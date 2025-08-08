# convert_verl_format.py
# pip install pandas pyarrow
import json
from pathlib import Path
import pandas as pd
import random

# 固定配置（零参数）
DEFAULT_PREFIX = (
    "You must wrap your final answer exactly in "
    "<finalanswer>YOUR ANSWER</finalanswer> tags; "
    "you may include any text before or after those tags."
)
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
OUT_TRAIN = "train.parquet"
OUT_TEST = "test.parquet"
ABILITY = "general"  # 文档示例里叫 "math"，你这里统一用 general

def iter_qa_from_json(obj):
    """
    允许以下结构：
      - [ { "question": "...", "answer": ... }, ... ]
      - { "data": [ {...}, ... ] }
      - { "question": "...", "answer": ... }
    其中 answer 可为 str 或 list
    """
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        items = obj["data"]
    elif isinstance(obj, dict) and ("question" in obj and "answer" in obj):
        items = [obj]
    else:
        return
    for r in items:
        if not isinstance(r, dict):
            continue
        q = (r.get("question") or "").strip()
        a = r.get("answer")
        if isinstance(a, list):
            a = " or ".join([str(x).strip() for x in a if x is not None and str(x).strip()])
        else:
            a = ("" if a is None else str(a)).strip()
        if q and a:
            yield q, a

def main():
    random.seed(RANDOM_SEED)
    rows = []
    json_files = sorted(Path(".").glob("*.json"))
    if not json_files:
        raise RuntimeError("No *.json files found in current directory.")

    # 先收集所有样本
    for p in json_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        idx = 0
        for q, a in iter_qa_from_json(obj):
            q_full = (DEFAULT_PREFIX.strip() + "\n\n" + q.strip()).strip()
            rows.append({
                "data_source": p.stem,
                "prompt": [{"role": "user", "content": q_full}],   # HF chat 格式
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": a},  # 文档要求 ground_truth 放这里
                "extra_info": {"split": "unknown", "index": idx},      # 必须是 dict，别用字符串
            })
            idx += 1

    if not rows:
        raise RuntimeError("No valid question/answer pairs found.")

    # 打乱并切分
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    n_train = int(len(df) * TRAIN_RATIO)
    df.iloc[:n_train].to_parquet(OUT_TRAIN, index=False, engine="pyarrow")
    df.iloc[n_train:].to_parquet(OUT_TEST, index=False, engine="pyarrow")
    print(f"OK — total={len(df)}, train={n_train}, test={len(df)-n_train}")
    print(f"Saved: {OUT_TRAIN}, {OUT_TEST}")

if __name__ == "__main__":
    main()
