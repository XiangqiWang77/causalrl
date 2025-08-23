# convert_verl_format_stratified.py
# pip install pandas pyarrow
import json
from pathlib import Path
import pandas as pd
import numpy as np
import random
import hashlib

# 固定配置（零参数）
DEFAULT_PREFIX = (
    "Solve the question and generate in assigned format."
    "Wrap your final answer between <finalanswer> </finalanswer> tags. "
    "It's mandatory and required to not include anything after </finalanswer> tag. Don't let your generation process to redundant and lengthy."
    "Be very clear in your explanation, and ensure the final answer is presented separately."
    "Example of output: We analyze A, solve B and answer is C. So final answer is <finalanswer> C is correct </finalanswer>"
)

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
OUT_TRAIN = "train.parquet"
OUT_TEST = "test.parquet"
ABILITY = "general"  # 文档示例里叫 "math"，这里统一用 general

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

def read_any_json(path: Path):
    # 支持 .json 与 .jsonl
    if path.suffix.lower() == ".jsonl":
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield from iter_qa_from_json(obj)
    else:
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        yield from iter_qa_from_json(obj)

def stable_uid(data_source: str, idx: int, q: str, a: str) -> str:
    m = hashlib.sha256()
    m.update(data_source.encode("utf-8"))
    m.update(str(idx).encode("utf-8"))
    m.update(q.encode("utf-8"))
    m.update(a.encode("utf-8"))
    return m.hexdigest()[:16]

def main():
    random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    rows = []
    json_files = sorted(list(Path(".").glob("*.json")) + list(Path(".").glob("*.jsonl")))
    if not json_files:
        raise RuntimeError("No *.json or *.jsonl files found in current directory.")

    # 收集所有样本（带 data_source 和稳定 uid）
    for p in json_files:
        idx = 0
        for q, a in read_any_json(p):
            q_full = (DEFAULT_PREFIX.strip() + "\n\n" + q.strip()).strip()
            uid = stable_uid(p.stem, idx, q_full, a)
            rows.append({
                "uid": uid,
                "data_source": p.stem,
                "prompt": [{"role": "user", "content": q_full}],   # HF chat 格式
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": a},  # 文档要求 ground_truth 放这里
                "extra_info": {"split": "unknown", "index": idx},      # 必须是 dict
            })
            idx += 1

    if not rows:
        raise RuntimeError("No valid question/answer pairs found.")

    df = pd.DataFrame(rows)

    # 可选：去重（按 uid 或按 prompt+ground_truth）
    df = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)

    # === 分层切分（按 data_source） ===
    train_parts = []
    test_parts = []

    for src, g in df.groupby("data_source", sort=False):
        n = len(g)
        n_train = int(round(n * TRAIN_RATIO))

        # 极小数据集保底：尽量让两边都有样本
        if n == 1:
            n_train = 1  # 单样本只能进 train
        else:
            n_train = max(1, min(n - 1, n_train))  # 既不为 0 也不为 n

        # 每个分组内部先洗牌，再切分，避免文件内顺序影响
        idx = rng.permutation(n)
        g_shuf = g.iloc[idx].reset_index(drop=True)

        train_parts.append(g_shuf.iloc[:n_train])
        test_parts.append(g_shuf.iloc[n_train:])

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True)

    # === 全局再次洗牌（确保不同数据集完全混合） ===
    df_train = df_train.sample(frac=1.0, random_state=RANDOM_SEED + 1).reset_index(drop=True)
    df_test  = df_test.sample(frac=1.0,  random_state=RANDOM_SEED + 2).reset_index(drop=True)

    # 保存
    df_train.to_parquet(OUT_TRAIN, index=False, engine="pyarrow")
    df_test.to_parquet(OUT_TEST, index=False, engine="pyarrow")

    # 简要统计：确认混合程度
    mix_train = df_train["data_source"].value_counts().to_dict()
    mix_test  = df_test["data_source"].value_counts().to_dict()
    print(f"OK — total={len(df)}, train={len(df_train)}, test={len(df_test)}")
    print("train per-source:", mix_train)
    print("test  per-source:", mix_test)
    print(f"Saved: {OUT_TRAIN}, {OUT_TEST}")

if __name__ == "__main__":
    main()
