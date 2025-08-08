# counterbench.py
# pip install -U datasets

from datasets import load_dataset
import json, random

random.seed(42)

# 只选 data_balanced_* 这两个文件，避免 schema 冲突
files = [
    "hf://datasets/CounterBench/CounterBench/data_balanced_alpha_V1.json",
    "hf://datasets/CounterBench/CounterBench/data_balanced_backdoor_V2.json",
]

# 用 json builder 显式加载指定文件（而不是整个仓库）
ds = load_dataset("json", data_files={"train": files}, split="train")

# （保险起见）只保留我们需要的列
need = {"question", "given_info", "answer"}
ds = ds.filter(lambda ex: all(k in ex and ex[k] is not None for k in need))

# 映射到 BBEH 风格：把 given_info 拼到题干前面
def to_bbeh(ex):
    q = (ex.get("question") or "").strip()
    bg = (ex.get("given_info") or "").strip()
    a = str(ex.get("answer") or "").strip()
    question = (bg + "\n\n" + q).strip() if bg else q
    return {"question": question, "answer": a}

ds_bbeh = ds.map(to_bbeh, remove_columns=ds.column_names)

# 抽样 300 条（若不足则全取）
n = min(200, len(ds_bbeh))
sampled = ds_bbeh.shuffle(seed=42).select(range(n))

# 保存
out_json = "bbeh_counterbench_200.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(sampled.to_list(), f, ensure_ascii=False, indent=2)

print(f"[OK] saved {len(sampled)} -> {out_json}")
