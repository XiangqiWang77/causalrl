# ifqa_to_bbeh_or_answers.py
# pip install -U datasets
from datasets import load_dataset
import json, random

random.seed(42)

# 1) 加载 IfQA（全量）
ds = load_dataset("jeggers/ifqa", split="train")

# 2) 转成 BBEH 风格，多个答案用 " or " 连接
def to_bbeh(ex):
    q = (ex.get("question") or "").strip()
    ans_list = ex.get("answers") or ex.get("answer") or []
    if not isinstance(ans_list, list):
        ans_list = [ans_list]
    ans_list = [str(a).strip() for a in ans_list if a is not None and str(a).strip()]
    ans_str = " or ".join(ans_list) if ans_list else ""
    return {"question": q, "answer": ans_str}

ds_bbeh = ds.map(to_bbeh, remove_columns=ds.column_names)

# 3) 随机抽样 200
n = min(200, len(ds_bbeh))
sampled = ds_bbeh.shuffle(seed=42).select(range(n))

# 4) 保存
out_path = "ifqa_200_bbeh_or_answers.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sampled.to_list(), f, ensure_ascii=False, indent=2)

print(f"[OK] Saved {len(sampled)} items -> {out_path}")
