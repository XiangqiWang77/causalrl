# pip install -U datasets
from datasets import load_dataset
import json, re, random

random.seed(42)
BOX = re.compile(r"\\boxed\{([^{}]+)\}")

def extract_boxed(s):
    vals = [v.strip() for v in BOX.findall(s or "") if v.strip()]
    return " or ".join(vals)

ds = load_dataset("lighteval/MATH-Hard", split="test")  # or "train" if available
def to_bbeh(ex):
    q = (ex.get("problem") or "").strip()
    sol = ex.get("solution") or ""
    ans = extract_boxed(sol) or sol.strip()
    return {"question": q, "answer": ans}

bb = ds.map(to_bbeh, remove_columns=ds.column_names)
bb = bb.shuffle(seed=42).select(range(min(200, len(bb))))
with open("MATH_Hard_200_bbeh.json", "w", encoding="utf-8") as f:
    json.dump(bb.to_list(), f, ensure_ascii=False, indent=2)
print("Saved", len(bb), "→ MATH_Hard_200_bbeh.json")
