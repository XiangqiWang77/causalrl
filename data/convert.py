# convert.py
# pip install pandas pyarrow

import argparse, json, os, random, uuid, sys
from pathlib import Path
import pandas as pd

random.seed(42)

DEFAULT_PREFIX = "You must wrap your final answer exactly in <finalanswer>YOUR ANSWER</finalanswer> tags; you may include any text before or after those tags."

def load_json_any(path: Path):
    items, text = [], path.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
            items = obj["data"]
        elif isinstance(obj, dict) and ("question" in obj):
            items = [obj]
    except Exception:
        for line in text.splitlines():
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    out = []
    for r in items:
        if not isinstance(r, dict): continue
        q = str(r.get("question") or "").strip()
        a = r.get("answer")
        if isinstance(a, list):
            a = " or ".join([str(x).strip() for x in a if x is not None and str(x).strip()])
        else:
            a = str(a or "").strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="default prefix for prompt")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--ability", type=str, default="general")
    args = ap.parse_args()

    files = [p for p in Path(".").glob("*.json*") if p.is_file()]
    if not files:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        for ex in load_json_any(f):
            rows.append({
                "data_source": f.stem,
                "question": ex["question"],
                "answer": ex["answer"],
                "extra_info": f.name,
            })

    if not rows:
        print("No valid QA rows parsed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(df))])
    df["ability"] = args.ability
    df["prompt"] = args.prefix.strip() + "\n\n" + df["question"]
    df["ground_truth"] = df["answer"]

    df = df[["id", "data_source", "ability", "prompt", "ground_truth", "extra_info"]]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df) * args.train_ratio)

    df.iloc[:n_train].to_parquet("train.parquet", index=False)
    df.iloc[n_train:].to_parquet("test.parquet", index=False)

    print(f"[OK] Total={len(df)}, Train={n_train}, Test={len(df)-n_train}")
    print("Saved: train.parquet, test.parquet")

if __name__ == "__main__":
    main()
