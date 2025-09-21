#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, random, re, sys
from pathlib import Path

Q_KEYS = ["question", "input", "prompt", "text", "query", "problem", "passage_question"]
A_KEYS = ["answer", "target", "label", "output", "gold", "final_answer", "y"]

def norm(s):
    return re.sub(r"\s+", " ", ("" if s is None else str(s)).strip())

def pick(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None

def iter_samples_from_json(obj):
    if isinstance(obj, list):
        for r in obj:
            if isinstance(r, dict):
                yield r
    elif isinstance(obj, dict):
        for k in ["data", "examples", "rows", "instances", "train", "test", "validation"]:
            if k in obj and isinstance(obj[k], list):
                for r in obj[k]:
                    if isinstance(r, dict):
                        yield r

def to_qa(rec):
    q = pick(rec, Q_KEYS)
    a = pick(rec, A_KEYS)
    if q is None or a is None:
        return None
    q = norm(q); a = norm(a)
    if not q or not a:
        return None
    return {"question": q, "answer": a}

def main():
    ap = argparse.ArgumentParser(description="Combine multiple QA JSONs: sample K from each file into combined.json")
    ap.add_argument("--dir", type=str, default=".")
    ap.add_argument("--k", type=int, default=110)
    ap.add_argument("--out", type=str, default="combined.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pattern", type=str, default="*.json")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.dir).resolve()
    out_path = (root / args.out).resolve()

    files = sorted([p for p in root.glob(args.pattern)
                    if p.is_file() and p.resolve() != out_path and not p.name.startswith(".")])

    if not files:
        print("[!] error", file=sys.stderr)
        sys.exit(2)

    combined = []
    total_in, total_kept = 0, 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn]: {f.name} ({e})")
            continue

        qa_list = []
        for rec in iter_samples_from_json(data):
            qa = to_qa(rec)
            if qa:
                qa_list.append(qa)

        n = len(qa_list)
        total_in += n
        if n == 0:
            print(f"[warn] {f.name}: question/answer")
            continue

        k = min(args.k, n)
        chosen = random.sample(qa_list, k) if n > k else qa_list
        combined.extend(chosen)
        total_kept += len(chosen)
        print(f"[ok] {f.name}: {n} {len(chosen)} ")

    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {len(files)}，{total_in}，{total_kept} -> {out_path}")

if __name__ == "__main__":
    main()
