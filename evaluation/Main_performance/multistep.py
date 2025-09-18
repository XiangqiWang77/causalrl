#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the K easiest & shortest {question, answer} PAIRS from MrLight/bbeh-eval.

Heuristic "ease" score (lower = simpler/shorter):
    score = 0.7 * len_words(question) + 0.3 * len_words(answer)

Output: a JSON array of objects: [{"question": "...", "answer": "..."}, ...]
"""
import argparse, json, re, sys, pathlib
from typing import List, Set

try:
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install -U datasets", file=sys.stderr)
    raise

DATASET_ID = "MrLight/bbeh-eval"  # HF dataset repo id

def norm_task(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-_]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def fuzzy_pick(all_tasks: Set[str], pattern: str) -> List[str]:
    """Substring match over normalized task names; comma/space separated."""
    wanted = []
    pats = [norm_task(p) for p in re.split(r"[,\s]+", pattern) if p.strip()]
    for t in sorted(all_tasks):
        nt = norm_task(t)
        if any(p in nt for p in pats):
            wanted.append(t)
    return wanted

def len_words(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))

def ease_score(q: str, a: str, wq: float = 0.7, wa: float = 0.3) -> float:
    return wq * len_words(q) + wa * len_words(a)

def main():
    ap = argparse.ArgumentParser(
        description="Export K easiest & shortest QA pairs from MrLight/bbeh-eval."
    )
    ap.add_argument("--task", type=str, default="multistep_arithmetic",
                    help="可模糊匹配的任务名，支持逗号分隔多个（示例：'multistep, arithmetic'）")
    ap.add_argument("--out", type=str, default="bbeh_easy_qa.json",
                    help="输出 JSON 文件名（包含 question/answer 对）")
    ap.add_argument("--list-tasks", action="store_true",
                    help="仅列出可用 task 名称后退出")
    ap.add_argument("--topk", type=int, default=120,
                    help="输出 QA 对数量（默认 120）")
    ap.add_argument("--wq", type=float, default=0.7,
                    help="问题词数权重（默认 0.7）")
    ap.add_argument("--wa", type=float, default=0.3,
                    help="答案词数权重（默认 0.3）")
    args = ap.parse_args()

    # Load dataset (split = train)
    ds = load_dataset(DATASET_ID, split="train")  # expects fields: 'task', 'question', 'answer'  :contentReference[oaicite:0]{index=0}
    all_tasks = sorted(set(ds["task"]))
    if args.list_tasks:
        print("\n# Available tasks ({}):".format(len(all_tasks)))
        for t in all_tasks:
            print("-", t)
        return

    picked = fuzzy_pick(set(all_tasks), args.task)
    if not picked:
        print(f"[!] No tasks matched '{args.task}'. Use --list-tasks to see all names.", file=sys.stderr)
        sys.exit(2)
    print("[info] matched tasks:", picked)

    picked_set = set(picked)

    # Collect (score, question, answer)
    scored = []
    for ex in ds:
        if ex.get("task") not in picked_set:
            continue
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        if not q or not a:
            continue
        scored.append((ease_score(q, a, args.wq, args.wa), q, a))

    if not scored:
        print("[!] No QA found for selected tasks.", file=sys.stderr)
        sys.exit(3)

    # Sort by increasing "ease" score and deduplicate (q,a) pairs
    scored.sort(key=lambda x: x[0])
    seen = set()
    qa_pairs = []
    for _, q, a in scored:
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        qa_pairs.append({"question": q, "answer": a})
        if len(qa_pairs) >= args.topk:
            break

    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(qa_pairs)} QA pairs -> {out_path.resolve()}")
    print(f"[note] score = {args.wq}*len_words(question) + {args.wa}*len_words(answer)")
    print("[ref] Dataset card:", "https://huggingface.co/datasets/MrLight/bbeh-eval")  # :contentReference[oaicite:1]{index=1}

if __name__ == "__main__":
    main()
