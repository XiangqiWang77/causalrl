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
    """
    支持两种顶层结构：
      - 列表：[ {question, answer, ...}, ... ]
      - 字典包裹：{ data/examples/rows/instances: [ ... ] }
    """
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
    ap.add_argument("--dir", type=str, default=".", help="包含多个 JSON 的目录（默认当前目录）")
    ap.add_argument("--k", type=int, default=110, help="每个文件采样条数（默认110）")
    ap.add_argument("--out", type=str, default="combined.json", help="输出文件名（默认 combined.json）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    ap.add_argument("--pattern", type=str, default="*.json", help="匹配文件模式（默认 *.json）")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.dir).resolve()
    out_path = (root / args.out).resolve()

    # 收集候选文件
    files = sorted([p for p in root.glob(args.pattern)
                    if p.is_file() and p.resolve() != out_path and not p.name.startswith(".")])

    if not files:
        print("[!] 未找到任何 JSON 文件。", file=sys.stderr)
        sys.exit(2)

    combined = []
    total_in, total_kept = 0, 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 跳过无法解析的文件: {f.name} ({e})")
            continue

        # 提取并规范化为 {question, answer}
        qa_list = []
        for rec in iter_samples_from_json(data):
            qa = to_qa(rec)
            if qa:
                qa_list.append(qa)

        n = len(qa_list)
        total_in += n
        if n == 0:
            print(f"[warn] {f.name}: 未发现有效的 question/answer 对")
            continue

        k = min(args.k, n)
        chosen = random.sample(qa_list, k) if n > k else qa_list
        combined.extend(chosen)
        total_kept += len(chosen)
        print(f"[ok] {f.name}: {n} 条，采样 {len(chosen)} 条")

    # 写出合并结果
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 共读取 {len(files)} 个文件，输入 {total_in} 条，合并输出 {total_kept} 条 -> {out_path}")

if __name__ == "__main__":
    main()
