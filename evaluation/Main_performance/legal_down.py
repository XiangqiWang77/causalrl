#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export LegalBench 'legal_reasoning_causality' to {question, answer} JSON
without using datasets.load_dataset (no trust_remote_code).

It first tries to read historical commit Parquet shards under:
  legal_reasoning_causality/train/*.parquet, test/*.parquet
If not found, it falls back to downloading the top-level data.tar.gz,
extracts, and searches for the task files (jsonl/csv/parquet).

Usage:
  pip install -U huggingface_hub pandas pyarrow
  python export_legalbench_causality_qa.py
"""

import io, os, re, sys, tarfile, tempfile, json, pathlib
from typing import List, Iterable
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID  = "nguha/legalbench"
# A commit that contains per-task Parquet folders:
PINNED_REV = "ed000c625af7242b058b8282b9d551c1b70ca7de"  # see HF file history
TASK_DIR = "legal_reasoning_causality"
OUT_FILE = "legalbench_legal_reasoning_causality_qa.json"

QUESTION_STEM = (
    "Context: {ctx} "
    "Question: Based on the excerpt above, did the judge's legal reasoning "
    "rely on statistical evidence (e.g., regression analysis) when determining "
    "whether there was a causal link? Answer Yes or No."
)

TEXT_CANDS   = ["text", "context", "passage", "input", "prompt"]
LABEL_CANDS  = ["label", "answer", "target", "y"]

def _norm(s) -> str:
    return re.sub(r"\s+", " ", ("" if s is None else str(s)).strip())

def _pick(row: pd.Series, names: List[str]) -> str:
    for n in names:
        if n in row and pd.notna(row[n]) and str(row[n]).strip():
            return str(row[n]).strip()
    return ""

def _load_parquet_paths(paths: Iterable[str]) -> pd.DataFrame:
    dfs = []
    for rel in paths:
        local = hf_hub_download(REPO_ID, rel, repo_type="dataset", revision=PINNED_REV)
        df = pd.read_parquet(local)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _try_historical_parquet(api: HfApi) -> pd.DataFrame:
    # List files at the pinned revision and collect all parquet under the task
    files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=PINNED_REV)
    wanted = [f for f in files
              if f.startswith(f"{TASK_DIR}/") and f.endswith(".parquet")]
    if not wanted:
        return pd.DataFrame()
    return _load_parquet_paths(sorted(wanted))

def _extract_tar_and_collect(tar_path: str, task_name: str) -> pd.DataFrame:
    # Search inside tarball for the task dir and load jsonl/csv/parquet
    with tarfile.open(tar_path, "r:gz") as tf, tempfile.TemporaryDirectory() as td:
        tf.extractall(td)
        root = pathlib.Path(td)
        task_root = None
        for p in root.rglob("*"):
            if p.is_dir() and p.name == task_name:
                task_root = p
                break
        if task_root is None:
            return pd.DataFrame()

        dfs = []
        for file in task_root.rglob("*"):
            if not file.is_file():
                continue
            try:
                if file.suffix.lower() == ".parquet":
                    dfs.append(pd.read_parquet(file))
                elif file.suffix.lower() in {".jsonl", ".json"}:
                    # jsonl preferred; json can be either list or dict-wrapped
                    if file.suffix.lower() == ".jsonl":
                        dfs.append(pd.read_json(file, lines=True))
                    else:
                        obj = json.loads(file.read_text(encoding="utf-8"))
                        if isinstance(obj, list):
                            dfs.append(pd.DataFrame(obj))
                        elif isinstance(obj, dict):
                            # try common keys
                            for k in ["data", "examples", "rows", "instances", "train", "test"]:
                                if k in obj and isinstance(obj[k], list):
                                    dfs.append(pd.DataFrame(obj[k]))
                                    break
                elif file.suffix.lower() in {".csv", ".tsv"}:
                    dfs.append(pd.read_csv(file, sep="\t" if file.suffix.lower()==".tsv" else ","))
            except Exception:
                continue

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _fallback_tarball(api: HfApi) -> pd.DataFrame:
    # Download top-level data.tar.gz from main and search
    tar_local = hf_hub_download(REPO_ID, "data.tar.gz", repo_type="dataset", revision="main")
    return _extract_tar_and_collect(tar_local, TASK_DIR)

def _to_qa(df: pd.DataFrame) -> List[dict]:
    out = []
    for _, row in df.iterrows():
        ctx = _pick(row, TEXT_CANDS)
        lab = _pick(row, LABEL_CANDS)
        if not ctx or not lab:
            continue
        q = QUESTION_STEM.format(ctx=_norm(ctx))
        out.append({"question": q, "answer": _norm(lab)})
    # dedup
    seen, dedup = set(), []
    for r in out:
        k = (r["question"], r["answer"])
        if k not in seen:
            seen.add(k); dedup.append(r)
    return dedup

def main():
    api = HfApi()
    df = _try_historical_parquet(api)
    if df.empty:
        print("[info] historical Parquet not found or inaccessible; using data.tar.gz fallback ...")
        df = _fallback_tarball(api)
    if df.empty:
        print("[!] Could not locate task files. Check the repo manually.", file=sys.stderr)
        sys.exit(2)

    qa = _to_qa(df)
    pathlib.Path(OUT_FILE).write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(qa)} samples -> {pathlib.Path(OUT_FILE).resolve()}")

if __name__ == "__main__":
    main()
