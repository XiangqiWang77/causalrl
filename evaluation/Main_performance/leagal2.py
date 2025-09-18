#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export LegalBench 'overruling' to {question, answer} JSON
(41 samples), without datasets.load_dataset.

It first tries historical Parquet shards under:
  overruling/train/*.parquet, test/*.parquet
If not found, falls back to top-level data.tar.gz.

Usage:
  pip install -U huggingface_hub pandas pyarrow
  python export_legalbench_overruling_qa.py
"""
import io, os, re, sys, tarfile, tempfile, json, pathlib, random
from typing import List, Iterable
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID   = "nguha/legalbench"
PINNED_REV = "ed000c625af7242b058b8282b9d551c1b70ca7de"  # same style as your script
TASK_DIR  = "overruling"
OUT_FILE  = "legalbench_overruling_qa_41.json"
N_SAMPLES = 51
SEED      = 42

QUESTION_STEM = (
    "Context: {ctx} "
    "Question: Does this sentence overrule a previous case? Answer Yes or No."
)

TEXT_CANDS  = ["text", "context", "passage", "input", "prompt", "sentence"]
LABEL_CANDS = ["label", "answer", "target", "y"]

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
    files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=PINNED_REV)
    wanted = [f for f in files if f.startswith(f"{TASK_DIR}/") and f.endswith(".parquet")]
    if not wanted:
        return pd.DataFrame()
    return _load_parquet_paths(sorted(wanted))

def _extract_tar_and_collect(tar_path: str, task_name: str) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as tf, tempfile.TemporaryDirectory() as td:
        tf.extractall(td)
        root = pathlib.Path(td)
        task_root = None
        for p in root.rglob("*"):
            if p.is_dir() and p.name == task_name:
                task_root = p; break
        if task_root is None:
            return pd.DataFrame()

        dfs = []
        for file in task_root.rglob("*"):
            if not file.is_file():
                continue
            try:
                suf = file.suffix.lower()
                if suf == ".parquet":
                    dfs.append(pd.read_parquet(file))
                elif suf in {".jsonl", ".json"}:
                    if suf == ".jsonl":
                        dfs.append(pd.read_json(file, lines=True))
                    else:
                        obj = json.loads(file.read_text(encoding="utf-8"))
                        if isinstance(obj, list):
                            dfs.append(pd.DataFrame(obj))
                        elif isinstance(obj, dict):
                            for k in ["data","examples","rows","instances","train","test"]:
                                if k in obj and isinstance(obj[k], list):
                                    dfs.append(pd.DataFrame(obj[k])); break
                elif suf in {".csv", ".tsv"}:
                    dfs.append(pd.read_csv(file, sep="\t" if suf==".tsv" else ","))
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _fallback_tarball(api: HfApi) -> pd.DataFrame:
    tar_local = hf_hub_download(REPO_ID, "data.tar.gz", repo_type="dataset", revision="main")
    return _extract_tar_and_collect(tar_local, TASK_DIR)

def _to_yesno(lab: str) -> str:
    s = str(lab).strip().lower()
    # Common encodings seen across LegalBench tasks:
    if s in {"0","no","false","f","neg","negative"}:
        return "No"
    if s in {"1","yes","true","t","pos","positive"}:
        return "Yes"
    # Fallback: anything else that's not clearly no -> map to Yes/No heuristically
    return "Yes" if re.match(r"y|1|true", s) else "No" if re.match(r"n|0|false", s) else s.title()

def _to_qa(df: pd.DataFrame) -> List[dict]:
    out = []
    for _, row in df.iterrows():
        ctx = _pick(row, TEXT_CANDS)
        lab = _pick(row, LABEL_CANDS)
        if not ctx or not lab:
            continue
        q = QUESTION_STEM.format(ctx=_norm(ctx))
        out.append({"question": q, "answer": _to_yesno(lab)})
    # dedup
    seen, dedup = set(), []
    for r in out:
        k = (r["question"], r["answer"])
        if k not in seen:
            seen.add(k); dedup.append(r)
    return dedup

def main():
    random.seed(SEED)
    api = HfApi()
    df = _try_historical_parquet(api)
    if df.empty:
        print("[info] parquet not found at pinned rev; using data.tar.gz fallback ...")
        df = _fallback_tarball(api)
    if df.empty:
        print("[!] Could not locate task files for 'overruling'.", file=sys.stderr)
        sys.exit(2)

    # sample 41 (deterministic via SEED)
    df = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
    qa = _to_qa(df)
    pathlib.Path(OUT_FILE).write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(qa)} samples -> {pathlib.Path(OUT_FILE).resolve()}")

if __name__ == "__main__":
    main()
