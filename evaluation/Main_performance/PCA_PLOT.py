# -*- coding: utf-8 -*-
# PCA distribution plot for four QA JSON files, saving figures to /mnt/data
# - Uses TF-IDF on "question + answer"
# - Reduces to 2D via PCA
# - Saves scatter plot and coordinates CSV
#
# NOTE: This script is robust to missing files and heterogeneous key names.
#       If no valid data rows are found, it still saves a placeholder figure.

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ---------- Config ----------
files = [
    "bbeh_multistep_arithmetic_qa.json",
    "cladder_qa.json",
    "legalbench_legal_reasoning_causality_qa.json",
    "LogiQA.json",
]

save_dir = Path("./")
save_dir.mkdir(parents=True, exist_ok=True)
fig_path_pca = save_dir / "qa_distribution_pca.png"
coords_csv_path = save_dir / "qa_pca_coords.csv"
summary_csv_path = save_dir / "qa_load_summary.csv"

# ---------- Helpers ----------
def pick_first_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Return the first key from candidates that exists in dict `d` (case-insensitive)."""
    lower_map = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None

def extract_qa(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Try to extract (question, answer) from a possibly heterogeneous record."""
    # Common possibilities
    q_candidates = [
        "question", "prompt", "input", "query", "instruction", "qn", "q"
    ]
    a_candidates = [
        "answer", "output", "response", "target", "label", "gold", "a"
    ]
    q_key = pick_first_key(record, q_candidates)
    a_key = pick_first_key(record, a_candidates)
    q_val = record.get(q_key) if q_key else None
    a_val = record.get(a_key) if a_key else None

    # Convert non-str to str (e.g., options arrays)
    if q_val is not None and not isinstance(q_val, str):
        q_val = json.dumps(q_val, ensure_ascii=False)
    if a_val is not None and not isinstance(a_val, str):
        a_val = json.dumps(a_val, ensure_ascii=False)

    return q_val, a_val

def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON file that may contain a list, a dict with 'data', or a single item."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # If there is a top-level array under a common key
        for k in ["data", "samples", "items", "instances", "examples"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        # Otherwise treat the dict itself as one record
        return [data]
    # Fallback: wrap anything else
    return [data]

# ---------- Ingest ----------
rows: List[Dict[str, Any]] = []
load_log: List[Dict[str, Any]] = []

for fp in files:
    p = Path(fp)
    status = {"file": fp, "exists": p.exists(), "total": 0, "parsed": 0}
    if not p.exists():
        load_log.append(status)
        continue

    try:
        records = load_json_records(p)
        status["total"] = len(records)
        for r in records:
            q, a = extract_qa(r) if isinstance(r, dict) else (None, None)
            if q is None and a is None:
                continue
            text = ((q or "") + " " + (a or "")).strip()
            if not text:
                continue
            rows.append({"source": p.name, "question": q or "", "answer": a or "", "text": text})
            status["parsed"] += 1
    except Exception as e:
        status["error"] = str(e)
    finally:
        load_log.append(status)

df = pd.DataFrame(rows)
log_df = pd.DataFrame(load_log)

# Save the load summary for transparency
log_df.to_csv(summary_csv_path, index=False)

# ---------- Vectorize ----------
if len(df) > 0:
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df["text"].tolist())
    # If the doc count is tiny, gracefully handle min_df issues
    if X.shape[0] < 3:
        # Fall back to a simpler vectorizer
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(df["text"].tolist())

    # ---------- PCA ----------
    # Guard against rank deficiency when features < 2
    n_components = 2 if X.shape[1] >= 2 and X.shape[0] >= 2 else 1
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    # Save coordinates
    coords = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    out = pd.concat([df[["source", "question", "answer"]].reset_index(drop=True), coords], axis=1)
    out.to_csv(coords_csv_path, index=False)

    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    # Avoid specifying colors; let matplotlib choose defaults
    sources = out["source"].unique().tolist()
    for s in sources:
        mask = out["source"] == s
        xs = out.loc[mask, "PC1"]
        ys = out.loc[mask, "PC2"] if "PC2" in out.columns else np.zeros_like(xs)
        plt.scatter(xs, ys, label=s, alpha=0.8, s=24)

    # Axis labels and title
    if n_components == 2:
        evr = getattr(pca, "explained_variance_ratio_", None)
        if evr is not None and len(evr) >= 2:
            title = f"PCA distribution (PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%)"
        else:
            title = "PCA distribution"
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(title)
    else:
        plt.xlabel("PC1")
        plt.title("PCA distribution (1D)")

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path_pca, dpi=180)
    plt.close()

else:
    # Create a placeholder figure to indicate no data was loaded
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    msg = (
        "No valid Q&A rows loaded.\n"
        "Please ensure the following files exist in the working directory:\n"
        "- bbeh_multistep_arithmetic_qa.json\n"
        "- cladder_qa.json\n"
        "- legalbench_legal_reasoning_causality_qa.json\n"
        "- LogiQA.json\n"
        "Each record should include keys like 'question'/'answer' (or 'input'/'target')."
    )
    plt.text(0.01, 0.6, msg, fontsize=11, va="top")
    plt.tight_layout()
    plt.savefig(fig_path_pca, dpi=180)
    plt.close()

# Display a small preview of the load summary so the user can confirm ingestion
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("QA Load Summary", log_df.head(50))
except Exception as e:
    print("Display error:", e)

{
    "pca_figure": str(fig_path_pca),
    "coords_csv": str(coords_csv_path),
    "load_summary_csv": str(summary_csv_path),
    "rows_loaded": len(df)
}
