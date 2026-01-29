# casehold_to_bbeh.py
# pip install -U datasets
from datasets import load_dataset
import json, random

random.seed(42)

def load_casehold_split(split="test"):
    """
    Try LexGLUE first (stable), then fallback to a standalone mirror if present.
    Available splits in LexGLUE: 'train', 'validation', 'test'
    """
    try:
        return load_dataset("lex_glue", "case_hold", split=split)
    except Exception:
        # fallback: some mirrors might expose 'case_hold' directly
        return load_dataset("case_hold", split=split)

def pick_first_key(d, keys, default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def to_bbeh(ex):
    # context / prompt / question text
    ctx = pick_first_key(ex, ["context", "prompt", "question", "text"], "")
    ctx = (ctx or "").strip()

    # options
    opts = pick_first_key(ex, ["endings", "choices", "options"], [])
    if not isinstance(opts, list):
        # some datasets store choices as dict with 'text' fields; normalize
        if isinstance(opts, dict):
            opts = [opts.get(k, "") for k in sorted(opts.keys())]
        else:
            opts = []

    # gold label (index or letter)
    lbl = pick_first_key(ex, ["label", "answer", "gold"], None)

    # normalize label to index
    if isinstance(lbl, str):
        lbl_str = lbl.strip().upper()
        # allow A-E letters
        if lbl_str in ["A", "B", "C", "D", "E"]:
            lbl_idx = ["A", "B", "C", "D", "E"].index(lbl_str)
        else:
            try:
                lbl_idx = int(lbl_str)
            except Exception:
                lbl_idx = None
    else:
        lbl_idx = int(lbl) if lbl is not None else None

    # build question string with labeled options
    letters = ["A", "B", "C", "D", "E", "F"]
    opt_lines = []
    for i, opt in enumerate(opts):
        opt_text = str(opt).strip()
        letter = letters[i] if i < len(letters) else f"Option{i+1}"
        opt_lines.append(f"{letter}. {opt_text}")
    options_block = "\n".join(opt_lines)

    question = (
        f"{ctx}\n\nOptions:\n{options_block}\n\n"
        f"Question: Select the correct holding. Answer with the option text."
    ).strip()

    answer = ""
    if isinstance(lbl_idx, int) and 0 <= lbl_idx < len(opts):
        answer = str(opts[lbl_idx]).strip()

    return {"question": question, "answer": answer}

def main():
    ds = load_casehold_split(split="test")

    
    ds_bbeh = ds.map(to_bbeh, remove_columns=ds.column_names)

    
    ds_bbeh = ds_bbeh.shuffle(seed=42).select(range(min(200, len(ds_bbeh))))

    out_path = "casehold_bbeh.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ds_bbeh.to_list(), f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(ds_bbeh)} items -> {out_path}")

if __name__ == "__main__":
    main()
