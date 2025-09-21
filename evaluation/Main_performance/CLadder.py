# make_cladder_qa_fixed.py
import json, re, pathlib, sys, io, zipfile, requests

GITHUB_ZIP = "https://github.com/causalNLP/cladder/raw/main/data/cladder-v1.zip"
PREFERRED_FILES = [
    "cladder-v1-balanced.json",
    "cladder-v1-aggregate.json",
    "cladder-v1-q-commonsense.json",
    "cladder-v1-q-easy.json",
    "cladder-v1-q-hard.json",
    "cladder-v1-q-anticommonsense.json",
    "cladder-v1-q-nonsense.json",
]

FIELD_ALIASES = {
    "given_info": ["given_info", "context", "given", "background", "givenInfo"],
    "question":   ["question", "question_text", "prompt", "q", "Question"],
    "answer":     ["answer", "ans", "label", "Answer", "gt", "groundtruth"],
}

def _pick(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return str(d[k])
    return ""

def _norm_answer(a: str) -> str:
    a = str(a).strip().lower()
    if a in {"yes","no"}: return a
    if a in {"true","1","y","t"}: return "yes"
    if a in {"false","0","n","f"}: return "no"
    return "yes" if a and a[0] in {"y","t","1"} else "no"

def _row_to_qa(r: dict) -> dict:
    gi = re.sub(r"\s+"," ", _pick(r, FIELD_ALIASES["given_info"]).strip())
    q  = re.sub(r"\s+"," ", _pick(r, FIELD_ALIASES["question"]).strip())
    question = f"Context: {gi.rstrip('.')}. Question: {q}" if gi else f"Question: {q}"
    a = _norm_answer(_pick(r, FIELD_ALIASES["answer"]))
    return {"question": question, "answer": a}

def _iter_json_bytes(b: bytes):
    obj = json.loads(b.decode("utf-8"))
    if isinstance(obj, list):
        for r in obj: yield r
    elif isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            for r in obj["data"]: yield r

def main(out_path="cladder_qa.json"):
    records = []

    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            with open(p, "rb") as f:
                records.extend(list(_iter_json_bytes(f.read())))
    else:
        resp = requests.get(GITHUB_ZIP, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            picked = None
            names = set(zf.namelist())
            for name in PREFERRED_FILES:
                if name in names:
                    picked = name; break
            if not picked:
                picked = next((n for n in names if n.endswith(".json")), None)
                if not picked:
                    raise RuntimeError("JSON file not found in zip.")
            with zf.open(picked) as f:
                records.extend(list(_iter_json_bytes(f.read())))

    out = []
    for r in records:
        qa = _row_to_qa(r)
        if qa["question"].strip() and qa["question"].strip() != "Question:" and len(qa["question"]) >= 12:
            out.append(qa)

    seen, dedup = set(), []
    for x in out:
        key = (x["question"], x["answer"])
        if key not in seen:
            seen.add(key); dedup.append(x)

    pathlib.Path(out_path).write_text(json.dumps(dedup, ensure_ascii=False, indent=2), "utf-8")
    print(f"Wrote {len(dedup)} pairs -> {pathlib.Path(out_path).resolve()}")

if __name__ == "__main__":
    main()
