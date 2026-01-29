from datasets import load_dataset
import json, os


ds = load_dataset("hubert233/BigBenchExtraHard", split="causal_understanding")


records = [{"question": ex["input"], "answer": ex["target"]} for ex in ds]


out_json = "bbeh_causal_understanding.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"Saved {len(records)} items to {os.path.abspath(out_json)}")
