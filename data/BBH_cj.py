from datasets import load_dataset
import json, os

# 1) 加载 BBEH 的 causal understanding split
ds = load_dataset("hubert233/BigBenchExtraHard", split="causal_understanding")

# 2) 规范化为 {question, answer}，源字段是 {input, target}
records = [{"question": ex["input"], "answer": ex["target"]} for ex in ds]

# 3) 存成 JSON（整数组合）
out_json = "bbeh_causal_understanding.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"Saved {len(records)} items to {os.path.abspath(out_json)}")