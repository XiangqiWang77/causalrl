from datasets import load_dataset
dataset = load_dataset("maveriq/bigbenchhard", "multistep_arithmetic_two", split="train", trust)
qa_pairs = [
    {"question": example["input"], "answer": example["target"]}
    for example in dataset
]
import json
print(json.dumps(qa_pairs, ensure_ascii=False, indent=2))
