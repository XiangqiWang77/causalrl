#!/usr/bin/env python
# BBH_cj.py

from datasets import load_dataset

#ds = load_dataset("lighteval/big_bench_hard", "causal_judgement")
def main():
    # Load the 'causal_judgement' configuration from the maveriq/bigbenchhard dataset
    ds = load_dataset(
        "lighteval/big_bench_hard",
        "causal_judgement",
        split="train"
    )
    # Print a brief summary
    print(f"Number of examples in train split: {len(ds)}")
    print("First example:")
    print(ds[0])

if __name__ == "__main__":
    main()
