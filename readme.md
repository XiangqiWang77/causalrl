# CausalRL

Official implementation of **Causal-Enhanced Policy Optimization (CE-PO)** for large language models.

This project improves reasoning faithfulness by combining:

- accuracy reward (reward model / BERTScore)
- Jacobian-based causal coherence signals
- PPO / GRPO policy optimization

---

## Installation

### 1. Environment

```bash
conda create -n causalrl python=3.11
conda activate causalrl
```

2. Dependencies

Refer to base**.md

Project Structure

CausalRL/

├── reward_part/      # reward activation

│   └── run.sh

├── trainer/          # policy optimization of verl

├── data/

├── checkpoints/

└── logs/


Quick Start

Step 1 — Activate Jacobian Reward calculation

cd reward_part
bash run.sh

This produces reward checkpoints used by policy training.

Step 2 — Run CE-PO training

cd ..
bash trainer/llama38.sh

Important
You must run:

reward_part/run.sh

before:

./llama38.sh

