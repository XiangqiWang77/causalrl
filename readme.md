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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install peft trl deepspeed
pip install sentencepiece bert-score
Project Structure
CausalRL/
├── reward_part/      # reward model training
│   └── run.sh
├── trainer/          # policy optimization
│   └── llama38.sh
├── data/
├── checkpoints/
└── logs/
Quick Start
Step 1 — Train reward model (required)
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

trainer/llama38.sh
Policy training depends on the reward model.
Skipping this step will cause missing reward signals or unstable training.
