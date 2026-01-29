
# CausalRL: Causal-Enhanced Policy Optimization (CE-PO)

This repository contains the official implementation of **Causal-Enhanced Policy Optimization (CE-PO)**. 

CE-PO is a model-agnostic reward fusion framework designed to mitigate "shortcut learning" in Large Language Models (LLMs). It augments standard policy optimization (PPO/GRPO) by combining task accuracy rewards with **Jacobian-based causal coherence signals**. This ensures that the model's reasoning pathway ($Z \rightarrow X \rightarrow Y$) is not only accurate but faithful

---

## Project Structure

The project is organized to separate the causal reward calculation from the policy optimization loop.

```text
CausalRL/
├── reward_part/        # Jacobian Reward Calculation
│   ├── run.sh          # Script to generate/calculate causal signals
│   └── ...             # Logic for counterfactual hardening & spectral normalization
├── trainer/            # Policy Optimization (Verl-based)
│   └── ...             # PPO/GRPO implementation with Minkowski combiner
├── data/               # Academic datasets (BBEH, CaseHOLD, etc.)
├── checkpoints/        # Saved models and reward checkpoints
├── llama38.sh      # Main training script (e.g., Llama-3-8B)
└── logs/               # Training logs
```

---

## Quick Start

### Prerequisites

The code relies on the **Verl** framework for efficient RLHF training. Ensure your environment is set up with Python 3.11+.

### Installation

You may refer to base01env.md, base02usage.md to install the Verl training environment.

### 1. Activate Jacobian Reward Calculation

CE-PO requires differentiable, model-internal proxies for causal coherence. You must first compute these signals or run the reward server thread.

```bash
cd reward_part
bash run.sh

```

This step computes the raw Jacobian sensitivities and applies **counterfactual hardening** to remove nuisance factors (e.g., length biases).

### 2. Run Policy Optimization (CE-PO)

Once the reward signals are available, run the training script. The trainer uses a **Minkowski (power-mean) combiner** to fuse the accuracy reward (e.g., BERTScore) with the causal signals.

```bash
cd ..
bash llama38.sh

```

> 
> **Note:** The training uses **PPO** or **GRPO** (Group Relative Policy Optimization) as the underlying RL algorithm.
> 
> 

---

## Parallel Computing Implementation

A critical implementation detail of CausalRL is the handling of computational overhead. Calculating Jacobian matrices for every token generation is computationally intensive.

To address this, this implementation modifies the base **Verl** framework to utilize a **separate thread for reward calculation**.

* **Main Thread:** Handles the policy optimization (actor/critic updates).
* **Reward Thread:** Asynchronously computes the Jacobian-based causal influence scores, and performs the spectral energy normalization.



---

## Dataset Sources (Please refer to the data folder for download)

The repository includes support for the following academic datasets used in the paper's experiments to evaluate reasoning faithfulness and robustness.

Training & Validation Sets 

* **BBEHCausal:** The causal reasoning split from the BIG-Bench Extra Hard dataset.


* **CaseHOLD:** Multiple-choice legal holdings requiring context from case citations.


* **MATHHARD:** A subset of the MATH dataset focusing on multi-step reasoning (Level 5).


* **IfQA:** Open-domain QA requiring hypothetical reasoning under counterfactual presuppositions.



Testing & OOD (Out-of-Distribution) Sets 

* **BBEHMATH:** Challenging multistep arithmetic tasks.


* **CLadder:** A benchmark for association, intervention, and counterfactual queries.


* **LegalBench:** Case-understanding split for analyzing legal reasoning.


* **LogiQA:** Deductive logical reasoning from officer entrance exams.


* **TruthfulQA / CodeMMLU / SuperGPQA:** Used for evaluating generalization and robustness.


