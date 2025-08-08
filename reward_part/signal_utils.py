import re
from typing import List
import torch
import re
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    StoppingCriteria, StoppingCriteriaList
)
from typing import Tuple

MODEL_NAME = "/users/xwang76/hf_models/qwen3-4b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
device = next(model.parameters()).device

class StopOnSubstr(StoppingCriteria):
    def __init__(self, stop_str: str, tokenizer: AutoTokenizer):
        super().__init__()
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids.shape[-1] < len(self.stop_ids):
            return False
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids

# prepare stopping criteria
stop_at_answer    = StoppingCriteriaList([StopOnSubstr("<finalanswer>", tokenizer)])
stop_at_endanswer = StoppingCriteriaList([StopOnSubstr("</finalanswer>", tokenizer)])

def g(Z: str, max_new_tokens: int = 512) -> str:
    """
    g(Z) → X: generate the chain-of-thought up to (but not including) "<finalanswer>"
    """
    inputs = tokenizer(Z, return_tensors="pt").to(device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stop_at_answer,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    full = tokenizer.decode(out_ids, skip_special_tokens=False)
    # strip Z prefix and the trailing "<answer>"
    X = full[len(Z):].rsplit("<finalanswer>", 1)[0]
    return X.strip()

def f(Z: str, X: str, max_new_tokens: int = 128) -> str:
    """
    f(Z, X) → Y: given Z and CoT X, generate the final answer Y up to "</finalanswer>"
    """
    prompt = Z + X + "<finalanswer>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stop_at_endanswer,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    full = tokenizer.decode(out_ids, skip_special_tokens=False)
    # strip prompt prefix and the trailing "</finalanswer>"
    Y = full[len(prompt):].rsplit("</finalanswer>", 1)[0]
    return Y.strip()

def extract_XY(resp: str) -> Tuple[str, str]:
    """
    Given a model output string resp containing "<finalanswer>Y</finalanswer>",
    return (X, Y) where:
      X = resp before "<finalanswer>"
      Y = the text inside "<finalanswer>…</finalanswer>"
    """
    m = re.search(r"<finalanswer>(.*?)</finalanswer>", resp, re.DOTALL)
    Y = m.group(1).strip() if m else ""
    X = resp.split("<finalanswer>")[0]
    return X, Y




def _compute_grads_matrix(full: str, pos: int, p: float) -> torch.Tensor:
    """
    Runs a forward+backward pass on `full` (tokenized without special tokens),
    takes the p-norm of logits at position `pos`, backprops, and returns the
    gradient tensor [seq_len, embed_dim].
    """
    enc = tokenizer(full, return_tensors="pt", add_special_tokens=False).to(device)
    ids = enc.input_ids              # [1, seq_len]
    mask = torch.ones_like(ids)
    embeds = model.get_input_embeddings()(ids).detach()
    embeds.requires_grad_(True)

    out = model(inputs_embeds=embeds, attention_mask=mask)
    logits = out.logits[:, pos, :]    # [1, vocab_size]
    score = logits.norm(p=p)
    score.backward()

    return embeds.grad[0]             # [seq_len, embed_dim]

def _split_sentences(text: str) -> List[str]:
    """
    Naively split on sentence-ending punctuation.
    """
    sents = re.split(r'(?<=[\.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def S_ZY(Z: str, p: float = 2.0) -> float:
    """
    Direct Causal Effect of Z on Y: S(Z→Y)
    """
    X = g(Z)
    Y = f(Z, X)
    full = Z + X + "<finalanswer>" + Y + "</finalanswer>"

    len_Z = len(tokenizer(Z, add_special_tokens=False).input_ids)
    len_X = len(tokenizer(X, add_special_tokens=False).input_ids)
    pos_Y = len(tokenizer(full, add_special_tokens=False).input_ids) - 1

    grads = _compute_grads_matrix(full, pos_Y, p)
    # gradient norm over Z segment
    return grads[:len_Z].norm(p=p).item()

def S_XY(Z: str, p: float = 2.0) -> float:
    """
    Direct Causal Effect of X on Y: S(X→Y)
    """
    X = g(Z)
    Y = f(Z, X)
    full = Z + X + "<finalanswer>" + Y + "</finalanswer>"

    len_Z = len(tokenizer(Z, add_special_tokens=False).input_ids)
    len_X = len(tokenizer(X, add_special_tokens=False).input_ids)
    pos_Y = len(tokenizer(full, add_special_tokens=False).input_ids) - 1

    grads = _compute_grads_matrix(full, pos_Y, p)
    # gradient norm over X segment
    return grads[len_Z:len_Z+len_X].norm(p=p).item()

def S_ZX(Z: str, p: float = 2.0) -> float:
    """
    Direct Causal Effect of Z on X: S(Z→X)
    """
    X = g(Z)
    full = Z + X

    len_Z = len(tokenizer(Z, add_special_tokens=False).input_ids)
    len_X = len(tokenizer(X, add_special_tokens=False).input_ids)
    pos_X = len(tokenizer(full, add_special_tokens=False).input_ids) - 1

    grads = _compute_grads_matrix(full, pos_X, p)
    # gradient norm over Z segment
    return grads[:len_Z].norm(p=p).item()

def S_X_in(Z: str, p: float = 2.0) -> float:
    """
    Internal coherence of X: average DCE among all sentence pairs in X
    """
    X = g(Z)
    full = Z + X

    len_Z = len(tokenizer(Z, add_special_tokens=False).input_ids)
    sentences = _split_sentences(X)
    tokenized = [tokenizer(s, add_special_tokens=False).input_ids for s in sentences]
    sent_lens = [len(toks) for toks in tokenized]
    # prefix sums to locate sentence boundaries
    prefix = [0]
    for l in sent_lens[:-1]:
        prefix.append(prefix[-1] + l)

    dce_values = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            # position of end-of-sentence j in the full sequence
            pos_j = len_Z + prefix[j] + sent_lens[j] - 1
            grads = _compute_grads_matrix(full, pos_j, p)
            # gradient norm of sentence i segment
            start_i = len_Z + prefix[i]
            end_i   = start_i + sent_lens[i]
            g_i = grads[start_i:end_i].norm(p=p).item()
            dce_values.append(g_i)

    return float(sum(dce_values) / len(dce_values)) if dce_values else 0.0
