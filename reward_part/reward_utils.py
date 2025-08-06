import re
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    StoppingCriteria, StoppingCriteriaList
)
from typing import Tuple

# — assume these are initialized once elsewhere —
# MODEL_NAME = "/path/to/your/model"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
# model.eval()
# device = next(model.parameters()).device

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
