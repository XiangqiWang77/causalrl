# -*- coding: utf-8 -*-
"""
Naive Jacobian scoring (no counterfactuals, easy to game).
Outputs three scores in [JAC_MIN_FLOOR, 1]: S(Z->X), S(Z->Y direct), S(X->Y).

Core (intentionally hackable):
- Compute base Jacobians for three channels:
    J_ZX = d (log p(X)) / d Z-emb
    J_ZY_total = d (log p(Y)) / d Z-emb
    J_XY = d (log p(Y)) / d X-emb
- Direct Z->Y uses a tiny linear removal along mean X->Y direction (optional, still naive).
- Score for a channel is ONLY the top-1 energy share of its Jacobian (spectral energy concentration):
      shape = (s_max^2) / sum(s_i^2)
  then squashed into [JAC_MIN_FLOOR, 1]. No residuals, no counterfactuals, no gates.
"""

import os, math, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- single (minimal) knob ----
JAC_MIN_FLOOR = float(os.getenv("JAC_MIN_FLOOR", "0.005"))

# ---- utils ----
def _squash_floor01(v: float, floor: float = JAC_MIN_FLOOR) -> float:
    v = 0.0 if v != v else float(v)  # NaN->0
    v = max(0.0, min(1.0, v))
    return float(floor + (1.0 - floor) * v)

def _svdvals(J: torch.Tensor):
    return torch.linalg.svdvals(J.float())

def _energy_top1(J: torch.Tensor) -> float:
    # shape-only (hackable): concentrate energy to 1st singular value -> score↑
    if J is None or J.numel() == 0:
        return 0.0
    s = _svdvals(J)
    fro2 = torch.sum(s**2)
    if fro2 <= 0:
        return 0.0
    val = (s[0]**2 / fro2).clamp(0, 1)
    return float(val.item())

# ---- main ----
class RMSNormalizedSignalCalculator:
    def __init__(self, model_path: str, device: str | None = None,
                 max_z: int | None = None, max_x: int | None = None, max_y: int | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_z, self.max_x, self.max_y = max_z, max_x, max_y

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(self.device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.emb_layer = self.model.get_input_embeddings()

    @staticmethod
    def _clip(ids: torch.Tensor, max_len: int | None) -> torch.Tensor:
        if max_len is None or ids.size(1) <= max_len:
            return ids
        return ids[:, -max_len:]

    def _forward_and_jac(self, Z: str, X: str, Y: str):
        tok, model, dev = self.tokenizer, self.model, self.device
        z_ids = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        x_ids = tok(X, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        y_ids = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        if self.max_z: z_ids = self._clip(z_ids, self.max_z)
        if self.max_x: x_ids = self._clip(x_ids, self.max_x)
        if self.max_y: y_ids = self._clip(y_ids, self.max_y)

        eos_id = tok.eos_token_id
        if eos_id is None:
            # fallback: try sep or pad, else last vocab id
            eos_id = getattr(tok, "sep_token_id", None) or getattr(tok, "pad_token_id", None) or (tok.vocab_size - 1)
        eos = torch.tensor([[eos_id]], device=dev)

        # [Z] <eos> [X] <eos> [Y]
        zxy_ids = torch.cat([z_ids, eos, x_ids, eos, y_ids], dim=-1)

        z_len_raw = z_ids.size(1); x_len_raw = x_ids.size(1); y_len_raw = y_ids.size(1)
        z_len = z_len_raw + 1  # include eos after Z
        x_start = z_len
        y_start = z_len + x_len_raw + 1
        x_slice = slice(x_start, x_start + x_len_raw)
        y_slice = slice(y_start, y_start + y_len_raw)

        emb_holder = {}
        def _emb_hook(mod, inp, out): emb_holder["emb"] = out
        h = self.emb_layer.register_forward_hook(_emb_hook)

        with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            enabled=True):
            out = model(input_ids=zxy_ids)
            logits = out.logits
            logp = F.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = zxy_ids[:, 1:]
            x_logp_tok = logp[:, x_slice, :].gather(-1, tgt[:, x_slice].unsqueeze(-1)).squeeze(-1)
            y_logp_tok = logp[:, y_slice, :].gather(-1, tgt[:, y_slice].unsqueeze(-1)).squeeze(-1)
            score_x = x_logp_tok.sum()
            score_y = y_logp_tok.sum()

        emb_out = emb_holder["emb"]; h.remove()

        # grads wrt the *embedding outputs* (one step)
        Jx = torch.autograd.grad(score_x, emb_out, retain_graph=True,  allow_unused=False)[0][0].float()
        Jy = torch.autograd.grad(score_y, emb_out, retain_graph=False, allow_unused=False)[0][0].float()

        # base Jacobians for channels
        J_ZX       = Jx[:z_len-1, :]                   # dX/dZ
        J_ZY_total = Jy[:z_len-1, :]                   # dY/dZ (total)
        J_XY       = Jy[z_len : z_len + x_len_raw, :]  # dY/dX

        return (J_ZX, J_ZY_total, J_XY)

    @staticmethod
    def _dce(J_ZX: torch.Tensor, J_ZY_total: torch.Tensor, J_XY: torch.Tensor) -> torch.Tensor:
        """
        Very naive 'direct' Z->Y: remove component along mean X->Y direction by scalar regression.
        Still trivial to game; kept only to mimic a DCE-like readout without robustness.
        """
        if J_XY is None or J_XY.numel() == 0:
            return J_ZY_total
        u_rows = J_XY / (J_XY.norm(dim=-1, keepdim=True) + 1e-8)
        u = u_rows.mean(dim=0, keepdim=True)                    # 1 x D
        v = (J_ZX * u).reshape(-1)                              # project dX/dZ onto u
        w = J_ZY_total.reshape(-1)
        denom = float((v @ v).item())
        if denom > 1e-12:
            alpha = float((w @ v).item() / (denom + 1e-8))
            return (w - alpha * v).view_as(J_ZY_total)
        return J_ZY_total

    def _score_channel_naive(self, J_base: torch.Tensor) -> float:
        shape = _energy_top1(J_base)            # ONLY shape; no magnitude/length normalization
        return _squash_floor01(shape)

    # -------- single example (naive) --------
    def compute_naive(self, Z: str, X: str, Y: str):
        J_ZX_b, J_ZYt_b, J_XY_b = self._forward_and_jac(Z, X, Y)
        J_ZY_b = self._dce(J_ZX_b, J_ZYt_b, J_XY_b)

        s_ZX = self._score_channel_naive(J_ZX_b)
        s_ZY = self._score_channel_naive(J_ZY_b)
        s_XY = self._score_channel_naive(J_XY_b)

        return float(s_ZX), float(s_ZY), float(s_XY)

    # -------- batch (naive) --------
    def compute_batch_naive(self, Zs, Xs, Ys):
        s_ZX_list, s_ZY_list, s_XY_list = [], [], []
        for Z, X, Y in zip(Zs, Xs, Ys):
            s_ZX, s_ZY, s_XY = self.compute_naive(Z, X, Y)
            s_ZX_list.append(s_ZX); s_ZY_list.append(s_ZY); s_XY_list.append(s_XY)
        return s_ZX_list, s_ZY_list, s_XY_list
