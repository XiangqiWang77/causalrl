# -*- coding: utf-8 -*-
"""
Minimal Counterfactual-Enhanced Jacobian (strong token reshuffle, residual-only).
Multi-GPU aware version: model weights sharded across multiple GPUs with
device_map="balanced" and per-GPU max_memory built from env.

ENV knobs (new):
  CUDA_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
  MAX_MEMORY_FRACTION=0.90   # fraction of each GPU VRAM used for weights

Existing:
  CFJ_K (default 4) — number of counterfactual draws
  JAC_MIN_FLOOR (default 0.005) — score floor
"""

import os, re, math, random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- minimal knobs ----
CFJ_K         = int(os.getenv("CFJ_K", "1"))
JAC_MIN_FLOOR = float(os.getenv("JAC_MIN_FLOOR", "0.005"))

# ---- multi-GPU helpers ----
def _normalize_devices(devs: str) -> List[str]:
    out: List[str] = []
    for d in devs.split(","):
        d = d.strip()
        if not d:
            continue
        if d.isdigit():
            out.append(f"cuda:{d}")
        else:
            out.append(d)
    return out

def _build_max_memory_map(CUDA_DEVICES: List[str], frac: float) -> Dict[int, str]:
    max_mem: Dict[int, str] = {}
    if not torch.cuda.is_available():
        return max_mem
    for d in CUDA_DEVICES:
        if not d.startswith("cuda"):
            continue
        idx = torch.device(d).index
        if idx is None:
            continue
        props = torch.cuda.get_device_properties(idx)
        gb = max(1, int(props.total_memory / (1024**3) * float(frac)))
        max_mem[idx] = f"{gb}GiB"
    return max_mem

# ---- tiny utils ----
def _squash_floor01(v: float, floor: float = JAC_MIN_FLOOR) -> float:
    v = 0.0 if v != v else float(v)  # NaN->0
    v = max(0.0, min(1.0, v))
    return float(floor + (1.0 - floor) * v)

def _strong_reshuffle(text: str) -> str:
    if not text:
        return text
    parts = re.findall(r'\S+|\s+', text)
    tokens = [p for p in parts if not p.isspace()]
    spaces = [p for p in parts if p.isspace()]
    if len(tokens) <= 1:
        return text
    random.shuffle(tokens)
    out = []
    t_i = 0; s_i = 0
    starts_with_token = not parts[0].isspace()
    while t_i < len(tokens) or s_i < len(spaces):
        if starts_with_token:
            if t_i < len(tokens):
                out.append(tokens[t_i]); t_i += 1
            if s_i < len(spaces):
                out.append(spaces[s_i]); s_i += 1
        else:
            if s_i < len(spaces):
                out.append(spaces[s_i]); s_i += 1
            if t_i < len(tokens):
                out.append(tokens[t_i]); t_i += 1
    return "".join(out)

def _svdvals(J: torch.Tensor):
    return torch.linalg.svdvals(J.float())

def _energy_top1(J: torch.Tensor) -> float:
    if J is None or J.numel() == 0:
        return 0.0
    s = _svdvals(J)
    fro2 = torch.sum(s**2)
    if fro2 <= 0:
        return 0.0
    val = (s[0]**2 / fro2).clamp(0, 1)
    return float(val.item())

def _norm_ratio_residual(J_res: torch.Tensor, J_base: torch.Tensor) -> float:
    if J_res is None or J_res.numel() == 0 or J_base is None or J_base.numel() == 0:
        return 0.0
    nr = float(torch.sum(J_res.float()**2).item())
    nb = float(torch.sum(J_base.float()**2).item()) + 1e-12
    r = nr / nb
    return float(max(0.0, min(1.0, r)))

def _proj_residual(J_base: torch.Tensor, J_cf_mean: torch.Tensor | None) -> torch.Tensor:
    if J_cf_mean is None or J_cf_mean.numel() == 0 or J_base is None or J_base.numel() == 0:
        return J_base
    a = (J_base.reshape(-1) @ J_cf_mean.reshape(-1)).item()
    b = (J_cf_mean.reshape(-1) @ J_cf_mean.reshape(-1)).item()
    if b <= 1e-12:
        return J_base
    return J_base - (a / (b + 1e-12)) * J_cf_mean

def _align_mean(mats: List[torch.Tensor], ref: torch.Tensor | None) -> torch.Tensor | None:
    if ref is None or ref.numel() == 0 or not mats:
        return None
    T, D = ref.shape
    aligned = []
    for M in mats:
        if M is None or M.numel() == 0:
            continue
        t, d = M.shape
        if d != D:
            d_common = min(d, D)
            M = M[:, :d_common]
            if d_common < D:
                pad_c = torch.zeros((t, D - d_common), device=M.device, dtype=M.dtype)
                M = torch.cat([M, pad_c], dim=1)
        if t < T:
            pad_r = torch.zeros((T - t, D), device=M.device, dtype=M.dtype)
            M = torch.cat([M, pad_r], dim=0)
        elif t > T:
            M = M[:T, :]
        aligned.append(M)
    if not aligned:
        return None
    return torch.stack(aligned, dim=0).mean(dim=0)

# ---- main ----
class RMSNormalizedSignalCalculator:
    """
    Multi-GPU aware calculator:
    - If multiple CUDA devices are visible (via env CUDA_DEVICES), load the model with
      device_map="balanced" and per-GPU max_memory constructed from MAX_MEMORY_FRACTION.
    - Otherwise, fall back to single device .to(device).
    - Supports optional hf_load_kwargs to override loading behavior.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_z: Optional[int] = None,
        max_x: Optional[int] = None,
        max_y: Optional[int] = None,
        hf_load_kwargs: Optional[dict] = None,
    ):
        self.device = device
        self.max_z, self.max_x, self.max_y = max_z, max_x, max_y

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

        # --- decide multi-GPU sharding ---
        CUDA_DEVICES_STR = os.environ.get("CUDA_DEVICES", "").strip()
        CUDA_DEVICES = _normalize_devices(CUDA_DEVICES_STR) if CUDA_DEVICES_STR else []
        use_multi = torch.cuda.is_available() and len(CUDA_DEVICES) >= 2

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if hf_load_kwargs is None:
            hf_load_kwargs = {}

        if use_multi:
            # build max_memory from env
            max_mem = _build_max_memory_map(
                CUDA_DEVICES=CUDA_DEVICES,
                frac=float(os.environ.get("MAX_MEMORY_FRACTION", "0.90")),
            )
            # default multi-GPU kwargs (can be overridden by hf_load_kwargs)
            default_multi = dict(
                device_map="balanced",
                max_memory=max_mem if max_mem else None,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            for k, v in default_multi.items():
                hf_load_kwargs.setdefault(k, v)

            # do NOT .to(device) when using device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                **hf_load_kwargs,
            )
            # derive embedding device
            self.emb_layer = self.model.get_input_embeddings()
            self.emb_device = next(self.emb_layer.parameters()).device
        else:
            # single device fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                **hf_load_kwargs,
            ).to(device)
            self.emb_layer = self.model.get_input_embeddings()
            self.emb_device = next(self.emb_layer.parameters()).device

        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    @staticmethod
    def _clip(ids: torch.Tensor, max_len: Optional[int]) -> torch.Tensor:
        if max_len is None or ids.size(1) <= max_len:
            return ids
        return ids[:, -max_len:]

    def _forward_and_jac(self, Z: str, X: str, Y: str):
        tok, model = self.tokenizer, self.model
        dev0 = self.emb_device if torch.cuda.is_available() else torch.device("cpu")

        # Keep tokenization on CPU; move to embedding device right before forward
        z_ids = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids
        x_ids = tok(X, add_special_tokens=False, return_tensors="pt").input_ids
        y_ids = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids
        if self.max_z: z_ids = self._clip(z_ids, self.max_z)
        if self.max_x: x_ids = self._clip(x_ids, self.max_x)
        if self.max_y: y_ids = self._clip(y_ids, self.max_y)

        z_ids = z_ids.to(dev0)
        x_ids = x_ids.to(dev0)
        y_ids = y_ids.to(dev0)

        eos = torch.tensor([[tok.eos_token_id]], device=dev0)
        zxy_ids = torch.cat([z_ids, eos, x_ids, eos, y_ids], dim=-1)

        z_len_raw = z_ids.size(1); x_len_raw = x_ids.size(1); y_len_raw = y_ids.size(1)
        z_len = z_len_raw + 1
        x_start = z_len
        y_start = z_len + x_len_raw + 1
        x_slice = slice(x_start, x_start + x_len_raw)
        y_slice = slice(y_start, y_start + y_len_raw)

        emb_holder: Dict[str, torch.Tensor] = {}
        def _emb_hook(mod, inp, out): emb_holder["emb"] = out
        h = self.emb_layer.register_forward_hook(_emb_hook)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=zxy_ids)
            logits = out.logits
            logp = F.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = zxy_ids[:, 1:]
            x_logp_tok = logp[:, x_slice, :].gather(-1, tgt[:, x_slice].unsqueeze(-1)).squeeze(-1)
            y_logp_tok = logp[:, y_slice, :].gather(-1, tgt[:, y_slice].unsqueeze(-1)).squeeze(-1)
            score_x = x_logp_tok.sum()
            score_y = y_logp_tok.sum()

        emb_out = emb_holder["emb"]; h.remove()

        # autograd across devices is supported; grads are taken wrt embedding output
        Jx = torch.autograd.grad(score_x, emb_out, retain_graph=True,  allow_unused=False)[0][0].float()
        Jy = torch.autograd.grad(score_y, emb_out, retain_graph=False, allow_unused=False)[0][0].float()

        # base Jacobians for channels
        J_ZX       = Jx[:z_len-1, :]                   # dX/dZ
        J_ZY_total = Jy[:z_len-1, :]                   # dY/dZ (total)
        J_XY       = Jy[z_len : z_len + x_len_raw, :]  # dY/dX

        return (J_ZX, J_ZY_total, J_XY)

    @staticmethod
    def _dce(J_ZX: torch.Tensor, J_ZY_total: torch.Tensor, J_XY: torch.Tensor) -> torch.Tensor:
        if J_XY is None or J_XY.numel() == 0:
            return J_ZY_total
        u_rows = J_XY / (J_XY.norm(dim=-1, keepdim=True) + 1e-8)
        u = u_rows.mean(dim=0, keepdim=True)
        v = (J_ZX * u).reshape(-1)
        w = J_ZY_total.reshape(-1)
        denom = float((v @ v).item())
        if denom > 1e-12:
            alpha = float((w @ v).item() / (denom + 1e-8))
            return (w - alpha * v).view_as(J_ZY_total)
        return J_ZY_total

    def _score_channel(self, J_base: torch.Tensor, cf_mats: List[torch.Tensor]) -> float:
        J_cf_mean = _align_mean(cf_mats, J_base)
        J_res = _proj_residual(J_base, J_cf_mean)
        shape = _energy_top1(J_res)
        mag   = _norm_ratio_residual(J_res, J_base)
        s = math.sqrt(max(0.0, min(1.0, shape)) * max(0.0, min(1.0, mag)))
        return _squash_floor01(s)

    # -------- single example --------
    def compute_cf_enhanced(self, Z: str, X: str, Y: str) -> Tuple[float, float, float]:
        J_ZX_b, J_ZYt_b, J_XY_b = self._forward_and_jac(Z, X, Y)
        J_ZY_b = self._dce(J_ZX_b, J_ZYt_b, J_XY_b)

        ZX_cf: List[torch.Tensor] = []
        ZY_cf: List[torch.Tensor] = []
        XY_cf: List[torch.Tensor] = []
        K = max(1, CFJ_K)
        for _ in range(K):
            Zt = _strong_reshuffle(Z)
            Xt = _strong_reshuffle(X)

            J_ZXc, J_ZYtc, _ = self._forward_and_jac(Zt, X, Y)
            J_ZYc = self._dce(J_ZXc, J_ZYtc, J_XY_b)
            ZX_cf.append(J_ZXc); ZY_cf.append(J_ZYc)

            _, _, J_XYc = self._forward_and_jac(Z, Xt, Y)
            XY_cf.append(J_XYc)

        s_ZX = self._score_channel(J_ZX_b, ZX_cf)
        s_ZY = self._score_channel(J_ZY_b, ZY_cf)
        s_XY = self._score_channel(J_XY_b, XY_cf)
        return float(s_ZX), float(s_ZY), float(s_XY)

    # -------- batch --------
    def compute_batch_cf_enhanced(self, Zs: List[str], Xs: List[str], Ys: List[str]) -> Tuple[List[float], List[float], List[float]]:
        s_ZX_list: List[float] = []
        s_ZY_list: List[float] = []
        s_XY_list: List[float] = []
        for Z, X, Y in zip(Zs, Xs, Ys):
            s_ZX, s_ZY, s_XY = self.compute_cf_enhanced(Z, X, Y)
            s_ZX_list.append(s_ZX); s_ZY_list.append(s_ZY); s_XY_list.append(s_XY)
        return s_ZX_list, s_ZY_list, s_XY_list
