import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def _rms(t: torch.Tensor) -> torch.Tensor:
    return (t.float().pow(2).mean().sqrt())

def _svd_stats(J: torch.Tensor, tol: float = 1e-6):
    """
    对 J ∈ R^{m×d} 做 SVD，返回：
      - s: 奇异值 (降序)
      - fro2: Frobenius 能量 (∑ s_i^2)
      - nuc: 核范数 (∑ s_i)
      - op: 谱范数 (s_1)
      - rank: 数值秩 (s_i > tol * s_1)
      - rmax: min(m,d)
    """
    m, d = J.shape
    # 用 float32 做 SVD 更稳
    s = torch.linalg.svdvals(J.float())  # [min(m,d)]
    op = s[0]
    fro2 = torch.sum(s**2)
    nuc = torch.sum(s)
    rmax = min(m, d)
    thr = float(max(op.item() * tol, 1e-12))
    rank = int((s > thr).sum().item())
    return s, fro2, nuc, op, rank, rmax

def _matrix_normalize(J: torch.Tensor, metric: str = "energy_top1") -> float:
    """
    用纯矩阵指标把 J 归一化到 [0,1]（无参数、无状态）。
    可选 metric：
      - 'energy_top1'     : σ1^2 / ∑σi^2                （默认；衡量主方向支配度）
      - 'spectral_ratio'  : σ1 / ∑σi                    （Top-1 相对谱重）
      - 'stable_rank'     : (∥J∥_F^2/∥J∥_2^2) / rmax    （稳定秩 / 最大秩）
      - 'cond_energy'     : (σ1-σ2)/σ1  （若 rank>=2，否则 1.0）
    """
    s, fro2, nuc, op, rank, rmax = _svd_stats(J)
    eps = 1e-12
    if metric == "energy_top1":
        val = (s[0]**2 / (fro2 + eps)).clamp(0, 1)          # ∈ (0,1]
    elif metric == "spectral_ratio":
        val = (s[0] / (nuc + eps)).clamp(0, 1)              # ∈ (0,1]
    elif metric == "stable_rank":
        # 稳定秩 sr = ||J||_F^2 / ||J||_2^2 ∈ [1, rmax]，再除 rmax → [0,1]
        sr = (fro2 / (op**2 + eps))
        val = (sr / rmax).clamp(0, 1)
    elif metric == "cond_energy":
        if len(s) >= 2:
            val = ((s[0] - s[1]) / (s[0] + eps)).clamp(0, 1)
        else:
            val = torch.tensor(1.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return float(val.item())

class RMSNormalizedSignalCalculator:
    """
    直接返回基于 Jacobian 的 0-1 归一化强度（纯矩阵指标）：
      D_ZX: 由 J_ZX = ∂ s_X / ∂ Z_emb 得（Z->X）
      D_ZY: 由 J_ZY^direct = ∂ s_Y / ∂ Z_emb 剥离 via-X 后得（Z->Y 的 DCE）
      D_XY: 由 J_XY = ∂ s_Y / ∂ X_emb 得（X->Y）
    归一化 metric 默认 'energy_top1'，可改为 ['spectral_ratio','stable_rank','cond_energy']。
    """
    def __init__(self, model_path: str, device="cuda", metric: str = "energy_top1"):
        self.device = device
        self.metric = metric
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(device)
        self.model.eval()

    def compute(self, Z: str, X: str, Y: str,
                return_debug: bool = False) -> tuple[float, float, float] | tuple:
        tok, model, dev = self.tokenizer, self.model, self.device
        emb_layer = model.get_input_embeddings()

        # ---- tokenize ----
        z_ids = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids.to(dev).long()
        x_ids = tok(X, add_special_tokens=False, return_tensors="pt").input_ids.to(dev).long()
        y_ids = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids.to(dev).long()
        eos = torch.tensor([[tok.eos_token_id]], device=dev)

        # === (1) J_ZX: ∂ s_X / ∂ Z_emb ===
        zx_ids = torch.cat([z_ids, eos, x_ids], dim=-1)
        zx_emb = emb_layer(zx_ids).detach().clone().requires_grad_(True)
        zx_logits = model(inputs_embeds=zx_emb).logits
        zx_logp = F.log_softmax(zx_logits, dim=-1)

        z_len = z_ids.size(1) + 1  # include eos between Z and X
        score_x = zx_logp[0, z_len:].gather(-1, zx_ids[0, z_len:].unsqueeze(-1)).sum()
        model.zero_grad(set_to_none=True)
        score_x.backward(retain_graph=False)
        grad_zx = zx_emb.grad[0].float()          # [|Z|+1+|X|, d]
        J_ZX = grad_zx[:z_len]                    # [|Z|+1, d]

        # === (2) J_ZY(total), J_XY: from Y score ===
        zxy_ids = torch.cat([z_ids, eos, x_ids, eos, y_ids], dim=-1)
        zxy_emb = emb_layer(zxy_ids).detach().clone().requires_grad_(True)
        zxy_logits = model(inputs_embeds=zxy_emb).logits
        zxy_logp = F.log_softmax(zxy_logits, dim=-1)

        y_start = z_len + x_ids.size(1) + 1
        score_y = zxy_logp[0, y_start:].gather(-1, zxy_ids[0, y_start:].unsqueeze(-1)).sum()
        model.zero_grad(set_to_none=True)
        score_y.backward(retain_graph=False)
        grad_zxy = zxy_emb.grad[0].float()

        J_ZY_total = grad_zxy[:z_len]                             # [|Z|+1, d]
        J_XY       = grad_zxy[z_len : z_len + x_ids.size(1) + 1]  # [|X|+1, d]

        # === (3) DCE for Z->Y: 剥离 via-X（搬运 + 投影） ===
        # 把 X->Y 的敏感方向池化成 u，把 J_ZX 搬运到与 J_ZY_total 同一空间，再正交化剔除
        u = J_XY.mean(dim=0, keepdim=True)         # [1, d]
        v = (J_ZX * u).reshape(-1)                 # via-X 候选通道向量
        w = J_ZY_total.reshape(-1)                 # 总梯度向量

        if float((v @ v).item()) > 1e-12:
            alpha = float((w @ v).item() / ((v @ v).item() + 1e-8))
            w_direct = w - alpha * v
        else:
            w_direct = w

        J_ZY_direct = w_direct.view_as(J_ZY_total)

        # === (4) 纯矩阵指标做 0-1 归一化 ===
        D_ZX = _matrix_normalize(J_ZX,        metric=self.metric)
        D_ZY = _matrix_normalize(J_ZY_direct, metric=self.metric)
        D_XY = _matrix_normalize(J_XY,        metric=self.metric)

        if not return_debug:
            return D_ZX, D_ZY, D_XY

        # 可选：回传一些原始矩阵统计，便于核查
        def pack_stats(J):
            s, fro2, nuc, op, rank, rmax = _svd_stats(J)
            return {
                "shape": tuple(J.shape),
                "spectral": float(op.item()),
                "fro": float(torch.sqrt(fro2 + 1e-12).item()),
                "nuclear": float(nuc.item()),
                "rank": rank,
                "rmax": rmax,
                "s_top3": [float(x) for x in s[:3].tolist()],
            }

        return (D_ZX, D_ZY, D_XY), {
            "ZX": pack_stats(J_ZX),
            "ZY_direct": pack_stats(J_ZY_direct),
            "XY": pack_stats(J_XY),
        }





