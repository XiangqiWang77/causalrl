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
    计算基于 Jacobian 的 0-1 归一化强度：
      D_ZX: J_ZX = ∂ s_X / ∂ Z_emb
      D_ZY: J_ZY^direct = ∂ s_Y / ∂ Z_emb （剥离 via-X）
      D_XY: J_XY = ∂ s_Y / ∂ X_emb
    关键优化：
      - 单次前向，双 autograd.grad（比两次前向+backward 快）
      - input_ids + Embedding hook（避免 inputs_embeds 克隆/复制）
      - autocast(bf16) + TF32
      - 可选：谱指标用幂迭代，免 SVD
    """
    def __init__(self, model_path: str, device="cuda", metric: str = "energy_top1",
                 max_z: int = None, max_x: int = None, max_y: int = None):
        self.device = device
        self.metric = metric
        self.max_z = max_z
        self.max_x = max_x
        self.max_y = max_y

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")  # Ampere+/Hopper 上有效

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(device)
        self.model.eval()
        # 训练时才需要 cache；这里做梯度分析，禁用 cache 更稳
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        # 嵌入层句柄（避免每次 get）
        self.emb_layer = self.model.get_input_embeddings()

    # ----------------- 可选：更快的矩阵指标（免SVD） -----------------
    @staticmethod
    def _top1_power(J: torch.Tensor, iters: int = 6) -> torch.Tensor:
        # 近似最大奇异值：幂迭代 ||J v||_2 / ||v||_2
        # J: [T, d]
        if J.numel() == 0:
            return torch.tensor(0.0, device=J.device, dtype=J.dtype)
        T, d = J.shape
        v = torch.randn(d, device=J.device, dtype=J.dtype)
        v = v / (v.norm() + 1e-8)
        for _ in range(iters):
            u = (J @ v)
            un = u.norm() + 1e-8
            u = u / un
            v = (J.T @ u)
            vn = v.norm() + 1e-8
            v = v / vn
        # 近似 σ_max
        s_max = (J @ v).norm()
        return s_max

    def _matrix_normalize_fast(self, J: torch.Tensor) -> float:
        # 一个“能量占比”的近似：σ1 / (σ1 + ε + Fro)
        # 既保留“头部能量”含义，又避免全 SVD 的开销
        if J.numel() == 0:
            return 0.0
        with torch.no_grad():
            s1 = self._top1_power(J.float())
            fro = torch.linalg.norm(J.float())  # Frobenius
            val = (s1 / (fro + 1e-12)).clamp(0, 1).item()
        return float(val)
    # ----------------------------------------------------------------

    def _clip_ids(self, ids: torch.Tensor, max_len: int | None) -> torch.Tensor:
        if max_len is None:
            return ids
        return ids[:, -max_len:] if ids.size(1) > max_len else ids

    def compute(self, Z: str, X: str, Y: str, return_debug: bool = False):
        tok, model, dev = self.tokenizer, self.model, self.device
        emb_layer = self.emb_layer

        # ---- tokenize（可选截断）----
        z_ids = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        x_ids = tok(X, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        y_ids = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

        if self.max_z: z_ids = self._clip_ids(z_ids, self.max_z)
        if self.max_x: x_ids = self._clip_ids(x_ids, self.max_x)
        if self.max_y: y_ids = self._clip_ids(y_ids, self.max_y)

        eos_id = tok.eos_token_id
        eos = torch.tensor([[eos_id]], device=dev)

        # 拼接：Z <eos> X <eos> Y
        zxy_ids = torch.cat([z_ids, eos, x_ids, eos, y_ids], dim=-1)  # [1, T]
        T = zxy_ids.size(1)

        # 位置切片
        z_len = z_ids.size(1) + 1                   # 含 Z 后的 eos
        x_len = x_ids.size(1) + 1                   # 含 X 后的 eos
        y_start = z_len + x_ids.size(1) + 1         # Y 起点
        # X 的 tokens 段（被预测的是位移后一个，所以 mask 需要错位）
        x_slice = slice(z_len, z_len + x_ids.size(1))
        y_slice = slice(y_start, y_start + y_ids.size(1))

        # -------- 单次前向 + 抓取 embedding 输出 --------
        emb_out_holder = {}
        def _emb_fwd_hook(mod, inp, out):
            emb_out_holder["emb"] = out  # [1, T, d]
        h = emb_layer.register_forward_hook(_emb_fwd_hook)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=zxy_ids, output_hidden_states=False)
            logits = out.logits  # [1, T, V]
            # 右移一位对齐标签（常规 LM loss 对齐方式）
            l_shift = logits[:, :-1, :]               # [1, T-1, V]
            tgt = zxy_ids[:, 1:]                      # [1, T-1]

            # 只对 X、Y 位置计算对数似然（避免无关位置）
            logp = F.log_softmax(l_shift, dim=-1)     # [1, T-1, V]
            # X 区间在对齐后是 [z_len-1, z_len-1 + |X| - 1]，更直观用 gather+切片：
            x_logp_tok = logp[:, x_slice, :].gather(-1, tgt[:, x_slice].unsqueeze(-1)).squeeze(-1)  # [1, |X|]
            y_logp_tok = logp[:, y_slice, :].gather(-1, tgt[:, y_slice].unsqueeze(-1)).squeeze(-1)  # [1, |Y|]
            score_x = x_logp_tok.sum()
            score_y = y_logp_tok.sum()

        emb_out = emb_out_holder["emb"]  # [1, T, d]
        h.remove()  # 及时移除 hook，避免重复注册

        # -------- 两次 autograd.grad（同一前向图）--------
        # 取对 emb_out 的梯度；不对权重构建/累积梯度，省内存省时间
        Jx = torch.autograd.grad(score_x, emb_out, retain_graph=True, allow_unused=False)[0][0].float()  # [T, d]
        Jy = torch.autograd.grad(score_y, emb_out, retain_graph=False, allow_unused=False)[0][0].float() # [T, d]

        # 切片得到各段 Jacobian
        J_ZX        = Jx[:z_len, :]                             # [|Z|+1, d]
        J_ZY_total  = Jy[:z_len, :]                             # [|Z|+1, d]
        J_XY        = Jy[z_len : z_len + x_len, :]              # [|X|+1, d]

        # -------- 剥离 via-X 分量（DCE）--------
        # 用 X->Y 的方向 u 做投影，去掉经 X 传递的分量
        u = J_XY.mean(dim=0, keepdim=True)                      # [1, d]
        v = (J_ZX * u).reshape(-1)                              # via-X 候选向量
        w = J_ZY_total.reshape(-1)                              # 总梯度
        if float((v @ v).item()) > 1e-12:
            alpha = float((w @ v).item() / ((v @ v).item() + 1e-8))
            w_direct = w - alpha * v
        else:
            w_direct = w
        J_ZY_direct = w_direct.view_as(J_ZY_total)

        # -------- 0-1 归一化指标 --------
        # 注：若你愿意免 SVD，把下行的 _matrix_normalize 换成 _matrix_normalize_fast
        D_ZX = _matrix_normalize(J_ZX,        metric=self.metric)
        D_ZY = _matrix_normalize(J_ZY_direct, metric=self.metric)
        D_XY = _matrix_normalize(J_XY,        metric=self.metric)
        # D_ZX = self._matrix_normalize_fast(J_ZX)
        # D_ZY = self._matrix_normalize_fast(J_ZY_direct)
        # D_XY = self._matrix_normalize_fast(J_XY)

        if not return_debug:
            return D_ZX, D_ZY, D_XY

        # 可选：调试统计（建议仅在需要时开启；SVD 很慢）
        def pack_stats_simple(J):
            with torch.no_grad():
                fro = torch.linalg.norm(J.float()).item()
                s1  = self._top1_power(J.float(), iters=6).item()
            return {"shape": tuple(J.shape), "fro": float(fro), "s1_approx": float(s1)}

        return (D_ZX, D_ZY, D_XY), {
            "ZX": pack_stats_simple(J_ZX),
            "ZY_direct": pack_stats_simple(J_ZY_direct),
            "XY": pack_stats_simple(J_XY),
        }

    # ----------------- 批处理（单 GPU，一次前向图） -----------------
    def compute_batch(self, Zs, Xs, Ys):
        """
        输入：等长列表 Zs/Xs/Ys（长度 B）
        输出：s_ZX_list, s_ZY_list, s_XY_list（长度 B）
        说明：同样只走“一次前向 + 两次 autograd.grad”，但把 batch 里各自的
             X/Y 分数先求和成两个标量，再各做一次 grad，最后按样本切片回填。
        """
        tok, model, dev = self.tokenizer, self.model, self.device
        emb_layer = self.emb_layer

        # 逐样本 tokenize 并拼接，随后 pad 到同长
        eos_id = tok.eos_token_id
        z_ids_list, x_ids_list, y_ids_list = [], [], []
        concat_list = []
        spans = []  # 记录每个样本的 z_len, x_len, y_start 等切片

        for Z, X, Y in zip(Zs, Xs, Ys):
            z = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids
            x = tok(X, add_special_tokens=False, return_tensors="pt").input_ids
            y = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids
            if self.max_z: z = self._clip_ids(z, self.max_z)
            if self.max_x: x = self._clip_ids(x, self.max_x)
            if self.max_y: y = self._clip_ids(y, self.max_y)

            z_len = z.size(1) + 1
            x_len = x.size(1) + 1
            y_start = z_len + x.size(1) + 1

            seq = torch.cat([z, torch.tensor([[eos_id]]), x, torch.tensor([[eos_id]]), y], dim=-1)  # [1, T_i]
            concat_list.append(seq)
            spans.append((z_len, x_len, y_start, x.size(1), y.size(1)))

        # pad
        maxT = max(s.size(1) for s in concat_list)
        ids = torch.full((len(concat_list), maxT), fill_value=tok.pad_token_id or 0, dtype=torch.long)
        for i, s in enumerate(concat_list):
            ids[i, :s.size(1)] = s[0]
        ids = ids.to(dev)

        # 前向 hook 抓 embedding 输出
        emb_out_holder = {}
        def _emb_fwd_hook(mod, inp, out):
            emb_out_holder["emb"] = out  # [B, T, d]
        h = emb_layer.register_forward_hook(_emb_fwd_hook)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, output_hidden_states=False)
            logits = out.logits  # [B, T, V]
            l_shift = logits[:, :-1, :]
            tgt     = ids[:, 1:]

            logp = F.log_softmax(l_shift, dim=-1)

            # 为所有样本构造两个“位置掩码”，累加成 batch 标量（两次 grad）
            score_x_total = torch.tensor(0.0, device=ids.device, dtype=logp.dtype)
            score_y_total = torch.tensor(0.0, device=ids.device, dtype=logp.dtype)

            for b, (z_len, x_len, y_start, x_raw_len, y_raw_len) in enumerate(spans):
                x_slice = slice(z_len, z_len + x_raw_len)
                y_slice = slice(y_start, y_start + y_raw_len)
                x_lp = logp[b:b+1, x_slice, :].gather(-1, tgt[b:b+1, x_slice].unsqueeze(-1)).squeeze(-1).sum()
                y_lp = logp[b:b+1, y_slice, :].gather(-1, tgt[b:b+1, y_slice].unsqueeze(-1)).squeeze(-1).sum()
                score_x_total = score_x_total + x_lp
                score_y_total = score_y_total + y_lp

        emb_out = emb_out_holder["emb"]  # [B, T, d]
        h.remove()

        Jx = torch.autograd.grad(score_x_total, emb_out, retain_graph=True, allow_unused=False)[0].float()  # [B,T,d]
        Jy = torch.autograd.grad(score_y_total, emb_out, retain_graph=False, allow_unused=False)[0].float() # [B,T,d]

        s_ZX_list, s_ZY_list, s_XY_list = [], [], []
        for b, (z_len, x_len, y_start, x_raw_len, y_raw_len) in enumerate(spans):
            J_ZX        = Jx[b, :z_len, :]
            J_ZY_total  = Jy[b, :z_len, :]
            J_XY        = Jy[b, z_len : z_len + x_len, :]

            # via-X 投影剥离
            u = J_XY.mean(dim=0, keepdim=True)
            v = (J_ZX * u).reshape(-1)
            w = J_ZY_total.reshape(-1)
            if float((v @ v).item()) > 1e-12:
                alpha = float((w @ v).item() / ((v @ v).item() + 1e-8))
                w_direct = w - alpha * v
            else:
                w_direct = w
            J_ZY_direct = w_direct.view_as(J_ZY_total)

            # 归一化
            DZ  = _matrix_normalize(J_ZX)
            DYZ = _matrix_normalize(J_ZY_direct)
            DXY = _matrix_normalize(J_XY)
            # 若需更快：改用 self._matrix_normalize_fast

            s_ZX_list.append(float(DZ))
            s_ZY_list.append(float(DYZ))
            s_XY_list.append(float(DXY))

        return s_ZX_list, s_ZY_list, s_XY_list
