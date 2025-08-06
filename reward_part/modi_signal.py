import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class RMSNormalizedSignalCalculator:
    """
    计算三路因果信号的 RMS 归一化版本，使得结果大约在 [0,1] 范围内：
      S(Z,X) = sqrt(mean((∂X/∂Z)^2))
      S(Z,Y) = sqrt(mean((∂Y/∂Z)^2))
      S(X,Y) = sqrt(mean((∂Y/∂X)^2))
    """

    def __init__(self, model_path: str, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(device)
        self.model.eval()

    def compute(self, Z: str, X: str, Y: str) -> tuple[float, float, float]:
        tok, model, dev = self.tokenizer, self.model, self.device

        # 编码并强制 LongTensor
        z_ids = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids.to(dev).long()
        x_ids = tok(X, add_special_tokens=False, return_tensors="pt").input_ids.to(dev).long()
        y_ids = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids.to(dev).long()
        eos = torch.tensor([[tok.eos_token_id]], device=dev)

        # --- 1. S(Z, X) RMS 归一化 ---
        zx_ids = torch.cat([z_ids, eos, x_ids], dim=-1)
        zx_emb = model.get_input_embeddings()(zx_ids).requires_grad_(True)
        zx_emb.retain_grad()
        zx_logits = model(inputs_embeds=zx_emb).logits
        zx_logp = F.log_softmax(zx_logits, dim=-1)

        # 定位 X 开始位置
        z_len = z_ids.size(1) + 1
        score_x = zx_logp[0, z_len:].gather(-1, zx_ids[0, z_len:].unsqueeze(-1)).sum()
        model.zero_grad()
        score_x.backward(retain_graph=True)
        grad_zx = zx_emb.grad[0]  # [seq_len, hidden_dim]
        block_z = grad_zx[:z_len]
        S_ZX = torch.sqrt((block_z**2).mean()).item()

        # --- 2. S(Z, Y) 和 S(X, Y) RMS 归一化 ---
        zxy_ids = torch.cat([z_ids, eos, x_ids, eos, y_ids], dim=-1)
        zxy_emb = model.get_input_embeddings()(zxy_ids).requires_grad_(True)
        zxy_emb.retain_grad()
        zxy_logits = model(inputs_embeds=zxy_emb).logits
        zxy_logp = F.log_softmax(zxy_logits, dim=-1)

        # 定位 Y 开始位置
        y_start = z_len + x_ids.size(1) + 1
        score_y = zxy_logp[0, y_start:].gather(-1, zxy_ids[0, y_start:].unsqueeze(-1)).sum()
        model.zero_grad()
        score_y.backward()
        grad_zxy = zxy_emb.grad[0]

        block_z_for_y = grad_zxy[:z_len]
        block_x_for_y = grad_zxy[z_len : z_len + x_ids.size(1) + 1]

        S_ZY = torch.sqrt((block_z_for_y**2).mean()).item()
        S_XY = torch.sqrt((block_x_for_y**2).mean()).item()

        return S_ZX/10, S_ZY/10, S_XY/10


# 使用示例
if __name__ == "__main__":
    model_dir = "/users/xwang76/hf_models/llama3-8b-instruct"
    calc = RMSNormalizedSignalCalculator(model_dir, device="cuda")

    Z = "Question: 为什么天空是蓝色的？"
    X = "1. 大气分子散射蓝光 2. 晴空时蓝光更易四处散射"
    Y = "因为大气分子对蓝光散射最明显，所以我们看到天空是蓝色的。"

    s_zx, s_zy, s_xy = calc.compute(Z, X, Y)
    print(f"S(Z,X)={s_zx:.4f}, S(Z,Y)={s_zy:.4f}, S(X,Y)={s_xy:.4f}")
