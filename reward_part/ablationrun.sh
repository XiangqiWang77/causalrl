#!/usr/bin/env bash
set -euo pipefail

# ==== 多 GPU 可见与分片均衡 ====
export CUDA_VISIBLE_DEVICES=0,1,2,3                 # 4 张卡可见
export CUDA_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"   # 供服务内做分片/均衡
export MAX_MEMORY_FRACTION=0.90                     # 每卡最多用 90% 显存装权重（可按需调）

# ==== 评分组件放置（避免把一张卡压爆；如需放 GPU 改成 cuda:0 等）====
export BERTSCORE_DEVICE=cpu

# ====（可选）运行时安全阈值；不需要可删 ====
export CUDA_FREE_MEM_FRACTION=0.85
export SAFE_MIN_FREE_MB=1000
export MAX_TOKENS_PER_BATCH=1024
export MAX_BATCH_SIZE=4
export CLEAR_CACHE_EVERY=5
export RESERVED_WATERMARK=0.92

# ==== 日志 ====
LOG_DIR="/users/xwang76/nano_rl/reward_part"
mkdir -p "$LOG_DIR"
LOG_NAME="llamablation_$(date +%Y%m%d_%H%M%S).log"

# ==== 启动（workers=1：单进程内多卡分片；多 worker 会重复占显存）====
uvicorn ablation_server:app \
  --host 0.0.0.0 --port 6010 \
  --workers 1 \
  --timeout-keep-alive 75 \
  > "${LOG_DIR}/${LOG_NAME}" 2>&1 & disown
