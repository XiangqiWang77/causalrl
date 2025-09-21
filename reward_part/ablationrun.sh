#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=0,1,2,3                 
export CUDA_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"   #
export MAX_MEMORY_FRACTION=0.90                     # 


export BERTSCORE_DEVICE=cpu


export CUDA_FREE_MEM_FRACTION=0.85
export SAFE_MIN_FREE_MB=1000
export MAX_TOKENS_PER_BATCH=1024
export MAX_BATCH_SIZE=4
export CLEAR_CACHE_EVERY=5
export RESERVED_WATERMARK=0.92


LOG_DIR="/path/to/reward_part"
mkdir -p "$LOG_DIR"
LOG_NAME="llamablation_$(date +%Y%m%d_%H%M%S).log"


uvicorn ablation_server:app \
  --host 0.0.0.0 --port 6010 \
  --workers 1 \
  --timeout-keep-alive 75 \
  > "${LOG_DIR}/${LOG_NAME}" 2>&1 & disown
