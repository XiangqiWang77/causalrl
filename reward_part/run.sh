export CUDA_VISIBLE_DEVICES=0
LOG_NAME=my_custom_name_$(date +%Y%m%d_%H%M%S).log
uvicorn updated_server:app --host 0.0.0.0 --port 6009 --workers 1 --timeout-keep-alive 75 \
  > /users/xwang76/nano_rl/reward_part/$LOG_NAME 2>&1 & disown