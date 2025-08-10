tmux new -s reward
# 不要占 GPU
export CUDA_VISIBLE_DEVICES=
nohup uvicorn reward_server:app --host 0.0.0.0 --port 6009 --workers 1 --timeout-keep-alive 75 \
  > /users/xwang76/nano_rl/reward_part/reward.log 2>&1 & disown
# Ctrl-b d 退出 tmux 会话，服务仍在
