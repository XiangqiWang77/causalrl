# 1) 干掉占用端口的进程（6009/8265/6378）
fuser -k 6010/tcp 2>/dev/null || true
fuser -k 8266/tcp 2>/dev/null || true   # Ray dashboard
fuser -k 6379/tcp 2>/dev/null || true   # 你的 Ray 端口

# 2) 停 Ray（如果开过）
ray stop --force || true

# 3) 兜底：按命令行特征杀掉后台 uvicorn / Ray 进程
pkill -f "uvicorn .*reward_server:app" || true
pkill -f "reward_server:app" || true
pkill -f "raylet|gcs_server|ray::|dashboard" || true

# 4) 验证端口是否已空
ss -lptn | egrep ':6009|:8265|:6378' || echo "OK – all free"
