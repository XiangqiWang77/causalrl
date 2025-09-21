
fuser -k 6010/tcp 2>/dev/null || true
fuser -k 8266/tcp 2>/dev/null || true   # Ray dashboard
fuser -k 6379/tcp 2>/dev/null || true   # 你的 Ray 端口

ray stop --force || true


pkill -f "uvicorn .*reward_server:app" || true
pkill -f "reward_server:app" || true
pkill -f "raylet|gcs_server|ray::|dashboard" || true


ss -lptn | egrep ':6009|:8265|:6378' || echo "OK – all free"
