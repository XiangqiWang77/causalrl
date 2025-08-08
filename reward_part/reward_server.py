# reward_server.py
import os
import json
import datetime
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# —— 轻量模块可在请求里 import；会初始化 CUDA 的库放到 lifespan 里 —— #
# from signal_utils import extract_XY  # 如果很轻量，也可以放到外面
# 这里先延迟到请求函数里 import，避免启动前找不到包出错

logger = logging.getLogger("reward_server")
logging.basicConfig(level=logging.INFO)

# 全局单例（启动时创建）
calc = None
gpu_lock: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    进程启动时只运行一次：在这里把模型加载到 GPU，并创建互斥锁。
    进程退出时可做清理。
    """
    global calc, gpu_lock
    logger.info(f"[lifespan] PID={os.getpid()} starting… loading model to GPU")

    # 这些 import / 构造会触发 CUDA 初始化，务必放在这里而不是请求里
    from modi_signal import RMSNormalizedSignalCalculator  # 你的类
    model_dir = "/users/xwang76/hf_models/qwen3-4b"

    # 只加载一次
    calc = RMSNormalizedSignalCalculator(model_dir, device="cuda")
    gpu_lock = asyncio.Semaphore(1)  # 串行化 GPU 使用，避免并发 OOM
    logger.info("[lifespan] model loaded, service is ready")

    try:
        yield
    finally:
        # 进程退出时清理（可选）
        try:
            import torch
            del calc
            torch.cuda.empty_cache()
            logger.info("[lifespan] cleaned CUDA cache")
        except Exception as e:
            logger.warning(f"[lifespan] cleanup err: {e}")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/get_reward2")
async def get_reward2(request: Request):
    """
    使用全局 calc（单例）计算奖励。
    注意：不要在这里再次加载模型或 import transformers/torch（会重复占显存）。
    """
    json_data = await request.json()

    # 串行化 GPU：一次只允许一个请求进入计算
    async with gpu_lock:
        # 把纯计算放到线程池里，避免阻塞事件循环
        def run_compute():
            # 轻量工具可以这里 import
            from signal_utils import extract_XY

            Z = json_data.get("prompt_str", "")
            resp = json_data.get("response_str", "")
            if isinstance(resp, list):
                resp = resp[0]

            X, Y = extract_XY(resp)

            # 你的 calc.compute 内部最好包 with torch.inference_mode()，
            # 并适时释放大 tensor，结束后可以 empty_cache（见下方注释）
            s_ZX, s_ZY, s_XY = calc.compute(Z, X, Y)
            p=2.0
            total = (abs(s_ZX)**p + abs(s_ZY)**p + abs(s_XY)**p)**(1.0/p)
            return s_ZX, s_ZY, s_XY, total

        s_ZX, s_ZY, s_XY, total = await asyncio.to_thread(run_compute)

        result = {
            "S(Z→X)": round(float(s_ZX), 6),
            "S(Z→Y)": round(float(s_ZY), 6),
            "S(X→Y)": round(float(s_XY), 6),
            "score": round(float(total), 6),
        }

        logger.info(
            json.dumps(
                {
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pid": os.getpid(),
                    "reward": result,
                },
                ensure_ascii=False,
            )
        )

        # 如果仍担心缓存积累，可适度清理（一般不必每次都清）
        # import torch; torch.cuda.empty_cache()

        return result


# 也可以保留你之前的 /get_reward 简单规则分 + 日志（略）
