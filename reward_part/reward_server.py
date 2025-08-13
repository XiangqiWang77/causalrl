# reward_server.py
import os
import json
import datetime
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from Judge_reward import evaluate_responses1

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
    model_dir = "~/hf_models/qwen3-4b"

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


import math

def minkowski_reward(s1, s2, s3, w1, w2, w3, p, eps=1e-12):
    """
    Robust weighted Minkowski aggregation for 3 signals.
    Inputs:
      s1,s2,s3: signals (expected in [0,1]); will be clipped to [0,1].
      w1,w2,w3: non-negative weights (can be any non-negative scale).
      p: Minkowski exponent. Supports p→0 (geometric mean), large |p|, and standard p.
      eps: small constant for numerical stability.
    Output:
      total in [0,1] if inputs in [0,1] and weights non-negative.
    """

    # 1) clip signals into [0,1] to avoid out-of-range explosions
    s = [min(1.0, max(0.0, float(x))) for x in (s1, s2, s3)]

    # 2) sanitize weights and normalize
    w = [max(0.0, float(x)) for x in (w1, w2, w3)]
    wsum = w[0] + w[1] + w[2]
    if wsum <= 0:
        # no weight info; default to uniform
        w = [1.0, 1.0, 1.0]
        wsum = 3.0
    wn = [wi / wsum for wi in w]

    # 3) handle special/edge cases of p
    # p -> +inf => max; p -> -inf => min
    if p is None or (isinstance(p, float) and math.isnan(p)):
        raise ValueError("p must be a real number.")
    if p >= 1e6:
        return max(s)
    if p <= -1e6:
        return min(s)

    # p ≈ 0 => weighted geometric mean (limit of Minkowski mean)
    if abs(p) < 1e-6:
        # use log-sum to avoid underflow; clamp by eps
        log_terms = [wn[i] * math.log(max(s[i], eps)) for i in range(3)]
        return math.exp(sum(log_terms))

    # If p < 0 and any s_i == 0, the term s_i^p is undefined/infinite.
    if p < 0 and any(si <= 0.0 for si in s):
        # define a safe fallback: treat zeros as eps to avoid inf
        s = [max(si, eps) for si in s]

    # 4) normalized Minkowski mean (a.k.a. power mean) with weights:
    # M_p = ( sum_i w_i * s_i^p / sum_i w_i )^(1/p)  => with wn it's just (sum wn * s_i^p)^(1/p)
    acc = 0.0
    for si, wi in zip(s, wn):
        # fast path for si in {0,1}
        if si == 0.0:
            term = 0.0
        elif si == 1.0:
            term = wi  # 1^p == 1
        else:
            # compute si^p stably via exp(p*log(si))
            term = wi * math.exp(p * math.log(si))
        acc += term

    # guard tiny negatives from round-off
    acc = max(acc, 0.0)

    # final power 1/p; use exp((1/p)*log(acc)) for stability
    # when acc==0, result is 0 for p>0; for p<0, it's undefined→return 0 as safe floor
    if acc == 0.0:
        return 0.0 if p > 0 else 0.0

    total = math.exp((1.0 / p) * math.log(acc))

    # final clamp to [0,1] due to tiny numeric drift
    return min(1.0, max(0.0, total))


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
            #from bandit import LinUCB

            Z = json_data.get("prompt_str", "")
            resp = json_data.get("response_str", "")
            ground_truth = json_data.get("ground_truth", "")
            judgescore = evaluate_responses1(Z, resp, ground_truth)
            X, Y = extract_XY(resp)

            # 你的 calc.compute 内部最好包 with torch.inference_mode()，
            # 并适时释放大 tensor，结束后可以 empty_cache（见下方注释）
            s_ZX, s_ZY, s_XY = calc.compute(Z, X, Y)
            p,w1,w2,w3=0.3, 1/3, 1/3, 1/3
            total = minkowski_reward(judgescore, s_ZX, s_XY, w1, w2, w3, p)
            return judgescore, s_ZX, s_XY, total

        judgescore, s_ZX, s_XY, total = await asyncio.to_thread(run_compute)

        result = {
            "Judgescore": round(float(judgescore), 6),
            "S(Z→X)": round(float(s_ZX), 6),
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
