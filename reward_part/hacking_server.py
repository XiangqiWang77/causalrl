# reward_server.py
import os
import json
import datetime
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


logger = logging.getLogger("hacking_server")
logging.basicConfig(level=logging.INFO)

# ------------------- 配置 -------------------
CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "4"))
MAX_WAIT_MS   = int(os.environ.get("MAX_WAIT_MS", "8"))

# --- 显存安全相关（可用环境变量覆盖） ---
MAX_TOKENS_PER_BATCH   = int(os.environ.get("MAX_TOKENS_PER_BATCH", "1024"))
CUDA_FREE_MEM_FRACTION = float(os.environ.get("CUDA_FREE_MEM_FRACTION", "0.85"))
OOM_BACKOFF_STEPS      = int(os.environ.get("OOM_BACKOFF_STEPS", "4"))
CLEAR_CACHE_EVERY      = int(os.environ.get("CLEAR_CACHE_EVERY", "5"))
RESERVED_WATERMARK     = float(os.environ.get("RESERVED_WATERMARK", "0.92"))
SAFE_MIN_FREE_MB       = int(os.environ.get("SAFE_MIN_FREE_MB", "1000"))  # 1.2GB 安全余量

# ------------------- 全局状态 -------------------
calc = None
judge = None
_task_queue: "asyncio.Queue[Tuple[Dict[str, Any], asyncio.Future]]" = None
_batcher_task: asyncio.Task = None
_global_batch_idx: int = 0  # 用于周期性清理



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动时只做一次：加载 Qwen(calc)、BERTScorer、创建队列与批处理后台协程。
    """
    global calc, judge, _task_queue, _batcher_task, _bs_scorer

    # 必须先 import，再用 os.getpid()
    import os, torch
    logger.info(f"[lifespan] PID={os.getpid()} starting… loading model to {CUDA_DEVICE}")

    # 固定当前 CUDA 设备
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device(CUDA_DEVICE))
    except Exception as _e:
        logger.warning(f"[lifespan] set_device warn: {_e}")

    # 只加载一份 Qwen：由 calc 持有
    from hack_signal import RMSNormalizedSignalCalculator
    calc = RMSNormalizedSignalCalculator("/users/xwang76/hf_models/qwen3-4b", device=CUDA_DEVICE)

    # 启动期一次性加载并缓存 BERTScorer（后续复用，不再重复加载）
    try:
        from bert_score import BERTScorer
        _bs_scorer = BERTScorer(
            model_type=os.getenv("BERTSCORE_MODEL", "microsoft/deberta-base-mnli"),
            idf=os.getenv("BERTSCORE_IDF", "0") == "1",
            rescale_with_baseline=os.getenv("BERTSCORE_RESCALE", "0") == "1",
            device=CUDA_DEVICE if torch.cuda.is_available() else "cpu",
        )
        logger.info("[lifespan] BERTScorer initialized")
    except Exception as e:
        _bs_scorer = None
        logger.warning(f"[lifespan] BERTScorer init failed: {e}")

    # 队列与批处理后台协程
    _task_queue = asyncio.Queue()
    _batcher_task = asyncio.create_task(_batcher_loop())
    logger.info("[lifespan] model loaded, queue started")

    try:
        yield
    finally:
        if _batcher_task:
            _batcher_task.cancel()
            try:
                await _batcher_task
            except asyncio.CancelledError:
                pass
        try:
            if _bs_scorer is not None:
                del _bs_scorer
                _bs_scorer = None
            if calc is not None:
                del calc
                calc = None
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

async def _compute_bertscore_batch(Ys: List[str], refs: List[str]) -> List[float]:
    """
    Minimal BERTScore (base only, precision-weighted).
    """
    assert _bs_scorer is not None, "BERTScorer not initialized"

    PREC_W = 0.8   # precision vs F1 权重
    BS_BATCH_SIZE = 8

    # 调用 scorer
    P, R, F1 = _bs_scorer.score(Ys, refs, batch_size=BS_BATCH_SIZE)

    # base score = precision 加权 F1
    scores = [PREC_W * P[i].item() + (1 - PREC_W) * F1[i].item() for i in range(len(Ys))]
    return scores



import math

def minkowski_reward(bertscore, s_ZX, s_XY, s_ZY, w1, w2, w3, w4, p, eps=1e-12):
    """
    Weighted Minkowski aggregation over four signals in [0,1].
      - p > 0  : soft-OR / soft-max (p→+∞ → max)
      - p = 0  : weighted geometric mean
      - p < 0  : soft-AND / soft-min (p→-∞ → min)  ← usually safer against hacking

    Args:
        bertscore, s_ZX, s_XY, s_ZY : floats in [0,1]
        w1..w4 : nonnegative weights (arbitrary scale; normalized inside)
        p      : real number (aggregation order)
        eps    : numerical floor to avoid log/zero issues

    Returns:
        float in [0,1]
    """
    # sanitize signals
    s_raw = [bertscore, s_ZX, s_XY, s_ZY]
    s = []
    for x in s_raw:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if not (v == v):  # NaN
            v = 0.0
        s.append(min(1.0, max(0.0, v)))

    # sanitize weights
    w_raw = [w1, w2, w3, w4]
    w = []
    for x in w_raw:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if not (v == v) or v < 0.0:  # NaN or negative
            v = 0.0
        w.append(v)

    wsum = sum(w)
    if wsum <= 0.0:
        wn = [0.25, 0.25, 0.25, 0.25]
    else:
        wn = [wi / wsum for wi in w]

    # p checks
    if p is None or (isinstance(p, float) and math.isnan(p)):
        raise ValueError("p must be a real number.")
    if p >= 1e6:   # → max
        return max(s)
    if p <= -1e6:  # → min
        return min(s)

    # p ≈ 0 → weighted geometric mean
    if abs(p) < 1e-6:
        # ignore zero-weight terms; use eps floor for log
        acc_log = 0.0
        for si, wi in zip(s, wn):
            if wi <= 0.0:
                continue
            acc_log += wi * math.log(max(si, eps))
        val = math.exp(acc_log)
        return min(1.0, max(0.0, val))

    # general p (weighted power mean)
    acc = 0.0
    for si, wi in zip(s, wn):
        if wi <= 0.0:
            continue
        # protect negatives/zeros for p<0
        base = max(si, eps) if p < 0 else si
        # compute wi * si^p robustly
        term = wi * math.exp(p * math.log(max(base, eps)))
        acc += term

    acc = max(acc, 0.0)
    if acc == 0.0:
        return 0.0  # safe fallback

    total = math.exp((1.0 / p) * math.log(acc))
    return min(1.0, max(0.0, total))


# ------------------- 显存安全辅助 -------------------
def _cuda_mem_info(device_str: str):
    import torch
    if not torch.cuda.is_available():
        return (0, 0, 0, 0)
    dev = torch.device(device_str)
    if dev.index is not None:
        torch.cuda.set_device(dev)
    free_b, total_b = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    return free_b, total_b, reserved, allocated

def _estimate_tokens_single(tok, Z: str, X: str, Y: str,
                            max_z: int | None, max_x: int | None, max_y: int | None) -> int:
    z = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids
    x = tok(X, add_special_tokens=False, return_tensors="pt").input_ids
    y = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids
    if max_z: z = z[:, -max_z:]
    if max_x: x = x[:, -max_x:]
    if max_y: y = y[:, -max_y:]
    # Z <eos> X <eos> Y
    return int(z.size(1) + 1 + x.size(1) + 1 + y.size(1))

async def _safe_compute_batch(Zs: List[str], Xs: List[str], Ys: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """
    不改 RMS：在 server 侧做切片 + 显存水位检查 + OOM 回退 + 周期清理。
    """
    import torch
    global _global_batch_idx

    # 1) 预分块：不超过 MAX_TOKENS_PER_BATCH（启发式柔约束）
    tok = getattr(calc, "tokenizer", None)
    max_z = getattr(calc, "max_z", None)
    max_x = getattr(calc, "max_x", None)
    max_y = getattr(calc, "max_y", None)

    if tok is None:
        # 若拿不到 tokenizer，就退化为样本数切片
        est_tokens = [len(Zs[i]) + len(Xs[i]) + len(Ys[i]) for i in range(len(Zs))]
        token_budget = MAX_TOKENS_PER_BATCH * 3  # 保守
    else:
        est_tokens = [_estimate_tokens_single(tok, Zs[i], Xs[i], Ys[i], max_z, max_x, max_y)
                      for i in range(len(Zs))]
        token_budget = MAX_TOKENS_PER_BATCH

    idxs = list(range(len(Zs)))
    chunks: List[List[int]] = []
    cur, cur_tokens = [], 0
    for i in idxs:
        t = est_tokens[i]
        if (len(cur) >= MAX_BATCH_SIZE) or (cur and cur_tokens + t > token_budget):
            chunks.append(cur)
            cur, cur_tokens = [i], t
        else:
            cur.append(i); cur_tokens += t
    if cur: chunks.append(cur)

    out_ZX = [None] * len(Zs)
    out_ZY = [None] * len(Zs)
    out_XY = [None] * len(Zs)

    # 2) 针对每个 chunk：显存水位检查 + OOM 回退（二分）
    for local_chunk_id, ch in enumerate(chunks):
        sub = ch[:]
        tries = 0
        while sub:
            try:
                free_b, total_b, reserved_b, alloc_b = _cuda_mem_info(CUDA_DEVICE)
                # 安全余量判断
                safe_free_b = int(free_b * CUDA_FREE_MEM_FRACTION)
                if (safe_free_b < SAFE_MIN_FREE_MB * 1024 * 1024) and len(sub) > 1:
                    # 主动先减半，降低 OOM 概率
                    sub = sub[: max(1, len(sub)//2)]

                # 真正执行
                Zb = [Zs[i] for i in sub]
                Xb = [Xs[i] for i in sub]
                Yb = [Ys[i] for i in sub]

                sZX, sZY, sXY = calc.compute_batch_naive(Zb, Xb, Yb)

                # 回填
                for k, i in enumerate(sub):
                    out_ZX[i] = float(sZX[k])
                    out_ZY[i] = float(sZY[k])
                    out_XY[i] = float(sXY[k])

                # 释放中间强引用，尽快让显存/缓存回收
                del Zb, Xb, Yb, sZX, sZY, sXY
                torch.cuda.synchronize()

                # 按需清理：水位高/周期性
                if torch.cuda.is_available():
                    _, total_b2, reserved_b2, _ = _cuda_mem_info(CUDA_DEVICE)
                    if RESERVED_WATERMARK < 1.0 and total_b2 > 0:
                        if reserved_b2 / total_b2 > RESERVED_WATERMARK:
                            torch.cuda.empty_cache()
                    if CLEAR_CACHE_EVERY > 0:
                        # 用全局批计数 + 本地 chunk id 组合决定是否清空
                        if ((_global_batch_idx + local_chunk_id) % CLEAR_CACHE_EVERY) == 0:
                            torch.cuda.empty_cache()

                break  # 成功，退出 while

            except RuntimeError as re:
                # 捕获可能的 CUDA OOM
                msg = str(re)
                is_oom = ("CUDA out of memory" in msg) or ("CUDA error: out of memory" in msg)
                if not is_oom:
                    raise
                tries += 1
                import torch
                torch.cuda.empty_cache()
                logger.warning(f"[compute_batch] OOM on sub-batch size={len(sub)}; try={tries}")
                if len(sub) == 1 or tries > OOM_BACKOFF_STEPS:
                    # 单条仍 OOM，直接抛
                    raise
                sub = sub[: max(1, len(sub)//2)]
            except Exception:
                raise

    _global_batch_idx += 1
    return out_ZX, out_ZY, out_XY

# ------------------- 批处理核心 -------------------
async def _batcher_loop():
    """
    单消费者：不断从队列取任务，按 MAX_WAIT_MS 或 MAX_BATCH_SIZE 触发一次 GPU 批前向。
    只在一张 GPU 上、只加载一次模型。包含：显存安全/自清理/OOM 回退。
    """
    from signal_utils import extract_XY
    import time

    while True:
        try:
            item = await _task_queue.get()
            batch: List[Tuple[Dict[str, Any], asyncio.Future]] = [item]

            start_t = time.perf_counter()
            while (len(batch) < MAX_BATCH_SIZE):
                timeout = MAX_WAIT_MS / 1000.0 - (time.perf_counter() - start_t)
                if timeout <= 0:
                    break
                try:
                    more = await asyncio.wait_for(_task_queue.get(), timeout=timeout)
                    batch.append(more)
                except asyncio.TimeoutError:
                    break

            # 组装批次输入
            Zs, Xs, Ys, gts, resps= [], [], [], [],[]
            futs = []
            for payload, fut in batch:
                Z = payload.get("prompt_str", "")
                resp = payload.get("response_str", "")
                gt = payload.get("ground_truth", "")
                X, Y = extract_XY(resp)
                Zs.append(Z); Xs.append(X); Ys.append(Y); gts.append(gt)
                #resps.append(resp[:512])
                futs.append(fut)

            # CPU 侧跑 judge（若它很重，建议另做 batched 版本）
            bertscores = await _compute_bertscore_batch(Ys, gts)


            # === 显存安全的批前向 ===
            s_ZX_list, s_ZY_list, s_XY_list = await _safe_compute_batch(Zs, Xs, Ys)
            #factual_list = await calc.factual_correctness_batch(Ys, gts)

            # 聚合 + 回填
            p, w1, w2, w3, w4 = 1, 6/10, 4/30, 4/30, 4/30
            for i, fut in enumerate(futs):
                bertscore = float(bertscores[i])
                s_ZX = float(s_ZX_list[i])
                s_XY = float(s_XY_list[i])
                s_ZY = float(s_ZY_list[i])
                total = minkowski_reward(bertscore, s_ZX, s_XY, s_ZY, w1, w2, w3, w4, p)

                result = {
                    "bertscore": round(bertscore, 6),
                    "S(Z→X)": round(s_ZX, 6),
                    "S(X→Y)": round(s_XY, 6),
                    "S(Z→Y)": round(s_ZY, 6),
                    "score": round(float(total), 6),
                }
                fut.set_result(result)

            for _ in batch:
                _task_queue.task_done()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"[batcher] error: {e}")
            try:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
            except Exception:
                pass

@app.post("/get_reward2")
async def get_reward2(request: Request):
    """
    高并发入口：把请求丢进队列，等待 batch 结果。
    GPU 只有一份模型、单消费者一次性合批前向（显存安全）。
    """
    json_data = await request.json()
    fut: "asyncio.Future[Dict[str, Any]]" = asyncio.get_running_loop().create_future()
    await _task_queue.put((json_data, fut))
    result = await fut

    logger.info(json.dumps(
        {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pid": os.getpid(),
            "reward": result,
        },
        ensure_ascii=False,
    ))
    return result
