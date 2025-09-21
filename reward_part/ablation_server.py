# reward_server.py
import os
import json
import datetime
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("updated_server")
logging.basicConfig(level=logging.INFO)


CUDA_DEVICES_STR = os.environ.get("CUDA_DEVICES", "cuda:0,cuda:1,cuda:2,cuda:3").strip()
MAX_BATCH_SIZE   = int(os.environ.get("MAX_BATCH_SIZE", "4"))
MAX_WAIT_MS      = int(os.environ.get("MAX_WAIT_MS", "8"))


MAX_TOKENS_PER_BATCH   = int(os.environ.get("MAX_TOKENS_PER_BATCH", "1024"))
CUDA_FREE_MEM_FRACTION = float(os.environ.get("CUDA_FREE_MEM_FRACTION", "0.85"))
OOM_BACKOFF_STEPS      = int(os.environ.get("OOM_BACKOFF_STEPS", "4"))
CLEAR_CACHE_EVERY      = int(os.environ.get("CLEAR_CACHE_EVERY", "5"))
RESERVED_WATERMARK     = float(os.environ.get("RESERVED_WATERMARK", "0.92"))
SAFE_MIN_FREE_MB       = int(os.environ.get("SAFE_MIN_FREE_MB", "1000"))  # 1GB
MAX_MEMORY_FRACTION    = float(os.environ.get("MAX_MEMORY_FRACTION", "0.90"))


BERTSCORE_DEVICE = os.environ.get("BERTSCORE_DEVICE", "cpu")


calc = None
_bs_scorer = None
_task_queue: "asyncio.Queue[Tuple[Dict[str, Any], asyncio.Future]]" = None
_batcher_task: asyncio.Task = None
_global_batch_idx: int = 0

def _normalize_devices(devs: str) -> List[str]:
    out = []
    for d in devs.split(","):
        d = d.strip()
        if not d:
            continue
        if d.isdigit():
            out.append(f"cuda:{d}")
        else:
            out.append(d)

    return out

CUDA_DEVICES = _normalize_devices(CUDA_DEVICES_STR)


def _cuda_mem_info(device_str: str):
    import torch
    if not torch.cuda.is_available() or not device_str.startswith("cuda"):
        return (0, 0, 0, 0)
    dev = torch.device(device_str)
    if dev.index is not None:
        torch.cuda.set_device(dev)
    free_b, total_b = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    return free_b, total_b, reserved, allocated

def _cuda_mem_info_all() -> Dict[str, Tuple[int,int,int,int]]:
    info = {}
    for d in CUDA_DEVICES:
        info[d] = _cuda_mem_info(d)
    return info

def _has_safe_margin_all() -> bool:
    infos = _cuda_mem_info_all()
    for d, (free_b, _, _, _) in infos.items():
        safe_free_b = int(free_b * CUDA_FREE_MEM_FRACTION)
        if safe_free_b < SAFE_MIN_FREE_MB * 1024 * 1024:
            return False
    return True

def _maybe_empty_cache_all():
    import torch
    for d in CUDA_DEVICES:
        if not d.startswith("cuda"):
            continue
        dev = torch.device(d)
        if dev.index is not None:
            torch.cuda.set_device(dev)
        _, total_b2, reserved_b2, _ = _cuda_mem_info(d)
        if RESERVED_WATERMARK < 1.0 and total_b2 > 0:
            if reserved_b2 / total_b2 > RESERVED_WATERMARK:
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()

"""
from typing import List, Tuple, Optional
import asyncio
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_MODEL_NAME = "roberta-large-mnli"   # 或 "facebook/bart-large-mnli"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
_MODEL = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME).to(_DEVICE).eval()

# ---------- utilities ----------
def _resolve_label_ids(model) -> Tuple[int, int, int]:

    l2i = {}
    if getattr(model.config, "label2id", None):
        l2i = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    else:
        if getattr(model.config, "id2label", None):
            l2i = {v.lower(): int(k) for k, v in model.config.id2label.items()}

    def pick(*names):
        for n in names:
            if n in l2i:
                return l2i[n]
        return None

    ent = pick("entailment", "entails")
    con = pick("contradiction", "contradicts")
    neu = pick("neutral")

    if any(v is None for v in (ent, con, neu)):
        ent, con, neu = 2, 0, 1
    return ent, con, neu

_ENT_ID, _CON_ID, _NEU_ID = _resolve_label_ids(_MODEL)

def _ensure_long_and_device(batch: dict, device: torch.device) -> dict:

    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # input_ids / attention_mask / token_type_ids 都转 long
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                v = v.to(device=device, dtype=torch.long, non_blocking=True)
            else:
                v = v.to(device=device, non_blocking=True)
        out[k] = v
    return out

def _chunked(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def _tokenize_pairs(premises: List[str], hyps: List[str], max_length: int = 512):
    return _TOKENIZER(
        premises,
        hyps,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def _nli_entail_probs_sync(
    premises: List[str],
    hyps: List[str],
    max_length: int = 512,
    chunk_size: int = 64,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    dev = device or (_DEVICE if torch.cuda.is_available() else torch.device("cpu"))
    probs_list = []
    with torch.no_grad():
        for p_chunk, h_chunk in zip(_chunked(premises, chunk_size), _chunked(hyps, chunk_size)):
            batch = _tokenize_pairs(p_chunk, h_chunk, max_length=max_length)
            batch = _ensure_long_and_device(batch, dev)

            try:
                logits = _MODEL(**batch).logits  # [B,3]
            except RuntimeError as e:
                msg = str(e).lower()
                needs_cpu_fallback = (
                    "embedding" in msg and "indices" in msg
                ) or ("cuda" in msg and ("fault" in msg or "illegal" in msg or "oom" in msg))
                if needs_cpu_fallback and dev.type != "cpu":
                    dev_cpu = torch.device("cpu")
                    _MODEL.to(dev_cpu)
                    batch_cpu = _ensure_long_and_device(batch, dev_cpu)
                    logits = _MODEL(**batch_cpu).logits
                    _MODEL.to(_DEVICE)
                else:
                    raise

            probs = torch.softmax(logits, dim=-1)  # [B,3]
            probs_list.append(probs[:, _ENT_ID].detach().cpu())

    return torch.cat(probs_list, dim=0)  # [len(premises)]

async def _compute_bertscore_batch(
    Ys: List[str],
    refs: List[str],
    threshold: float = 0.5,
    max_length: int = 512,
    chunk_size: int = 64,
) -> List[float]:
    
    if not Ys or not refs:
        raise ValueError("Ys and refs cannot be empty.")
    Ys_clean = [y if (isinstance(y, str) and y.strip()) else "" for y in Ys]
    refs_clean = [r if (isinstance(r, str) and r.strip()) else "" for r in refs]

    premises, hyps = [], []
    for y in Ys_clean:
        premises.extend([y] * len(refs_clean))
        hyps.extend(refs_clean)

    n_y, n_r = len(Ys_clean), len(refs_clean)
    total_pairs = n_y * n_r

    loop = asyncio.get_running_loop()
    entail_flat: torch.Tensor = await loop.run_in_executor(
        None,
        _nli_entail_probs_sync,
        premises,
        hyps,
        max_length,
        chunk_size,
        _DEVICE,
    )

    if not isinstance(entail_flat, torch.Tensor):
        entail_flat = torch.tensor(entail_flat, dtype=torch.float32)
    entail_flat = entail_flat.flatten().contiguous().cpu()

    if entail_flat.numel() != total_pairs:
        if entail_flat.numel() > total_pairs:
            entail_flat = entail_flat[:total_pairs]
        else:
            pad = torch.zeros(total_pairs - entail_flat.numel(), dtype=entail_flat.dtype)
            entail_flat = torch.cat([entail_flat, pad], dim=0)

    entail_mat = entail_flat.view(n_y, n_r)          # [N_y, N_r]
    max_entail = entail_mat.max(dim=1).values        # [N_y], 0~1


    return (max_entail >= threshold).to(dtype=torch.float32).tolist()
"""





# ------------------- BERTScore -------------------
async def _compute_bertscore_batch(Ys: List[str], refs: List[str]) -> List[float]:
    #Minimal BERTScore（precision 加权 F1）
    assert _bs_scorer is not None, "BERTScorer not initialized"
    PREC_W = 0.8
    BS_BATCH_SIZE = 8
    P, R, F1 = _bs_scorer.score(Ys, refs, batch_size=BS_BATCH_SIZE)
    return [PREC_W * P[i].item() + (1 - PREC_W) * F1[i].item() for i in range(len(Ys))]


# ------------------- Minkowski-------------------
import math
def minkowski_reward(bertscore, s_ZX, s_XY, s_ZY, w1, w2, w3, w4, p, eps=1e-12):
    s_raw = [bertscore, s_ZX, s_XY, s_ZY]
    s = []
    for x in s_raw:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if not (v == v):
            v = 0.0
        s.append(min(1.0, max(0.0, v)))
    w_raw = [w1, w2, w3, w4]
    w = []
    for x in w_raw:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if not (v == v) or v < 0.0:
            v = 0.0
        w.append(v)
    wsum = sum(w)
    wn = [wi / wsum for wi in w] if wsum > 0 else [0.25]*4
    if p is None or (isinstance(p, float) and math.isnan(p)):
        raise ValueError("p must be a real number.")
    if p >= 1e6:   return max(s)
    if p <= -1e6:  return min(s)
    if abs(p) < 1e-6:
        acc_log = 0.0
        for si, wi in zip(s, wn):
            if wi <= 0.0: continue
            acc_log += wi * math.log(max(si, eps))
        return min(1.0, max(0.0, math.exp(acc_log)))
    acc = 0.0
    for si, wi in zip(s, wn):
        if wi <= 0.0: continue
        base = max(si, eps) if p < 0 else si
        acc += wi * math.exp(p * math.log(max(base, eps)))
    if acc <= 0: return 0.0
    return min(1.0, max(0.0, math.exp((1.0/p) * math.log(acc))))

# ------------------- token -------------------
def _estimate_tokens_single(tok, Z: str, X: str, Y: str,
                            max_z: int | None, max_x: int | None, max_y: int | None) -> int:
    z = tok(Z, add_special_tokens=False, return_tensors="pt").input_ids
    x = tok(X, add_special_tokens=False, return_tensors="pt").input_ids
    y = tok(Y, add_special_tokens=False, return_tensors="pt").input_ids
    if max_z: z = z[:, -max_z:]
    if max_x: x = x[:, -max_x:]
    if max_y: y = y[:, -max_y:]
    return int(z.size(1) + 1 + x.size(1) + 1 + y.size(1))


def _build_max_memory_map():
    import torch
    max_mem = {}
    for d in CUDA_DEVICES:
        if not d.startswith("cuda"):  
            continue
        idx = torch.device(d).index
        if idx is None:
            continue
        props = torch.cuda.get_device_properties(idx)
        gb = int(props.total_memory / (1024**3) * MAX_MEMORY_FRACTION)
        max_mem[idx] = f"{gb}GiB"
    return max_mem

def _load_rms_sharded(model_path: str):
    import torch
    from distri_signal import RMSNormalizedSignalCalculator
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    
    device_map = "balanced" if len(CUDA_DEVICES) > 1 and torch.cuda.is_available() else None
    max_memory = _build_max_memory_map() if device_map else None

   
    try:
        calc = RMSNormalizedSignalCalculator(
            model_path,
            device="cuda",
            hf_load_kwargs=dict(  # 可省略，用 ENV 自动生成
                device_map="balanced",
                max_memory={0: "36GiB", 1: "36GiB", 2: "36GiB", 3: "36GiB"},
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ),
        )
        logger.info("[lifespan] RMS loaded with internal hf_load_kwargs (sharded).")
        return calc
    except TypeError:
        logger.info("[lifespan] RMS does not accept hf_load_kwargs; trying fallback dispatch…")

    
    calc = RMSNormalizedSignalCalculator(
        model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",  # 临时放 0 卡，稍后分发
        dtype=str(dtype).replace("torch.", "")
    )
    try:
        from accelerate import load_checkpoint_and_dispatch
        base_model = getattr(calc, "model", None) or getattr(calc, "_model", None)
        base_name_or_path = getattr(base_model, "name_or_path", model_path)

        if base_model is None:
            raise RuntimeError("Cannot access underlying HF model from RMS calculator.")

        if device_map is None:
            logger.info("[lifespan] Single device detected; keep as-is.")
            return calc

        sharded_model = load_checkpoint_and_dispatch(
            base_model,
            checkpoint=base_name_or_path,
            device_map="balanced",
            max_memory=max_memory,
            dtype=dtype,
            no_split_module_classes=getattr(base_model, "_no_split_modules", None)
        )

        if hasattr(calc, "model"):
            calc.model = sharded_model
        elif hasattr(calc, "_model"):
            calc._model = sharded_model
        logger.info("[lifespan] Fallback dispatch applied (sharded).")
    except Exception as e:
        logger.warning(f"[lifespan] Fallback shard failed; using single GPU model. err={e}")
    return calc


@asynccontextmanager
async def lifespan(app: FastAPI):
    global calc, _bs_scorer, _task_queue, _batcher_task

    import torch
    import os as _os

    
    logger.info(f"[lifespan] Visible devices: {CUDA_DEVICES}")

   
    logger.info("[lifespan] loading RMSNormalizedSignalCalculator (sharded)…")
    calc = _load_rms_sharded("/path/to/llama-3-8B")

    # BERTScorer
    try:
        from bert_score import BERTScorer
        _bs_scorer = BERTScorer(
            model_type=_os.getenv("BERTSCORE_MODEL", "microsoft/deberta-base-mnli"),
            idf=_os.getenv("BERTSCORE_IDF", "0") == "1",
            rescale_with_baseline=_os.getenv("BERTSCORE_RESCALE", "0") == "1",
            device=BERTSCORE_DEVICE,
        )
        logger.info(f"[lifespan] BERTScorer initialized on {BERTSCORE_DEVICE}")
    except Exception as e:
        _bs_scorer = None
        logger.warning(f"[lifespan] BERTScorer init failed: {e}")

    _task_queue = asyncio.Queue()
    _batcher_task = asyncio.create_task(_batcher_loop())
    logger.info("[lifespan] queue/batcher started")

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
            if calc is not None:
                del calc
            if torch.cuda.is_available():
                _maybe_empty_cache_all()
            logger.info("[lifespan] cleaned CUDA cache")
        except Exception as e:
            logger.warning(f"[lifespan] cleanup err: {e}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


async def _safe_compute_batch(Zs: List[str], Xs: List[str], Ys: List[str]) -> Tuple[List[float], List[float], List[float]]:
    
    import torch
    global _global_batch_idx

    
    tok = getattr(calc, "tokenizer", None)
    max_z = getattr(calc, "max_z", None)
    max_x = getattr(calc, "max_x", None)
    max_y = getattr(calc, "max_y", None)

    if tok is None:
        est_tokens = [len(Zs[i]) + len(Xs[i]) + len(Ys[i]) for i in range(len(Zs))]
        token_budget = MAX_TOKENS_PER_BATCH * 3
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

    for local_chunk_id, ch in enumerate(chunks):
        sub = ch[:]
        tries = 0
        while sub:
            try:
                
                if not _has_safe_margin_all() and len(sub) > 1:
                    sub = sub[: max(1, len(sub)//2)]

               
                Zb = [Zs[i] for i in sub]
                Xb = [Xs[i] for i in sub]
                Yb = [Ys[i] for i in sub]

                sZX, sZY, sXY = calc.compute_batch_cf_enhanced(Zb, Xb, Yb)

                for k, i in enumerate(sub):
                    out_ZX[i] = float(sZX[k]); out_ZY[i] = float(sZY[k]); out_XY[i] = float(sXY[k])

                del Zb, Xb, Yb, sZX, sZY, sXY

                
                if RESERVE_WATERMARK := RESERVED_WATERMARK:
                    _maybe_empty_cache_all()
                if CLEAR_CACHE_EVERY > 0:
                    if ((_global_batch_idx + local_chunk_id) % CLEAR_CACHE_EVERY) == 0:
                        _maybe_empty_cache_all()
                break

            except RuntimeError as re:
                msg = str(re)
                is_oom = ("CUDA out of memory" in msg) or ("CUDA error: out of memory" in msg)
                if not is_oom:
                    raise
                tries += 1
                try:
                    _maybe_empty_cache_all()
                except Exception:
                    pass
                logger.warning(f"[compute_batch] OOM on sub-batch size={len(sub)}; try={tries}")
                if len(sub) == 1 or tries > OOM_BACKOFF_STEPS:
                    raise
                sub = sub[: max(1, len(sub)//2)]
            except Exception:
                raise

    _global_batch_idx += 1
    return out_ZX, out_ZY, out_XY


async def _batcher_loop():
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

            Zs, Xs, Ys, gts = [], [], [], []
            resps=[]
            futs = []
            for payload, fut in batch:
                Z = payload.get("prompt_str", "")
                resp = payload.get("response_str", "")
                gt = payload.get("ground_truth", "")
                X, Y = extract_XY(resp)
                Zs.append(Z); Xs.append(X); Ys.append(Y); gts.append(gt); resps.append(resp)
                futs.append(fut)

            bertscores = await _compute_bertscore_batch(Ys, gts)

            s_ZX_list, s_ZY_list, s_XY_list = await _safe_compute_batch(Zs, Xs, Ys)

            #p, w1, w2, w3, w4 = 0.7, 8/10, 3/30, 0/30, 3/30
            #p, w1, w2, w3, w4 = 1, 10/10, 0/30, 0/30, 0/30
            #p, w1, w2, w3, w4 = 0.2, 7/10, 3/30, 3/30, 3/30
            p, w1, w2, w3, w4 = 0.2, 0/10, 10/30, 10/30, 10/30
            #p, w1, w2, w3, w4 = 1, 7/10, 3/30, 3/30, 3/30
            #p, w1, w2, w3, w4 = 10, 7/10, 3/30, 3/30, 3/30
            #p, w1, w2, w3, w4 = -2, 7/10, 3/30, 3/30, 3/30
            #p, w1, w2, w3, w4 = 0.2, 1/2, 1/6, 1/6, 1/6
            for i, fut in enumerate(futs):
                bertscore = float(bertscores[i])
                s_ZX = float(s_ZX_list[i]); s_XY = float(s_XY_list[i]); s_ZY = float(s_ZY_list[i])
                total = minkowski_reward(bertscore, s_ZX, s_XY, s_ZY, w1, w2, w3, w4, p)
                #total = total + np.random.normal(0, 0.05)
                fut.set_result({
                    "basecore": round(bertscore, 6),
                    "S(Z→X)": round(s_ZX, 6),
                    "S(X→Y)": round(s_XY, 6),
                    "S(Z→Y)": round(s_ZY, 6),
                    "score": round(float(total), 6),
                })

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

# ------------------- API -------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/get_reward2")
async def get_reward2(request: Request):
    json_data = await request.json()
    fut: "asyncio.Future[Dict[str, Any]]" = asyncio.get_running_loop().create_future()
    await _task_queue.put((json_data, fut))
    result = await fut
    logger.info(json.dumps(
        {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "pid": os.getpid(), "reward": result}, ensure_ascii=False))
    return result
