from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, FastAPI, Body, Response, Query
from .settings import settings
from .middleware import detect_lang, mask_pii, normalize_text
import time
import httpx
from .retrieval import (
    bm25_rebuild, bm25_search, dense_search, rrf_fuse, rerank, mmr_select, rrf_search_prepare_context
)
import requests
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import json, math
from pathlib import Path
from .eval_utils import recall_at_k, mrr_at_k
from typing import List, Dict, Any
import numpy as np
from collections import OrderedDict
import hashlib

ANSWER_LAT_S = Histogram(
    "answer_latency_seconds",
    "End-to-end /answer latency (seconds)",
    buckets=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2, 3, 5, 8, 13, 21]
)

CACHE_MAX = 256  # saklanacak en fazla cevap
_answer_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()

gen_ms = Histogram(
    "gen_ms",
    "LLM üretim süresi (ms)",
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, float("inf")),
)

tokens_per_s = Gauge("tokens_per_s", "LLM çıkış token hızı (token/s)")

app = FastAPI(title="GPT-OSS-20B RAG API")
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
GPU_MEM = Gauge("gpu_mem_mb", "GPU memory usage in MB")
EVAL_RECALL_G = Gauge("eval_recall_at_k", "Eval recall@k", ["set_name"])
EVAL_MRR_G    = Gauge("eval_mrr_at_k", "Eval MRR@k", ["set_name"])
EVAL_SAMPLES_G= Gauge("eval_samples", "Eval sample count", ["set_name"])
GEN_SIM_AVG = Gauge("eval_gen_semantic_sim_avg", "Average semantic similarity of generated answers", ["set_name"])
GEN_FAITH_AVG = Gauge("eval_gen_faithfulness_avg", "Average faithfulness proxy (answer vs context similarity)", ["set_name"])
GEN_SAMPLES = Gauge("eval_gen_samples", "Number of samples evaluated (generation eval)", ["set_name"])
# Prometheus
CACHE_HIT_C        = Counter("cache_hit_total",  "Answer cache hits")
CACHE_MISS_C       = Counter("cache_miss_total", "Answer cache misses")
CACHE_SIZE_G       = Gauge("cache_size",         "Answer cache current size")
CACHE_HIT_RATIO_G  = Gauge("cache_hit_ratio",    "Answer cache hit ratio (0..1)")
# app/main.py (cache’i başlatırken)
USE_CACHE = getattr(settings, "CACHE_ENABLED", True)
CACHE_TTL = getattr(settings, "CACHE_TTL", 300)
CACHE_MAX_ITEMS = getattr(settings, "CACHE_MAX_ITEMS", 1024)



_cache_hits = 0
_cache_misses = 0

class AnswerIn(BaseModel):
    query: str
    k_final: int | None = 5   # kaç pasajı bağlama alalım?

class AnswerOut(BaseModel):
    answer: str
    sources: list[dict]
    durations_ms: dict
    tokens_per_s: float | None = None



@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "vector_db": settings.VECTOR_DB,
        "ollama_host": settings.OLLAMA_HOST,
        "qdrant": {"host": settings.QDRANT_HOST, "port": settings.QDRANT_PORT},
        "model": settings.MODEL_NAME,
    }

@app.get("/ping/ollama")
def ping_ollama():
    try:
        r = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/ping/qdrant")
def ping_qdrant():
    try:
        r = requests.get(f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/healthz", timeout=5)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}
@app.post("/guard/sanitize")
def guard_sanitize(payload: dict = Body(...)):
    text = str(payload.get("text", ""))
    lang = detect_lang(text)
    masked, flags = mask_pii(text)
    return {"lang": lang, "masked_text": masked, "flags": flags}

@app.post("/admin/bm25/rebuild")
def admin_bm25_rebuild():
    try:
        n = bm25_rebuild()
        return {"indexed": n}
    except Exception as e:
        # hata olursa burada JSON döndür; bağlantı kırılmasın
        return {"error": str(e)}


@app.post("/search/rrf")
def search_rrf(payload: dict = Body(...)):
    q = str(payload.get("query", "")).strip()
    k_dense = int(payload.get("k_dense", 20))
    k_bm25  = int(payload.get("k_bm25", 50))
    k_rrf   = int(payload.get("k_rrf", 50))
    k_rer   = int(payload.get("k_rerank", 20))
    k_final = int(payload.get("k_final", 5))
    lam     = float(payload.get("mmr_lambda", 0.6))
    if not q:
        return {"matches": [], "trace": {"note": "empty query"}}

    t0 = time.perf_counter()
    dense = dense_search(q, top_k=k_dense)
    t1 = time.perf_counter()
    bm25  = bm25_search(q, top_k=k_bm25)
    t2 = time.perf_counter()
    fused = rrf_fuse(dense, bm25, k=60, limit=k_rrf)
    t3 = time.perf_counter()
    rer   = rerank(q, fused, top_r=k_rer)
    t4 = time.perf_counter()
    final = mmr_select(q, rer, final_k=k_final, lam=lam)
    t5 = time.perf_counter()

    return {
        "matches": final,
        "trace": {
            "dense_top": [{"source": d.get("source"), "score": d.get("score")} for d in dense[:5]],
            "bm25_top":  [{"source": b.get("source"), "bm25": b.get("bm25_score")} for b in bm25[:5]],
            "rrf_top":   [{"source": x.get("source"), "rrf": x.get("rrf")} for x in fused[:5]],
            "rerank_top":[{"source": x.get("source"), "rerank": x.get("rerank")} for x in rer[:5]],
            "timings_ms": {
                "dense_ms": round((t1-t0)*1000,2),
                "bm25_ms":  round((t2-t1)*1000,2),
                "rrf_ms":   round((t3-t2)*1000,2),
                "rerank_ms":round((t4-t3)*1000,2),
                "mmr_ms":   round((t5-t4)*1000,2),
                "total_ms": round((t5-t0)*1000,2)
            },
            "params": {"k_dense":k_dense,"k_bm25":k_bm25,"k_rrf":k_rrf,"k_rerank":k_rer,"k_final":k_final,"lambda":lam}
        }
    }
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



@app.post("/answer")
async def answer(req: dict):
    start_ts = time.perf_counter()
    query = (req.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    k_final = int(req.get("k_final") or 5)

    # 1) Retrieval
    try:
        context, hits = rrf_search_prepare_context(query=query, k_final=k_final)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval_failed: {type(e).__name__}: {e}")

    # -------- Cache (LRU) kontrolü --------
    key = _cache_key(query, context)
    global _cache_hits, _cache_misses
    cached = _cache_get(key)
    if cached is not None:
        _cache_hits += 1
        CACHE_HIT_C.inc()
        _update_hit_ratio()
        return {
            "answer": cached,
            "used_context": hits,
            "model": settings.MODEL_NAME,
            "from_cache": True,
        }
    _cache_misses += 1
    CACHE_MISS_C.inc()
    _update_hit_ratio()
    # --------------------------------------

    # 2) Prompt (system + user)
    sys_prompt = (
        "Aşağıdaki bağlama dayanarak soruyu yanıtla. "
        "Bağlam yetersizse dürüstçe söyle. Türkçe yanıtla."
    )
    user_content = f"Soru: {query}\n\nBağlam:\n{context}\n\nCevap:"

    payload = {
        "model": settings.MODEL_NAME,  # settings.MODEL_NAME = "gpt-oss:20b"
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        # "options": {"num_ctx": 4096, "temperature": 0.2}
    }

    # 3) Ollama chat çağrısı + süre ölçümü
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post("http://ollama:11434/api/chat", json=payload)
        if resp.status_code >= 400:
            # anlamlı hata mesajı döndür
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=500, detail=f"ollama_failed: {resp.status_code} {detail}")
        data = resp.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"ollama_request_error: {e}")
    t1 = time.perf_counter()
    gen_ms.observe((t1 - t0) * 1000.0)

    # 4) Yanıtı topla (Ollama chat yanıtı: { message: { content: "..." }, ... })
    msg = (data or {}).get("message") or {}
    answer_text = (msg.get("content") or "").strip()

    # 4.a) tokens_per_s (varsa) hesapla
    # Ollama çoğu modelde aşağıdaki alanları döndürüyor (ns cinsinden):
    # total_duration, eval_count, eval_duration
    tps = None
    try:
        eval_count = float((data or {}).get("eval_count") or 0)
        eval_duration_ns = float((data or {}).get("eval_duration") or 0)  # ns
        if eval_count > 0 and eval_duration_ns > 0:
            tps = eval_count / (eval_duration_ns / 1e9)
            tokens_per_s.set(tps)
    except Exception:
        tps = None  # sessiz geç

    # 5) Cache’e yaz
    if answer_text:
        _cache_set(key, answer_text)
    time_stop = time.perf_counter() - start_ts
    ANSWER_LAT_S.observe(time_stop)
    return {
        "answer": answer_text,
        "used_context": hits,
        "model": settings.MODEL_NAME,
        "tokens_per_s": tps,
        "from_cache": False,
    }




@app.post("/admin/eval/run")
async def run_eval(payload: dict = None):
    """
    Body seçenekleri:
    {
      "path": "/data/eval/eval.jsonl",   # opsiyonel (varsayılan bu)
      "set_name": "local_smoke",         # Grafana label'ı
      "k_default": 5                     # her satırda yoksa buradan al
    }
    """
    payload = payload or {}
    set_name = (payload.get("set_name") or "local_smoke").strip()
    path = Path(payload.get("path") or "/data/eval/eval.jsonl")
    k_default = int(payload.get("k_default") or 5)

    if not path.exists():
        raise HTTPException(400, f"eval file not found: {path}")

    n = 0
    sum_recall = 0.0
    sum_mrr = 0.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = (record.get("query") or "").strip()
            gold_any = record.get("gold_any") or []
            k = int(record.get("k") or k_default)
            if not query or not gold_any:
                continue

            # retrieval zincirini içten çağır
            ctx_text, hits = rrf_search_prepare_context(query=query, k_final=k)
            pred_sources = [f'{h.get("source")}#{h.get("chunk_id")}' for h in hits]  # retrieval.py 'de source alanı var

            r = recall_at_k(pred_sources, gold_any, k)
            m = mrr_at_k(pred_sources, gold_any, k)

            sum_recall += r
            sum_mrr += m
            n += 1

    if n == 0:
        raise HTTPException(400, "no valid eval records")

    recall = sum_recall / n
    mrr = sum_mrr / n

    # Prometheus gauge’lara yaz
    EVAL_RECALL_G.labels(set_name).set(recall)
    EVAL_MRR_G.labels(set_name).set(mrr)
    EVAL_SAMPLES_G.labels(set_name).set(n)

    return {"set_name": set_name, "samples": n, "recall_at_k": recall, "mrr_at_k": mrr}

@app.post("/admin/eval/gen")
async def eval_generation(req: dict):
    """
    JSON:
    {
      "path": "/data/eval/eval_v2.jsonl",
      "set_name": "local_smoke_v2",
      "k_default": 5
    }
    JSONL satırları:
    {"query":"...", "positive":["...","..."]}  # positive zorunlu; negative yok sayıyoruz
    """
    path = (req.get("path") or "").strip()
    set_name = (req.get("set_name") or "gen_eval").strip()
    k_final = int(req.get("k_default") or 5)

    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    samples = list(_iter_jsonl(path))
    if not samples:
        raise HTTPException(status_code=400, detail="no valid eval records")

    sim_scores: List[float] = []
    faith_scores: List[float] = []

    # Cevap üretiminde /answer’daki şablona benzer chat çağrısı
    async with httpx.AsyncClient(timeout=300.0) as client:
        for ex in samples:
            q = (ex.get("query") or "").strip()
            positives = ex.get("positive") or []

            # 1) retrieval
            try:
                context, hits = rrf_search_prepare_context(query=q, k_final=k_final)
            except Exception as e:
                # retrieval fail -> skoru 0 say
                sim_scores.append(0.0)
                faith_scores.append(0.0)
                continue

            sys_prompt = (
                "Aşağıdaki bağlama dayanarak soruyu yanıtla. "
                "Bağlam yetersizse dürüstçe söyle. Türkçe yanıtla."
            )
            user_content = f"Soru: {q}\n\nBağlam:\n{context}\n\nCevap:"
            payload = {
                "model": settings.MODEL_NAME,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            }

            try:
                resp = await client.post("http://ollama:11434/api/chat", json=payload)
                if resp.status_code >= 400:
                    # hata olursa metrikte 0 verelim
                    sim_scores.append(0.0)
                    faith_scores.append(0.0)
                    continue
                data = resp.json()
            except Exception:
                sim_scores.append(0.0)
                faith_scores.append(0.0)
                continue

            msg = (data or {}).get("message") or {}
            answer_text = (msg.get("content") or "").strip()
            if not answer_text:
                sim_scores.append(0.0)
                faith_scores.append(0.0)
                continue

            # 2) skorlar
            sim = _best_sim(answer_text, positives)
            faith = _faithfulness_proxy(answer_text, context)
            sim_scores.append(sim)
            faith_scores.append(faith)

    # Prometheus’a yaz
    n = float(len(sim_scores))
    avg_sim = float(sum(sim_scores) / n) if n else 0.0
    avg_faith = float(sum(faith_scores) / n) if n else 0.0
    GEN_SIM_AVG.labels(set_name=set_name).set(avg_sim)
    GEN_FAITH_AVG.labels(set_name=set_name).set(avg_faith)
    GEN_SAMPLES.labels(set_name=set_name).set(n)

    return {
        "set_name": set_name,
        "samples": int(n),
        "avg_semantic_sim": round(avg_sim, 4),
        "avg_faithfulness": round(avg_faith, 4),
        "details": [
            {"i": i, "sim": round(sim_scores[i], 4), "faith": round(faith_scores[i], 4)}
            for i in range(len(sim_scores))
        ],
    }

@app.post("/admin/cache/clear")
async def cache_clear():
    _answer_cache.clear()
    CACHE_SIZE_G.set(0)
    CACHE_HIT_RATIO_G.set(0.0)
    return {"ok": True, "size": 0}

@app.get("/admin/cache/stats")
async def cache_stats():
    return {
        "enabled": USE_CACHE,
        "ttl": CACHE_TTL,
        "max_items": CACHE_MAX_ITEMS,
        "size": len(_answer_cache),
    }

# Yardımcı: bir hit gold_any listesiyle eşleşiyor mu?
def _hit_matches_gold(hit: dict, gold_any: list[str]) -> bool:
    txt = (hit.get("text") or "").casefold()
    src = (hit.get("source") or "")
    cid = str(hit.get("chunk_id"))
    key = f"{src}#{cid}"
    for g in gold_any or []:
        if not g:
            continue
        g_norm = g.strip()
        # 1) ID eşleşmesi (örn. nvidia_dlss_overview#0)
        if g_norm == key:
            return True
        # 2) Metin içinde substring eşleşmesi
        if g_norm.casefold() in txt:
            return True
    return False


def _norm_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = _norm_vec(a)
    b = _norm_vec(b)
    return float(np.clip(a @ b, -1.0, 1.0))

def _embed_text(text: str) -> np.ndarray:
    # bge-m3 embedder zaten ingestion için kullanılıyor; burada da reuse ediyoruz.
    from app.retrieval import get_embedder  # sende bu fonksiyon var
    emb = get_embedder()(text)
    return np.array(emb, dtype=np.float32)

def _best_sim(ans: str, refs: List[str]) -> float:
    if not refs:
        return 0.0
    ans_vec = _embed_text(ans)
    best = 0.0
    for r in refs:
        best = max(best, _cos_sim(ans_vec, _embed_text(r)))
    return best

def _faithfulness_proxy(ans: str, context: str) -> float:
    # Basit ama etkili: answer ↔ tüm context birleştirmesi (cosine)
    return _cos_sim(_embed_text(ans), _embed_text(context))


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cache_key(query: str, context: str) -> str:
    m = hashlib.sha1()
    m.update(query.encode("utf-8")); m.update(b"\x00"); m.update(context.encode("utf-8"))
    return m.hexdigest()

def _cache_get(key: str):
    v = _answer_cache.get(key)
    if v is None:
        return None
    _answer_cache.move_to_end(key)
    CACHE_HIT_C.inc()
    _update_hit_ratio()  
    return v[0]

def _cache_set(key: str, answer_text: str):
    if key in _answer_cache:
        _answer_cache.move_to_end(key)
    _answer_cache[key] = (answer_text, time.time())
        # LRU: kapasiteyi aşarsa en eskiyi at
    if len(_answer_cache) > CACHE_MAX:
        _answer_cache.popitem(last=False)
    CACHE_SIZE_G.set(len(_answer_cache))
    CACHE_MISS_C.inc()
    _update_hit_ratio()

def _update_hit_ratio():
    total = _cache_hits + _cache_misses
    if total > 0:
        CACHE_HIT_RATIO_G.set(_cache_hits / total)



