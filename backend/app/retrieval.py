import re, threading
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer
import time
from prometheus_client import Histogram

from .settings import settings
from functools import lru_cache
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # requirements'ta varsa dolacaktır

FlagReranker = None  # lazy import

_COLLECTION = "rag_chunks"          # ingestion ile aynı
_EMB_NAME   = "BAAI/bge-m3"         # dense embedder
_RERANK_NAME= "BAAI/bge-reranker-v2-m3"  # cross-encoder

RETRIEVAL_STAGE_MS = Histogram(
    "retrieval_stage_ms",
    "Retrieval aşama süreleri (ms)",
    ["stage"],
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 3000, 5000)
)

# ---- Tekil nesneler (lazy) ----
_q_client = None
_embedder = None
_reranker = None
_bm25 = None
_bm25_docs: List[Dict] = []
_lock = threading.Lock()




# ---- Basit TR/EN tokenizer ----
_WORD = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", re.UNICODE)
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]

def get_qdrant_only():
    from .settings import settings
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

def timed(stage: str):
    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                RETRIEVAL_STAGE_MS.labels(stage=stage).observe(dt_ms)
        return wrapper
    return deco

# --- Embedder singleton (for eval/gen etc.) ---
def _pick_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

@lru_cache(maxsize=1)
def get_embedder():
    """
    Return a callable: str -> List[float]
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")

    model = SentenceTransformer("BAAI/bge-m3", device=_pick_device())

    def _encode(text: str):
        if not text:
            return []
        vec = model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()

    return _encode
# --- /Embedder singleton ---

def get_clients():
    global _q_client, _embedder
    if _q_client is None:
        from .settings import settings
        _q_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    if _embedder is None:
        _embedder = SentenceTransformer(_EMB_NAME)
    return _q_client, _embedder

# ---- BM25 in-memory index (Qdrant'tan yükler) ----
@timed("bm25_rebuild")
def bm25_rebuild(max_points: int = 50_000):
    global _bm25, _bm25_docs
    client = get_qdrant_only()
    _bm25_docs = []
    # güvenli scroll sayfalama
    next_offset = None
    fetched = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=_COLLECTION,
            with_payload=True,
            limit=min(2048, max_points - fetched),
            offset=next_offset
        )
        if not points:
            break
        for p in points:
            pl = p.payload or {}
            txt = pl.get("text") or ""
            if not txt:
                continue
            _bm25_docs.append({
                "id": str(p.id),
                "doc_id": pl.get("doc_id"),
                "chunk_id": pl.get("chunk_id"),
                "source": pl.get("source"),
                "doc_title": pl.get("doc_title"),
                "text": txt,
            })
        fetched += len(points)
        if not next_offset or fetched >= max_points:
            break

    if not _bm25_docs:
        _bm25 = BM25Okapi([[]])   # boş corpus için
        return 0

    corpus = [tokenize(d["text"]) for d in _bm25_docs]
    _bm25 = BM25Okapi(corpus)
    return len(_bm25_docs)


@timed("bm25")
def bm25_search(query: str, top_k: int = 50) -> list:
    global _bm25, _bm25_docs
    if _bm25 is None or not _bm25_docs:
        bm25_rebuild()
        if _bm25 is None or not _bm25_docs:
            return []
    toks = tokenize(query)
    scores = _bm25.get_scores(toks)
    idxs = np.argsort(scores)[::-1][:top_k]
    out = []
    for i in idxs:
        d = _bm25_docs[int(i)]
        out.append({**d, "bm25_score": float(scores[int(i)])})
    return out

# ---- Dense search (Qdrant) ----
@timed("dense")
def dense_search(query: str, top_k: int = 20) -> List[Dict]:
    client, embedder = get_clients()

    q_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = client.search(
        collection_name=_COLLECTION,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )
    out = []
    for r in res:
        p = r.payload or {}
        out.append({
            "id": str(r.id),
            "score": float(r.score),
            "doc_id": p.get("doc_id"),
            "chunk_id": p.get("chunk_id"),
            "source": p.get("source"),
            "doc_title": p.get("doc_title"),
            "text": p.get("text"),
        })
    return out

# ---- RRF füzyon (rank temelli) ----
@timed("rrf")
def rrf_fuse(dense: List[Dict], bm25: List[Dict], k: int = 60, limit: int = 50) -> List[Dict]:
    # id olarak (doc_id, chunk_id, source, text) kombinasyonunu kullanıyoruz
    def key(d): return (d.get("doc_id"), d.get("chunk_id"), d.get("source"))
    ranks: Dict[Tuple, float] = {}
    def add(list_):
        for rank, d in enumerate(list_, start=1):
            ranks[key(d)] = ranks.get(key(d), 0.0) + 1.0 / (k + rank)
    add(dense)
    add(bm25)
    # birleşik küme
    merged = { key(d): d for d in (dense + bm25) }
    scored = [(kk, vv, ranks.get(kk, 0.0)) for kk, vv in merged.items()]
    scored.sort(key=lambda x: x[2], reverse=True)
    return [s[1] | {"rrf": s[2]} for s in scored[:limit]]

def rrf_search(
    query: str,
    k_dense: int = 20,
    k_bm25: int = 50,
    k_rrf: int = 50,
    k_rerank: int = 20,
    k_final: int = 5,
    lam: float = 0.6,
):
    """
    Dense + BM25 -> RRF -> Rerank -> MMR zinciri.
    'matches' anahtarında son listeyi döndürür (main.py ile aynı sıralama).
    """
    dense = dense_search(query, top_k=k_dense)
    bm25  = bm25_search(query, top_k=k_bm25)
    fused = rrf_fuse(dense, bm25, k=60, limit=k_rrf)
    rer   = rerank(query, fused, top_r=k_rerank)
    final = mmr_select(query, rer, final_k=k_final, lam=lam)
    return {"matches": final}

# ---- Reranker (cross-encoder) ----
@timed("rerank")
def rerank(query: str, items: list, top_r: int = 20) -> list:
    global _reranker, FlagReranker
    if _reranker is None:
        if FlagReranker is None:
            from FlagEmbedding import FlagReranker as _FR
            FlagReranker = _FR
        _reranker = FlagReranker(_RERANK_NAME, use_fp16=True)
    pairs = [[query, it["text"]] for it in items]
    scores = _reranker.compute_score(pairs, normalize=True)
    for it, sc in zip(items, scores):
        it["rerank"] = float(sc)
    items.sort(key=lambda x: x["rerank"], reverse=True)
    return items[:top_r]

# ---- MMR (çeşitlilik) ----
@timed("mmr")
def mmr_select(query: str, items: List[Dict], final_k: int = 5, lam: float = 0.6) -> List[Dict]:
    _, embedder = get_clients()
    qv = embedder.encode([query], normalize_embeddings=True)[0]
    tv = embedder.encode([it["text"] for it in items], normalize_embeddings=True)
    selected = []
    candidates = list(range(len(items)))
    if not candidates:
        return []
    # ilk en alakalıyı al (rerank skoruna göre)
    best = max(candidates, key=lambda i: items[i].get("rerank", 0.0))
    selected.append(best)
    candidates.remove(best)
    while len(selected) < min(final_k, len(items)) and candidates:
        def gain(i):
            rel = float(np.dot(qv, tv[i]))
            div = max(float(np.dot(tv[i], tv[j])) for j in selected) if selected else 0.0
            return lam * rel - (1 - lam) * div
        next_i = max(candidates, key=gain)
        selected.append(next_i)
        candidates.remove(next_i)
    return [items[i] for i in selected]


def rrf_search_prepare_context(query: str, k_final: int = 5) -> tuple[str, list[dict]]:
    """
    RRF zincirini çalıştırır, top-k parçaları birleştirip bağlam string’i döner.
    """
    result = rrf_search(query=query, k_final=k_final)  # mevcut fonksiyonun
    hits = result["matches"] if isinstance(result, dict) else result

    # bağlam metni (başlık/prefix varsa ekle)
    chunks = []
    for h in hits:
        prefix = ""
        if "doc_title" in h and h["doc_title"]:
            prefix = f"[{h['doc_title']}] "
        elif "section" in h and h["section"]:
            prefix = f"[{h['section']}] "
        text = h.get("text") or h.get("chunk") or ""
        chunks.append(f"{prefix}{text}".strip())

    context = "\n\n---\n\n".join(chunks)
    # uzunluk emniyeti: 6–8k pencerene göre kırpabilirsiniz (opsiyonel)
    max_chars = 12000
    if len(context) > max_chars:
        context = context[:max_chars]

    return context, hits
