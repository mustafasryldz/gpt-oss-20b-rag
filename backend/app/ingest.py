import os, time, hashlib, uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from .settings import settings
from .ingest_utils import load_any, scan_folder, chunk_text, clean_text

COLLECTION = "rag_chunks"
EMB_DIM = 1024  # bge-m3
OVERWRITE = False

def ensure_collection(client: QdrantClient):
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION not in cols:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMB_DIM, distance=Distance.COSINE),
        )


def hash_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def embed_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        v = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).tolist()
        embs.extend(v)
    return embs

def upsert(client: QdrantClient, vectors, payloads):
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
        for v, p in zip(vectors, payloads)
    ]
    client.upsert(collection_name=COLLECTION, points=points)

def run_ingest(data_dir: str = "/data"):
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, prefer_grpc=False)
    ensure_collection(client)

    model = SentenceTransformer("BAAI/bge-m3")
    files = scan_folder(data_dir)
    print(f"[ingest] found {len(files)} files under {data_dir}")

    for path in files:
        raw = load_any(path)
        if raw is None:
            print(f"[skip] unsupported: {path}")
            continue
        text = clean_text(raw)
        if not text:
            print(f"[skip] empty: {path}")
            continue

        # metadata
        doc_id = hash_id(os.path.abspath(path))
        title = os.path.basename(path)

        # chunking
        chunks = chunk_text(text, target_chars=2600, overlap_ratio=0.25)
        print(f"[{title}] chunks: {len(chunks)}")

        # payloads
        payloads = []
        for i, ch in enumerate(chunks):
            payloads.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "source": os.path.relpath(path, data_dir),
                "page": None,
                "section": None,
                "created_at": int(time.time()),
                "tags": [],
                "doc_title": title,
                "text": ch,
            })

        # embeddings
        vecs = embed_batches(model, [p["text"] for p in payloads])

        # upsert
        upsert(client, vecs, payloads)
        print(f"[{title}] upserted {len(payloads)} chunks")

if __name__ == "__main__":
    run_ingest()
