# GPT-OSS-20B RAG Chatbot (FastAPI + React + Qdrant + Prometheus)

Gelişmiş bir **RAG (Retrieval-Augmented Generation)** chatbot mimarisi.  
Backend → FastAPI, Frontend → React + Tailwind, LLM → GPT-OSS-20B (Ollama),  
Vector DB → Qdrant, Fusion → RRF + BM25, Reranker → BGE-Reranker-v2-M3,  
Metrics → Prometheus + Grafana.

---

## Projeyi Çalıştırmak İçin

```bash
# 1) Depoyu klonla
git clone <repo_url>
cd gpt-oss20-rag

# 2) Ortam dosyasını hazırla (.env.example → .env kopyala)
cp .env.example .env
# Windows PowerShell:
# copy .env.example .env

# 3) Ollama modelini indir
ollama pull gpt-oss:20b

# 4) Docker Compose ile tüm servisleri ayağa kaldır
docker compose up -d --build

# Servis adresleri:
# Frontend (React UI):   http://localhost:3000
# Backend (FastAPI):     http://localhost:8000
# Swagger dokümantasyonu: http://localhost:8000/docs
# Prometheus metrics:    http://localhost:8000/metrics
# Grafana dashboard:     http://localhost:3001

# 5) Veri ingest etmek için (örnek):
docker exec -it api python -m app.ingest

# 6) Servisleri durdurmak için
docker compose down
# (veya sadece geçici durdurma için: docker compose stop)
