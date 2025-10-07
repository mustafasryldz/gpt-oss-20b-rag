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

# Eğer Ollama Docker container’ı içinde çalışıyorsa:
docker exec -it ollama ollama pull gpt-oss:20b

# 4) Docker Compose ile tüm servisleri ayağa kaldır
docker compose up -d --build

# Servis adresleri:
# Frontend (React UI):   http://localhost:3000
# Backend (FastAPI):     http://localhost:8000
# Swagger dokümantasyonu: http://localhost:8000/docs
# Prometheus metrics:    http://localhost:8000/metrics
# Grafana dashboard:     http://localhost:3001

# 5) Veri ingest etmek için proje klasöründe "data" klasörü oluştur
# Ingest edilecek verileri "data" klasörünün içine aktar.
# Ardından ingest işlemini tamamlamak için:
docker exec -it api python -m app.ingest

# (İlk çalıştırmada Hugging Face'ten BGE-M3 ve BGE-Reranker modelleri indirilecek (~4 GB toplam).
# Bu işlem sırasında tokenizer.json, pytorch_model.bin, model.safetensors gibi dosyalar otomatik olarak indirilir.
# İndirme tamamlandıktan sonra ingest işlemi devam eder ve Qdrant veritabanına veri yüklenir.)


# 6) Servisleri durdurmak için
docker compose down
# (veya sadece geçici durdurma için: docker compose stop)
