FROM python:3.11-slim

# Sistem paketleri (curl: healthcheck için)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bağımlılıkları önce kopyala (build cache avantajı)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Uygulama kodu
COPY . /app

# Uvicorn ile (reload açık; dev ortamında klasörü mount edeceğiz)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
