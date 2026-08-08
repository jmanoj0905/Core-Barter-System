FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV HF_TOKEN=***REMOVED-LEAKED-HF-TOKEN*** \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_SYMLINKS=1

RUN mkdir -p /data
COPY requirements.txt .
RUN pip install --no-cache-dir hf-transfer && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
