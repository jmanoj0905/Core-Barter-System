FROM python:3.11-slim
WORKDIR /app

# Install build tools for hf-transfer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV HF_TOKEN=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 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
