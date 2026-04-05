# ── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.docker.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY api/           ./api/
COPY mlflow_pipeline/ ./mlflow_pipeline/
COPY models/        ./models/
COPY model.pkl      ./model.pkl
COPY scaler.pkl     ./scaler.pkl
COPY balanced_vad_dataset.csv ./balanced_vad_dataset.csv

# Ensure __init__.py in mlflow_pipeline
RUN touch mlflow_pipeline/__init__.py

# Environment
ENV MODEL_POLL_SEC=60
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Entrypoint: optionally retrain first, then serve
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
