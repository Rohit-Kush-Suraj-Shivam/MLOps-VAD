# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy source code
COPY features/   features/
COPY api/        api/
COPY train.py    train.py
COPY model_selector.py model_selector.py

# Copy models (active model must exist at build time, or mount at runtime)
COPY models/     models/

# Create feedback and log files
RUN touch feedback_data.csv prediction_log.csv && \
    echo "label,mfcc_1,mfcc_2,mfcc_3,mfcc_4,mfcc_5,mfcc_6,mfcc_7,mfcc_8,mfcc_9,mfcc_10,mfcc_11,mfcc_12,mfcc_13,energy,zcr,spectral_centroid,timestamp,source" > feedback_data.csv

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app
ENV ACTIVE_MODEL_DIR=/app/models/active
ENV MODEL_OUT_DIR=/app/models
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PORT=8000
ENV FEEDBACK_THRESHOLD=50

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
