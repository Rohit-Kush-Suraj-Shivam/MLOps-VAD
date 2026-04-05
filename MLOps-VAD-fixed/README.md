# VAD MLOps Pipeline

Voice Activity Detection with automated MLOps pipeline using MLflow, GitHub Actions, and Docker.

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MLOps-VAD.git
cd MLOps-VAD

# Install dependencies (for local development)
pip install -r requirements.txt
```

### 2. Run with Docker Compose (Recommended)

```bash
# Start MLflow and API
docker-compose up -d mlflow api

# Access:
# - API: http://localhost:8000
# - MLflow UI: http://localhost:5000
# - API Docs: http://localhost:8000/docs

# Train models
docker-compose run --rm trainer

# Check feedback status
docker-compose run --rm auto-updater
```

### 3. Run with GitHub Actions

```bash
# Trigger unified pipeline (trains all models, selects best, deploys)
git commit --allow-empty -m "Trigger training pipeline"
git push origin main
```

## Features

- **Automated Training:** Trains 3 models with different feature sets in parallel
- **Model Selection:** Automatically selects best model based on F1 score
- **MLflow Tracking:** Complete experiment tracking and model versioning
- **Auto-Update:** Retrains models when feedback threshold is reached
- **Feedback Collection:** API endpoints for continuous improvement
- **Docker Deployment:** Containerized API with health checks

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML Dashboard |
| `GET /health` | Health check |
| `GET /model-info` | Active model info |
| `POST /predict` | Upload audio for prediction |
| `POST /feedback` | Submit feedback |
| `GET /feedback-stats` | Feedback statistics |

## Project Structure

```
├── .github/workflows/     # CI/CD workflows
├── api/main.py           # FastAPI application
├── features/             # Feature extraction
├── models/active/        # Active model storage
├── train.py              # Training script
├── model_selector.py     # Model selection
└── docker-compose.yml    # Docker orchestration
```

## Workflows

1. **Unified Pipeline** - Train, select, and deploy in one workflow
2. **Auto Update** - Feedback-based retraining
3. **Individual Training** - Train specific feature sets

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FEATURE_SET` | combined | Feature set to use |
| `MLFLOW_TRACKING_URI` | http://localhost:5000 | MLflow server URL |
| `MODEL_OUT_DIR` | models | Model output directory |
| `FEEDBACK_THRESHOLD` | 50 | Samples needed for retraining |

## Documentation

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed instructions.

## License

MIT
