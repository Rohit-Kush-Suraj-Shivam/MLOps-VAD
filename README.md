# VAD MLOps Pipeline

> Automated Voice Activity Detection with MLflow experiment tracking, multi-branch feature engineering, and Docker deployment.

---

## Architecture Overview

```
GitHub
├── main                     ← production branch; triggers Docker deploy
├── feature/combined         ← trains on MFCC + Energy + ZCR + Spectral Centroid (16 features)
├── feature/mfcc-only        ← trains on MFCC only (13 features)
└── feature/zcr-others       ← trains on Energy + ZCR + Spectral features (5 features)

Pipeline per feature branch
  push → GitHub Actions → train.py → MLflow tracking → upload model artifact

main branch (after merge / workflow_dispatch)
  download all 3 artifacts → model_selector.py → pick best F1 → Docker build → GHCR push

Runtime (Docker Compose)
  mlflow:5000  ←  experiment UI
  api:8000     ←  FastAPI dashboard + /predict upload endpoint
```

---

## Project Structure

```
vad-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml               # runs pytest on every push / PR
│       ├── train_combined.yml   # trains on feature/combined branch
│       ├── train_mfcc_only.yml  # trains on feature/mfcc-only branch
│       ├── train_zcr_others.yml # trains on feature/zcr-others branch
│       └── deploy.yml           # selects best model + builds Docker image
├── api/
│   └── main.py                  # FastAPI app (dashboard + /predict)
├── features/
│   ├── combined.py              # 16-feature extractor
│   ├── mfcc_only.py             # 13-feature extractor
│   └── zcr_others.py            # 5-feature extractor
├── models/
│   └── .gitkeep                 # generated .pkl files go here (git-ignored)
├── tests/
│   └── test_pipeline.py         # pytest suite
├── train.py                     # MLflow-tracked training script
├── model_selector.py            # picks best branch model by F1
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Quick Start (Local, No Docker)

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/vad-mlops.git
cd vad-mlops
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Copy your dataset

```bash
# Copy balanced_vad_dataset.csv from the original MLOps-VAD repo
cp /path/to/MLOps-VAD/balanced_vad_dataset.csv .
```

### 3. Start the MLflow server

```bash
# In a separate terminal
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts
```

MLflow UI is now at **http://localhost:5000**

### 4. Train all three feature branches

```bash
# In your main terminal (venv activated)
FEATURE_SET=combined       python train.py
FEATURE_SET=mfcc_only      python train.py
FEATURE_SET=zcr_others     python train.py
```

Each run logs to MLflow and writes to `models/`:
- `model_combined.pkl` + `scaler_combined.pkl` + `meta_combined.json`
- `model_mfcc_only.pkl` + `scaler_mfcc_only.pkl` + `meta_mfcc_only.json`
- `model_zcr_others.pkl` + `scaler_zcr_others.pkl` + `meta_zcr_others.json`

### 5. Select the best model

```bash
python model_selector.py
```

Output example:
```
  combined         F1=0.8693  Acc=0.8693  AUC=0.9412  model=LogisticRegression
  mfcc_only        F1=0.8543  Acc=0.8543  AUC=0.9308  model=SVC
  zcr_others       F1=0.7812  Acc=0.7812  AUC=0.8621  model=LogisticRegression

🏆 Winner: combined  (F1=0.8693, model=LogisticRegression)
✅ Active model updated → models/active/
```

### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Dashboard: **http://localhost:8000**
API docs: **http://localhost:8000/docs**

### 7. Upload a file via the dashboard or curl

```bash
# curl
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/audio.wav"

# Response:
{
  "prediction": "Speech",
  "speech_probability": 0.873,
  "energy": 0.000512,
  "variation": 0.182,
  "model_branch": "combined",
  "model_type": "LogisticRegression",
  "features_used": ["mfcc_1", ..., "spectral_centroid"],
  "filename": "audio.wav",
  "duration_s": 3.2
}
```

---

## Docker Compose (Recommended for Full Stack)

### Prerequisites
- Docker Desktop (or Docker Engine + Compose plugin)
- The dataset CSV and pre-built `models/` directory

### Build and run

```bash
# 1. Train locally first (or the container has no model)
FEATURE_SET=combined   python train.py
FEATURE_SET=mfcc_only  python train.py
FEATURE_SET=zcr_others python train.py
python model_selector.py

# 2. Start everything
docker compose up --build

# Services:
#   http://localhost:8000  →  VAD API + Dashboard
#   http://localhost:5000  →  MLflow UI
```

### Re-train inside Docker (no local Python needed)

```bash
# One-shot trainer container (reads dataset, trains all, selects best, exits)
docker compose --profile train run --rm trainer
```

### Rebuild only the API after code changes

```bash
docker compose up --build api
```

---

## GitHub Actions Setup

### Step 1: Repository setup

```bash
# Create the repo and push
git init
git remote add origin https://github.com/YOUR_USERNAME/vad-mlops.git
git add .
git commit -m "Initial MLOps pipeline"
git push -u origin main
```

### Step 2: Create the three feature branches

```bash
git checkout -b feature/combined
git push -u origin feature/combined

git checkout main
git checkout -b feature/mfcc-only
git push -u origin feature/mfcc-only

git checkout main
git checkout -b feature/zcr-others
git push -u origin feature/zcr-others
```

### Step 3: Add your dataset to each branch

The CSV is git-ignored (too large). You have two options:

**Option A – Git LFS** (recommended)
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes balanced_vad_dataset.csv
git commit -m "Add dataset via LFS"
git push
```

**Option B – GitHub Release asset**
Upload `balanced_vad_dataset.csv` as a Release asset, then add a download step to each workflow:
```yaml
- name: Download dataset
  run: |
    curl -L -o balanced_vad_dataset.csv \
      https://github.com/YOUR_USERNAME/vad-mlops/releases/download/v1.0/balanced_vad_dataset.csv
```

### Step 4: Trigger training on each branch

Push any change to each feature branch to trigger its workflow:

```bash
# Trigger combined training
git checkout feature/combined
git commit --allow-empty -m "Trigger training"
git push

# Trigger mfcc-only training
git checkout feature/mfcc-only
git commit --allow-empty -m "Trigger training"
git push

# Trigger zcr-others training
git checkout feature/zcr-others
git commit --allow-empty -m "Trigger training"
git push
```

Watch progress at: `https://github.com/YOUR_USERNAME/vad-mlops/actions`

### Step 5: Deploy (model selection + Docker)

After all 3 branch workflows complete successfully:

```bash
# Trigger the deploy workflow manually
# GitHub UI: Actions → "Select Best Model & Deploy" → Run workflow
# OR:
git checkout main
git commit --allow-empty -m "Trigger deploy"
git push
```

The deploy workflow will:
1. Download model artifacts from all 3 branch workflows
2. Run `model_selector.py` to pick the best F1
3. Build the Docker image with the winning model baked in
4. Push to `ghcr.io/YOUR_USERNAME/vad-mlops:latest`

### Step 6: Pull and run the published image

```bash
docker pull ghcr.io/YOUR_USERNAME/vad-mlops:latest
docker run -p 8000:8000 ghcr.io/YOUR_USERNAME/vad-mlops:latest
```

---

## API Endpoints

| Method | Path          | Description                              |
|--------|---------------|------------------------------------------|
| GET    | `/`           | Interactive dashboard (HTML)             |
| GET    | `/health`     | Health check                             |
| GET    | `/model-info` | Active model metadata + metrics          |
| GET    | `/metrics`    | All branch model metrics (JSON)          |
| POST   | `/predict`    | Upload audio file → VAD prediction       |
| GET    | `/docs`       | Auto-generated Swagger UI                |
| GET    | `/redoc`      | ReDoc API documentation                  |

### POST /predict

- **Content-Type**: `multipart/form-data`
- **Field**: `file` (audio file)
- **Supported formats**: WAV, MP3, MP4, FLAC, OGG, M4A
- **Max size**: 50 MB

---

## Feature Branches Explained

| Branch             | Features Used                                          | Count |
|--------------------|--------------------------------------------------------|-------|
| `feature/combined` | MFCC (13) + Energy + ZCR + Spectral Centroid           | 16    |
| `feature/mfcc-only`| MFCC coefficients only                                 | 13    |
| `feature/zcr-others`| Energy + ZCR + Spectral Centroid + Rolloff + Bandwidth | 5     |

Each branch trains two models (Logistic Regression + SVC), picks the better one by F1, and logs everything to MLflow.

---

## MLflow Experiments

After training, you'll see these experiments in the MLflow UI:

- `VAD-combined` — runs for the combined feature set
- `VAD-mfcc_only` — runs for MFCC-only
- `VAD-zcr_others` — runs for ZCR + others
- `VAD-ModelSelection` — the cross-branch selection run

Each run tracks: accuracy, precision, recall, F1, ROC-AUC, CV F1 mean/std, confusion matrix cells.

---

## Running Tests

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v --tb=short
```

---

## Updating the Model (Re-training Workflow)

When you have new data:

1. Update `balanced_vad_dataset.csv`
2. Push to any feature branch → GitHub Actions re-trains automatically
3. Go to Actions → "Select Best Model & Deploy" → Run workflow
4. The API image is rebuilt with the new best model

Or locally:
```bash
python train.py                # re-trains current FEATURE_SET
python model_selector.py       # re-picks best
docker compose up --build api  # redeploys
```

---

## Troubleshooting

**`No active model found`**
→ Run `python model_selector.py` — you must train at least one branch first.

**`ModuleNotFoundError: No module named 'librosa'`**
→ Run `pip install -r requirements.txt` in your venv.

**`PortAudio library not found`** (sounddevice error)
→ This project uses file upload for inference, not mic recording. This error can be safely ignored.

**Docker build fails on `COPY models/active/`**
→ You must run `model_selector.py` locally before the first Docker build so `models/active/` exists.

**GitHub Actions: `No meta file for combined`**
→ The branch workflow hasn't run yet, or the artifact expired (30-day retention). Re-trigger the branch workflow.

**MLflow UI shows no experiments**
→ Make sure `MLFLOW_TRACKING_URI` points to the same server used during training.
