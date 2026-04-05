# MLOps VAD – Complete Setup & Run Guide

## What Was Fixed

### Bug: `NameError` in `model_selector.py` (line 222)

**Root Cause:**  
Inside the `main()` function, the list comprehension at line 218–228 referenced variable `f`, but the loop unpacked tuples as `(f1, b, m)`. Separately, Python also saw `f` assigned later in the same function via `with open(...) as f:` (line 232). This caused Python to treat `f` as a "cell variable" — a local that hadn't been initialized yet when the list comprehension ran — triggering `NameError: cannot access free variable 'f'`.

**Fix:**
```python
# BEFORE (buggy)
"f1": f,
for rank, (f1, b, m) in enumerate(results, 1)

# AFTER (fixed)
"f1": f1,
for rank, (f1, b, m) in enumerate(results, 1)
```

**Also fixed:** Both `model_selector.py` and `train.py` defaulted to `http://localhost:5000` for MLflow when no env var was set. Changed to `file:///tmp/mlflow` so they work without a running MLflow server.

---

## Architecture Overview

```
feature/combined  ──┐
feature/mfcc-only ──┤──► train.py ──► artifacts ──► model_selector.py ──► models/active/ ──► Docker image
feature/zcr-others ─┘                               (picks best F1)
```

Three training branches each train a model independently. The deploy job downloads all three artifacts, runs `model_selector.py` to pick the best by F1, and builds/pushes a Docker image.

---

## Option A – Local Development (Simplest)

### 1. Prerequisites
```bash
python -m pip install -r requirements.txt
```

### 2. Start the MLflow UI (optional but recommended)
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
# Open http://localhost:5000
```

### 3. Train all three models
```bash
# In separate terminals or sequentially:
MLFLOW_TRACKING_URI=http://localhost:5000 FEATURE_SET=combined    python train.py
MLFLOW_TRACKING_URI=http://localhost:5000 FEATURE_SET=mfcc_only   python train.py
MLFLOW_TRACKING_URI=http://localhost:5000 FEATURE_SET=zcr_others  python train.py
```

### 4. Select the best model
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python model_selector.py
```

### 5. Start the API
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn api.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

---

## Option B – Docker Compose (Recommended for Teams)

### Start everything
```bash
# Start MLflow + API
docker compose up -d mlflow api

# Run training (one-shot)
docker compose run --rm trainer

# Start auto-updater (feedback-based retraining)
docker compose --profile auto-update run --rm auto-updater
```

### Access
| Service   | URL                        |
|-----------|----------------------------|
| MLflow UI | http://localhost:5000      |
| VAD API   | http://localhost:8000/docs |

---

## Option C – GitHub Actions CI/CD with Remote MLflow

This is the full MLOps setup with persistent experiment tracking.

### Step 1 — Set up a remote MLflow server (DagsHub — free)

1. Create an account at https://dagshub.com
2. Create a new repo and link it to your GitHub repo
3. In your DagsHub repo, go to **Remote** → **Experiments** — you'll see a tracking URI like:
   ```
   https://dagshub.com/<your-username>/<your-repo>.mlflow
   ```
4. Get your DagsHub token from your profile → **Settings** → **Tokens**

### Step 2 — Add GitHub Secrets

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name            | Value                                                          |
|------------------------|----------------------------------------------------------------|
| `MLFLOW_TRACKING_URI`  | `https://dagshub.com/<username>/<repo>.mlflow`                |
| `DAGSHUB_USERNAME`     | Your DagsHub username                                          |
| `DAGSHUB_TOKEN`        | Your DagsHub access token                                      |

> If you skip these, the pipeline still works using local file-based MLflow — runs are uploaded as GitHub artifacts (`mlflow-tracking-db-*`) and you can inspect them locally.

### Step 3 — Create feature branches

```bash
git checkout -b feature/combined
git push origin feature/combined

git checkout -b feature/mfcc-only
git push origin feature/mfcc-only

git checkout -b feature/zcr-others
git push origin feature/zcr-others
```

### Step 4 — Trigger the CI/CD pipeline

Push to any training branch to trigger that branch's training workflow:

```bash
git checkout feature/combined
git commit --allow-empty -m "trigger training"
git push origin feature/combined
```

Each training job:
1. Trains the model
2. Logs metrics to MLflow (DagsHub or file-based)
3. Uploads `model_combined.pkl`, `scaler_combined.pkl`, `meta_combined.json` as GitHub artifacts

Once all three training workflows complete, push to `main` to trigger the deploy workflow:

```bash
git checkout main
git merge feature/combined
git push origin main
```

The deploy workflow:
1. Downloads artifacts from all three training jobs
2. Runs `model_selector.py` (picks best F1, logs selection run to MLflow)
3. Commits `models/active/` back to the repo
4. Builds and pushes Docker image to `ghcr.io`

### Step 5 — View MLflow experiments

**If using DagsHub:**  
Go to `https://dagshub.com/<username>/<repo>` → **Experiments** tab — all training and selection runs are visible.

**If using file-based tracking:**  
Download the `mlflow-tracking-db-*` artifact from any GitHub Actions run, then:
```bash
unzip mlflow-tracking-db-*.zip -d /tmp/mlflow
mlflow ui --backend-store-uri file:///tmp/mlflow
# Open http://localhost:5000
```

---

## Automated Feedback-Based Retraining

The `auto_update.yml` workflow retrains automatically when enough user feedback is collected.

### How feedback flows
1. Users interact with the API and corrections are appended to `feedback_data.csv`
2. The workflow checks every 6 hours (scheduled cron) or on push to `feedback_data.csv`
3. If `feedback_data.csv` has ≥ 50 rows, retraining is triggered automatically
4. After training, `model_selector.py` picks the best model and updates `models/active/`
5. A new Docker image is built and pushed
6. Processed feedback is archived and `feedback_data.csv` is reset

### Manual trigger
```
GitHub → Actions → "Auto Model Update" → Run workflow → Optionally set force_retrain=true
```

---

## Project File Reference

```
.
├── train.py               # Trains one model (FEATURE_SET env var)
├── model_selector.py      # Picks best model by F1, logs to MLflow  ← BUG FIXED HERE
├── api/main.py            # FastAPI inference server
├── balanced_vad_dataset.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml     # Local: MLflow server + API + trainer
└── .github/workflows/
    ├── train_combined.yml    # Triggers on feature/combined
    ├── train_mfcc_only.yml   # Triggers on feature/mfcc-only
    ├── train_zcr_others.yml  # Triggers on feature/zcr-others
    ├── deploy.yml            # Triggers on main push or workflow_run
    └── auto_update.yml       # Feedback-based scheduled retraining
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `NameError: cannot access free variable 'f'` | Fixed — was `"f1": f`, now `"f1": f1` |
| `MLflow connection refused` | Either start `mlflow server` locally or set `MLFLOW_TRACKING_URI` secret |
| `No meta file for combined` | Training hasn't run yet — run `train.py` with `FEATURE_SET=combined` first |
| `Model file not found: models/model_combined.pkl` | Same as above |
| `Artifact expired` in deploy job | Re-run the training workflow — GitHub artifacts expire after 30 days |
| Docker image push fails | Ensure `packages: write` permission is set in the job, and GHCR is enabled for your org |
