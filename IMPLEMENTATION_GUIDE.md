# VAD MLOps Pipeline - Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Fixed Issues](#fixed-issues)
3. [New Features](#new-features)
4. [File Structure](#file-structure)
5. [Setup Instructions](#setup-instructions)
6. [Running the Pipeline](#running-the-pipeline)
7. [GitHub Actions Workflows](#github-actions-workflows)
8. [MLflow Integration](#mlflow-integration)
9. [Auto-Update Mechanism](#auto-update-mechanism)
10. [API Endpoints](#api-endpoints)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the fixed and enhanced MLOps pipeline for Voice Activity Detection (VAD). The pipeline now includes:

- **Fixed deploy workflow** - Proper artifact handling and model persistence
- **MLflow integration** - Complete experiment tracking
- **Auto-update mechanism** - Automatic model retraining based on feedback
- **Feedback collection** - API endpoints for continuous improvement
- **Unified CI/CD pipeline** - Single workflow for train, select, and deploy

---

## Fixed Issues

### 1. Deploy Workflow Artifact Download Issue
**Problem:** The original `deploy.yml` tried to download artifacts from other workflows, but GitHub Actions artifacts are workflow-specific and expire.

**Solution:** 
- Created `unified_pipeline.yml` that trains all models in parallel and then selects/deploys
- Added fallback options to download from releases or use existing models
- Models are now committed back to the repository for persistence

### 2. Missing Model Persistence
**Problem:** Models were only stored as artifacts and would expire.

**Solution:**
- Active models are now committed to `models/active/` directory
- GitHub Releases are created with model artifacts
- Docker images are tagged with both `latest` and commit SHA

---

## New Features

### 1. Unified Pipeline Workflow
- Trains all 3 feature sets in parallel
- Automatically selects the best model
- Deploys to GitHub Container Registry
- Creates GitHub Release with artifacts

### 2. Auto-Update Workflow
- Monitors feedback data collection
- Automatically triggers retraining when threshold reached
- Merges feedback into training data
- Creates new model releases

### 3. Feedback Collection API
- `/feedback` endpoint for submitting corrections
- `/feedback-stats` for monitoring collection progress
- `/trigger-update` for manual retraining trigger

### 4. Enhanced MLflow Integration
- All training runs logged to MLflow
- Model comparison reports
- Confusion matrices and classification reports
- Artifact versioning

---

## File Structure

```
MLOps-VAD/
├── .github/
│   └── workflows/
│       ├── unified_pipeline.yml    # Main CI/CD pipeline
│       ├── deploy.yml               # Fixed deploy workflow
│       ├── auto_update.yml          # Feedback-based retraining
│       ├── train_combined.yml       # Individual training workflows
│       ├── train_mfcc_only.yml
│       └── train_zcr_others.yml
├── api/
│   └── main.py                      # Enhanced API with feedback
├── features/                        # Feature extraction modules
├── models/
│   └── active/                      # Active model storage
├── train.py                         # Enhanced training script
├── model_selector.py                # Enhanced model selection
├── requirements.txt                 # Updated dependencies
├── docker-compose.yml               # Updated compose file
├── Dockerfile                       # Updated Dockerfile
└── IMPLEMENTATION_GUIDE.md          # This guide
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- GitHub account with repository access
- (Optional) GitHub CLI (`gh`)

### 1. Clone and Setup Repository

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/MLOps-VAD.git
cd MLOps-VAD

# Copy the fixed files
cp -r /path/to/MLOps-VAD-fixed/* .

# Commit the changes
git add .
git commit -m "Fix deploy workflow and add auto-update mechanism"
git push origin main
```

### 2. Configure GitHub Secrets

No additional secrets required! The workflows use:
- `GITHUB_TOKEN` (automatically provided)

### 3. Initial Model Training

#### Option A: Using GitHub Actions (Recommended)
```bash
# Trigger unified pipeline
git commit --allow-empty -m "Trigger initial training"
git push origin main
```

#### Option B: Local Training
```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow (optional)
mlflow server --host 0.0.0.0 --port 5000 &

# Train all models
FEATURE_SET=combined python train.py
FEATURE_SET=mfcc_only python train.py
FEATURE_SET=zcr_others python train.py

# Select best model
python model_selector.py
```

---

## Running the Pipeline

### Local Development with Docker Compose

```bash
# Start MLflow and API
docker-compose up -d mlflow api

# Access services
# - MLflow UI: http://localhost:5000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs

# Train models locally
docker-compose run --rm trainer

# Check feedback and auto-update
docker-compose run --rm auto-updater
```

### Production Deployment

```bash
# Pull and run latest Docker image
docker pull ghcr.io/YOUR_USERNAME/vad-mlops:latest
docker run -p 8000:8000 ghcr.io/YOUR_USERNAME/vad-mlops:latest
```

---

## GitHub Actions Workflows

### 1. Unified Pipeline (`unified_pipeline.yml`)

**Triggers:**
- Push to `main`, `feature/combined`, `feature/mfcc-only`, `feature/zcr-others`
- Manual dispatch

**Jobs:**
1. `train-combined` - Trains combined feature model
2. `train-mfcc-only` - Trains MFCC-only model
3. `train-zcr-others` - Trains ZCR-others model
4. `select-and-deploy` - Selects best model and deploys
5. `test-deployment` - Runs tests on deployed model

**Usage:**
```bash
# Trigger manually via GitHub UI or CLI
git push origin main
```

### 2. Auto Update (`auto_update.yml`)

**Triggers:**
- Every 6 hours (scheduled)
- Push to `feedback_data.csv`
- Manual dispatch

**Jobs:**
1. `check-feedback` - Checks if enough feedback collected
2. `retrain-models` - Retrains all models with feedback
3. `select-and-deploy` - Selects best and deploys

**Usage:**
```bash
# Trigger manually
git commit --allow-empty -m "Check feedback"
git push origin main
```

### 3. Deploy (`deploy.yml`)

**Triggers:**
- Push to `main`
- Completion of training workflows
- Manual dispatch

**Options:**
- `use_artifacts` - Download from artifacts (default: true)
- `use_releases` - Download from releases (default: false)

**Usage:**
```bash
# Trigger via GitHub UI with options
```

---

## MLflow Integration

### Accessing MLflow UI

```bash
# Local
docker-compose up -d mlflow
open http://localhost:5000

# GitHub Actions
# MLflow data is uploaded as artifacts
```

### MLflow Experiments

- **VAD-combined** - Combined feature training runs
- **VAD-mfcc_only** - MFCC-only training runs
- **VAD-zcr_others** - ZCR-others training runs
- **VAD-ModelSelection** - Model selection runs

### Logged Metrics

- `test_f1`, `test_accuracy`, `test_precision`, `test_recall`, `test_roc_auc`
- `cv_f1_mean`, `cv_f1_std` (cross-validation scores)
- `winner_f1`, `winner_accuracy` (model selection)

### Logged Artifacts

- `model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `meta.json` - Model metadata
- `confusion_matrix_*.json` - Confusion matrices
- `classification_report_*.json` - Classification reports

---

## Auto-Update Mechanism

### How It Works

1. **Feedback Collection:**
   - Users submit feedback via `/feedback` endpoint
   - Feedback includes true label and optional audio file
   - Data stored in `feedback_data.csv`

2. **Threshold Monitoring:**
   - Default threshold: 50 feedback samples
   - Checked every 6 hours via scheduled workflow
   - Also checked on each prediction

3. **Automatic Retraining:**
   - Feedback merged with original training data
   - All 3 models retrained
   - Best model selected and deployed
   - New GitHub Release created

4. **Model Archiving:**
   - Previous models archived in `models/archive/`
   - Timestamped for version control

### Configuration

```python
# In docker-compose.yml or environment
FEEDBACK_THRESHOLD=50  # Number of samples to trigger retraining
```

### Manual Trigger

```bash
# Via API
curl -X POST http://localhost:8000/trigger-update?force=true

# Via GitHub Actions
git commit --allow-empty -m "Force retraining"
git push origin main
```

---

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | HTML Dashboard |
| `/health` | GET | Health check |
| `/model-info` | GET | Active model metadata |
| `/metrics` | GET | All model metrics |

### Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload audio, get VAD prediction |

### Feedback Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/feedback` | POST | Submit feedback for predictions |
| `/feedback-stats` | GET | Feedback collection statistics |
| `/trigger-update` | POST | Trigger model retraining |

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reload-model` | POST | Reload active model |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model-info

# Make prediction
curl -X POST -F "file=@test.wav" http://localhost:8000/predict

# Submit feedback
curl -X POST \
  -F "true_label=1" \
  -F "predicted_label=0" \
  -F "confidence=0.85" \
  -F "notes=Incorrect prediction" \
  http://localhost:8000/feedback

# Get feedback stats
curl http://localhost:8000/feedback-stats

# Trigger update
curl -X POST http://localhost:8000/trigger-update
```

---

## Troubleshooting

### Issue: "No active model found"

**Cause:** Model hasn't been trained or selected yet.

**Solution:**
```bash
# Train and select models locally
FEATURE_SET=combined python train.py
FEATURE_SET=mfcc_only python train.py
FEATURE_SET=zcr_others python train.py
python model_selector.py

# Or trigger GitHub Actions workflow
git commit --allow-empty -m "Trigger training"
git push origin main
```

### Issue: "Artifacts not found" in deploy workflow

**Cause:** Training workflows haven't completed or artifacts expired.

**Solution:**
1. Use the unified pipeline workflow instead
2. Or trigger training workflows first, then deploy
3. Or use `use_releases: true` option

### Issue: MLflow connection refused

**Cause:** MLflow server not running.

**Solution:**
```bash
# Start MLflow
docker-compose up -d mlflow

# Or use file-based tracking
export MLFLOW_TRACKING_URI="file:///tmp/mlflow"
```

### Issue: Docker build fails

**Cause:** Missing dependencies or incorrect Dockerfile.

**Solution:**
```bash
# Rebuild with no cache
docker-compose build --no-cache

# Check Dockerfile syntax
docker build -t vad-test .
```

### Issue: Feedback not being collected

**Cause:** Feedback file not mounted or permissions issue.

**Solution:**
```bash
# Create feedback file with correct permissions
touch feedback_data.csv
chmod 666 feedback_data.csv

# In docker-compose.yml, ensure volume is mounted:
# - ./feedback_data.csv:/app/feedback_data.csv
```

---

## Best Practices

1. **Always use the unified pipeline** for complete training and deployment
2. **Monitor MLflow** for model performance trends
3. **Collect feedback** regularly to improve model accuracy
4. **Archive old models** before deploying new ones
5. **Test the API** after each deployment using `/health` and `/predict`
6. **Set up alerts** for workflow failures in GitHub Actions

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Check MLflow experiment logs
4. Open an issue in the repository

---

## Changelog

### v2.1.0 (Current)
- Fixed deploy workflow artifact download issue
- Added unified pipeline workflow
- Added auto-update mechanism
- Enhanced MLflow integration
- Added feedback collection API
- Improved model archiving

### v2.0.0 (Previous)
- Initial MLOps pipeline
- Basic GitHub Actions workflows
- MLflow tracking
- Docker deployment
