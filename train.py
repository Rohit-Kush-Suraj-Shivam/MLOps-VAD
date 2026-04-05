"""
train.py  –  MLflow-tracked training script.
Reads FEATURE_SET env var to select the feature branch.
Run:
    FEATURE_SET=combined       python train.py
    FEATURE_SET=mfcc_only      python train.py
    FEATURE_SET=zcr_others     python train.py
"""

import os
import importlib
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
FEATURE_SET   = os.getenv("FEATURE_SET", "combined")          # combined | mfcc_only | zcr_others
DATASET_PATH  = os.getenv("DATASET_PATH", "balanced_vad_dataset.csv")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_OUT_DIR = os.getenv("MODEL_OUT_DIR", "models")
RANDOM_STATE  = 42

FEATURE_COLUMNS = {
    "combined":  [f"mfcc_{i}" for i in range(1, 14)] + ["energy", "zcr", "spectral_centroid"],
    "mfcc_only": [f"mfcc_{i}" for i in range(1, 14)],
    "zcr_others": ["energy", "zcr", "spectral_centroid"],   # dataset subset
}

os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_name = f"VAD-{FEATURE_SET}"
mlflow.set_experiment(experiment_name)

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)

cols = FEATURE_COLUMNS[FEATURE_SET]
# For zcr_others we may not have rolloff/bandwidth in the CSV – use what's available
available = [c for c in cols if c in df.columns]
print(f"[{FEATURE_SET}] Using {len(available)} features: {available}")

X = df[available].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── train & evaluate two models, pick best ────────────────────────────────────
def evaluate(model, X_tr, y_tr, X_te, y_te, name, feature_set):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall":    recall_score(y_te, y_pred, zero_division=0),
        "f1":        f1_score(y_te, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_te, y_prob)

    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="f1")
    metrics["cv_f1_mean"] = float(cv.mean())
    metrics["cv_f1_std"]  = float(cv.std())

    cm = confusion_matrix(y_te, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel().tolist()

    print(f"  [{name}] acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  auc={metrics.get('roc_auc', 'N/A')}")
    return metrics


candidates = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ("SVC",                SVC(probability=True, random_state=RANDOM_STATE)),
]

best_run_id   = None
best_f1       = -1
best_model    = None
best_scaler   = None
best_name     = None
best_metrics  = None

for model_name, model in candidates:
    with mlflow.start_run(run_name=f"{FEATURE_SET}-{model_name}") as run:
        mlflow.log_param("feature_set",  FEATURE_SET)
        mlflow.log_param("model_type",   model_name)
        mlflow.log_param("n_features",   len(available))
        mlflow.log_param("n_train",      len(X_train))
        mlflow.log_param("n_test",       len(X_test))
        mlflow.log_param("feature_names", ",".join(available))

        metrics = evaluate(model, X_train_s, y_train, X_test_s, y_test, model_name, FEATURE_SET)
        mlflow.log_metrics(metrics)

        # Save artifacts locally too
        model_path  = os.path.join(MODEL_OUT_DIR, f"model_{FEATURE_SET}_{model_name.lower()}.pkl")
        scaler_path = os.path.join(MODEL_OUT_DIR, f"scaler_{FEATURE_SET}.pkl")
        meta_path   = os.path.join(MODEL_OUT_DIR, f"meta_{FEATURE_SET}.json")

        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)

        import json
        meta = {
            "feature_set":   FEATURE_SET,
            "model_type":    model_name,
            "feature_names": available,
            "n_features":    len(available),
            "metrics":       metrics,
            "run_id":        run.info.run_id,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(meta_path)
        mlflow.sklearn.log_model(model, artifact_path="sklearn_model")

        mlflow.set_tag("feature_set", FEATURE_SET)
        mlflow.set_tag("best_model",  "false")

        if metrics["f1"] > best_f1:
            best_f1      = metrics["f1"]
            best_run_id  = run.info.run_id
            best_model   = model
            best_scaler  = scaler
            best_name    = model_name
            best_metrics = metrics

# ── save the winner as the canonical branch model ─────────────────────────────
print(f"\n✅ Best: {best_name}  F1={best_f1:.4f}  (run_id={best_run_id})")

# Overwrite canonical files (these are what model_selector.py reads)
joblib.dump(best_model,  os.path.join(MODEL_OUT_DIR, f"model_{FEATURE_SET}.pkl"))
joblib.dump(best_scaler, os.path.join(MODEL_OUT_DIR, f"scaler_{FEATURE_SET}.pkl"))

import json
final_meta = {
    "feature_set":   FEATURE_SET,
    "model_type":    best_name,
    "feature_names": available,
    "n_features":    len(available),
    "metrics":       best_metrics,
    "run_id":        best_run_id,
}
with open(os.path.join(MODEL_OUT_DIR, f"meta_{FEATURE_SET}.json"), "w") as f:
    json.dump(final_meta, f, indent=2)

# Tag the best run
client = mlflow.tracking.MlflowClient()
client.set_tag(best_run_id, "best_model", "true")
client.set_tag(best_run_id, "branch_winner", FEATURE_SET)

print(f"Models saved to {MODEL_OUT_DIR}/")
print("Done.")
