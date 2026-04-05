"""
MLflow Training Pipeline — VAD (Voice Activity Detection)
Trains 3 model branches, logs metrics, selects best model, promotes to active.
"""

import os, json, shutil
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_CSV    = ROOT / "balanced_vad_dataset.csv"
MODELS_DIR  = ROOT / "models"
ACTIVE_DIR  = MODELS_DIR / "active"
ARCHIVE_DIR = MODELS_DIR / "archive"
BRANCHES_DIR= MODELS_DIR / "branches"
MLFLOW_URI  = f"sqlite:///{ROOT / 'mlflow.db'}"

for d in [ACTIVE_DIR, ARCHIVE_DIR, BRANCHES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Feature sets per branch ───────────────────────────────────────────────
BRANCH_FEATURES = {
    "mfcc_only": [f"mfcc_{i}" for i in range(1, 14)],
    "zcr_others": ["energy", "zcr", "spectral_centroid"],
    "combined":   [f"mfcc_{i}" for i in range(1, 14)] + ["energy", "zcr", "spectral_centroid"],
}

# ── Helper: evaluate ──────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0,
    }
    cm_dict = {
        "confusion_matrix": cm.tolist(),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
    }
    return metrics, cm_dict

# ── Train one branch ──────────────────────────────────────────────────────
def train_branch(branch_name: str, df: pd.DataFrame) -> dict:
    feature_cols = BRANCH_FEATURES[branch_name]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train_s, y_train)

    metrics, cm_dict = evaluate(model, X_test_s, y_test)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="f1")

    # ── MLflow run ──
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(f"VAD-{branch_name}")

    with mlflow.start_run(run_name=branch_name) as run:
        mlflow.log_params({"branch": branch_name, "feature_set": branch_name,
                           "model_type": "SVC", "n_features": len(feature_cols)})
        mlflow.log_metrics({**metrics,
                             "cv_f1_mean": float(cv_scores.mean()),
                             "cv_f1_std":  float(cv_scores.std())})
        # Save artefacts
        branch_dir = BRANCHES_DIR / branch_name
        branch_dir.mkdir(exist_ok=True)
        model_path  = branch_dir / "model.pkl"
        scaler_path = branch_dir / "scaler.pkl"
        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        run_id = run.info.run_id

    meta = {
        "feature_set":    branch_name,
        "feature_names":  feature_cols,
        "model_type":     "SVC",
        "metrics":        metrics,
        "cv_f1_mean":     float(cv_scores.mean()),
        "cv_f1_std":      float(cv_scores.std()),
        "confusion_matrix": cm_dict,
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
        "timestamp":      datetime.now().isoformat(),
        "mlflow_run_id":  run_id,
        "mlflow_experiment": f"VAD-{branch_name}",
        "active_branch":  branch_name,
    }
    with open(branch_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{branch_name}] F1={metrics['f1']:.4f}  Accuracy={metrics['accuracy']:.4f}  "
          f"ROC-AUC={metrics['roc_auc']:.4f}")
    return meta

# ── Select best & promote ─────────────────────────────────────────────────
def promote_best(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["metrics"]["f1"], reverse=True)
    best   = ranked[0]
    branch = best["active_branch"]

    # Archive current active model
    if ACTIVE_DIR.exists():
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch  = ARCHIVE_DIR / f"model_{ts}"
        shutil.copytree(ACTIVE_DIR, arch)
        best["archive_location"] = str(arch.relative_to(ROOT))

    # Promote best branch → active
    src_dir = BRANCHES_DIR / branch
    for fname in ("model.pkl", "scaler.pkl"):
        shutil.copy2(src_dir / fname, ACTIVE_DIR / fname)
    # Also copy to repo root for backward compatibility with existing API
    shutil.copy2(src_dir / "model.pkl",  ROOT / "model.pkl")
    shutil.copy2(src_dir / "scaler.pkl", ROOT / "scaler.pkl")

    # Build rich meta
    best["selection_timestamp"] = datetime.now().isoformat()
    best["all_results"] = [
        {"rank": i+1, "branch": r["active_branch"], "f1": r["metrics"]["f1"],
         "model_type": r["model_type"], "accuracy": r["metrics"]["accuracy"],
         "roc_auc": r["metrics"]["roc_auc"]}
        for i, r in enumerate(ranked)
    ]
    best["comparison_report"] = {
        "timestamp": datetime.now().isoformat(),
        "total_models_evaluated": len(results),
        "models": [
            {"rank": i+1, "branch": r["active_branch"], "model_type": r["model_type"],
             "feature_set": r["feature_set"], "metrics": r["metrics"],
             "feature_names": r["feature_names"]}
            for i, r in enumerate(ranked)
        ]
    }
    with open(ACTIVE_DIR / "meta.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n✅  Best model: [{branch}] promoted to active "
          f"(F1={best['metrics']['f1']:.4f})")
    return best

# ── Entry point ───────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("VAD MLflow Training Pipeline")
    print("=" * 60)
    df = pd.read_csv(DATA_CSV)
    print(f"Dataset loaded: {len(df)} rows, {df['label'].value_counts().to_dict()}")

    results = []
    for branch in BRANCH_FEATURES:
        print(f"\nTraining branch: {branch}")
        meta = train_branch(branch, df)
        results.append(meta)

    best = promote_best(results)
    print("\n📊  All branch results:")
    for r in best["all_results"]:
        marker = "★" if r["branch"] == best["active_branch"] else " "
        print(f"  {marker} #{r['rank']} {r['branch']:<12}  F1={r['f1']:.4f}  "
              f"Acc={r['accuracy']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
    print("=" * 60)
    return best

if __name__ == "__main__":
    run_pipeline()
