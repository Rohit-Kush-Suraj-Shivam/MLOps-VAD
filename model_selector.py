"""
model_selector.py  –  Reads meta JSON files from all 3 branch models,
picks the best by F1, and writes models/active_model/ for the API to load.

Run after all 3 branches have trained:
    python model_selector.py
"""

import os
import json
import shutil
import joblib
import mlflow

MODELS_DIR  = os.getenv("MODEL_OUT_DIR", "models")
ACTIVE_DIR  = os.path.join(MODELS_DIR, "active")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

os.makedirs(ACTIVE_DIR, exist_ok=True)

BRANCHES = ["combined", "mfcc_only", "zcr_others"]

def load_meta(feature_set: str) -> dict | None:
    path = os.path.join(MODELS_DIR, f"meta_{feature_set}.json")
    if not os.path.exists(path):
        print(f"  ⚠️  No meta file for {feature_set} ({path})")
        return None
    with open(path) as f:
        return json.load(f)

# ── gather all branch results ─────────────────────────────────────────────────
results = []
for branch in BRANCHES:
    meta = load_meta(branch)
    if meta:
        f1 = meta["metrics"].get("f1", 0)
        acc = meta["metrics"].get("accuracy", 0)
        auc = meta["metrics"].get("roc_auc", 0)
        print(f"  {branch:15s}  F1={f1:.4f}  Acc={acc:.4f}  AUC={auc:.4f}  model={meta['model_type']}")
        results.append((f1, branch, meta))

if not results:
    raise RuntimeError("No trained branch models found! Run train.py on each branch first.")

results.sort(key=lambda x: x[0], reverse=True)
best_f1, best_branch, best_meta = results[0]

print(f"\n🏆 Winner: {best_branch}  (F1={best_f1:.4f}, model={best_meta['model_type']})")

# ── copy winner to active/ ────────────────────────────────────────────────────
src_model  = os.path.join(MODELS_DIR, f"model_{best_branch}.pkl")
src_scaler = os.path.join(MODELS_DIR, f"scaler_{best_branch}.pkl")

shutil.copy(src_model,  os.path.join(ACTIVE_DIR, "model.pkl"))
shutil.copy(src_scaler, os.path.join(ACTIVE_DIR, "scaler.pkl"))

# Write active meta
active_meta = {**best_meta, "active_branch": best_branch, "all_results": [
    {"branch": b, "f1": f, "model_type": m["model_type"], "accuracy": m["metrics"].get("accuracy")}
    for f, b, m in results
]}
with open(os.path.join(ACTIVE_DIR, "meta.json"), "w") as f:
    json.dump(active_meta, f, indent=2)

print(f"✅ Active model updated → {ACTIVE_DIR}/")
print(f"   Feature set : {best_meta['feature_set']}")
print(f"   Features    : {best_meta['feature_names']}")
print(f"   Model type  : {best_meta['model_type']}")

# ── log selection to MLflow ───────────────────────────────────────────────────
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("VAD-ModelSelection")
    with mlflow.start_run(run_name="model_selection"):
        mlflow.log_param("winner_branch",  best_branch)
        mlflow.log_param("winner_model",   best_meta["model_type"])
        mlflow.log_param("winner_features", ",".join(best_meta["feature_names"]))
        mlflow.log_metric("winner_f1",       best_f1)
        mlflow.log_metric("winner_accuracy", best_meta["metrics"].get("accuracy", 0))
        mlflow.log_metric("winner_roc_auc",  best_meta["metrics"].get("roc_auc", 0))
        for rank, (f1, branch, meta) in enumerate(results):
            mlflow.log_metric(f"rank{rank+1}_f1", f1)
            mlflow.log_param(f"rank{rank+1}_branch", branch)
        mlflow.log_artifact(os.path.join(ACTIVE_DIR, "meta.json"))
        mlflow.set_tag("stage", "model_selection")
    print("📊 Selection logged to MLflow.")
except Exception as e:
    print(f"⚠️  MLflow logging skipped: {e}")
