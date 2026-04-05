"""
model_selector.py - Reads meta JSON files from all 3 branch models,
picks the best by F1, and writes models/active/ for the API to load.

Features:
- Automatic model comparison and selection
- MLflow experiment tracking
- Model versioning and metadata storage
- Performance comparison reporting

Run after all 3 branches have trained:
    python model_selector.py
"""
import os
import json
import shutil
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path

# Configuration
MODELS_DIR = os.getenv("MODEL_OUT_DIR", "models")
ACTIVE_DIR = os.path.join(MODELS_DIR, "active")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlflow")
EXPERIMENT_NAME = "VAD-ModelSelection"

# Ensure directories exist
os.makedirs(ACTIVE_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "archive"), exist_ok=True)

# Branch configurations
BRANCHES = ["combined", "mfcc_only", "zcr_others"]


def load_meta(feature_set: str) -> dict | None:
    """Load metadata JSON for a specific feature set."""
    path = os.path.join(MODELS_DIR, f"meta_{feature_set}.json")
    if not os.path.exists(path):
        print(f"  ⚠️  No meta file for {feature_set} ({path})")
        return None
    with open(path) as f:
        return json.load(f)


def archive_current_model():
    """Archive the current active model before replacing it."""
    active_model = os.path.join(ACTIVE_DIR, "model.pkl")
    if os.path.exists(active_model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(MODELS_DIR, "archive", f"model_{timestamp}")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Copy current active model to archive
        for file in ["model.pkl", "scaler.pkl", "meta.json"]:
            src = os.path.join(ACTIVE_DIR, file)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(archive_dir, file))
        
        print(f"📦 Archived current model to {archive_dir}")
        return archive_dir
    return None


def compare_models(results: list) -> dict:
    """Generate detailed comparison report of all models."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "total_models_evaluated": len(results),
        "models": []
    }
    
    for rank, (f1, branch, meta) in enumerate(results, 1):
        model_info = {
            "rank": rank,
            "branch": branch,
            "model_type": meta["model_type"],
            "feature_set": meta["feature_set"],
            "metrics": {
                "f1": f1,
                "accuracy": meta["metrics"].get("accuracy", 0),
                "precision": meta["metrics"].get("precision", 0),
                "recall": meta["metrics"].get("recall", 0),
                "roc_auc": meta["metrics"].get("roc_auc", 0)
            },
            "feature_names": meta.get("feature_names", [])
        }
        comparison["models"].append(model_info)
    
    return comparison


def log_to_mlflow(best_f1, best_branch, best_meta, results, comparison):
    """Log model selection results to MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"model_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("winner_branch", best_branch)
            mlflow.log_param("winner_model", best_meta["model_type"])
            mlflow.log_param("winner_features", ",".join(best_meta["feature_names"]))
            mlflow.log_param("feature_count", len(best_meta["feature_names"]))
            mlflow.log_param("total_models_compared", len(results))
            
            # Log winner metrics
            mlflow.log_metric("winner_f1", best_f1)
            mlflow.log_metric("winner_accuracy", best_meta["metrics"].get("accuracy", 0))
            mlflow.log_metric("winner_precision", best_meta["metrics"].get("precision", 0))
            mlflow.log_metric("winner_recall", best_meta["metrics"].get("recall", 0))
            mlflow.log_metric("winner_roc_auc", best_meta["metrics"].get("roc_auc", 0))
            
            # Log all model rankings
            for rank, (f1, branch, meta) in enumerate(results, 1):
                mlflow.log_metric(f"rank{rank}_f1", f1)
                mlflow.log_metric(f"rank{rank}_accuracy", meta["metrics"].get("accuracy", 0))
                mlflow.log_param(f"rank{rank}_branch", branch)
                mlflow.log_param(f"rank{rank}_model_type", meta["model_type"])
            
            # Log artifacts
            mlflow.log_artifact(os.path.join(ACTIVE_DIR, "meta.json"))
            mlflow.log_artifact(os.path.join(ACTIVE_DIR, "model.pkl"))
            
            # Log comparison report
            comparison_path = os.path.join(MODELS_DIR, "comparison_report.json")
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)
            mlflow.log_artifact(comparison_path)
            
            # Set tags
            mlflow.set_tag("stage", "model_selection")
            mlflow.set_tag("winner_branch", best_branch)
            mlflow.set_tag("winner_model_type", best_meta["model_type"])
            mlflow.set_tag("selection_timestamp", datetime.now().isoformat())
            
            print("📊 Selection logged to MLflow.")
            print(f"   Run ID: {mlflow.active_run().info.run_id}")
            
    except Exception as e:
        print(f"⚠️  MLflow logging skipped: {e}")


def main():
    """Main model selection logic."""
    print("=" * 70)
    print("🔍 VAD Model Selector")
    print("=" * 70)
    print(f"Models directory: {MODELS_DIR}")
    print(f"Active directory: {ACTIVE_DIR}")
    print(f"MLflow URI: {MLFLOW_URI}")
    print()
    
    # Gather all branch results
    print("📊 Evaluating models from all branches...")
    results = []
    
    for branch in BRANCHES:
        meta = load_meta(branch)
        if meta:
            metrics = meta.get("metrics", {})
            f1 = metrics.get("f1", 0)
            acc = metrics.get("accuracy", 0)
            auc = metrics.get("roc_auc", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            
            print(f"  {branch:15s}  F1={f1:.4f}  Acc={acc:.4f}  AUC={auc:.4f}  "
                  f"Precision={precision:.4f}  Recall={recall:.4f}  "
                  f"model={meta['model_type']}")
            results.append((f1, branch, meta))
    
    if not results:
        raise RuntimeError(
            "❌ No trained branch models found! Run train.py on each branch first.\n"
            "Expected files:\n"
            f"  - {MODELS_DIR}/meta_combined.json\n"
            f"  - {MODELS_DIR}/meta_mfcc_only.json\n"
            f"  - {MODELS_DIR}/meta_zcr_others.json"
        )
    
    # Sort by F1 score (descending)
    results.sort(key=lambda x: x[0], reverse=True)
    best_f1, best_branch, best_meta = results[0]
    
    print()
    print("🏆 Winner Selected:")
    print(f"   Branch:      {best_branch}")
    print(f"   F1 Score:    {best_f1:.4f}")
    print(f"   Model Type:  {best_meta['model_type']}")
    print(f"   Features:    {', '.join(best_meta['feature_names'])}")
    
    # Archive current model before replacing
    archive_dir = archive_current_model()
    
    # Copy winner to active/
    src_model = os.path.join(MODELS_DIR, f"model_{best_branch}.pkl")
    src_scaler = os.path.join(MODELS_DIR, f"scaler_{best_branch}.pkl")
    
    if not os.path.exists(src_model):
        raise RuntimeError(f"❌ Model file not found: {src_model}")
    if not os.path.exists(src_scaler):
        raise RuntimeError(f"❌ Scaler file not found: {src_scaler}")
    
    shutil.copy(src_model, os.path.join(ACTIVE_DIR, "model.pkl"))
    shutil.copy(src_scaler, os.path.join(ACTIVE_DIR, "scaler.pkl"))
    
    # Generate comparison report
    comparison = compare_models(results)
    
    # Write active meta with additional info
    active_meta = {
        **best_meta,
        "active_branch": best_branch,
        "selection_timestamp": datetime.now().isoformat(),
        "archive_location": archive_dir,
        "all_results": [
            {
                "rank": rank,
                "branch": b,
                "f1": f1,
                "model_type": m["model_type"],
                "accuracy": m["metrics"].get("accuracy"),
                "roc_auc": m["metrics"].get("roc_auc")
            }
            for rank, (f1, b, m) in enumerate(results, 1)
        ],
        "comparison_report": comparison
    }
    
    with open(os.path.join(ACTIVE_DIR, "meta.json"), "w") as f:
        json.dump(active_meta, f, indent=2)
    
    print()
    print("✅ Active model updated:")
    print(f"   Location:    {ACTIVE_DIR}/")
    print(f"   Feature Set: {best_meta['feature_set']}")
    print(f"   Features:    {len(best_meta['feature_names'])} features")
    print(f"   Model Type:  {best_meta['model_type']}")
    
    # Log to MLflow
    log_to_mlflow(best_f1, best_branch, best_meta, results, comparison)
    
    print()
    print("=" * 70)
    print("✨ Model selection complete!")
    print("=" * 70)
    
    return active_meta


if __name__ == "__main__":
    try:
        result = main()
        # Exit with success code
        exit(0)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
