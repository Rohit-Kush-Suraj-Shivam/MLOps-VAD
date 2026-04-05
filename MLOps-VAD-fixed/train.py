"""
train.py - MLflow-tracked training script with enhanced monitoring.

Reads FEATURE_SET env var to select the feature branch.
Run:
    FEATURE_SET=combined       python train.py
    FEATURE_SET=mfcc_only      python train.py
    FEATURE_SET=zcr_others     python train.py

Features:
- Automatic model selection (Logistic Regression vs SVM)
- MLflow experiment tracking
- Cross-validation
- Model artifact generation
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# Configuration
FEATURE_SET = os.getenv("FEATURE_SET", "combined")  # combined | mfcc_only | zcr_others
DATASET_PATH = os.getenv("DATASET_PATH", "balanced_vad_dataset.csv")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_OUT_DIR = os.getenv("MODEL_OUT_DIR", "models")
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature column definitions
FEATURE_COLUMNS = {
    "combined": [f"mfcc_{i}" for i in range(1, 14)] + ["energy", "zcr", "spectral_centroid"],
    "mfcc_only": [f"mfcc_{i}" for i in range(1, 14)],
    "zcr_others": ["energy", "zcr", "spectral_centroid"],
}

os.makedirs(MODEL_OUT_DIR, exist_ok=True)


def get_best_model(X_train, y_train, X_val, y_val):
    """
    Train and compare Logistic Regression and SVM, return the best model.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "SVC": SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
    }
    
    best_f1 = 0
    best_model = None
    best_model_name = None
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Validate
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred)
        
        results[name] = {
            "f1": val_f1,
            "model": model
        }
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_model_name = name
    
    return best_model, best_model_name, results


def log_confusion_matrix(y_true, y_pred, run_id):
    """Log confusion matrix as artifact."""
    cm = confusion_matrix(y_true, y_pred)
    cm_dict = {
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }
    
    # Save to temp file and log
    cm_path = os.path.join(MODEL_OUT_DIR, f"confusion_matrix_{FEATURE_SET}.json")
    with open(cm_path, "w") as f:
        json.dump(cm_dict, f, indent=2)
    mlflow.log_artifact(cm_path)
    
    return cm_dict


def main():
    print("=" * 70)
    print(f"🚀 Training VAD Model - Feature Set: {FEATURE_SET}")
    print("=" * 70)
    
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment_name = f"VAD-{FEATURE_SET}"
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow Experiment: {experiment_name}")
    print(f"MLflow Tracking URI: {MLFLOW_URI}")
    print()
    
    # Load data
    print(f"📊 Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    cols = FEATURE_COLUMNS[FEATURE_SET]
    available = [c for c in cols if c in df.columns]
    
    if len(available) < len(cols):
        missing = set(cols) - set(available)
        print(f"⚠️  Missing columns: {missing}")
    
    print(f"Using {len(available)} features: {available}")
    print(f"Dataset shape: {df.shape}")
    
    X = df[available].values
    y = df["label"].values
    
    # Class distribution
    class_dist = pd.Series(y).value_counts().to_dict()
    print(f"Class distribution: {class_dist}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{FEATURE_SET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("feature_set", FEATURE_SET)
        mlflow.log_param("features", ",".join(available))
        mlflow.log_param("feature_count", len(available))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        
        # Train and select best model
        print("🎯 Training models...")
        model, model_type, model_results = get_best_model(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        print(f"Selected model: {model_type}")
        print(f"Model comparison:")
        for name, result in model_results.items():
            print(f"  {name}: F1={result['f1']:.4f}")
        
        mlflow.log_param("model_type", model_type)
        
        # Cross-validation
        print(f"\n📈 Running {CV_FOLDS}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        
        print(f"CV F1 scores: {cv_scores}")
        print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_f1_fold_{i+1}", score)
        
        # Final training on train+val
        print("\n🔁 Final training on combined train+validation set...")
        X_train_full = np.vstack([X_train_scaled, X_val_scaled])
        y_train_full = np.concatenate([y_train, y_val])
        model.fit(X_train_full, y_train_full)
        
        # Test evaluation
        print("\n🧪 Evaluating on test set...")
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        print("\n📊 Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            mlflow.log_metric(f"test_{metric}", value)
        
        # Log confusion matrix
        cm_dict = log_confusion_matrix(y_test, y_pred, mlflow.active_run().info.run_id)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(MODEL_OUT_DIR, f"classification_report_{FEATURE_SET}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model", registered_model_name=f"VAD-{FEATURE_SET}")
        
        # Save artifacts locally
        model_path = os.path.join(MODEL_OUT_DIR, f"model_{FEATURE_SET}.pkl")
        scaler_path = os.path.join(MODEL_OUT_DIR, f"scaler_{FEATURE_SET}.pkl")
        meta_path = os.path.join(MODEL_OUT_DIR, f"meta_{FEATURE_SET}.json")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        meta = {
            "feature_set": FEATURE_SET,
            "feature_names": available,
            "model_type": model_type,
            "metrics": metrics,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "confusion_matrix": cm_dict,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "timestamp": datetime.now().isoformat(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "mlflow_experiment": experiment_name
        }
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(meta_path)
        
        # Set tags
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("feature_set", FEATURE_SET)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("status", "completed")
        
        print()
        print("=" * 70)
        print("✅ Training complete!")
        print(f"Model saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")
        print(f"Meta saved: {meta_path}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 70)
        
        return meta


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
