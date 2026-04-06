import os
import shutil
import tempfile
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import requests

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# ---------------- MLFLOW SETUP ----------------
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    temp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file:{temp_dir}")

# ---------------- CONFIG ----------------
BASE_DIR = os.getcwd()

MODEL_NAME = "vad-model"
EXPERIMENT_NAME = "vad-experiment"

DATA_PATH = os.path.join(BASE_DIR, "balanced_vad_dataset.csv")

ACTIVE_DIR = Path(BASE_DIR) / "models" / "active"
ARCHIVE_DIR = Path(BASE_DIR) / "models" / "archive"

GITHUB_REPO = "Rohit-Kush-Suraj-Shivam/MLOps-VAD"
WORKFLOW_FILE = "mlops_pipeline.yml"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1)
y = df["label"]

# ---------------- FEATURE BRANCHES ----------------
FEATURE_BRANCHES = {
    "mfcc_only": [col for col in X.columns if "mfcc" in col],
    "zcr_only": [col for col in X.columns if "zcr" in col],
    "combined": list(X.columns)
}

results = []

# ---------------- TRAIN EACH BRANCH ----------------
for branch_name, features in FEATURE_BRANCHES.items():
    print(f"\nTraining branch: {branch_name}")

    X_branch = X[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_branch, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"{branch_name} F1: {f1}")

    # log to MLflow
    with mlflow.start_run(run_name=branch_name):
        mlflow.log_param("branch", branch_name)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:1]
        )

        run_id = mlflow.active_run().info.run_id

    results.append({
        "branch": branch_name,
        "model": model,
        "f1": f1,
        "run_id": run_id
    })

# ---------------- SELECT BEST MODEL ----------------
best_model = max(results, key=lambda x: x["f1"])

print(f"\nBest branch: {best_model['branch']}")
print(f"Best F1: {best_model['f1']}")

# ---------------- GET PRODUCTION MODEL SCORE ----------------
def get_production_f1():
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            data = client.get_run(run_id)
            return data.data.metrics.get("f1_score", 0)
    except Exception:
        return 0

old_f1 = get_production_f1()
print(f"Production F1: {old_f1}")

# ---------------- COMPARE WITH PRODUCTION ----------------
if best_model["f1"] > old_f1:
    print("\nNew BEST model is better → promoting...")

    # archive old model
    if ACTIVE_DIR.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            ACTIVE_DIR,
            ARCHIVE_DIR / f"model_{old_f1:.4f}",
            dirs_exist_ok=True
        )

    # save best model
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model["model"], ACTIVE_DIR / "model.pkl")

    # register model
    result = mlflow.register_model(
        f"runs:/{best_model['run_id']}/model",
        MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print("Model promoted to Production!")

    # trigger GitHub workflow
    if GITHUB_TOKEN:
        try:
            requests.post(
                f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches",
                headers={"Authorization": f"token {GITHUB_TOKEN}"},
                json={"ref": "main"}
            )
            print("Triggered GitHub workflow")
        except Exception as e:
            print("GitHub trigger failed:", e)

else:
    print("\nBest model NOT better → skipping deployment")