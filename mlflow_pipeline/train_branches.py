import os
import shutil
import tempfile
import mlflow
import mlflow.sklearn
import requests
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# ---------------- SAFE MLFLOW SETUP ----------------
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    # ALWAYS safe in GitHub Actions
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f"New Model F1: {f1}")

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
print(f"Old Model F1: {old_f1}")

# ---------------- LOG TO MLFLOW ----------------
with mlflow.start_run() as run:
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        model,
        name="model",   # fixed deprecation warning
        input_example=X_train.iloc[:1]
    )

    run_id = run.info.run_id

# ---------------- COMPARE MODELS ----------------
if f1 > old_f1:
    print("New model is better → promoting...")

    # archive old model
    if ACTIVE_DIR.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        arch_path = ARCHIVE_DIR / f"model_{old_f1:.4f}"
        shutil.copytree(ACTIVE_DIR, arch_path, dirs_exist_ok=True)

    # save new model
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ACTIVE_DIR / "model.pkl")

    # register model in MLflow
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print("Model promoted to Production!")

    # ---------------- TRIGGER GITHUB WORKFLOW ----------------
    if GITHUB_TOKEN:
        print("Triggering GitHub workflow...")

        url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"

        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        data = {"ref": "main"}

        try:
            response = requests.post(url, headers=headers, json=data)
            print("GitHub Trigger Status:", response.status_code)
        except Exception as e:
            print("GitHub trigger failed:", str(e))

else:
    print("New model NOT better → skipping deployment")