from fastapi import FastAPI
import joblib
import os

app = FastAPI()

MODEL_PATH = "models/active/model.pkl"

model = None

# ---------------- LOAD MODEL SAFELY ----------------
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print("Model load failed:", e)
else:
    print("Model not found, API will still run")

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "VAD API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict():
    if model is None:
        return {"error": "Model not available"}
    
    return {"prediction": "dummy"}  # replace later