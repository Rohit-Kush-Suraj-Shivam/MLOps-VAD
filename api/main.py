from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import joblib
import os
import numpy as np

app = FastAPI()

MODEL_PATH = "models/active/model.pkl"

model = None

# ---------------- SAFE MODEL LOAD ----------------
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded")
    else:
        print("Model NOT found")
except Exception as e:
    print("Model load failed:", e)

# ---------------- ROOT UI ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Upload File</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file"/>
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """

# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- AUTO PREDICT ----------------
@app.get("/predict")
def predict():
    try:
        if model is None:
            return {"error": "Model not loaded"}

        n_features = model.n_features_in_
        sample = np.random.rand(1, n_features)

        pred = model.predict(sample)[0]
        result = "speech" if pred == 1 else "non-speech"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}

# ---------------- FILE UPLOAD ----------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if model is None:
            return {"error": "Model not loaded"}

        n_features = model.n_features_in_
        sample = np.random.rand(1, n_features)

        pred = model.predict(sample)[0]
        result = "speech" if pred == 1 else "non-speech"

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception as e:
        return {"error": str(e)}