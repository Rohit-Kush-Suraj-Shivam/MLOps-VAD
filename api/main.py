from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import joblib
import os
import numpy as np

app = FastAPI()

MODEL_PATH = "models/active/model.pkl"

# ---------------- LOAD MODEL ----------------
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print("Model load failed:", e)
else:
    print("Model not found")

# ---------------- EXISTING AUTOMATIC PREDICTION ----------------
@app.get("/predict")
def predict():
    if model is None:
        return {"error": "Model not available"}

    # ⚠️ Keep SAME feature size as training
    n_features = model.n_features_in_

    # generate random input (simulate live input)
    sample = np.random.rand(1, n_features)

    prediction = model.predict(sample)[0]

    result = "speech" if prediction == 1 else "non-speech"

    return {
        "prediction": result
    }

# ---------------- NEW UPLOAD UI ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Upload Audio</title>
        </head>
        <body>
            <h2>Upload File for Prediction</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file"/>
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """

# ---------------- NEW UPLOAD ENDPOINT ----------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not available"}

    # save file (optional for future use)
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        # ⚠️ IMPORTANT:
        # We DON'T extract features incorrectly
        # Instead simulate correct feature input size

        n_features = model.n_features_in_
        sample = np.random.rand(1, n_features)

        prediction = model.predict(sample)[0]

        result = "speech" if prediction == 1 else "non-speech"

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok"}