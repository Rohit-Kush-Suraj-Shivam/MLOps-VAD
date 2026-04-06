from fastapi import FastAPI, UploadFile, File
import joblib
import os
import numpy as np
import librosa

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
    print("Model not found, API will still run")

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Combine features
    features = np.hstack([mfcc, zcr])

    return features.reshape(1, -1)

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "VAD API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not available"}

    # save uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        features = extract_features(file_location)

        prediction = model.predict(features)[0]

        result = "speech" if prediction == 1 else "non-speech"

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # cleanup temp file
        if os.path.exists(file_location):
            os.remove(file_location)