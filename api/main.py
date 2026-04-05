"""
VAD FastAPI — with MLflow-backed auto model reload.
The active model is determined by models/active/meta.json.
A background scheduler checks for updates every 60 s (configurable).
"""

import os, json, time, threading, io
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
ACTIVE_DIR = ROOT / "models" / "active"
META_PATH  = ACTIVE_DIR / "meta.json"

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

def extract_features(audio: np.ndarray, sr: int, feature_names: list) -> np.ndarray:
    feats = {}
    if LIBROSA_AVAILABLE:
        mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=13)
        for i, m in enumerate(np.mean(mfcc, axis=1)):
            feats[f"mfcc_{i+1}"] = m
        feats["energy"] = float(np.mean(librosa.feature.rms(y=audio)))
        feats["zcr"]    = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        feats["spectral_centroid"] = float(
            np.mean(librosa.feature.spectral_centroid(y=audio.astype(float), sr=sr)))
    return np.array([feats[n] for n in feature_names]).reshape(1, -1)


class ModelManager:
    def __init__(self):
        self._lock     = threading.Lock()
        self.model     = None
        self.scaler    = None
        self.meta      = {}
        self._last_ts  = None
        self._poll_sec = int(os.getenv("MODEL_POLL_SEC", "60"))
        self._load()
        self._start_watcher()

    def _load(self):
        try:
            model_path  = ACTIVE_DIR / "model.pkl"
            scaler_path = ACTIVE_DIR / "scaler.pkl"
            if not model_path.exists():
                model_path  = ROOT / "model.pkl"
                scaler_path = ROOT / "scaler.pkl"
            mtime = model_path.stat().st_mtime
            if mtime == self._last_ts:
                return
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            meta   = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            with self._lock:
                self.model = model; self.scaler = scaler
                self.meta  = meta;  self._last_ts = mtime
            print(f"[ModelManager] Loaded: branch={meta.get('active_branch','?')} "
                  f"F1={meta.get('metrics',{}).get('f1','?')}")
        except Exception as e:
            print(f"[ModelManager] ERROR: {e}")

    def _start_watcher(self):
        def _w():
            while True:
                time.sleep(self._poll_sec)
                self._load()
        threading.Thread(target=_w, daemon=True).start()

    @property
    def feature_names(self):
        return self.meta.get("feature_names",
               [f"mfcc_{i}" for i in range(1,14)] + ["energy","zcr","spectral_centroid"])

    def predict(self, audio, sr):
        with self._lock:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            feats = extract_features(audio, sr, self.feature_names)
            if self.scaler is not None:
                feats = self.scaler.transform(feats)
            prob = self.model.predict_proba(feats)[0][1]
        pred = "Speech" if prob >= 0.5 else "Noise"
        return {"prediction": pred, "speech_probability": round(float(prob), 4)}


manager = ModelManager()
app = FastAPI(title="VAD API - MLOps Edition", version="2.0.0")

@app.get("/")
def root():
    m = manager.meta
    return {"status": "ok", "active_branch": m.get("active_branch"),
            "model_type": m.get("model_type"), "metrics": m.get("metrics"),
            "last_updated": m.get("selection_timestamp")}

@app.get("/model/info")
def model_info():
    return manager.meta

@app.post("/detect/file")
async def detect_file(file: UploadFile = File(...)):
    if not LIBROSA_AVAILABLE:
        raise HTTPException(503, "librosa not installed")
    import librosa as _lb
    data = await file.read()
    try:
        audio, sr = _lb.load(io.BytesIO(data), sr=None, mono=True)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")
    result = manager.predict(audio, sr)
    return {**result, "energy": round(float(np.mean(audio**2)),6),
            "variation": round(float(np.std(audio)),6),
            "sample_rate": sr, "duration_sec": round(len(audio)/sr,2)}

@app.get("/model/reload")
def force_reload():
    manager._last_ts = None
    manager._load()
    return {"status": "reloaded", "branch": manager.meta.get("active_branch")}

@app.get("/health")
def health():
    return {"healthy": manager.model is not None,
            "branch": manager.meta.get("active_branch")}
