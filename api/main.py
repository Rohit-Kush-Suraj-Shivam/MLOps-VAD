"""
FastAPI application for Voice Activity Detection.
- Extracts real audio features (MFCC, energy, ZCR, spectral centroid) from uploads
- Uses the trained scaler + model for prediction
- Logs every prediction as an MLflow run under the "VAD-predictions" experiment
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import json
import os
import tempfile
import numpy as np

app = FastAPI(title="VAD — Voice Activity Detection")

# ─── paths ───────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "models/active/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/active/scaler.pkl")
META_PATH   = os.getenv("META_PATH", "models/active/meta.json")

# ─── load model artefacts ────────────────────────────────────────────────
model  = None
scaler = None
meta   = None

def load_model():
    global model, scaler, meta
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"[API] Model loaded from {MODEL_PATH}")
        else:
            print(f"[API] Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"[API] Model load failed: {e}")
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"[API] Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"[API] Scaler load failed: {e}")
    try:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
            print(f"[API] Meta loaded — branch: {meta.get('active_branch')}")
    except Exception:
        pass

load_model()

# ─── optional MLflow setup ───────────────────────────────────────────────
mlflow = None
try:
    import mlflow as _mlflow
    mlflow = _mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[API] MLflow tracking → {tracking_uri}")
    else:
        mlflow.set_tracking_uri("./mlruns")
        print("[API] MLflow tracking → ./mlruns (local)")
except ImportError:
    print("[API] MLflow not installed — prediction logging disabled")


def extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract the same 16 features used during training."""
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    energy = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    sc = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features = np.hstack([mfcc_means, energy, zcr, sc])
    return features.reshape(1, -1)


def log_prediction_to_mlflow(filename: str, prediction: str, features: np.ndarray):
    if mlflow is None:
        return
    try:
        mlflow.set_experiment("VAD-predictions")
        with mlflow.start_run(run_name=f"predict-{filename}"):
            mlflow.log_param("filename", filename)
            mlflow.log_param("prediction", prediction)
            mlflow.log_param("model_branch", meta.get("active_branch", "unknown") if meta else "unknown")
            feature_names = (
                [f"mfcc_{i}" for i in range(1, 14)]
                + ["energy", "zcr", "spectral_centroid"]
            )
            flat = features.flatten()
            for name, val in zip(feature_names, flat):
                mlflow.log_metric(name, float(val))
    except Exception as e:
        print(f"[API] MLflow log failed: {e}")


# ─── HTML UI ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    branch    = meta.get("active_branch", "–") if meta else "–"
    model_type = meta.get("model_type", "–") if meta else "–"
    n_features = len(meta.get("feature_names", [])) if meta else "–"
    trained    = meta.get("timestamp", "–")[:10] if meta else "–"
    metrics    = meta.get("metrics", {}) if meta else {}
    f1  = f"{metrics.get('f1', 0) * 100:.1f}" if metrics else "–"
    acc = f"{metrics.get('accuracy', 0) * 100:.1f}" if metrics else "–"
    pre = f"{metrics.get('precision', 0) * 100:.1f}" if metrics else "–"
    rec = f"{metrics.get('recall', 0) * 100:.1f}" if metrics else "–"
    auc = f"{metrics.get('roc_auc', 0) * 100:.1f}" if metrics else "–"
    model_active = model is not None

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VAD — Voice Activity Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg-primary:    #0b1120;
    --bg-card:       #111827;
    --bg-surface:    #1a2236;
    --border-dim:    #1e2d45;
    --border-bright: #2a3f5f;
    --text-primary:  #e2e8f0;
    --text-muted:    #64748b;
    --text-label:    #8892a8;
    --accent-green:  #22c55e;
    --accent-coral:  #f0564a;
    --gradient-start:#6366f1;
    --gradient-mid:  #38bdf8;
    --gradient-end:  #34d399;
  }}

  body {{
    font-family: 'DM Sans', system-ui, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    padding: 40px 24px;
  }}

  .container {{
    max-width: 920px;
    margin: 0 auto;
  }}

  /* ── Header ────────────────────────────────── */
  .header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 36px;
  }}
  .header-left {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  .vad-badge {{
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-mid));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 14px; color: #fff;
    letter-spacing: 1px;
    flex-shrink: 0;
  }}
  .header-text h1 {{
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }}
  .header-text .subtitle {{
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 2px;
    font-weight: 500;
  }}
  .status-badge {{
    display: flex; align-items: center; gap: 8px;
    border: 1px solid var(--border-dim);
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-label);
    white-space: nowrap;
  }}
  .status-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {"var(--accent-green)" if model_active else "#ef4444"};
    box-shadow: 0 0 8px {"rgba(34,197,94,.5)" if model_active else "rgba(239,68,68,.5)"};
  }}

  /* ── Divider ───────────────────────────────── */
  .divider {{
    height: 1px;
    background: var(--border-dim);
    margin: 0 0 32px 0;
  }}

  /* ── Metric Cards ──────────────────────────── */
  .metrics-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 20px;
  }}
  .metric-card {{
    border: 1px solid var(--border-dim);
    border-radius: 14px;
    padding: 22px 12px 18px;
    text-align: center;
    background: var(--bg-card);
    transition: border-color .25s, box-shadow .25s;
  }}
  .metric-card:hover {{
    border-color: var(--border-bright);
    box-shadow: 0 0 20px rgba(99,102,241,.08);
  }}
  .metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-mid), var(--gradient-end));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.3;
  }}
  .metric-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-top: 8px;
    font-weight: 600;
  }}

  /* ── Model Info Bar ────────────────────────── */
  .model-bar {{
    background: var(--bg-surface);
    border-radius: 10px;
    padding: 14px 24px;
    display: flex;
    gap: 32px;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 32px;
    flex-wrap: wrap;
  }}
  .model-bar span {{
    color: var(--text-primary);
    font-weight: 600;
  }}

  /* ── Tabs ───────────────────────────────────── */
  .tabs {{
    display: flex;
    gap: 6px;
    margin-bottom: 18px;
  }}
  .tab {{
    padding: 10px 22px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all .2s;
  }}
  .tab.active {{
    background: var(--accent-coral);
    color: #fff;
  }}
  .tab.inactive {{
    background: transparent;
    color: var(--text-muted);
  }}
  .tab.inactive:hover {{
    color: var(--text-primary);
    background: var(--bg-surface);
  }}

  /* ── Drop Zone ─────────────────────────────── */
  .dropzone {{
    border: 2px dashed var(--border-bright);
    border-radius: 16px;
    padding: 56px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .3s, background .3s;
    background: var(--bg-card);
    position: relative;
  }}
  .dropzone:hover, .dropzone.dragover {{
    border-color: var(--gradient-mid);
    background: rgba(56,189,248,.04);
  }}
  .dropzone-icon {{
    font-size: 2.4rem;
    margin-bottom: 14px;
    opacity: .5;
  }}
  .dropzone-title {{
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 6px;
  }}
  .dropzone-sub {{
    font-size: 0.78rem;
    color: var(--text-muted);
  }}
  .dropzone input[type=file] {{
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }}

  /* ── Result Panel ──────────────────────────── */
  .result-panel {{
    margin-top: 24px;
    border: 1px solid var(--border-dim);
    border-radius: 14px;
    overflow: hidden;
    display: none;
  }}
  .result-panel.show {{ display: block; animation: fadeUp .35s ease; }}

  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  .result-header {{
    padding: 18px 24px;
    background: var(--bg-surface);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    color: var(--text-muted);
  }}
  .result-prediction {{
    padding: 28px 24px;
    text-align: center;
  }}
  .result-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }}
  .result-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
  }}
  .result-value.speech {{
    background: linear-gradient(135deg, var(--accent-green), var(--gradient-mid));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .result-value.non-speech {{
    background: linear-gradient(135deg, var(--accent-coral), #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .result-confidence {{
    margin-top: 6px;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
  }}
  .result-file {{
    color: var(--text-primary);
    font-weight: 500;
  }}

  /* ── Loading Spinner ───────────────────────── */
  .spinner {{
    display: none;
    margin: 32px auto;
    width: 36px; height: 36px;
    border: 3px solid var(--border-dim);
    border-top-color: var(--gradient-mid);
    border-radius: 50%;
    animation: spin .7s linear infinite;
  }}
  .spinner.show {{ display: block; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* ── Responsive ────────────────────────────── */
  @media (max-width: 680px) {{
    .metrics-row {{ grid-template-columns: repeat(3, 1fr); }}
    .metric-value {{ font-size: 1.3rem; }}
    .header {{ flex-direction: column; align-items: flex-start; gap: 12px; }}
    .model-bar {{ gap: 16px; }}
  }}
  @media (max-width: 420px) {{
    .metrics-row {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <div class="vad-badge">VAD</div>
      <div class="header-text">
        <h1>Voice Activity Detection</h1>
        <div class="subtitle">MLOps Pipeline &middot; {model_type} &middot; {branch}</div>
      </div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      {"Model active" if model_active else "Model offline"}
    </div>
  </div>

  <div class="divider"></div>

  <!-- Metrics -->
  <div class="metrics-row">
    <div class="metric-card">
      <div class="metric-value">{f1}%</div>
      <div class="metric-label">F1 Score</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{acc}%</div>
      <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{pre}%</div>
      <div class="metric-label">Precision</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{rec}%</div>
      <div class="metric-label">Recall</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{auc}%</div>
      <div class="metric-label">ROC-AUC</div>
    </div>
  </div>

  <!-- Model Info -->
  <div class="model-bar">
    <div>Model <span>{model_type}</span></div>
    <div>Feature set <span>{branch}</span></div>
    <div>Trained <span>{trained}</span></div>
    <div>Features <span>{n_features}</span></div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('upload')">Upload file</button>
    <button class="tab inactive" onclick="showTab('mic')">Record mic</button>
  </div>

  <!-- Upload Zone -->
  <div id="tab-upload">
    <div class="dropzone" id="dropzone">
      <div class="dropzone-icon">&#9835;</div>
      <div class="dropzone-title">Drop audio file here</div>
      <div class="dropzone-sub">or click to browse &middot; WAV, MP3, FLAC, OGG supported</div>
      <input type="file" id="fileInput" accept="audio/*,video/mp4,.wav,.mp3,.flac,.ogg" />
    </div>
  </div>

  <!-- Mic Zone (placeholder) -->
  <div id="tab-mic" style="display:none">
    <div class="dropzone" style="cursor:default; opacity:.5;">
      <div class="dropzone-icon">&#127908;</div>
      <div class="dropzone-title">Microphone recording</div>
      <div class="dropzone-sub">Coming soon &middot; requires browser mic permission</div>
    </div>
  </div>

  <!-- Spinner -->
  <div class="spinner" id="spinner"></div>

  <!-- Result -->
  <div class="result-panel" id="result">
    <div class="result-header">
      <div>File: <span class="result-file" id="res-file">—</span></div>
      <div id="res-confidence"></div>
    </div>
    <div class="result-prediction">
      <div class="result-label">Prediction</div>
      <div class="result-value" id="res-value">—</div>
    </div>
  </div>

</div>

<script>
  /* ── Tabs ── */
  function showTab(t) {{
    document.getElementById('tab-upload').style.display = t === 'upload' ? 'block' : 'none';
    document.getElementById('tab-mic').style.display    = t === 'mic'    ? 'block' : 'none';
    document.querySelectorAll('.tab').forEach((btn, i) => {{
      btn.className = 'tab ' + ((i === 0 && t === 'upload') || (i === 1 && t === 'mic') ? 'active' : 'inactive');
    }});
  }}

  /* ── Drag & Drop ── */
  const dz = document.getElementById('dropzone');
  dz.addEventListener('dragover',  e => {{ e.preventDefault(); dz.classList.add('dragover'); }});
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {{
    e.preventDefault();
    dz.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  }});

  document.getElementById('fileInput').addEventListener('change', e => {{
    if (e.target.files.length) uploadFile(e.target.files[0]);
  }});

  /* ── Upload ── */
  async function uploadFile(file) {{
    const spinner = document.getElementById('spinner');
    const result  = document.getElementById('result');
    result.classList.remove('show');
    spinner.classList.add('show');

    const fd = new FormData();
    fd.append('file', file);

    try {{
      const res  = await fetch('/upload', {{ method: 'POST', body: fd }});
      const data = await res.json();
      spinner.classList.remove('show');

      if (data.error) {{
        alert('Error: ' + data.error);
        return;
      }}

      document.getElementById('res-file').textContent  = data.filename || '—';
      const valEl = document.getElementById('res-value');
      valEl.textContent = data.prediction || '—';
      valEl.className   = 'result-value ' + (data.prediction === 'speech' ? 'speech' : 'non-speech');
      document.getElementById('res-confidence').textContent = data.confidence ? 'Confidence: ' + data.confidence : '';
      result.classList.add('show');
    }} catch (err) {{
      spinner.classList.remove('show');
      alert('Request failed: ' + err);
    }}

    // reset input so same file can be re-uploaded
    document.getElementById('fileInput').value = '';
  }}
</script>
</body>
</html>"""


# ─── health check ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "active_branch": meta.get("active_branch") if meta else None,
    }


# ─── reload model (hot-swap after retraining) ────────────────────────────
@app.post("/reload")
def reload():
    load_model()
    return {"status": "reloaded", "model_loaded": model is not None}


# ─── upload + predict ─────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    try:
        import librosa

        suffix = os.path.splitext(file.filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=22050)
        os.unlink(tmp_path)

        features = extract_features_from_audio(audio, sr)

        if scaler is not None:
            features = scaler.transform(features)

        pred = model.predict(features)[0]
        result = "speech" if pred == 1 else "non-speech"

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = f"{max(proba) * 100:.1f}%"

        log_prediction_to_mlflow(file.filename or "unknown", result, features)

        return {
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence,
        }

    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {e}"}, status_code=500)


# ─── simple GET predict with random features (for quick tests) ────────────
@app.get("/predict")
def predict_random():
    if model is None:
        return {"error": "Model not loaded"}

    n_features = model.n_features_in_
    sample = np.random.rand(1, n_features)
    if scaler is not None:
        sample = scaler.transform(sample)
    pred = model.predict(sample)[0]
    result = "speech" if pred == 1 else "non-speech"
    return {"prediction": result, "note": "random features — use /upload for real predictions"}
