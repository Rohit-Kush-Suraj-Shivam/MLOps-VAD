"""
api/main.py  –  FastAPI VAD service
  GET  /           → HTML dashboard
  GET  /health     → health check
  GET  /model-info → active model metadata
  POST /predict    → upload audio file, get VAD result
  GET  /metrics    → all branch metrics
"""

import os
import json
import tempfile
import warnings
import numpy as np
import librosa
import joblib
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
ACTIVE_DIR  = Path(os.getenv("ACTIVE_MODEL_DIR", BASE_DIR / "models" / "active"))
MODELS_DIR  = Path(os.getenv("MODEL_OUT_DIR",    BASE_DIR / "models"))

SAMPLE_RATE         = 22050
VARIATION_THRESHOLD = 0.14

# ── load active model ─────────────────────────────────────────────────────────
def load_active():
    model_path  = ACTIVE_DIR / "model.pkl"
    scaler_path = ACTIVE_DIR / "scaler.pkl"
    meta_path   = ACTIVE_DIR / "meta.json"

    if not model_path.exists():
        raise RuntimeError(f"No active model found at {model_path}. Run model_selector.py first.")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta   = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, scaler, meta

model, scaler, active_meta = load_active()
FEATURE_NAMES = active_meta.get("feature_names", [])
FEATURE_SET   = active_meta.get("feature_set", "combined")

# ── import the right extractor dynamically ────────────────────────────────────
import importlib
_feat_mod = importlib.import_module(f"features.{FEATURE_SET}")
extract_features = _feat_mod.extract_features

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="VAD MLOps API",
    description="Voice Activity Detection – automated MLOps pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ───────────────────────────────────────────────────────────────────
def highpass_filter(data, cutoff=100, fs=22050, order=5):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="high", analog=False)
    return filtfilt(b, a, data)


def run_inference(audio: np.ndarray) -> dict:
    """Full inference pipeline on a 1-D float audio array."""
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    audio = highpass_filter(audio)

    energy    = float(np.mean(audio ** 2))
    variation = float(np.std(audio))

    features = extract_features(audio, SAMPLE_RATE)
    features_scaled = scaler.transform(features)

    prob = float(model.predict_proba(features_scaled)[0][1])

    prediction = "Speech" if variation > VARIATION_THRESHOLD else "Noise"

    return {
        "prediction":        prediction,
        "speech_probability": round(prob, 4),
        "energy":            round(energy, 6),
        "variation":         round(variation, 6),
        "model_branch":      FEATURE_SET,
        "model_type":        active_meta.get("model_type", "unknown"),
        "features_used":     FEATURE_NAMES,
    }

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_branch": FEATURE_SET}


@app.get("/model-info")
def model_info():
    return JSONResponse(content=active_meta)


@app.get("/metrics")
def branch_metrics():
    """Return metrics from all trained branches."""
    branches = ["combined", "mfcc_only", "zcr_others"]
    out = {}
    for b in branches:
        p = MODELS_DIR / f"meta_{b}.json"
        if p.exists():
            out[b] = json.loads(p.read_text())
    return JSONResponse(content=out)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a WAV / MP3 / MP4 / FLAC / OGG file and get VAD prediction."""
    allowed = {".wav", ".mp3", ".mp4", ".flac", ".ogg", ".m4a"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50 MB cap
        raise HTTPException(413, "File too large (max 50 MB)")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(422, f"Could not decode audio: {e}")
    finally:
        os.unlink(tmp_path)

    result = run_inference(audio)
    result["filename"] = file.filename
    result["duration_s"] = round(len(audio) / SAMPLE_RATE, 2)
    return JSONResponse(content=result)


# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VAD MLOps Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --success: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Space Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* grid background */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(124,58,237,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,58,237,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container { max-width: 1100px; margin: 0 auto; padding: 2rem; position: relative; z-index: 1; }

  header {
    display: flex; align-items: center; gap: 1.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 2rem; margin-bottom: 3rem;
  }
  .logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
    box-shadow: 0 0 30px rgba(124,58,237,.4);
  }
  h1 { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em; }
  h1 span { color: var(--accent2); }
  .subtitle { color: var(--muted); font-size: .75rem; margin-top: .2rem; }

  .badge {
    margin-left: auto;
    padding: .3rem .8rem;
    border-radius: 999px;
    font-size: .7rem; font-weight: 700;
    background: rgba(16,185,129,.15);
    color: var(--success);
    border: 1px solid rgba(16,185,129,.3);
    display: flex; align-items: center; gap: .4rem;
  }
  .badge::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
  .grid3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-bottom: 1.5rem; }
  @media(max-width:700px) { .grid2,.grid3 { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
  }
  .card-title {
    font-size: .65rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 1rem;
  }

  /* upload zone */
  .upload-zone {
    border: 2px dashed var(--accent);
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    background: rgba(124,58,237,.05);
    margin-bottom: 1rem;
    position: relative;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent2); background: rgba(6,182,212,.08); }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-icon { font-size: 2.5rem; margin-bottom: .8rem; display: block; }
  .upload-label { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
  .upload-sub { color: var(--muted); font-size: .72rem; margin-top: .4rem; }

  .btn {
    width: 100%; padding: .9rem;
    background: linear-gradient(135deg, var(--accent), #5b21b6);
    border: none; border-radius: 10px;
    color: #fff; font-family: 'Space Mono', monospace;
    font-size: .85rem; font-weight: 700;
    cursor: pointer; transition: all .2s;
    letter-spacing: .05em;
  }
  .btn:hover { opacity: .9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,58,237,.4); }
  .btn:disabled { opacity: .4; transform: none; cursor: not-allowed; }

  /* result */
  .result-box {
    border-radius: 14px; padding: 1.5rem;
    border: 1px solid var(--border);
    display: none;
    animation: slideIn .3s ease;
  }
  @keyframes slideIn { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:translateY(0) } }
  .result-box.speech { border-color: rgba(16,185,129,.4); background: rgba(16,185,129,.07); }
  .result-box.noise  { border-color: rgba(245,158,11,.4);  background: rgba(245,158,11,.07); }

  .result-pred {
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
    margin-bottom: .5rem;
  }
  .result-pred.speech { color: var(--success); }
  .result-pred.noise  { color: var(--warn); }

  .prob-bar-wrap { margin: 1rem 0; }
  .prob-bar-label { font-size: .7rem; color: var(--muted); margin-bottom: .4rem; display: flex; justify-content: space-between; }
  .prob-bar-bg { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; }
  .prob-bar-fill { height: 100%; border-radius: 999px; transition: width .6s ease; background: linear-gradient(90deg, var(--accent), var(--accent2)); }

  .result-meta { display: grid; grid-template-columns: repeat(2,1fr); gap: .6rem; margin-top: 1rem; }
  .meta-chip {
    background: rgba(255,255,255,.04); border: 1px solid var(--border);
    border-radius: 8px; padding: .5rem .8rem;
    font-size: .72rem;
  }
  .meta-chip span { display: block; color: var(--muted); font-size: .62rem; margin-bottom: .2rem; }

  /* metric cards */
  .metric-val {
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
    color: var(--accent2);
  }
  .metric-lbl { color: var(--muted); font-size: .7rem; margin-top: .3rem; }

  /* branch table */
  table { width: 100%; border-collapse: collapse; font-size: .78rem; }
  th { color: var(--muted); font-weight: 400; padding: .5rem .7rem; text-align: left; border-bottom: 1px solid var(--border); }
  td { padding: .6rem .7rem; border-bottom: 1px solid rgba(255,255,255,.04); }
  tr.winner td { color: var(--accent2); }
  tr.winner td:first-child::before { content: '🏆 '; }

  /* log */
  .log {
    background: #050508; border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem;
    font-size: .72rem; color: var(--muted);
    max-height: 140px; overflow-y: auto;
    margin-top: 1rem;
  }
  .log p { margin: .15rem 0; }
  .log .info  { color: var(--accent2); }
  .log .ok    { color: var(--success); }
  .log .err   { color: var(--danger); }

  #file-chosen { font-size: .72rem; color: var(--muted); text-align: center; margin-bottom: .8rem; height: 1rem; }

  footer { text-align: center; color: var(--muted); font-size: .68rem; margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo">🎙</div>
    <div>
      <h1>VAD <span>MLOps</span></h1>
      <div class="subtitle">Automated Voice Activity Detection Pipeline</div>
    </div>
    <div class="badge" id="status-badge">LIVE</div>
  </header>

  <!-- metric strip -->
  <div class="grid3" id="metric-strip">
    <div class="card">
      <div class="card-title">Active Branch</div>
      <div class="metric-val" id="m-branch">—</div>
      <div class="metric-lbl" id="m-model">loading…</div>
    </div>
    <div class="card">
      <div class="card-title">F1 Score</div>
      <div class="metric-val" id="m-f1">—</div>
      <div class="metric-lbl">test set</div>
    </div>
    <div class="card">
      <div class="card-title">Accuracy</div>
      <div class="metric-val" id="m-acc">—</div>
      <div class="metric-lbl">test set</div>
    </div>
  </div>

  <div class="grid2">

    <!-- upload card -->
    <div class="card">
      <div class="card-title">Analyze Audio File</div>

      <div class="upload-zone" id="drop-zone">
        <input type="file" id="file-input" accept=".wav,.mp3,.mp4,.flac,.ogg,.m4a">
        <span class="upload-icon">🎵</span>
        <div class="upload-label">Drop file or click to browse</div>
        <div class="upload-sub">WAV · MP3 · MP4 · FLAC · OGG · M4A · max 50 MB</div>
      </div>

      <div id="file-chosen"></div>
      <button class="btn" id="analyze-btn" disabled>Analyze</button>

      <div class="log" id="log"><p class="info">Ready.</p></div>
    </div>

    <!-- result card -->
    <div class="card" style="display:flex;flex-direction:column;gap:1rem;">
      <div class="card-title">Prediction Result</div>

      <div class="result-box" id="result-box">
        <div class="result-pred" id="result-pred">—</div>
        <div style="color:var(--muted);font-size:.75rem;" id="result-file">—</div>

        <div class="prob-bar-wrap">
          <div class="prob-bar-label">
            <span>Speech Probability</span>
            <span id="prob-val">0%</span>
          </div>
          <div class="prob-bar-bg"><div class="prob-bar-fill" id="prob-bar" style="width:0%"></div></div>
        </div>

        <div class="result-meta" id="result-meta"></div>
      </div>

      <div id="placeholder" style="flex:1;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:.8rem;text-align:center;border:1px dashed var(--border);border-radius:12px;padding:2rem;">
        Upload an audio file to see<br>the VAD prediction here.
      </div>
    </div>
  </div>

  <!-- branch comparison -->
  <div class="card">
    <div class="card-title">Branch Model Comparison</div>
    <table id="branch-table">
      <thead><tr>
        <th>Branch</th><th>Model</th><th>Features</th>
        <th>F1</th><th>Accuracy</th><th>AUC</th>
      </tr></thead>
      <tbody id="branch-body">
        <tr><td colspan="6" style="color:var(--muted)">Loading…</td></tr>
      </tbody>
    </table>
  </div>

  <footer>VAD MLOps Pipeline · MLflow · FastAPI · Docker · GitHub Actions</footer>
</div>

<script>
const log = (msg, cls='') => {
  const d = document.getElementById('log');
  const p = document.createElement('p');
  if (cls) p.className = cls;
  p.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  d.appendChild(p);
  d.scrollTop = d.scrollHeight;
};

// Load model info
async function loadModelInfo() {
  try {
    const r = await fetch('/model-info');
    const d = await r.json();
    document.getElementById('m-branch').textContent = (d.feature_set || '—').replace('_', '-');
    document.getElementById('m-model').textContent  = d.model_type || '—';
    const m = d.metrics || {};
    document.getElementById('m-f1').textContent  = m.f1   ? (m.f1*100).toFixed(1)+'%'  : '—';
    document.getElementById('m-acc').textContent = m.accuracy ? (m.accuracy*100).toFixed(1)+'%' : '—';
  } catch(e) { log('Could not load model info', 'err'); }
}

// Load branch metrics
async function loadMetrics() {
  try {
    const r = await fetch('/metrics');
    const d = await r.json();
    const active = document.getElementById('m-branch').textContent.replace('-','_');
    const tbody = document.getElementById('branch-body');
    tbody.innerHTML = '';
    const order = ['combined','mfcc_only','zcr_others'];
    let bestF1 = -1, bestBranch = '';
    order.forEach(b => { if(d[b]) { const f = d[b].metrics?.f1||0; if(f>bestF1){bestF1=f;bestBranch=b;} } });
    order.forEach(b => {
      if (!d[b]) { tbody.innerHTML += `<tr><td>${b}</td><td colspan="5" style="color:var(--muted)">Not trained</td></tr>`; return; }
      const m = d[b].metrics || {};
      const isWinner = b === bestBranch;
      tbody.innerHTML += `<tr class="${isWinner?'winner':''}">
        <td>${b.replace('_','-')}</td>
        <td>${d[b].model_type||'—'}</td>
        <td>${d[b].n_features||'—'}</td>
        <td>${m.f1?(m.f1*100).toFixed(1)+'%':'—'}</td>
        <td>${m.accuracy?(m.accuracy*100).toFixed(1)+'%':'—'}</td>
        <td>${m.roc_auc?(m.roc_auc*100).toFixed(1)+'%':'—'}</td>
      </tr>`;
    });
  } catch(e) {}
}

// File input
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const dropZone   = document.getElementById('drop-zone');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    document.getElementById('file-chosen').textContent = '📎 ' + fileInput.files[0].name;
    analyzeBtn.disabled = false;
  }
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  if (e.dataTransfer.files[0]) {
    fileInput.files = e.dataTransfer.files;
    document.getElementById('file-chosen').textContent = '📎 ' + fileInput.files[0].name;
    analyzeBtn.disabled = false;
  }
});

analyzeBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  log(`Uploading ${file.name} (${(file.size/1024).toFixed(0)} KB)…`, 'info');

  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Server error');

    // show result
    const isS = d.prediction === 'Speech';
    const box = document.getElementById('result-box');
    box.className = 'result-box ' + (isS ? 'speech' : 'noise');
    box.style.display = 'block';
    document.getElementById('placeholder').style.display = 'none';

    document.getElementById('result-pred').className = 'result-pred ' + (isS?'speech':'noise');
    document.getElementById('result-pred').textContent = (isS ? '🟢 ' : '🟡 ') + d.prediction;
    document.getElementById('result-file').textContent = `${d.filename} · ${d.duration_s}s`;

    const pct = Math.round(d.speech_probability * 100);
    document.getElementById('prob-val').textContent = pct + '%';
    document.getElementById('prob-bar').style.width  = pct + '%';

    document.getElementById('result-meta').innerHTML = `
      <div class="meta-chip"><span>Energy</span>${d.energy}</div>
      <div class="meta-chip"><span>Variation</span>${d.variation}</div>
      <div class="meta-chip"><span>Branch</span>${d.model_branch}</div>
      <div class="meta-chip"><span>Model</span>${d.model_type}</div>
    `;
    log(`✅ ${d.prediction}  prob=${pct}%  branch=${d.model_branch}`, 'ok');
  } catch(e) {
    log('Error: ' + e.message, 'err');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
  }
});

loadModelInfo();
loadMetrics();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML
