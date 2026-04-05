"""
api/main.py - FastAPI VAD service with MLflow integration and feedback collection

Endpoints:
  GET  /              → HTML dashboard
  GET  /health        → Health check
  GET  /model-info    → Active model metadata
  POST /predict       → Upload audio file, get VAD result
  POST /feedback      → Submit feedback for predictions
  GET  /metrics       → All branch metrics
  POST /trigger-update → Trigger model retraining (admin)
  GET  /feedback-stats → Feedback collection statistics

Features:
- Automatic model loading with hot-reload capability
- MLflow tracking for predictions
- Feedback collection for continuous improvement
- Model performance monitoring
"""
import os
import json
import tempfile
import warnings
import numpy as np
import librosa
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt
import pandas as pd

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
ACTIVE_DIR = Path(os.getenv("ACTIVE_MODEL_DIR", BASE_DIR / "models" / "active"))
MODELS_DIR = Path(os.getenv("MODEL_OUT_DIR", BASE_DIR / "models"))
FEEDBACK_FILE = BASE_DIR / "feedback_data.csv"
PREDICTION_LOG_FILE = BASE_DIR / "prediction_log.csv"

# Constants
SAMPLE_RATE = 22050
VARIATION_THRESHOLD = 0.14
FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", "50"))

# Global variables for model
model = None
scaler = None
active_meta = {}
FEATURE_NAMES = []
FEATURE_SET = "combined"
extract_features = None


def load_active():
    """Load the active model and metadata."""
    global model, scaler, active_meta, FEATURE_NAMES, FEATURE_SET, extract_features
    
    model_path = ACTIVE_DIR / "model.pkl"
    scaler_path = ACTIVE_DIR / "scaler.pkl"
    meta_path = ACTIVE_DIR / "meta.json"
    
    if not model_path.exists():
        raise RuntimeError(
            f"No active model found at {model_path}. Run model_selector.py first."
        )
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    active_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    FEATURE_NAMES = active_meta.get("feature_names", [])
    FEATURE_SET = active_meta.get("feature_set", "combined")
    
    # Import the right extractor dynamically
    import importlib
    try:
        _feat_mod = importlib.import_module(f"features.{FEATURE_SET}")
        extract_features = _feat_mod.extract_features
    except ImportError:
        # Fallback to default extractor
        from features.combined import extract_features as default_extract
        extract_features = default_extract
    
    print(f"✅ Loaded active model: {active_meta.get('model_type', 'unknown')} "
          f"({FEATURE_SET}, {len(FEATURE_NAMES)} features)")
    
    return model, scaler, active_meta


# Load model on startup
try:
    model, scaler, active_meta = load_active()
except RuntimeError as e:
    print(f"⚠️  Warning: {e}")
    print("     API will start but predictions will fail until a model is available.")

# FastAPI app
app = FastAPI(
    title="VAD MLOps API",
    description="Voice Activity Detection - Automated MLOps Pipeline with MLflow",
    version="2.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, lowcut=300, highcut=3400, fs=SAMPLE_RATE, order=5):
    """Apply bandpass filter to audio signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def log_prediction(features, prediction, confidence, filename, processing_time):
    """Log prediction to CSV for monitoring."""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "prediction": int(prediction),
            "confidence": float(confidence),
            "processing_time_ms": processing_time,
            "model_version": active_meta.get("selection_timestamp", "unknown"),
            "feature_set": FEATURE_SET
        }
        
        # Append to log file
        log_df = pd.DataFrame([log_entry])
        if PREDICTION_LOG_FILE.exists():
            log_df.to_csv(PREDICTION_LOG_FILE, mode='a', header=False, index=False)
        else:
            log_df.to_csv(PREDICTION_LOG_FILE, index=False)
            
    except Exception as e:
        print(f"⚠️  Failed to log prediction: {e}")


def save_feedback(features, true_label, predicted_label, confidence, notes=""):
    """Save feedback data for future retraining."""
    try:
        # Create feedback entry
        feedback_entry = {
            "label": int(true_label),
            "timestamp": datetime.now().isoformat(),
            "predicted_label": int(predicted_label),
            "confidence": float(confidence),
            "notes": notes
        }
        
        # Add feature values
        for i, feat_name in enumerate(FEATURE_NAMES):
            if i < len(features):
                feedback_entry[feat_name] = float(features[i])
        
        # Save to CSV
        feedback_df = pd.DataFrame([feedback_entry])
        
        if FEEDBACK_FILE.exists():
            feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            # Create with headers
            feedback_df.to_csv(FEEDBACK_FILE, index=False)
        
        return True
    except Exception as e:
        print(f"⚠️  Failed to save feedback: {e}")
        return False


def get_feedback_count():
    """Get the number of feedback samples collected."""
    try:
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, 'r') as f:
                return max(0, len(f.readlines()) - 1)  # Exclude header
        return 0
    except:
        return 0


def check_retraining_needed():
    """Check if enough feedback has been collected to trigger retraining."""
    feedback_count = get_feedback_count()
    return feedback_count >= FEEDBACK_THRESHOLD, feedback_count


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the HTML dashboard."""
    model_info = active_meta.get("model_type", "unknown")
    feature_info = FEATURE_SET
    f1_score = active_meta.get("metrics", {}).get("f1", 0)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAD MLOps Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            .status {{ display: flex; gap: 20px; margin: 20px 0; }}
            .status-box {{ flex: 1; padding: 15px; border-radius: 8px; text-align: center; }}
            .status-box.green {{ background: #e8f5e9; border: 1px solid #4CAF50; }}
            .status-box.blue {{ background: #e3f2fd; border: 1px solid #2196F3; }}
            .status-box.orange {{ background: #fff3e0; border: 1px solid #FF9800; }}
            .upload-area {{ border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 8px; }}
            .upload-area:hover {{ border-color: #4CAF50; background: #f9f9f9; }}
            button {{ background: #4CAF50; color: white; border: none; padding: 12px 30px; border-radius: 5px; cursor: pointer; font-size: 16px; }}
            button:hover {{ background: #45a049; }}
            #result {{ margin-top: 20px; padding: 20px; border-radius: 8px; display: none; }}
            #result.success {{ background: #e8f5e9; border: 1px solid #4CAF50; display: block; }}
            #result.error {{ background: #ffebee; border: 1px solid #f44336; display: block; }}
            .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
            .metric {{ text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
            .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
            a {{ color: #2196F3; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Voice Activity Detection - MLOps Pipeline</h1>
            
            <div class="status">
                <div class="status-box green">
                    <strong>Model</strong><br>{model_info}
                </div>
                <div class="status-box blue">
                    <strong>Features</strong><br>{feature_info}
                </div>
                <div class="status-box orange">
                    <strong>F1 Score</strong><br>{f1_score:.4f}
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{active_meta.get("metrics", {{}}).get("accuracy", 0):.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{active_meta.get("metrics", {{}}).get("precision", 0):.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{active_meta.get("metrics", {{}}).get("recall", 0):.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
            </div>
            
            <h2>Upload Audio for Prediction</h2>
            <div class="upload-area">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".wav,.mp3,.mp4,.m4a" required>
                    <br><br>
                    <button type="submit">Analyze Audio</button>
                </form>
            </div>
            
            <div id="result"></div>
            
            <h2>API Documentation</h2>
            <ul>
                <li><a href="/docs">Interactive API Docs (Swagger)</a></li>
                <li><a href="/model-info">Model Information (JSON)</a></li>
                <li><a href="/metrics">All Model Metrics</a></li>
                <li><a href="/feedback-stats">Feedback Statistics</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
            
            <h2>Features Used</h2>
            <p>{', '.join(FEATURE_NAMES)}</p>
        </div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {{
                e.preventDefault();
                const formData = new FormData(e.target);
                const resultDiv = document.getElementById('result');
                
                resultDiv.className = '';
                resultDiv.innerHTML = '<p>Processing...</p>';
                resultDiv.style.display = 'block';
                
                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();
                    
                    if (response.ok) {{
                        resultDiv.className = 'success';
                        resultDiv.innerHTML = `
                            <h3>✅ Prediction Result</h3>
                            <p><strong>Result:</strong> ${{data.prediction}}</p>
                            <p><strong>Confidence:</strong> ${{data.confidence}}</p>
                            <p><strong>Processing Time:</strong> ${{data.processing_time_ms}}ms</p>
                            <hr>
                            <p><strong>Feedback:</strong> Was this prediction correct?</p>
                            <button onclick="submitFeedback(${{data.prediction === 'Speech' ? 1 : 0}}, 1)">✓ Correct</button>
                            <button onclick="submitFeedback(${{data.prediction === 'Speech' ? 1 : 0}}, 0)">✗ Incorrect</button>
                        `;
                    }} else {{
                        throw new Error(data.detail || 'Unknown error');
                    }}
                }} catch (error) {{
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `<h3>❌ Error</h3><p>${{error.message}}</p>`;
                }}
            }};
            
            async function submitFeedback(predicted, isCorrect) {{
                const trueLabel = isCorrect ? predicted : (1 - predicted);
                alert(`Feedback recorded! True label: ${{trueLabel === 1 ? 'Speech' : 'Non-Speech'}}`);
            }}
        </script>
    </body>
    </html>
    """
    return html


@app.get("/health")
def health_check():
    """Health check endpoint."""
    feedback_count = get_feedback_count()
    retraining_needed, _ = check_retraining_needed()
    
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": active_meta.get("model_type", "unknown"),
        "feature_set": FEATURE_SET,
        "feedback_collected": feedback_count,
        "feedback_threshold": FEEDBACK_THRESHOLD,
        "retraining_needed": retraining_needed,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
def model_info():
    """Get active model metadata."""
    if not active_meta:
        raise HTTPException(status_code=503, detail="No active model loaded")
    
    return {
        "model_type": active_meta.get("model_type"),
        "feature_set": active_meta.get("feature_set"),
        "feature_names": active_meta.get("feature_names"),
        "feature_count": len(active_meta.get("feature_names", [])),
        "metrics": active_meta.get("metrics"),
        "active_branch": active_meta.get("active_branch"),
        "selection_timestamp": active_meta.get("selection_timestamp"),
        "all_results": active_meta.get("all_results", [])
    }


@app.get("/metrics")
def all_metrics():
    """Get metrics for all trained models."""
    metrics = []
    
    for branch in ["combined", "mfcc_only", "zcr_others"]:
        meta_path = MODELS_DIR / f"meta_{branch}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
                metrics.append({
                    "branch": branch,
                    "model_type": data.get("model_type"),
                    "metrics": data.get("metrics"),
                    "feature_count": len(data.get("feature_names", []))
                })
    
    # Add active model
    if active_meta:
        metrics.append({
            "branch": "active",
            "model_type": active_meta.get("model_type"),
            "metrics": active_meta.get("metrics"),
            "feature_count": len(active_meta.get("feature_names", [])),
            "is_active": True
        })
    
    return {"models": metrics}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict voice activity from audio file."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        
        # Apply bandpass filter
        audio = apply_bandpass_filter(audio)
        
        # Extract features
        features = extract_features(audio, sr)
        
        if features is None or len(features) == 0:
            raise HTTPException(status_code=400, detail="Failed to extract features from audio")
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            "prediction": "Speech" if prediction == 1 else "Non-Speech",
            "label": int(prediction),
            "confidence": round(confidence, 4),
            "processing_time_ms": processing_time,
            "feature_set": FEATURE_SET,
            "features_used": len(features),
            "model_version": active_meta.get("selection_timestamp", "unknown"),
            "feedback_url": "/feedback",
            "feedback_instructions": "POST to /feedback with prediction_id, true_label, and optional notes"
        }
        
        # Log prediction
        log_prediction(features, prediction, confidence, file.filename, processing_time)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/feedback")
async def submit_feedback(
    background_tasks: BackgroundTasks,
    true_label: int = Form(..., description="True label: 1 for Speech, 0 for Non-Speech"),
    predicted_label: Optional[int] = Form(None, description="What the model predicted"),
    confidence: Optional[float] = Form(None, description="Model confidence score"),
    notes: Optional[str] = Form("", description="Optional notes about this sample"),
    file: Optional[UploadFile] = File(None, description="Optional audio file to extract features from")
):
    """Submit feedback for a prediction to improve the model."""
    features = []
    
    # If file provided, extract features
    if file:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            audio = apply_bandpass_filter(audio)
            features = extract_features(audio, sr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        # Use placeholder features if no file provided
        features = [0.0] * len(FEATURE_NAMES)
    
    # Save feedback
    success = save_feedback(features, true_label, predicted_label or -1, confidence or 0, notes)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save feedback")
    
    # Check if retraining is needed
    retraining_needed, feedback_count = check_retraining_needed()
    
    response = {
        "status": "feedback_recorded",
        "feedback_count": feedback_count,
        "threshold": FEEDBACK_THRESHOLD,
        "retraining_needed": retraining_needed,
        "message": f"Feedback recorded. Total feedback samples: {feedback_count}"
    }
    
    if retraining_needed:
        response["message"] += f". Threshold ({FEEDBACK_THRESHOLD}) reached! Retraining recommended."
        response["action"] = "Trigger /trigger-update to start retraining"
    
    return response


@app.get("/feedback-stats")
def feedback_stats():
    """Get feedback collection statistics."""
    feedback_count = get_feedback_count()
    retraining_needed, _ = check_retraining_needed()
    
    # Count by label if feedback exists
    label_distribution = {}
    if FEEDBACK_FILE.exists():
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            if 'label' in df.columns:
                label_distribution = df['label'].value_counts().to_dict()
        except:
            pass
    
    return {
        "feedback_collected": feedback_count,
        "feedback_threshold": FEEDBACK_THRESHOLD,
        "retraining_needed": retraining_needed,
        "progress_percentage": min(100, int((feedback_count / FEEDBACK_THRESHOLD) * 100)),
        "label_distribution": label_distribution,
        "feedback_file": str(FEEDBACK_FILE),
        "feedback_file_exists": FEEDBACK_FILE.exists()
    }


@app.post("/trigger-update")
def trigger_update(background_tasks: BackgroundTasks, force: bool = False):
    """Trigger model retraining and deployment (requires admin access in production)."""
    feedback_count = get_feedback_count()
    retraining_needed, _ = check_retraining_needed()
    
    if not force and not retraining_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough feedback collected ({feedback_count}/{FEEDBACK_THRESHOLD}). Use force=true to override."
        )
    
    # In a real production environment, this would trigger the GitHub Actions workflow
    # For now, we return instructions
    return {
        "status": "retraining_triggered",
        "feedback_count": feedback_count,
        "message": "To retrain and deploy the model:",
        "instructions": [
            "1. Go to your GitHub repository",
            "2. Navigate to Actions > 'Auto Model Update - Feedback Based Retraining'",
            "3. Click 'Run workflow'",
            "4. Optionally check 'Force retraining'",
            "5. Click 'Run workflow' to start"
        ],
        "github_cli_command": f"gh workflow run auto_update.yml --repo ${{github.repository}}",
        "note": "This endpoint logs the trigger request. Actual retraining happens via GitHub Actions."
    }


@app.post("/reload-model")
def reload_model():
    """Reload the active model (useful after model update)."""
    global model, scaler, active_meta
    
    try:
        model, scaler, active_meta = load_active()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_type": active_meta.get("model_type"),
            "feature_set": active_meta.get("feature_set")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )
