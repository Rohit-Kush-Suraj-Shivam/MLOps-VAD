"""
tests/test_pipeline.py
Run with:  pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
import tempfile
import json


# ─── Feature extraction tests ─────────────────────────────────────────────────

def make_audio(duration=1.0, sr=22050, freq=440):
    """Synthetic sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


class TestCombinedFeatures:
    def test_shape(self):
        from features.combined import extract_features, N_FEATURES
        audio = make_audio()
        feats = extract_features(audio, 22050)
        assert feats.shape == (1, N_FEATURES), f"Expected (1, {N_FEATURES}), got {feats.shape}"

    def test_no_nan(self):
        from features.combined import extract_features
        audio = make_audio()
        feats = extract_features(audio, 22050)
        assert not np.any(np.isnan(feats)), "Features contain NaN"

    def test_feature_names_count(self):
        from features.combined import feature_names, N_FEATURES
        assert len(feature_names()) == N_FEATURES


class TestMFCCOnlyFeatures:
    def test_shape(self):
        from features.mfcc_only import extract_features, N_FEATURES
        audio = make_audio()
        feats = extract_features(audio, 22050)
        assert feats.shape == (1, N_FEATURES)

    def test_no_nan(self):
        from features.mfcc_only import extract_features
        feats = extract_features(make_audio(), 22050)
        assert not np.any(np.isnan(feats))


class TestZCROthersFeatures:
    def test_shape(self):
        from features.zcr_others import extract_features, N_FEATURES
        audio = make_audio()
        feats = extract_features(audio, 22050)
        assert feats.shape == (1, N_FEATURES)

    def test_no_nan(self):
        from features.zcr_others import extract_features
        feats = extract_features(make_audio(), 22050)
        assert not np.any(np.isnan(feats))

    def test_silence_features(self):
        """Silence should give near-zero energy and ZCR."""
        from features.zcr_others import extract_features
        silence = np.zeros(22050, dtype=np.float32)
        feats = extract_features(silence, 22050)
        energy = feats[0][0]
        assert energy < 1e-5, f"Energy for silence should be near 0, got {energy}"


# ─── Training smoke test ──────────────────────────────────────────────────────

class TestTraining:
    """Quick smoke test: can we train on a tiny synthetic dataset?"""

    def _make_tiny_dataset(self, path):
        import pandas as pd
        np.random.seed(42)
        n = 100
        # speech: higher mfcc values; noise: lower
        speech = np.random.randn(n//2, 13) + 2
        noise  = np.random.randn(n//2, 13) - 2
        data   = np.vstack([speech, noise])
        labels = np.array([1]*(n//2) + [0]*(n//2))
        cols   = [f"mfcc_{i}" for i in range(1,14)] + ["energy","zcr","spectral_centroid"]
        df     = pd.DataFrame(
            np.hstack([data, np.random.rand(n,3)]),
            columns=cols
        )
        df["label"] = labels
        df.to_csv(path, index=False)

    def test_combined_train(self, tmp_path):
        import subprocess
        csv_path = tmp_path / "dataset.csv"
        self._make_tiny_dataset(str(csv_path))
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        env = {
            **os.environ,
            "FEATURE_SET": "combined",
            "DATASET_PATH": str(csv_path),
            "MODEL_OUT_DIR": str(model_dir),
            "MLFLOW_TRACKING_URI": f"file://{tmp_path}/mlruns",
        }
        result = subprocess.run(
            [sys.executable, "train.py"],
            env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
        assert (model_dir / "model_combined.pkl").exists()
        assert (model_dir / "scaler_combined.pkl").exists()
        assert (model_dir / "meta_combined.json").exists()

    def test_mfcc_only_train(self, tmp_path):
        import subprocess
        csv_path = tmp_path / "dataset.csv"
        self._make_tiny_dataset(str(csv_path))
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        env = {
            **os.environ,
            "FEATURE_SET": "mfcc_only",
            "DATASET_PATH": str(csv_path),
            "MODEL_OUT_DIR": str(model_dir),
            "MLFLOW_TRACKING_URI": f"file://{tmp_path}/mlruns",
        }
        result = subprocess.run(
            [sys.executable, "train.py"],
            env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
        assert (model_dir / "model_mfcc_only.pkl").exists()


# ─── Model selector test ──────────────────────────────────────────────────────

class TestModelSelector:
    def test_selector_picks_best(self, tmp_path):
        """Selector should pick the branch with highest F1."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create fake branch outputs
        branches = {
            "combined":  0.92,
            "mfcc_only": 0.85,
            "zcr_others": 0.78,
        }
        for branch, f1 in branches.items():
            m = LogisticRegression()
            m.fit([[0,0],[1,1]], [0,1])
            s = StandardScaler()
            s.fit([[0,0],[1,1]])
            joblib.dump(m, model_dir / f"model_{branch}.pkl")
            joblib.dump(s, model_dir / f"scaler_{branch}.pkl")
            meta = {
                "feature_set": branch,
                "model_type": "LogisticRegression",
                "feature_names": ["f1","f2"],
                "n_features": 2,
                "metrics": {"f1": f1, "accuracy": f1, "roc_auc": f1},
                "run_id": "test",
            }
            with open(model_dir / f"meta_{branch}.json", "w") as f:
                json.dump(meta, f)

        env = {
            **os.environ,
            "MODEL_OUT_DIR": str(model_dir),
            "MLFLOW_TRACKING_URI": f"file://{tmp_path}/mlruns",
        }
        import subprocess
        result = subprocess.run(
            [sys.executable, "model_selector.py"],
            env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        active_meta_path = model_dir / "active" / "meta.json"
        assert active_meta_path.exists()
        with open(active_meta_path) as f:
            active = json.load(f)
        assert active["feature_set"] == "combined"  # highest F1


# ─── API tests ────────────────────────────────────────────────────────────────

class TestAPI:
    """Integration tests for the FastAPI endpoints."""

    @pytest.fixture
    def client(self, tmp_path):
        """Set up a test client with a pre-trained model."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Build a tiny working active model (combined, 16 features)
        active_dir = tmp_path / "models" / "active"
        active_dir.mkdir(parents=True)

        X = np.random.randn(40, 16)
        y = np.array([0]*20 + [1]*20)
        scaler = StandardScaler().fit(X)
        model  = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)

        joblib.dump(model,  active_dir / "model.pkl")
        joblib.dump(scaler, active_dir / "scaler.pkl")
        meta = {
            "feature_set": "combined",
            "model_type": "LogisticRegression",
            "feature_names": [f"mfcc_{i}" for i in range(1,14)] + ["energy","zcr","spectral_centroid"],
            "n_features": 16,
            "metrics": {"f1": 0.9, "accuracy": 0.9},
            "run_id": "test",
        }
        with open(active_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Patch env before importing app
        os.environ["ACTIVE_MODEL_DIR"] = str(active_dir)
        os.environ["MODEL_OUT_DIR"]    = str(tmp_path / "models")

        # Import fresh
        import importlib
        if "api.main" in sys.modules:
            del sys.modules["api.main"]

        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_info(self, client):
        r = client.get("/model-info")
        assert r.status_code == 200
        data = r.json()
        assert "feature_set" in data

    def test_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "VAD" in r.text

    def test_predict_wav(self, client):
        """Upload a synthetic WAV and check response structure."""
        import io, wave, struct
        sr, duration = 22050, 1
        samples = [int(32767 * np.sin(2 * np.pi * 440 * t / sr)) for t in range(sr * duration)]
        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        buf.seek(0)

        r = client.post("/predict", files={"file": ("test.wav", buf, "audio/wav")})
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert data["prediction"] in ("Speech", "Noise")
        assert 0.0 <= data["speech_probability"] <= 1.0
        assert "model_branch" in data

    def test_predict_invalid_type(self, client):
        r = client.post("/predict", files={"file": ("test.txt", b"hello", "text/plain")})
        assert r.status_code == 400
