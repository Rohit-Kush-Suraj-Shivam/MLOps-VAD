"""
Branch: feature/mfcc-only
Features: MFCC (13) only  →  13 features total
"""
import numpy as np
import librosa

FEATURE_SET = "mfcc_only"
N_FEATURES = 13


def extract_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract 13 MFCC features from a raw audio array."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # (13,)
    return mfcc_mean.reshape(1, -1)


def feature_names() -> list:
    return [f"mfcc_{i}" for i in range(1, 14)]
