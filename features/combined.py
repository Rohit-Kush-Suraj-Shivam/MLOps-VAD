"""
Branch: feature/combined
Features: MFCC (13) + Energy + ZCR + Spectral Centroid  →  16 features total
"""
import numpy as np
import librosa

FEATURE_SET = "combined"
N_FEATURES = 16


def extract_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract all 16 features from a raw audio array."""
    # 13 MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # (13,)

    # Energy (RMS)
    energy = np.mean(librosa.feature.rms(y=audio))  # scalar

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))  # scalar

    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))  # scalar

    features = np.hstack([mfcc_mean, energy, zcr, spectral_centroid])  # (16,)
    return features.reshape(1, -1)


def feature_names() -> list:
    return [f"mfcc_{i}" for i in range(1, 14)] + ["energy", "zcr", "spectral_centroid"]
