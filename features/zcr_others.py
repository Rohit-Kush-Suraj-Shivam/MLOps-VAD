"""
Branch: feature/zcr-others
Features: Energy + ZCR + Spectral Centroid + Spectral Rolloff + Spectral Bandwidth  →  5 features
"""
import numpy as np
import librosa

FEATURE_SET = "zcr_others"
N_FEATURES = 5


def extract_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract 5 non-MFCC features from a raw audio array."""
    # Energy (RMS)
    energy = np.mean(librosa.feature.rms(y=audio))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    # Spectral Rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    # Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    features = np.array([energy, zcr, spectral_centroid, spectral_rolloff, spectral_bandwidth])
    return features.reshape(1, -1)


def feature_names() -> list:
    return ["energy", "zcr", "spectral_centroid", "spectral_rolloff", "spectral_bandwidth"]
