import numpy as np
import librosa
from .mfcc import extract_mfcc
from .energy import extract_energy

def extract_features(audio, sr):
    """
    Combine MFCC, Energy, ZCR, Spectral Centroid
    """

    # MFCC (13)
    mfcc_features = extract_mfcc(audio, sr)

    # Energy (1)
    energy_feature = extract_energy(audio)

    # Zero Crossing Rate (1)
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    # Spectral Centroid (1)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    features = np.hstack([
        mfcc_features,
        energy_feature,
        zcr_mean,
        spectral_centroid_mean
    ])

    return features.reshape(1, -1)