import librosa
import numpy as np

def extract_mfcc(audio, sr, n_mfcc=13):
    """
    Extract MFCC features from audio signal
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)