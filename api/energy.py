import librosa
import numpy as np

def extract_energy(audio):
    """
    Extract RMS energy from audio signal
    """
    rms = librosa.feature.rms(y=audio)
    return np.mean(rms)