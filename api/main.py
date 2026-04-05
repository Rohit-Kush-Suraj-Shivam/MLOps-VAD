from fastapi import FastAPI
import joblib
import numpy as np
import sounddevice as sd
from .combined import extract_features

app = FastAPI(title="Live VAD API")

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

SAMPLE_RATE = 22050
DURATION = 3.0          # record 3 seconds
THRESHOLD = 0.15         # speech probability threshold
SILENCE_ENERGY = 0.0003 # sfilterilence 


from scipy.signal import butter, filtfilt

def highpass_filter(data, cutoff=100, fs=22050, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


@app.get("/detect")
def detect():

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()

    audio = audio.flatten()

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Apply high-pass filter (removes hum / low noise)
    audio = highpass_filter(audio, cutoff=100)

    energy = np.mean(audio ** 2)
    variation = np.std(audio)

    print("Energy:", energy)
    print("Variation:", variation)

    # ML probability (monitor only)
    features = extract_features(audio, SAMPLE_RATE)
    features = scaler.transform(features)
    prob = model.predict_proba(features)[0][1]

    print("Speech Probability:", prob)

    VARIATION_THRESHOLD = 0.14

    if variation > VARIATION_THRESHOLD:
        result = "Speech"
    else:
        result = "Noise"

    return {
        "Prediction": result,
        "Speech_Probability": round(float(prob), 3),
        "Energy": round(float(energy), 6),
        "Variation": round(float(variation), 6)
    }