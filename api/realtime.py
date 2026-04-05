import sounddevice as sd
import numpy as np

SAMPLE_RATE = 22050
DURATION = 1  # seconds

def record_audio():
    """
    Record audio from microphone for DURATION seconds
    """
    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()

    print("Recording finished.")

    return audio.flatten()