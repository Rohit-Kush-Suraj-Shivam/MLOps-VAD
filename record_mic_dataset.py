import sounddevice as sd
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE = 22050
DURATION = 3  # seconds

def record_samples(label, folder, num_samples):
    print(f"\nRecording {num_samples} samples for: {label}")
    print("Press Enter to start each recording...")

    for i in range(num_samples):
        input(f"\nPress Enter to record sample {i+1}")

        audio = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1)
        sd.wait()

        file_path = os.path.join(folder, f"{label}_{i+1}.wav")
        sf.write(file_path, audio, SAMPLE_RATE)

        print(f"Saved: {file_path}")

    print(f"\nFinished recording {label} samples.")


# Create folders if not exist
os.makedirs("mic_dataset/speech", exist_ok=True)
os.makedirs("mic_dataset/noise", exist_ok=True)

# Record Speech Samples
record_samples("speech", "mic_dataset/speech", 30)

# Record Noise Samples
record_samples("noise", "mic_dataset/noise", 30)

print("\nDataset recording complete!")