import os
import librosa
import numpy as np
import pandas as pd
import random

SAMPLE_RATE = 22050
MAX_FILES_KAGGLE = 1000  # limit per Kaggle class


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # 13 MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Energy
    energy = np.mean(audio ** 2)

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    # Spectral Centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sr)
    )

    return np.hstack([mfcc_mean, energy, zcr, spectral_centroid])


def process_folder(folder_path, label, max_files=None):
    data = []
    files = []

    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))

    random.shuffle(files)

    if max_files:
        files = files[:max_files]

    for file_path in files:
        try:
            features = extract_features(file_path)
            row = list(features) + [label]
            data.append(row)
        except:
            continue

    return data


# -------------------------------
# Kaggle Google Speech Commands
# -------------------------------
speech_data_kaggle = process_folder(
    "C:/Users/SHIVAM/Downloads/archive (3)",
    1,
    MAX_FILES_KAGGLE
)

# -------------------------------
# Kaggle MUSAN Noise
# -------------------------------
noise_data_kaggle = process_folder(
    "C:/Users/SHIVAM/Downloads/archive (4)/musan/noise",
    0,
    MAX_FILES_KAGGLE
)

# -------------------------------
# Mic Recorded Speech
# -------------------------------
speech_data_mic = process_folder(
    "mic_dataset/speech",
    1
)

# -------------------------------
# Mic Recorded Noise
# -------------------------------
noise_data_mic = process_folder(
    "mic_dataset/noise",
    0
)

# Combine everything
all_data = (
    speech_data_kaggle +
    noise_data_kaggle +
    speech_data_mic +
    noise_data_mic
)

random.shuffle(all_data)

columns = [f"mfcc_{i}" for i in range(1, 14)] + \
          ["energy", "zcr", "spectral_centroid", "label"]

df = pd.DataFrame(all_data, columns=columns)

df.to_csv("balanced_vad_dataset.csv", index=False)

print("New dataset created successfully with Kaggle + Mic data!")