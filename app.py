import librosa
import numpy as np
import pandas as pd

def extract_features(y, sr):
    # Your existing feature extraction logic
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return np.hstack((mfcc_mean, energy, zcr, spectral_centroid))

def process_single_file(file_path):
    try:
        # librosa handles mp4 by using soundfile or audioread backends
        y, sr = librosa.load(file_path, sr=None)
        
        features = extract_features(y, sr)
        
        # Define names for clarity
        feature_names = ([f"MFCC_{i+1}" for i in range(13)] + 
                         ["Energy", "ZCR", "SpectralCentroid"])
        
        # Create a clean display
        results = pd.Series(features, index=feature_names)
        return results

    except Exception as e:
        return f"Error processing file: {e}"

# --- EXECUTION ---
file_to_check = "test_Speech.mp4" # Replace with your actual file name
feature_results = process_single_file(file_to_check)

print(f"--- Features for {file_to_check} ---")
print(feature_results)