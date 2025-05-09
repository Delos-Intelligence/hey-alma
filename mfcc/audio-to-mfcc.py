import librosa
import numpy as np
import os

# Input and output folders
input_folder = "dataset/positive_samples"
output_folder = "dataset-mfcc/positive_samples"

os.makedirs(output_folder, exist_ok=True)

# Librosa settings
sample_rate = 16000
n_mfcc = 13  # or 20 for more detail
max_duration = 1.0  # seconds
max_length = int(sample_rate * max_duration)

for filename in os.listdir(input_folder):
    if not filename.endswith(".wav"):
        continue

    filepath = os.path.join(input_folder, filename)

    # Load and trim/pad to 1 second
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=512,
        n_fft=1024
    )

    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Save as .npy file
    np.save(os.path.join(output_folder, filename.replace(".wav", ".npy")), mfcc)

print("✅ MFCC preprocessing complete!")
