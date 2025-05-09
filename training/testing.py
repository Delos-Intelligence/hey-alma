#!/usr/bin/env python3

import librosa
import numpy as np
import tensorflow as tf

# Load model (add .keras extension!)
model = tf.keras.models.load_model("wakeword_model.keras")

# Config (must match training settings)
sample_rate = 16000
n_mfcc = 13
max_duration = 1.0  # seconds
max_length = int(sample_rate * max_duration)
max_time_steps = 44  # adjust to match training if needed

def predict_from_wav(filepath):
    # Load and pad audio
    print(f"Loading audio from {filepath}")
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=512,
        n_fft=1024
    )

    # Pad / truncate time steps
    if mfcc.shape[1] < max_time_steps:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_time_steps - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_time_steps]

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Add batch and channel dimensions
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Predict
    prediction = model.predict(mfcc)[0][0]
    print(f"Prediction Score: {prediction:.4f} {'✅ Wake word detected!' if prediction > 0.5 else '❌ Not wake word'}")

# 🔊 Test with a .wav file
predict_from_wav("test_audio/positive3.wav")
