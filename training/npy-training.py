import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Config
positive_dir = "../dataset-raw/positive_samples"
negative_dir = "../dataset-raw/negative_samples"
sample_rate = 16000  # number of samples per audio clip
input_shape = (sample_rate, 1)  # shape: [16000, 1]

# Load data
def load_raw_waveform(folder, label):
    X = []
    y = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            waveform = np.load(os.path.join(folder, file))
            if waveform.shape[0] != sample_rate:
                continue  # skip corrupted or misaligned data
            X.append(waveform)
            y.append(label)
    return np.array(X), np.array(y)

# Load positive and negative samples
X_pos, y_pos = load_raw_waveform(positive_dir, 1)
X_neg, y_neg = load_raw_waveform(negative_dir, 0)

# Combine and shuffle
X = np.concatenate((X_pos, X_neg), axis=0)
y = np.concatenate((y_pos, y_neg), axis=0)

# Normalize (optional, but helps)
X = (X - np.mean(X)) / np.std(X)

# Reshape to [samples, 16000, 1]
X = X[..., np.newaxis]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model (1D CNN for raw audio)
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(16, 13, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(32, 11, activation='relu'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(64, 9, activation='relu'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val))

# Save the model
model.save("wakeword_model.keras")
print("✅ Model training complete and saved as 'wakeword_model.keras'")
