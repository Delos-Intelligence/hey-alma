import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Config
positive_dir = "../mfcc/dataset-mfcc/positive_samples"
negative_dir = "../mfcc/dataset-mfcc/negative_samples"
n_mfcc = 13
max_time_steps = 44

# Load data
def load_mfcc_data(folder, label):
    X = []
    y = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            mfcc = np.load(os.path.join(folder, file))
            if mfcc.shape[1] < max_time_steps:
                pad_width = max_time_steps - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_time_steps]
            X.append(mfcc)
            y.append(label)
    return np.array(X), np.array(y)

# Load MFCCs
X_pos, y_pos = load_mfcc_data(positive_dir, 1)
X_neg, y_neg = load_mfcc_data(negative_dir, 0)

# Combine and shuffle
X = np.concatenate((X_pos, X_neg), axis=0)
y = np.concatenate((y_pos, y_neg), axis=0)

# Normalize
X = (X - np.mean(X)) / np.std(X)

# Add channel dimension for Conv2D
X = X[..., np.newaxis]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(n_mfcc, max_time_steps, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val))

# Save model in .keras format for easy local loading
model.save("wakeword_model.keras")
print("✅ Model training complete and saved as 'wakeword_model.keras'")
