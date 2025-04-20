import os
import librosa
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from helper import *

TRAIN_DATA_DIR = "data/training"
AUGMENTATION_RANGE = np.arange(-0.02, 0.02, 0.001)

# Load features
X = []
files = []
for f in glob.glob(TRAIN_DATA_DIR + "/*.wav"):
    audio, sr = librosa.load(f, sr=None)
    mid = len(audio) / 2 / sr

    # Use training augmentation to create more samples and improve generalization
    for i in AUGMENTATION_RANGE:
        audio_slice = onset_clip(audio, sr, mid + i)
        feats = features(audio_slice, sr)
        if len(feats) == 0:
            continue
        X.append(feats)
        files.append(os.path.basename(f).split(".")[0])

# Load labels
with open(TRAIN_DATA_DIR + "/labels.csv") as f:
    paddle_hits = set([line.strip().split(".")[0] for line in f])
y = [1 if f in paddle_hits else 0 for f in files]

# Sanity check
for f in paddle_hits:
    if f not in files:
        raise ValueError(f"File {f} not found in training data.")
print("All files in labels.csv found in training data.")
print(f"Positive cases: {AUGMENTATION_RANGE.size*len(paddle_hits)}. Positive labels: {sum(y)}.")
if AUGMENTATION_RANGE.size*len(paddle_hits) != sum(y):
    raise ValueError("Positive label mismatch.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Cross-validation accuracy on {len(y_test)} samples: {accuracy}")

# Save the model
model = RandomForestClassifier()
model.fit(X, y)  # retrain on full data
accuracy = accuracy_score(y, model.predict(X))
model.nfeatures = len(X[0])
print(f"Accuracy on full training data, {len(y)} samples: {accuracy}")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
