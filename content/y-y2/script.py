import librosa
import numpy as np
import soundfile as sf
import os
import random

# Chemins (ajuste le tiret si ton dossier est y-y2)
BASE_PATH = "content/y-y2/"
OUTPUT_PATH = BASE_PATH + "outputs/latest.wav"
COUNT_PATH = BASE_PATH + "outputs/count.txt"

sr = 22050
duration = 20

# 1. GESTION DU COMPTEUR
if os.path.exists(COUNT_PATH):
    with open(COUNT_PATH, "r") as f:
        count = int(f.read()) + 1
else:
    count = 1

with open(COUNT_PATH, "w") as f:
    f.write(str(count))

# 2. CHARGEMENT
if os.path.exists(OUTPUT_PATH):
    y, _ = librosa.load(OUTPUT_PATH, sr=sr)
else:
    y = np.random.normal(0, 0.01, sr * duration)

# 3. TRANSFORMATIONS AGRESSIVES
pitch = random.uniform(-2.0, 2.0)
stretch = random.uniform(0.85, 1.15)
y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# Gain fort pour montée rapide
y2 = np.tanh(y2 * 8.0) 
y2 += np.random.normal(0, 0.005, len(y2))

if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

sf.write(OUTPUT_PATH, y2, sr)
