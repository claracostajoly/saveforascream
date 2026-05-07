import librosa
import numpy as np
import soundfile as sf
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LATEST_FILE = os.path.join(OUTPUT_DIR, "latest.wav")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

sr = 22050
duration = 20

# 1. POINT DE DÉPART : Le "bruit de fond" du système
if os.path.exists(LATEST_FILE):
    y, _ = librosa.load(LATEST_FILE, sr=sr)
else:
    # Amplitude de 10^-5 : un silence "technique" qui contient des micro-données
    y = np.random.normal(0, 0.00001, sr * duration)

# 2. LOGIQUE y = y2 (Substitution récursive)
pitch = random.uniform(-0.3, 0.3) 
stretch = random.uniform(0.99, 1.01)

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# 3. ÉMERGENCE (L'architecture du gain)
# Le gain léger (1.02) permet au "vide" de monter en résonance au fil des heures
y2 = np.tanh(y2 * 1.02)
y2 += np.random.normal(0, 0.00001, len(y2))

# 4. MAINTIEN DU CADRE
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

# 5. y = y2
sf.write(LATEST_FILE, y2, sr)
