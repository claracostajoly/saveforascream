import librosa
import numpy as np
import soundfile as sf
import os
import random

OUTPUT_PATH = "content/y-y2/outputs/latest.wav"

sr = 22050
duration = 20
target_len = sr * duration

# 🔁 charger ou initialiser
if os.path.exists(OUTPUT_PATH):
    y, _ = librosa.load(OUTPUT_PATH, sr=sr)
else:
    y = np.random.randn(target_len)

# ⚠️ éviter NaN / silence
if np.isnan(y).any() or np.max(np.abs(y)) < 1e-6:
    y = np.random.randn(target_len)

# 🎛️ transformations (plus audibles)
pitch = random.uniform(-5.0, 5.0)
stretch = random.uniform(0.75, 1.25)

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# 🔥 saturation plus marquée (donne du caractère)
gain = random.uniform(2.0, 4.0)
y2 = np.tanh(y2 * gain)

# 🌫️ bruit (évite stabilisation totale)
noise_level = random.uniform(0.002, 0.01)
y2 += np.random.normal(0, noise_level, len(y2))

# 📏 longueur fixe
if len(y2) < target_len:
    y2 = np.pad(y2, (0, target_len - len(y2)))
else:
    y2 = y2[:target_len]

# 🔊 normalisation (évite saturation morte)
max_val = np.max(np.abs(y2))
if max_val > 0:
    y2 = y2 / max_val * 0.9

# 💾 sauvegarde
sf.write(OUTPUT_PATH, y2, sr)
