import librosa
import numpy as np
import soundfile as sf
import os
import random

OUTPUT_PATH = "content/y-y2/outputs/latest.wav"

sr = 22050
duration = 20

# --- INPUT ---
if os.path.exists(OUTPUT_PATH):
    y, sr = librosa.load(OUTPUT_PATH, sr=sr)
else:
    y = np.random.randn(sr * duration) * 0.1  # moins agressif au départ

# --- TRANSFORMATIONS (plus lentes) ---
pitch = random.uniform(-0.5, 0.5)          # avant ±2 → maintenant ±0.5
stretch = random.uniform(0.95, 1.05)       # avant 0.85–1.15 → resserré

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# --- DISTORSION DOUCE ---
y2 = np.tanh(y2 * 1.2)  # moins violent

# --- BRUIT (très réduit) ---
y2 += np.random.normal(0, 0.0003, len(y2))  # avant 0.001

# --- MÉLANGE AVEC L'ÉTAT PRÉCÉDENT ---
# clé : ralentit la disparition sans recréer une "mémoire forte"
y2 = 0.97 * y2 + 0.03 * y

# --- LONGUEUR FIXE ---
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

# --- NORMALISATION DOUCE ---
peak = np.max(np.abs(y2)) + 1e-9
y2 = y2 / peak * 0.85

# --- SOFT LIMITER ---
threshold = 0.75
ratio = 0.3

y2 = np.where(np.abs(y2) > threshold,
              np.sign(y2) * (threshold + (np.abs(y2) - threshold) * ratio),
              y2)

# --- SMOOTHING (anti-clics) ---
y2 = np.convolve(y2, np.ones(5)/5, mode='same')

# --- EXPORT ---
sf.write(OUTPUT_PATH, y2, sr)
