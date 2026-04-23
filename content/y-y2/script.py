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
    y = np.random.randn(sr * duration)

# --- TRANSFORMATIONS ---
pitch = random.uniform(-2.0, 2.0)
stretch = random.uniform(0.85, 1.15)

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# distorsion douce (moins agressive)
y2 = np.tanh(y2 * 1.5)

# bruit très léger
y2 += np.random.normal(0, 0.001, len(y2))


# --- LONGUEUR FIXE ---
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]


# --- LIMITER / PROTECTION OREILLE ---

# peak
peak = np.max(np.abs(y2)) + 1e-9

# normalisation avec headroom
y2 = y2 / peak * 0.85

# soft limiter (évite pics agressifs)
threshold = 0.75
ratio = 0.2

y2 = np.where(np.abs(y2) > threshold,
              np.sign(y2) * (threshold + (np.abs(y2) - threshold) * ratio),
              y2)

# léger smoothing (évite clics numériques)
y2 = np.convolve(y2, np.ones(5)/5, mode='same')


# --- EXPORT ---
sf.write(OUTPUT_PATH, y2, sr)
