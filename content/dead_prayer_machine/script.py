import librosa
import numpy as np
import soundfile as sf
import os
import random

OUTPUT_PATH = "content/dead_prayer_machine/outputs/latest.wav"

sr = 22050
duration = 20

# si fichier existe → continuer
if os.path.exists(OUTPUT_PATH):
    y, sr = librosa.load(OUTPUT_PATH, sr=sr)
else:
    y = np.random.randn(sr * duration)

# transformations
pitch = random.uniform(-2.0, 2.0)
stretch = random.uniform(0.85, 1.15)

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

y2 = np.tanh(y2 * random.uniform(1.2, 2.0))
y2 += np.random.normal(0, 0.001, len(y2))

# durée constante
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

sf.write(OUTPUT_PATH, y2, sr)
