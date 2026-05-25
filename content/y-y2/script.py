import librosa
import numpy as np
import soundfile as sf
import os
import random

# Chemins
BASE_PATH = "content/y-y2/"
OUTPUT_PATH = BASE_PATH + "outputs/latest.wav"
COUNT_PATH = BASE_PATH + "outputs/count.txt"

sr = 22050
duration = 20
fade_samples = int(sr * 0.1) # 100ms de fondu pour éliminer le pop physique

# 1. GESTION DU COMPTEUR
if os.path.exists(COUNT_PATH):
    with open(COUNT_PATH, "r") as f:
        try:
            count = int(f.read().strip()) + 1
        except ValueError:
            count = 1
else:
    count = 1

with open(COUNT_PATH, "w") as f:
    f.write(str(count))

# 2. CHARGEMENT
if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 44:
    y, _ = librosa.load(OUTPUT_PATH, sr=sr)
else:
    # Premier résidu : un souffle initial très bas
    y = np.random.normal(0, 0.005, sr * duration)

# 3. TRANSFORMATIONS
pitch = random.uniform(-2.0, 2.0)
stretch = random.uniform(0.85, 1.15)

# Sécurité si le fichier précédent était altéré
if len(y) == 0:
    y = np.random.normal(0, 0.005, sr * duration)

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# Gain contrôlé pour éviter le clipping numérique absolu (qui fait poper les navigateurs)
y2 = np.tanh(y2 * 4.0) 
y2 += np.random.normal(0, 0.003, len(y2))

# Normalisation de la taille
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

# --- PROTECTION ANTI-POP PHYSIQUE (FADE) ---
# On crée une rampe progressive au début et à la fin du tableau de données
window = np.ones(len(y2))
window[:fade_samples] = np.linspace(0, 1, fade_samples)
window[-fade_samples:] = np.linspace(1, 0, fade_samples)
y2 = y2 * window
# --------------------------------------------

# Exportation en PCM_16 standard (le format universel des navigateurs)
sf.write(OUTPUT_PATH, y2, sr, subtype='PCM_16')
