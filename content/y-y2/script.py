import librosa
import numpy as np
import soundfile as sf
import os
import random

# Configuration des chemins
BASE_PATH = "content/y-y2/"
OUTPUT_PATH = BASE_PATH + "outputs/latest.wav"
COUNT_PATH = BASE_PATH + "outputs/count.txt"

sr = 22050
duration = 20
fade_samples = int(sr * 0.05)  # Fenêtre de protection de 50ms pour lisser le tableau de données

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
    # Seuil de départ clinique (presque inaudible)
    y = np.random.normal(0, 0.001, sr * duration)

# 3. ACCUMULATION CONCEPTUELLE DE LA MATIÈRE
pitch = random.uniform(-1.5, 1.5)
stretch = random.uniform(0.9, 1.1)

if len(y) == 0:
    y = np.random.normal(0, 0.001, sr * duration)

# Déplacement de l'empreinte temporelle
y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# Saturation mathématique
y2 = np.tanh(y2 * 3.5)

# Intensification du résidu liée au compteur (l'erreur s'épaissit avec le temps)
facteur_matiere = min(count * 0.0002, 0.015)
y2 += np.random.normal(0, 0.002 + facteur_matiere, len(y2))

# Normalisation stricte de la matrice
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

# 4. ALIGNEMENT DES EXTRÉMITÉS À ZÉRO SÉCURISÉ
window = np.ones(len(y2))
window[:fade_samples] = np.linspace(0, 1, fade_samples)
window[-fade_samples:] = np.linspace(1, 0, fade_samples)
y2 = y2 * window

# Export au format universel non compressé
sf.write(OUTPUT_PATH, y2, sr, subtype='PCM_16')
