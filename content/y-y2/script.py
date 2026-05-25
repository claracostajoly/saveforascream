import librosa
import numpy as np
import soundfile as sf
import os
import random

BASE_PATH = "content/y-y2/"
OUTPUT_PATH = BASE_PATH + "outputs/latest.wav"
COUNT_PATH = BASE_PATH + "outputs/count.txt"

sr = 22050
duration = 20
fade_samples = int(sr * 0.2) # 200ms pour lisser mathématiquement l'entrée/sortie du bloc

# 1. COMPTEUR
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

# 2. CHARGEMENT / MATIÈRE INITIALE
if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 44:
    try:
        y, _ = librosa.load(OUTPUT_PATH, sr=sr)
        if len(y) == 0 or np.all(y == 0):
            raise ValueError
    except Exception:
        # Si le fichier était vide ou corrompu, on génère un grondement sourd
        y = np.random.normal(0, 0.02, sr * duration)
else:
    y = np.random.normal(0, 0.02, sr * duration)

# 3. TRANSFORMATION ET ACCUMULATION
pitch = random.uniform(-1.0, 1.0)
stretch = random.uniform(0.95, 1.05)

y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# Gain progressif lié au temps + injection constante d'erreur thermique
facteur_matiere = min(count * 0.0005, 0.05)
y2 = np.tanh(y2 * (2.0 + facteur_matiere))
y2 += np.random.normal(0, 0.005 + (facteur_matiere * 0.1), len(y2))

# Normalisation de la taille
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

# --- FENÊTRAGE STRICT ANTI-POP ---
# Rampe progressive pour amener et raccompagner le signal à zéro aux extrémités
window = np.ones(len(y2))
window[:fade_samples] = np.linspace(0, 1, fade_samples)
window[-fade_samples:] = np.linspace(1, 0, fade_samples)
y2 = y2 * window

# Normalisation du volume général pour éviter la saturation numérique destructrice
if np.max(np.abs(y2)) > 0:
    y2 = y2 / np.max(np.abs(y2)) * 0.5

sf.write(OUTPUT_PATH, y2, sr, subtype='PCM_16')
