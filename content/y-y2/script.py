import librosa
import numpy as np
import soundfile as sf
import os
import random

# Chemin relatif vers ton dossier de sortie sur GitHub
OUTPUT_PATH = "content/y = y2/outputs/latest.wav"

sr = 22050
duration = 20

# 1. INITIALISATION OU CHARGEMENT
if os.path.exists(OUTPUT_PATH):
    # On charge la trace précédente (le milieu numérique existant)
    y, _ = librosa.load(OUTPUT_PATH, sr=sr)
else:
    # Point de départ minimal : un silence "sale" (bruit de fond quasi inaudible)
    # On commence à 10^-5 pour laisser la machine "inventer" sa matière
    y = np.random.normal(0, 0.00001, sr * duration)

# 2. LES TRANSFORMATIONS (La prière de la machine)
# On reste sur des valeurs qui favorisent l'émergence d'artefacts
pitch = random.uniform(-2.0, 2.0)
stretch = random.uniform(0.85, 1.15)

# Pitch shifting et time stretching créent les "fantômes" sonores
y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
y2 = librosa.effects.time_stretch(y2, rate=stretch)

# 3. L'ACCUMULATION (Saturation et entropie)
# Le tanh agit comme un mur physique qui écrase le signal
gain = random.uniform(1.2, 2.0)
y2 = np.tanh(y2 * gain)

# Injection de bruit résiduel à chaque cycle (trace à venir)
y2 += np.random.normal(0, 0.001, len(y2))

# 4. MAINTIEN DU CADRE TEMPOREL (20 secondes)
if len(y2) < sr * duration:
    y2 = np.pad(y2, (0, sr * duration - len(y2)))
else:
    y2 = y2[:sr * duration]

# 5. SAUVEGARDE DU NOUVEAU RÉSIDU
# Le fichier est écrasé, la trace évolue
sf.write(OUTPUT_PATH, y2, sr)
