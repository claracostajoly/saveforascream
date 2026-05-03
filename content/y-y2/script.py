import numpy as np
import librosa
import soundfile as sf
import os
import random
from glob import glob

def machine_a_oublier():
    # 1. Trouver le dernier état généré dans /outputs
    states = sorted(glob("outputs/state_*.wav"))
    
    if not states:
        # Si rien n'existe, on part du fichier initial
        input_file = "input_init.wav"
        current_idx = 0
    else:
        input_file = states[-1]
        current_idx = int(input_file.split("_")[-1].split(".")[0])

    # 2. Charger le signal (y)
    y, sr = librosa.load(input_file, sr=None)
    duration_samples = len(y)
    
    # 3. Transformer (y2)
    # Légère dérive temporelle et spectrale
    rate = random.uniform(0.99, 1.01)
    y_transformed = librosa.effects.time_stretch(y, rate=rate)
    y_transformed = librosa.effects.pitch_shift(y_transformed, sr=sr, n_steps=random.uniform(-0.2, 0.2))
    
    # Ajout du résidu de calcul (bruit d'érosion)
    y_transformed += np.random.normal(0, 0.0005, len(y_transformed))
    
    # 4. Aligner (Correction du bug de durée)
    if len(y_transformed) > duration_samples:
        y_next = y_transformed[:duration_samples]
    else:
        y_next = np.pad(y_transformed, (0, duration_samples - len(y_transformed)))

    # 5. Sauvegarder le nouvel état (Substituer)
    new_idx = current_idx + 1
    output_path = f"outputs/state_{new_idx:04d}.wav"
    sf.write(output_path, y_next, sr)
    print(f"Nouvel état généré : {output_path}")

if __name__ == "__main__":
    machine_a_oublier()
