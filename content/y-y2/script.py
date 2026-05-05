import numpy as np
import librosa
import soundfile as sf
import os
import random
from glob import glob

def machine_a_oublier():
    # Définir le dossier de base (celui où se trouve le script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Trouver le dernier état généré
    states = sorted(glob(os.path.join(output_dir, "state_*.wav")))
    
    if not states:
        # Chemin absolu vers le fichier initial
        input_file = os.path.join(base_dir, "input_init.wav")
        current_idx = 0
    else:
        input_file = states[-1]
        # Extraction de l'index de manière plus robuste
        filename = os.path.basename(input_file)
        current_idx = int(filename.split("_")[-1].split(".")[0])

    # Vérification de l'existence du fichier d'entrée
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} est introuvable.")
        return

    # 2. Charger le signal
    y, sr = librosa.load(input_file, sr=None)
    duration_samples = len(y)
    
    # 3. Transformer
    rate = random.uniform(0.99, 1.01)
    y_transformed = librosa.effects.time_stretch(y, rate=rate)
    y_transformed = librosa.effects.pitch_shift(y_transformed, sr=sr, n_steps=random.uniform(-0.2, 0.2))
    
    # Ajout du bruit (attention à la taille du tableau de bruit)
    noise = np.random.normal(0, 0.0005, len(y_transformed))
    y_transformed = y_transformed + noise
    
    # 4. Aligner la durée
    if len(y_transformed) > duration_samples:
        y_next = y_transformed[:duration_samples]
    else:
        y_next = np.pad(y_transformed, (0, duration_samples - len(y_transformed)))

    # 5. Sauvegarder
    new_idx = current_idx + 1
    output_path = os.path.join(output_dir, f"state_{new_idx:04d}.wav")
    sf.write(output_path, y_next, sr)
    print(f"Nouvel état généré : {output_path}")

if __name__ == "__main__":
    machine_a_oublier()
