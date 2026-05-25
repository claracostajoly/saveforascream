import os
import time
import subprocess

# Chemins absolus ou relatifs au dépôt
BASE_PATH = "content/y-y2/"
COUNT_PATH = BASE_PATH + "outputs/count.txt"

def run_git_commands():
    try:
        # 1. Ajoute uniquement le fichier de comptage pour éviter les conflits
        subprocess.run(["git", "add", COUNT_PATH], check=True)
        
        # 2. Crée le commit avec un message automatique
        subprocess.run(["git", "commit", "-m", "system: loop degradation update"], check=True)
        
        # 3. Pousse la mise à jour sur GitHub (branche main)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("-> résidu synchronisé sur github.")
    except subprocess.CalledProcessError as e:
        print(f"erreur de synchronisation git : {e}")

def main():
    print("machine à oublier activée. synchronisation toutes les 5 minutes.")
    
    while True:
        # Gestion du compteur
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

        print(f"\nitération {count} générée en local.")
        
        # Envoi automatique sur GitHub
        run_git_commands()
        
        # Attente de 5 minutes (300 secondes) avant le prochain cycle
        time.sleep(300)

if __name__ == "__main__":
    main()
