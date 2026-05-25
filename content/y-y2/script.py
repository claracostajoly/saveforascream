import os
import time

BASE_PATH = "content/y-y2/"
COUNT_PATH = BASE_PATH + "outputs/count.txt"

# S'assurer que le dossier existe
os.makedirs(os.path.dirname(COUNT_PATH), exist_ok=True)

# 1. LECTURE ET INCRÉMENTATION
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

print(f"Iteration {count} enregistrée.")
