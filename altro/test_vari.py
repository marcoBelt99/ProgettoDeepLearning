import os

# Percorso relativo
percorso_relativo = "../documenti/file.txt"

## Configurazione dei percorsi
data_raw_dir = "../data/raw/"  # Dati grezzi (su archivio zip)
dataset_dir = "../data/dataset"  # Dati unzippati dal dataset

# Ottiene il percorso assoluto
percorso_assoluto = os.path.abspath(percorso_relativo)

print(f"Percorso relativo: {data_raw_dir}")
print(f"Percorso assoluto: {data_raw_dir}")