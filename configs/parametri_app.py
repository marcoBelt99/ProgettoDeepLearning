import os

from pathlib import Path
import yaml

# Caricamento della configurazione
with open(Path(__file__).parent / 'project_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Dal file attuale mi recupero la directory root del progetto
PROJECT_ROOT = Path(__file__).parent.parent

def get_project_path(relative_path) -> Path:
    """
    Converte un percorso relativo in assoluto rispetto alla root del progetto.
    """
    return PROJECT_ROOT / relative_path

## Variabili per i percorsi principali.
## Uso le chiavi che ho definito nel file yaml di configurazione per i vari parametri
DATA_DIR = get_project_path(config['paths']['data']['root'])
RAW_DATA_DIR = get_project_path(config['paths']['data']['raw'])
DATASET_ZIP = get_project_path(config['paths']['data']['dataset_zip'])

# Cartella 'dataset' dopo che è stata estratta
DATASET_DIR = get_project_path(config['paths']['data']['dataset_dir'])

# Lista di tutti i nomi dei files presenti dentro alla cartella in cui e' contenuto il dataset
FILES = os.listdir(DATASET_DIR) if DATASET_DIR.exists() else []

# Percorso al dataframe completo di tutte le colonne
DATAFRAME_MASTER = get_project_path(config['paths']['data']['dataframe_master'])

# Percorsi ai dataframe specifici
# ...

num_totale_punti = 14
