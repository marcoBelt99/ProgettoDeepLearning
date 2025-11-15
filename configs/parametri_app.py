import os

from pathlib import Path
import yaml

## Caricamento della configurazione yaml
## FIXME: controllare le eccezioni
with open(Path(__file__).parent / 'project_config.yml', 'r') as f:
    config = yaml.safe_load(f)



########################################################
####################### PERCORSI #######################
########################################################

## Dal file attuale mi recupero la directory root del progetto
PROJECT_ROOT = Path(__file__).parent.parent

def get_project_path(relative_path) -> Path:
    """
    Converte un percorso relativo in assoluto rispetto alla root del progetto.
    Utile per costruire gli altri path "alla bisogna".
    """
    return PROJECT_ROOT / relative_path

## Costanti per i percorsi principali.
#  (uso le chiavi che ho definito nel file yaml di configurazione
#  per i vari parametri).


############################### DATI ###############################
#  Cartella: 'data'
DATA_DIR = get_project_path(config['paths']['data']['root'])

#  Cartella: 'data/raw'
RAW_DATA_DIR = get_project_path(config['paths']['data']['raw'])

#  File:     'data/raw/dataset_14_punti_pulito_06072023.zip'
DATASET_ZIP = get_project_path(config['paths']['data']['dataset_zip'])

#  Cartella: 'data/dataset' (dopo che è stata estratta. Contiene sia le immagini che i file di testo).
DATASET_DIR = get_project_path(config['paths']['data']['dataset_dir'])


############################### DATAFRAME (in csv) ###############################
#  File:     'data/dataframe_master.csv' (dataframe completo di tutte le colonne, cioè di tutti i punti).
DATAFRAME_MASTER = get_project_path(config['paths']['data']['dataframe_master'])

# Percorsi ai dataframe specifici
DATAFRAME_GRUPPO_1 = get_project_path(config['paths']['data']['dataframe_gruppo_1'])
DATAFRAME_GRUPPO_2 = get_project_path(config['paths']['data']['dataframe_gruppo_2'])
DATAFRAME_GRUPPO_3 = get_project_path(config['paths']['data']['dataframe_gruppo_3'])
DATAFRAME_GRUPPO_4 = get_project_path(config['paths']['data']['dataframe_gruppo_4'])






############################### MODELLI ###############################
#  Cartella:   'src/models/' alla cartella models
MODELS_DIR = get_project_path(config['paths']['src']['models']['models'])

#  Cartella:   'src/models/checkpoints' (memorizza i pesi dei vari modelli di RN).
CHECKPOINTS_DIR = get_project_path(config['paths']['src']['models']['checkpoints'])


########################################################
################### ALTRI PARAMETRI ####################
########################################################

# Lista di tutti i nomi dei files presenti dentro alla cartella DATASET_DIR
FILES = os.listdir(DATASET_DIR) if DATASET_DIR.exists() else []

num_totale_punti : int = 14

'''
 Ogni modello avrà un numero di output pari a len( GRUPPO )*2
 Mi conviene salvarmi / isolarmi i punti in diversi "sotto-dataframe", ognuno associato allo specifico modello
 Credo che questo sia importante anche per poter avere una maggior granularità verso lo specifico gruppo di punti.
 Infatti, se per caso trovo difficoltà per un
'''
RAGGRUPPAMENTI = {
    "GRUPPO1": [0, 1, 4, 5],    # S, N, A, B
    "GRUPPO2": [2, 3, 7, 8],    # Sna, Snp, Gn, Go
    "GRUPPO3": [9, 10, 11, 12], # U1r, U1t, L1r, L1t
    "GRUPPO4": [6, 13]          # Pg, Mesial
}

## Mi salvo tutti gli indici dei raggruppamenti per il modello più generale
lista_indici_di_tutti_i_punti = sum( RAGGRUPPAMENTI.values(), [] )