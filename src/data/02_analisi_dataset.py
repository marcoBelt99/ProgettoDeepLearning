# Parametri da usare in giro per l'app
from configs.parametri_app import *

# Per usare funzioni di uso comune
from utils.utils import get_num_files




def analizza_struttura_dataset() -> None:
    '''
    Effettua semplici cosiderazioni / verifiche numeriche sulla struttura della cartella contente
    i dati grezzi.
    '''
    print("\n(2) Analisi Strutturale del dataset:\n")

    NUM_TOTALE_FILES : int = len( FILES )  # conto gli elementi totali presenti dentro la cartella 'dataset'
    print(f"\n{'#' * 30}\nNumero di files totali: {NUM_TOTALE_FILES}\n{'#' * 30}\n")

    ## 1) Conto quanti files ".jpg" e ".txt" sono presenti
    NUM_FILE_JPG : int = get_num_files("jpg")
    NUM_FILE_TXT: int = get_num_files("txt")
    NUM_OTHERS_FILES = NUM_TOTALE_FILES - (NUM_FILE_JPG + NUM_FILE_TXT)
    print(f"\n{'#' * 30}\nNumero files '.jpg' totali: {NUM_FILE_JPG}\n{'#' * 30}\n")
    print(f"\n{'#' * 30}\nNumero files '.txt' totali: {NUM_FILE_TXT}\n{'#' * 30}\n")
    print(f"\n{'#' * 30}\nNumero di eventuali altri files rimanenti: {NUM_OTHERS_FILES}\n{'#' * 30}\n")

    ## 2) Verifico che ci siano gli stessi numeri, cioè che file di testo e di immagine siano in corrispondenza biunivoca
    lista_nomi_files_txt: list[str] = list(filter(lambda nome_file: (".txt" in nome_file), FILES))
    lista_nomi_files_jpg: list[str] = list(filter(lambda nome_file: (".jpg" in nome_file), FILES))

    # Funzione anonima che mi recupera i singoli nomi
    rimuovi_estensione = lambda nf: int(nf.split('.')[0])

    lista_numeri_files_txt: list[int] = sorted(list(map(rimuovi_estensione, lista_nomi_files_txt)))
    lista_numeri_files_jpg: list[int] = sorted(list(map(rimuovi_estensione, lista_nomi_files_jpg)))
    # In entrambe le liste devo avere esattamente gli stessi numeri affinchè ci sia una corrispondenza 1:1
    # tra immagine e file di testo ad essa associato.
    stessi_numeri: bool = all(n_txt == n_jpg for n_txt, n_jpg in zip(lista_numeri_files_txt, lista_numeri_files_jpg))

    print("Le due liste sono in corrispondenza biunivoca (stessi numeri)" if stessi_numeri else "Ci sono numeri diversi")

def main():

    ## Richiamo la funzione
    analizza_struttura_dataset()


if __name__ == '__main__':
    main()

