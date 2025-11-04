import zipfile

# Parametri da usare in giro per l'app
from configs.parametri_app import *



def unzip_dataset() -> bool:
    '''
    Unzip archivio del dataset.

    Returns:
        - True se il dataset è stato estratto.
        - False se il dataset non è stato estratto.
    '''

    estratto : bool = False

    # Verifico che la directory principale (che contiene sia lo zip che il dataset grezzo) esista
    if os.path.exists( DATA_DIR ):
        print("Directory trovata: ", DATA_DIR)
        print("Contenuto della directory:", os.listdir( DATA_DIR ))

        # Verifico che il file zip esista
        if os.path.exists( DATASET_ZIP ):
            print("File zip del Dataset presente.")
        else:
            print("File zip del Dataset non trovato.")
    else:
        print("Directory non trovata:", DATA_DIR)

    # Estrazione del file: solo quando la cartella 'dataset' non e' gia' presente
    if 'dataset' not in os.listdir( DATA_DIR ):
        with zipfile.ZipFile( DATASET_ZIP, 'r' ) as zip_ref:
            zip_ref.extractall( DATA_DIR )
            estratto = True
    else:
        print("\nLa cartella e' gia' stata estratta dallo zip\n")

    # Controllo del contenuto estratto
    # print(os.listdir( DATA_DIR ))
    return estratto



def main():
    """
    Funzione principale per il preprocessing.
    """

    print("\n(1) Inizio preprocessing del dataset...")

    ## La prima operazione è quella di unzippare il DS se è la prima volta
    dataset_estratto : bool = unzip_dataset()

    if dataset_estratto:
        print("\nDataset estratto con successo.")
    else:
        print("\nDataset non estratto.")



if __name__ == '__main__':
    main()
