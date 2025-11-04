import pandas as pd
import os
import zipfile

# Parametri da usare in giro per l'app
from configs.parametri_app import *



def unzip_dataset():
    '''
    Unzip archivio del dataset.
    '''


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
    else:
        print("\nLa cartella e' gia' stata estratta dallo zip\n")

    # Controllo del contenuto estratto
    print(os.listdir( DATA_DIR ))




lista_files_txt = sorted( [t for t in FILES if t.endswith(".txt") ], key=lambda nf: int(nf.split('.')[0]) )
lista_immagini =  sorted( [i for i in FILES if i.endswith(".jpg") ], key=lambda nf: int(nf.split('.')[0]) )

# A scopo di debug
# print( lista_files_txt )
# print('\n')
# print( lista_immagini )

print( f'Lunghezza lista file TXT: {len( lista_files_txt )}' )
print( f'Lunghezza lista file JPG: {len( lista_immagini )}' )

print( f'Primo file TXT: {lista_files_txt[0]}' )
print( f'Primo file JPG: {lista_immagini[0]}' )




#### Creo un nuovo DataFrame
num_punti = 14
dati: pd.DataFrame = pd.DataFrame(columns=['path_img'] +
                                          [f'punto_{n}_{coord}' for n in range(1, num_punti + 1) for coord in
                                           ('X', 'Y')])  # +1 per comprendere anche il 14°-esimo punto nel range


def prepara_dataset_completo(DIR_ANNOTAZIONI, lista_nomi_files_jpg: list[str], lista_nomi_files_txt: list[str],
                             dati: pd.DataFrame) -> pd.DataFrame:
    '''
    Si occupa di riempire il DataFrame con i soli dati di interesse.
    Args:
    DIR_ANNOTAZIONI: directory in cui sono salvati tutti i file di testo che annotano ogni singola immagine.
    lista_nomi_files_jpg: lista ordinata dei nomi di tutte le immagini presenti nel dataset.
    lista_nomi_files_txt: lista ordinata dei nomi di tutte i file di testo presenti nel dataset.
    dati: pd.DataFrame, dataframe pandas da usare per immagazzinare i dati.
    '''

    ## Definizione del dataframe
    # num_punti = 14
    # dati : pd.DataFrame = pd.DataFrame(
    #     columns=['path_img'] +
    #             [f'punto_{n}_{coord}' for n in range(1, num_punti+1) for coord in ('X', 'Y')] # +1 per comprendere anche il 14°-esimo punto nel range
    # )

    for t, i in zip(lista_nomi_files_txt,
                    lista_nomi_files_jpg):  # posso usare la zip dato che ho appurato che le due liste abbiano la medesima lunghezza
        try:

            ### Inizio col definirmi la variabile che rappresenta la singola riga del DF
            ##  La prima colonna deve avere il nome dell'immagine
            riga = {'path_img': i}  # costruisco il dizionario per questa riga

            ##  Le altre colonne, ciascuna deve avere la coordinata
            df_txt = pd.read_csv(f"{DIR_ANNOTAZIONI}/{t}", delimiter=',',
                                 header=0)  # prima colonna è quella delle intestazioni

            ##  Recupero le coordinate X ed Y rispettivamente
            X = df_txt['X'].values
            Y = df_txt['Y'].values

            riga.update({f'punto_{idx + 1}_X': x for idx, x in enumerate(X)})
            riga.update({f'punto_{idy + 1}_Y': y for idy, y in enumerate(Y)})

            ##  Aggiungo la riga al dataframe
            dati.loc[len(dati)] = riga

        except FileNotFoundError:
            print(f"Immagine non trovata per {t}. Salto.")
        except Exception as e:
            print(f"Errore durante l'elaborazione di {t}: {e}")

            # Restituisco il dataframe creato
    return dati


def main():
    """
    Funzione principale per il preprocessing.
    """

    print("\nInizio preprocessing del dataset...")

    ## La prima operazione è quella di unzippare il DS se è la prima volta
    unzip_dataset()


    ## Richiamo la funzione
    dati = prepara_dataset_completo(DATASET_PATH, lista_immagini, lista_files_txt, dati)

    # Salvataggio dei dati processati in file csv.
    # In questo modo posso leggere il DF direttamente.
    dati.to_csv(f"{DATA_DIR}/dataframe_master.csv", index=False)