import pandas as pd

# Parametri da usare in giro per l'app
from configs.parametri_app import *
from utils.utils import get_num_files


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

    for t, i in zip(lista_nomi_files_txt, #  posso usare la zip dato che ho appurato che le due liste
                    lista_nomi_files_jpg): # abbiano la medesima lunghezza
        try:

            ### Inizio col definirmi la variabile che rappresenta la singola riga del DF
            ##  La prima colonna deve avere il nome dell'immagine
            riga : dict[str, str] = {'path_img': i}  # costruisco il dizionario per questa riga

            ##  Le altre colonne, ciascuna deve avere la coordinata
            df_txt : pd.DataFrame = pd.read_csv(f"{DIR_ANNOTAZIONI}/{t}", delimiter=',',
                                 header=0)  # prima colonna è quella delle intestazioni

            ##  Recupero le coordinate X ed Y rispettivamente
            X : pd.Series = df_txt['X'].values
            Y : pd.Series = df_txt['Y'].values

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

    #### Definisco un nuovo DataFrame
    num_punti = 14
    dati: pd.DataFrame = pd.DataFrame(columns=['path_img'] +
                                              [f'punto_{n}_{coord}' for n in range(1, num_punti + 1) for coord in
                                               ('X', 'Y')])  # +1 per comprendere anche il 14°-esimo punto nel range

    ### Richiamo la funzione, solo se non esiste e contiene il giusto numero di righe
    # dati = prepara_dataset_completo( DATASET_PATH, lista_nomi_files_jpg, lista_nomi_files_txt, dati )
    NUM_FILE_JPG: int = get_num_files("jpg")


    if not os.path.exists(DATAFRAME_MASTER):
        # Il giusto numero di righe me lo dice la shape
        if dati.shape[0] != NUM_FILE_JPG:

            # Ottengo la lista di files e di immagini
            lista_files_txt : list[str] = sorted([t for t in FILES if t.endswith(".txt")], key=lambda nf: int(nf.split('.')[0]))
            lista_immagini : list[str] = sorted([i for i in FILES if i.endswith(".jpg")], key=lambda nf: int(nf.split('.')[0]))

            ## Richiamo la funzione per preparare il dataframe
            dati = prepara_dataset_completo(DATASET_DIR, lista_immagini, lista_files_txt, dati)

            # Salvataggio dei dati processati in file csv.
            # In questo modo posso leggere il DF direttamente.
            dati.to_csv(f"{DATA_DIR}/dataframe_master.csv", index=False)
    else:
        print(f"Non ho preparato il dataframe perchè era gia' presente il file {DATAFRAME_MASTER}")

if __name__ == "__main__":
    main()