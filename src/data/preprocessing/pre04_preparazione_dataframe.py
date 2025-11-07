import pandas as pd

# Parametri da usare in giro per l'app
from configs.parametri_app import *
from utils.utils import get_num_files, ok, fail


def prepara_dataframe_completo(DIR_ANNOTAZIONI, LISTA_FILES,
                               dati: pd.DataFrame) -> pd.DataFrame:
    '''
    Si occupa di riempire il DataFrame con i soli dati di interesse.

    Args:
        DIR_ANNOTAZIONI: directory in cui sono salvati tutti i file di testo che annotano ogni singola immagine.
        LISTA_FILES: Lista di tutti i nomi dei files presenti dentro alla cartella in cui e' contenuto il dataset. (L'ho messo come parametro, così posso riutilizzare questa funzione per creare altri dataset che fanno riferimento a cartelle differenti).
        dati: pd.DataFrame, dataframe pandas da usare per immagazzinare i dati.
    '''

    ## Definizione del dataframe
    # num_punti = 14
    # dati : pd.DataFrame = pd.DataFrame(
    #     columns=['path_img'] +
    #             [f'punto_{n}_{coord}' for n in range(1, num_punti+1) for coord in ('X', 'Y')] # +1 per comprendere anche il 14°-esimo punto nel range
    # )

    ## Ottengo le due liste: rispettivamente, di tutti i nomi di files txt e di immagini presenti nel dataset, ordinate in ordine crescente.
    lista_nomi_files_txt: list[str] = sorted([t for t in LISTA_FILES if t.endswith(".txt")], key=lambda nf: int(nf.split('.')[0]))
    lista_nomi_files_jpg: list[str] = sorted([i for i in LISTA_FILES if i.endswith(".jpg")], key=lambda nf: int(nf.split('.')[0]))

    ## Uso le due liste per costruirmi il dataframe
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

            ## Scrivo il valore X, e dopo il valore Y
            riga.update({f'punto_{idx + 1}_X': x for idx, x in enumerate(X)})
            riga.update({f'punto_{idy + 1}_Y': y for idy, y in enumerate(Y)})

            ##  Aggiungo la riga al dataframe
            dati.loc[len(dati)] = riga

        except FileNotFoundError:
            print(f"{fail}Immagine non trovata per {t}. Salto.")
        except Exception as e:
            print(f"{fail}Errore durante l'elaborazione di {t}: {e}")

    # Restituisco il dataframe creato
    return dati

def main():

    #### Definisco un nuovo DataFrame
    num_punti = 14
    dati: pd.DataFrame = pd.DataFrame(columns=['path_img'] +
                                              [f'punto_{n}_{coord}' for n in range(1, num_punti + 1) for coord in
                                               ('X', 'Y')])  # +1 per comprendere anche il 14°-esimo punto nel range

    # Ottengo il numero totale di immagini presenti dentro al cartella FILES
    num_file_jpg: int = get_num_files("jpg")

    ### Richiamo la funzione, solo se non esiste e contiene il giusto numero di righe
    if not os.path.exists(DATAFRAME_MASTER):
        # Il giusto numero di righe me lo dice la shape
        if dati.shape[0] != num_file_jpg:


            ## Richiamo la funzione per preparare il dataframe
            dati = prepara_dataframe_completo(DATASET_DIR, FILES, dati)

            # Salvataggio dei dati processati in file csv.
            # In questo modo posso leggere il DF direttamente.
            dati.to_csv(f"{DATA_DIR}/dataframe_master.csv", index=False)

            print(f"\n{ok}File csv contenente il dataset completo salvato correttamente in: {DATA_DIR}/dataframe_master.csv\n")
    else:
        print(f"{fail}Non ho preparato il dataframe perchè era gia' presente il file {DATAFRAME_MASTER}")

if __name__ == "__main__":
    main()