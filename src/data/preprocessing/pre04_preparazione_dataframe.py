import pandas as pd
from sklearn.utils import deprecated

# Parametri da usare in giro per l'app
from configs.parametri_app import *
from models.resnet18.parametri_modello import NUM_TOTALE_PUNTI
from utils.utils import get_num_files, ok, fail


def prepara_dataframe_completo(DIR_ANNOTAZIONI, LISTA_FILES,
                               dati : pd.DataFrame) -> pd.DataFrame:
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


# TODO: deprecato
# Creo diversi DF ognuno correlato ad uno specifico gruppo di punti
def crea_sottodataframe_per_gruppo(dati_completi, nome_gruppo) -> pd.DataFrame:
    """


    Args:
      dati_completi: dataframe "master" di partenza, completo di tutti i punti.
      nome_gruppo: chiave del dizionario che ha per valori la lista dei punti di interesse per ognuno dei raggruppamenti.
    """

    # Inizio creando la lista di colonne desiderate, che è composta inzialmente dalla sola
    # colonna 'path_img'
    colonne_selezionate = ['path_img']  # Mantengo sempre il path dell'immagine

    # Estendo tale lista, seleziono tutte le colonne desiderate relative ai punti
    # dello specifico gruppo (sia X che Y).
    for idx in RAGGRUPPAMENTI[nome_gruppo]:
        punto_numero = idx + 1  # Converto da indice 0-based a numero punto 1-based
        colonne_selezionate.extend( [f'punto_{punto_numero}_X', f'punto_{punto_numero}_Y'] )

    # Crea il sotto-dataframe, selezionando solo le colonne di interesse, e facendo una copia di quello originale
    dataframe_gruppo = dati_completi[colonne_selezionate].copy()

    return dataframe_gruppo

# TODO: deprecato
def prepara_dataframe_gruppo(df_completo : pd.DataFrame, nome_gruppo: str) -> pd.DataFrame:
    '''
    Crea sotto-dataframe per il raggruppamento specificato.
    Questa procedura funziona per tutti i gruppi di punti. Infatti, basta che passo la chiave del gruppo di punti
    di cui voglio creare il dataframe.


    Args:
        df_completo:  dataframe di partenza, da cui tenere le sole colonne di interesse per il gruppo di punti.
        nome_gruppo: chiave del nome del gruppo di punti di cui si vuol creare il dataframe

    '''

    # Per poter creare il sotto-dataframe specifico per quel gruppo di punti, non ha senso che
    # il dataframe master sia vuoto
    try:
        # if not df_completo.empty:
        if os.path.exists(DATAFRAME_MASTER):
            # Mi serve leggere il csv per poter passare i dati ai sottodataframe
            df_completo = pd.read_csv(DATAFRAME_MASTER)
            punti : list[int] = RAGGRUPPAMENTI[nome_gruppo]
            colonne : list[str] = ['path_img'] + [coord for i in punti for coord in (f'punto_{i + 1}_X', f'punto_{i + 1}_Y')]
            print(f"{ok} Dataframe per il gruppo {nome_gruppo} creato correttamente. ")

            ## Ritorno il dataframe filtrato, con le sole colonne di interesse
            return df_completo[colonne]
        else:
            # raise ValueError(f"\n{fail}Il Dataframe master non puo' essere vuoto, altrimenti non ha senso filtrare il dataframe dello specifico gruppo.")
            raise ValueError(f"\n{fail}Il Dataframe master deve esistere, altrimenti non ha senso filtrare il dataframe dello specifico gruppo.")
    except Exception as e:
        print(f"Errore durante la preparazione del DataFrame per {nome_gruppo}: {e}")
        return None



def main():

    #######################################################
    ### PREPARAZIONE DATAFRAME COMPLETO (MASTER) ##########
    #######################################################
    #### Definisco un nuovo DataFrame
    dati : pd.DataFrame = pd.DataFrame(columns=['path_img'] +
                                               [f'punto_{n}_{coord}' for n in range(1, NUM_TOTALE_PUNTI + 1) for coord in
                                               ('X', 'Y')])  # il +1 è per comprendere anche il 14°-esimo punto nel range

    # Ottengo il numero totale di immagini presenti dentro al cartella FILES
    num_file_jpg : int = get_num_files("jpg")

    ### Richiamo la funzione, solo se non esiste e solo se contiene il giusto numero di righe
    if not os.path.exists(DATAFRAME_MASTER):
        # Il giusto numero di righe me lo dice la shape
        if dati.shape[0] != num_file_jpg:

            ## Richiamo la funzione per preparare il dataframe completo
            dati = prepara_dataframe_completo(DATASET_DIR, FILES, dati)

            # Salvataggio dei dati processati in file csv.
            # In questo modo posso leggere il DF direttamente.
            dati.to_csv(f"{DATA_DIR}/dataframe_master.csv", index=False)

            print(f"\n{ok} File csv contenente il dataset completo salvato correttamente in: {DATA_DIR}/dataframe_master.csv\n")
    else:
        print(f"{fail} Non ho preparato il dataframe perchè era gia' presente il file {DATAFRAME_MASTER}.")


    print("\nDATI COMPLETI:\n", dati.head(3))


    ######################################################
    ######## TODO: depracato -> PREPARAZIONE DATAFRAMEs PARZIALI ############
    ######################################################

    dict_dataframe_parziali : dict[str, pd.DataFrame] = {}

    conta_gruppi = 0
    for path_df in [DATAFRAME_GRUPPO_1, DATAFRAME_GRUPPO_2, DATAFRAME_GRUPPO_3, DATAFRAME_GRUPPO_4]:
        if not os.path.exists(path_df):
            conta_gruppi += 1
            dict_dataframe_parziali[f"GRUPPO{conta_gruppi}"] =  prepara_dataframe_gruppo(dati, f"GRUPPO{conta_gruppi}")
            print(f"\nGRUPPO {conta_gruppi}:\n", dict_dataframe_parziali[f"GRUPPO{conta_gruppi}"].head(3))
        else:
            print(f"{fail} Non ho preparato il dataframe parziale perchè era gia' presente il file {path_df}.")



    # lista_dati_raggruppamenti: list[pd.DataFrame] = [dati_gruppo_1, dati_gruppo_2, dati_gruppo_3, dati_gruppo_4]
    lista_dati_raggruppamenti : list[pd.DataFrame] = list( dict_dataframe_parziali.values() )

    ## Lista di bool come "sentinella" per vedere se sono stati creati tutti i gruppi
    lista_files_dati_gruppi_creati : list[bool] = [os.path.exists(os.path.join(DATA_DIR,  f'dati_gruppo_{i}.csv')) for i in
                                                  range(1, len(lista_dati_raggruppamenti) + 1)]

    # Se ho creato correttamente tutti i file csv dei gruppi
    if not all(lista_files_dati_gruppi_creati):
        print("\n")
        for i, df_gruppo in zip(range(1, len(lista_dati_raggruppamenti) + 1), lista_dati_raggruppamenti):
            df_gruppo.to_csv(os.path.join(DATA_DIR , f'dati_gruppo_{i}.csv'), index=False)
            print(f"{ok} Salvato file: dati_gruppo{i}.csv")
    else:
        print(f"{fail} Gruppi non creati correttamente.")

if __name__ == "__main__":
    main()