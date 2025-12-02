import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2


from configs.parametri_app import DATAFRAME_MASTER, DATASET_DIR, num_totale_punti, FILES
from utils.utils import fail, ok



def mostra_punti_annotati_random(df, etichette_punti , path_base=DATASET_DIR, n_immagini=10 ):
    """
    Mostra una griglia di immagini casuali con i punti sovrapposti e le relative sigle anatomiche corrette.

    Args:
        df: DataFrame Pandas contenente 'path_immagine' e colonne punto_<n>_<X|Y>
        etichette_punti: lista delle corrispondenti labels dei vari punti sotto forma di stringhe con le iniziali.
        path_base: percorso della cartella contenente le immagini. (Di default è DATASET_DIR)
        n_immagini: numero di immagini da visualizzare. (Di default = 10)
    """

    # Seleziono n immagini casuali
    sample_df = df.sample( n=n_immagini, random_state=random.randint(0,10000) ).reset_index(drop=True)

    # Calcola le dimensioni della griglia
    cols = 2
    rows = (n_immagini + cols - 1) // cols # // mi calcola la divisione senza resto

    # Ho bisogno di più sotto-plots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()

    for idx, (i, row) in enumerate(sample_df.iterrows()): # itero sulle righe del DF => ogni riga == un'immagine
        img_path = f"{path_base}/{row['path_img']}"

        # Leggo l'immagine con OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"Immagine non trovata: {img_path}")
            continue

        # Recupero altezza e larghezza rispettivamente dell'immagine
        height, width = img.shape[0], img.shape[1]

        # Disegno i punti
        for j in range(1, num_totale_punti + 1): # Scorro sulle colonne, a partire dalla 2° che ha il primo punto

            # Unpacking delle coordinate: Open CV accetta le coordinate come numeri interi,
            # altrimenti non riesce a calcolare il centro del cerchio
            x, y = int( row[f"punto_{j}_X"]  ), int( row[f"punto_{j}_Y"]   )

            # print('\n')
            # print("X: ", x, " Y: ", y )

            cv2.circle( img, (x, y), 8, (0, 0, 255), -1 )   # cerchio rosso
            # cv2.circle(, img, coord, raggio, colore, -1) # coord valeva 5

            # Scrivo il testo
            cv2.putText(
                img,
                etichette_punti[j-1],
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, # prima era 0.6
                (255, 255, 0),  # giallo
                2
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axes[idx]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{row['path_img']} (idx={df.index[df['path_img']==row['path_img']][0]})")


    # Nascondo eventuali assi vuoti
    for k in range(idx + 1, len(axes)):
        axes[k].axis("off")

    # Mostro a video
    plt.tight_layout()
    plt.show()



def check_missing_values():
    missing_stats = []

    lista_nomi_files_txt : list[str] = sorted([t for t in FILES if t.endswith(".txt")], key=lambda nf: int(nf.split('.')[0]))

    for txt_file in lista_nomi_files_txt:
        file_path = os.path.join(DATASET_DIR, txt_file)
        if not os.path.exists(file_path):
            print(f"File non trovato: {file_path}")
            continue

        # Leggo il CSV
        df: pd.DataFrame = pd.read_csv(f"{DATASET_DIR}/{txt_file}", delimiter=',',
                                           header=0)  # prima colonna è quella delle intestazioni

        # Prendo solo le colonne X e Y
        coords = df[['X', 'Y']]

        # Controllo valori anomali
        invalid = (coords == 0).any(axis=1) | coords.isna().any(axis=1)
        missing_stats.append({
            'file': txt_file,
            'punti_totali': len(coords),
            'punti_invalidi': invalid.sum()
        })

    return pd.DataFrame(missing_stats)




def dimensioni_medie_immagini(df : pd.DataFrame, path_base=DATASET_DIR) -> tuple[float, float]:
    """
    Restituisce rispettivamente (larghezza, altezza) medie del dataset (in formato dataframe) passato
    come argomento.

    Args:
        df: dataframe del dataset su cui operare.
    Returns:
        (larghezza media, altezza media)
    """
    larghezze = []
    altezze = []

    for _, row in df.iterrows():
        img_path = f"{path_base}/{row['path_img']}"
        img = cv2.imread(img_path)

        if img is None:
            print(f"Immagine non trovata: {img_path}")
            continue

        h, w = img.shape[:2]  # più compatto di: height, width = img.shape[0], img.shape[1]

        larghezze.append(w)
        altezze.append(h)

    if not larghezze or not altezze:
        raise ValueError("Nessuna immagine valida trovata.")

    return np.mean(larghezze), np.mean(altezze)




def main():

    ## Se il dataframe master esiste
    if os.path.exists(DATAFRAME_MASTER):

        ## Lo prendo
        dati : pd.DataFrame = pd.read_csv(DATAFRAME_MASTER)

        ## Visualizzo le prime 4 righe
        print("\nPrime 4 righe del dataframe master:")
        print( dati.head(4) )

        ## Visualizzo le ultime 4 righe
        print("\nUltime 4 righe del dataframe master:")
        print( dati.tail(4) )

        ## Verifico se ci sono dei valori mancanti o anomali
        print("\nVerifico se ci sono dei valori mancanti o anomali:")
        print( dati.isna().sum() )

        ## Vedo alcune statistiche riassuntive del Dataset
        print("\nVedo alcune statistiche riassuntive del Dataset:")

        print( dati.describe() )


        '''
        Noto che:
        
        Ho per ogni colonna esattamente 1599 valori di conteggio, quindi non ci sono valori mancanti.
        Leggendo i valori min e max mi sembra di capire che le X variano fino a circa 5000 px e le Y fino a circa 4200 px. Le dimensioni sono piuttosto grandi, e come visto prima ogni immagine ha dimensioni diverse.
        Dalla media dei punti vedo che ad esempio:
        punto_1 è intorno a (978, 713)
        punto_14 è intorno a (1373, 1291)
        punto_4 è intorno a (1666, 1072)
        I punti sembrano distribuiti su un'area che si sposta verso destra e verso il basso (come atteso per i punti del cranio delle cefalometrie)
        
        std: la deviazione standard mi indica come variano i punti.
        alcuni punti, come punto_2, punto_4, punto_11 hanno std nelle ascisse > 700 px.
        altri punti, come il punto_1 hanno std per le Y < 330px
        Quindi, alcuni punti variano molto di più, probabilmente perchè:
        
        sono in zone più mobili (come mandibola o estremità)
        oppure le immagini non sono perfettamente allineate
        '''

        etichette_punti : list[str] = ["S", "N", "Sna", "Snp", "A", "B", "pg",
                                      "Gn", "Go", "U1r", "U1t", "L1r", "L1t", "mes"]

        mostra_punti_annotati_random(
            dati,
            etichette_punti,
            n_immagini=4  # quante immagini mostrare
        )
    else:
        print(f"{fail}Dataframe in csv non trovato.")



    ## Verifico se esistono valori mancanti o problematici
    #  settando a True la seguente variabile
    check_valori_mancanti = False

    if check_valori_mancanti:

        df_valori_mancanti = check_missing_values()
        print( "Dataframe valori mancanti:\n", df_valori_mancanti )
        print( "Numero valori mancanti: ", df_valori_mancanti['punti_invalidi'].sum() )

    '''
    Noto che non ne ho da nessuna parte di valori mancanti. Ottimo segno.
    '''

    larghezza_media, altezza_media = dimensioni_medie_immagini(dati)

    print(f"\nLarghezza media: {larghezza_media}\t|Altezza media: {altezza_media}\n")


if __name__ == "__main__":

    main()