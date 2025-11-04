import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import cv2

# Parametri da usare in giro per l'app
from configs.parametri_app import *


from utils.utils import *
import random

'''
Analisi di un'immagine e del relativo file di testo ad essa associato.
'''


########################
# Analisi immagine
########################


def analizza_immagine(immagine, campione_random):
    '''
    Funzione principale di analisi dell'immagine.
    '''

    #### Apertura ed operazioni su immagine
    if os.path.exists(immagine):
        try:
            print(f"\n{'#' * 30}\nAnalisi dell'immagine numero: {campione_random}\n{'#' * 30}\n")

            ##### Analizzo l'immmagine con PIL #####
            analizza_immagine_PIL(immagine)

            print(f"{'-'*30}")

            ### Analizzo l'immmagine con Open CV ###
            analizza_immagine_CV2(immagine)

        except Exception as e:
            print(f"Errore su immagine {campione_random}: {e}")
    else:
        print(f"L'immagine {campione_random}.jpg non esiste.")



def controlla_modalita_colore(immagine_PIL):
  '''
  Tramite PIL vado a controllare il numero di canali dell'immagine,
  per verificare se essa è a colori (3 canali) o in scala di grigi (1 canale).
  '''
  if immagine_PIL.mode == 'L': # grayscale
      print("L'immagine è in scala di grigi.\n")
  elif immagine_PIL.mode == 'RGB':
      print("L'immagine è a colori (RGB).\n")
  elif immagine_PIL.mode == 'RGBA':
      print("L'immagine è a colori (RGBA).\n")
  else:
      print(f"La modalità dell'immagine è: {immagine_PIL.mode}\n")


def analizza_immagine_PIL(immagine):
    '''
    Analisi dell'immagine con PIL.
    '''

    print("\nAnalisi dell'immagine con PIL:\n")

    img = Image.open(immagine)

    # NB: img.size mi da rispettivamente la tupla: (width, height) in pixels
    print(f"Dimensioni immagine: {img.size}\n")

    # Quindi provo a mostrarla con matplotlib (perchè in teoria PIL non funziona qui su colab)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')  # Rimuove gli assi
    plt.show()

    # Chiamata a funzione di check grayscale o RGB
    controlla_modalita_colore(img)

def analizza_immagine_CV2(immagine):
    print("\nAnalisi dell'immagine con OpenCV:\n")

    # 1) La apro
    img_opencv = cv2.imread(immagine)

    # L'immagine è grayscale o a colori (RGB ad esempio)? lo dovrei vedere dalla shape
    print(f'Dimensione array OpenCV: {img_opencv.shape}') # NB: (height, weigth)
    # Se shape è (h, w, 3) --> RGB/BGR
    # Se shape è (h, w) o (h, w, 1) --> Grayscale
    # Se shape è (h, w, 4) --> RGBA

    if len(img_opencv.shape) == 3:
        if img_opencv.shape[2] == 3:
            print("L'immagine è a COLORI (BGR - 3 canali)")
        elif img_opencv.shape[2] == 4:
            print("L'immagine è RGBA (4 canali)")
    else:
        print("L'immagine è in GRAYSCALE (1 canale)")

    # 2) Ne divido i canali
    canale_blu, canale_verde, canale_rosso = cv2.split(img_opencv)
    # canale_blu, canale_verde, canale_rosso = cv2.split(immagine)

    # 3) Visualizzo i canali
    print("\nCanali dell'immagine:\n")

    # Visualizzo i canali
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(canale_blu)
    plt.title('Canale Blu')

    plt.subplot(1, 3, 2)
    plt.imshow(canale_verde)
    plt.title('Canale Verde')

    plt.subplot(1, 3, 3)
    plt.imshow(canale_rosso)
    plt.title('Canale Rosso')

    plt.show()

    print("\nVisualizzazione Istogramma per ciascun canale:\n")

    # Calcolo gli istogrammi per ciascun canale
    hist_blu = cv2.calcHist([img_opencv], [0], None, [256], [0, 256])
    hist_verde = cv2.calcHist([img_opencv], [1], None, [256], [0, 256])
    hist_rosso = cv2.calcHist([img_opencv], [2], None, [256], [0, 256])

    # Visualizzo gli istogrammi
    plt.figure(figsize=(8, 6))

    plt.plot(hist_blu, color='blue', label='Blu')
    plt.plot(hist_verde, color='green', label='Verde')
    plt.plot(hist_rosso, color='red', label='Rosso')

    plt.title('Istogrammi dei Canali RGB')
    plt.xlabel('Intensità')
    plt.ylabel('Numero di Pixel')
    plt.legend()

    plt.show()




####################################
# Analisi del relativo file di testo
####################################

def analizza_file_di_testo(annotazione, campione_random):
    '''
    Analizza il file di testo che annota la singola immagine
    '''
    if os.path.exists( annotazione ):
      try:
          ## Mi salvo in un DataFrame pandas un SINGOLO file di testo per analizzarne / visualizzarne la struttura
          print(f"\n{'#'*30}\nAnalisi del relativo file di testo numero: {campione_random}\n{'#' * 30}\n")

          # Ho notato che la prima riga di ogni file di testo funge da intestazione
          # data = pd.read_csv(sample_annotation, delimiter=',', header=None)
          # Pertanto, con header=0 sto indicando che la prima riga del file è l'intestazione
          struttura_file = pd.read_csv(annotazione, delimiter=',', header=0)

          # Colonne di un singolo file
          struttura_file.columns = ['Index', 'Area', 'Mean', 'Min', 'Max', 'X', 'Y']

          print(f"\n{'#'*30}\nPrime 10 righe del dataframe del singolo file:\n{'#'*30}\n")
          print( struttura_file.head(20) ) # stampo le prime 10 righe

          print(f"\n{'#'*30}\nTipo di dato delle varie colonne:\n{'#'*30}\n")
          print(struttura_file.dtypes) # vedo il tipo di dato delle varie colonne che costituiscono il dataset

          print(f"\n{'#'*30}\nStatistiche riassuntive:\n{'#'*30}\n")
          print( struttura_file.describe() )  # Statistiche riassuntive

          print(f"\n{'#'*30}\nMedia delle coordinate:\n{'#'*30}\n")
          print( struttura_file['X'].mean(), struttura_file['Y'].mean() )  # Media delle coordinate (lavoro per colonne)

      except Exception as e:
        print(f"Errore su file {campione_random}: {e}")
    else:
        print(f"Il file {campione_random}.txt non esiste.")



def main():
    num_file_jpg: int = get_num_files("jpg")

    # Per visualizzare il singolo file e la singola immagine ad esso associata,
    # prendo un campione a caso
    campione_random = random.randint(1, num_file_jpg)
    # (Noto che per n=1237 non sono presenti nè l'img, nè il suo txt)

    ## Costruisco il path (dinamico), a seconda della singola immagine campione che prelevo dentro il DS
    immagine = os.path.join(DATASET_DIR, str(campione_random) + '.jpg')

    # Do il nome del path per trovare il singolo file di testo dentro il DS
    annotazione = os.path.join(DATASET_DIR, str(campione_random) + '.txt')

    ## Chiamata alle funzioni
    analizza_immagine(immagine, campione_random)
    analizza_file_di_testo(annotazione, campione_random)


if __name__ == "__main__":
    main()

