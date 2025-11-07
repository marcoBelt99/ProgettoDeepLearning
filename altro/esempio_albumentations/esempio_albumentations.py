import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import pandas as pd
import numpy as np
from albumentations import Compose

from data.preprocessing.pre04_preparazione_dataframe import prepara_dataframe_completo
from configs.parametri_app import num_totale_punti


'''
Provo a:
- creare un dataset composto da due immagini e due relativi file txt di annotazione.
- crare delle augmentation sia delle immagini che dei relativi keypoints, tramite
  le trasformazioni offerte da Albumentations.
'''

### Seguo la documentazione di albumentations:
### https://albumentations.ai/docs/3-basic-usage/keypoint-augmentations/


def visualizzaza_immagine_con_keypoints(img, keypoints, titolo):

    '''
    Visualizza l'immagine e i suoi keypoints associati.

    Args:
        img: immagine in formato RGB (compatibile con matplotlib).
        keypoints: annotazioni delle coordinate (x,y) dell'immagine.
                   sono rappresentati mediante una lista di coppie.
    '''
    for kp in keypoints:
        cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)

    ## Visualizzo l'immagine con matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(titolo)
    plt.axis('off')
    plt.show()



'''
Costruzione delle trasformazioni desiderate (e sensate).
'''
transform : Compose = A.Compose([
    # A.RandomCrop(width=900, height=900),
    A.Resize(width=512, height=512), # Resize per portare ogni immagine a cui applico
            # le trasformazioni ad una stessa dimensione, utile come input per la rete neurale
            # che sceglierò di adottare.
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # La variabile transform contiene una funzione ( trasform(image, keypoints) ) a cui passerò sia l'immagine che i keypoints


def main():

    #######################################
    ######## (1) Preparazione dataset #####
    #######################################
    dati : pd.DataFrame = pd.DataFrame(columns=['path_img'] + # parto da 1 per escludere la colonna 'path_img'
                                               [f'punto_{n}_{coord}' for n in range(1, num_totale_punti + 1) for coord in
                                               ('X', 'Y')])  # +1 per comprendere anche il 14°-esimo punto nel range

    # (i dati si trovano nella current directory in questo caso)
    dati = prepara_dataframe_completo(os.getcwd(), os.listdir(os.getcwd()), dati)

    print("\nDataframe completo per i due campioni:\n", dati)


    # Dovrei ciclare per ogni riga del DF (in questo caso 2 righe)
    for i in range(0, dati.shape[0]):
        print(f"Immagine {i+1}")

        ## Estraggo la riga del DataFrame
        riga = dati.iloc[i]
        img_path = riga['path_img']

        ## Lettura immagine con opencv
        #  visto che devo visualizzare l'immagine con matplotlib
        #  converto da BGR (formato usato da opencv) in RGB (formato
        #  usato da matplotlib).
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Costruisco i keypoints prendendoli dal dataset
        keypoints = [ (riga[f'punto_{k}_X'], riga[f'punto_{k}_Y']) for k in range(1, num_totale_punti + 1)]

        # trasformo da lista python ad array numpy per vederli meglio su stdout
        print("\nKeypoints originali:\n", np.asarray(keypoints, dtype=np.float32))

        ## Visualizzazione d'esempio
        visualizzaza_immagine_con_keypoints(image, keypoints, f"Originale: {i + 1}")


        ###############################################################################
        ######## (2) Applico la funzione trasform() all'immagine e ai keypoints.  #####
        ###############################################################################
        transformed = transform(image=image, keypoints=keypoints)

        ## Recupero sia l'immagine aumentata che i relativi keypoints aumentati (ad essa associati)
        aug_image = transformed["image"]
        aug_keypoints = transformed["keypoints"]

        print("\nKeypoints trasformati:\n", aug_keypoints)

        ## Visualizzazione d'esempio
        visualizzaza_immagine_con_keypoints(aug_image, aug_keypoints, f"Aumentata: {i + 1}")

if __name__ == '__main__':
    main()