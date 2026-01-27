import torch
from albumentations import Compose
from torch.utils.data import Dataset

import cv2
import pandas as pd
import numpy as np
import os

from models.resnet18.trasformazioni import resnet18_data_transforms

class RepereKeypointsDataset(Dataset):


    def __init__(self, dataframe: pd.DataFrame, img_dir: str, trasformazioni : dict[str, Compose],
                 img_size: int = 224, gruppo_punti: list[int] = None):
        """
        Dataset per la regressione di keypoints su immagini di cefalometrie.

        Args:
            dataframe (pd.DataFrame): Contiene path_img e colonne punto_X/Y.
            img_dir (str): Directory delle immagini.
            trasformazioni (Compose) : trasformazioni specifiche del modello che si sta usando.
            img_size (int): Dimensione a cui ridimensionare le immagini. Di default l'ho messa a 512, devo ovviamente renderla aderente al modello che uso
            gruppo_punti (list[int]): eventuale specifico gruppo di punti da considerare. Se non presente (None) allora vengono considerate tutte le colonne
                                      del dataset. Se presente, vengono considerate solo un sottoinsieme delle colonne.
                                      [Questo al fine di non creare più tutti e 4 i files csv separati, a partire dal csv master].
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size

        ## Memorizzo le colonne dei punti di interesse
        if gruppo_punti is not None:

            # TODO: filtro le sole colonne del gruppo di punti di interesse
            # Converto gli indici 0-based in numeri punto 1-based
            self.keypoint_cols = []
            for i in gruppo_punti:
                punto_num = i + 1  # +1 perchè i punti vanno da 0 a 13 (e non da 1 a 14 come richiesto)
                self.keypoint_cols.extend([
                    f'punto_{punto_num}_X',
                    f'punto_{punto_num}_Y'
                ])
        else:
            self.keypoint_cols = [c for c in dataframe.columns if c.startswith("punto_")]

        # Applico le giuste trasformazioni
        self.transform = trasformazioni


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.img_dir, row["path_img"])

        # Carica immagine
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Estrazione coordinate (in pixel)
        coords = row[self.keypoint_cols].values.astype("float32") # TODO: provare il float64
        keypoints = [ (coords[i], coords[i + 1] ) for i in range(0, len(coords), 2) ]

        # TODO: Applico le trasformazioni sia alle immagini, sia ai keypoints grazie ad albumentations
        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed["image"]

        # Converto i keypoints in array prima della normalizzazione
        keypoints = np.array(transformed["keypoints"], dtype="float32") # provare con il float64

        ## Normalizzo le coordinate in [0,1] (dopo il resize)
        #  dovrebbe far convergere molto meglio il MSE
        keypoints[:, 0] /= self.img_size  # X
        keypoints[:, 1] /= self.img_size  # Y

        # Flatten ==> [x1, y1, x2, y2, ...]
        keypoints = torch.tensor(keypoints.flatten(), dtype=torch.float32)

        # Restituisco una tupla (immagine, keypoints) pronta per il DataLoader
        return image, keypoints




def crea_split_dataset(dataframe: pd.DataFrame,
                       img_dir: str,
                       img_size: int = 512,
                       train_ratio: float = 0.8,
                        gruppo_punti: list[int] = None
                       ):
    """
    Divide il dataset in train e test in modo casuale.

    Args:
        dataframe: struttura dati per immagazzinare il dataset
        img_size: dimensione delle immagini (Se previsto, di default uso quella del modello pre-addestrato che scego)
        train_ratio (float): % di split per il training set.
        train_ratio (float): % di split per il training set.
        gruppo_punti (list[int]): eventuale specifico gruppo di punti da considerare. Se non presente (None) allora vengono considerate tutte le colonne
                                  del dataset. Se presente, vengono considerate solo un sottoinsieme delle colonne.

    """



    ## Splitting
    train_df = dataframe.sample(frac=train_ratio, random_state=42)

    test_df = dataframe.drop(train_df.index)

    ## Assegnazione delle giuste trasformazioni ai due dataset di training e di test
    train_set = RepereKeypointsDataset(train_df, img_dir,
                                       trasformazioni=resnet18_data_transforms['train'],
                                       img_size=img_size,
                                       gruppo_punti=gruppo_punti)

    test_set = RepereKeypointsDataset(test_df, img_dir,
                                      trasformazioni=resnet18_data_transforms['test'],
                                      img_size=img_size,
                                      gruppo_punti=gruppo_punti)

    return train_set, test_set
