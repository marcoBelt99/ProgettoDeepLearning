import torch
from torch.utils.data import Dataset, random_split

import cv2
import pandas as pd
import numpy as np
import os

from src.data.dataset.trasformazioni import data_transforms

class RepereKeypointsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, img_size: int = 512, augment: bool = True):
        """
        Dataset per la regressione di keypoints su immagini di cefalometrie.

        Args:
            dataframe (pd.DataFrame): Contiene path_img e colonne punto_X/Y.
            img_dir (str): Directory delle immagini.
            img_size (int): Dimensione a cui ridimensionare le immagini. Di default a 512
            augment (bool): Se True, applica data augmentation.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size

        # Memorizzo le colonne dei punti
        # Posso pensare di prendere solo le coordinate dei punti desiderati,
        # basta usare la funzione corretta (prendi spunto dal colab).
        self.keypoint_cols = [c for c in dataframe.columns if c.startswith("punto_")]

        # Applico le giuste trasformazioni
        self.transform = self._get_transforms(augment)

    def _get_transforms(self, augment: bool):
        """Restituisce la pipeline di trasformazioni Albumentations."""
        return data_transforms['train' if augment else 'test']


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.img_dir, row["path_img"])

        # Carica immagine
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Estrazione coordinate (in pixel)
        coords = row[self.keypoint_cols].values.astype("float32") # provare il float64
        keypoints = [ (coords[i], coords[i + 1] ) for i in range(0, len(coords), 2) ]

        # Applico le trasformazioni
        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed["image"]
        keypoints = np.array(transformed["keypoints"], dtype="float32") # provare con il float64

        # Normalizzo le coordinate in [0,1] (dopo il resize)
        keypoints[:, 0] /= self.img_size  # X
        keypoints[:, 1] /= self.img_size  # Y

        # Flatten → [x1, y1, x2, y2, ...]
        keypoints = torch.tensor(keypoints.flatten(), dtype=torch.float32)

        return image, keypoints


# Funzione di utilità per split train/test
def crea_split_dataset(dataframe: pd.DataFrame, img_dir: str, img_size: int = 512, train_ratio: float = 0.8):
    """
    Divide il dataset in train e test in modo casuale.
    """
    full_dataset = RepereKeypointsDataset(dataframe, img_dir=img_dir, img_size=img_size, augment=True)

    n_train = int(len(full_dataset) * train_ratio)
    n_test = len(full_dataset) - n_train

    train_set, test_set = random_split(full_dataset, [n_train, n_test])

    # Cambia il flag augment=False nel test set
    test_set.dataset.transform = test_set.dataset._get_transforms(augment=False)

    return train_set, test_set
