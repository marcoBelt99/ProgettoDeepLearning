import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


class KeypointDataset(Dataset):
    """
    Dataset custom per la regressione di keypoint.
    """

    def __init__(self, dataframe, transform=None, target_size=(224, 224)):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame con colonne 'image_path' e 'keypoints'
            transform (callable, optional): Trasformazioni da applicare all'immagine
            target_size (tuple): Dimensione target per il ridimensionamento (width, height)
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_size = target_size

        # Calcola il numero massimo di keypoint per padding
        self.max_keypoints = self.df['num_keypoints'].max()
        print(f"Dataset inizializzato con {len(self.df)} campioni")
        print(f"Numero massimo di keypoint: {self.max_keypoints}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Restituisce un campione dal dataset.

        Returns:
            tuple: (image_tensor, keypoints_tensor)
        """
        sample = self.df.iloc[idx]

        # Carica l'immagine
        image_path = sample['../../data/dataset/2.jpg']
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)

        # Ottieni i keypoint originali
        keypoints = sample['keypoints']  # Lista di tuple (x, y)

        # Calcola i fattori di scala per il ridimensionamento
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]

        # Ridimensiona i keypoint in base alla nuova dimensione dell'immagine
        scaled_keypoints = []
        for x, y in keypoints:
            new_x = x * scale_x
            new_y = y * scale_y
            scaled_keypoints.append([new_x, new_y])

        # Padding per avere sempre lo stesso numero di keypoint
        while len(scaled_keypoints) < self.max_keypoints:
            scaled_keypoints.append([-1, -1])  # Valore sentinella per keypoint mancanti

        # Converte in tensore
        keypoints_tensor = torch.tensor(scaled_keypoints, dtype=torch.float32)
        # Flatten per avere shape [max_keypoints * 2]
        keypoints_tensor = keypoints_tensor.flatten()

        # Applica le trasformazioni all'immagine
        if self.transform:
            image = self.transform(image)

        return image, keypoints_tensor


class KeypointDatasetFixed(Dataset):
    """
    Versione semplificata per dataset con numero fisso di keypoint.
    """

    def __init__(self, dataframe, num_keypoints, transform=None, target_size=(224, 224)):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame con colonne 'image_path' e 'keypoints'
            num_keypoints (int): Numero fisso di keypoint per ogni immagine
            transform (callable, optional): Trasformazioni da applicare all'immagine
            target_size (tuple): Dimensione target per il ridimensionamento
        """
        # Filtra solo le immagini con il numero corretto di keypoint
        self.df = dataframe[dataframe['num_keypoints'] == num_keypoints].reset_index(drop=True)
        self.num_keypoints = num_keypoints
        self.transform = transform
        self.target_size = target_size

        print(f"Dataset filtrato: {len(self.df)} campioni con {num_keypoints} keypoint")

        if len(self.df) == 0:
            raise ValueError(f"Nessun campione trovato con {num_keypoints} keypoint!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        # Carica immagine
        image_path = sample['../../data/dataset/2.jpg']
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Keypoint originali
        keypoints = sample['keypoints']

        # Scala keypoint
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]

        scaled_keypoints = []
        for x, y in keypoints:
            scaled_keypoints.extend([x * scale_x, y * scale_y])

        # Converte in tensore
        keypoints_tensor = torch.tensor(scaled_keypoints, dtype=torch.float32)

        # Trasformazioni immagine
        if self.transform:
            image = self.transform(image)

        return image, keypoints_tensor


def get_transforms(is_training=True, target_size=(224, 224)):
    """
    Restituisce le trasformazioni appropriate per training o validation.

    Args:
        is_training (bool): Se True, include data augmentation
        target_size (tuple): Dimensione target dell'immagine

    Returns:
        torchvision.transforms.Compose: Trasformazioni composte
    """

    if is_training:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.3),  # Attenzione: richiede di flippare anche i keypoint!
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    return transform


def create_dataloaders(df, batch_size=32, train_ratio=0.8, num_keypoints=None):
    """
    Crea DataLoader per training e validation.

    Args:
        df (pd.DataFrame): DataFrame del dataset
        batch_size (int): Dimensione del batch
        train_ratio (float): Frazione per il training set
        num_keypoints (int, optional): Numero fisso di keypoint. Se None, usa padding.

    Returns:
        tuple: (train_loader, val_loader, num_output_features)
    """

    # Split train/validation
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    val_df = df[train_size:]

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Crea i dataset
    if num_keypoints is not None:
        # Numero fisso di keypoint
        train_dataset = KeypointDatasetFixed(
            train_df,
            num_keypoints=num_keypoints,
            transform=get_transforms(is_training=True)
        )
        val_dataset = KeypointDatasetFixed(
            val_df,
            num_keypoints=num_keypoints,
            transform=get_transforms(is_training=False)
        )
        num_output_features = num_keypoints * 2
    else:
        # Padding per keypoint variabili
        train_dataset = KeypointDataset(
            train_df,
            transform=get_transforms(is_training=True)
        )
        val_dataset = KeypointDataset(
            val_df,
            transform=get_transforms(is_training=False)
        )
        max_keypoints = df['num_keypoints'].max()
        num_output_features = max_keypoints * 2

    # Crea i DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, num_output_features


# Funzione di test
def test_dataset():
    """Funzione di test per verificare il dataset."""

    # Carica il dataset processato
    df = pd.read_pickle("dataset.pkl")

    # Test con il dataset
    dataset = KeypointDataset(df, transform=get_transforms(is_training=False))

    print(f"Dimensione dataset: {len(dataset)}")

    # Test di un campione
    image, keypoints = dataset[0]
    print(f"Forma immagine: {image.shape}")
    print(f"Forma keypoints: {keypoints.shape}")
    print(f"Keypoints campione: {keypoints[:10]}")  # Primi 10 valori

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch_images, batch_keypoints = next(iter(loader))

    print(f"Batch immagini: {batch_images.shape}")
    print(f"Batch keypoints: {batch_keypoints.shape}")


if __name__ == "__main__":
    test_dataset()