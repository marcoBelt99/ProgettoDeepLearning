import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import os

from configs.parametri_app import DATAFRAME_MASTER, DATASET_DIR, CHECKPOINTS_DIR
from src.data.dataset.repere_dataset import RepereKeypointsDataset
import pandas as pd

# ==============================
# CONFIGURAZIONE
# ==============================

# dirname = os.path.dirname(__file__)
MODEL_PATH = f"{CHECKPOINTS_DIR}/resnet18_keypoints.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# CARICA DATASET E MODELLO
# ==============================
df = pd.read_csv(DATAFRAME_MASTER)
dataset = RepereKeypointsDataset(df, img_dir=DATASET_DIR, img_size=IMG_SIZE, augment=False)

from torchvision import models
import torch.nn as nn

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 28)

# Carico i pesi del mio modello che ho allenato
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==============================
# FUNZIONE DI VISUALIZZAZIONE
# ==============================
def visualizza_predizione(idx):
    """
    Mostra l'immagine idx del dataset con i keypoints reali e predetti.
    """
    # Ottieni immagine e keypoints reali
    image, keypoints_true = dataset[idx]
    image_np = image.permute(1, 2, 0).numpy()  # da [C,H,W] → [H,W,C]
    image_np = np.clip(image_np, 0, 1)  # per sicurezza

    # Predizione
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()

    # Reshape da [28] → [14,2]
    keypoints_true = keypoints_true.numpy().reshape(-1, 2)
    keypoints_pred = pred.reshape(-1, 2)

    # Converti coordinate normalizzate in pixel se serve (nel tuo caso già in 0–224)
    H, W = IMG_SIZE, IMG_SIZE
    keypoints_true = keypoints_true * np.array([W, H])
    keypoints_pred = keypoints_pred * np.array([W, H])

    # Disegna
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.scatter(keypoints_true[:, 0], keypoints_true[:, 1], c='lime', s=40, label='Reali')
    ax.scatter(keypoints_pred[:, 0], keypoints_pred[:, 1], c='red', marker='x', s=40, label='Predetti')

    for i, (t, p) in enumerate(zip(keypoints_true, keypoints_pred)):
        ax.plot([t[0], p[0]], [t[1], p[1]], color='orange', linestyle='--', linewidth=1)  # collega i due punti

    ax.legend()
    # ax.set_title(f"Immagine {dataset.data.iloc[idx]['path_img']}")
    ax.set_title(f"Immagine {dataset.dataframe.iloc[idx]['path_img']}")
    ax.axis('off')
    plt.show()


# ==============================
# ESEMPIO: visualizza un'immagine a scelta
# ==============================
visualizza_predizione(0)
visualizza_predizione(5)
visualizza_predizione(10)


