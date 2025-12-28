import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from factory import build_resnet18
from src.data.dataset.repere_dataset import RepereKeypointsDataset
from models.resnet18.trasformazioni import resnet18_data_transforms
from parametri_modello import *

from configs.parametri_app import (
    CHECKPOINTS_DIR,
    DATASET_DIR,
    TESTING_DIR,
    DATAFRAME_MASTER,
    RAGGRUPPAMENTI
)

# ============================================================
# CONFIGURAZIONE
# ============================================================

MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "GRUPPO1_resnet18_best.pth")
GRUPPO = "GRUPPO1"

# ============================================================
# CARICAMENTO MODELLO
# ============================================================

def carica_modello(model_path):
    model = build_resnet18(
        num_outputs=len(RAGGRUPPAMENTI[GRUPPO]) * 2,
        pretrained=True,
        head="linear",
        freeze_until="layer3"
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# PREDIZIONE KEYPOINTS
# ============================================================

def predict_keypoints(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = resnet18_data_transforms["test"]
    transformed = transform(image=img)

    tensor_img = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_img).cpu().numpy().reshape(-1, 2)

    # denormalizzo keypoints
    keypoints_pred = output * IMG_SIZE

    # denormalizzo immagine
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    img_norm = transformed["image"].permute(1, 2, 0).numpy()
    img_denorm = np.clip(img_norm * std + mean, 0, 1)

    return img_denorm, keypoints_pred


# ============================================================
# KEYPOINTS REALI DAL DATASET
# ============================================================

def get_keypoints_from_dataset(dataset, img_path):
    img_name = os.path.basename(img_path)

    matches = dataset.dataframe.index[
        dataset.dataframe["path_img"] == img_name
    ]

    if len(matches) == 0:
        return None

    idx = matches[0]
    _, keypoints = dataset[idx]

    keypoints = keypoints.numpy().reshape(-1, 2) * IMG_SIZE
    return keypoints


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def show_prediction(img_path, model, dataset=None):
    img, keypoints_pred = predict_keypoints(img_path, model)

    keypoints_true = None
    if dataset is not None:
        keypoints_true = get_keypoints_from_dataset(dataset, img_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)

    # Predetti
    ax.scatter(
        keypoints_pred[:, 0],
        keypoints_pred[:, 1],
        c="red",
        marker="x",
        s=50,
        label="Predetti"
    )

    # Reali (solo se disponibili)
    if keypoints_true is not None:
        ax.scatter(
            keypoints_true[:, 0],
            keypoints_true[:, 1],
            c="lime",
            s=40,
            label="Reali"
        )

        for t, p in zip(keypoints_true, keypoints_pred):
            ax.plot(
                [t[0], p[0]],
                [t[1], p[1]],
                color="orange",
                linestyle="--",
                linewidth=1
            )

    ax.set_title(os.path.basename(img_path))
    ax.legend()
    ax.axis("off")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Carico modello:", MODEL_PATH)
    model = carica_modello(MODEL_PATH)

    # Dataset (serve solo per immagini del dataset)
    df = pd.read_csv(DATAFRAME_MASTER)
    dataset = RepereKeypointsDataset(
        df,
        img_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        trasformazioni=resnet18_data_transforms["test"]
    )

    # ---- Immagine del dataset (reali + predetti)
    img_dataset = os.path.join(DATASET_DIR, "5.jpg")
    show_prediction(img_dataset, model, dataset)

    # ---- Immagine esterna (solo predetti)
    img_external = os.path.join(TESTING_DIR, "Convex_Concave/234.jpg")
    show_prediction(img_external, model)
