import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset.repere_dataset import RepereKeypointsDataset
from factory import build_resnet18
from models.resnet18.trasformazioni import resnet18_data_transforms
from configs.parametri_app import (
    DATAFRAME_MASTER,
    DATASET_DIR,
    CHECKPOINTS_DIR,
    RAGGRUPPAMENTI
)
from parametri_modello import IMG_SIZE, DEVICE, BATCH_SIZE

import matplotlib.pyplot as plt



def errori_per_landmark(model, dataloader, device, img_size):
    '''

    '''
    model.eval()
    errori = None

    with torch.no_grad():
        for images, keypoints_true in dataloader:
            images = images.to(device)
            keypoints_true = keypoints_true.to(device)

            pred = model(images)

            # effettuo un reshape da: [B, 2N] a: [B, N, 2]
            pred = pred.view(pred.size(0), -1, 2)
            keypoints_true = keypoints_true.view(keypoints_true.size(0), -1, 2)

            # denormalizzazione
            pred *= img_size
            keypoints_true *= img_size

            # distanza euclidea per punto
            dist = torch.norm(pred - keypoints_true, dim=2)  # [B, N]

            if errori is None:
                errori = [[] for _ in range(dist.size(1))]

            for i in range(dist.size(1)):
                errori[i].extend(dist[:, i].cpu().numpy())

    return errori


# ==========================================================
# BOX PLOT
# ==========================================================
def plot_boxplot_landmark(errori, titolo):
    plt.figure(figsize=(10, 5))
    plt.boxplot(errori, showfliers=False)
    plt.xlabel("Landmark")
    plt.ylabel("Errore (pixel)")
    plt.title(titolo)
    plt.grid(True)
    plt.show()


# ==========================================================
# MAIN
# ==========================================================
def main():

    # ===============================
    # CONFIG
    # ===============================
    NOME_GRUPPO = "GRUPPO1"  # ← cambia qui (GRUPPO1, GRUPPO2, ...)
    MODEL_PATH = os.path.join(
        CHECKPOINTS_DIR,
        f"{NOME_GRUPPO}_resnet18_best.pth"
    )

    print(f"Valutazione {NOME_GRUPPO}")
    print("Carico modello da:", MODEL_PATH)

    # ===============================
    # DATASET (solo TEST)
    # ===============================
    df = pd.read_csv(DATAFRAME_MASTER)

    dataset_test = RepereKeypointsDataset(
        dataframe=df,
        img_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        trasformazioni=resnet18_data_transforms["test"],
        gruppo_punti=RAGGRUPPAMENTI[NOME_GRUPPO]
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # ===============================
    # MODELLO
    # ===============================
    model = build_resnet18(
        num_outputs=len(RAGGRUPPAMENTI[NOME_GRUPPO]) * 2,
        pretrained=False,
        head="linear",
        freeze_until=None
    ).to(DEVICE)

    # Carico il modello più preciso sperimentato
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    # ===============================
    # CALCOLO ERRORI
    # ===============================
    errori = errori_per_landmark(
        model,
        test_loader,
        DEVICE,
        IMG_SIZE
    )

    # ===============================
    # STAMPA RISULTATI
    # ===============================
    print("\nErrore per landmark (pixel):")
    for i, e in enumerate(errori):
        print(
            f"  Punto {i+1}: "
            f"MAE = {np.mean(e):.2f} px | "
            f"STD = {np.std(e):.2f} px"
        )

    # ===============================
    # BOXPLOT
    # ===============================
    plot_boxplot_landmark(
        errori,
        titolo=f"Errore per landmark – {NOME_GRUPPO}"
    )


# ==========================================================
if __name__ == "__main__":
    main()
