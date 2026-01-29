import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from factory import build_resnet18
from src.data.dataset.repere_dataset_1 import RepereKeypointsDataset # uso repere_dataset_1
from models.resnet18.trasformazioni_1 import resnet18_data_transforms # uso trasformazioni_1
from parametri_modello import *

from configs.parametri_app import (
    CHECKPOINTS_DIR,
    DATASET_DIR,
    TESTING_DIR,
    DATAFRAME_MASTER,
    RAGGRUPPAMENTI
)


#######################################################
#################### TODO: CONFIGURAZIONE #############
#######################################################

# 1) Scelgo se e quale gruppo voglio valutare
#    assegnare None se non voglio valutare i gruppi

GRUPPO = "GRUPPO1"
# GRUPPO = None # (se voglio il modello che predice tutti i punti contemporaneamente)

# 2) Scelgo quale modello di rete neurale usare

# MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "GRUPPO1_resnet18_best.pth")
# MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "GRUPPO1_resnet18_BEST_EARLY.pth")
# MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'layer3_4_fc_best.pth')
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'GRUPPO1_resnet18_layer3_linear_lr0.0001_epoche60_BEST_EARLY.pth')



# CARICAMENTO MODELLO scelto

def carica_modello(model_path):

    # carico la struttura del modello
    model = build_resnet18(
        num_outputs= ( len(RAGGRUPPAMENTI[GRUPPO]) if GRUPPO is not None else NUM_TOTALE_PUNTI) * 2,
        # pretrained=True,
        pretrained= False,
        head="linear"#,
        # freeze_until="layer3"
    ).to(DEVICE)

    # ne carico i pesi
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_keypoints(img_path, model):
    """
    Effettua l'effettiva predizione con il modello di rete neurale scelto.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Eseguo lo stesso preprocessing del test set
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




def get_keypoints_from_dataset(dataset, img_path):
    """
    Recupera le etichette reali (ossia i keypoints) dell'immagine
    (avente path img_path) dal dataset.
    """
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



def show_prediction(img_path, model, dataset=None):
    """
    Funzione per visualizzare i keypoints sovrapposti all'immagine.

    """

    # 1) EFFETTUA LA PREDIZIONE, TRAMITE IL MODELLO DI RETE NEURALE
    img, keypoints_pred = predict_keypoints(img_path, model)

    # 2) PROVO AD OTTENERE LE ETICHETTE REALI
    #    dell'immagine (img_path) del dataset
    keypoints_true = None
    if dataset is not None:
        keypoints_true = get_keypoints_from_dataset(dataset, img_path)

    # 3) Parte di visualizzazione
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





if __name__ == "__main__":

    # 1) CARICAMENTO MODELLO DI RETE NEURALE
    print("Carico modello:", MODEL_PATH)
    model = carica_modello(MODEL_PATH)

    # Dataset (serve solo per immagini del dataset)
    df = pd.read_csv(DATAFRAME_MASTER)
    dataset = RepereKeypointsDataset(
        df,
        img_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        trasformazioni=resnet18_data_transforms["test"],

        # SE VOGLIO VALUTARE SUL SINGOLO GRUPPO DI PUNTO LASCIO
        # ALTRIMENTI COMMENTA SE VOGLIO VALUTARE SU TUTTI I PUNTI
        gruppo_punti=RAGGRUPPAMENTI[GRUPPO] if GRUPPO in RAGGRUPPAMENTI else None,
    )



    # 2) MOSTRA PREDIZIONI

    # Immagine interna del dataset (reali + predetti)
    img_dataset = os.path.join(DATASET_DIR, "5.jpg")
    show_prediction(img_dataset, model, dataset)

    #  Immagine esterna (solo predetti)
    img_external = os.path.join(TESTING_DIR, "Convex_Concave/234.jpg")
    show_prediction(img_external, model)
