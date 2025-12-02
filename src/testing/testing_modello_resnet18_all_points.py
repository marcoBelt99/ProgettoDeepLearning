import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import os
from torchvision import models
import torch.nn as nn

from configs.parametri_app import DATAFRAME_MASTER, DATASET_DIR, CHECKPOINTS_DIR
from models.resnet18.modello_resnet18_all_points import ResNet18_Keypoints
from src.data.dataset.repere_dataset import RepereKeypointsDataset
import pandas as pd

from models.resnet18.trasformazioni import resnet18_data_transforms
from models.resnet18.parametri_modello import *



# PARAMETRI DI CONFIGURAZIONE

MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'resnet18_all_points.pth')
print("checkpoints dir: ", CHECKPOINTS_DIR)
print("file .pth: ", MODEL_PATH)

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# CARICA DATASET E MODELLO
df = pd.read_csv(DATAFRAME_MASTER)
dataset = RepereKeypointsDataset(df, img_dir=DATASET_DIR, img_size=IMG_SIZE, trasformazioni=resnet18_data_transforms['test'], augment=False)


#### TODO: questa roba qui deve SPARIRE perchè:
'''
- ricreare un modello con pesi ImageNet sovrascrive gran parte delle strutture interne
- poi carichi la state_dict, ma questo funziona solo se la definizione del modello è identica a quella usata nell’addestramento
- nel tuo codice di training probabilmente hai aggiunto dropout, batchnorm, o altro
- quindi reinizializzare così il modello è pericoloso
La definizione che devi usare è SOLO quella di training.
'''
# model = models.resnet18(weights='IMAGENET1K_V1')
# model.fc = nn.Linear(model.fc.in_features, 28)
#############################


### TODO: parte nuova

model = ResNet18_Keypoints(num_outputs=28)

# Carico i pesi del mio modello che ho allenato
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

model.eval()





def denormalizza_immagine(image_tensor, mean, std):
    """
    Denormalizza un tensore immagine normalizzato con mean/std ImageNet.
    Input:  tensor [3, H, W]
    Output: numpy array [H, W, 3] in range [0,1]

    TODO: pensare di parametrizzare anche MEAN e STD in futuro.
    """
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


    mean = torch.Tensor(mean).view(3,1,1)
    std = torch.Tensor(std).view(3,1,1)

    image = image_tensor.clone()

    # Serve denormalizzare con la giusta MEAN e STD,
    # prima del np.clip
    image = image * std + mean  # de-normalize

    # # Passo da: [C,H,W] → [H,W,C]
    image = image.permute(1, 2, 0).numpy()


    image = np.clip(image, 0, 1)                 # clamp sicuro

    return image




#################################
## FUNZIONE DI VISUALIZZAZIONE ##
#################################
def visualizza_predizione(idx):
    """
    Mostra l'idx-esima immagina dal con i suoi relativi keypoints reali e,
    poi anche con quelli predetti.

    Args:
        idx (int): indice del dataframe da cui prendere l'idx-esima immagine con i suoi keypoints.
    """

    # Ottengo l'immagine e keypoints REALI dal dataset
    image, keypoints_true = dataset[idx]

    # Ripristino l'immagine originale.
    image_np = denormalizza_immagine(image, IMAGENET_MEAN, IMAGENET_STD)


    # Effettuo le Predizioni
    with torch.no_grad():
        pred = model( image.unsqueeze(0).to(DEVICE) ).cpu().squeeze().numpy()

    # Reshape da [28] → [14,2]
    keypoints_true = keypoints_true.numpy().reshape(-1, 2)
    keypoints_pred = pred.reshape(-1, 2)

    # Converto le coordinate normalizzate in pixel se serve (nel tuo caso già in 0–224)
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
    ax.set_title(f"Immagine {dataset.dataframe.iloc[idx]['path_img']}")
    ax.axis('off')
    plt.show()




# ==============================
# ESEMPIO: visualizza un'immagine a scelta
# ==============================
visualizza_predizione(0)
visualizza_predizione(5)
visualizza_predizione(10)


