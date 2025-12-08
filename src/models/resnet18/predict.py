import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from factory import build_resnet18
from models.resnet18.trasformazioni import resnet18_data_transforms
from parametri_modello import *
from configs.parametri_app import CHECKPOINTS_DIR, DATASET_DIR, TESTING_DIR

## 1) CARICO IL MODELLO ADDRESTRATO

MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "layer3_4_fc_best.pth")
print("Carico modello da:", MODEL_PATH)

def carica_modello(model_path):
    model = build_resnet18(
        num_outputs=NUM_TOTALE_PUNTI * 2,
        pretrained=True,
        head="linear",
        freeze_until="layer3"
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return model


model = carica_modello(MODEL_PATH)



# 2) FUNZIONE PER PREDIRE I KEYPOINTS SU UNA SINGOLA IMMAGINE


def predict_keypoints(img_path, model):
    """
    Predice i keypoints per un’immagine esterna o del dataset.
    Ritorna:
        - immagine denormalizzata (np array)
        - keypoints predetti in pixel (numpy)
    """

    ## Caricamento immagine con open cv
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_img = img.copy()  # per visualizzazione

    ## Applica trasformazioni del test set
    transform = resnet18_data_transforms["test"]
    transformed = transform(image=img)
    tensor_img = transformed["image"].unsqueeze(0).to(DEVICE)

    # Predizione
    with torch.no_grad():
        output = model(tensor_img).cpu().numpy().flatten()

    # ---- reshape in [14,2] e denormalizza ----
    keypoints = output.reshape(-1, 2)
    keypoints = keypoints * np.array([IMG_SIZE, IMG_SIZE])  # da [0,1] a pixel

    # ---- denormalizza immagine per plotting ----
    mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    std = np.array(IMAGENET_STD).reshape(1,1,3)

    # ricostruisco immagine normalizzata → denormalizzata
    img_norm = transformed["image"].permute(1,2,0).numpy()
    img_denorm = img_norm * std + mean
    img_denorm = np.clip(img_denorm, 0, 1)

    return img_denorm, keypoints



# 3) FUNZIONE DI VISUALIZZAZIONE

def show_prediction(img_path):
    img, kpts = predict_keypoints(img_path, model)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.scatter(kpts[:,0], kpts[:,1], c='red', s=30)

    for x, y in kpts:
        plt.text(x+2, y+2, ".", color="yellow")  # piccolo marcatore opzionale

    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()



if __name__ == "__main__":

    # metti qui una immagine qualsiasi del dataset o esterna
    esempio_from_dataset = os.path.join(DATASET_DIR, "5.jpg")  # <-- SOSTITUISCI CON IL TUO FILE
    show_prediction(esempio_from_dataset)

    # Quest'immagine l'ho presa online da un dataset su Kaggle
    esempio_esterno = os.path.join( TESTING_DIR, "Convex_Concave/234.jpg")
    show_prediction(esempio_esterno)
