import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet

from configs.parametri_app import DATAFRAME_MASTER, DATASET_DIR, CHECKPOINTS_DIR
from src.data.dataset.repere_dataset import  crea_split_dataset
import pandas as pd
from utils.utils import ok, fail

from timeit import default_timer as timer # timer per monitorare il tempo che impiego ad allenare
# il modello

###################################
### PARAMETRI DI CONFIGURAZIONE ###
###################################


IMG_SIZE = 224      # Dimensione delle immagini che vanno in input al modello pre-addestrato
BATCH_SIZE = 4      # Provato con: 4,    Provare con: 8, 16,
EPOCHS = 4          # Provato con: 4,    Provare con: 20,
LR = 1e-4           # Provato con: 1e-4  Provare con: 0.001
TRAIN_RATIO = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###################################
####### CARICAMENTO DATASET #######
###################################

## Sto caricando il dataframe master
df = pd.read_csv(DATAFRAME_MASTER)

## Suddivisione dataset in train e test set
train_set, test_set = crea_split_dataset(df, img_dir=DATASET_DIR, img_size=IMG_SIZE, train_ratio=TRAIN_RATIO)

## Preparazione dei data loader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


#######################################
### RECUPERO MODELLO PRE-ADDESTRATO ###
#######################################
model : ResNet = models.resnet18(weights='IMAGENET1K_V1')

## Sostituisco l'ultimo layer per la regressione dei 28 valori (14 punti × 2)
model.fc = nn.Linear(model.fc.in_features, 28)


model = model.to(DEVICE)

## Loss ed ottimizzatore
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


#######################################
### LOOP DI TRAINING ###
#######################################
print(f"{ok}Arrivato al loop di training\n")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, keypoints in train_loader:
        images = images.to(DEVICE)
        keypoints = keypoints.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    #######################################
    ######### LOOP DI VALIDAZIONE #########
    #######################################
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, keypoints in test_loader:
            images = images.to(DEVICE)
            keypoints = keypoints.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, keypoints)
            val_loss += loss.item() * images.size(0)

    avg_val_loss = val_loss / len(test_loader.dataset)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

#######################################
####### SALVATAGGIO DEL MODELLO #######
#######################################
nome_modello : str = "resnet18_keypoints_all_points.pth"

torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, nome_modello) )
print(f"{ok}Modello salvato come {nome_modello}")

