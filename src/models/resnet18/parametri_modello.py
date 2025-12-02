import torch

###################################
### PARAMETRI DI CONFIGURAZIONE ###
###################################

'''
Principali parametri e iperparametri.
'''

IMG_SIZE : int = 224  # Dimensione delle immagini che vanno in input al modello pre-addestrato
BATCH_SIZE : int = 16  # Provato con: 4,8                          Provare con: 8, 16,
EPOCHS : int = 20  # Provato con: 4,8,10, 20, 50                   Provare con: 20,
LR : float = 1e-4  # Provato con: 1e-4                             Provare con: 0.001
TRAIN_RATIO : float = 0.8  # Provato con: 80% train, 20% test      Provare con: 75, 25
DEVICE : str = "cuda" if torch.cuda.is_available() else "cpu"


# Mean e std di ImageNet (necessarie per modelli pre-addestrati)
# (infatti mi serve avere la media e la dev standard del dataset su cui è stato fatto l'apprendimento)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


'''
Molti setup di keypoints detection con backbone ResNet18 lavorano proprio con LR intorno a 1e-4.
Il fatto che non abbia dovuto abbassare il LR a metà training per mantenere la convergenza è un buon segno:
il gradiente non esplode, non collassa, non oscilla.
'''

# Numero di punti in totale da predire
NUM_TOTALE_PUNTI = 14

