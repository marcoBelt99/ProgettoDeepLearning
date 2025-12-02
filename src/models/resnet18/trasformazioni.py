import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from models.resnet18.parametri_modello import *


'''
Per meglio isolare le trasformazioni.
'''


# Normalizzazione a valori di media e dev. standard del modello pre-addestrato
# (normalizzazione su immagini che contengono valori [0, 255]
normalizzazione = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


resnet18_data_transforms = {

    # Trasformazioni da applicare per la data augmentation al training set
    'train' : A.Compose([

                # Ridimensionamento a IMG_SIZE x IMG_SIZE
                A.Resize(IMG_SIZE, IMG_SIZE),

                # FIXME: dovrei forse usare A.Affine() che è più moderna,
                #  al posto della A.ShiftScaleRotate().
                # A.ShiftScaleRotate(
                #     shift_limit=0.05, scale_limit=0.05, rotate_limit=5,
                #     p=0.5, border_mode=cv2.BORDER_CONSTANT
                # ),
                A.Affine(
                    # La traslazione (shift) è specificata come percentuale
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    # La scala è definita come un intervallo, calcolato da 1 +/- scale_limit
                    # scale_limit=0.05 -> intervallo [1 - 0.05, 1 + 0.05] = [0.95, 1.05]
                    scale=(0.95, 1.05),
                    # La rotazione (rotate) è definita in gradi
                    rotate=(-5, 5),
                    # I parametri di controllo (p e border_mode) sono diretti
                    p=0.6, # prima era a 0.5
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                # FIXME: sto avendo problemi con A.GaussNoise:
                # 'var_limit' are not valid for transform GaussNoise
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2), # quello di prima
                # A.GaussNoise(std_range=(5.0, 15.0), p=0.2), # nuova modifica, però da errore
                # TODO: Per GaussNoise, visto che ha cambiato API, la versione corretta dovrebbe essere:
                # A.GaussNoise(var_limit=(5, 30), mean=0, p=0.2)
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),

                normalizzazione, # va sempre prima della tensorizzazione
                ToTensorV2()
            ],

            # Gli passo i keypoints tramite la classe
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),

            # con remove_invisible=False evito che Albumentations elimini punti invisibili

    # Trasformazioni da applicare per il test
    'test' : A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                normalizzazione, # va sempre prima della tensorizzazione
                ToTensorV2()
            ],

            # Gli passo i keypoints tramite la classe
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
}