import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


'''
Per meglio isolare le trasformazioni.
'''


# Mean e std di ImageNet (necessarie per modelli pre-addestrati)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dimensione immagine
IMG_SIZE = 224

# Normalizzazione a valori di media e dev. standard del modello pre-addestrato
normalizzazione = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


data_transforms = {

    # Trasformazioni da applicare per la data augmentation al training set
    'train' : A.Compose([
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
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                # FIXME: sto avendo problemi con A.GaussNoise:
                # 'var_limit' are not valid for transform GaussNoise
                # A.GaussNoise(var_limit=(5.0, 15.0), p=0.2), # quello di prima
                A.GaussNoise(std_range=(5.0, 15.0), p=0.2), # nuova modifica
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                normalizzazione, # va sempre prima della tensorizzazione
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),

    # Trasformazioni da applicare per il test
    'test' : A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                normalizzazione, # va sempre prima della tensorizzazione
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
}