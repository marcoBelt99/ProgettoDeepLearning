import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from parametri_modello import IMAGENET_MEAN, IMAGENET_STD

# Normalizzazione ImageNet (include /255.0 + standardizzazione)
_normalizzazione = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def get_transforms_for_group(group_name: str, img_size: int):
    """
    Restituisce dict con:
      - 'train': A.Compose(...)
      - 'test' : A.Compose(...)
    Trasformazioni pensate per regressione keypoints con ResNet/MobileNet.
    """

    # Base resize senza distorcere
    resize_pad = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
    ]

    # Default: leggero (sicuro)
    affine_light = A.Affine(
        translate_percent={'x': (-0.04, 0.04), 'y': (-0.04, 0.04)},
        scale=(0.97, 1.03),
        rotate=(-4, 4),
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT
    )

    # Per gruppo: regolazioni
    if group_name == "GRUPPO1":
        # punti globali: regolarizzazione leggera, niente flip
        train_aug = [
            affine_light,
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.25),
            A.CLAHE(clip_limit=2.0, p=0.15),
            A.GaussNoise(std_range=(5/255, 12/255), mean_range=(0.0, 0.0), p=0.15),
            # dropout molto leggero o anche senza
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.05, 0.08),
                            hole_width_range=(0.05, 0.08), fill=0, p=0.08),
        ]
    elif group_name == "GRUPPO2":
        # zona della mandibola e del naso: più variabilità, ma provo ad evitare distruzioni e flip
        train_aug = [
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-5, 5),
                p=0.6,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.30),
            A.CLAHE(clip_limit=2.0, p=0.20),
            A.GaussNoise(std_range=(5/255, 15/255), mean_range=(0.0, 0.0), p=0.20),
            A.GaussianBlur(blur_limit=3, p=0.03), # prima era a 0.10
            A.CoarseDropout(num_holes_range=(1, 2),
                            hole_height_range=(0.05, 0.10),
                            hole_width_range=(0.05, 0.10),
                            fill=0,
                            p=0.05), # prima a 0.12
        ]
    elif group_name == "GRUPPO3":
        # denti: serve precisione -> aug molto gentile, niente dropout/flip
        train_aug = [
            A.Affine(
                translate_percent={'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
                scale=(0.98, 1.02),
                rotate=(-3, 3),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.06, p=0.20),
            A.CLAHE(clip_limit=2.0, p=0.10),
            # A.GaussianBlur(blur_limit=3, p=0.05), # non rende evidenti i contorni tra i denti?
            A.GaussNoise(std_range=(3/255, 10/255), mean_range=(0.0, 0.0), p=0.10),
        ]
    elif group_name == "GRUPPO4":
        # 2 punti: minimo necessario, molto stabile
        train_aug = [
            A.Affine(
                translate_percent={'x': (-0.04, 0.04), 'y': (-0.04, 0.04)},
                scale=(0.97, 1.03),
                rotate=(-4, 4),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.20),
        ]
    else:
        train_aug = [affine_light]

    train = A.Compose(
        resize_pad + train_aug + [_normalizzazione, ToTensorV2()],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    test = A.Compose(
        resize_pad + [_normalizzazione, ToTensorV2()],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    return {"train": train, "test": test}
