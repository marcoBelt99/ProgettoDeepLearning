import torch

###################################
### PARAMETRI DI CONFIGURAZIONE ###
###################################

# Device
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet mean/std (per modelli pretrained ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Numero totale punti nel dataset (14)
NUM_TOTALE_PUNTI: int = 14

# Seed (riproducibilità)
SEED: int = 42

# Dataloader
NUM_WORKERS: int = 0  # su CPU spesso conviene 0 o 2; su Windows 0 evita problemi
PIN_MEMORY: bool = (DEVICE == "cuda")

# Training stability
GRAD_CLIP_NORM: float = 1.0  # None per disabilitare
WEIGHT_DECAY: float = 1e-4

# Default (fallback)
DEFAULT_IMG_SIZE: int = 224
DEFAULT_EPOCHS: int = 60

# Config per gruppo: iperparametri CPU-friendly + modello diverso per gruppo.
# Nota: puoi modificarli senza toccare codice.
GROUP_CONFIG = {
    "GRUPPO1": {
        "img_size": 224,
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-4,
        "model_name": "resnet18_g1",
        "pretrained": True,
        "freeze_until": "layer3",   # più stabile su punti globali (con layer 4 ottenevo: TEST: loss=0.009435 | mae=2.566px | med=4.034px)
        "head": "linear",
        "scheduler": {"type": "plateau", "factor": 0.5, "patience": 4},
        "early_stop": {"patience": 7, "min_delta": 1e-4},
    },
    "GRUPPO2": {
        "img_size": 224,
        "batch_size": 8,
        "epochs": 40,
        "lr": 2e-4,
        "model_name": "resnet34_g2",
        "pretrained": True,
        "freeze_until": "layer3",   # più adattabile su mandibola/naso
        "head": "mlp",
        "scheduler": {"type": "plateau", "factor": 0.5, "patience": 4},
        "early_stop": {"patience": 12, "min_delta": 1e-4},
    },
    "GRUPPO3": {
        "img_size": 256,            # prima era nel suo default, cioè 224 ==> con 256 ho più dettaglio
        "batch_size": 16,
        "epochs": 45,
        "lr": 3e-4,
        "model_name": "mobilenetv3_g3",
        "pretrained": True,
        "freeze_until": "none",
        "head": "mlp", # MLP
        "scheduler": {"type": "plateau", "factor": 0.5, "patience": 4},
        "early_stop": {"patience": 14, "min_delta": 1e-4},
    },
    "GRUPPO4": {
        "img_size": 224,
        "batch_size": 32, # mi alzo anche il batch size
        "epochs": 30,
        "lr": 5e-4,
        "model_name": "smallcnn_g4",
        "pretrained": False,
        "freeze_until": "none",
        "head": "internal", #
        "scheduler": {"type": "plateau", "factor": 0.5, "patience": 4},
        "early_stop": {"patience": 9, "min_delta": 1e-4},
    },
}
