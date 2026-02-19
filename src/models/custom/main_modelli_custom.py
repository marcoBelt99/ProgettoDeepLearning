import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.parametri_app import DATAFRAME_MASTER, RAGGRUPPAMENTI, DATASET_DIR, NOMI_PUNTI_RAGGRUPPAMENTI

from src.data.dataset.repere_dataset_1 import RepereKeypointsDataset

from parametri_modello import (
    DEVICE, GROUP_CONFIG, SEED, NUM_WORKERS, PIN_MEMORY, WEIGHT_DECAY
)
from trasformazioni_custom import get_transforms_for_group
from factory import build_model_for_group
from train_and_validate_custom import execute


#
# CONFIG: scegliere gruppo o None per allenarli tutti
#
# GRUPPO_DA_ALLENARE = "GRUPPO1"   # es: "GRUPPO1" | "GRUPPO2" | "GRUPPO3" | "GRUPPO4" | None
# GRUPPO_DA_ALLENARE = "GRUPPO2"
# GRUPPO_DA_ALLENARE = "GRUPPO3" # provo con mobilenetv3
GRUPPO_DA_ALLENARE = "GRUPPO4"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_group(nome_gruppo: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    cfg = GROUP_CONFIG[nome_gruppo]
    img_size = int(cfg["img_size"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    epochs = int(cfg["epochs"])
    model_name = cfg["model_name"]
    pretrained = bool(cfg["pretrained"])
    freeze_until = cfg["freeze_until"]
    head = cfg["head"]

    print("\n")
    print(f"Allenamento {nome_gruppo} ({NOMI_PUNTI_RAGGRUPPAMENTI[nome_gruppo]})")
    print("Config:", {k: cfg[k] for k in ['img_size','batch_size','epochs','lr','model_name','freeze_until','head']})
    print("\n")

    # Trasformazioni per-group
    tfs = get_transforms_for_group(nome_gruppo, img_size)

    # Dataset
    train_set = RepereKeypointsDataset(
        train_df,
        img_dir=DATASET_DIR,
        img_size=img_size,
        trasformazioni=tfs["train"],
        gruppo_punti=RAGGRUPPAMENTI[nome_gruppo]
    )
    val_set = RepereKeypointsDataset(
        val_df,
        img_dir=DATASET_DIR,
        img_size=img_size,
        trasformazioni=tfs["test"],
        gruppo_punti=RAGGRUPPAMENTI[nome_gruppo]
    )
    test_set = RepereKeypointsDataset(
        test_df,
        img_dir=DATASET_DIR,
        img_size=img_size,
        trasformazioni=tfs["test"],
        gruppo_punti=RAGGRUPPAMENTI[nome_gruppo]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Modello
    model = build_model_for_group(
        nome_gruppo=nome_gruppo,
        model_name=model_name,
        pretrained=pretrained,
        head=head,
        freeze_until=freeze_until
    ).to(DEVICE)



    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, # ignoro i pesi non allenabili
               model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY # applico una penalità ai pesi
    )



    """
    #  Questo era per il gruppo 2, per vedere se funzionava meglio con il discriminative LR, ma a quanto pare non va bene
    #  SOLO PER MODELLO 2: Discriminative LR: backbone più basso, head più alto
    head_params = list(model.backbone.fc.parameters())  # per ResNet: la head è fc
    head_param_ids = {id(p) for p in head_params}

    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.25},  # backbone LR più piccolo (2e-4 -> 5e-5)
            {"params": head_params, "lr": lr},  # head LR pieno
        ],
        weight_decay=WEIGHT_DECAY
    )
    """


    # Output root per gruppo
    group_folder = nome_gruppo.lower()
    output_root = Path(__file__).resolve().parent / group_folder

    name_train = f"{nome_gruppo}_{model_name}_{freeze_until}_{head}_lr{lr}_ep{epochs}"
    num_points = len(RAGGRUPPAMENTI[nome_gruppo])

    execute(
        name_train=name_train,
        nome_gruppo=nome_gruppo,
        model=model,
        optimizer=optimizer,
        starting_lr=lr,
        num_epochs=epochs,
        num_points=num_points,
        img_size=img_size,
        batch_size=batch_size,
        freeze_until=freeze_until,
        head=head,
        output_root=str(output_root),
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        data_loader_test=test_loader,
        scheduler_cfg=cfg.get("scheduler", {}),
        early_stop_cfg=cfg.get("early_stop", {}),
        seed=SEED,
    )


def main():
    set_seed(SEED)

    df = pd.read_csv(DATAFRAME_MASTER)

    # split 80/10/10
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, shuffle=True)

    print(f"Device: {DEVICE}")
    print(f"Dimensione Training set: {len(train_df)}")
    print(f"Dimensione Validation set: {len(val_df)}")
    print(f"Dimensione Test set: {len(test_df)}")

    if GRUPPO_DA_ALLENARE is None:
        for nome_gruppo in ["GRUPPO1", "GRUPPO2", "GRUPPO3", "GRUPPO4"]:
            train_group(nome_gruppo, train_df, val_df, test_df)
    else:
        train_group(GRUPPO_DA_ALLENARE, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
