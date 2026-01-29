from configs.parametri_app import * # parametri dell'applicazione da usare in giro per il progetto
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.models.resnet18.parametri_modello import *
import torch.nn as nn
from torch.utils.data import DataLoader

'''
Modulo in cui inserisco funzioni di uso comune e di utilità varie.
'''

def get_num_files(estensione : str) -> int:
    '''
    Restituisce il numero di file con quella determinata estensione presenti dentro la lista
    di tutti i file (jpg, txt) del dataset.

    Args:
        estensione: può essere 'jpg' o 'txt'.
    '''
    return len(list(filter( lambda nome_file: (f"{estensione.lower()}" in nome_file), FILES )))

ok = '[✓]'
fail = '[x]'










### Le seguenti due funzioni mi servono per creare la griglia di immagini quando valuto sul test set
def denormalize_batch_images(images_t: torch.Tensor) -> np.ndarray:
    """
    images_t: (B, C, H, W) normalizzate con IMAGENET mean/std.
    ritorna: np.ndarray (B, H, W, 3) in [0,1]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=images_t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images_t.device).view(1, 3, 1, 1)
    imgs = images_t * std + mean
    imgs = imgs.clamp(0, 1)
    imgs = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
    return imgs


@torch.no_grad()
def plot_test_grid(
    model: nn.Module,
    test_loader: DataLoader,
    num_outputs_modello: int,
    img_size: int,
    n_images: int = 10,
    cols: int = 5,
    save_path: str | None = None,
    title: str = ""
) -> None:
    """
    Crea una griglia (es. 10 immagini) dal test_loader con keypoints predetti e reali.
    - Predetti: rosso (x)
    - Reali: verde (o)
    """
    model.eval()

    # prendo un batch dal test loader
    batch = next(iter(test_loader))
    images, keypoints_true = batch
    images = images.to(DEVICE)
    keypoints_true = keypoints_true.to(DEVICE)

    # forward
    outputs = model(images)

    # clamp SOLO per visualizzazione (come fai nelle metriche)
    outputs_viz = outputs.clamp(0.0, 1.0)

    # denormalizzo immagini
    imgs_den = denormalize_batch_images(images)  # (B,H,W,3)

    # denormalizzo keypoints (true e pred) -> pixel
    kp_true_px = (keypoints_true.view(-1, num_outputs_modello, 2).detach().cpu().numpy()) * img_size
    kp_pred_px = (outputs_viz.view(-1, num_outputs_modello, 2).detach().cpu().numpy()) * img_size

    # preparo griglia
    B = imgs_den.shape[0]
    n = min(n_images, B)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")

        if i >= n:
            continue

        ax.imshow(imgs_den[i])

        # predetti
        ax.scatter(kp_pred_px[i, :, 0], kp_pred_px[i, :, 1],
                   c="red", marker="x", s=35, label="Predetti")

        # reali
        ax.scatter(kp_true_px[i, :, 0], kp_true_px[i, :, 1],
                   c="lime", marker="o", s=20, label="Reali")

        # opzionale: linee tra vero e predetto
        for t, p in zip(kp_true_px[i], kp_pred_px[i]):
            ax.plot([t[0], p[0]], [t[1], p[1]], color="orange", linestyle="--", linewidth=1)

        if i == 0:  # legenda solo una volta
            ax.legend(loc="lower right")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)