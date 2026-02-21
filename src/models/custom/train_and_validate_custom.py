import os
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from utils.early_stopping import EarlyStopping
from utils.metriche import calcola_mae_pixel, mean_euclidean_distance
from utils.logger import log_experiment

from parametri_modello import DEVICE, IMAGENET_MEAN, IMAGENET_STD, GRAD_CLIP_NORM


def _denormalize_batch_images(images_t: torch.Tensor) -> np.ndarray:
    """
    images_t: (B, C, H, W) normalizzate con ImageNet mean/std
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
    num_points: int,
    img_size: int,
    save_path: str,
    n_images: int = 10,
    cols: int = 5,
    title: str = ""
) -> None:
    model.eval()
    batch = next(iter(test_loader))
    images, keypoints_true = batch
    images = images.to(DEVICE)
    keypoints_true = keypoints_true.to(DEVICE)

    outputs = model(images)
    outputs_viz = outputs.clamp(0.0, 1.0)

    imgs_den = _denormalize_batch_images(images)
    kp_true_px = (keypoints_true.view(-1, num_points, 2).detach().cpu().numpy()) * img_size
    kp_pred_px = (outputs_viz.view(-1, num_points, 2).detach().cpu().numpy()) * img_size

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
        ax.scatter(kp_pred_px[i, :, 0], kp_pred_px[i, :, 1], c="red", marker="x", s=35)
        ax.scatter(kp_true_px[i, :, 0], kp_true_px[i, :, 1], c="lime", marker="o", s=25)
        ax.set_title(f"{i}", fontsize=10)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_one_epoch(
    writer: SummaryWriter,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epoch: int,
    log_interval: int = 0,
) -> float:
    model.train()
    samples = 0
    loss_acc = 0.0
    num_batches = len(train_loader)

    for i, (images, keypoints) in enumerate(train_loader):
        images = images.to(DEVICE)
        keypoints = keypoints.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, keypoints)

        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        loss_acc += loss.item() * len(images)
        samples += len(images)

        if log_interval and (i % log_interval == 0):
            global_step = i + epoch * num_batches
            writer.add_scalar("Metrics/Loss_Train_IT", loss_acc / max(1, samples), global_step)

    return loss_acc / max(1, samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_points: int,
    img_size: int
) -> Tuple[float, float, float]:
    model.eval()
    samples = 0
    loss_acc = 0.0
    mae_acc = 0.0
    med_acc = 0.0

    for images, keypoints in loader:
        images = images.to(DEVICE)
        keypoints = keypoints.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, keypoints)

        loss_acc += loss.item() * len(images)
        samples += len(images)

        outputs_m = outputs.clamp(0.0, 1.0)  # clamp solo per metriche
        mae_acc += calcola_mae_pixel(outputs_m, keypoints, img_size) * len(images)
        med_acc += mean_euclidean_distance(outputs_m, keypoints, img_size, num_points) * len(images)

    loss_mean = loss_acc / max(1, samples)
    mae_mean = mae_acc / max(1, samples)
    med_mean = med_acc / max(1, samples)
    return loss_mean, mae_mean, med_mean


def execute(
    name_train: str,
    nome_gruppo: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    starting_lr: float,
    num_epochs: int,
    num_points: int,
    img_size: int,
    batch_size: int,
    freeze_until: str,
    head: str,
    output_root: str,
    data_loader_train: DataLoader,
    data_loader_val: DataLoader,
    data_loader_test: DataLoader,
    scheduler_cfg: dict,
    early_stop_cfg: dict,
    seed: int = 42,
):
    """Training completo (train/val + early stopping + test + output su cartelle del gruppo)."""

    out_root = Path(output_root)
    logs_dir = out_root / "logs" / name_train
    plots_dir = out_root / "plots"
    ckpt_dir = out_root / "checkpoints"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=str(logs_dir))

    # Loss: SmoothL1 con beta ~ 1px
    criterion = nn.SmoothL1Loss(beta=1.0 / float(img_size))

    # Scheduler
    sched = None
    if scheduler_cfg and scheduler_cfg.get("type") == "plateau":
        sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 4)),
            # verbose=True
        )

    # Early stopping: salva state_dict best
    best_ckpt = ckpt_dir / f"{name_train}_BEST_EARLY.pth"
    early = EarlyStopping(
        patience=int(early_stop_cfg.get("patience", 10)),
        min_delta=float(early_stop_cfg.get("min_delta", 1e-4)),
        checkpoint_path=str(best_ckpt)
    )

    val_mae_values: List[float] = []
    start_time = timer()

    for epoch in range(num_epochs):
        loss_train = train_one_epoch(writer, model, data_loader_train, optimizer, criterion, epoch, log_interval=0)

        loss_val, mae_val, med_val = evaluate(model, data_loader_val, criterion, num_points, img_size)
        val_mae_values.append(mae_val)

        writer.add_scalar("Metrics/Loss_Train", loss_train, epoch)
        writer.add_scalar("Metrics/Loss_Val", loss_val, epoch)
        writer.add_scalar("Metrics/MAE_Val_px", mae_val, epoch)
        writer.add_scalar("Metrics/MED_Val_px", med_val, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"[{epoch+1:03d}/{num_epochs}] "
              f"train_loss={loss_train:.6f} | val_loss={loss_val:.6f} | val_mae={mae_val:.3f}px | val_med={med_val:.3f}px | lr={optimizer.param_groups[0]['lr']:.2e}")

        if sched is not None:
            sched.step(mae_val)

        early(mae_val, model)
        if early.early_stop:
            print("Early stopping attivato.")
            break

    elapsed = timer() - start_time
    epochs_run = len(val_mae_values)
    best_epoch = int(np.argmin(val_mae_values)) + 1
    best_mae = float(np.min(val_mae_values))

    # carico best model e valuto su test
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    loss_test, mae_test, med_test = evaluate(model, data_loader_test, criterion, num_points, img_size)

    print(f"TEST: loss={loss_test:.6f} | mae={mae_test:.3f}px | med={med_test:.3f}px")
    print(f"Tempo totale: {elapsed:.1f}s (epoche eseguite: {epochs_run})")

    # plot curve val mae
    fig = plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs_run + 1), val_mae_values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE (px)")
    plt.title(f"{name_train} - Val MAE")
    plt.grid(True)
    curve_path = plots_dir / f"{name_train}_val_mae.png"
    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    # plot test grid
    grid_path = plots_dir / f"{name_train}_test_grid.png"
    try:
        plot_test_grid(model, data_loader_test, num_points, img_size, save_path=str(grid_path), title=f"{nome_gruppo} - {name_train}")
    except Exception as e:
        print("Plot test grid fallito:", e)

    # log CSV (nel gruppo)
    csv_path = out_root / "log_esperimenti.csv"
    log_experiment(str(csv_path), {
        "esperimento": name_train,
        "best_epoch": best_epoch,
        "best_mae": best_mae,
        "lr": starting_lr,
        "batch_size": batch_size,
        "epochs_run": epochs_run,
        "freeze_until": freeze_until,
        "head": head,
        "img_size": img_size,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": (scheduler_cfg.get("type") if scheduler_cfg else ""),
        "seed": seed,
    })

    writer.close()
