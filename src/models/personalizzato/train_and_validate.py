import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Tuple, Union
from torch.optim import lr_scheduler

# Assicurati che questi import puntino ai file corretti nel tuo progetto
from parametri_modello import IMG_SIZE, DEVICE
from configs.parametri_app import CHECKPOINTS_DIR
from utils.early_stopping import EarlyStopping
from utils.logger import log_experiment
from utils.metriche import *
from timeit import default_timer as timer
from utils.utils import plot_test_grid


# --- Train di un'epoca ---
def train(writer: SummaryWriter,
          model: nn.Module,
          train_loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          criterion: Callable[[torch.Tensor, torch.Tensor], float],
          log_interval: int,
          epoch: int) -> float:
    '''
    Allena la rete neurale per una epoca.
    '''

    samples_train = 0
    loss_train = 0
    num_batches = len(train_loader)

    model.train()

    for idx_batch, (images, keypoints) in enumerate(train_loader):
        images, keypoints = images.to(device), keypoints.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, keypoints)

        loss_train += loss.item() * len(images)
        samples_train += len(images)

        loss.backward()
        optimizer.step()

        # Logging Tensorboard
        if log_interval > 0:
            if idx_batch % log_interval == 0:
                running_loss = loss_train / samples_train
                global_step = idx_batch + (epoch * num_batches)
                writer.add_scalar('Metrics/Loss_Train_IT', running_loss, global_step)

    loss_train /= samples_train
    return loss_train


# --- Validazione su una epoca ---
def validate(model: nn.Module,
             data_loader: DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float],
             num_outputs_modello: int
             ) -> Tuple[float, float, float]:
    """
     Valuta il modello e ritorna Loss, MAE e MED.
     """
    samples_val = 0
    loss_val = 0.
    mae_val = 0.0
    med_val = 0.0

    model = model.eval()

    with torch.no_grad():
        for images, keypoints in data_loader:
            images, keypoints = images.to(device), keypoints.to(device)

            outputs = model(images)

            loss = criterion(outputs, keypoints)
            loss_val += loss.item() * len(images)
            samples_val += len(images)

            # Metriche su output clampato (0-1) per reporting reale in pixel
            outputs_m = outputs.clamp(0.0, 1.0)

            mae_val += calcola_mae_pixel(outputs_m, keypoints, IMG_SIZE) * images.size(0)
            med_val += mean_euclidean_distance(outputs_m, keypoints, IMG_SIZE, num_outputs_modello) * images.size(0)

    loss_val /= samples_val
    mae_val /= samples_val
    med_val /= samples_val

    return loss_val, mae_val, med_val


def plotta_loss(train_losses, val_losses):
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()


# --- Training Loop Principale ---
def training_loop(writer: SummaryWriter,
                  num_epochs: int,
                  optimizer: optim,
                  lr_scheduler: lr_scheduler,
                  log_interval: int,
                  model: nn.Module,
                  num_outputs_modello: int,
                  loader_train: DataLoader,
                  loader_val: DataLoader,
                  verbose: bool = True,
                  run_name: str = "",
                  save_plots: bool = True,
                  show_plots: bool = False) -> Dict:
    # Loss function: SmoothL1Loss è ottima per evitare che outlier "rompano" il gradiente
    criterion = nn.SmoothL1Loss(beta=1.0 / IMG_SIZE)
    loop_start = timer()

    if run_name == "":
        run_name = writer.log_dir.split(os.sep)[-1]

    plots_dir = "plots"
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    losses_values = []
    val_losses = []
    val_mae_values = []
    val_med_values = []

    # Early Stopping
    early_stopper = EarlyStopping(
        patience=10,
        min_delta=1e-4,
        checkpoint_path=os.path.join(CHECKPOINTS_DIR, f"{run_name}_BEST_EARLY.pth")
    )

    for epoch in range(1, num_epochs + 1):
        time_start = timer()

        # 1. TRAIN
        loss_train = train(writer, model, loader_train, DEVICE, optimizer, criterion, log_interval, epoch)

        # 2. VALIDATION
        loss_val, mae_val, med_val = validate(model, loader_val, DEVICE, criterion, num_outputs_modello)

        time_end = timer()

        losses_values.append(loss_train)
        val_losses.append(loss_val)
        val_mae_values.append(mae_val)
        val_med_values.append(med_val)

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoca: {epoch}/{num_epochs} | Lr: {current_lr:.8f} | '
                  f'Train Loss: {loss_train:.5f} | Val Loss: {loss_val:.5f} | '
                  f'MAE: {mae_val:.2f}px | MED: {med_val:.2f}px | '
                  f'Time: {(time_end - time_start):.2f}s')

        # Tensorboard Logging
        writer.add_scalar('Iperparametri/Learning Rate', current_lr, epoch)
        writer.add_scalars('Metriche/Losses', {"Train": loss_train, "Val": loss_val}, epoch)
        writer.add_scalar('Metriche/MAE_Val_px', mae_val, epoch)
        writer.add_scalar('Metriche/MED_Val_px', med_val, epoch)
        writer.flush()

        # Scheduler step (ReduceLROnPlateau lavora sulla metrica di validation)
        if lr_scheduler:
            lr_scheduler.step(mae_val)

        # Early Stopping check
        early_stopper(mae_val, model)
        if early_stopper.early_stop:
            print(f"Early stopping attivato all'epoca: {epoch}. Best MAE: {early_stopper.best_score:.4f}")
            break

    loop_end = timer()
    time_loop = loop_end - loop_start

    if verbose:
        print(f'Training completato in: {(time_loop):.3f} secondi')

    # Salvataggio Grafici finali
    # LOSS
    plt.figure()
    plotta_loss(losses_values, val_losses)
    plt.title(f"Loss - {run_name}")
    if save_plots:
        plt.savefig(os.path.join(plots_dir, f"{run_name}_loss.png"), dpi=150, bbox_inches="tight")
    if show_plots: plt.show()
    plt.close()

    # MAE
    plt.figure()
    plotta_mae(val_mae_values)
    plt.title(f"MAE (Val, px) - {run_name}")
    if save_plots:
        plt.savefig(os.path.join(plots_dir, f"{run_name}_mae.png"), dpi=150, bbox_inches="tight")
    if show_plots: plt.show()
    plt.close()

    return {
        'loss_values': losses_values,
        'val_mae_values': val_mae_values,
        'time': time_loop
    }


# --- EXECUTE (Funzione Wrapper) ---
def execute(name_train: str,
            rete: nn.Module,
            starting_lr: float,
            optimizer,
            num_epochs: int,
            num_outputs_modello: int,
            data_loader_train: DataLoader,
            data_loader_val: DataLoader,
            data_loader_test: DataLoader) -> None:
    """
    Esegue il training loop e gestisce il logging finale.
    """

    log_interval = 20
    log_dir = os.path.join("logs", name_train)
    writer = SummaryWriter(log_dir)

    # Scheduler (Generico per tutti i modelli, si adatta se il loss non scende)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,  # Leggermente aumentata la pazienza
    )

    # Avvio Training Loop
    statistics: dict = training_loop(
        writer, num_epochs, optimizer, scheduler, log_interval,
        rete, num_outputs_modello,
        data_loader_train, data_loader_val,
        run_name=name_train,
        save_plots=True,
        show_plots=False
    )

    writer.close()

    # Recupero miglior risultato
    best_epoch = np.argmin(statistics['val_mae_values']) + 1
    best_mae = statistics['val_mae_values'][best_epoch - 1]
    print(f'Miglior MAE Validation: {best_mae:.2f} (Epoca: {best_epoch})')

    # --- VALUTAZIONE SU TEST SET (Usando il Best Model) ---
    best_path = os.path.join(CHECKPOINTS_DIR, f"{name_train}_BEST_EARLY.pth")

    if os.path.exists(best_path):
        # Ricarico i pesi migliori
        rete.load_state_dict(torch.load(best_path, map_location=DEVICE))
        rete.to(DEVICE)

        criterion = nn.SmoothL1Loss(beta=1.0 / IMG_SIZE)

        loss_test, mae_test, med_test = validate(
            rete, data_loader_test, DEVICE, criterion, num_outputs_modello
        )

        print(f"\n>>> RISULTATI TEST (Best Checkpoint) <<<")
        print(f"Loss: {loss_test:.4f} | MAE: {mae_test:.3f} px | MED: {med_test:.3f} px\n")

        # Test Grid Plot
        try:
            grid_path = os.path.join("plots", f"{name_train}_testgrid.png")
            plot_test_grid(
                model=rete,
                test_loader=data_loader_test,
                num_outputs_modello=num_outputs_modello,
                img_size=IMG_SIZE,
                n_images=10,
                cols=5,
                save_path=grid_path,
                title=f"TEST - {name_train}"
            )
        except Exception as e:
            print(f"[WARN] Impossibile creare test grid: {e}")

    else:
        print(f"[ERROR] Checkpoint {best_path} non trovato!")

    # --- LOGGING SUL CSV ---
    # Qui gestiamo il fatto che i modelli nuovi non hanno attributi espliciti come 'head' o 'freeze_until'
    # Recuperiamo il nome della classe (es. Group4Model)
    model_class_name = rete.__class__.__name__

    log_experiment(
        csv_path="log_esperimenti.csv",
        data_dict={
            "esperimento": name_train,
            "tot_epochs": num_epochs,
            "best_epoch": best_epoch,
            "best_mae": float(best_mae),
            "loss_function": "SmoothL1Loss",
            "lr": starting_lr,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__ if scheduler else "None",
            "batch_size": data_loader_train.batch_size,
            "epochs_run": len(statistics["loss_values"]),

            # MODIFICATO: Invece di cercare attributi che non esistono, logghiamo info generiche o il nome classe
            "freeze_until": "Specialized",
            "head": model_class_name,  # Salvo il nome della classe qui (es. Group4Model) così sai quale hai usato
            "img_size": IMG_SIZE,
        }
    )