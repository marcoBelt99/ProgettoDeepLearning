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

from parametri_modello import IMG_SIZE, DEVICE

from configs.parametri_app import CHECKPOINTS_DIR


from timeit import default_timer as timer # timer per monitorare il tempo che impiego ad allenare il modello



def calcola_mae_pixel(predictions, targets, img_size):
    """
    Calcola il MAE come metrica per i pixel
    """
    pred_pixels = predictions * img_size # moltiplicando per img_size vado a denormalizzare e passo da valori in [0,1] a valori in [0, 255]
    target_pixels = targets * img_size #   moltiplicando per img_size vado a denormalizzare e passo da valori in [0,1] a valori in [0, 255]

    return torch.abs(pred_pixels - target_pixels) .mean(dim=1).mean().item() # nuovo

def mean_euclidean_distance(preds, targets, img_size, num_outputs_modello):
    # preds = preds.view(-1, 14, 2) * img_size # moltiplicando per img_size vado a denormalizzare, passando da valori in [0,1] ai veri valori assunti dai pixel
    preds = preds.view(-1, num_outputs_modello, 2) * img_size # moltiplicando per img_size vado a denormalizzare, passando da valori in [0,1] ai veri valori assunti dai pixel
    # targets = targets.view(-1, 14, 2) * img_size
    targets = targets.view(-1, num_outputs_modello, 2) * img_size
    dists = torch.norm(preds - targets, dim=2)
    return dists.mean().item()



# Train di un'epoca
def train( writer : SummaryWriter,
           model : nn.Module,
           train_loader : DataLoader,
           device : torch.device,
           optimizer : torch.optim,
           criterion: Callable[[torch.Tensor, torch.Tensor], float],
           log_interval : int,
           epoch: int) -> float:
    '''
    Allena la rete neurale per una epoca.

    Args:
        model: il modello da allenare.
        train_loader: il data loader contenente i dati di training. Il train loader serve per caricare
                      i vari batch dai dati di training
        device: il device da usare per allenare il modello
        optimizer: l'ottimizzatore da usare per allenare il modello. (Potrebbe essere Adam, SGD, etc.)
        criterion: la loss da ottimizzare.
        log_interval: intervallo logaritmico. Questa è una costante per plottare gli intervalli. (serve per printare o non printare stuff)
        epoch: numero dell'epoca corrente.
    '''

    samples_train = 0
    loss_train = 0
    # size_ds_train = len(train_loader.dataset)
    num_batches = len(train_loader)

    # E' importante mettere il modello in modalità train() prima di iniziare il training!
    # questo al fine del Dropout e della BatchNormalization: perchè durante il training uso il
    # Dropout, e vado a modificare il valore per la batch norm
    model.train()

    for idx_batch, (images, keypoints) in enumerate(train_loader):
        images, keypoints = images.to(device), keypoints.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, keypoints)

        loss_train += loss.item() * len(images) # tensore.item() mi fa passare da tensore[ numero ] a numero
        samples_train += len(images)

        loss.backward()

        optimizer.step()

        # Gestione intervallo di logging per tensorboard
        if log_interval > 0:  # se definisco l'intervallo per loggare
            if idx_batch % log_interval == 0:  # allora ogni volta che raggiungo un punto faccio le seguenti cose
                running_loss = loss_train / samples_train
                global_step = idx_batch + (epoch * num_batches)
                writer.add_scalar('Metrics/Loss_Train_IT', running_loss,
                                  global_step)  # running loss e global step rappresentano: valore della loss a quell'iterazione e numero di iterazione

                # TODO: opzionale: visualizzo immagini su tensorboard
                # indices_random = torch.randperm(images.size(0))[:4]  # seleziono alcune immagini random
                # writer.add_images('Samples/Train', denormalize(images[indices_random]), global_step)

    # Alla fine del for, quando ho finito
    loss_train /= samples_train # calcolo la loss
    # TODO: valutare se inserire anche come metrica per il train anche mae e/o med
    return loss_train


# Validazione su una epoca
def validate(model : nn.Module,
             data_loader : DataLoader,
             device : torch.device,
             criterion : Callable[[torch.Tensor, torch.Tensor], float],
             num_outputs_modello : int
             ) -> Tuple[float, float, float]:
    """
     Valuta il modello.

     Args:
       model: modello da valutare.
       data_loader: il data loader contenente i dati di validation o di test.
       device: il device da usare per valutare il modello.
       criterion: la funzione di loss.

     Returns:
       (il valore di loss sui dati di validazione,
       il valore di MAE sui dati di validazione,
       il valore di MED sui dati di validazione)
     """
    samples_val = 0
    loss_val = 0.
    mae_val = 0.0
    med_val = 0.0

    # Metto il modello in modalità valutazione, quindi non supporto più nè Dropout (eventuale), nè BatchNorm
    model = model.eval()

    with torch.no_grad():  # inoltre, chiedo affinchè non venga computato il gradiente
        for images, keypoints in data_loader: # enumerate da errore
            images, keypoints = images.to(device), keypoints.to(device)

            outputs = model(images)

            loss = criterion(outputs, keypoints)

            loss_val += loss.item() * len(images)
            samples_val += len(images)

            mae_val += calcola_mae_pixel(outputs, keypoints, IMG_SIZE) * images.size(0)
            med_val += mean_euclidean_distance(outputs, keypoints, IMG_SIZE, num_outputs_modello) * images.size(0)

    loss_val /= samples_val
    mae_val /= samples_val
    med_val /= samples_val

    return loss_val, mae_val, med_val



def plotta_loss(train_losses, val_losses):
    """
    Plotta il grafico delle due loss a confronto.
    """
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()



def training_loop(  writer : SummaryWriter, # writer di tensorboard
                    num_epochs : int,
                    optimizer : optim,
                    lr_scheduler : lr_scheduler,
                    log_interval : int,
                    model : nn.Module,
                    num_outputs_modello : int,
                    loader_train : DataLoader,
                    loader_val : DataLoader,
                    verbose : bool=True) -> Dict:
    '''
    Esegue il loop di training.

    Args:
        writer: il summary writer per tensorboard.
        num_epochs: il numero di epoche.
        optimizer: l'ottimizzatore da usare.
        lr_scheduler: lo scheduler per il learning rate.
        log_interval: intervallo per printare con tensoarboard.
        model: il modello da allenare.
        loader_train: il data loader contenente i dati di training.
        loader_val: il data loader contenente i dati di validazione.
    '''

    criterion = nn.MSELoss() # TODO: valutare se passarglielo come parametro
    loop_start = timer()

    ## Inizializzo delle liste per salvare, ad ogni epoca le mie loss e le distanze (mae, med) sul training set e sul validation set
    losses_values = []
    val_losses = [] # mi serve solo per stamparla con matplotlib
    # train_mae_values = [] # TODO: ha senso salvarmi anche per il testing le metriche come MAD e MAE ?
    # train_med_values = []
    val_mae_values = []
    val_med_values = []

    # TODO: implemento una strategia di best per salvarmi il miglior modello
    #  quello avente metriche migliori
    best_mae = float('inf')  # Inizializzo con infinito

    for epoch in range(1, num_epochs + 1):  # potevo fare anche in range(0, num_epochs), ma io ho shiftato il default, per farlo funzionare con tensorboard
        time_start = timer()  # chissenefrega..

        ## Richiamo la funzione train. Ovviamente, le passo il loader solo di training
        # TODO: pensare se è il sensato salvarmi anche per il testing delle metriche come MAD e MAE
        loss_train = train(writer, model, loader_train, DEVICE, optimizer, criterion, log_interval, epoch)

        ## Appena finisce il train, chiamo la validate
        loss_val, mae_val, med_val = validate(model, loader_val, DEVICE, criterion, num_outputs_modello)

        time_end = timer()

        # Salvo tutte le loss e le metriche di distanza
        losses_values.append(loss_train)
        val_losses.append(loss_val) # solo a scopo di usarla con matplotlib
        val_mae_values.append(mae_val)
        val_med_values.append(med_val)

        if mae_val < best_mae:
            best_mae = mae_val
            # Salvataggio del modello
            #torch.save( model.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_resnet18.pth.pth") )
            save_path = os.path.join(CHECKPOINTS_DIR, f"{writer.log_dir.split(os.sep)[-1]}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🟢 Salvato miglior modello: {save_path}")

        ## Se uso un l.r. scheduler, allora:
        #     se voglio mostrare il l.r. nel mio plots in tensorboard o a console,
        #     io devo chiedere all'ottimizzatore per quale l.r. sto usando a questo momento; questo dipende da
        #     che tipo di scheduler sto usando. Qui sto usando un "simple step scheduler", in cui definisco di quanto
        #     voglio cambiare il l.r.
        lr = optimizer.param_groups[0]['lr']  # per ottenere il l.r. devo chiederlo all'optimizer, accedendo a quella posizione [0]['lr']

        # Metto questi report per mostrare le loss (formattate) e le metriche
        if verbose:
            print(f'Epoca: {epoch}\t'
                  f'Lr: {lr:.8f}\t'
                  f'Loss: Train = [{loss_train:.4f}] - Val = [ {loss_val:.4f}]\t'
                  # TODO: ha senso mettere anche mae e med per train?
                  # f'Accuratezza: Train = [{accuracy_train:.2f}] - Val = [{accuracy_val:.2f}]'
                  f'Tempo di una epoca (s): {(time_end - time_start):.4f}')

        # Plot su tensorboard
        writer.add_scalar('Iperparametri/Learning Rate', lr, epoch)  # (lo step è l'epoca)
        writer.add_scalars('Metriche/Losses', {"Train": loss_train, "Val": loss_val}, epoch)
        # TODO: ha senso mettere anche mae e med per train?
        # writer.add_scalars('Metriche/Accuratezza', {"Train": accuracy_train, "Val": accuracy_val}, epoch)
        writer.flush()  # flusho, così la prossima volta posso scrivere un nuovo valore

        # Ogni volta in questa epoca incremento lo step
        if lr_scheduler:
            lr_scheduler.step()

        loop_end = timer()
        time_loop = loop_end - loop_start

        if verbose:
            print(f'Tempo per {num_epochs} epoche (s): {(time_loop):.3f}')

    plotta_loss(losses_values, val_losses)
    plt.show()

    # Ritorno un dizionario con le metriche
    return {'loss_values': losses_values,
            'val_mae_values': val_mae_values,
            'val_med_values': val_med_values,
            'time': time_loop
            }





def execute(name_train: str,
            network : nn.Module,
            starting_lr : float,
            optimizer,
            num_epochs: int,
            num_outputs_modello: int,
            data_loader_train : DataLoader,
            data_loader_val : DataLoader) -> None:

    """
    Esegue il training loop.

    Args:
        name_train: il nome per la subfolder di log.
        network: la rete da allenare.
        starting_lr: il learning rate di partenza.
        num_epochs: il numero di epoche.
        data_loader_train: il data loader con i dati di training.
        data_loader_val: il data loader con i dati di validazione.
    """

    # Visualizzazione (per tensorboard)
    log_interval = 20  # quello che mi consente di vedere i risultati in tensorboard o in console
    log_dir = os.path.join("logs", name_train)  # directory in cui voglio salvare i logs
    writer = SummaryWriter(log_dir)  # istanzio il SummaryWriter, a cui passo la cartella su cui deve leggere le loss


    # Ottimizzazione da passare alla rete.
    # optimizer = optim.SGD(network.parameters(), lr=starting_lr, momentum=0.9, weight_decay=0.0001)

    # TODO: con Adam sto ottenendo buoni risultati e migliori rispetto a SGD

    # optimizer = torch.optim.Adam([
    #     {"params": network.layer3.parameters(), "lr": 5e-5},
    #     {"params": network.layer4.parameters(), "lr": 1e-4},
    #     {"params": network.fc.parameters(), "lr": 1e-3},
    # ])

    ## Learning Rate schedule: decade il learning rate di un fattore `gamma` ogni `step_size` epoche.
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5,
    #                                gamma=0.1)  # quindi, qui sto usando uno StepLR in cui cambio il lr moltiplicandolo per 0.1 ogni 5 epoche

    scheduler = None # lascindolo a None dovrei ottenere lo stesso effetto senza scheduler. Se uso Adam, lui già di suo lo implementa

    # Richiamo il loop di training, e salvo le statistiche (il dizionario che mi viene ritornato)
    statistics = training_loop(writer, num_epochs, optimizer, scheduler, log_interval, network, num_outputs_modello , data_loader_train,
                               data_loader_val)

    # Quando finisce, chiudo il writer
    writer.close()

    # Visualizzo statistiche
    best_epoch = np.argmin(statistics['val_mae_values']) + 1
    best_mae = statistics['val_mae_values'][best_epoch - 1]

    print(f'Miglior valore di MAE: {best_mae:.2f} epoca: {best_epoch}.')