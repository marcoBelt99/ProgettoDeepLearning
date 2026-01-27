import torch
import pandas as pd
from torch.utils.data import DataLoader

from configs.parametri_app import DATAFRAME_MASTER, RAGGRUPPAMENTI, DATASET_DIR
from models.resnet18.trasformazioni import resnet18_data_transforms
from parametri_modello import (
    IMG_SIZE,
    TRAIN_RATIO,
    BATCH_SIZE,
    DEVICE,
    LR,
    EPOCHS
)

from src.data.dataset.repere_dataset import RepereKeypointsDataset
from factory import build_model_for_group
from train_and_validate import execute
from torchsummary import summary




def train_experiment_for_group(
        nome_esperimento : str,
        nome_gruppo : str,
        lista_indici_punti_gruppo : list[int], # specifici punti del gruppo
        train_df : pd.DataFrame,
        test_df : pd.DataFrame,
        head : str ="linear",
        freeze_until : str ="layer3",
        lr : float =LR,
        ):

                     #num_outputs=NUM_TOTALE_PUNTI * 2 ):
    """
    Rifattorizzazione di TUTTE le funzioni di training in una sola.
    Esegue quindi un esperimento di training usando la factory.
    """


    print(f"\nTraining esperimento: {nome_esperimento}")

    # Dataset
    train_set = RepereKeypointsDataset(
         train_df,
         img_dir=DATASET_DIR,
         img_size=IMG_SIZE,
         trasformazioni=resnet18_data_transforms["train"],
         gruppo_punti=lista_indici_punti_gruppo
    )

    test_set = RepereKeypointsDataset(
        test_df,
        img_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        trasformazioni=resnet18_data_transforms["test"],
        gruppo_punti=lista_indici_punti_gruppo
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    # 1) Creo il modello, scegliendo anche quali layer sbloccare
    rete = build_model_for_group(
        nome_gruppo,
        head=head,
        freeze_until=freeze_until,
    ).to(DEVICE)


    summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))

    # 2) Ottimizzatore
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, rete.parameters()),
        lr=lr)
    # optimizer = torch.optim.Adam(rete.parameters(), lr=LR)

    # 4) Training
    execute(name_train=nome_esperimento,
            rete=rete,
            starting_lr=lr,
            optimizer=optimizer,
            num_epochs=EPOCHS,
            num_outputs_modello=len(RAGGRUPPAMENTI[nome_gruppo]),
            data_loader_train=train_loader,
            data_loader_test=test_loader)



def main():

    ###################################
    # TODO: CONFIGURAZIONE ESPERIMENTO
    ###################################
    head = "linear" # tipo di testa ("linear" = ho direttamente la testa lineare; "mlp" = testa custom)
    freeze_until = "layer3" # fin quale layer freezare. In questo caso freeza fino al layer2 (compreso),
    # mentre allena layer3, layer4, fc
    lr = LR # learning rate usato



    ## Carico il dataframe master
    df = pd.read_csv(DATAFRAME_MASTER)

    ## Splitting (DA FARE UNA SOLA VOLTA)
    train_df = df.sample( frac=TRAIN_RATIO, random_state=42 )
    test_df = df.drop( train_df.index )

    print(f"Dimensione Training set: {len(train_df)}")
    print(f"Dimensione Test  set: {len(test_df)}")


    ## Ciclo su tutti gli indici dei punti
    for nome_gruppo, indici_punti in RAGGRUPPAMENTI.items():

       nome_esperimento = (f"{nome_gruppo}_resnet18_"f"{freeze_until}_{head}_lr{lr}")

       """
       Conviene chiamare così, in questo modo so sempre:
       - che backbone
       - che freeze
       - che head
       - che learning rate
       hanno prodotto quel risultato.
       """

       train_experiment_for_group(
           # nome_esperimento=f"{nome_gruppo}_resnet18",
           nome_esperimento=nome_esperimento,
           nome_gruppo=nome_gruppo,
           lista_indici_punti_gruppo=indici_punti,
           train_df=train_df,
           test_df=test_df,
           head=head,
           freeze_until=freeze_until,
           lr=LR
       )


if __name__ == "__main__":
    main()