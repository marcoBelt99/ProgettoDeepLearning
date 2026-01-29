import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.parametri_app import DATAFRAME_MASTER, RAGGRUPPAMENTI, DATASET_DIR, NOMI_PUNTI_RAGGRUPPAMENTI
from models.resnet18.trasformazioni_1 import resnet18_data_transforms # uso trasformazioni_1
from parametri_modello import (
    IMG_SIZE,
    TRAIN_RATIO,
    BATCH_SIZE,
    DEVICE,
    LR,
    EPOCHS
)

from src.data.dataset.repere_dataset_1 import RepereKeypointsDataset # uso repere_dataset_1
from factory import build_model_for_group
from train_and_validate_1 import execute
from torchsummary import summary




def train_experiment_for_group(
        nome_esperimento : str,
        nome_gruppo : str,
        lista_indici_punti_gruppo : list[int], # specifici punti del gruppo
        train_df : pd.DataFrame,
        val_df : pd.DataFrame,
        test_df : pd.DataFrame,
        head : str ="linear",
        freeze_until : str ="layer3",
        lr : float =LR,
        num_epoche : int=EPOCHS,
        ):

                     #num_outputs=NUM_TOTALE_PUNTI * 2 ):
    """
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

    val_set = RepereKeypointsDataset(
        val_df,
        img_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        trasformazioni=resnet18_data_transforms["test"], # solo trasformazioni per VAL e TEST
        gruppo_punti=lista_indici_punti_gruppo
    )

    test_set = RepereKeypointsDataset(
        test_df,
        img_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        trasformazioni=resnet18_data_transforms["test"], # solo trasformazioni per VAL e TEST
        gruppo_punti=lista_indici_punti_gruppo
    )

    # corrispettivi DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    ##################################################
    #### 1) Creo il modello, scegliendo anche quali layer sbloccare
    ##################################################
    rete = build_model_for_group(
        nome_gruppo,
        head=head,
        freeze_until=freeze_until,
    ).to(DEVICE)


    summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))

    ##################################################
    #### 2) Ottimizzatore
    ##################################################
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, rete.parameters()),
        lr=lr)
    # optimizer = torch.optim.Adam(rete.parameters(), lr=LR)


    ##################################################
    ##### 4) Training
    ##################################################
    execute(name_train=nome_esperimento,
            rete=rete,
            starting_lr=lr,
            optimizer=optimizer,
            # num_epochs=EPOCHS,
            num_epochs=num_epoche,
            num_outputs_modello=len(RAGGRUPPAMENTI[nome_gruppo]),
            data_loader_train=train_loader,
            data_loader_val=val_loader,
            data_loader_test=test_loader)





def allena_modello_by_nome_gruppo(nome_gruppo,
                                  train_df,
                                  val_df,
                                  test_df,
                                  head,
                                  freeze_until,
                                  lr,
                                  num_epoche  # Provo con 60 epoche per il gruppo 1):
                                  ):

    nome_esperimento = f"{nome_gruppo}_resnet18_"f"{freeze_until}_{head}_lr{lr}_epoche{num_epoche}"

    print(f"\nGruppo di punti di interesse: {NOMI_PUNTI_RAGGRUPPAMENTI[nome_gruppo]}.\n")

    """
    Conviene chiamare così, in questo modo so sempre:
    - che backbone
    - che freeze
    - che head
    - che learning rate
    hanno prodotto quel risultato.
    """

    train_experiment_for_group(
        nome_esperimento=nome_esperimento,
        nome_gruppo=nome_gruppo,
        lista_indici_punti_gruppo=RAGGRUPPAMENTI[nome_gruppo],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        head=head,
        freeze_until=freeze_until,
        lr=lr,  # lr=LR
        num_epoche=num_epoche  # Provo con 60 epoche per il gruppo 1
    )


def main():

    ###################################
    # TODO: CONFIGURAZIONE ESPERIMENTO
    ###################################
    head = "linear" # tipo di testa ("linear" = ho direttamente la testa lineare; "mlp" = testa custom)
    freeze_until = "layer3" # fin quale layer freezare. In questo caso (impostando il valore "layer3") freeza fino al layer2 (compreso),
    # mentre, appunto, allena: layer3, layer4, fc


    ## Carico il dataframe master
    df = pd.read_csv(DATAFRAME_MASTER)


    ### Splitting (DA FARE UNA SOLA VOLTA)

    # Vecchio
    # train_df = df.sample( frac=TRAIN_RATIO, random_state=42 )
    # test_df = df.drop( train_df.index )

    # Nuovo
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True) # 80% per il train_set, 20% di dati disponibili da dividere tra validation e test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True) # 10% va al test set, l'altro 10% va al validation set

    print(f"Dimensione Training set: {len(train_df)}")
    print(f"Dimensione Validation set: {len(val_df)}")
    print(f"Dimensione Test  set: {len(test_df)}")

    """
    ## Ciclo su tutti gli indici dei punti
    for nome_gruppo, indici_punti in RAGGRUPPAMENTI.items():

       nome_esperimento = (f"{nome_gruppo}_resnet18_"f"{freeze_until}_{head}_lr{lr}")
       print(f"\nGruppo di punti di interesse: {NOMI_PUNTI_RAGGRUPPAMENTI[nome_gruppo]}.\n")

       ###
       # Conviene chiamare così, in questo modo so sempre:
       # - che backbone
       # - che freeze
       # - che head
       # - che learning rate
       # hanno prodotto quel risultato.
       ###

       train_experiment_for_group(
           nome_esperimento=nome_esperimento,
           nome_gruppo=nome_gruppo,
           lista_indici_punti_gruppo=indici_punti,
           train_df=train_df,
           val_df=val_df,
           test_df=test_df,
           head=head,
           freeze_until=freeze_until,
           lr=lr, # lr=LR
           num_epoche=20 # Provo con 60 epoche per il gruppo 1
       )
       """

    allena_modello_by_nome_gruppo("GRUPPO1",
                                  train_df,
                                  val_df,
                                  test_df,
                                  head,
                                  freeze_until,
                                  lr=LR,
                                  num_epoche=10)




if __name__ == "__main__":
    main()