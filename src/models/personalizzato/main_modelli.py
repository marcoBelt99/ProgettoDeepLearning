import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsummary import summary


from configs.parametri_app import DATAFRAME_MASTER, RAGGRUPPAMENTI, DATASET_DIR, NOMI_PUNTI_RAGGRUPPAMENTI
from src.data.dataset.repere_dataset_1 import RepereKeypointsDataset
from trasformazioni import trasformazioni_custom
from train_and_validate import execute
from parametri_modello import (
    IMG_SIZE,
    BATCH_SIZE,
    DEVICE,
    LR,
    EPOCHS
)


from specialized_models import ModelloGruppo1, ModelloGruppo2, ModelloGruppo3, ModelloGruppo4


#  FUNZIONE HELPER PER SELEZIONARE IL MODELLO
def get_specialized_model(nome_gruppo):
    """
    Restituisce l'istanza della classe specifica per il gruppo richiesto.
    """
    if nome_gruppo == "GRUPPO1":
        # Cranio: ResNet18 parzialmente freezata
        return ModelloGruppo1(pretrained=True)

    elif nome_gruppo == "GRUPPO2":
        # Naso/Mandibola: ResNet18 sbloccata quasi tutta
        return ModelloGruppo2(pretrained=True)

    elif nome_gruppo == "GRUPPO3":
        # Denti: ResNet18 con head profonda (o architettura custom)
        return ModelloGruppo3(pretrained=True)

    elif nome_gruppo == "GRUPPO4":
        # Mento: MobileNetV3 (veloce ed efficiente)
        return ModelloGruppo4(pretrained=True)

    else:
        raise ValueError(f"Gruppo {nome_gruppo} non riconosciuto o non implementato.")


def train_experiment_for_group(
        nome_esperimento: str,
        nome_gruppo: str,
        lista_indici_punti_gruppo: list[int],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        lr: float = LR,
        num_epoche: int = EPOCHS,
):
    print(f"\nTraining esperimento: {nome_esperimento}")

    # Dataset (Invariati)
    train_set = RepereKeypointsDataset(
        train_df, img_dir=DATASET_DIR, img_size=IMG_SIZE,
        trasformazioni=trasformazioni_custom["train"],
        gruppo_punti=lista_indici_punti_gruppo
    )

    val_set = RepereKeypointsDataset(
        val_df, img_dir=DATASET_DIR, img_size=IMG_SIZE,
        trasformazioni=trasformazioni_custom["test"],
        gruppo_punti=lista_indici_punti_gruppo
    )

    test_set = RepereKeypointsDataset(
        test_df, img_dir=DATASET_DIR, img_size=IMG_SIZE,
        trasformazioni=trasformazioni_custom["test"],
        gruppo_punti=lista_indici_punti_gruppo
    )

    # DataLoader (Invariati)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                              pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                             pin_memory=(DEVICE == "cuda"))

    ##################################################
    #### 1) ISTANZIO IL MODELLO SPECIALIZZATO
    ##################################################

    rete = get_specialized_model(nome_gruppo).to(DEVICE)

    # Nota: la summary potrebbe stampare output diversi ora che le architetture cambiano
    try:
        summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))
    except Exception as e:
        print(f"Summary non disponibile (forse per MobileNet/Layer imprevisti): {e}")


    ##################################################
    #### 2) Ottimizzatore
    ##################################################
    # Le nuove classi hanno già impostato
    # 'requires_grad=False' sui layer che devono stare fermi.
    # Quindi filter() prenderà automaticamente solo i layer che ho deciso di allenare.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, rete.parameters()),
        lr=lr
    )

    ##################################################
    ##### 3) Training
    ##################################################
    execute(name_train=nome_esperimento,
            rete=rete,
            starting_lr=lr,
            optimizer=optimizer,
            num_epochs=num_epoche,
            num_outputs_modello=len(lista_indici_punti_gruppo),
            data_loader_train=train_loader,
            data_loader_val=val_loader,
            data_loader_test=test_loader)


def allena_modello_by_nome_gruppo(nome_gruppo,
                                  train_df,
                                  val_df,
                                  test_df,
                                  lr,
                                  num_epoche
                                  ):
    # Aggiorno il nome dell'esperimento per riflettere che è "specialized"
    nome_esperimento = f"{nome_gruppo}_specialized_model_lr{lr}_ep{num_epoche}"

    print(f"\nGruppo di punti di interesse: {NOMI_PUNTI_RAGGRUPPAMENTI[nome_gruppo]}.\n")

    train_experiment_for_group(
        nome_esperimento=nome_esperimento,
        nome_gruppo=nome_gruppo,
        lista_indici_punti_gruppo=RAGGRUPPAMENTI[nome_gruppo],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        lr=lr,
        num_epoche=num_epoche
    )


def main():

    ## Carico il dataframe master
    df = pd.read_csv(DATAFRAME_MASTER)

    ### Splitting
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    print(f"Dimensione Training set: {len(train_df)}")
    print(f"Dimensione Validation set: {len(val_df)}")
    print(f"Dimensione Test  set: {len(test_df)}")



    allena_modello_by_nome_gruppo(
        "GRUPPO3",
        train_df,
        val_df,
        test_df,
        lr=1e-4,
        num_epoche=EPOCHS
    )

    # Se voglio allenare il GRUPPO 3, basta demmentare:
    # allena_modello_by_nome_gruppo("GRUPPO3", train_df, val_df, test_df, lr=1e-4, num_epoche=EPOCHS)


if __name__ == "__main__":
    main()