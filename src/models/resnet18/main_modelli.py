import torch
import pandas as pd
from torch.utils.data import DataLoader


from configs.parametri_app import DATAFRAME_MASTER, RAGGRUPPAMENTI, DATASET_DIR
from parametri_modello import IMG_SIZE, TRAIN_RATIO, BATCH_SIZE, DEVICE, NUM_TOTALE_PUNTI, LR, EPOCHS
from src.data.dataset.repere_dataset import crea_split_dataset
from factory import build_model_for_group
from torchsummary import summary
from train_and_validate import execute




def train_experiment_for_group(nome_esperimento : str,
                               train_loader,
                               test_loader,
                               nome_gruppo : str,
                               # lista_indici_punti_gruppo : list[int],
                               head="linear",
                               freeze_until="layer3",
                               lr=LR,

                               ):

                     #num_outputs=NUM_TOTALE_PUNTI * 2 ):
    """
    Rifattorizzazione di TUTTE le tue funzioni di training in una sola.
    Esegue quindi un esperimento di training usando la factory.
    """

    # 1) Creo il modello, scegliendo anche quali layer sbloccare
    # rete = build_resnet18(num_outputs=num_outputs,
    #                       pretrained=True,
    #                       head=head,
    #                       freeze_until=freeze_until
    # ).to(DEVICE)



    rete = build_model_for_group(nome_gruppo,
                                 RAGGRUPPAMENTI,
                                  head="linear",
                                  freeze_until="layer3"
                                  ).to(DEVICE)

    summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))

    # 2) Ottimizzatore
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, rete.parameters()), lr=lr)
    # optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    # 4) Training
    execute(nome_esperimento, rete, lr, optimizer, EPOCHS,len(RAGGRUPPAMENTI[nome_gruppo]), train_loader, test_loader)




def main():

    df = pd.read_csv(DATAFRAME_MASTER)

    for nome_gruppo, indici_punti in RAGGRUPPAMENTI.items():
        print(f"\nAvvio training per {nome_gruppo}")

        train_set, test_set = crea_split_dataset(
            df,
            img_dir=DATASET_DIR,
            img_size=IMG_SIZE,
            train_ratio=TRAIN_RATIO,
            gruppo_punti=indici_punti
        )

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


        # num_outputs = len(indici_punti) * 2

        train_experiment_for_group(
            nome_esperimento=f"{nome_gruppo}_resnet18",
            train_loader=train_loader,
            test_loader=test_loader,
            nome_gruppo=nome_gruppo,
            head="linear",
            freeze_until="layer3",
            lr=LR,


            # num_outputs=num_outputs
        )


if __name__ == "__main__":
    main()