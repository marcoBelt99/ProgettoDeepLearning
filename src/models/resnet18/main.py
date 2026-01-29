import pandas as pd
from tensorboard import summary
from torch.utils.data import DataLoader
from torchsummary import summary

from configs.parametri_app import DATAFRAME_MASTER, DATASET_DIR, num_totale_punti
from altro.deprecati.repere_dataset import  crea_split_dataset # per splittare e creare il dataset usando come struttura dati il dataframe
from factory import *
from parametri_modello import *
from train_and_validate_1 import  execute # metodo che chiama il training loop. Il training loop cicla su tutte le epoche e fa prima training
# tante volte quanti sono i batch, e poi fa tante validation quanti sono i batch.


def train_experiment(nome_esperimento : str,
                     train_loader,
                     test_loader,
                     head="linear",
                     freeze_until="layer3",
                     lr=LR,
                     num_outputs=NUM_TOTALE_PUNTI * 2 ):
    """
    Rifattorizzazione di TUTTE le funzioni di training in una sola.
    Esegue quindi un esperimento di training usando la factory.
    """

    # 1) Creo il modello, scegliendo anche quali layer sbloccare
    rete = build_resnet18(num_outputs=num_outputs,
                          pretrained=True,
                          head=head,
                          freeze_until=freeze_until
    ).to(DEVICE)

    summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))

    # 2) Ottimizzatore
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, rete.parameters()), lr=lr)
    # optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    # 4) Training
    execute(nome_esperimento, rete, lr, optimizer, EPOCHS, num_totale_punti, train_loader, test_loader)



def main():

    #### (1) CARICAMENTO DATASET ######

    ## Sto caricando il dataframe master
    df = pd.read_csv(DATAFRAME_MASTER)

    ## Suddivisione dataset in train e test set
    train_set, test_set = crea_split_dataset(df, img_dir=DATASET_DIR, img_size=IMG_SIZE, train_ratio=TRAIN_RATIO)

    ## Preparazione dei data loader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, # provare con valori 8 / 16 / 32 / 64
                              shuffle=True, # è bene mescolare i dati solo per il train
                              # pin_memory=True, # speed-up CPU-GPU traning: utile a scopo di training, è un miglioramento a livello di memoria
                              num_workers=2) # di solito è 2 o 4

    test_loader = DataLoader(test_set,
                             batch_size=BATCH_SIZE,
                             shuffle=False, # non si mescola per il test loader
                             num_workers=2)

    #### (2) ESPERIMENTI

    ## (A) ALLENAMENTO RETE DA 0
    # allena_rete_da_zero(train_loader, test_loader) # TODO: devo migliorare questo. Ci mette troppo tempo
    #train_experiment("scratch", train_loader, test_loader, head="linear", freeze_until=None)


    ## (B) TL: alleno solo ultimo layer
    # allena_pesi_solo_ultimo_layer(train_loader, test_loader) # TODO: fa schifo
    # train_experiment("solo_fc", train_loader, test_loader, head="linear", freeze_until="fc")

    ## (C) TL: alleno solo ultimo layer + layer 4
    # allena_pesi_solo_layers_4_and_ultimofc(train_loader, test_loader) # TODO: da migliorare
    # train_experiment("layer4_fc", train_loader, test_loader, head="linear", freeze_until="layer4")

    ## (D) TL: alleno solo ultimo layer + layer 4 + layer 3
    # allena_pesi_solo_layers_3_and_4_and_ultimofc(train_loader, test_loader)
    train_experiment("layer3_4_fc", train_loader, test_loader, head="linear", freeze_until="layer3")

    ## (E) TL: alleno la mia head MLP
    # train_experiment("layer4_fc", train_loader, test_loader, head="mlp", freeze_until="layer3")


if __name__ == "__main__":
    main()