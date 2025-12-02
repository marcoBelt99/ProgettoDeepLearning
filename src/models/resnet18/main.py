import os
import pandas as pd
import torchvision
from tensorboard import summary
from torch.utils.data import DataLoader

from torchsummary import summary



from configs.parametri_app import DATAFRAME_MASTER, DATASET_DIR, CHECKPOINTS_DIR
from models.resnet18.modello_resnet18_all_points import ResNet18_Keypoints # modello di rete neurale che fa uso del TL
from src.data.dataset.repere_dataset import  crea_split_dataset # per splittare e creare il dataset usando come struttura dati il dataframe

from utils.utils import ok, fail

from parametri_modello import *

from train_and_validate import  execute # metodo che chiama il training loop. Il training loop cicla su tutte le epoche e fa prima training
# tante volte quanti sono i batch, e poi fa tante validation quanti sono i batch.



def allena_rete_da_zero(train_loader, test_loader):
    """
    1.
    Provo l'allenamento di una rete resnet18 da 0.
    Senza TL, ma solamente rimuovendo lo strato finale e adattandolo
    alle 28 coordinate che devo predire.

    Questa sarà una baseline da cui dovrò migliorare sempre più.
    """

    # Importo il modello senza pesi (weights=None)
    net_from_scratch = torchvision.models.resnet18(weights=None)

    # Sostituisco le 512 features con i miei 28 valori di coordinate
    num_features: int = net_from_scratch.fc.in_features  # mi aspetto sia 512
    print(f"\nL'ultimo layer di resnet18 ha: {num_features} features.\n")
    net_from_scratch.fc = torch.nn.Linear(num_features, NUM_TOTALE_PUNTI * 2)

    # Riepilogo dei pesi allenabili
    summary(net_from_scratch, input_size=(3, IMG_SIZE, IMG_SIZE))

    name_train = "resnet_18_all_points_from_scratch"  # nome da usare negli esperimenti per il writer di tensorboard; utile per logging

    # Eseguo il loop di training e l'inferenza
    execute(name_train, net_from_scratch, LR, EPOCHS, train_loader, test_loader)


def allena_pesi_solo_ultimo_layer(train_loader, test_loader):
    """
    2.
    Freezo tutti i pesi, tranne quelli dell'ultimo layer.

    Questa sarà una baseline da cui dovrò migliorare sempre più.
    """

    ## Importo il modello con i pesi
    rete = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    # Sostituisco le 512 features con i miei 28 valori di coordinate
    num_features: int = rete.fc.in_features  # mi aspetto sia 512
    print(f"\nL'ultimo layer di resnet18 ha: {num_features} features.\n")
    rete.fc = torch.nn.Linear(num_features, NUM_TOTALE_PUNTI * 2)

    ## Congelamento dei pesi pre-addestrati del backbone
    for param in rete.parameters():  # Per questi parametri qui
        param.requires_grad = False  # non deve essere calcolato nessun gradiente

    ## Tuttavia, non devo congelare i pesi dell' ultimo layer
    for param in rete.fc.parameters():  # Soltanto questi devo allenare
        param.requires_grad = True  # quindi solo questi parametri saranno aggiornati

    # Riepilogo dei pesi allenabili
    summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))

    name_train = "resnet_18_all_points_solo_ultimo_layer"  # nome da usare negli esperimenti per il writer di tensorboard; utile per logging

    # Eseguo il loop di training e l'inferenza
    execute(name_train, rete, LR, EPOCHS, train_loader, test_loader)



def allena_pesi_solo_layers_4_and_ultimofc(train_loader, test_loader):
    """
    3. ...

    """

    ## Importo il modello con i pesi
    rete = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    # Sostituisco le 512 features con i miei 28 valori di coordinate
    num_features: int = rete.fc.in_features  # mi aspetto sia 512
    print(f"\nL'ultimo layer di resnet18 ha: {num_features} features.\n")
    rete.fc = torch.nn.Linear(num_features, NUM_TOTALE_PUNTI * 2)

    ## Congelamento dei pesi pre-addestrati del backbone
    for param in rete.parameters():  # Per questi parametri qui
        param.requires_grad = False  # non deve essere calcolato nessun gradiente

    ## Non devo congelare i pesi ne' dell' ultimo layer
    #  e questa volta neanche del layer 4
    for name, param in rete.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    # Riepilogo dei pesi allenabili
    summary(rete, input_size=(3, IMG_SIZE, IMG_SIZE))

    name_train = "resnet_18_all_points_solo_4_and_ultimo"  # nome da usare negli esperimenti per il writer di tensorboard; utile per logging

    # Eseguo il loop di training e l'inferenza
    execute(name_train, rete, LR, EPOCHS, train_loader, test_loader)

def main():

    ###################################
    #### (1) CARICAMENTO DATASET ######
    ###################################

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

    ##########################################
    ### (2) ALLENAMENTO RETE DA 0 ###
    ##########################################
    # allena_rete_da_zero(train_loader, test_loader) # TODO: devo migliorare questo. Ci mette troppo tempo

    ###########################################
    ### (3) TRANSFER LEARNING ###
    ###########################################

    # allena_pesi_solo_ultimo_layer(train_loader, test_loader) # TODO: fa schifo


    # allena_pesi_solo_layers_4_and_ultimofc(train_loader, test_loader) # TODO: da migliorare

    ## Per fare TL uso il modello che ho definito io, come istanza della classe che estende nn.Module
    #  NUM_TOTALE_PUNTI * 2 perchè ho 2 coordinate per punto
    model : ResNet18_Keypoints = ResNet18_Keypoints(num_outputs=NUM_TOTALE_PUNTI*2, pretrained=True)

    model = model.to(DEVICE)

    summary(model.backbone,input_size=(3, IMG_SIZE, IMG_SIZE))

    ###########################################
    ### (3) Loss ed ottimizzatore ###
    ###########################################


    name_train = "resnet_18_all_points" # nome da usare negli esperimenti per il writer di tensorboard; utile per logging

    # Eseguo il loop di training e l'inferenza
    execute(name_train, model, LR, EPOCHS, train_loader, test_loader)


    #######################################
    #### (4)SALVATAGGIO DEL MODELLO ####
    #######################################
    # TODO: d'ora in avanti il salvataggio del modello è
    #  fatto in automatico da dentro il training_loop

    # nome_modello: str = "resnet18_all_points.pth"
    #
    # torch.save( model.state_dict(), os.path.join(CHECKPOINTS_DIR, nome_modello) )
    #
    # print(f"{ok}Modello salvato come {nome_modello}")


if __name__ == "__main__":
    main()