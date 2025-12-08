import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet



class ResNet18_Keypoints(nn.Module):
    '''
    Questa classe serve da proxy come contenitore di pesi tra la parte di training
    e nella parte di inferenza (validazione), e nella parte di visualizzazione.

    ## Per fare TL uso il modello che ho definito io, come istanza della classe che estende nn.Module
    #  NUM_TOTALE_PUNTI * 2 perchè ho 2 coordinate per punto
    '''
    def __init__(self, num_outputs=28, pretrained=True):
        '''
        Args:
            num_outputs (int): Indica il numero di valori di coordinate x,y (n_punti × 2) la cui regressione dev'essere predetta.
            Di default vale 28, quindi si vogliono predire tutti i punti assieme.
        '''

        super().__init__()

        '''
        Recupero il modello pre-addestrato e lo adatto alla mia situazione.
        '''
        self.backbone : ResNet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        ## Congelamento dei pesi pre-addestrati del backbone
        for param in self.backbone.parameters():  # Per questi parametri qui
            param.requires_grad = False  # non deve essere calcolato nessun gradiente

        ## Tuttavia, non devo congelare i pesi dell' ultimo layer
        for param in self.backbone.fc.parameters():  # Soltanto questi devo allenare
            param.requires_grad = True  # quindi solo questi parametri saranno aggiornati


        ## Sostituisco l'ultimo layer per la regressione.
        ## (taglio il classificatore e adatto l'architettura per risolvere il mio task)

        num_features: int = self.backbone.fc.in_features  # mi aspetto sia 512
        print(f"\nL'ultimo layer di resnet18 ha: {num_features} features.\n")

        ## VECCHIA
        # self.backbone.fc = nn.Linear(num_features, num_outputs)

        ## NUOVA
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_outputs)
        )



    # TODO: pensare di sostituirlo con il metodo __call__()
    def forward(self, x):
        return self.backbone(x)
