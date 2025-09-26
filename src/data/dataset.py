import torch
from torchvision import datasets, transforms, utils 
import matplotlib.pyplot as plt # per poter visualizzare le mie immagini


## Trasformazioni che uso per normalizzare il DS
transform = transforms.Compose(
    [transforms.ToTensor(), # converto il mio dato in un tensore
    transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) # Normalizzo un tensore di immagini fornendo la media e la dev. standard per ciascun canale
])


## Caricamento del DS CIFAR-10. Con transform=transform applico le trasformazioni sulle immagini. download=False perchè ho il DS in locale e non deve essere
#  scaricato da internet.
train_dataset = datasets.CIFAR10( root='./ProgettoFinale/dataset', train=True, download=False, transform=transform )
test_dataset = datasets.CIFAR10( root='./ProgettoFinale/dataset', train=False, download=False, transform=transform )

## Creazione dei Dataloader (il DL facilita l'iterazione sul Dataset). batch_size è il numero di campioni per ogni batch da caricare
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=64, shuffle=True ) # shuffle=True mescola i dati ad ogni epoch (utile per l'addestramento)
test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=64, shuffle=False )


## Creo una funzione per visualizzare le immagini
def im_show(img):
    '''
    Denormalizza l'immagine riportandola al suo formato originale, e la visualizza.
    '''
    img = img / 2 + 0.5
    plt.imshow( torch.transpose( torch.transpose(img, 0, 1), 1, 2 ) ) # con torch.transpose cambio l'immagine
    # del tensore in modo che possa essere correttamente interpretate da plt.imshow
    plt.show()

## Visualizzo alcune immagini per vedere se è tutto ok:
#  - carico un batch di immagini e creo una griglia da questo batch di immagini per visualizzarle insieme.
#  - se la visualizzazione va a buon fine, il mio DS è stato caricato correttamente.
dataiter = iter( train_loader )
images, labels = next( dataiter )
im_show( utils.make_grid(images[:4]) ) # guardo solo una riga composta da 4 immagini

'''
Ok. Fin qui: ho caricato, normalizzato e visualizzato il DS di immagini utilizzando PyTorch e torchvision.
'''