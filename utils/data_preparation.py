import torch
from torchvision import datasets, transforms # per caricare e pre-elaborare i dataset
from torch.utils.data import DataLoader, random_split # per facilitare il batching, il campionamento e la suddivisione casuale dei dataset

def prepare_data_loaders(batch_size):
    '''
    - batch_size: parametro che definisce il numero di esempi per ogni batch durante il training e il testing.
    '''
    transform = transforms.Compose([
        transforms.ToTensor(), # per convertire le immagini in tensori di PyTorch
        transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) # normalizzo il tensore a media e deviazione standard specifiche per ogni canale (qui uso media 0.5 e dev standard 0.5)
    ])

    ## Carico il DS CIFAR-10
    train_dataset = datasets.CIFAR10( root='./ProgettoFinale/dataset', train=True, download=False, transform=transform )
    test_dataset = datasets.CIFAR10( root='./ProgettoFinale/dataset', train=False, download=False, transform=transform )    

    ## Suddivido il Dataset
    train_size = int( 0.8 * len( train_dataset ) ) # 80% delle immagini per l'addestramento
    validation_size = len( train_dataset ) - train_size #20% per la validazione
    # suddivido il train dataset in due subset non sovrapposti, secondo le dimensioni specificate
    train_set, validation_set = random_split( train_dataset, [train_size, validation_size] )

    ## Creo i DataLoader. shuffle=True è necessario per mescolare i dati ad ogni epoch: molto importante per l'addestramento per prevenire l'overfitting
    train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=True )
    validation_loader = DataLoader( validation_set, batch_size=batch_size, shuffle=True )
    test_loader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False ) # invece per il test non è necessario mescolare i dati

    '''
    Quindi, questa funzione fornisce un modo utile e strutturato per preparare i dati per il training, la validazione e il test,
    gestendo adeguatamente le trasformazioni e la suddivisione dei dati usando PyTorch.
    '''
    return train_loader, validation_loader, test_loader
