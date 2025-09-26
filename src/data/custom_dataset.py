import os # per interagire con il filesystem
import pandas as pd # per leggere i file csv che contengono le etichette delle immagini
from torchvision.io import read_image # read_image per leggere le immagini
from torchvision.io import decode_image
from torch.utils.data import Dataset # Dataset è la classe base per creare dataset personalizzati

class CustomImageDataset(Dataset):
    '''
    Il costruttore come parametri ha:
    :param str annotation_file: percorso del file csv che contiene le annotazioni delle immagini.
    Ogni riga nel file csv contiene il nome del file dell'immagine e la sua etichetta.
    :param: str image_dir: directory in cui sono salvate le immagini
    :param Function transform: opzionale. Funzione di trasformazione da applicare a ciascuna immagine, ad esempio
    normalizzazione, ridimensionamento, etc.
    :param Function target_transform: opzionale. Funzione di trasformazione da applicare alle etichette. Ad esempio,
    la conversione da etichette testuali ad interi.
    '''

    '''
    ######
    Questa classe:
    - Deve essere in grado di gestire i diversi gruppi di punti. 
    - Durante l'inizializzazione del dataset, dovrai specificare quale gruppo di punti stai addestrando. 
    
    '''
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # pd.csv carica le annotazioni delle immagini in un dataframe di pandas. Questo DF è accessibile
        # a tutta la classe.
        self.img_labels = pd.csv( annotations_file )
        self.img_dir = img_dir # memorizza il percorso della directory dell'immagine.
        self.transform = transform # memorizzo la funzione di trasformazione
        self.target_transform = target_transform # memorizzo la funzione di trasformazione
    
    def __len__(self):
        '''
        Questo metodo è richiesto da PyTorch, e deve restituire il numero di elementi nel dataset,
        ossia il numero di righe nel dataframe delle etichette.
        '''
        return len( self.img_labels )
    
    def __getitem__(self, idx):
        '''
        Questo metodo è chiamato da PyTorch per ottenere un singolo elemento dal dataset.
        :param int idx: è l'indice dell'elemento da restituire
        '''
        '''
        ######
        La logica è di:
        - Leggere le coordinate di tutti i 14 punti dal file .txt corrispondente all'immagine.
        - Filtrare e restituire solo le coordinate dei keypoint che fanno parte del gruppo specificato
          (ad esempio, se il gruppo è [0, 1, 4, 5], dovrai estrarre solo le coordinate di questi quattro punti).
        '''
        # Costruisco il percorso completo dell'immagine, combinando la directory di base 'img_dir'
        # e il nome del file dell'immagine, ottenuto dal dataframe
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # Carico l'immagine da img_path usando il metodo read_image, che restituisce un tensore di immagine. 
        #image = read_image(img_path) # dovrebbe essere obsoleto 
        image = decode_image(img_path) #questo dovrebbe essere quello consigliato

        # Estraggo la corrispondente etichetta dall'indice speficato.
        label = self.img_labels.iloc[idx, 1]

        # Se sono state fornite, con il seguente if dovrei applicare le trasformazioni all'immagine e alla label
        # Questo è comune per la normalizzazione dell'immagine e la conversione delle etichette in un formato 
        # adatto per l'addestramento. 
        if self.transform:
            image = self.transform(image)
        if self.target_transform(label):
            label = self.target_transform(label)
        
        # Poi, restituisce l'immagine trasformata e la label
        return image, label
