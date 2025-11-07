'''
La mia rete sarà composta da 2 strati convoluzionali, seguiti da 3 strati fully connected.
'''
import torch.nn as nn # modulo di PyTorch che contiene tutte le classi necessarie per costruire i layers di reti neurali, come
                      # gli strati convoluzionali, di pooling, fully connected, etc. 
import torch.nn.functional as F # modulo che contiene le funzioni utili per eseguire operazioni come: le funzioni di attivazione e 
                                # le operazioni di convoluzione

## Definisco la classe Net, che sarà il nome del mio modello
class Net(nn.Module):
    def __init__(self):
        '''
        Definizione della mia Rete Neurale
        '''
        super(Net, self).__init__()
        ## Definisco i layer convoluzionali e di pooling
        #  primo strato convoluzionale, che prende immagini RGB (3 canali) come input, e produce 6 mappe di features (filtri / kernel) 
        # di dimensione 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        #  inserisco il layer di pooling, che riduce la dimensione spaziale dell'input di un fattore di 2, usando un kernel 2x2 e stride 2
        self.pool = nn.MaxPool2d(2, 2)
        #  inserisco il secondo strato convoluzionale, che prende i 6 canali in output dal primo strato, e produce 16 filtri usando anch'esso un kernel 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        '''
        Note:
        - i filtri hanno dimensione 5x5 px: questa è una scelta abbastanza comune che consente di catturare le features dell'immagine
        - pooling: dopo ciascun strato convoluzionale applico il MaxPooling con un kernel 2x2 per ridurre la dimensione delle caratteristiche che sono mappate,
                   rendendo così la rete più efficiente, e riducendo il rischio di overfitting.
        '''

        ## Ora Definisco i layer fully connected
        #  fc1 è il 1° strato completamente connesso, che trasforma i dati da una dimensione di 16 mappe di caratteristiche ciascuna di
        #  dimensione 5x5, dopo aver applicato appunto le convoluzioni e il pooling, e trasforma in 120 nodi
        self.fc1 = nn.Linear( 16 * 5 * 5, 120)
        #  fc2 è il 2° strato completamente connesso, che collega 120 nodi a 84 nodi
        self.fc2 = nn.Linear( 120, 84 )
        #  fc3 è l'ultimo strato completamente connesso, che collega 84 nodi a 10 nodi, che sono corrispondenti alle 10 classi del Dataset
        #  CIFAR-10
        self.fc3 = nn.Linear( 84, 10 )

        '''
        Note:
        - i layer fully-connected, progressivamente riducono il numero di nodi da 16*5*5 a 10.
          10 rappresentano le classi di output di CIFAR-10.
        - ReLU: uso la f.d.att. ReLU per gli strati convoluzionali e per i primi 2 strati FC. La ReLU è popolare per la sua semplicità computazionale,
                e per la capacità di ridurre il problema del gradient vanishing, permettendo che la mia rete apprenda più velocemente
        '''

    
    def forward(self, x):
        '''
        Definisco come l'input x attraversa la rete.
        '''
        # L'input passa attraverso il 1° layer convoluzionale, seguito da una funzione di attivazione ReLU, e poi attraverso il pooling
        x = self.pool( F.relu( self.conv1( x ) ) )
        # Lo stesso processo si ripete per il 2° layer convoluzionale
        x = self.pool( F.relu( self.conv2( x ) ) )
        # L'output convoluzionale viene poi appiattito (flattened) in un vettore prima di passare attraverso i 3 strati completamente connessi
        x = x.view( -1, 16 * 5 * 5 )
        # Tra i primi due strati FC uso la ReLU come funz. di attiv.
        x = F.relu( self.fc1( x ) )
        x = F.relu( self.fc2( x ) )
        # L'output del metodo forward è il risultato del 3° strato completamente connesso
        x = self.fc3( x )
        return x

## Creo un'istanza della rete Net() pronta per essere addestrata o testata

         
