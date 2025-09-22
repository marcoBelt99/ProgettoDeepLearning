'''
Qui creo una funzione per l'addestramento e la validazione del mio modello. Questa funzione gestirà tutto il processo di 
addestramento, inclusa la gestione del dispositivo (CPU o GPU); l'ottimizzazione; il calcolo della perdita; e fornirò anche
le valutazioni di performance al termine di ogni epoch.
'''

import torch
import torch.nn as nn
import torch.optim as optim

def train_and_validate(model, train_loader, validation_loader, epochs=10):
    '''
    Parametri:
    - modello: modello di RN
    - train_loader: DataLoader per il training
    - validation_loader: DataLoader per la validazione
    - epochs: numero di epoche per cui eseguire l'addestramento
    '''
    ## Imposto il dispositivo: se la GPU è disponibile la uso per l'addestramento (che so accelerare il training rispetto 
    #  all'uso della classica CPU).
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    ## Inizializzo le variabili che mi autano a tracciare le prestazioni del mio modello
    #  uso una lista per memorizzare la perdita durante l'addestramento
    train_losses = []
    #  uso una lista per memorizzare la loss per la validazione
    validation_losses = []
    #  uso una lista per memorizzare l'accuratezza di validazione al termine di ogni epoch
    validation_accuracy = []

    ## Configuro la perdita e l'ottimizzatore
    criterion = nn.CrossEntropyLoss() # questa la uso per problemi di classificazione multiclasse
    #  uso come ottimizzatore SGD con lr 0.001 e un momentum di 0.9 per aggiornare i pesi del modello
    optimizer = optim.SGD( model.parameters(), lr=0.001, momentum=0.9 )

    ### Ciclo di addestramento e validazione
    for epoch in range(epochs):
        # entro in modalità addestramento:
        model.train()
        running_loss = 0.0

        ## Ciclo di addestramento visto più volte (zero_grad; output; loss; backward step)
        #  Quindi: addestro il modello per un numero specificato di epoche; durante l'addestramento la perdita viene calcolata
        #  e ottimizzata; poi, la media delle perdite per ciascun epoch viene salvata.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to( device ), labels.to( device )
            optimizer.zero_grad()
            outputs = model( inputs )
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append( running_loss / len(train_loader) )

        ## Passo alla validazione:
        #  si entra in modalità valutazione, con model.eval()
        model.eval()
        validation_loss = 0.0
        total = 0
        correct = 0
        # Si valuta il modello usando un set di validazione, senza aggiornare i pesi del modello (torch.no_grad())
        with torch.no_grad():
            # calcolo e salvo la perdita di validazione e l'accuratezza:
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to( device ), labels.to( device )
                outputs = model( inputs )
                loss = criterion( outputs, labels )
                validation_loss += loss.item()
                _, predicted = torch.max( outputs.data, 1 )
                total += labels.size( 0 )
                correct += ( predicted == labels ).sum().item()
        validation_losses.append( validation_loss / len( validation_loader ) )
        validation_accuracy.append( 100* correct / total )

        ## Poi si passa alla stampa della perdita di addestramento e di validazione e l'accuratezza di validazione al termine di ogni epoch
        print( f'Epoch {epoch+1}, Loss: {train_losses[-1]}' )
        print( f'Validation Accuracy: {validation_accuracy[-1]}%' )
        print( f'Validation Loss: {validation_losses[-1]}' )

    ## Al termine dell'addestramento e della validazione, la mia funzione va a stampare un messaggio di completamento,
    #  e va a restituire le metriche di addestramento e validazione raccolte durante il processo.
    #  In questo modo riesco a monitorare le prestazioni del mio modello durante i vari cicli di addestramento, e quindi
    #  effettuare gli aggiuntamenti
    print( f'Training and validation complete' ) 
    return train_losses, validation_losses

## Il passo successivo sarà ora quello di testare il mio modello sul set di test per valutarne le prestazioni complessive
    