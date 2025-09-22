import torch

def test_model(net, test_loader, classes):
    '''
    - net: modello di RN che sarà testato.
    - test_loader: oggetto DataLoader di PyTorch che fornisce accesso ai dati di test.
    - classes: un elenco delle classi del Dataset, usato per stampare i risultati dell'accuratezza per ciascuna classe
    '''
    # scelgo se usare una GPU o la CPU per il test del modello
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Imposto il modello per la valutazione
    net = net.to( device )
    net.eval() #disattiva le caratteristiche del training come il Dropout e la Normalizzazione Batch, rendendolo adatto per la valutazione

    ## Inizializzo le variabili per tracciare le prestazioni
    class_correct = list(0. for i in range(10)) # tiene traccia del numero di predizioni corrette per ciascuna classe
    class_total = list(0. for i in range(10)) # conto il numero totale di esempi per ciascuna classe

    ## Blocco di codice di valutazione
    # disattivo il calcolo del gradiente (riduco il consumo di memoria e velocizzo il processo, poichè i gradienti per il test non sono necessari)
    with torch.no_grad():
        # Per ogni batch di immagini e di etichette
        for images, labels in test_loader:
            # Le immagini e le etichette vengono spostate sul dispositivo appropriato
            images, labels = images.to(device)
            # calcolo l'output del modello
            outputs = net(images)
            # funzione toch.max che estrai le classi predette
            _, predicted = torch.max(outputs, 1)
            # calcolo la corrispondenza tra le classi predette e le vere etichette (variabile c)
            c = (predicted == labels)
            # aggiorno il conteggio delle predizioni corrette e totali per ogni classe
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    ## Stampo i risultati
    for i in range(10):
        # Stampo l'accuratezza per ciascuna classe, calcolata come rapporto tra il numero di predizioni corrette e il totale di
        # ogni classe, convertito in  %. Se non ci sono dati per una classe, allora l'accuratezza per quella classe è riportata come 0. 
        print(f'Accuracy of {classes[i]}:  {100* class_correct[i] / class_total[i] if class_total[i] else 0:.2f}%')
    
    '''
    Quindi, questa funzione è utile per valutare in maniera dettagliata le prestazioni del mio modello di classificazione, 
    consentendo di identificare come il modello si comporta su ciascuna classe specifica del dataset.
    E' molto utile per comprendere la capacità del modello e le eventuali limitazioni.
    '''