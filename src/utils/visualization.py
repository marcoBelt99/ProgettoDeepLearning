'''
In questo file creo due funzioni che saranno utili per la visualizzazione di immagini e le metriche di addestramento
durante la creazione del mio modello.
'''

import matplotlib.pyplot as plt
import numpy as np

## Ridefinisco la funzione im_show(), in cui denormalizzo le immagini. 
def im_show(img):
    '''
    Denormalizzare è necessario al fine di visualizzare correttamente con matplotlib. 
    Poi converto da tensore pytorch in array numpy: passaggio necessario perchè matplotlib non può lavorare direttamente coi tensori.
    Poi c'è la trasposizione dell'array: le immagini in pytorch sono in formato (canale, altezza, larghezza), ma matplotlib si aspetta
    immagini nel formato (altezza, larghezza, canali) ecco perchè devo fare il transpose().
    Infine, visualizzo l'immagine con plt.imshow(...), mentre plt.show() rende visibile la figura.
    '''
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow( np.transpose( npimg, (1, 2, 0) ) )
    plt.show()

def plot_metrics(metric1, metric2=None, metric_name="Metric"):
    '''
    Con questa funzione vado a monitorare visivamente l'andamento dell'addestramento, e la logica
    presente in questa funzione mi permette di fare i confronti diretti tra le performance di
    addestramento e di validazione.
    '''
    ## Inizializzazione del grafico:
    #  creo una nuova figura con dimensioni specificate in pollici
    plt.figure( figsize=(10, 5) )
    #  disegno un grafico con plt.plot() della metrica passata come primo argomento. Questa metrica
    #  può rappresentare una qualsiasi misurazione durante l'addestramento, come: la loss, o l'accuracy.
    plt.plot( metric1, label=f'Training {metric_name}' )
    #  opzionale c'è metric_2,, quindi viene aggiunto al grafico un 2° tracciato: questo è utile per confrontare
    #  le varie performance di addestramento e validazione.
    if metric2 is not None:
        plt.plot( metric2, label=f'Validation {metric_name}' )
    #  configuro gli assi e i titoli. Le etichette e i titoli degli assi sono impostati per chiarire cosa effettivamente
    #  il grafico rappresenta.
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Training {metric_name}')
    #  aggiungo la legenda al grafico
    plt.legend()
    #  aggiungo una griglia per facilitare la lettura dei valori
    plt.grid( True )
    #  rendo visibile il grafico
    plt.show()

