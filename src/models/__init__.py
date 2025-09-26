"""
Modulo `src/models`

Definizione del modello, inclusa la classe ResNet modificata.

L'idea di usare una ResNet pre-addestrata rimane valida. Dovrai implementare un ciclo di addestramento che
esegua i seguenti passi per ciascuno dei quattro gruppi:

    - Definizione del modello: Per ogni gruppo, dovrai creare un'istanza del tuo modello basato su ResNet.
      L'unica cosa che cambia è lo strato finale.

        - Gruppo 1 (0,1,4,5): 4 punti, quindi 4 x 2 = 8 neuroni in output.
        - Gruppo 2 (2,3,7,8): 4 punti, quindi 4 x 2 = 8 neuroni in output.
        - Gruppo 3 (9,10,11,12): 4 punti, quindi 4 x 2 = 8 neuroni in output.
        - Gruppo 4 (6,13): 2 punti, quindi 2 x 2 = 4 neuroni in output.
    
    - Creazione del DataLoader: Dovrai creare un DataLoader specifico per ogni gruppo, utilizzando l'istanza del
      CustomDataset che restituisce solo le coordinate corrette.

    - Addestramento: Addestra ogni modello separatamente, salvando i pesi (.pt o .pth) con un nome significativo 
    (es. model_group1.pth, model_group2.pth, ecc.).

    
La cartella checkpoints/ è la libreria per i quattro modelli addestrati. Ad esempio, 
poisso salvare i pesi come gruppo_1.pth, gruppo_2.pth, etc. per poterli caricare in seguito per il testing.
"""

