- La ReLU è preferita in molti contesti nel Deep Learning, perchè introduce non linearità senza influenzare negativamente la propagazione
  del gradiente (come invece possono fare altre funzioni di attivazione). Questo aiuta a prevenire il problema del Gradiente Evanescente,
  migliorando la capacità di apprendimento della mia rete su Dataset anche complessi.
- Dimensione del Kernel e il pooling: la scelta di un kernel di dimensione 5x5 negli strati convoluzionali è proprio necessaria per 
  catturare sufficienti dettagli dell'immagine, mentenendo una buona copertura dell'area di input.
  Il pooling, dall'altra parte, serve a ridurre le dimensioni spaziali delle rappresentazioni apprese, diminuendo quindi il numero di
  parametri e di computazioni necessarie negli strati successivi, e quindi contribuendo a controllare l'overfitting.
- Architettura: questa architettura mira un po' ad un equilibrio tra profondità della rete (numero di layers) e complessità
  computazionale, rendendola adatta all'addestramento su hardware relativamente anche accessibili (come potrebbe essere il mio 
  computer), senza sacrificare troppo la precisione.
  La progressione degli strati convoluzionali e poi a quelli fully connected segue un approccio che è canonico, come visto nel
  design delle CNN: le features (caratteristche estratte) vengono via via condensate fino alla classificazione finale.