# Introduzione
Progetto d'esame per il corso di Deep Learning.

## Testo

Bisogna andare a localizzare i punti di Repere.

Come possibile aiuto, guarderei qualche articolo su keypoint detection e pose estimation. Il nostro obiettivo è leggermente diverso ma la base è la stessa, <u>dobbiamo localizzare dei punti specifici all'interno della immagine</u>.

- https://keras.io/examples/vision/keypoint_detection/
- https://towardsdatascience.com/facial-keypoints-detection-image-and-keypoints-augmentation-6c2ea824a59
- https://www.tensorflow.org/lite/examples/pose_estimation/overview
- https://www.analyticsvidhya.com/blog/2021/10/human-pose-estimation-using-machine-learning-in-python/

L'idea è quella di implementare diverse reti, una per gruppo di punti, per predire le coordinate dei seguenti raggruppamenti (il numero indica l'indice nel file di testo)
- 0,1,4,5 (S,N,A,B)
- 2,3,7,8 (Sna, Snp, Gn, Go)
- 9,10,11,12 (U1 e L1 root e tip)
- 6,13 (pg, mesial)

Nel file di testo, ogni riga contiene l'indice del punto, alcune informazioni inutili, e le coordinate del punto (le ultime due colonne)

Eventualmente, se si ha tempo, confrontando i risultati con una rete che prova a predirli tutti insieme.


## Dataset

Il Dataset usato è così composto:

![alt text](/altro/dataset-composizione.png)

Ogni file di testo è strutturato nel seguente modo:

![alt text](/altro/file-txt-composizione.png)

## Fasi del progetto:

1. Esplorazione e preparazione dei Dati: 
    - caricamento del Dataset
    - visualizzazione di alcune immagini
    - normalizzazione delle immagini
    - preparazione dei DataLoader per l'addestramento e la validazione
2. Progettazione della Rete Neurale (CNN), che includerà 
    - diversi strati: convoluzionali, pooling, fc-
    - funzioni di attivazione
    - dimensioni del kernel
    - etc.
3. Addestramento e Validazione del Modello:
    - implementazione del ciclo di addestramento (inclusa la loss function e l'ottimizzatore)
    - utilizzo una parte del Dataset per la Validazione del modello durante l'addestramento per monitorare l'overfitting
4. Test e Valutazione del Modello: valuterò le prestazioni del modello su un set di test per determinare la sua accuratezza. Infine, analizzerò i risultati per identificare eventualmente quelle categorie di immagini con performance migliori e peggiori, e discussione su alcuni possibili miglioramenti.
5. Documentazione e Presentazione: vedo come preparare una breve relazione, con i punti fondamentali, che descriva il processo di sviluppo del modello e le scelte effettuate.

## Struttura del progetto
- `dataset/`
    - `dataset.py`: per caricare, normalizzare e visualizzare il DS di immagini utilizzando PyTorch e torchvision.
- `documentation/`
- `models/`
- `testing/`
- `training/`
- `utils/`
    - data_preparation.py: funzione prepare_data_loaders() progettata per facilitare la gestione del flusso di dati durante l'addestramento del mio modello di ML. Inoltre, suddivido il set di dati in: addestramento, validazione e test.
- `main.py`
- `test_torch.py`


### Passaggi:
1. Ho agito sulla cartella `/dataset` e importazione dataset scaricato da internet e scompattato su cartella del progetto
2. Ho agito sulla cartella `/utils`, prima col file `data_preparation.py` e poi `visualization.py`. 
3. Ho agito sulla cartella `/models` . 
    - `cnn_model.py`: qui ho lavorato sulla Progettazione della rete neurale. Questa è una fase cruciale, perchè la struttura della CNN determina come i dati vengono elaborati, e infine l'efficacia del modello nel classificare le immagini.
    - `SCELTE_PROGETTUALI.md`: riassume le scelte fatte e spiega anche il funzionamento di convoluzione e pooling e ReLU.
4. Ho agito sulla cartella `/training/`
    - `train_validate.py`: qui procedo con l'addestramento e la valutazione del modello. In particolare, qui imparo come implementare
   un ciclo di addestramento che sia efficace; scelgo la funzione di perdita giusta; seleziono un ottimizzatore, e quindi poi uso il set di dati per la validazione per monitorare, e quindi per prevenire l'overfitting.
5. Ho agito sulla cartella `/testing`: Qui sono nella fase di test e valutazione del modello. Tale fase è importante perchè mi fornisce un'indicazione chiara di come il modello si comporterà con dati completamente nuovi, che non ha mai visto durante l'addestramento o la validazione.
    - test_model.py: contiene la funzione test_model() che valuta l'accuratezza di classificazione del modello su un set di dati di test, calcolando l'accuratezza per ciascuna classe.

## Possibili miglioramenti / controlli:
- Se alcune categorie sono più difficili da classificare, il mio modello ha delle difficoltà con alcune categorie, e può essere
interessante aumentare il Dataset con immagini trasformate (quindi, fare un Data Augmentation, con immagini ruotate, traslate, etc. ),
in modo che venga aumentata la varietà di dati da cui il modello può apprendere.
- Posso modificare l'architettura della rete, aggiungendo o modificando layers, che può essere interessante per migliorare l'apprendimento, magari di caratteristiche complesse
- Posso aggiustare il learning rate, il numero di epoche, applicare tecniche come il dropout
