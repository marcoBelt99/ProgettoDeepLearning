
## Commenti personali

Devo fare 4 Reti Neurali, probabilmente CNN, ognuna che mi predica il suo gruppo di punti di appartenenza
- S, N, A, B
- Sna, Snp, Gn, Go
- U1, L1, root, tip
- Pg, mesial

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
    - `data_preparation.py`: funzione `prepare_data_loaders()` progettata per facilitare la gestione del flusso di dati durante l'addestramento del mio modello di ML. Inoltre, suddivido il set di dati in: addestramento, validazione e test.
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
    - `test_model.py`: contiene la funzione `test_model()` che valuta l'accuratezza di classificazione del modello su un set di dati di test, calcolando l'accuratezza per ciascuna classe.



---

# Linee guida per lo svolgimento

Il dataset è composto solo da $img_{i}.jpg$ e relativo $file-txt_{i}.txt$ (che annota l'immagine).

In ogni singolo file di testo mi dovrebbero interessare solo le colonne $X$ ed $Y$.

## 1. Preprocessing

1. Carico immagini con PIL
2. Normalizzare:
    - porto tutte le immagini alla stessa dimensione
    - scalo i pixel su $[0, 1]$.
3. Normalizzare le coordinate X, Y: vanno scalate in base alla nuova dimensione dell'immagine; Es) se l'immagine originale è $2000$x$2000$ e la ridimensiono a $256$ x $256$, allora devo moltiplicare le coordinate per $256/2000$.

### Dettagli
- Come visto, i file txt hanno delle colonne inutili, e a me servono solo X ed Y.
- Prima di allenare la rete, devo:
    - leggere tutti i txt con pandas
    - estrarre solo le colonne di interesse
    - fare statistiche: come media, varianza, deviazione standard, distribuzione per punto, etc.
    - fare qualche grafico con matplotlib / seaborn, ad esempio per la distribuzione delle X e delle Y per i diversi punti.

Questo processo dovrebbe servirmi per capire se ci sono outlier o errori anomali sui dati.
- Inoltre, devo fare anche Resizing + Rescaling: fondamentale perchè le immagini hanno dimensioni diverse e la rete vuole input fissi
- calcolare le coordinate normalizzate rispetto alle nuove dimensioni

## 2. Dataset e DataLoader in PyTorch

1. Fare classe `torch.util.data.Dataset`. Tale serve per crearmi un mio dataset custom. Questa classe, dato l'indice ritorna l'immagine preprocessata (un tensore) e la coordinata di keypoints (un tensore). In particolare, la classe:
    - legge un'immagine
    - trova il corrispondente file.txt
    - estrae solo i punti del gruppo richiesto
    ritorna la coppia: `(immagine_tensor, coords_tensor)`

## 3. Definizione del Modello
- Uso una CNN standard (ResNet18 pre-addestrata o una rete custom più leggera).
- Modifico l'ultimo layer Linear per avere 2 * n_punti neuroni in output. Es) per il gruppo (0, 1, 4, 5) ho 4 punti, quindi 8 valori di output.
- Come Loss Function:
    - visto che si tratta di un problema di regressione di coordinate, uso una MSELoss
    - (un' alternativa può essere la MAE)

## 4. Training
- Divido il DS in set di: training; validazione; test.
- Parto con un learning rate basso, come 0.001
- Valuto con la Mean Euclidean Distance.

In particolare, nel loop di training avrò:
- forward pass: in cui faccio le predizioni delle coordinate
- calcolo della loss: con MSE(pred, target)
- backpropagation: loss.backward()
- ottimizzazione: usando Adam o SGD

Ovvio che l'output layer, essendo una regressione non ha la softmax, ma ha numeri reali (coordinate)

## 5. Alcune estensioni
- Data augmentation: rotazioni leggere; zoom; flip orizzontali
- Un modello unico che predice tutti i punti
- Heatmap ??


# Fonti

Gli articoli consigliati parlano di:
1. Keras example keypoint detection: 
    - per predire keypoints sul Dataset MPII (di pose umane)
    - usa una CNN che predice direttamente le coordinate (regressione).
    - Mostra bene come costruire dataset, loss e loop di training => è praticamente lo stesso task che devo
    fare io, solo che anzichè Keras, io uso PyTorch.
2. Towards DataScience: Facial KeyPoints Detection: 
    - è un tutorial con Dataset di volti e annotazioni di punti facciali (occhi, naso, bocca)
    - spiega bene la parte di preprocessing, normalizzazione e data augmentation;
    - M0stra anche come dividere i punti in sottoinsiemi => è praticamente lo stesso problema, solo che i miei sono
    i Punti di Repere
3. TensorFlow Lite: Pose Estimation:
    - mostra che i keypoints possono essere predetti usando heatmap probabilistiche (anzichè come coordinate dirette)
    - => è però più complicato
4. Analytics Vidhya: Human Pose estimation:
    - da una panoramica su tecniche di pose estimation;
    - spiega concetti chiave: keypoints; heatmap; modelli noti tipo OpenPose
    - utile a livello teorico e spiega il perchè di certe scelte.