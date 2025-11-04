# Introduzione
Progetto d'esame per il corso di Deep Learning.

## Testo

Bisogna andare a localizzare i punti di Repere.

Come possibile aiuto, guarderei qualche articolo su keypoint detection e pose estimation. Il nostro obiettivo è leggermente diverso ma la base è la stessa, <u>dobbiamo localizzare dei punti specifici all'interno della immagine</u>.

- https://keras.io/examples/vision/keypoint_detection/
- https://towardsdatascience.com/facial-keypoints-detection-image-and-keypoints-augmentation-6c2ea824a59
- https://www.tensorflow.org/lite/examples/pose_estimation/overview
- https://www.analyticsvidhya.com/blog/2021/10/human-pose-estimation-using-machine-learning-in-python/

In allegato anche [un file](https://github.com/marcoBelt99/ProgettoDeepLearning/blob/main/PUNTI%20DI%20REPERE%20CEFALOMETRIA.docx) che spiega i punti di Repere che dobbiamo andare a localizzare.

L'idea è quella di implementare diverse reti, una per gruppo di punti, per predire le coordinate dei seguenti raggruppamenti (il numero indica l'indice nel file di testo)
- 0,1,4,5 (S,N,A,B)
- 2,3,7,8 (Sna, Snp, Gn, Go)
- 9,10,11,12 (U1 e L1 root e tip)
- 6,13 (pg, mesial)

Nel file di testo, ogni riga contiene l'indice del punto, alcune informazioni inutili, e le coordinate del punto (le ultime due colonne)

Eventualmente, se si ha tempo, confrontando i risultati con una rete che prova a predirli tutti insieme.


Altre considerazioni:
- Lavoro in ambito di immagini: ho un DS con tante immagini radiografiche
- Su ogni immagine devo analizzare dei punti (14 punti) per poter fare una regressione di valori
- (L'accuratezza se anche è al 60% devo sapere il perchè lo è)
- Devo analizzare il DS molto bene, facendo un buon pre-processing!


## Dataset

Il Dataset usato è così composto:

![alt text](/altro/dataset-composizione.png)

Ogni file di testo è strutturato nel seguente modo:

![alt text](/altro/file-txt-composizione.png)

## Funzionamento

Inserire lo zip del dataset dentro `/data/raw`.

