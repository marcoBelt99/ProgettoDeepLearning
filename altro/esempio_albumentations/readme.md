## Esempio di pipeline minimale che funziona con i keypoints

Il parametro `keypoint_params` di `A.Compose` definisce il formato per le coordinate
dei keypoints

```python
transform = A.Compose([
    A.RandomCrop(width=330, height=330),
    A.RandomBrightnessContrast(p=0.2),
], keypoint_params=A.KeypointParams(format='xy'))
```



 In modo più completo, con tutti i possibili settaggi  per `A.KeypointParams`, come:
 - `label_fields`: oltre ai keypoints, per alcuni task di computer vision ci possono essere
                 altre etichette, come elbow, knee o wrist (nel caso ad esempio della Pose Estimation).
 - `remove_invisible`: se True, Albumentations non ritorna i keypoints invisibili
                     (quindi, se non passo questo argomento Albumentations non ritorna i keypoints invisibili).
                     Perchè invisibile? Perchè dopo la augmentation, alcuni keypoints possono essere invisibili
                     poichè risulteranno posizionati al di fuori dell'area dell'immagine aumentata
                     Es) se ritaglio (crop) una parte di immagine, tutti i keypoints al di fuori dell'area croppata
                     diventeranno invisibili.
 - `angle_in_degrees`: se True (di default lo è), Albumentations si aspetta che il valore dell'angolo nei
                     formati xya, xyas e xysa sia definito in angoli. Se angle_degrees è impostato a False, Albumentations
                     si aspetta che il valore dell'angolo sia specificato in radianti.
 Questi settaggi non hanno effetto per i formati xy e yx, perchè questi formati non usano gli angoli.


```python
transform = A.Compose([
    A.RandomCrop(width=330, height=330),
    A.RandomBrightnessContrast(p=0.2),
], keypoint_params=A.KeypointParams(format='xy',
                                    label_fields=['class_labels'],
                                    remove_invisible=True,
                                    angle_in_degrees=True))
```


Albumentations expects keypoints to be represented as a NumPy array with shape (num_keypoints, 2+).


Nella pagina github di Albumentations si fa riferimento ad alcune trasformazioni
in ambito immagini mediche: https://github.com/albumentations-team/AlbumentationsX come:
- GridDistortion
- ElasticTransform

Anche per la sezione Keypoints augmentations sono mezionate:
- HorizontalFlip
- ShiftScaleRotate


