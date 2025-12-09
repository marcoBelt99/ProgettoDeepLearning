import torch
from utils.metriche import calcola_mae_pixel, mean_euclidean_distance


##########################################
############# calcola_mae_pixel ##########
##########################################

def test_calcola_mae_pixel_zero_error():
    """
    Se predictions == targets, il MAE deve essere banalmente 0.
    """
    predizioni = torch.tensor([[0.5, 0.2, 0.8, 0.4]])
    targets = predizioni.clone()
    img_size = 224

    mae = calcola_mae_pixel( predizioni, targets, img_size )

    assert mae == 0.0


def test_calcola_mae_pixel_known_value():
    """
    Test su valori noti: pred = target + costante.

   1) In pratica io ho:
    - pred → (0.1, 0.1)
    - true → (0.2, 0.3)
    che rappresentano keypoints in [0,1].
   2) Allora intanto li denormalizzo, moltiplicando per img_size che ipotizzo essere=100
    per semplificare i calcoli:
    - pred * 100    → (10, 10)
    - target * 100  → (20, 30)
   3) Mi calcolo la differenza in valore assoluto:
    diff = |pred_pixels - target_pixels|
         = |(10,10)     - (20,30)      |
         = (10,20)
   4) Faccio la media delle differenze assolute:
    MAE = (10 + 20) / 2 = 15
    """
    predetti = torch.tensor( [[0.1, 0.1]] )
    targets = torch.tensor( [[0.2, 0.3]] )
    img_size = 100  # (così è più facile fare i conti)

    # differenze in pixel:
    # pred * 100 = (10,10)
    # target * 100 = (20,30)
    # diff = (10,20)
    # MAE = media = (10+20)/2 = 15
    expected_mae = 15.0

    mae = calcola_mae_pixel( predetti, targets, img_size )

    # Non posso confrontare numeri in floating point con l’uguaglianza diretta (==).
    # Ad esempio:
    # mae = 15.0000000001
    # expected = 15.0
    # sono praticamente uguali, ma a livello numerico non sono esattamente identici.
    # pertanto assert mae == expected_mae può fallire, quindi testo sotto un certo troncamento
    assert abs(mae - expected_mae) < 1e-6





################################################
############# mean_euclidean_distance ##########
################################################
"""
Vediamo il codice, riga per riga
1️⃣ Si fa il reshape (view)
preds = preds.view(-1, num_outputs_modello, 2) * img_size
targets = targets.view(-1, num_outputs_modello, 2) * img_size

Cosa significa?

Il tuo modello produce un tensore:

[batch_size, num_outputs_modello * 2]


Esempio per 14 punti:

shape = [batch, 28]


Ogni punto ha 2 valori: x, y

Quindi si trasforma in:

[batch, num_outputs_modello, 2]


Esempio:

[8, 14, 2]

2️⃣ Si moltiplica per img_size

Perché?

Durante training tu normalizzi i keypoints in [0,1].

Quindi:

0.0 → 0 pixel

1.0 → img_size pixel (224 nel tuo caso)

Moltiplicando:

* img_size


li riporti alle coordinate reali in pixel.

3️⃣ Calcolo della distanza euclidea punto per punto
dists = torch.norm(preds - targets, dim=2)


Per ogni coppia di coordinate (x_pred, y_pred) e (x_true, y_true):

dist = sqrt( (x_pred - x_true)^2 + (y_pred - y_true)^2 )


La norma L2.

E produce un tensore:

shape = [batch, num_outputs_modello]


Ovvero:
la distanza di ogni punto, per ogni immagine.

4️⃣ Media su tutti i punti e su tutto il batch
return dists.mean().item()


Questo fa:

media sulle immagini del batch

media sui keypoint della stessa immagine

E produce un solo numero:

Distanza euclidea media globale, in pixel.

🎯 RISULTATO

La funzione restituisce:

👉 “In media, quanto sono lontani i keypoint predetti da quelli reali?”

Espresso in pixel, quindi interpretabile.

🔢 Esempio numerico chiarissimo

Supponiamo 2 keypoint:

Pred:

[(0.2, 0.3), (0.7, 0.8)]


True:

[(0.3, 0.4), (0.6, 0.9)]


img_size = 100 → denormalizzo:

Pred:

(20, 30), (70, 80)


True:

(30, 40), (60, 90)


Calcolo distanze:

Punto 1:

sqrt( (20–30)^2 + (30–40)^2 ) = sqrt(100 + 100) ≈ 14.14


Punto 2:

sqrt( (70–60)^2 + (80–90)^2 ) ≈ 14.14


Media:

MED = (14.14 + 14.14) / 2 = 14.14 pixel


La funzione restituisce:

14.14

🧠 In sintesi

La funzione:

✔ prende keypoints normalizzati
✔ li trasforma in coordinate reali (pixel)
✔ calcola la distanza euclidea punto per punto
✔ fa la media su batch e su punti
✔ restituisce quanto il modello sbaglia in pixel
"""
def test_mean_euclidean_distance_zero():
    """Se pred = target, la distanza euclidea media è 0."""
    preds = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    targets = preds.clone()
    img_size = 224
    num_outputs_modello = 2  # 2 punti → 4 valori

    dist = mean_euclidean_distance(preds, targets, img_size, num_outputs_modello)

    assert dist == 0.0


def test_mean_euclidean_distance_known_case():
    """Test con valori semplici e prevedibili."""
    preds = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
    img_size = 100
    num_outputs_modello = 2  # 2 punti (x1,y1,x2,y2)

    # Dopo scaling:
    # preds  → (0,0), (100,100)
    # targets → (0,100), (100,0)
    #
    # distanze:
    # punto1: sqrt( (0-0)^2 + (0-100)^2 ) = 100
    # punto2: sqrt( (100-100)^2 + (100-0)^2 ) = 100
    # media = 100

    expected_med = 100.0

    dist = mean_euclidean_distance(preds, targets, img_size, num_outputs_modello)

    assert abs(dist - expected_med) < 1e-6


import torch


def test_mean_euclidean_distance_variable_points():
    img_size = 200

    for num_pts in [1, 5, 14]:
        preds = torch.rand((2, num_pts * 2))
        targets = preds.clone()  # MED deve essere zero

        dist = mean_euclidean_distance(preds, targets, img_size, num_pts)

        assert dist == 0.0
