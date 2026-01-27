import torch

def calcola_mae_pixel(predictions, targets, img_size):
    """
    Calcola il MAE come metrica per i pixel.
    In particolare, i passaggi eseguiti dalla funzione sono:
    - Prende keypoint normalizzati (valori tra 0 e 1)
    - Li riporta a coordinate pixel, moltiplicando per img_size
    - Calcola la differenza assoluta
    - Fa la media, ottienendo il Mean Absolute Error
    """
    pred_pixels = predictions * img_size # moltiplicando per img_size vado a denormalizzare e passo da valori in [0,1] a valori in [0, img_size]
    target_pixels = targets * img_size #   moltiplicando per img_size vado a denormalizzare e passo da valori in [0,1] a valori in [0, img_size]

    return (torch.abs(pred_pixels - target_pixels) .mean(dim=1).mean().item())
    # pred_pixels - target_pixels ha shape [batch, 2*K]
    # mean(dim=1) fa la media su tutte le coordinate x,y di tutti i punti. Ottengo quindi un numero per sample
    # .mean() faccio la media per gli esempi

def mean_euclidean_distance(preds, targets, img_size, num_outputs_modello):
    """
    Calcola la distanza euclidea media tra i keypoints predetti e quelli reali, espressa in pixel.
    È una metrica molto usata in rilevamento di punti anatomici, facial landmarks e keypoint detection.

    """
    # preds = preds.view(-1, 14, 2) * img_size # moltiplicando per img_size vado a denormalizzare, passando da valori in [0,1] ai veri valori assunti dai pixel
    preds = preds.view(-1, num_outputs_modello, 2) * img_size # moltiplicando per img_size vado a denormalizzare, passando da valori in [0,1] ai veri valori assunti dai pixel
    # targets = targets.view(-1, 14, 2) * img_size
    targets = targets.view(-1, num_outputs_modello, 2) * img_size
    dists = torch.norm(preds - targets, dim=2)
    return dists.mean().item()