import torch

def mean_euclidean_distance(preds, targets, num_punti=14):
    '''
    Misura di valutazione delle performance sperabilmente piu' efficace della semplice
    MSE. La MSE potrebbe essere difficile da interpretare in termini spaziali.
    Invece, con la MED posso dire "In media, i punti sono predetti con un errore di x pixel."
    '''
    preds = preds.view(-1, num_punti, 2)
    targets = targets.view(-1, num_punti, 2)
    dists = torch.norm(preds - targets, dim=2)
    return dists.mean().item()