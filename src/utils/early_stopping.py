import numpy as np
import torch
import os

class EarlyStopping:
    '''
    Implementa la seguente logica:
    tiene il best, incrementa counter quando non migliora, stop quando supera patience.
    Salva state_dict() del modello sul best
    '''
    def __init__(self, patience=8, min_delta=0.0, checkpoint_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False

    def __call__(self, metric, model):

        if metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter = 0

            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
