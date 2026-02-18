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
        self.min_delta = min_delta # impedisce al modello di continuare l'addestramento se il miglioramento è insignificante (ad es. di solo 0.000001 pixel nel MAE)
        self.checkpoint_path = checkpoint_path

        self.counter = 0
        self.best_score = np.inf # così mi assicuro che la prima epoca sia sempre considerata come il nuovo record
        self.early_stop = False

    def __call__(self, metrica, model):

        if metrica < self.best_score - self.min_delta:
            self.best_score = metrica
            self.counter = 0

            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
