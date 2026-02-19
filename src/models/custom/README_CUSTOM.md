# models/custom

Pacchetto "custom" per allenare 4 modelli diversi (uno per gruppo) mantenendo la pipeline del progetto:
- stesso dataset: `src.data.dataset.repere_dataset_1.RepereKeypointsDataset`
- stesso split 80/10/10
- stessa loss/metriche (SmoothL1 + MAE/MED in pixel)
- trasformazioni Albumentations (per-group)

## Struttura
- `main_modelli_custom.py`: avvio training (un gruppo o tutti)
- `factory.py`: costruzione modello per gruppo
- `trasformazioni_custom.py`: trasformazioni per-group (train/test)
- `train_and_validate_custom.py`: training loop + early stopping + scheduler + output (logs/plots/checkpoints)
- `parametri_modello.py`: iperparametri per-group (CPU-friendly)
- `gruppo1/`, `gruppo2/`, `gruppo3/`, `gruppo4/`: definizioni dei modelli + cartelle output

## Come usare
1) Copia questa cartella in: `src/models/custom/` del tuo progetto.
2) Da root del progetto, lancia:
   `python src/models/custom/main_modelli_custom.py`

Nel file `main_modelli_custom.py` trovi la variabile `GRUPPO_DA_ALLENARE`.
- Se è `"GRUPPO1"` allena solo gruppo 1
- Se è `None` allena tutti e 4 in sequenza

Output:
- `src/models/custom/gruppoX/logs/` (TensorBoard)
- `src/models/custom/gruppoX/plots/` (griglia test + curve)
- `src/models/custom/gruppoX/checkpoints/` (best early checkpoint)
