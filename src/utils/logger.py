import csv
import os
from datetime import datetime



INTESTAZIONI_FILE_LOG = [
    "timestamp",
    "esperimento",
    "best_epoch",
    "best_mae",
    "lr",
    "batch_size",
    "epochs_run",
    "freeze_until",
    "head",
    "img_size",
    "optimizer",
    "scheduler",
    "seed",
]


def log_experiment(csv_path, data_dict):
    """
    Semplice sistema di registrazione di tutte le metriche in un file CSV.
    """

    file_exists = os.path.isfile(csv_path)

    # timestamp di default se non fornito
    row = {k: "" for k in INTESTAZIONI_FILE_LOG}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row.update(data_dict)  # sovrascrive i campi presenti

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)
