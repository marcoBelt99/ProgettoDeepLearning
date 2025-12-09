import csv
import os

def log_experiment(csv_path, data_dict):
    """
    Semplice sistema di registrazione di tutte le metriche in un file CSV.
    """

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)
