import os
import zipfile
import cv2
import pandas as pd
import matplotlib.pyplot as plt


# === CONFIGURAZIONE ===
DATASET_DIR = "dataset"
ZIP_NAME = "dataset_14_punti_pulito_06072023.zip"
EXTRACT_DIR = os.path.join(DATASET_DIR, "raw")
OUTPUT_CSV = os.path.join(DATASET_DIR, "keypoints.csv")

# === 1. Estrazione archivio ZIP ===
def estrai_dataset():
    zip_path = os.path.join(DATASET_DIR, ZIP_NAME)
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"[OK] Archivio estratto in {EXTRACT_DIR}")
    else:
        print(f"[INFO] Cartella {EXTRACT_DIR} già presente, salto estrazione.")

# === 2. Parsing immagini + file txt ===
def data_preprocessing():
    '''
    Ha il compito di fare il preprocessing dei dati
    '''
    rows = []
    for file in os.listdir(EXTRACT_DIR):
        if file.endswith(".txt"):
            txt_path = os.path.join(EXTRACT_DIR, file)
            img_name = file.replace(".txt", ".jpg")  # cambia se non è .jpg
            img_path = os.path.join(EXTRACT_DIR, img_name)

            if not os.path.exists(img_path):
                print(f"[ATTENZIONE] Immagine mancante per {file}")
                continue

            # dimensioni immagine
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape

            # parsing del file txt
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    idx = int(parts[0])   # indice del punto
                    x = float(parts[-2])  # penultima colonna
                    y = float(parts[-1])  # ultima colonna

                    # normalizzazione
                    x_norm = x / w
                    y_norm = y / h

                    rows.append({
                        "image": img_name,
                        "point_idx": idx,
                        "x": x_norm,
                        "y": y_norm
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Preprocessing completato. CSV salvato in: {OUTPUT_CSV}")
    return df

# === 3. Analisi statistica e grafici ===
def analyze_data(df: pd.DataFrame):
    print("\n=== STATISTICHE SUI PUNTI ===")
    print(df[["x", "y"]].describe())

    # distribuzione x
    plt.figure(figsize=(10,4))
    plt.hist(df["x"], bins=50, alpha=0.7)
    plt.title("Distribuzione coordinate X normalizzate")
    plt.xlabel("x")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.savefig(os.path.join(DATASET_DIR, "hist_x.png"))
    plt.close()

    # distribuzione y
    plt.figure(figsize=(10,4))
    plt.hist(df["y"], bins=50, alpha=0.7, color="orange")
    plt.title("Distribuzione coordinate Y normalizzate")
    plt.xlabel("y")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.savefig(os.path.join(DATASET_DIR, "hist_y.png"))
    plt.close()

    print(f"[OK] Grafici salvati in {DATASET_DIR}")

# === MAIN ===
if __name__ == "__main__":
    estrai_dataset()
    df = data_preprocessing()
    analyze_data(df)
