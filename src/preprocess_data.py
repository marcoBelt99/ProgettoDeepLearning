import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import zipfile
from PIL import Image



def parse_txt_file(txt_path):
    """
    Parsa un file .txt e estrae le coordinate dei keypoint.
    
    Args:
        txt_path: Percorso al file .txt
    
    Returns:
        List di tuple (x, y) rappresentanti le coordinate
    """
    keypoints = []
    
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if line:  # Ignora righe vuote
            # Cerca pattern numerici per le coordinate
            # Assumendo formato: indice, info_inutile, x, y
            numbers = re.findall(r'-?\d+\.?\d*', line)
            
            if len(numbers) >= 2:
                # Prendi gli ultimi due numeri come coordinate x, y
                x, y = float(numbers[-2]), float(numbers[-1])
                keypoints.append((x, y))
    
    return keypoints

def create_dataset_dataframe(data_dir):
    """
    Crea un DataFrame con tutte le immagini e i relativi keypoint.
    
    Args:
        data_dir: Directory contenente immagini e file .txt
    
    Returns:
        pandas.DataFrame con colonne: image_path, keypoints, num_keypoints
    """
    data_dir = Path(data_dir)
    dataset_rows = []

    
    # Trova tutte le immagini
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in data_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Trovate {len(image_files)} immagini")
    
    for img_path in image_files:
        # Cerca il file .txt corrispondente
        txt_path = img_path.with_suffix('.txt')
        
        if txt_path.exists():
            try:
                keypoints = parse_txt_file(txt_path) # chiamata a parse_txt_file()
                
                if keypoints:  # Solo se abbiamo trovato keypoint
                    dataset_rows.append({
                        'image_path': str(img_path),
                        'keypoints': keypoints,
                        'num_keypoints': len(keypoints)
                    })
                    
            except Exception as e:
                print(f"Errore nel parsing di {txt_path}: {e}")
        else:
            print(f"File .txt non trovato per {img_path}")
    
    df = pd.DataFrame(dataset_rows)
    print(f"Dataset creato con {len(df)} campioni")
    return df

def analyze_dataset(df):
    """
    Analizza il dataset e genera statistiche e visualizzazioni.
    """
    print("=== ANALISI DATASET ===")
    print(f"Numero totale di campioni: {len(df)}")
    print(f"Numero medio di keypoint per immagine: {df['num_keypoints'].mean():.2f}")
    print(f"Min keypoint per immagine: {df['num_keypoints'].min()}")
    print(f"Max keypoint per immagine: {df['num_keypoints'].max()}")
    
    # Analisi delle coordinate
    all_x_coords = []
    all_y_coords = []
    
    for keypoints in df['keypoints']:
        for x, y in keypoints:
            all_x_coords.append(x)
            all_y_coords.append(y)
    
    coord_stats = pd.DataFrame({
        'x': all_x_coords,
        'y': all_y_coords
    })
    
    print("\n=== STATISTICHE COORDINATE ===")
    print(coord_stats.describe())
    
    # Visualizzazioni
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribuzione numero keypoint
    axes[0, 0].hist(df['num_keypoints'], bins=20, edgecolor='black')
    axes[0, 0].set_title('Distribuzione Numero Keypoint per Immagine')
    axes[0, 0].set_xlabel('Numero Keypoint')
    axes[0, 0].set_ylabel('Frequenza')
    
    # Distribuzione coordinate X
    axes[0, 1].hist(all_x_coords, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribuzione Coordinate X')
    axes[0, 1].set_xlabel('Coordinata X')
    axes[0, 1].set_ylabel('Frequenza')
    
    # Distribuzione coordinate Y
    axes[1, 0].hist(all_y_coords, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribuzione Coordinate Y')
    axes[1, 0].set_xlabel('Coordinata Y')
    axes[1, 0].set_ylabel('Frequenza')
    
    # Scatter plot delle coordinate
    axes[1, 1].scatter(all_x_coords, all_y_coords, alpha=0.6)
    axes[1, 1].set_title('Scatter Plot Coordinate X vs Y')
    axes[1, 1].set_xlabel('Coordinata X')
    axes[1, 1].set_ylabel('Coordinata Y')
    
    plt.tight_layout()
    # plt.savefig('data/processed/dataset_analysis.png')
    plt.savefig('src/utils/dataset_analysis.png')
    plt.show()
    
    return coord_stats

def visualize_sample_with_keypoints(df, sample_idx=0):
    """
    Visualizza un'immagine di esempio con i suoi keypoint sovrapposti.
    """
    sample = df.iloc[sample_idx]
    img_path = sample['image_path']
    keypoints = sample['keypoints']
    
    # Carica l'immagine
    img = Image.open(img_path)
    img_array = np.array(img)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_array)
    
    # Sovrapponi i keypoint
    x_coords = [kp[0] for kp in keypoints]
    y_coords = [kp[1] for kp in keypoints]
    
    plt.scatter(x_coords, y_coords, c='red', s=50, marker='x', linewidths=2)
    
    # Numera i keypoint
    for i, (x, y) in enumerate(keypoints):
        plt.annotate(str(i), (x, y), xytext=(5, 5), 
                    textcoords='offset points', color='red', fontweight='bold')
    
    plt.title(f'Immagine con {len(keypoints)} Keypoint')
    plt.axis('off')
    plt.savefig(f'data/processed/sample_{sample_idx}_with_keypoints.png', 
                bbox_inches='tight', dpi=150)
    plt.show()


def unzip_dataset(data_dir):
    '''
    Unzip archivio del dataset.
    '''
    DATASET_NAME_FILE = 'dataset_14_punti_pulito_06072023'
    DATASET_NAME_FILE_ZIP = data_dir + DATASET_NAME_FILE + '.zip'

    # Verifico che la directory esista
    if os.path.exists(data_dir):
        print("Directory trovata: ", data_dir)
        print("Contenuto della directory:", os.listdir(data_dir))

        # Verifico che il file zip esista
        if os.path.exists(DATASET_NAME_FILE_ZIP):
            print("File zip del Dataset presente.")
        else:
            print("File zip del Dataset non trovato.")
    else:
        print("Directory non trovata:", data_dir)

    if 'dataset' not in os.listdir('../data'):
        # Estrazione del file: solo quando la cartella 'dataset' non e' gia' presente
        with zipfile.ZipFile(DATASET_NAME_FILE_ZIP, 'r') as zip_ref:
            # zip_ref.extractall(data_dir)
            zip_ref.extractall('../data')
    else:
        print("\nLa cartella e' gia' stata estratta dallo zip\n")

    # Controllo del contenuto estratto
    print(os.listdir(data_dir))


def main():

    """Funzione principale per il preprocessing."""
    
    ## Configurazione dei percorsi
    data_raw_dir = "../data/raw/"  #  Dati grezzi (su archivio zip)
    dataset_dir = "../data/dataset" # Dati unzippati dal dataset

    
    print("\nInizio preprocessing del dataset...")

    ## La prima operazione è quella di unzippare il DS se è la prima volta
    unzip_dataset( data_raw_dir )

    ## Creo un DataFrame Pandas a partire dai dati estratti
    df = create_dataset_dataframe(dataset_dir)
    
    if len(df) == 0:
        print("Nessun dato trovato! Verifica il percorso del dataset.")
        return
    
    # Analizza il dataset
    coord_stats = analyze_dataset(df)
    
    # Visualizza alcuni campioni
    print("\nVisualizzazione campioni...")
    for i in range(min(3, len(df))):  # Primi 3 campioni
        try:
            visualize_sample_with_keypoints(df, i)
        except Exception as e:
            print(f"Errore nella visualizzazione del campione {i}: {e}")
    
    # Salva il DataFrame processato
    df.to_pickle("data/processed/dataset.pkl")
    coord_stats.to_csv("data/processed/coordinate_stats.csv", index=False)
    
    print(f"\nPreprocessing completato!")
    print(f"Dataset salvato in: data/processed/dataset.pkl")
    print(f"Statistiche salvate in: data/processed/coordinate_stats.csv")

if __name__ == "__main__":
    main()