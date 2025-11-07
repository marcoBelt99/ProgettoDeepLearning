import torch

from src.dataset import CustomKeypointDataset
from src.model import KeypointModel
from src.train import train_model
 
def main():

    '''
    File di avvio progetto.
    Punto di ingresso del progetto: da qui chiamo le funzioni di addestramento per i 4 diversi modelli.
    '''
   # Definizione dei gruppi
    gruppi_punti = {
        "gruppo_1": [0, 1, 4, 5],
        "gruppo_2": [2, 3, 7, 8],
        "gruppo_3": [9, 10, 11, 12],
        "gruppo_4": [6, 13]
    }

    for nome_gruppo, indici_punti in gruppi_punti.items():
        print(f"--- Addestramento per {nome_gruppo} ---")
        
        # 1. Preparazione dei dati
        num_keypoints = len(indici_punti)
        dataset = CustomDataset(img_dir="...", txt_dir="...", keypoint_indices=indici_punti)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 2. Inizializzazione del modello
        modello = KeypointModel(num_output=num_keypoints * 2)
        
        # 3. Ciclo di addestramento e validazione
        # ... (Il tuo codice di addestramento, ottimizzatore, loss, ecc.)
        
        # 4. Salvataggio del modello
        torch.save(modello.state_dict(), f"modelli/{nome_gruppo}_model.pth")
        
    print("--- Addestramento completato per tutti i gruppi! ---")
 
if __name__ == '__main__':
   

   '''
   Questa strategia ti permetterà di specializzare ogni rete per un compito più circoscritto,
   il che spesso porta a prestazioni migliori rispetto a un'unica rete che deve imparare a prevedere
   tutti i punti contemporaneamente, specialmente se i raggruppamenti hanno una logica anatomica o strutturale.
   
   '''
   main()   