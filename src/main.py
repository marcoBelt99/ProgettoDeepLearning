from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd

from configs.parametri_app import DATASET_DIR, DATAFRAME_MASTER
from data.dataset.repere_dataset import crea_split_dataset


# Leggo il DataFrame
df = pd.read_csv(DATAFRAME_MASTER)

# Creo train e test set
train_set, test_set = crea_split_dataset(df, img_dir=DATASET_DIR, img_size=224, train_ratio=0.8)

# Creo i DataLoader
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# Controllo

## Ricordo che il dataloader può diventare un iterator (ricorda il pattern di ISA)
#  La stessa cosa posso ottenerla facendo un for
img, keypoints = next(iter(train_loader))
print(img.shape)        # torch.Size([16, 3, 224, 224])
print(keypoints.shape)  # torch.Size([16, 28]) (14 punti × 2)
