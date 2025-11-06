import torch
import albumentations as A

# Dato che voglio fare feature extraction e fine tuning devo normalizzare i dati
# nel range in cui il dataset è già stato allenato.
# Questi valori posso trovarli cercandoli su google
mean_image_net = [0.485, 0.456, 0.406]
std_image_net = [0.229, 0.224, 0.225]

# La cosa bella che possa fare nel mio DS è che posso passare al metodo di creazione della classe del dataset tutte
# le trasformazioni che voglio applicare alle mie immagini. Queste trasformazioni sono applicate quando il batch è
# caricato. Anche per la data augmentation, viene aumentato ogni volta che viene chiamato un batch, poichè questo
# emula differenti dati. La trasformazione è applicata quando viene richiesto un batch del Dataloader.

# Creo una trasformazione per normalizzazione
normalize = A.Normalize(mean=mean_image_net, std=std_image_net)
size_image = 128  # 64 o 224  ==> il DS originale è a 64 px, ma io sto usando 128 perchè, dato che la rete è pre-allenata,
# qui sto usando una risoluzione differente perchè sto usando una rete pre-trained, che alla fine usa un AvgPooling
# se provo questa altre due risoluzioni (64 o 224), ottengo differenti risultati

# Altre trasformazioni per aumentare i dati?
data_transforms = {
    # a training time ha senso fare trasformazioni random perchè la rete ha bisogno di "imparare"
    # il placement, e servono differenti dati
    'train': transforms.Compose([A.RandomResizedCrop(size_image),
                                 A.RandomHorizontalFlip(),
                                 A.ToTensor(),  # butto tutto dentro un tensore
                                 normalize  # applico la normalizzazione. Va fatta dopo la augmentation!
                                 ]),

    # a test-time non posso usare un RandomCrop, perchè per ogni possibile crop posso avere un risultato diverso
    # invece, devo usare CenterCrop, perchè di solito gli oggetti sono nel centro dell'immagine
    'val': transforms.Compose([transforms.Resize(int(size_image * 1.2)),
                               # sto facendo un 20% in più del resize della dimensione dell'immagine: se ho un'immagine grande
                               # ad esempio 224, la ridimensiono a (ad esempio) 225 e ...
                               transforms.CenterCrop(size_image),  # questo non può essere random
                               transforms.ToTensor(),  # trasformo in tensore
                               normalize  # normalizzo
                               ])
}

# Poi creo il dataset, che ho già splittato prima tramite split-folders.
# Avendo la suddivisione in cartelle posso usare la classe ImageFolder.
# Qui creo due differenti dataset: uno per training e uno per validazione, ai quali
# passo differenti trasformazioni!
# [Invece, in altri esercizi solitamente per creare il dataset devo: estendere la classe Dataset; etc.]
# Quando faccio Fine-tuning è molto più velcoe fare così come sto facendo ora: preparo i dati separati in cartelle
data_train = torchvision.datasets.ImageFolder(path_ds_train, data_transforms['train'])
data_val = torchvision.datasets.ImageFolder(path_ds_val, data_transforms['val'])

# Leggo il numero di classi
classes = data_train.classes
num_classes = len(classes)

print(
    f'Samples -> Train = {len(data_train)} - Val = {len(data_val)} - Classi = {classes}')  # numero di esempi in entrambi i dataset. classes mi mostra il nome delle classi esistenti
