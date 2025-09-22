import torch
from models.cnn_model import Net
from utils.data_preparation import prepare_data_loaders
from utils.visualization import im_show, plot_metrics
import torchvision
from training.train_validate import train_and_validate
from testing.test_model import test_model
 
def main():
    # Configurazioni iniziali
    epochs = 10
    batch_size = 64
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
    # Preparazione dei DataLoader per il training, la validazione e il test
    train_loader, validation_loader, test_loader = prepare_data_loaders(batch_size)
    
    # Inizializzazione del modello
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    
    # Visualizzazione di alcune immagini dal training set con le etichette corrispondenti
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    im_show(torchvision.utils.make_grid(images))
 
    # Stampa delle etichette corrispondenti
    print('Labels:', ' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))
 
    # Esegui la funzione di addestramento e validazione e ottieni le metriche
    train_losses, validation_losses, validation_accuracies = train_and_validate(net, train_loader, validation_loader, epochs)
   
    # Visualizza la loss per training e validazione su un grafico
    plot_metrics(train_losses, validation_losses, metric_name="Loss")

    # Visualizza l'accuratezza della validazione su un grafico separato
    plot_metrics(validation_accuracies, metric_name="Accuracy")
 
    # Test del modello
    test_model(net, test_loader, classes)
 
if __name__ == '__main__':
   main()