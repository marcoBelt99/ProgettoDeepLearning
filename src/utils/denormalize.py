import torch
import torchvision
import matplotlib.pyplot as plt


class NormalizeInverse(torchvision.transforms.Normalize):
  def __init__(self, mean: List[float], std: List[float]) -> None:
    """
    Ricostruisci le immagini di input invertendo le trasformazioni di normalizzazione.

    Args:
      mean: La media usata per normalizzare le immagini
      std: la deviazione standard usata per normalizzare le immagini.
    """
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    super().__init__(mean=mean_inv, std=std_inv)

  def __call__(self, tensor):
    return super().__call__(tensor.clone())

def show_grid(dataset: torchvision.datasets.ImageFolder,
              process: Callable = None) -> None:
  """
  Mostra una grid con immagini random prese da un batch del dataset.

  Args:
    dataset: il dataset contenente le immagini.
    process: la funzione da applicare alle immagini al fine di poterle visualizzare
  """
  fig = plt.figure(figsize=(15, 5))
  indices_random = np.random.randint(10, size=10, high=len(dataset))

  for count, idx in enumerate(indices_random):
    fig.add_subplot(2, 5, count + 1 ) # 2 righe e 5 colonne
    title = dataset.classes[dataset[idx][1]]
    plt.title(title)
    image_processed = process(dataset[idx][0]) if process is not None else data_val
    plt.imshow(transforms.ToPILImage()(image_processed))
    plt.axis("off")

# Mostra alcuni esempi
denormalize = NormalizeInverse(mean_image_net, std_image_net)
show_grid(data_val, process=denormalize)