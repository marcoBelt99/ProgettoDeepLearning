import matplotlib.pyplot as plt

def visualizza_immagine(nome_immagine):
    image = plt.imread(nome_immagine) # image è un array numpy (ndarray)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(image)
    ax.axis('off') # rimuovo gli assi cartesiani che non mi interessano
    plt.title("Cefalometria")
    plt.show()

visualizza_immagine('cefalometria.jpg')