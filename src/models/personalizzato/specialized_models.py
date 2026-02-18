import torch
import torch.nn as nn
from torchvision import models


class BaseLandmarkModel(nn.Module):
    """
    Classe base per evitare di riscrivere codice comune.
    """

    def __init__(self, num_punti):
        super(BaseLandmarkModel, self).__init__()
        self.num_output_coords = num_punti * 2
        self.backbone = None
        self.head = None

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# --- GRUPPO 1: CRANIO (Strutture grandi) ---
class ModelloGruppo1(BaseLandmarkModel):
    def __init__(self, pretrained=True):
        super().__init__(num_punti=4)  # 4 punti

        # Qui usiamo ResNet18 come hai fatto finora, funziona bene per strutture globali
        full_resnet = models.resnet18(weights='DEFAULT' if pretrained else None)

        # Strategia di freezing: Freeziamo fino al layer 2
        for name, param in full_resnet.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Prendiamo tutto tranne l'ultimo layer fc
        self.backbone = nn.Sequential(*list(full_resnet.children())[:-1])
        self.input_features = full_resnet.fc.in_features

        # Head personalizzata
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_output_coords)
        )




#  GRUPPO 2: NASO/MANDIBOLA (Forme complesse)
class ModelloGruppo2(BaseLandmarkModel):
    def __init__(self, pretrained=True):
        super().__init__(num_punti=4)

        # Anche qui ResNet18 va bene, ma magari voglio più capacità di apprendimento
        # Sblocco tutto tranne il layer0 per permettere più adattamento
        full_resnet = models.resnet18(weights='DEFAULT' if pretrained else None)

        for name, param in full_resnet.named_parameters():
            if "layer1" in name or "bn1" in name or "conv1" in name:
                param.requires_grad = False  # Freezo solo l'inizio
            else:
                param.requires_grad = True

        self.backbone = nn.Sequential(*list(full_resnet.children())[:-1])
        self.input_features = full_resnet.fc.in_features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_features, self.num_output_coords)  # Head semplice
        )



# Sta andando bene (ottengo  6.354 di MAE)
#  GRUPPO 3: DENTI (Dettaglio fine)
class ModelloGruppo3(BaseLandmarkModel):
    def __init__(self, pretrained=True):
        super().__init__(num_punti=4)

        # Per i denti serve alta risoluzione. Uso ResNet18 ma cambio strategia.
        # EfficientNet sarebbe meglio, ma rimango su ResNet per ora per stabilità su CPU.
        full_resnet = models.resnet18(weights='DEFAULT' if pretrained else None)

        # Qui NON freezo nulla o quasi, perché i feature dei denti sono molto diversi da ImageNet
        for name, param in full_resnet.named_parameters():
            # Blocco tutto tranne layer4 e fc
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False  # FREEZED
            else:
                param.requires_grad = True  # TRAIN

        self.backbone = nn.Sequential(*list(full_resnet.children())[:-1])
        self.input_features = full_resnet.fc.in_features

        # Head più profonda per imparare relazioni non lineari complesse
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_output_coords)
        )


# Il gruppo 4 fa schifo con mobilenetv3 come backbone
"""
#  GRUPPO 4: MENTO (Semplice e pochi punti)
class ModelloGruppo4(BaseLandmarkModel):
    def __init__(self, pretrained=True):
        super().__init__(num_punti=2)  # Solo 2 punti!

        # CAMBIAMO BACKBONE!
        # Usiamo MobileNetV3 Small: è leggerissima, ideale per la CPU
        # e per un task semplice con soli 2 punti.
        mobilenet = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)

        # Freeziamo la prima parte
        for param in mobilenet.features.parameters():
            param.requires_grad = False

        self.backbone = mobilenet.features  # MobileNet outputta feature map diverse
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),  # 576 è l'uscita tipica di MobileNetV3 Small
            nn.ReLU(),
            nn.Linear(128, self.num_output_coords)
        )
"""




#  GRUPPO 4: MENTO (Semplice e pochi punti)
class ModelloGruppo4(BaseLandmarkModel):
    def __init__(self, pretrained=True):
        super().__init__(num_punti=2)  # Qui passo 2, perché il Mento ha solo 2 punti (Pg, Mesial)

        # Abbandono MobileNet. Uso la strategia vincente del GRUPPO 1 (ResNet18)
        full_resnet = models.resnet18(weights='DEFAULT' if pretrained else None)

        # Copio la strategia di freezing che ha funzionato bene:
        # Freezo i primi layer (feature generiche) e alleno layer3, layer4 e head.
        for name, param in full_resnet.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Prendiamo tutto tranne l'ultimo layer fc originale
        self.backbone = nn.Sequential(*list(full_resnet.children())[:-1])
        self.input_features = full_resnet.fc.in_features

        # Uso la stessa Head "potente" del Gruppo 1
        # (Flatten -> Linear -> ReLU -> Dropout -> Linear finale)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_output_coords)  # num_output_coords sarà 4 (2 punti * 2 coordinate)
        )