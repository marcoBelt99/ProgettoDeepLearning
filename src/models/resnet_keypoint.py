import torch
import torch.nn as nn
import torchvision.models as models


class ResNetKeypoint(nn.Module):
    """
    ResNet adattata per la regressione di keypoint.
    """

    def __init__(self, num_keypoints, pretrained=True, freeze_backbone=True):
        """
        Args:
            num_keypoints (int): Numero di keypoint da predire (output = num_keypoints * 2)
            pretrained (bool): Se usare pesi pre-addestrati
            freeze_backbone (bool): Se congelare i pesi del backbone
        """
        super(ResNetKeypoint, self).__init__()

        self.num_keypoints = num_keypoints
        self.num_outputs = num_keypoints * 2  # x, y per ogni keypoint

        # Carica ResNet pre-addestrata
        self.backbone = models.resnet50(weights='IMAGENET1K_V2') if pretrained else models.resnet50(weights=None)


        # Ottieni il numero di features dell'ultimo layer
        num_features = self.backbone.fc.in_features

        # Sostituisci l'ultimo layer per la regressione
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # provo a mettere il dropout
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_outputs)
        )

        # Congelo i pesi del backbone se richiesto
        if freeze_backbone and pretrained:
            self._freeze_backbone()

        print(f"Modello inizializzato:")
        print(f"- Backbone: ResNet50 ({'pre-allenata' if pretrained else 'da zero'})")
        print(f"- Keypoints: {num_keypoints}")
        print(f"- Output features: {self.num_outputs}")
        print(f"- Backbone frozen: {freeze_backbone}")

    def _freeze_backbone(self):
        """Congela i pesi del backbone (tutti tranne l'ultimo layer)."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Non congelare l'ultimo layer
                param.requires_grad = False

        print("Backbone congelato (solo l'ultimo layer sarà addestrato)")

    def unfreeze_backbone(self):
        """Scongela tutti i pesi del modello."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone scongelato (tutto il modello sarà addestrato)")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [batch_size, 3, H, W]

        Returns:
            torch.Tensor: Output tensor [batch_size, num_keypoints * 2]
        """
        return self.backbone(x)

    def predict_keypoints(self, x):
        """
        Predice i keypoint e li restituisce in formato strutturato.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Keypoints [batch_size, num_keypoints, 2]
        """
        outputs = self.forward(x)
        # Reshape da [batch_size, num_keypoints * 2] a [batch_size, num_keypoints, 2]
        return outputs.view(-1, self.num_keypoints, 2)


class ResNetKeypointAdvanced(nn.Module):
    """
    Versione avanzata con architettura più complessa.
    """

    def __init__(self, num_keypoints, backbone='resnet50', pretrained=True,
                 freeze_backbone=True, dropout_rate=0.5):
        super(ResNetKeypointAdvanced, self).__init__()

        self.num_keypoints = num_keypoints
        self.num_outputs = num_keypoints * 2

        # Selezione del backbone
        backbone_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }

        if backbone not in backbone_dict:
            raise ValueError(f"Backbone non supportato: {backbone}")

        # Carica il modello
        if pretrained:
            weights = 'IMAGENET1K_V2' if backbone in ['resnet50', 'resnet101', 'resnet152'] else 'IMAGENET1K_V1'
            self.backbone = backbone_dict[backbone](weights=weights)
        else:
            self.backbone = backbone_dict[backbone](weights=None)

        # Numero di features
        num_ftrs = self.backbone.fc.in_features

        # Head più sofisticato per la regressione
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),

            nn.Linear(256, self.num_outputs)
        )

        # Congela il backbone se richiesto
        if freeze_backbone and pretrained:
            self._freeze_backbone()

        print(f"Modello avanzato inizializzato:")
        print(f"- Backbone: {backbone} ({'pre-trained' if pretrained else 'from scratch'})")
        print(f"- Keypoints: {num_keypoints}")
        print(f"- Dropout rate: {dropout_rate}")

    def _freeze_backbone(self):
        """Congela il backbone."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Scongela il backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def predict_keypoints(self, x):
        outputs = self.forward(x)
        return outputs.view(-1, self.num_keypoints, 2)


def create_model(num_keypoints, model_type='simple', **kwargs):
    """
    Factory function per creare il modello.

    Args:
        num_keypoints (int): Numero di keypoint
        model_type (str): 'simple' o 'advanced'
        **kwargs: Argomenti aggiuntivi per il modello

    Returns:
        nn.Module: Il modello creato
    """

    if model_type == 'simple':
        return ResNetKeypoint(num_keypoints, **kwargs)
    elif model_type == 'advanced':
        return ResNetKeypointAdvanced(num_keypoints, **kwargs)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")


def count_parameters(model):
    """Conta i parametri del modello."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Parametri totali: {total:,}")
    print(f"Parametri addestrabili: {trainable:,}")
    print(f"Parametri congelati: {total - trainable:,}")

    return trainable, total


# Test del modello
def test_model():
    """Test del modello con dati dummy."""