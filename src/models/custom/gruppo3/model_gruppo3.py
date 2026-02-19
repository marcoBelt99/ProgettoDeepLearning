import torch
import torch.nn as nn
from torchvision import models

class Gruppo3MobileNetV3(nn.Module):
    """MobileNetV3-Large pretrained: veloce su CPU, adatta a dettagli (denti)."""
    def __init__(self, num_outputs: int, pretrained: bool = True, head: str = "mlp"):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        # classifier originale: [Linear(960->1280), Hardswish, Dropout, Linear(1280->1000)]
        # teniamo i primi 3 layer pretrained (fino al Dropout)
        prefix = list(self.backbone.classifier[:-1])   # Linear(960->1280) + act + dropout
        in_f = self.backbone.classifier[-1].in_features  # 1280

        if head == "linear":
            # uso il prefix pretrained e sostituisco l'ultimo layer
            self.backbone.classifier = nn.Sequential(
                *prefix,
                nn.Linear(in_f, num_outputs)
            )

        elif head == "mlp":
            self.backbone.classifier = nn.Sequential(
                *prefix,
                nn.Linear(in_f, 512),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, num_outputs)
            )
        else:
            raise ValueError(f"head non valido: {head}")

        # (opzionale ma consigliato) freeze feature extractor se pretrained
        if pretrained:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

        # ...ma sblocco gli ultimi K blocchi (ad es. 4)
        for p in self.backbone.features[-4:].parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
