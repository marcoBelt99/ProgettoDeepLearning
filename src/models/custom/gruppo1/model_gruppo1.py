import torch
import torch.nn as nn
from torchvision import models

def _freeze_resnet(resnet: nn.Module, freeze_until: str):
    """
    freeze_until: 'all' | 'fc' | 'layer4' | 'layer3' | 'none'
    Coerente con quanto già fatto.
    """
    if freeze_until == "none":
        for p in resnet.parameters():
            p.requires_grad = True
        return

    # freeza tutto
    for p in resnet.parameters():
        p.requires_grad = False

    # unfreezza progressivamente
    if freeze_until == "fc":
        for p in resnet.fc.parameters():
            p.requires_grad = True
        return

    if freeze_until == "layer4":
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        for p in resnet.fc.parameters():
            p.requires_grad = True
        return

    if freeze_until == "layer3":
        for p in resnet.layer3.parameters():
            p.requires_grad = True
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        for p in resnet.fc.parameters():
            p.requires_grad = True
        return

    # 'all' per lasciare tutti freezati

class Gruppo1ResNet18(nn.Module):
    """ResNet18 pretrained, pensata per punti globali (S,N,A,B)."""
    def __init__(self, num_outputs: int, pretrained: bool = True, head: str = "linear", freeze_until: str = "layer4"):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if head == "linear":
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_outputs)
        elif head == "mlp":
            self.backbone.fc = nn.Sequential(
                nn.Linear(self.backbone.fc.in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_outputs)
            )
        else:
            raise ValueError(f"head non valido: {head}")

        _freeze_resnet(self.backbone, freeze_until)

    def forward(self, x):
        return self.backbone(x)
