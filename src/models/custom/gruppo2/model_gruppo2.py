import torch
import torch.nn as nn
from torchvision import models

def _freeze_resnet(resnet: nn.Module, freeze_until: str):
    if freeze_until == "none":
        for p in resnet.parameters():
            p.requires_grad = True
        return

    for p in resnet.parameters():
        p.requires_grad = False

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

class Gruppo2ResNet34(nn.Module):
    """ResNet34 pretrained, più capiente per mandibola/naso (Sna,Snp,Gn,Go)."""
    def __init__(self, num_outputs: int, pretrained: bool = True, head: str = "mlp", freeze_until: str = "layer3"):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        in_f = self.backbone.fc.in_features
        if head == "linear":
            self.backbone.fc = nn.Linear(in_f, num_outputs)
        elif head == "mlp":
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_f, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, num_outputs),
            )
        else:
            raise ValueError(f"head non valido: {head}")

        _freeze_resnet(self.backbone, freeze_until)

    def forward(self, x):
        return self.backbone(x)
