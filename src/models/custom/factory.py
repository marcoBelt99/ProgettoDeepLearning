from typing import Tuple
import torch.nn as nn

from configs.parametri_app import RAGGRUPPAMENTI

from gruppo1.model_gruppo1 import Gruppo1ResNet18
from gruppo2.model_gruppo2 import Gruppo2ResNet34
from gruppo3.model_gruppo3 import Gruppo3MobileNetV3
from gruppo4.model_gruppo4 import Groppo4SmallCNN

def build_model_for_group(
    nome_gruppo: str,
    model_name: str,
    pretrained: bool = True,
    head: str = "linear",
    freeze_until: str = "layer3",
) -> nn.Module:
    """
    Costruisce il modello specifico per gruppo.
    num_outputs = (num_punti_gruppo * 2)
    """
    if nome_gruppo not in RAGGRUPPAMENTI:
        raise ValueError(f"Gruppo non valido: {nome_gruppo}. Disponibili: {list(RAGGRUPPAMENTI.keys())}")

    num_points = len(RAGGRUPPAMENTI[nome_gruppo])
    num_outputs = num_points * 2

    if model_name == "resnet18_g1":
        return Gruppo1ResNet18(num_outputs=num_outputs, pretrained=pretrained, head=head, freeze_until=freeze_until)

    if model_name == "resnet34_g2":
        return Gruppo2ResNet34(num_outputs=num_outputs, pretrained=pretrained, head=head, freeze_until=freeze_until)

    if model_name == "mobilenetv3_g3":
        return Gruppo3MobileNetV3(num_outputs=num_outputs, pretrained=pretrained, head=head)

    if model_name == "smallcnn_g4":
        return Groppo4SmallCNN(num_outputs=num_outputs)

    raise ValueError(f"model_name non valido: {model_name}")
