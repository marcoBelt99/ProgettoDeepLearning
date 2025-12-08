import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet

from configs.parametri_app import RAGGRUPPAMENTI


def build_resnet18(num_outputs,
                   pretrained=True,
                   head="linear",
                   mlp_hidden=(256, 128),
                   freeze_until=None):
    """
    Restituisce una ResNet18 customizzata.
    Args:
        num_outputs (int): Numero di outputs della rete
        pretrained (bool, optional): Bool di pretraine. Default: True
        mlp_hidden: numero di neuroni degli strati nascosti (dal più "profondo" al meno) che aggiungo
                    alla testa.

        head: come dev'essere fatta la head
            - "linear" = Linear(512, outputs)
            - "mlp"    = MLP 512 → mlp_hidden → outputs

        freeze_until: None = non congela nulla (train all)
                      "fc" = congela tutto tranne fc
                      "layer4" = congela layer1-3
                      "layer3" = congela layer1-2
                      "all" = congela tutto
    """

    backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    num_features = backbone.fc.in_features


    ## Scelta della tipologia di head
    if head == "linear":
        backbone.fc = nn.Linear(num_features, num_outputs)

    elif head == "mlp":
        h1, h2 = mlp_hidden
        backbone.fc = nn.Sequential(
            nn.Linear(num_features, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, num_outputs)
        )
    else:
        raise ValueError(f"Tipo di head non riconosciuto: {head}")

    ## Scelta di quali layers freezare
    if freeze_until == "all":
        for p in backbone.parameters():
            p.requires_grad = False

    elif freeze_until == "layer3":
        # train layer3, layer4 e fc
        for name, p in backbone.named_parameters():
            p.requires_grad = any(l in name for l in ["layer3", "layer4", "fc"])

    elif freeze_until == "layer4":
        # allena layer4 e fc
        for name, p in backbone.named_parameters():
            p.requires_grad = any(l in name for l in ["layer4", "fc"])

    elif freeze_until == "fc":
        # allena solo la testa
        for name, p in backbone.named_parameters():
            p.requires_grad = ("fc" in name)

    elif freeze_until in (None, "none"):
        # allena tutto, quindi non fa freeze
        for p in backbone.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Opzione freeze_until non valida: {freeze_until}")

    # Restituisco la backbone costruita
    return backbone



def build_model_for_group(nome_gruppo: str,
                          pretrained=True,
                          head="linear",
                          freeze_until="layer3"):
    """
    Costruisce automaticamente un modello ResNet18
    con il giusto numero di output per lo specifico gruppo.

    Questa funzione deve essere richiamata solo dai singoli modelli.
    """

    num_punti = len(RAGGRUPPAMENTI[nome_gruppo])
    num_outputs = num_punti * 2  # x e y per ciascun punto

    print(f"Creo ResNet18 per {nome_gruppo} → {num_punti} punti → {num_outputs} output")

    return build_resnet18(
        num_outputs=num_outputs,
        pretrained=pretrained,
        head=head,
        freeze_until=freeze_until
    )




def freeze_backbone_except(backbone : ResNet, trainable_levels=("fc",)):
    """
    Attiva requires_grad solo per alcuni layer.
    Ad esempio:
        - ("fc",) solo ultimo layer (fine tuning debole)
        - ("layer4","fc") ultimi 2 blocchi
        - ("layer3","layer4","fc") strategia migliore (fine tuning moderato)
    """
    for name, param in backbone.named_parameters():
        param.requires_grad = any(level in name for level in trainable_levels)