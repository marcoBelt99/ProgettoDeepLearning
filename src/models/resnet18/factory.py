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
        pretrained (bool, optional): Bool di pretrained. Default: True
        mlp_hidden: numero di neuroni degli strati nascosti (dal più "profondo" al meno) che aggiungo
                    alla testa.

        head: come dev'essere fatta la head
            - "linear" = Linear(512, num_outputs_che_mi_servono)
            - "mlp"    = MLP 512 => mlp_hidden => num_outputs_che_mi_servono

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
        # allena layer3, layer4 e fc
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


    # Aggiungo dei metadati al modello, che mi serviranno per la funzione log_experiment()
    backbone.freeze_until = freeze_until
    backbone.head_type = head
    backbone.pretrained_flag = bool(pretrained)
    backbone.num_outputs = int(num_outputs)

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

    print(f"Creo ResNet18 per {nome_gruppo} che ha {num_punti} punti ==> pertanto la rete ha {num_outputs} uscite.")

    model =  build_resnet18(
        num_outputs=num_outputs,
        pretrained=pretrained,
        head=head,
        freeze_until=freeze_until
    )

    # Aggiungo metadati necessari al logging
    model.group_name = nome_gruppo
    model.num_points = int(num_punti)

    return model




