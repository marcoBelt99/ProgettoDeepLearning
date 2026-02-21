"""

Contiene un gruppo di diversi tentativi di specializzazione per ciascuno dei 4 modelli
(ognuno interessa un diverso gruppo di punti).

In models/resnet18/* ho fatto dei tentativi "generici" usando la singola architettura ResNet18, con 4 modelli
che di fatto condividono tutti gli stessi iperparametri e le stesse trasformazioni di data agumentation.
Tale package mi è servito come "baseline".

Nel presente package models/custom/* provo a specializzare e dunque diversificare le architetture di ciascuno dei 4 modelli,
provando varie configurazioni che credo possano essere adeguate al gruppo di punti (ad esempio, trasformazioni più sensate
per un determinato gruppo e altre trasformazioni diverse per un altro gruppo).
Oppure una diversa backbone per un gruppo rispetto agli altri.

"""