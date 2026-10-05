"""Modelos carregados uma vez por run. Um backend por função, sem cascata.

`depth`, `segmentation` e `deblurnet` importam torch dentro dos métodos, então este
pacote é importável sem GPU — o que permite testar assinatura, geometria e proveniência
localmente.

A DeblurNet tem um eixo a mais que os outros dois: a **variante** do checkpoint. Não é
um segundo backend (não há cascata), é a convenção de inferência que o peso exige, e ela
vem amarrada ao peso em `DeblurVariant`. Ver o docstring de `deblurnet`.
"""
from .deblurnet import (
    BACKEND_NAME as DEBLUR_BACKEND,
    DeblurNetRuntime,
    DeblurredAIF,
    DeblurVariant,
    DeblurVariantMismatch,
    ProcessingPlan,
    ResizePolicy,
    plan_processing,
    resolve_weights,
)
from .depth import BACKEND_NAME as DEPTH_BACKEND, DepthProRuntime
from .segmentation import BACKEND_NAME as MASK_BACKEND, BiRefNetRuntime

__all__ = [
    "DEPTH_BACKEND", "MASK_BACKEND", "DEBLUR_BACKEND",
    "DepthProRuntime", "BiRefNetRuntime", "DeblurNetRuntime",
    "DeblurredAIF", "DeblurVariant", "DeblurVariantMismatch",
    "ProcessingPlan", "ResizePolicy", "plan_processing", "resolve_weights",
]
