"""Condicionamento geométrico da BokehNet, contra `metric_disparity_official_v1`.

Reimplementação da campanha de 09/2026 (`referencia_antiga/geo_cond_v1/`), que
ficou inválida porque os modelos contra os quais ela mediu foram treinados com o
mapa de defocus errado. Nada é importado de lá: física e derivações voltam
REESCRITAS, números não voltam.

O que mudou de contrato, em uma linha: o PNG do release é DISPARIDADE `uint16`
linear entre `disparity_min` e `disparity_max`, e esses dois números vêm no
`meta/` de CADA amostra. A tabela externa de escalares (`geo_escalares`, casada
por `stem`, alimentada pelos jobs f0b/f0e) não existe mais e não é mais
necessária.

O CONTRATO ESTÁ EM `contrato.py`, e é o único lugar onde ele está.

IMPORTAÇÃO PREGUIÇOSA, DE PROPÓSITO
-----------------------------------
Só `contrato` é importado na carga do pacote. Os módulos de cálculo
(`signals`, `constants`, `dataloader`, `loss_weight`, `ebleed`) puxam `numpy` —
e `loss_weight` puxa `torch` — então importá-los aqui tornaria
`from geocond.contrato import CANAIS` dependente de `torch`. Além disso os
módulos deste pacote são escritos em paralelo: um `__init__` que importa tudo
faz um módulo ausente derrubar os outros seis.

Quem precisa de cálculo importa o submódulo (`from geocond.dataloader import
stack_para_amostra`), ou usa os atributos preguiçosos abaixo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contrato import (
    CANAIS,
    CONTRATO_DE_CONTROLE,
    G1_IDX,
    G2_IDX,
    AmostraGeo,
    PilhaGeo,
)

if TYPE_CHECKING:  # pragma: no cover - só para o type checker
    from .dataloader import espelhar_pilha, stack_para_amostra

__all__ = [
    "CANAIS",
    "CONTRATO_DE_CONTROLE",
    "G1_IDX",
    "G2_IDX",
    "AmostraGeo",
    "PilhaGeo",
    "espelhar_pilha",
    "stack_para_amostra",
]

#: Atributo preguiçoso -> submódulo que o define. Ver a nota do módulo.
_PREGUICOSOS = {
    "espelhar_pilha": "dataloader",
    "stack_para_amostra": "dataloader",
}


def __getattr__(nome: str):
    submodulo = _PREGUICOSOS.get(nome)
    if submodulo is None:
        raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")
    from importlib import import_module

    return getattr(import_module(f".{submodulo}", __name__), nome)


def __dir__() -> list[str]:
    return sorted(__all__)
