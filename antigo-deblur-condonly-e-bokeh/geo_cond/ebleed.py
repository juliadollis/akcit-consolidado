"""E_bleed -- erro concentrado na regiao de borda de profundidade.

Do PLANO secao 6, e a metrica que o documento de proposta coloca como
PRE-CONDICAO da campanha: as metricas usuais diluem o artefato, porque a regiao
de borda e uma fracao pequena dos pixels e uma melhora real aparece na terceira
casa decimal de qualquer metrica global.

    E_bleed = (1/|B_t|) * soma_{x in B_t} || I_chapeu(x) - I(x) ||_1
    B_t     = { x : O(x) > theta }

`O` e o MESMO mapa de oclusao do condicionamento (geo_cond.signals.occlusion),
com o MESMO tau fixo. Isso nao e detalhe: se a metrica usasse outro limiar, ela
mediria uma regiao diferente da que a perda ponderada supervisiona, e a
comparacao entre as condicoes deixaria de ser sobre a mesma coisa.

REPORTA TAMBEM A REGIAO COMPLEMENTAR
------------------------------------
Sem isso, um modelo que borra tudo poderia baixar o E_bleed por acidente. O par
(dentro, fora) mostra se a melhora na borda veio as custas do resto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GeoConstants
from .signals import depth01_to_metric, grad_normalized, occlusion

__all__ = ["ResultadoEbleed", "e_bleed"]


@dataclass(frozen=True)
class ResultadoEbleed:
    e_bleed: float
    """L1 medio DENTRO da regiao de borda. Menor e melhor."""
    e_fora: float
    """L1 medio FORA dela. O par mostra se a melhora custou o resto."""
    fracao_borda: float
    """Fracao de pixels em B_theta. Se for ~0 ou ~1, o limiar nao serve."""
    n_pixels_borda: int


def e_bleed(
    predicao: np.ndarray,
    alvo: np.ndarray,
    depth01: np.ndarray,
    *,
    z_min_m: float,
    z_max_m: float,
    consts: GeoConstants,
    theta: float = 0.3,
    field: str = "inverse",
) -> ResultadoEbleed:
    """
    Parameters
    ----------
    predicao, alvo
        (H, W, 3) em [0,1].
    depth01
        (H, W) em [0,1], na MESMA resolucao das imagens.
    z_min_m, z_max_m
        Faixa metrica da amostra. `z_max_m` e o valor BRUTO (ver
        `signals.depth01_to_metric`).
    theta
        Limiar sobre `O`. B_theta = {x : O(x) > theta}.
    """
    if predicao.shape != alvo.shape:
        raise ValueError(f"shapes diferentes: {predicao.shape} vs {alvo.shape}")
    if depth01.shape != predicao.shape[:2]:
        raise ValueError(
            f"depth {depth01.shape} nao casa com a imagem {predicao.shape[:2]}; "
            "reamostre ANTES, para que a borda caia no mesmo lugar"
        )
    z = depth01_to_metric(depth01, z_min_m, z_max_m)
    f = (1.0 / z) if field == "inverse" else z
    fx, fy = grad_normalized(f, float(min(depth01.shape)))
    O = occlusion(np.hypot(fx, fy), consts.tau_occlusion)

    l1 = np.abs(predicao.astype(np.float64) - alvo.astype(np.float64)).mean(axis=2)
    borda = O > theta
    n = int(borda.sum())
    dentro = float(l1[borda].mean()) if n else float("nan")
    fora = float(l1[~borda].mean()) if n < borda.size else float("nan")
    return ResultadoEbleed(
        e_bleed=dentro, e_fora=fora,
        fracao_borda=float(n / borda.size), n_pixels_borda=n,
    )
