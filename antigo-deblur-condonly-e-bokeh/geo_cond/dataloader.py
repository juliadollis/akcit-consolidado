"""Ponte entre o dataloader do treino e os sinais geometricos.

Fica aqui, e nao em `genfocus_train/data.py`, para ser testavel sem rede e sem
GPU. O gancho em `data.py` chama `stack_para_amostra` e nada mais.

A PARTE DELICADA E A GEOMETRIA, NAO O SINAL
-------------------------------------------
`prepare_aligned_bokeh` faz tres transformacoes, nesta ordem, e cada uma mexe nos
intrinsecos que a curvatura retroprojetada consome:

  1. RESIZE do lado menor para `image_size`.  fx' = fx * escala
  2. CROP de S x S na posicao (x, y).         cx' = cx*escala - x
  3. HFLIP com probabilidade 1/2.             cx'' = S-1-cx'  E  n_x troca de SINAL

O item 3 e o que passa despercebido: as normais sao um CAMPO VETORIAL, entao
espelhar a imagem nao e so espelhar o array. A componente x aponta para o outro
lado. Como o canal e gravado como (n_x+1)/2, espelhar corretamente e

    canal_espelhado = 1 - flip_horizontal(canal)

e nao apenas `flip_horizontal(canal)`. Errar isso ensina a rede a associar
inclinacao para a direita com inclinacao para a esquerda em metade das amostras,
o que nao levanta excecao nenhuma.

ESTRATEGIA: CALCULAR ANTES DO CROP, RECORTAR DEPOIS
--------------------------------------------------
Calculamos a pilha na imagem REDIMENSIONADA e INTEIRA, e so entao recortamos.
Duas razoes:

  * evita artefato de borda: as derivadas usam padding replicado, entao calcular
    no recorte poria uma borda falsa em cada lado do crop, em toda amostra.
  * elimina a contabilidade de `cx, cy` do crop: so a escala do resize entra.

O custo e calcular em ~1024x683 em vez de 512x512, o que e desprezivel perto do
VAE. O teste `test_recorte_comuta_com_o_calculo_no_interior` confirma que as duas
ordens batem no interior, entao a escolha e por robustez de borda, nao por
resultado diferente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GeoConstants
from .signals import CANAIS, geometric_stack

__all__ = ["GeoAmostra", "IDX_NX", "stack_para_amostra", "espelhar_stack"]

#: Indice do canal n_x em `CANAIS`. E o unico que troca de sinal sob espelhamento.
IDX_NX = CANAIS.index("n_x")


@dataclass(frozen=True)
class GeoAmostra:
    """Escalares por amostra que a pilha geometrica exige.

    Vem da tabela kfix (rota b) ou do job F0b (rota c). Casados por `stem`.
    """

    z_min_m: float
    z_max_m: float
    focallength_px: float
    """Focal em pixels NA RESOLUCAO ORIGINAL da imagem armazenada."""
    largura_px: int
    altura_px: int
    """Dimensoes originais, para deduzir a escala do resize."""


def espelhar_stack(canais: np.ndarray) -> np.ndarray:
    """Espelha a pilha horizontalmente, tratando n_x como campo vetorial.

    Todos os canais sao espelhados no array; alem disso `n_x` tem o sinal
    invertido, o que na codificacao [0,1] significa `1 - valor`.
    """
    out = canais[:, :, ::-1].copy()
    out[IDX_NX] = 1.0 - out[IDX_NX]
    return out


def stack_para_amostra(
    depth01_redimensionado: np.ndarray,
    caixa: tuple[int, int, int, int],
    flip: bool,
    amostra: GeoAmostra,
    consts: GeoConstants,
    *,
    field: str = "inverse",
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, int]:
    """Pilha (6, S, S) em [0,1] para uma amostra ja redimensionada.

    Parameters
    ----------
    depth01_redimensionado
        (H', W') em [0,1], DEPOIS do resize e ANTES do crop.
    caixa
        (x0, y0, x1, y1), a mesma que `prepare_aligned_bokeh` usou.
    flip
        Se o hflip foi aplicado. Ver a nota do modulo.
    amostra
        Escalares casados por `stem`.

    Returns
    -------
    (canais (6,S,S) float32, segunda_ordem_valida, niveis_quantizacao)
    """
    H, W = depth01_redimensionado.shape
    if amostra.largura_px <= 0 or amostra.altura_px <= 0:
        raise ValueError("dimensoes originais invalidas em GeoAmostra")

    # Escala do resize. `prepare_aligned_bokeh` usa o LADO MENOR, entao a escala
    # e a mesma nos dois eixos e nao ha anisotropia introduzida.
    escala = min(H / amostra.altura_px, W / amostra.largura_px)
    fx = amostra.focallength_px * escala
    if fx <= 0:
        raise ValueError(f"focal invalida apos resize: {fx}")

    # Centro optico: sem coluna de principal point nos dfs, assumimos o centro
    # geometrico, que e a convencao do proprio Depth Pro.
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    saida = geometric_stack(
        depth01_redimensionado,
        z_min_m=amostra.z_min_m, z_max_m=amostra.z_max_m,
        fx_px=fx, fy_px=fx, cx=cx, cy=cy,
        px_per_unit=float(min(H, W)),
        consts=consts, field=field, valid=valid,
    )

    x0, y0, x1, y1 = caixa
    recorte = saida.canais[:, y0:y1, x0:x1]
    if flip:
        recorte = espelhar_stack(recorte)
    return (np.ascontiguousarray(recorte),
            saida.segunda_ordem_valida, saida.niveis_quantizacao)
