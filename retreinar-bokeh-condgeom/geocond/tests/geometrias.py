"""Cenas sintéticas de geometria CONHECIDA, e o codificador do release.

Duas metades:

1. Geradores de profundidade Z (m) analíticos — esfera, plano, cilindro,
   frontoparalelo, rampa, degrau. Sem interpolação em lugar nenhum, para que a
   comparação entre resoluções seja exata e não misture erro de reamostragem com
   erro de fórmula. É o que permite exigir 2% de erro na curvatura da esfera.

2. `codifica_release`, que leva Z a `(disp01, AmostraGeo)` passando pela MESMA
   codificação do release `metric_disparity_official_v1`: disparidade métrica,
   min-max sobre a faixa da cena, quantização uint16. Os testes entram pelo
   mesmo buraco que o dataloader, e não por uma porta de serviço.
"""

from __future__ import annotations

import numpy as np

from geocond.contrato import AmostraGeo
from geocond.signals import NIVEIS_U16


# =============================================================================
# Profundidade Z analítica
# =============================================================================

def grade_raios(n: int, fx: float):
    """Direções de raio (a, b) para uma imagem n x n com focal fx.

    a = (u − cx)/fx, b = (v − cy)/fy, com cx = cy = (n−1)/2 e fy = fx. É o MESMO
    centro que `pilha_geometrica` assume, então cena e cálculo concordam sobre
    onde fica o eixo óptico.
    """
    c = (n - 1) / 2.0
    v, u = np.mgrid[0:n, 0:n]
    return (u - c) / fx, (v - c) / fx, c


def esfera(n: int, fx: float, raio_m: float, dist_m: float) -> np.ndarray:
    """Profundidade Z de uma esfera de raio R centrada em (0, 0, d).

    K = 1/R² em todo ponto. É o teste que o `RETESTE_CURVATURA.md` usou para
    mostrar que a formulação de Monge devolve ~10 onde o valor verdadeiro é 0,25.
    """
    a, b, _ = grade_raios(n, fx)
    q = a * a + b * b + 1.0
    disc = dist_m ** 2 - q * (dist_m ** 2 - raio_m ** 2)
    if np.any(disc <= 0):
        raise ValueError(
            "a esfera não cobre a imagem inteira; use fx maior, R maior ou d menor"
        )
    return (dist_m - np.sqrt(disc)) / q          # raiz próxima


def plano_inclinado(n: int, fx: float, normal, dist_m: float) -> np.ndarray:
    """Profundidade de um plano n·X = c. Curvatura gaussiana exatamente 0."""
    a, b, _ = grade_raios(n, fx)
    n1, n2, n3 = normal
    den = n1 * a + n2 * b + n3
    if np.any(np.abs(den) < 1e-6):
        raise ValueError("plano paralelo a algum raio")
    return dist_m / den


def cilindro(n: int, fx: float, raio_m: float, dist_m: float) -> np.ndarray:
    """Cilindro de eixo vertical (Y). Superfície desenvolvível: K = 0 apesar de
    curvar numa direção. É o teste que separa curvatura GAUSSIANA de média."""
    a, b, _ = grade_raios(n, fx)
    q = a * a + 1.0
    disc = dist_m ** 2 - q * (dist_m ** 2 - raio_m ** 2)
    if np.any(disc <= 0):
        raise ValueError("o cilindro não cobre a imagem inteira")
    return (dist_m - np.sqrt(disc)) / q


def frontoparalelo(n: int, z_m: float) -> np.ndarray:
    """Plano frontoparalelo: gradiente nulo, K = 0."""
    return np.full((n, n), float(z_m))


def rampa(n: int, z0: float, z1: float) -> np.ndarray:
    """Rampa linear em Z ao longo de x."""
    return np.tile(np.linspace(z0, z1, n), (n, 1))


def degrau(n: int, z_perto: float, z_longe: float) -> np.ndarray:
    """Degrau de profundidade no meio: a descontinuidade que O deve marcar."""
    z = np.full((n, n), float(z_longe))
    z[:, : n // 2] = float(z_perto)
    return z


# =============================================================================
# Codificação do release
# =============================================================================

def codifica_release(
    z: np.ndarray,
    *,
    focallength_px: float | None = None,
    quantizar: bool = True,
    faixa: tuple[float, float] | None = None,
) -> tuple[np.ndarray, AmostraGeo]:
    """Z (m) -> (`disp01`, `AmostraGeo`), pela convenção do release.

        disp   = 1/Z                                   (1/m)
        d01    = (disp − disp_min) / (disp_max − disp_min)
        PNG    = round(d01 · 65535)                    (uint16_linear_in_disparity)

    `quantizar=True` (default) passa pelo uint16, porque é o que o treino vê. Os
    testes de FÓRMULA (curvatura da esfera a 2%) usam `quantizar=False`: com a
    esfera de R=2 m a 10 m a faixa de disparidade da cena é de apenas 0,009 1/m,
    o passo do uint16 vira 1,4e-7 1/m, e o ruído que isso injeta na derivada
    SEGUNDA é da ordem de 4% do sinal — ou seja o uint16 não é o objeto sob teste
    ali, e misturar os dois erros num limiar só esconderia qual deles falhou.

    `faixa` força `disparity_min`/`max` em vez de tirá-los da cena. Necessário
    para o frontoparalelo, cuja disparidade é constante e cujo min-max seria
    degenerado — e realista, porque no release os extremos são da CENA, que
    quase nunca é uma parede lisa e nada mais.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z deve ser (H, W); recebido {z.shape}")
    if np.any(z <= 0):
        raise ValueError("profundidade não positiva")

    disp = 1.0 / z
    if faixa is None:
        d_min, d_max = float(disp.min()), float(disp.max())
        if d_max - d_min < 1e-12:
            raise ValueError(
                "cena de disparidade constante: passe `faixa=(min, max)` "
                "explicitamente (ver docstring)"
            )
    else:
        d_min, d_max = float(faixa[0]), float(faixa[1])

    d01 = (disp - d_min) / (d_max - d_min)
    if quantizar:
        d01 = np.rint(np.clip(d01, 0.0, 1.0) * NIVEIS_U16) / NIVEIS_U16

    H, W = z.shape
    amostra = AmostraGeo(
        disparity_min=d_min,
        disparity_max=d_max,
        largura_px=W,
        altura_px=H,
        focallength_px=focallength_px,
    )
    return d01, amostra
