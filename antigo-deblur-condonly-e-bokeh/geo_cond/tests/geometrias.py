"""Cenas sintéticas de geometria CONHECIDA, para os testes.

Todas geradas analiticamente, sem interpolação, para que a comparação entre
resoluções seja exata e não misture erro de reamostragem com erro de fórmula.
"""

from __future__ import annotations

import numpy as np


def grade_raios(n: int, fx: float):
    """Direções de raio (a, b) para uma imagem n x n com focal fx.

    a = (u - cx)/fx, b = (v - cy)/fy, com cx = cy = (n-1)/2 e fy = fx.
    """
    c = (n - 1) / 2.0
    v, u = np.mgrid[0:n, 0:n]
    return (u - c) / fx, (v - c) / fx, c


def esfera(n: int, fx: float, raio_m: float, dist_m: float) -> np.ndarray:
    """Profundidade Z de uma esfera de raio R centrada em (0, 0, d).

    A curvatura gaussiana da esfera é K = 1/R^2 em todo ponto. É o teste que o
    `RETESTE_CURVATURA.md` usou para mostrar que a formulação de Monge devolve
    ~10 onde o valor verdadeiro é 0,25.
    """
    a, b, _ = grade_raios(n, fx)
    q = a * a + b * b + 1.0
    disc = dist_m ** 2 - q * (dist_m ** 2 - raio_m ** 2)
    if np.any(disc <= 0):
        raise ValueError("a esfera não cobre a imagem inteira; use fx maior ou R maior")
    return (dist_m - np.sqrt(disc)) / q          # raiz próxima


def plano_inclinado(n: int, fx: float, normal, dist_m: float) -> np.ndarray:
    """Profundidade de um plano n . X = c. Curvatura gaussiana exatamente 0."""
    a, b, _ = grade_raios(n, fx)
    n1, n2, n3 = normal
    den = n1 * a + n2 * b + n3
    if np.any(np.abs(den) < 1e-6):
        raise ValueError("plano paralelo a algum raio")
    return dist_m / den


def cilindro(n: int, fx: float, raio_m: float, dist_m: float) -> np.ndarray:
    """Cilindro de eixo vertical (Y). Superfície desenvolvível: K = 0."""
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
    """Rampa linear em Z ao longo de x. Gradiente constante."""
    return np.tile(np.linspace(z0, z1, n), (n, 1))


def degrau(n: int, z_perto: float, z_longe: float) -> np.ndarray:
    """Degrau de profundidade no meio: a descontinuidade que O deve marcar."""
    z = np.full((n, n), float(z_longe))
    z[:, : n // 2] = float(z_perto)
    return z
