"""Métricas de imagem usadas na calibração e nos gates.

SSIM aqui é a da Eq. 5 do paper: `K* = argmax_K SSIM(R(I_aif, D; D_focus, K), I_real)`.
Implementação própria em numpy para não depender de `scikit-image` no cluster, com as
constantes padrão de Wang et al. 2004 e janela gaussiana 11×11 σ=1,5.
"""

from __future__ import annotations

import numpy as np

#: Constantes padrão de SSIM para dados em [0, 255].
_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2


def _gaussian_kernel_1d(sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolução separável com borda replicada, sem depender de cv2.

    Duas passadas explícitas: horizontal `(H+2r, W+2r) -> (H+2r, W)`, depois vertical
    `(H+2r, W) -> (H, W)`. Os shapes intermediários são escritos à mão de propósito —
    a primeira versão deste código somava um array `(H+2r, W)` num acumulador
    `(H+2r, W+2r)` e quebrava por broadcast.
    """
    height, width = image.shape
    r = len(kernel) // 2
    padded = np.pad(np.asarray(image, dtype=np.float64), ((r, r), (r, r)), mode="edge")

    horizontal = np.zeros((height + 2 * r, width), dtype=np.float64)
    for i, weight in enumerate(kernel):
        horizontal += weight * padded[:, i:i + width]

    out = np.zeros((height, width), dtype=np.float64)
    for i, weight in enumerate(kernel):
        out += weight * horizontal[i:i + height, :]
    return out


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR/RGB uint8 -> cinza float64 em [0, 255]. Luma ITU-R BT.601."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"imagem com shape inesperado: {arr.shape}")
    b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """SSIM global entre duas imagens do mesmo shape, em [0, 255].

    É a função objetivo da Eq. 5. Não recebe máscara de propósito: o paper compara a
    imagem inteira, e restringir a máscara mudaria o que está sendo otimizado.
    """
    a, b = to_gray(image_a), to_gray(image_b)
    if a.shape != b.shape:
        raise ValueError(f"shapes diferentes: {a.shape} vs {b.shape}")
    k = _gaussian_kernel_1d()
    mu_a, mu_b = _blur(a, k), _blur(b, k)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_aa = _blur(a * a, k) - mu_aa
    sigma_bb = _blur(b * b, k) - mu_bb
    sigma_ab = _blur(a * b, k) - mu_ab
    num = (2 * mu_ab + _C1) * (2 * sigma_ab + _C2)
    den = (mu_aa + mu_bb + _C1) * (sigma_aa + sigma_bb + _C2)
    return float(np.mean(num / den))


def laplacian_variance(image: np.ndarray) -> float:
    """Medida de nitidez do paper (supp. B.1 e B.2): variância do Laplaciano.

    Usada para ranquear o pool de AIF da rota A e como gate de nitidez. Comparável
    apenas DENTRO de uma mesma fonte: a variância escala com resolução e com
    compressão, então ranking global entre datasets diferentes enviesa a seleção.
    """
    gray = to_gray(image)
    lap = (
        -4.0 * gray
        + np.pad(gray, ((1, 0), (0, 0)), mode="edge")[:-1, :]
        + np.pad(gray, ((0, 1), (0, 0)), mode="edge")[1:, :]
        + np.pad(gray, ((0, 0), (1, 0)), mode="edge")[:, :-1]
        + np.pad(gray, ((0, 0), (0, 1)), mode="edge")[:, 1:]
    )
    return float(lap.var())
