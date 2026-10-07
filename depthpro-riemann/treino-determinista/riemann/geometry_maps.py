"""
riemann/geometry_maps.py
========================
Extrai os sinais de geometria diferencial de um mapa de profundidade, para
visualização e para decidir qual sinal alimentar o DeblurNet / BokehNet.

Tratamos a profundidade como uma superfície de Monge z = f(x, y) em R^3. Disso saem:

  PRIMEIRA ORDEM (orientação e estiramento)
    grad          |∇z|, magnitude do gradiente. Marca qualquer variação de profundidade.
    normals       vetor normal unitário por pixel. Diz para onde a superfície aponta.
    det_g         determinante do tensor métrico g = I + ∇z∇zᵀ, ou seja 1 + |∇z|².
                  Mede o quanto a superfície está esticada em relação ao plano da imagem.
                  É literalmente a primeira forma fundamental, o "tensor métrico".

  SEGUNDA ORDEM (curvatura)
    K             curvatura Gaussiana, k1·k2. INTRÍNSECA: invariante por isometria.
                  Distingue região elíptica (K>0, domo/vale) de hiperbólica (K<0, sela).
    H             curvatura média, (k1+k2)/2. EXTRÍNSECA. É o análogo contínuo do
                  Laplaciano usado pelo DepthPro (MALE), e por isso serve de contraste
                  com K na hora de justificar a novidade do trabalho.
    k1, k2        curvaturas principais.
    curvedness    sqrt((k1²+k2²)/2). Intensidade da curvatura, independente do tipo.
    shape_index   (2/π)·arctan((k2+k1)/(k2−k1)). Classifica a FORMA local num único
                  número em [-1,1]: -1 taça, -0.5 sulco, 0 sela, +0.5 crista, +1 domo.
                  É a visualização mais interpretável para julgar geometria de objeto.

  DERIVADO (bordas e oclusão)
    occlusion     mapa de descontinuidade: normaliza |∇z| de forma robusta e realça
                  onde a superfície "quebra". É o sinal mais direto para o bokeh, porque
                  é exatamente onde o kernel de desfoque não pode atravessar.

Todas as funções recebem depth (B,1,H,W) e devolvem (B,1,H,W), exceto `normals`, que
devolve (B,3,H,W).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .losses import _grad_xy, _second_derivs, _smooth, _surface_normals


def gradient_magnitude(depth: torch.Tensor, h: Optional[float] = None) -> torch.Tensor:
    """|∇z|. Primeira ordem: marca toda variação de profundidade."""
    zx, zy = _grad_xy(depth, h)
    return torch.sqrt(zx ** 2 + zy ** 2 + 1e-12)


def metric_determinant(depth: torch.Tensor, h: Optional[float] = None) -> torch.Tensor:
    """
    det(g) = 1 + |∇z|², com g = I + ∇z∇zᵀ a primeira forma fundamental.

    Interpretação: a raiz de det(g) é o fator de estiramento de área entre o plano da
    imagem e a superfície. Vale 1 em regiões frontoparalelas e cresce em superfícies
    inclinadas ou em quebras de profundidade.
    """
    zx, zy = _grad_xy(depth, h)
    return 1.0 + zx ** 2 + zy ** 2


def area_element(depth: torch.Tensor, h: Optional[float] = None) -> torch.Tensor:
    """
    Elemento de área riemanniano:  sqrt(det(g)) = sqrt(1 + |∇z|²).

    Com g_ij = δ_ij + ∂_i z · ∂_j z (primeira forma fundamental de uma superfície de
    Monge), sqrt(det(g)) é o fator que converte área no plano da imagem em área sobre a
    superfície em R³. Vale exatamente 1 em regiões frontoparalelas e cresce onde a
    superfície se inclina ou quebra.

    Por que este mapa é tão mais limpo que a curvatura: ele é de PRIMEIRA ordem. A
    curvatura Gaussiana usa segundas derivadas, que amplificam ruído por 1/h², enquanto
    aqui o ruído entra apenas linearmente. Por isso o elemento de área produz uma
    visualização estável mesmo sem suavização pesada, e serve bem como mapa de "bordas
    3D": regiões brilhantes são exatamente onde a superfície deixa de ser paralela ao
    plano da imagem.
    """
    return torch.sqrt(metric_determinant(depth, h) + 1e-12)


def principal_curvatures(depth: torch.Tensor, smooth_sigma: float = 0.0,
                         h: Optional[float] = None, clamp_val: float = 50.0):
    """
    Curvaturas principais (k1, k2) da superfície de Monge, com k1 <= k2.

    A partir delas obtemos K = k1·k2 e H = (k1+k2)/2, o que garante consistência entre
    todos os mapas de segunda ordem (todos derivam das mesmas derivadas).
    """
    z = _smooth(depth, smooth_sigma) if smooth_sigma > 0 else depth
    zx, zy = _grad_xy(z, h)
    zxx, zyy, zxy = _second_derivs(z, h)

    p2 = zx ** 2 + zy ** 2
    denom = torch.sqrt(1.0 + p2 + 1e-12)

    # K = (zxx*zyy - zxy^2) / (1 + |∇z|^2)^2
    K = (zxx * zyy - zxy ** 2) / ((1.0 + p2) ** 2 + 1e-12)
    # H = ((1+zy^2)zxx - 2 zx zy zxy + (1+zx^2)zyy) / (2 (1+|∇z|^2)^{3/2})
    H = (((1.0 + zy ** 2) * zxx - 2.0 * zx * zy * zxy + (1.0 + zx ** 2) * zyy)
         / (2.0 * denom ** 3 + 1e-12))

    disc = torch.clamp(H ** 2 - K, min=0.0).sqrt()
    k1 = H - disc
    k2 = H + disc
    if clamp_val and clamp_val > 0:
        K = K.clamp(-clamp_val, clamp_val)
        H = H.clamp(-clamp_val, clamp_val)
        k1 = k1.clamp(-clamp_val, clamp_val)
        k2 = k2.clamp(-clamp_val, clamp_val)
    return k1, k2, K, H


def shape_index(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """
    Índice de forma em [-1, 1] (Koenderink & van Doorn).

    -1.0 taça (côncavo)   -0.5 sulco   0.0 sela   +0.5 crista   +1.0 domo (convexo)

    CONVENÇÃO: aqui z é PROFUNDIDADE, então valores maiores estão mais LONGE da câmera.
    Um objeto que se projeta na direção do observador tem profundidade menor no centro e
    aparece com índice positivo; uma reentrância aparece com índice negativo. Se a
    convenção do consumidor do sinal for altura em vez de profundidade, basta inverter.

    É invariante à intensidade da curvatura, então separa TIPO de forma de MAGNITUDE.
    Útil para decidir se o sinal que importa ao bokeh é o tipo de superfície ou só a
    intensidade da quebra.
    """
    dif = k2 - k1
    s = (2.0 / torch.pi) * torch.atan((k2 + k1) / (dif + 1e-8))
    return torch.where(dif.abs() < 1e-8, torch.zeros_like(s), s).clamp(-1.0, 1.0)


def curvedness(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """Intensidade da curvatura, sqrt((k1²+k2²)/2). Complementa o índice de forma."""
    return torch.sqrt((k1 ** 2 + k2 ** 2) / 2.0 + 1e-12)


def occlusion_map(depth: torch.Tensor, h: Optional[float] = None,
                  percentil: float = 97.0) -> torch.Tensor:
    """
    Mapa de descontinuidade de profundidade, normalizado por percentil robusto em [0,1].

    Por que importa para o bokeh: a quebra de profundidade é exatamente onde o kernel de
    desfoque não pode atravessar. Um bokeh que ignora essa fronteira gera o artefato de
    vazamento de cor entre primeiro plano e fundo.
    """
    g = gradient_magnitude(depth, h)
    B = g.shape[0]
    flat = g.view(B, -1)
    lim = torch.quantile(flat, percentil / 100.0, dim=1).view(B, 1, 1, 1)
    return (g / (lim + 1e-8)).clamp(0.0, 1.0)


def compute_all(depth: torch.Tensor, smooth_sigma: float = 0.5,
                clamp_val: float = 50.0,
                h: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """
    Calcula todos os sinais de uma vez, a partir das mesmas derivadas.

    Retorna um dicionário nome -> tensor (B,1,H,W), exceto "normals" (B,3,H,W).
    """
    k1, k2, K, H = principal_curvatures(depth, smooth_sigma, h, clamp_val)
    return {
        "depth": depth,
        "grad": gradient_magnitude(depth, h),
        "det_g": metric_determinant(depth, h),
        "area_element": area_element(depth, h),
        "normals": _surface_normals(depth, h),
        "K": K,
        "H": H,
        "k1": k1,
        "k2": k2,
        "curvedness": curvedness(k1, k2),
        "shape_index": shape_index(k1, k2),
        "occlusion": occlusion_map(depth, h),
    }


# Metadados de visualização: como cada sinal deve ser colorido.
#   divergente=True  -> colormap divergente centrado em zero (o sinal tem significado)
#   divergente=False -> colormap sequencial
VIS_INFO = {
    "depth":       dict(cmap="turbo",   divergente=False, titulo="Mapa de profundidade"),
    "grad":        dict(cmap="magma",   divergente=False, titulo="Magnitude do gradiente |∇z|"),
    "det_g":       dict(cmap="viridis", divergente=False, titulo="Tensor métrico det(g)"),
    "area_element": dict(cmap="hot",    divergente=False,
                         titulo="√det(g): elemento de área riemanniano"),
    "normals":     dict(cmap=None,      divergente=False, titulo="Vetores normais"),
    "K":           dict(cmap="RdBu_r",  divergente=True,  titulo="Curvatura Gaussiana K"),
    "H":           dict(cmap="PuOr_r",  divergente=True,  titulo="Curvatura média H"),
    "k1":          dict(cmap="RdBu_r",  divergente=True,  titulo="Curvatura principal k1"),
    "k2":          dict(cmap="RdBu_r",  divergente=True,  titulo="Curvatura principal k2"),
    "curvedness":  dict(cmap="inferno", divergente=False, titulo="Curvedness"),
    "shape_index": dict(cmap="Spectral", divergente=True, titulo="Índice de forma"),
    "occlusion":   dict(cmap="hot",     divergente=False, titulo="Descontinuidade / oclusão"),
}
