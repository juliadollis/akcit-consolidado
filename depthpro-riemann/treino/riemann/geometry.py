"""
riemann/geometry.py
===================
Geometria diferencial da superfície 3D reconstruída a partir de um mapa de profundidade,
com os intrínsecos da câmera.

MOTIVAÇÃO — por que este módulo existe
--------------------------------------
A implementação anterior calculava a curvatura do GRÁFICO z = D(u,v), usando espaçamento
`h = 1/max(H,W)`, isto é, coordenadas de imagem normalizadas em [0,1], enquanto a
profundidade está em METROS. Isso mistura unidades nos dois eixos e a quantidade
resultante não é a curvatura de nenhuma superfície física.

O erro não é pequeno. Numa esfera de raio 2 m a 10 m de distância, cuja curvatura
verdadeira é K = 1/R² = 0,25, aquela formulação devolve ~2756, um erro de seis ordens de
magnitude. O erro depende da resolução e da faixa de profundidade da cena, o que explica
por que a distribuição de |K| tinha cauda de milhões e por que o teto de estabilidade
cortava boa parte dos pixels.

O teste de sanidade não pegava isso porque construía a esfera com x, y e z nas MESMAS
unidades. A inconsistência só aparece com dados reais.

A FORMULAÇÃO CORRETA
--------------------
Com uma câmera pinhole de focal (fx, fy) e centro óptico (cx, cy), o pixel (u,v) com
profundidade D retroprojeta para

    S(u,v) = ( a·D , b·D , D ),      a = (u-cx)/fx ,   b = (v-cy)/fy

Esta é uma superfície paramétrica em R³. Suas derivadas são

    S_u = ( D/fx + a·D_u ,  b·D_u ,  D_u )
    S_v = ( a·D_v ,  D/fy + b·D_v ,  D_v )

    S_uu = ( 2·D_u/fx + a·D_uu ,  b·D_uu ,  D_uu )
    S_uv = ( D_v/fx + a·D_uv ,  D_u/fy + b·D_uv ,  D_uv )
    S_vv = ( a·D_vv ,  2·D_v/fy + b·D_vv ,  D_vv )

A primeira forma fundamental é E = S_u·S_u, F = S_u·S_v, G = S_v·S_v; a normal unitária é
N = (S_u × S_v)/‖S_u × S_v‖; e a segunda forma fundamental é L = S_uu·N, M = S_uv·N,
Nn = S_vv·N. Daí

    K = (L·Nn − M²)/(E·G − F²)          curvatura Gaussiana (intrínseca)
    H = (E·Nn − 2F·M + G·L)/(2(E·G − F²))   curvatura média (extrínseca)

As derivadas D_u, D_v, D_uu, ... são tomadas em unidades de PIXEL (espaçamento 1), porque
a conversão para métrica já está embutida no termo D/f. Isso é importante: não se deve
aplicar nenhum outro fator de escala.

Validado contra esfera, plano e cilindro em várias combinações de raio, distância, focal e
resolução. Ver `scripts/test_geometry_metrica.py`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Derivadas em unidades de pixel (espaçamento unitário)
# ---------------------------------------------------------------------------
def _d1(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Primeiras derivadas por diferenças centrais, espaçamento de 1 pixel."""
    p = F.pad(t, (1, 1, 1, 1), mode="replicate")
    du = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) * 0.5
    dv = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) * 0.5
    return du, dv


def _d2(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segundas derivadas por diferenças centrais, espaçamento de 1 pixel."""
    p = F.pad(t, (1, 1, 1, 1), mode="replicate")
    duu = p[:, :, 1:-1, 2:] - 2 * t + p[:, :, 1:-1, :-2]
    dvv = p[:, :, 2:, 1:-1] - 2 * t + p[:, :, :-2, 1:-1]
    duv = (p[:, :, 2:, 2:] - p[:, :, 2:, :-2]
           - p[:, :, :-2, 2:] + p[:, :, :-2, :-2]) * 0.25
    return duu, dvv, duv


def _suavizar(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Suavização gaussiana separável. Aplicada à profundidade antes de derivar."""
    if not sigma or sigma <= 0:
        return t
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=t.dtype, device=t.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    a = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(a, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def grade_normalizada(depth: torch.Tensor, fx: float, fy: Optional[float] = None,
                      cx: Optional[float] = None, cy: Optional[float] = None):
    """
    Devolve os campos a = (u-cx)/fx e b = (v-cy)/fy, com a forma de `depth`.

    Se cx/cy não forem dados, assume o centro geométrico da imagem, que é a convenção
    usual quando os intrínsecos completos não estão disponíveis. O efeito de um centro
    óptico ligeiramente deslocado sobre a curvatura é de segunda ordem.
    """
    B, _, H, W = depth.shape
    fy = fx if fy is None else fy
    cx = (W - 1) / 2.0 if cx is None else cx
    cy = (H - 1) / 2.0 if cy is None else cy
    u = torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, 1, 1, W)
    v = torch.arange(H, dtype=depth.dtype, device=depth.device).view(1, 1, H, 1)
    return (u - cx) / fx, (v - cy) / fy


# ---------------------------------------------------------------------------
# Curvaturas da superfície 3D
# ---------------------------------------------------------------------------
def surface_curvatures(depth: torch.Tensor, fx: float, fy: Optional[float] = None,
                       cx: Optional[float] = None, cy: Optional[float] = None,
                       smooth_sigma: float = 0.0,
                       clamp_val: Optional[float] = None,
                       eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    """
    Curvaturas Gaussiana e média da superfície S(u,v) = ((u-cx)D/fx, (v-cy)D/fy, D).

    Args:
        depth: (B,1,H,W) profundidade MÉTRICA (metros), não normalizada.
        fx, fy: distância focal em PIXELS. `fy` assume `fx` se omitido.
        cx, cy: centro óptico em pixels. Assume o centro da imagem se omitido.
        smooth_sigma: suavização da profundidade antes de derivar. As segundas
            derivadas amplificam ruído, então algum valor é recomendado em
            profundidade estimada; use 0 para ground truth sintético limpo.
        clamp_val: teto simétrico por pixel, só para estabilidade numérica. Com a
            parametrização correta os valores ficam em faixa fisiológica e o teto
            raramente atua; ele existe para conter singularidades em descontinuidades.

    Retorna dict com K, H, k1, k2 e as formas fundamentais E, F, G, L, M, Nn.

    Unidades: K em 1/m², H em 1/m. Uma esfera de raio R dá K = 1/R² e |H| = 1/R.
    """
    fy = fx if fy is None else fy
    D = _suavizar(depth, smooth_sigma)
    a, b = grade_normalizada(D, fx, fy, cx, cy)

    Du, Dv = _d1(D)
    Duu, Dvv, Duv = _d2(D)

    # Vetores tangentes
    Su = torch.stack([D / fx + a * Du, b * Du, Du], dim=-1)
    Sv = torch.stack([a * Dv, D / fy + b * Dv, Dv], dim=-1)

    # Derivadas segundas da parametrização
    Suu = torch.stack([2 * Du / fx + a * Duu, b * Duu, Duu], dim=-1)
    Suv = torch.stack([Dv / fx + a * Duv, Du / fy + b * Duv, Duv], dim=-1)
    Svv = torch.stack([a * Dvv, 2 * Dv / fy + b * Dvv, Dvv], dim=-1)

    # Primeira forma fundamental
    E = (Su * Su).sum(-1)
    Ff = (Su * Sv).sum(-1)
    G = (Sv * Sv).sum(-1)

    # Normal unitária
    n = torch.cross(Su, Sv, dim=-1)
    n_norm = n.norm(dim=-1, keepdim=True).clamp(min=eps)
    N = n / n_norm

    # Segunda forma fundamental
    L = (Suu * N).sum(-1)
    M = (Suv * N).sum(-1)
    Nn = (Svv * N).sum(-1)

    den = (E * G - Ff ** 2).clamp(min=eps)
    K = (L * Nn - M ** 2) / den
    Hc = (E * Nn - 2 * Ff * M + G * L) / (2 * den)

    if clamp_val is not None and clamp_val > 0:
        K = K.clamp(-clamp_val, clamp_val)
        Hc = Hc.clamp(-clamp_val, clamp_val)

    disc = torch.clamp(Hc ** 2 - K, min=0.0).sqrt()
    return {"K": K, "H": Hc, "k1": Hc - disc, "k2": Hc + disc,
            "E": E, "F": Ff, "G": G, "L": L, "M": M, "Nn": Nn,
            "normal": N.permute(0, 1, 4, 2, 3).squeeze(1)}


def surface_normals(depth: torch.Tensor, fx: float, fy: Optional[float] = None,
                    cx: Optional[float] = None, cy: Optional[float] = None,
                    smooth_sigma: float = 0.0) -> torch.Tensor:
    """
    Normais unitárias da superfície 3D, em (B,3,H,W).

    Diferente da normal do gráfico em coordenadas de imagem, esta é a normal geométrica
    verdadeira, obtida do produto vetorial dos tangentes da superfície retroprojetada.
    """
    fy = fx if fy is None else fy
    D = _suavizar(depth, smooth_sigma)
    a, b = grade_normalizada(D, fx, fy, cx, cy)
    Du, Dv = _d1(D)
    Su = torch.stack([D / fx + a * Du, b * Du, Du], dim=-1)
    Sv = torch.stack([a * Dv, D / fy + b * Dv, Dv], dim=-1)
    n = torch.cross(Su, Sv, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return n.permute(0, 1, 4, 2, 3).squeeze(1)


def area_element_3d(depth: torch.Tensor, fx: float, fy: Optional[float] = None,
                    cx: Optional[float] = None, cy: Optional[float] = None,
                    smooth_sigma: float = 0.0) -> torch.Tensor:
    """
    Elemento de área da superfície 3D por pixel, sqrt(EG − F²), em m²/pixel².

    É o fator que converte área em pixels para área métrica sobre a superfície. Cresce
    com a distância (porque cada pixel cobre mais área) e com a inclinação.
    """
    r = surface_curvatures(depth, fx, fy, cx, cy, smooth_sigma, clamp_val=None)
    return (r["E"] * r["G"] - r["F"] ** 2).clamp(min=0).sqrt()
