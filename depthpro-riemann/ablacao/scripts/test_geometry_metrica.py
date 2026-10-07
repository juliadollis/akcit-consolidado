#!/usr/bin/env python3
"""
scripts/test_geometry_metrica.py
================================
Validação rigorosa da curvatura em unidades métricas.

Diferente do `test_geometry.py` antigo, que construía as superfícies com x, y e z nas
MESMAS unidades e por isso não detectava inconsistência de parametrização, aqui as cenas
são construídas como um sensor real as veria: uma câmera pinhole com focal em pixels
observando objetos com dimensões em metros a distâncias em metros.

O teste varre várias combinações de raio, distância, focal e resolução. Se a
parametrização estiver correta, o erro deve ser pequeno e, principalmente, NÃO deve
depender sistematicamente de nenhum desses parâmetros. Uma dependência residual indicaria
que algum fator de escala ficou fora do lugar.

Rode antes de qualquer treino que use o termo de curvatura.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.geometry import (area_element_3d, surface_curvatures,  # noqa: E402
                              surface_normals)

OK, FALHA = "OK  ", "FALHA"


def _grade_ab(H, W, f, cx=None, cy=None):
    """Campos a=(u-cx)/f e b=(v-cy)/f. O raio do pixel (u,v) e (a·t, b·t, t)."""
    cx = (W - 1) / 2 if cx is None else cx
    cy = (H - 1) / 2 if cy is None else cy
    u = torch.arange(W, dtype=torch.float64) - cx
    v = torch.arange(H, dtype=torch.float64) - cy
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uu / f, vv / f


def cena_esfera(R, Zc, f, H=256, W=256):
    """
    Esfera de raio R com centro em (0, 0, Zc+R), vista por camera pinhole.

    A profundidade vem do TRACADO DE RAIO, nao de uma aproximacao. O raio do pixel e
    (a·t, b·t, t); intersectando com |P - C|² = R² obtem-se a equacao quadratica

        t²(a²+b²+1) - 2·Cz·t + (Cz² - R²) = 0

    e tomamos a raiz proxima. Construir a cena de outra forma (por exemplo supondo
    x = u·Zc/f, com Zc fixo em vez da profundidade local) produz uma superficie que NAO
    e uma esfera, e o teste passaria a medir o erro da cena em vez do erro do codigo.

    Na superficie visivel, K = 1/R² e |H| = 1/R exatamente.
    """
    a, b = _grade_ab(H, W, f)
    Cz = float(Zc + R)
    A = a ** 2 + b ** 2 + 1.0
    disc = Cz ** 2 - A * (Cz ** 2 - R ** 2)
    atinge = disc > 0
    t = torch.full((H, W), float(Zc + 2 * R), dtype=torch.float64)   # fundo atras
    t[atinge] = ((Cz - torch.sqrt(disc[atinge])) / A[atinge])
    # miolo da calota, longe da silhueta onde a normal fica rasante
    ang = torch.sqrt(a ** 2 + b ** 2)
    ang_max = float(torch.asin(torch.tensor(min(R / Cz, 0.999))))
    m = atinge & (ang < 0.6 * ang_max)
    return t.view(1, 1, H, W), m.view(1, 1, H, W)


def cena_plano(Zc, f, nx=0.0, ny=0.0, H=256, W=256):
    """
    Plano com normal (nx, ny, 1) normalizada, passando pela distancia Zc no eixo optico.

    Para um plano n·P = d, com P = (a·t, b·t, t), a profundidade e

        t = d / (nx·a + ny·b + nz)

    que NAO e linear em a e b. Supor linearidade produziria uma superficie curva, e o
    teste acusaria curvatura onde nao ha.

    K = 0 e H = 0 em todo ponto, qualquer que seja a inclinacao.
    """
    a, b = _grade_ab(H, W, f)
    n = torch.tensor([nx, ny, 1.0], dtype=torch.float64)
    n = n / n.norm()
    d = float(n[2] * Zc)
    den = n[0] * a + n[1] * b + n[2]
    t = d / den.clamp(min=1e-6)
    m = torch.ones((H, W), dtype=torch.bool)
    m[:10] = m[-10:] = False
    m[:, :10] = m[:, -10:] = False
    return t.view(1, 1, H, W), m.view(1, 1, H, W)


def cena_cilindro(R, Zc, f, H=256, W=256):
    """
    Cilindro de raio R e eixo paralelo a Y, com centro em (0, *, Zc+R).

    Tracado de raio: t²(a²+1) - 2·Cz·t + (Cz² - R²) = 0.
    Superficie desenvolvivel, entao K = 0; e |H| = 1/(2R).
    """
    a, b = _grade_ab(H, W, f)
    Cz = float(Zc + R)
    A = a ** 2 + 1.0
    disc = Cz ** 2 - A * (Cz ** 2 - R ** 2)
    atinge = disc > 0
    t = torch.full((H, W), float(Zc + 2 * R), dtype=torch.float64)
    t[atinge] = ((Cz - torch.sqrt(disc[atinge])) / A[atinge])
    ang_max = float(torch.asin(torch.tensor(min(R / Cz, 0.999))))
    m = atinge & (a.abs() < 0.6 * ang_max)
    m[:10] = m[-10:] = False
    return t.view(1, 1, H, W), m.view(1, 1, H, W)


def mediana(t, m):
    v = t[m]
    return float(v.median()) if v.numel() else float("nan")


def main():
    print("=" * 78)
    print("VALIDACAO DA CURVATURA EM UNIDADES METRICAS")
    print("=" * 78)
    falhas = 0

    # ---------------- 1. Esfera: K = 1/R^2, |H| = 1/R -------------------
    print("\n[1] ESFERA — K deve valer 1/R^2 e |H| deve valer 1/R")
    print(f"    {'R(m)':>5s} {'Z(m)':>6s} {'f(px)':>7s} {'res':>6s} "
          f"{'K esperado':>11s} {'K medido':>11s} {'erro K':>8s} {'erro H':>8s}")
    casos = [(2.0, 10.0, 500.0, 256), (2.0, 10.0, 500.0, 512),
             (0.5, 3.0, 500.0, 256), (5.0, 30.0, 500.0, 256),
             (2.0, 10.0, 250.0, 256), (2.0, 10.0, 1000.0, 256),
             (1.0, 50.0, 800.0, 256), (10.0, 60.0, 500.0, 256)]
    for R, Zc, f, res in casos:
        z, m = cena_esfera(R, Zc, f, res, res)
        r = surface_curvatures(z, fx=f, smooth_sigma=0.0, clamp_val=None)
        Kesp, Hesp = 1.0 / R ** 2, 1.0 / R
        Kmed = mediana(r["K"], m)
        Hmed = abs(mediana(r["H"], m))
        eK = abs(Kmed - Kesp) / Kesp * 100
        eH = abs(Hmed - Hesp) / Hesp * 100
        marca = OK if (eK < 5 and eH < 5) else FALHA
        falhas += marca == FALHA
        print(f"    {R:5.1f} {Zc:6.1f} {f:7.0f} {res:6d} "
              f"{Kesp:11.4f} {Kmed:11.4f} {eK:7.2f}% {eH:7.2f}%  {marca}")

    # ---------------- 2. Plano: K = 0, H = 0 -----------------------------
    print("\n[2] PLANO — K e H devem ser nulos, mesmo inclinado")
    print(f"    {'Z(m)':>6s} {'incl.x':>7s} {'incl.y':>7s} {'|K| med':>12s} "
          f"{'|H| med':>12s}   criterio: |H|*Z < 1e-2")
    for Zc, ix, iy in [(10.0, 0.0, 0.0), (10.0, 0.5, 0.0), (10.0, 0.5, 0.3),
                       (50.0, 1.0, 0.0), (3.0, 0.2, 0.2)]:
        z, m = cena_plano(Zc, 500.0, ix, iy)
        r = surface_curvatures(z, fx=500.0, smooth_sigma=0.0, clamp_val=None)
        aK = abs(mediana(r["K"], m))
        aH = abs(mediana(r["H"], m))
        # Criterio fisico, nao numerico: um plano e "plano o bastante" quando o raio de
        # curvatura aparente e muito maior que a propria distancia da cena. Exigimos
        # |H|·Z < 1e-2, ou seja, raio de curvatura acima de 100x a distancia. Um limiar
        # absoluto puniria cenas proximas por precisao de ponto flutuante, nao por erro.
        marca = OK if (aK * Zc ** 2 < 1e-2 and aH * Zc < 1e-2) else FALHA
        falhas += marca == FALHA
        print(f"    {Zc:6.1f} {ix:7.2f} {iy:7.2f} {aK:12.3e} {aH:12.3e}  {marca}")

    # ---------------- 3. Cilindro: K = 0, |H| = 1/(2R) -------------------
    print("\n[3] CILINDRO — K deve ser nulo (desenvolvivel) e |H| = 1/(2R)")
    print(f"    {'R(m)':>5s} {'Z(m)':>6s} {'|K| med':>12s} {'|H| esp':>9s} "
          f"{'|H| med':>9s} {'erro H':>8s}")
    for R, Zc in [(2.0, 10.0), (1.0, 5.0), (5.0, 25.0)]:
        z, m = cena_cilindro(R, Zc, 500.0)
        r = surface_curvatures(z, fx=500.0, smooth_sigma=0.0, clamp_val=None)
        aK = abs(mediana(r["K"], m))
        Hesp = 1.0 / (2 * R)
        Hmed = abs(mediana(r["H"], m))
        eH = abs(Hmed - Hesp) / Hesp * 100
        marca = OK if (aK < 1e-3 and eH < 8) else FALHA
        falhas += marca == FALHA
        print(f"    {R:5.1f} {Zc:6.1f} {aK:12.3e} {Hesp:9.4f} {Hmed:9.4f} "
              f"{eH:7.2f}%  {marca}")

    # ---------------- 4. Independencia de resolucao e focal --------------
    print("\n[4] INVARIANCIA — o mesmo objeto deve dar o mesmo K, mudando camera")
    print("    (uma dependencia residual indicaria fator de escala fora do lugar)")
    vals = []
    for f, res in [(300.0, 192), (500.0, 256), (800.0, 384), (1200.0, 512)]:
        z, m = cena_esfera(2.0, 10.0, f, res, res)
        r = surface_curvatures(z, fx=f, smooth_sigma=0.0, clamp_val=None)
        k = mediana(r["K"], m)
        vals.append(k)
        print(f"    f={f:6.0f} px, {res}x{res}  ->  K = {k:.4f}")
    disp = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100
    marca = OK if disp < 5 else FALHA
    falhas += marca == FALHA
    print(f"    dispersao entre configuracoes: {disp:.2f}%  {marca}")

    # ---------------- 5. Normais e elemento de area ----------------------
    print("\n[5] NORMAIS E ELEMENTO DE AREA")
    z, m = cena_plano(10.0, 500.0, 0.0, 0.0)
    n = surface_normals(z, fx=500.0)
    nz = float(n[0, 2][m[0, 0]].abs().median())
    marca = OK if nz > 0.99 else FALHA
    falhas += marca == FALHA
    print(f"    plano frontoparalelo: |n_z| = {nz:.4f} (esperado ~1)  {marca}")

    dA = area_element_3d(z, fx=500.0)
    esp = (10.0 / 500.0) ** 2          # (Z/f)^2 m^2 por pixel
    med = float(dA[m].median())
    e = abs(med - esp) / esp * 100
    marca = OK if e < 5 else FALHA
    falhas += marca == FALHA
    print(f"    elemento de area: esperado {esp:.3e} m2/px, medido {med:.3e} "
          f"({e:.2f}%)  {marca}")

    # ---------------- 6. Comparacao com a formulacao antiga --------------
    print("\n[6] MAGNITUDE DO ERRO DA FORMULACAO ANTERIOR")
    try:
        from riemann.losses import gaussian_curvature
        print(f"    {'R(m)':>5s} {'Z(m)':>6s} {'K verdadeiro':>13s} "
              f"{'K antigo':>14s} {'erro':>14s}")
        for R, Zc in [(2.0, 10.0), (0.5, 3.0), (5.0, 30.0)]:
            z, m = cena_esfera(R, Zc, 500.0)
            Kv = 1.0 / R ** 2
            Ka = mediana(gaussian_curvature(z, 0.0, 1e12), m)
            print(f"    {R:5.1f} {Zc:6.1f} {Kv:13.4f} {Ka:14.2f} "
                  f"{abs(Ka-Kv)/Kv*100:13.0f}%")
        print("    A formulacao anterior usava h = 1/max(H,W), ou seja, coordenadas")
        print("    de imagem normalizadas com profundidade em metros. As unidades dos")
        print("    eixos nao batem e a quantidade resultante nao e curvatura fisica.")
    except Exception as e:
        print(f"    [pulado] {e}")

    print("\n" + "=" * 78)
    if falhas == 0:
        print("TODOS OS TESTES PASSARAM — a curvatura esta em unidades metricas corretas.")
    else:
        print(f"{falhas} TESTE(S) FALHARAM — nao use o termo de curvatura ate resolver.")
    print("=" * 78)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
