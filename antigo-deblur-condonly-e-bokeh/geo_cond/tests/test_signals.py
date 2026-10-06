"""Testes dos sinais geométricos. Sem rede, sem GPU.

A régua é geometria de resposta conhecida, não regressão contra a saída atual:
esfera de raio R tem K = 1/R^2, plano e cilindro têm K = 0, plano frontoparalelo
tem gradiente nulo. É o mesmo teste que o `RETESTE_CURVATURA.md` usou para
descobrir que a formulação de Monge devolve ~10 onde o valor verdadeiro é 0,25.
"""

from __future__ import annotations

import numpy as np
import pytest

from geo_cond.constants import GeoConstants
from geo_cond import signals as S
from geo_cond.tests import geometrias as G


def _consts(**kw) -> GeoConstants:
    base = dict(
        tau_occlusion=1.0, u_max=1.0, s_max=1.0, k0_curvature=1.0,
        kt_max=5.0, z_percentile_max=99.5, smooth_sigma=0.5, min_quant_levels=256,
    )
    base.update(kw)
    return GeoConstants(**base)


def _miolo(a: np.ndarray, m: int = 12) -> np.ndarray:
    """Interior, longe da borda: fora dela as diferenças são unilaterais e a
    suavização replica, então o valor ali não é a fórmula que se quer testar."""
    return a[m:-m, m:-m]


# ---------------------------------------------------------------- curvatura

def test_esfera_tem_curvatura_um_sobre_r2():
    R, d, fx, n = 2.0, 10.0, 800.0, 96
    z = G.esfera(n, fx, R, d)
    K = S.gaussian_curvature_backprojected(z, fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5)
    esperado = 1.0 / R ** 2
    obtido = float(np.median(_miolo(K)))
    assert abs(obtido - esperado) / esperado < 0.02, (
        f"K da esfera: obtido {obtido:.4f}, esperado {esperado:.4f}. "
        "A formulação de Monge devolve ~10 aqui; a retroprojetada devolve 0,25."
    )


@pytest.mark.parametrize("R", [0.5, 2.0, 5.0])
def test_curvatura_da_esfera_escala_com_o_raio(R):
    d, fx, n = R * 5.0, 1600.0, 96
    K = S.gaussian_curvature_backprojected(
        G.esfera(n, fx, R, d), fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5
    )
    assert abs(float(np.median(_miolo(K))) - 1.0 / R ** 2) / (1.0 / R ** 2) < 0.03


def test_plano_inclinado_tem_curvatura_nula():
    fx, n = 800.0, 96
    z = G.plano_inclinado(n, fx, (0.3, 0.1, 1.0), 5.0)
    K = S.gaussian_curvature_backprojected(z, fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5)
    assert float(np.abs(_miolo(K)).max()) < 1e-3


def test_cilindro_tem_curvatura_nula():
    """Superfície desenvolvível: curva numa direção, K = 0 mesmo assim.
    É o teste que separa curvatura gaussiana de curvatura média."""
    fx, n = 1600.0, 96
    z = G.cilindro(n, fx, 1.0, 6.0)
    K = S.gaussian_curvature_backprojected(z, fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5)
    assert float(np.abs(_miolo(K)).max()) < 5e-3


def test_curvatura_e_invariante_a_resolucao_com_fx_corrigido():
    """Se `fx` acompanha a resolução, K não muda. É o contrato que a correção
    do resize (512/min(W,H)) tem de cumprir."""
    R, d = 2.0, 10.0
    K1 = S.gaussian_curvature_backprojected(
        G.esfera(96, 800.0, R, d), 800.0, 800.0, 47.5, 47.5, 0.5)
    K2 = S.gaussian_curvature_backprojected(
        G.esfera(192, 1600.0, R, d), 1600.0, 1600.0, 95.5, 95.5, 0.5)
    a, b = float(np.median(_miolo(K1))), float(np.median(_miolo(K2, 24)))
    assert abs(a - b) / a < 0.02, f"K a 96px={a:.4f} vs a 192px={b:.4f}"


def test_curvatura_com_fx_errado_da_resposta_errada():
    """Guarda contra o defeito que o plano descreve: se ninguém aplicar o fator
    de resize ao `fx`, a curvatura sai errada. O teste existe para que isso
    apareça como falha e não como número plausível."""
    R, d = 2.0, 10.0
    K_certo = S.gaussian_curvature_backprojected(
        G.esfera(192, 1600.0, R, d), 1600.0, 1600.0, 95.5, 95.5, 0.5)
    K_errado = S.gaussian_curvature_backprojected(
        G.esfera(192, 1600.0, R, d), 800.0, 800.0, 95.5, 95.5, 0.5)
    assert abs(float(np.median(_miolo(K_certo, 24))) - 0.25) / 0.25 < 0.02
    assert abs(float(np.median(_miolo(K_errado, 24))) - 0.25) / 0.25 > 0.5


# ---------------------------------------------------------- primeira ordem

def test_frontoparalelo_zera_gradiente_oclusao_e_area():
    n = 64
    z = G.frontoparalelo(n, 5.0)
    fx_, fy_ = S.grad_normalized(1.0 / z, px_per_unit=n)
    assert float(np.abs(fx_).max()) < 1e-12 and float(np.abs(fy_).max()) < 1e-12
    s, nx, ny = S.area_element_and_normals(fx_, fy_)
    assert float(np.abs(s).max()) < 1e-12
    assert float(np.abs(nx).max()) < 1e-12 and float(np.abs(ny).max()) < 1e-12
    assert float(S.occlusion(np.sqrt(fx_**2 + fy_**2), tau=1.0).max()) == 0.0


def test_rampa_linear_no_campo_da_gradiente_constante():
    """Linear NO CAMPO que se diferencia, que é u quando field='inverse'."""
    n = 64
    u = np.tile(np.linspace(1 / 2.0, 1 / 6.0, n), (n, 1))
    fx_, fy_ = S.grad_normalized(u, px_per_unit=n)
    mi = _miolo(fx_, 2)
    assert float(mi.std()) / abs(float(mi.mean())) < 1e-9, "gradiente tem de ser constante"
    assert float(np.abs(_miolo(fy_, 2)).max()) < 1e-9


def test_rampa_linear_em_z_concentra_o_gradiente_de_u_no_perto():
    """A propriedade que justifica field='inverse' como default.

    Numa rampa linear em Z, grad(1/Z) = -Z'/Z^2, ou seja o sinal se concentra no
    que está PERTO e desaparece no fundo. Com grad(Z) puro seria o contrário: o
    fundo distante domina, que é justamente onde o borrão é mais uniforme e a
    geometria menos útil.

    (Este teste nasceu de um teste ERRADO que afirmava gradiente constante aqui.
    O erro era a premissa, não o código.)"""
    n = 64
    z = G.rampa(n, 2.0, 6.0)
    gu = np.abs(S.grad_normalized(1.0 / z, px_per_unit=n)[0])
    gz = np.abs(S.grad_normalized(z, px_per_unit=n)[0])
    perto, longe = slice(4, 12), slice(n - 12, n - 4)
    razao_u = float(gu[:, perto].mean() / gu[:, longe].mean())
    razao_z = float(gz[:, perto].mean() / gz[:, longe].mean())

    # A expectativa vem do proprio array, nao de um numero chutado: como
    # |grad(1/z)| = |z'|/z^2 e z' e constante, a razao prevista e a razao das
    # medias de 1/z^2 nas duas fatias.
    prev = float((1.0 / z[:, perto] ** 2).mean() / (1.0 / z[:, longe] ** 2).mean())
    assert abs(razao_u - prev) / prev < 0.01, f"medido {razao_u:.3f} vs previsto {prev:.3f}"
    assert razao_u > 4.0, f"o sinal tem de se concentrar no perto; razao {razao_u:.2f}"
    assert abs(razao_z - 1.0) < 1e-6, f"em z, perto/longe = {razao_z:.4f} (esperado 1)"


def test_degrau_satura_a_oclusao_na_borda_e_so_nela():
    n = 64
    u = 1.0 / G.degrau(n, 1.0, 20.0)
    fx_, fy_ = S.grad_normalized(u, px_per_unit=n)
    mag = np.sqrt(fx_**2 + fy_**2)
    O = S.occlusion(mag, tau=float(mag.max()) * 0.5)
    col = n // 2
    assert float(O[:, col - 1:col + 1].min()) == 1.0, "a borda tem de saturar"
    assert float(O[:, :col - 3].max()) == 0.0, "longe da borda tem de ser zero"
    assert float(O[:, col + 3:].max()) == 0.0


def test_primeira_ordem_e_invariante_a_resolucao():
    """Mesma cena a 96 e a 192 px, com px_per_unit = lado da imagem, dá o mesmo
    gradiente. Sem isso, treinar a 512 e inferir com tiling na resolução
    original mudaria o significado do canal."""
    R, d = 2.0, 10.0
    u1 = 1.0 / G.esfera(96, 800.0, R, d)
    u2 = 1.0 / G.esfera(192, 1600.0, R, d)
    m1 = np.hypot(*S.grad_normalized(u1, px_per_unit=96))
    m2 = np.hypot(*S.grad_normalized(u2, px_per_unit=192))
    a, b = float(np.median(_miolo(m1))), float(np.median(_miolo(m2, 24)))
    assert abs(a - b) / max(a, 1e-12) < 0.02, f"96px={a:.6g} vs 192px={b:.6g}"


def test_gradiente_por_pixel_NAO_e_invariante_a_resolucao():
    """O contraste do teste acima: sem a normalização, o mesmo campo a 2x a
    resolução dá metade do gradiente. É exatamente o `h = 1/max(H,W)` calculado
    a partir do array que a auditoria apontou."""
    R, d = 2.0, 10.0
    m1 = np.hypot(*S.grad_normalized(1.0 / G.esfera(96, 800.0, R, d), 1.0))
    m2 = np.hypot(*S.grad_normalized(1.0 / G.esfera(192, 1600.0, R, d), 1.0))
    a, b = float(np.median(_miolo(m1))), float(np.median(_miolo(m2, 24)))
    assert abs(a / b - 2.0) < 0.1, f"esperado fator 2, obtido {a/b:.3f}"


def test_oclusao_e_invariante_a_reescala_do_campo_quando_tau_acompanha():
    """A propriedade que sustenta usar oclusão mesmo sem escala métrica: se o
    campo e o tau são reescalados pelo mesmo fator, O não muda."""
    n = 64
    mag = np.hypot(*S.grad_normalized(1.0 / G.rampa(n, 2.0, 6.0), n))
    assert np.allclose(S.occlusion(mag, 0.3), S.occlusion(3.0 * mag, 0.9))


def test_erosao_da_mascara_impede_borda_falsa_de_invalido():
    n = 32
    mag = np.ones((n, n))
    valid = np.ones((n, n), dtype=bool)
    valid[:, :8] = False                       # região inválida à esquerda
    O = S.occlusion(mag, tau=1.0, valid=valid)
    assert float(O[:, :9].max()) == 0.0, "a fronteira do inválido tem de ser erodida"
    assert float(O[:, 12:].min()) == 1.0


# --------------------------------------------------------------- utilidades

@pytest.mark.parametrize("sigma,taps", [(0.5, 5), (1.0, 7), (2.0, 13), (3.0, 19)])
def test_raio_do_nucleo_segue_tres_sigmas(sigma, taps):
    """`riemann/losses.py:86` fixa radius=2 para qualquer sigma. Aqui não."""
    k = S.gaussian_kernel_1d(sigma)
    assert len(k) == taps
    assert abs(float(k.sum()) - 1.0) < 1e-12


def test_depth01_para_metrico_bate_com_a_formula_do_dataloader():
    d01 = np.array([[0.0, 0.5, 1.0]])
    z = S.depth01_to_metric(d01, 2.0, 10.0)
    assert np.allclose(z, [[2.0, 6.0, 10.0]])


def test_faixa_metrica_invalida_levanta():
    with pytest.raises(ValueError):
        S.depth01_to_metric(np.zeros((4, 4)), 10.0, 2.0)


def test_niveis_de_quantizacao():
    assert S.quantization_levels(np.linspace(0, 1, 1000).reshape(1, -1)) == 1000
    assert S.quantization_levels(np.full((8, 8), 0.5)) == 1


# ------------------------------------------------------------------ driver

def _cena_valida(n=96, fx=800.0):
    z = G.esfera(n, fx, 2.0, 10.0)
    zmin, zmax = float(z.min()), float(z.max())
    return (z - zmin) / (zmax - zmin), zmin, zmax, fx


def test_pilha_tem_seis_canais_em_zero_um_e_finitos():
    d01, zmin, zmax, fx = _cena_valida()
    out = S.geometric_stack(
        d01, z_min_m=zmin, z_max_m=zmax, fx_px=fx, fy_px=fx, cx=47.5, cy=47.5,
        px_per_unit=96, consts=_consts(u_max=0.2, s_max=1.0),
    )
    assert out.canais.shape == (6, 96, 96)
    assert out.canais.dtype == np.float32
    assert float(out.canais.min()) >= 0.0 and float(out.canais.max()) <= 1.0
    assert np.all(np.isfinite(out.canais))


def test_gate_de_quantizacao_neutraliza_a_curvatura():
    """Com poucos níveis, o canal K sai NEUTRO (0,5) em vez de ruído, e a flag
    avisa. Medido na auditoria: 2.870 amostras da rota b (24,7%) têm a cena útil
    em menos de 256 níveis, mediana de 23 no subgrupo com z_max sentinela."""
    d01, zmin, zmax, fx = _cena_valida()
    grosso = np.rint(d01 * 20) / 20.0            # 21 níveis
    out = S.geometric_stack(
        grosso, z_min_m=zmin, z_max_m=zmax, fx_px=fx, fy_px=fx, cx=47.5, cy=47.5,
        px_per_unit=96, consts=_consts(u_max=0.2), )
    assert out.segunda_ordem_valida is False
    assert out.niveis_quantizacao < 256
    assert np.allclose(out.canais[5], 0.5)


def test_field_inverse_e_depth_dao_resultados_diferentes():
    """Se dessem o mesmo, a escolha não seria ablacionável e a discussão física
    sobre grad u contra grad D não teria consequência."""
    d01, zmin, zmax, fx = _cena_valida()
    kw = dict(z_min_m=zmin, z_max_m=zmax, fx_px=fx, fy_px=fx, cx=47.5, cy=47.5,
              px_per_unit=96, consts=_consts(u_max=0.2))
    a = S.geometric_stack(d01, field="inverse", **kw).canais
    b = S.geometric_stack(d01, field="depth", **kw).canais
    assert not np.allclose(a[1], b[1])           # oclusão
    assert np.allclose(a[0], b[0])               # u não depende da escolha


def test_field_invalido_levanta():
    d01, zmin, zmax, fx = _cena_valida()
    with pytest.raises(ValueError):
        S.geometric_stack(d01, z_min_m=zmin, z_max_m=zmax, fx_px=fx, fy_px=fx,
                          cx=47.5, cy=47.5, px_per_unit=96, consts=_consts(),
                          field="disparidade")


def test_recorte_comuta_com_o_calculo_no_interior():
    """Calcular no mapa inteiro e recortar tem de bater com recortar e calcular,
    longe da borda. É o que garante que o crop aleatório do dataloader não muda
    o significado do canal, e só vale porque tau e px_per_unit vêm de fora."""
    d01, zmin, zmax, fx = _cena_valida(n=128, fx=1000.0)
    c = _consts(u_max=0.2)
    kw = dict(z_min_m=zmin, z_max_m=zmax, fx_px=fx, fy_px=fx, consts=c, px_per_unit=128)
    inteiro = S.geometric_stack(d01, cx=63.5, cy=63.5, **kw).canais
    y0, x0, L = 16, 24, 64
    recorte = S.geometric_stack(
        d01[y0:y0 + L, x0:x0 + L], cx=63.5 - x0, cy=63.5 - y0, **kw).canais
    a = inteiro[:5, y0 + 8:y0 + L - 8, x0 + 8:x0 + L - 8]
    b = recorte[:5, 8:L - 8, 8:L - 8]
    assert np.abs(a - b).max() < 1e-5, f"máx {np.abs(a-b).max():.2e}"


# ------------------------------------------------- gate por regiao util

def test_niveis_uteis_penaliza_a_sentinela_do_ceu():
    """A contagem no mapa inteiro nao vê o problema; a da regiao util vê.

    Cena tipica (z_max 12 m) contra cena com ceu no teto (z_max 10.000 m), com o
    MESMO plano de foco. A codificacao uint16 e a mesma; o que muda e quanto dela
    sobra para a cena util."""
    tipica = S.niveis_uteis(z_min_m=1.0, z_max_m=12.0, z_focus_m=3.0)
    ceu    = S.niveis_uteis(z_min_m=1.0, z_max_m=10000.0, z_focus_m=3.0)
    assert tipica == pytest.approx(65535, rel=1e-9), "cena tipica usa a faixa toda"
    assert ceu < 500, f"cena com ceu devia ficar com poucos niveis uteis, deu {ceu:.0f}"
    assert ceu < tipica / 100


def test_niveis_uteis_reproduz_a_medicao_da_rota_c():
    """Numeros medidos no job F0b, 2.932 amostras: as saturadas ficam com mediana
    de ~921 niveis uteis, as nao saturadas com 65.535."""
    assert S.niveis_uteis(1.0, 10.0, 3.0) == pytest.approx(65535, rel=1e-9)
    saturada = S.niveis_uteis(z_min_m=0.5, z_max_m=10000.0, z_focus_m=7.0)
    assert 500 < saturada < 1500


@pytest.mark.parametrize("z_min,z_max,z_focus", [(1.0, 0.5, 3.0), (1.0, 10.0, 0.0)])
def test_niveis_uteis_degenerado_da_zero(z_min, z_max, z_focus):
    assert S.niveis_uteis(z_min, z_max, z_focus) == 0.0
