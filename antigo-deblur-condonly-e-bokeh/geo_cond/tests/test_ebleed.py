"""Testes do E_bleed. Sem rede, sem GPU."""

from __future__ import annotations

import numpy as np
import pytest

from geo_cond.constants import GeoConstants
from geo_cond.ebleed import e_bleed
from geo_cond.tests import geometrias as G


def _consts(tau=1.0):
    return GeoConstants(tau_occlusion=tau, u_max=1.0, s_max=1.0, k0_curvature=1.0,
                        kt_max=5.0, z_percentile_max=99.5, smooth_sigma=0.5,
                        min_quant_levels=256)


def _cena_degrau(n=64):
    z = G.degrau(n, 1.0, 20.0)
    zmin, zmax = float(z.min()), float(z.max())
    return (z - zmin) / (zmax - zmin), zmin, zmax


def test_erro_so_na_borda_aparece_no_ebleed_e_nao_fora():
    d01, zmin, zmax = _cena_degrau()
    alvo = np.zeros((64, 64, 3)); pred = alvo.copy()
    pred[:, 30:34] = 1.0                     # erro colado na descontinuidade
    r = e_bleed(pred, alvo, d01, z_min_m=zmin, z_max_m=zmax,
                consts=_consts(tau=50.0), theta=0.3)
    assert r.n_pixels_borda > 0, "o degrau tem de produzir regiao de borda"
    assert r.e_bleed > r.e_fora * 5, f"dentro={r.e_bleed:.4f} fora={r.e_fora:.4f}"


def test_erro_so_longe_da_borda_nao_entra_no_ebleed():
    d01, zmin, zmax = _cena_degrau()
    alvo = np.zeros((64, 64, 3)); pred = alvo.copy()
    pred[:, :8] = 1.0                        # erro no canto, longe do degrau
    r = e_bleed(pred, alvo, d01, z_min_m=zmin, z_max_m=zmax,
                consts=_consts(tau=50.0), theta=0.3)
    assert r.e_bleed < 1e-9 and r.e_fora > 0.05


def test_predicao_perfeita_zera_os_dois():
    d01, zmin, zmax = _cena_degrau()
    a = np.random.default_rng(0).random((64, 64, 3))
    r = e_bleed(a, a.copy(), d01, z_min_m=zmin, z_max_m=zmax,
                consts=_consts(tau=50.0))
    assert r.e_bleed == pytest.approx(0.0) and r.e_fora == pytest.approx(0.0)


def test_cena_plana_nao_tem_regiao_de_borda():
    """Sem descontinuidade, B_theta e vazio e o E_bleed nao e definido. Melhor
    devolver nan explicito do que um numero que parece medida."""
    n = 32
    z = G.frontoparalelo(n, 5.0)
    d01 = np.zeros((n, n))
    a = np.zeros((n, n, 3))
    r = e_bleed(a, a, d01, z_min_m=4.0, z_max_m=6.0, consts=_consts())
    assert r.n_pixels_borda == 0 and np.isnan(r.e_bleed)


def test_depth_em_resolucao_diferente_levanta():
    """Se o depth nao casar com a imagem, a borda cai no lugar errado e o numero
    sai plausivel mas sem sentido. Melhor abortar."""
    with pytest.raises(ValueError):
        e_bleed(np.zeros((64, 64, 3)), np.zeros((64, 64, 3)), np.zeros((32, 32)),
                z_min_m=1.0, z_max_m=2.0, consts=_consts())
