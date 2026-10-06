"""Testes da ponte com o dataloader. Sem rede, sem GPU.

O foco e a GEOMETRIA, nao o sinal: resize, crop e flip. O teste do flip existe
porque espelhar um campo vetorial nao e espelhar o array, e errar isso nao
levanta excecao nenhuma.
"""

from __future__ import annotations

import numpy as np
import pytest

from geo_cond.constants import GeoConstants
from geo_cond.dataloader import IDX_NX, GeoAmostra, espelhar_stack, stack_para_amostra
from geo_cond.signals import CANAIS
from geo_cond.tests import geometrias as G


def _consts(**kw):
    base = dict(tau_occlusion=1.0, u_max=1.0, s_max=1.0, k0_curvature=1.0,
                kt_max=5.0, z_percentile_max=99.5, smooth_sigma=0.5,
                min_quant_levels=256)
    base.update(kw)
    return GeoConstants(**base)


def _cena(n=128, fx=1000.0):
    """Esfera, com os escalares que o job F0b produziria."""
    z = G.esfera(n, fx, 2.0, 10.0)
    zmin, zmax = float(z.min()), float(z.max())
    d01 = (z - zmin) / (zmax - zmin)
    return d01, GeoAmostra(z_min_m=zmin, z_max_m=zmax, focallength_px=fx,
                           largura_px=n, altura_px=n)


# ------------------------------------------------------------------- flip

def test_espelhar_inverte_o_sinal_de_nx_e_so_dele():
    canais = np.random.default_rng(0).random((6, 8, 8)).astype(np.float32)
    esp = espelhar_stack(canais)
    for i, nome in enumerate(CANAIS):
        if i == IDX_NX:
            assert np.allclose(esp[i], 1.0 - canais[i][:, ::-1]), f"{nome} devia inverter"
        else:
            assert np.allclose(esp[i], canais[i][:, ::-1]), f"{nome} NAO devia inverter"


def test_espelhar_duas_vezes_e_identidade():
    canais = np.random.default_rng(1).random((6, 8, 8)).astype(np.float32)
    assert np.allclose(espelhar_stack(espelhar_stack(canais)), canais)


def test_flip_de_rampa_troca_o_lado_da_inclinacao():
    """O teste que pega o erro de verdade: numa rampa, a inclinacao aponta para
    um lado. Depois do flip tem de apontar para o OUTRO, ou seja n_x cruza 0,5
    para o lado oposto. Espelhar sem inverter o sinal deixaria n_x do mesmo lado."""
    n = 64
    z = G.rampa(n, 2.0, 8.0)
    zmin, zmax = float(z.min()), float(z.max())
    d01 = (z - zmin) / (zmax - zmin)
    am = GeoAmostra(z_min_m=zmin, z_max_m=zmax, focallength_px=800.0,
                    largura_px=n, altura_px=n)
    c = _consts(u_max=0.6, s_max=2.0)
    caixa = (8, 8, 8 + 32, 8 + 32)
    sem, _, _ = stack_para_amostra(d01, caixa, False, am, c)
    com, _, _ = stack_para_amostra(d01, caixa, True, am, c)
    a = float(np.median(sem[IDX_NX])) - 0.5
    b = float(np.median(com[IDX_NX])) - 0.5
    assert abs(a) > 1e-3, "a rampa tem de produzir n_x fora do neutro"
    assert a * b < 0, f"n_x devia trocar de lado: sem flip {a:+.4f}, com flip {b:+.4f}"
    assert abs(abs(a) - abs(b)) < 1e-6, "a magnitude tem de ser a mesma"


# ------------------------------------------------------------------- crop

def test_o_recorte_tem_o_tamanho_pedido_e_a_posicao_certa():
    d01, am = _cena()
    c = _consts(u_max=0.2)
    x0, y0, L = 24, 16, 64
    rec, _, _ = stack_para_amostra(d01, (x0, y0, x0 + L, y0 + L), False, am, c)
    assert rec.shape == (6, L, L)
    inteiro, _, _ = stack_para_amostra(d01, (0, 0, 128, 128), False, am, c)
    assert np.allclose(rec, inteiro[:, y0:y0 + L, x0:x0 + L])


def test_saida_e_contigua_finita_e_em_zero_um():
    d01, am = _cena()
    rec, _, _ = stack_para_amostra(d01, (10, 10, 74, 74), True, am, _consts(u_max=0.2))
    assert rec.flags["C_CONTIGUOUS"], "tensor nao contiguo custa uma copia no collate"
    assert np.all(np.isfinite(rec))
    assert rec.min() >= 0.0 and rec.max() <= 1.0


# ----------------------------------------------------------------- resize

def test_a_focal_e_corrigida_pela_escala_do_resize():
    """O defeito que a auditoria apontou: o dataloader reescala e ninguem
    corrige `fx`. Aqui a correcao e automatica, e o resultado tem de bater com o
    da resolucao original."""
    zg = G.esfera(256, 2000.0, 2.0, 10.0)
    zmin, zmax = float(zg.min()), float(zg.max())
    am = GeoAmostra(z_min_m=zmin, z_max_m=zmax, focallength_px=2000.0,
                    largura_px=256, altura_px=256)
    c = _consts(u_max=0.2, k0_curvature=0.05, kt_max=3.0)
    grande, _, _ = stack_para_amostra((zg - zmin) / (zmax - zmin),
                                      (64, 64, 192, 192), False, am, c)
    # mesma cena "redimensionada" para 128: a GeoAmostra ainda diz 256, entao a
    # ponte tem de deduzir escala 0.5 e usar fx = 1000
    zp = G.esfera(128, 1000.0, 2.0, 10.0)
    pequeno, _, _ = stack_para_amostra((zp - zmin) / (zmax - zmin),
                                       (32, 32, 96, 96), False, am, c)
    kg = float(np.median(grande[5, 32:-32, 32:-32]))
    kp = float(np.median(pequeno[5, 16:-16, 16:-16]))
    assert abs(kg - kp) < 0.01, f"curvatura normalizada: 256px={kg:.4f} vs 128px={kp:.4f}"


def test_amostra_com_dimensao_invalida_levanta():
    d01, _ = _cena()
    ruim = GeoAmostra(z_min_m=1.0, z_max_m=2.0, focallength_px=800.0,
                      largura_px=0, altura_px=128)
    with pytest.raises(ValueError):
        stack_para_amostra(d01, (0, 0, 64, 64), False, ruim, _consts())


# ------------------------------------------------------------- gate 2a ordem

def test_gate_de_quantizacao_propaga_pela_ponte():
    d01, am = _cena()
    grosso = np.rint(d01 * 20) / 20.0
    _, ok, niveis = stack_para_amostra(grosso, (0, 0, 64, 64), False, am,
                                       _consts(u_max=0.2))
    assert ok is False and niveis < 256
