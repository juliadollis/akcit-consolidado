"""A ponte entre `signals` e o crop/flip do dataloader.

O teste central deste arquivo é o do ESPELHAMENTO DAS NORMAIS. Ele existe
porque o erro que ele pega não levanta exceção, não muda shape, não muda range,
não aparece em nenhuma métrica de treino — e ensinaria a rede a associar
inclinação para a direita com inclinação para a esquerda em metade das
amostras, já que o hflip é aplicado a ~50% delas.

Os demais testes trancam as convenções que, se divergirem entre `dataloader` e
`signals`, quebram em silêncio: o `L` da normalização, a ordem entre recortar e
calcular, e a conferência de shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from geocond.constants import GeoConstants
from geocond.contrato import CANAIS
from geocond.dataloader import IDX_NX, espelhar_pilha, stack_para_amostra
from geocond.signals import comprimento_de_normalizacao, pilha_geometrica
from geocond.tests.geometrias import codifica_release, plano_inclinado, rampa


def _consts(**kw) -> GeoConstants:
    base = dict(
        tau_occlusion=1.0,
        u_max=1.0,
        s_max=1.0,
        k0_curvature=0.01,
        kt_max=8.0,
        smooth_sigma=0.5,
        min_niveis_para_segunda_ordem=256,
    )
    base.update(kw)
    return GeoConstants(**base)


def _cena_inclinada(n: int = 96):
    """Plano inclinado: tem gradiente NÃO NULO e de direção conhecida, que é
    o que o teste das normais precisa. Numa cena frontoparalela `n_x` vale 0,5
    em todo lugar e o erro do espelhamento ficaria invisível."""
    z = plano_inclinado(n, fx=float(n), normal=(0.6, 0.2, 1.0), dist_m=5.0)
    return codifica_release(z, focallength_px=float(n))


# ─────────────────────────────────────────────────────────────────────────────
# O teste que justifica o arquivo
# ─────────────────────────────────────────────────────────────────────────────


def test_espelhar_inverte_o_sinal_de_nx_e_nao_so_o_array():
    """`n` é campo VETORIAL: sob hflip a componente x aponta para o outro lado.

    Na codificação `(n_x+1)/2 ∈ [0,1]` isso é `1 - valor`, não `valor`. Um
    `[:, ::-1]` puro passaria por todos os outros testes deste arquivo.
    """
    disp01, amostra = _cena_inclinada()
    pilha = pilha_geometrica(disp01, amostra, _consts())
    canais = np.asarray(pilha.canais)

    espelhada = espelhar_pilha(canais)
    so_array = canais[:, :, ::-1]

    # todos os canais MENOS n_x são espelhamento puro
    for i, nome in enumerate(CANAIS):
        if i == IDX_NX:
            continue
        np.testing.assert_allclose(
            espelhada[i], so_array[i], atol=0, rtol=0,
            err_msg=f"canal {nome!r} não deveria mudar de valor sob hflip",
        )

    # n_x é 1 - espelhado
    np.testing.assert_allclose(espelhada[IDX_NX], 1.0 - so_array[IDX_NX], atol=1e-6)

    # e o erro é MENSURÁVEL: num plano inclinado o n_x está longe de 0,5, então
    # trocar `1-v` por `v` desloca o canal de forma grosseira. Se este delta for
    # pequeno, a cena de teste é fraca e o teste não discrimina.
    delta = float(np.abs(espelhada[IDX_NX] - so_array[IDX_NX]).mean())
    assert delta > 0.05, (
        f"cena de teste fraca: o hflip errado mudaria n_x em só {delta:.4f}. "
        "Use um plano com inclinação em x maior."
    )


def test_espelhar_duas_vezes_e_identidade():
    """`1 - (1 - v) = v` e `flip(flip(a)) = a`. Tranca a involução."""
    disp01, amostra = _cena_inclinada()
    canais = np.asarray(pilha_geometrica(disp01, amostra, _consts()).canais)
    np.testing.assert_allclose(espelhar_pilha(espelhar_pilha(canais)), canais, atol=1e-6)


def test_espelhar_recusa_pilha_com_numero_errado_de_canais():
    with pytest.raises(ValueError, match=r"6, H, W"):
        espelhar_pilha(np.zeros((3, 8, 8), dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# A integração: `stack_para_amostra` fala com `signals` de verdade
# ─────────────────────────────────────────────────────────────────────────────


def test_stack_para_amostra_produz_o_contrato():
    """Shape, dtype e range. É o que a fiação do `data.py` vai assumir."""
    n, lado = 96, 32
    disp01, amostra = _cena_inclinada(n)
    pilha = stack_para_amostra(disp01, (10, 12, 10 + lado, 12 + lado), False,
                               amostra, _consts())
    canais = np.asarray(pilha.canais)
    assert canais.shape == (len(CANAIS), lado, lado)
    assert canais.dtype == np.float32
    assert np.isfinite(canais).all()
    assert canais.min() >= 0.0 and canais.max() <= 1.0


def test_stack_recorta_o_mesmo_que_calcular_e_recortar():
    """Calcular no mapa INTEIRO e recortar depois — e não o contrário.

    A ordem importa: recortar antes faria a suavização e as diferenças finitas
    verem a borda do recorte como borda de cena, criando oclusão onde não há.
    Este teste tranca a ordem escolhida, comparando com o cálculo explícito.
    """
    n, lado = 96, 32
    caixa = (20, 8, 20 + lado, 8 + lado)
    disp01, amostra = _cena_inclinada(n)

    via_ponte = np.asarray(stack_para_amostra(disp01, caixa, False, amostra,
                                              _consts()).canais)
    inteiro = np.asarray(pilha_geometrica(disp01, amostra, _consts()).canais)
    x0, y0, x1, y1 = caixa
    np.testing.assert_allclose(via_ponte, inteiro[:, y0:y1, x0:x1], atol=1e-6)


def test_crop_e_flip_comutam_na_ordem_declarada():
    """Recortar e depois espelhar o recorte == o que a ponte faz."""
    n, lado = 96, 32
    caixa = (20, 8, 20 + lado, 8 + lado)
    disp01, amostra = _cena_inclinada(n)

    com_flip = np.asarray(stack_para_amostra(disp01, caixa, True, amostra,
                                             _consts()).canais)
    sem_flip = np.asarray(stack_para_amostra(disp01, caixa, False, amostra,
                                             _consts()).canais)
    np.testing.assert_allclose(com_flip, espelhar_pilha(sem_flip), atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# As convenções compartilhadas com `signals`
# ─────────────────────────────────────────────────────────────────────────────


def test_L_e_o_lado_menor_e_e_o_mesmo_nos_dois_modulos():
    """A ponte deixou de calcular `L` por conta própria: `signals` é a fonte.

    Este teste é o que sustenta essa centralização. Se a convenção mudar lá,
    ele quebra aqui, em vez de as duas cópias divergirem em silêncio. E cobra
    `min`, não `max`: `max` daria a uma imagem 16:9 um `L` 1,78× maior, e o
    `tau_occlusion` calibrado sob uma convenção não vale sob a outra.
    """
    z = rampa(64, 2.0, 8.0)
    disp01, amostra = codifica_release(z)
    # array quadrado: min == max, então não discrimina. Force retângulo.
    from geocond.contrato import AmostraGeo
    retangular = AmostraGeo(
        disparity_min=amostra.disparity_min,
        disparity_max=amostra.disparity_max,
        largura_px=128,
        altura_px=72,
        focallength_px=None,
    )
    assert comprimento_de_normalizacao(retangular) == 72.0


def test_amostra_que_descreve_outro_array_e_erro_nomeado():
    """O erro natural é passar `image_w`/`image_h` do `meta/` em vez das do
    array depois do resize. Isso põe um fator de escala por amostra na
    normalização do gradiente e desfaz a invariância a redimensionamento — em
    silêncio. Por isso é erro, e não ajuste."""
    disp01, amostra = _cena_inclinada(96)
    from geocond.contrato import AmostraGeo
    errada = AmostraGeo(
        disparity_min=amostra.disparity_min,
        disparity_max=amostra.disparity_max,
        largura_px=1561,   # a da imagem ORIGINAL, como vem no meta/
        altura_px=1024,
        focallength_px=None,
    )
    with pytest.raises(ValueError, match="array diferente"):
        stack_para_amostra(disp01, (0, 0, 32, 32), False, errada, _consts())


def test_caixa_fora_do_array_e_erro():
    disp01, amostra = _cena_inclinada(96)
    with pytest.raises(ValueError, match="fora do array"):
        stack_para_amostra(disp01, (80, 80, 200, 200), False, amostra, _consts())


def test_sem_focal_a_curvatura_sai_neutra_e_a_flag_cai():
    """O `meta/` do release NÃO traz `focallength_px` — conferido campo a campo.
    A ausência tem de viajar como ausência até o chamador."""
    n = 96
    z = plano_inclinado(n, fx=float(n), normal=(0.6, 0.2, 1.0), dist_m=5.0)
    disp01, amostra = codifica_release(z, focallength_px=None)
    pilha = stack_para_amostra(disp01, (10, 10, 42, 42), False, amostra, _consts())
    canais = np.asarray(pilha.canais)
    idx_k = CANAIS.index("K~")
    np.testing.assert_allclose(canais[idx_k], 0.5, atol=1e-6)
    assert pilha.segunda_ordem_valida is False


def test_diagnostico_traz_o_maximo_absoluto_do_recorte():
    """É o número que discrimina normalização por imagem.

    Correlação NÃO discrimina, porque é invariante a escala — foi exatamente
    assim que o mapa de defocus quebrado passou despercebido. O teste que
    discrimina é comparar o máximo absoluto ENTRE AMOSTRAS.
    """
    disp01, amostra = _cena_inclinada(96)
    d = stack_para_amostra(disp01, (10, 10, 42, 42), False, amostra, _consts()).diagnostico
    assert "max_abs_por_canal_recorte" in d
    assert len(d["max_abs_por_canal_recorte"]) == len(CANAIS)
    assert d["recorte_caixa"] == (10, 10, 42, 42)
