"""Testes do E_bleed. Sem rede, sem GPU, sem torch.

Os mapas `O` são construídos à mão aqui dentro, e não a partir de uma
profundidade sintética, por uma razão de contrato: a função RECEBE `O` — o mesmo
mapa e o mesmo `tau` do condicionamento — justamente para que ninguém possa
medir uma região diferente da que a perda supervisiona. Reconstruir `O` no teste
reintroduziria o acoplamento que a assinatura elimina.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from geocond.ebleed import e_bleed, erodir  # noqa: E402

N = 64
THETA = 0.3


def _O_faixa(largura=4, col=30) -> np.ndarray:
    """`O` com uma faixa vertical de borda — o análogo de um degrau de
    profundidade depois do suavizador."""
    O = np.zeros((N, N))
    O[:, col:col + largura] = 0.9
    return O


def _img(v=0.0) -> np.ndarray:
    return np.full((N, N, 3), float(v))


# ---------------------------------------------------- o que a métrica mede

def test_vazamento_injetado_so_na_borda_sobe_o_ebleed_e_nao_mexe_no_efora():
    O = _O_faixa()
    alvo = _img(0.0)
    limpo = _img(0.0)

    r0 = e_bleed(limpo, alvo, O, theta=THETA)
    assert r0["e_bleed"] == pytest.approx(0.0) and r0["e_fora"] == pytest.approx(0.0)

    vazou = alvo.copy()
    vazou[:, 30:34] = 0.5                       # erro EXATAMENTE sobre a faixa
    r = e_bleed(vazou, alvo, O, theta=THETA)

    assert r["n_pixels_borda"] == 4 * N
    assert r["e_bleed"] == pytest.approx(0.5)
    assert r["e_fora"] == pytest.approx(0.0), "o erro não encostou fora da borda"
    assert r["frac_borda"] == pytest.approx(4 / N)


def test_erro_so_longe_da_borda_nao_entra_no_ebleed():
    O = _O_faixa()
    alvo = _img(0.0)
    pred = alvo.copy()
    pred[:, :8] = 1.0                           # erro no canto, longe do degrau
    r = e_bleed(pred, alvo, O, theta=THETA)
    assert r["e_bleed"] == pytest.approx(0.0)
    assert r["e_fora"] > 0.1


def test_predicao_identica_ao_alvo_da_zero_nos_dois():
    O = _O_faixa()
    a = np.random.default_rng(0).random((N, N, 3))
    r = e_bleed(a, a.copy(), O, theta=THETA)
    assert r["e_bleed"] == 0.0 and r["e_fora"] == 0.0
    # 0/0 não é 1: sem erro nenhum a razão não é definida, e um "1,000" aqui
    # seria lido como "borda igual ao resto", que é uma afirmação diferente.
    assert math.isnan(r["razao"])


# -------------------------------- por que E_fora tem de ser reportado SEMPRE

def test_o_caso_A_linha_melhora_relativa_com_piora_absoluta_fica_visivel():
    """O modo de falha que derrubou a campanha v1, reproduzido.

    O A' baixou a RAZÃO borda/fora de 1,427 para 1,286 enquanto o erro absoluto
    piorava em toda parte (E_fora +31%). Se a tabela tivesse só a coluna do
    E_bleed — ou só a razão — isso teria passado por melhora.
    """
    O = _O_faixa()
    alvo = _img(0.0)

    base = alvo.copy()
    base[:, 30:34] = 0.1427                     # borda
    base[:, :30] = 0.1                          # fora
    base[:, 34:] = 0.1
    r_base = e_bleed(base, alvo, O, theta=THETA)

    pior = alvo.copy()
    pior[:, 30:34] = 0.1686                     # borda: erro ABSOLUTO maior
    pior[:, :30] = 0.131                        # fora: +31%
    pior[:, 34:] = 0.131
    r_pior = e_bleed(pior, alvo, O, theta=THETA)

    assert r_base["razao"] == pytest.approx(1.427, abs=1e-3)
    assert r_pior["razao"] == pytest.approx(1.287, abs=2e-3)
    assert r_pior["razao"] < r_base["razao"], "a razão melhorou…"
    assert r_pior["e_bleed"] > r_base["e_bleed"], "…e o erro na borda PIOROU"
    assert r_pior["e_fora"] == pytest.approx(r_base["e_fora"] * 1.31, rel=1e-3)


# ------------------------------------------------------------------- erosão

def test_erodir_nao_come_uma_mascara_cheia():
    """Padding por réplica: o quadro da imagem não é fronteira de invalidez."""
    m = np.ones((8, 8), dtype=bool)
    assert erodir(m, 1).all() and erodir(m, 3).all()


def test_erodir_alarga_o_buraco_em_um_pixel_por_iteracao():
    m = np.ones((9, 9), dtype=bool)
    m[4, 4] = False
    e1 = erodir(m, 1)
    assert int((~e1).sum()) == 9, "3×3 em volta do buraco"
    e2 = erodir(m, 2)
    assert int((~e2).sum()) == 25, "5×5"


def test_a_fronteira_do_buraco_invalido_nao_entra_como_borda():
    """A fronteira de uma região sem profundidade produz um degrau ARTIFICIAL em
    `O`. Sem erodir a validade, esse anel entraria em B e a métrica passaria a
    medir buraco de reprojeção, não oclusão de cena — é o espírito do
    `boundary_fscore` de depth-riemannian."""
    O = np.zeros((N, N))
    valido = np.ones((N, N), dtype=bool)
    valido[20:30, 20:30] = False
    # o anel de 1 px em volta do buraco acende em O, como um degrau falso
    O[19:31, 19:31] = 0.9
    O[20:30, 20:30] = 0.0

    alvo = _img(0.0)
    pred = alvo.copy()
    pred[19:31, 19:31] = 1.0                    # "erro" grande no anel falso

    sem = e_bleed(pred, alvo, O, theta=THETA, valido=valido, raio_erosao=0)
    com = e_bleed(pred, alvo, O, theta=THETA, valido=valido, raio_erosao=1)

    assert sem["n_pixels_borda"] > 0
    assert com["n_pixels_borda"] == 0, (
        f"a erosão tinha de eliminar o anel falso; sobraram {com['n_pixels_borda']} px"
    )
    assert math.isnan(com["e_bleed"])
    # e o anel não migra para "fora": ele sai das DUAS regiões
    assert com["n_pixels_validos"] == com["n_pixels_borda"] + com["n_pixels_fora"]
    assert com["e_fora"] == pytest.approx(0.0)


def test_pixels_invalidos_nao_entram_em_nenhuma_das_duas_regioes():
    O = _O_faixa()
    valido = np.ones((N, N), dtype=bool)
    valido[:, 50:] = False
    alvo = _img(0.0)
    pred = alvo.copy()
    pred[:, 50:] = 1.0                          # erro inteiro dentro do inválido
    r = e_bleed(pred, alvo, O, theta=THETA, valido=valido)
    assert r["e_fora"] == pytest.approx(0.0), "erro em pixel inválido vazou para E_fora"
    assert r["n_pixels_validos"] < N * N


# ------------------------------------------------------ região vazia e limites

def test_cena_sem_borda_da_nan_e_nao_zero():
    """Sem descontinuidade, B é vazio e o E_bleed não é definido. `nan` explícito
    é melhor do que um 0,0 que se parece com "erro nenhum na borda"."""
    O = np.zeros((N, N))
    a = _img(0.0)
    r = e_bleed(a, a, O, theta=THETA)
    assert r["n_pixels_borda"] == 0
    assert math.isnan(r["e_bleed"]) and math.isnan(r["razao"])
    assert r["frac_borda"] == pytest.approx(0.0)


def test_cena_toda_borda_da_nan_no_e_fora():
    O = np.full((N, N), 0.9)
    a = _img(0.0)
    r = e_bleed(a, a, O, theta=THETA)
    assert r["n_pixels_fora"] == 0 and math.isnan(r["e_fora"])
    assert r["frac_borda"] == pytest.approx(1.0)


def test_limiar_estrito_O_igual_a_theta_fica_de_fora():
    O = np.full((N, N), THETA)
    a = _img(0.0)
    r = e_bleed(a, a, O, theta=THETA)
    assert r["n_pixels_borda"] == 0, "o teste é O > θ, não O >= θ"


# ----------------------------------------------------------------- validações

def test_depth_em_resolucao_diferente_levanta():
    """Se `O` não casar com a imagem, a borda cai no lugar errado e o número sai
    plausível mas mede outra região. Melhor abortar."""
    with pytest.raises(ValueError):
        e_bleed(np.zeros((N, N, 3)), np.zeros((N, N, 3)), np.zeros((32, 32)),
                theta=THETA)


def test_shapes_diferentes_levantam():
    with pytest.raises(ValueError):
        e_bleed(np.zeros((N, N, 3)), np.zeros((N, N, 1)), np.zeros((N, N)),
                theta=THETA)


def test_theta_fora_de_zero_um_levanta():
    with pytest.raises(ValueError):
        e_bleed(_img(), _img(), np.zeros((N, N)), theta=1.0)
    with pytest.raises(ValueError):
        e_bleed(_img(), _img(), np.zeros((N, N)), theta=-0.1)


def test_theta_nao_tem_default():
    """O θ tem de ser o MESMO da perda ponderada. Um default aqui convidaria a
    medir uma região e supervisionar outra."""
    with pytest.raises(TypeError):
        e_bleed(_img(), _img(), np.zeros((N, N)))      # type: ignore[call-arg]


def test_nan_na_entrada_levanta_em_vez_de_contaminar_a_media():
    a = _img(0.0)
    b = _img(0.0)
    b[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        e_bleed(b, a, _O_faixa(), theta=THETA)


# ------------------------------------------------------------ formas aceitas

def test_imagem_monocromatica_funciona():
    O = _O_faixa()
    alvo = np.zeros((N, N))
    pred = alvo.copy()
    pred[:, 30:34] = 0.5
    r = e_bleed(pred, alvo, O, theta=THETA)
    assert r["e_bleed"] == pytest.approx(0.5) and r["e_fora"] == pytest.approx(0.0)


def test_O_com_eixo_de_canal_sobrando_e_aceito():
    O = _O_faixa()[..., None]                   # (H, W, 1)
    a = _img(0.0)
    r = e_bleed(a, a, O, theta=THETA)
    assert r["n_pixels_borda"] == 4 * N


def test_aceita_tensor_torch_se_houver():
    torch = pytest.importorskip("torch")
    O = torch.from_numpy(_O_faixa())
    alvo = torch.zeros(N, N, 3)
    pred = alvo.clone()
    pred[:, 30:34] = 0.5
    r = e_bleed(pred, alvo, O, theta=THETA)
    assert r["e_bleed"] == pytest.approx(0.5)


def test_chaves_do_contrato_estao_todas_la():
    r = e_bleed(_img(), _img(), _O_faixa(), theta=THETA)
    for k in ("e_bleed", "e_fora", "razao", "n_pixels_borda", "frac_borda"):
        assert k in r, f"chave do contrato ausente: {k}"
