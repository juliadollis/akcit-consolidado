"""Testes dos sinais geométricos. Sem rede, sem GPU.

A régua é geometria de resposta FECHADA, não regressão contra a saída atual:
esfera de raio R tem K = 1/R², plano e cilindro têm K = 0, plano frontoparalelo
tem gradiente nulo. Um teste de regressão congelaria o erro junto com o acerto —
e foi exatamente assim que a formulação de Monge de
`geometry_maps.principal_curvatures`, que devolve ~10 onde o valor verdadeiro é
0,25, sobreviveu tempo suficiente para gerar as figuras do documento de
proposta.

Três famílias de teste aqui NÃO existiam na `geo_cond_v1`, e são as que guardam
o contrato novo (`metric_disparity_official_v1`):

  * `test_contrato_*`   — que `u` é linear na DISPARIDADE, não em 1/z;
  * `test_resolucao_*`  — incluindo o par que erra DE PROPÓSITO;
  * `test_sem_focal_*`  — que `K~` sai neutro em vez de inventar `fx`.
"""

from __future__ import annotations

import numpy as np
import pytest

from geocond.constants import GeoConstants
from geocond.contrato import CANAIS, AmostraGeo
from geocond import signals as S
from geocond.tests import geometrias as G


# =============================================================================
# Andaimes
# =============================================================================

def _consts(**kw) -> GeoConstants:
    """Constantes de TESTE, não de treino.

    São escolhidas para deixar cada canal na região informativa da sua
    normalização (nada saturado por acidente), não medidas no conjunto de
    treino. `GeoConstants` não tem default de propósito — ver `constants.py` —
    então o valor tem de aparecer explícito em algum lugar, e aqui é o lugar.

    `k0_curvature = 0,01` e `kt_max = 8` dão folga confortável para a esfera de
    R = 2 m: log1p(0,25/0,01) = 3,26, bem dentro de ±8, logo o canal `K~` não
    satura e é invertível para conferir o K original.
    """
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


def _miolo(a: np.ndarray, m: int = 12) -> np.ndarray:
    """Interior, longe da borda: fora dela as diferenças são unilaterais e a
    suavização replica, então o valor ali não é a fórmula que se quer testar."""
    return a[m:-m, m:-m]


def _K_do_canal(ch_K: np.ndarray, consts: GeoConstants) -> np.ndarray:
    """Inverte a normalização do canal `K~` de volta para K em 1/m².

    Existe para que os testes de ponta a ponta possam cobrar o MESMO número
    fechado (1/R²) que os testes da função de baixo nível cobram, em vez de
    cobrarem um valor normalizado que ninguém sabe conferir de cabeça.
    """
    Kt = (np.asarray(ch_K, dtype=np.float64) - 0.5) * (2.0 * consts.kt_max)
    return np.sign(Kt) * np.expm1(np.abs(Kt)) * consts.k0_curvature


# =============================================================================
# Curvatura — o teste que separa a formulação certa da errada
# =============================================================================

@pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0])
def test_esfera_tem_curvatura_um_sobre_r2(R):
    """K = 1/R², erro < 2%, de R = 0,5 m a R = 5 m.

    A razão d/R é mantida fixa em 5 para que a esfera ocupe a MESMA fração do
    quadro em todos os raios: assim o teste varia só a escala métrica, que é o
    que se quer cobrar, e não a amostragem.

    Este é o teste do `RETESTE_CURVATURA.md`: a formulação de Monge devolve ~10
    aqui (fator 40 para R = 2 m), a retroprojetada devolve 0,25.
    """
    d, fx, n = R * 5.0, 1600.0, 96
    z = G.esfera(n, fx, R, d)
    K = S.curvatura_gaussiana_retroprojetada(z, fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5)
    esperado = 1.0 / R ** 2
    obtido = float(np.median(_miolo(K)))
    assert abs(obtido - esperado) / esperado < 0.02, (
        f"K da esfera R={R} m: obtido {obtido:.6g}, esperado {esperado:.6g} "
        f"(erro {abs(obtido - esperado) / esperado:.2%})"
    )


def test_plano_inclinado_tem_curvatura_nula():
    fx, n = 800.0, 96
    z = G.plano_inclinado(n, fx, (0.3, 0.1, 1.0), 5.0)
    K = S.curvatura_gaussiana_retroprojetada(z, fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5)
    assert float(np.abs(_miolo(K)).max()) < 1e-3


def test_cilindro_tem_curvatura_nula():
    """Superfície desenvolvível: curva numa direção, K = 0 mesmo assim.
    É o teste que separa curvatura GAUSSIANA de curvatura MÉDIA — uma
    implementação que devolvesse H passaria no plano e falharia aqui."""
    fx, n = 1600.0, 96
    z = G.cilindro(n, fx, 1.0, 6.0)
    K = S.curvatura_gaussiana_retroprojetada(z, fx, fx, (n - 1) / 2, (n - 1) / 2, 0.5)
    assert float(np.abs(_miolo(K)).max()) < 5e-3


def test_esfera_pela_pilha_completa_com_quantizacao_uint16():
    """O mesmo 1/R², mas entrando pelo caminho de produção: disparidade
    quantizada em uint16, `AmostraGeo`, `pilha_geometrica`, canal normalizado e
    desnormalizado de volta.

    A mesma tolerância de 2%, e isso é um resultado, não uma escolha: a esfera
    de R = 2 m a 10 m ocupa só 0,0090 1/m de faixa de disparidade, o passo do
    uint16 vira 1,4e-7 1/m, e esse degrau entra na derivada SEGUNDA amplificado.
    Medido: 0,16% de erro contra os 0,03% do caminho sem quantização. Ou seja o
    uint16 custa um fator 5 no erro e continua uma ordem de grandeza abaixo do
    limiar — a mediana no miolo absorve o ruído, que é zero-média.
    """
    R, d, fx, n = 2.0, 10.0, 400.0, 96
    c = _consts()
    d01, am = G.codifica_release(G.esfera(n, fx, R, d), focallength_px=fx)
    out = S.pilha_geometrica(d01, am, c)
    assert out.segunda_ordem_valida is True
    K = _K_do_canal(out.canais[CANAIS.index("K~")], c)
    obtido = float(np.median(_miolo(K)))
    assert abs(obtido - 0.25) / 0.25 < 0.02, f"K de ponta a ponta = {obtido:.4f}"


# =============================================================================
# Primeira ordem
# =============================================================================

def test_frontoparalelo_zera_oclusao_e_da_direcao_neutra():
    """Parede lisa: ‖∇f‖ = 0, logo O = 0 e a normal é (0,0,1) — que depois do
    mapeamento (n+1)/2 é exatamente (0,5; 0,5). O canal de área também vai a 0.

    A `faixa` é forçada porque a disparidade é constante e o min-max da cena
    seria degenerado; no release os extremos são da CENA inteira, que quase
    nunca é uma parede lisa e nada mais."""
    n = 64
    c = _consts()
    d01, am = G.codifica_release(G.frontoparalelo(n, 5.0), faixa=(0.1, 0.3))
    out = S.pilha_geometrica(d01, am, c)
    ch = {nome: out.canais[i] for i, nome in enumerate(CANAIS)}
    assert float(ch["O"].max()) == 0.0
    assert np.allclose(ch["n_x"], 0.5, atol=1e-7)
    assert np.allclose(ch["n_y"], 0.5, atol=1e-7)
    assert float(ch["s"].max()) == 0.0
    # u = disp/u_max = 0,2/1,0 — a disparidade métrica reconstruída, constante.
    assert np.allclose(ch["u"], 0.2, atol=1e-6)


def test_degrau_satura_a_oclusao_exatamente_na_borda():
    """O = 1 nas duas colunas que a diferença central toca, e 0 em todo o resto.

    Com o degrau 1 m → 20 m e L = 64, ‖∇u‖ na borda vale
    (1 − 0,05)·0,5·64 = 30,4, contra tau = 1,0: satura com folga de 30x. Longe
    da borda o campo é constante e o gradiente é exatamente zero, não
    aproximadamente — por isso a comparação é `== 0.0` e não `< eps`."""
    n = 64
    c = _consts()
    d01, am = G.codifica_release(G.degrau(n, 1.0, 20.0))
    out = S.pilha_geometrica(d01, am, c)
    O = out.canais[CANAIS.index("O")]
    col = n // 2
    assert float(O[:, col - 1:col + 1].min()) == 1.0, "a borda tem de saturar"
    assert float(O[:, : col - 1].max()) == 0.0, "à esquerda da borda tem de ser 0"
    assert float(O[:, col + 1:].max()) == 0.0, "à direita da borda tem de ser 0"
    assert out.diagnostico["frac_O_saturado"] == pytest.approx(2.0 / n, rel=1e-9)


def test_mascara_de_validade_nao_deixa_a_fronteira_virar_a_maior_oclusao():
    """Sem a erosão, o degrau artificial na borda de uma região inválida produz
    o maior ‖∇f‖ do mapa e vira "a maior oclusão da imagem" — um artefato de
    máscara supervisionando o modelo como se fosse geometria."""
    n = 32
    mag = np.ones((n, n))
    valida = np.ones((n, n), dtype=bool)
    valida[:, :8] = False
    O = S.oclusao(mag, tau=1.0, valida=valida)
    assert float(O[:, :9].max()) == 0.0, "a fronteira do inválido tem de ser erodida"
    assert float(O[:, 12:].min()) == 1.0, "longe dela, nada muda"


def test_elemento_de_area_e_normais_sao_a_polar_do_gradiente():
    """s, n_x, n_y são uma REPARAMETRIZAÇÃO de ∇f: 2 graus de liberdade em 3
    canais. O teste confere a identidade que sustenta a afirmação, porque o
    texto do paper não pode dizer "informação complementar"."""
    rng = np.random.default_rng(0)
    fx = rng.normal(size=(16, 16)) * 3.0
    fy = rng.normal(size=(16, 16)) * 3.0
    s, nx, ny = S.elemento_de_area_e_normais(fx, fy)
    q = fx * fx + fy * fy
    assert np.allclose(np.exp(2.0 * s) - 1.0, q)
    assert np.allclose(nx * np.sqrt(1.0 + q), -fx)
    assert np.allclose(nx ** 2 + ny ** 2 + 1.0 / (1.0 + q), 1.0)


# =============================================================================
# Contrato novo: `u` é linear na DISPARIDADE, não em 1/z
# =============================================================================

def test_contrato_u_e_linear_na_disparidade_reconstruida():
    """d01 -> disp = disparity_min + d01·(max − min), e `u` é isso dividido por
    `u_max`. Sem inversão de profundidade no caminho.

    A segunda metade do teste é a que importa: mostra que a leitura ANTIGA do
    mesmo array (d01 = profundidade métrica normalizada, u = 1/z) daria outro
    canal. Se as duas leituras coincidissem, a mudança de contrato não teria
    consequência e não precisaria de teste."""
    n = 64
    d_min, d_max = 0.05, 1.0
    d01 = np.tile(np.linspace(0.0, 1.0, n), (n, 1))
    am = AmostraGeo(disparity_min=d_min, disparity_max=d_max,
                    largura_px=n, altura_px=n, focallength_px=None)
    c = _consts(u_max=1.0)
    u = S.pilha_geometrica(d01, am, c).canais[CANAIS.index("u")]

    disp = d_min + d01 * (d_max - d_min)
    assert np.allclose(u, disp, atol=1e-6), "u tem de ser a disparidade métrica"

    # Linearidade exata em d01: o coeficiente angular é (max − min) em todo o
    # domínio, com desvio nulo. 1/z NÃO tem essa propriedade.
    du = np.diff(u[0].astype(np.float64))
    assert float(du.std()) < 1e-6 and du.mean() == pytest.approx(
        (d_max - d_min) / (n - 1), rel=1e-4)

    # A leitura antiga: d01 seria Z normalizado entre z_min = 1/d_max e
    # z_max = 1/d_min, e u seria 1/Z. Mesmo array, outro canal.
    z_antigo = (1.0 / d_max) + d01 * ((1.0 / d_min) - (1.0 / d_max))
    u_antigo = 1.0 / z_antigo
    erro = float(np.abs(u_antigo - disp).max() / d_max)
    assert erro > 0.3, (
        "as duas leituras do mesmo PNG têm de divergir; se não divergissem, a "
        f"invalidação da campanha antiga não faria sentido (divergência {erro:.1%})"
    )


def test_contrato_z_e_o_inverso_da_disparidade():
    d01 = np.array([[0.0, 0.5, 1.0]])
    am = AmostraGeo(disparity_min=0.1, disparity_max=1.1,
                    largura_px=3, altura_px=1)
    disp = S.disparidade_metrica(d01, am)
    assert np.allclose(disp, [[0.1, 0.6, 1.1]])
    assert np.allclose(S.profundidade_metrica(disp), [[10.0, 1 / 0.6, 1 / 1.1]])


@pytest.mark.parametrize("d_min,d_max", [(1.0, 0.5), (0.5, 0.5), (-0.1, 1.0)])
def test_faixa_de_disparidade_invalida_levanta(d_min, d_max):
    am = AmostraGeo(disparity_min=d_min, disparity_max=d_max,
                    largura_px=4, altura_px=4)
    with pytest.raises(ValueError):
        S.disparidade_metrica(np.zeros((4, 4)), am)


# =============================================================================
# field="inverse" contra "depth": a física, medida
# =============================================================================

def test_field_inverse_ordena_os_degraus_e_depth_nao():
    """Dois degraus, 1 m → 20 m e 20 m → 40 m.

    O segundo é um salto MAIOR em profundidade (20 m contra 19 m) e um salto
    muito MENOR em disparidade (0,025 contra 0,95 1/m). Opticamente o segundo é
    quase invisível: o raio do CoC é linear na disparidade, então 0,025 1/m de
    salto quase não muda o borrão, enquanto 0,95 muda tudo.

    Só `field="inverse"` ordena os dois na ordem óptica. Com `field="depth"` a
    razão fica ABAIXO de 1, ou seja o degrau irrelevante domina o canal — e é
    esse o motivo de "depth" existir apenas como ablação.
    """
    n = 64
    d01_a, am_a = G.codifica_release(G.degrau(n, 1.0, 20.0))
    d01_b, am_b = G.codifica_release(G.degrau(n, 20.0, 40.0))

    def pico(d01, am, field):
        disp = S.disparidade_metrica(d01, am)
        f = disp if field == "inverse" else S.profundidade_metrica(disp)
        gx, gy = S.gradiente_normalizado(f, S.comprimento_de_normalizacao(am))
        return float(np.hypot(gx, gy).max())

    razao_u = pico(d01_a, am_a, "inverse") / pico(d01_b, am_b, "inverse")
    razao_z = pico(d01_a, am_a, "depth") / pico(d01_b, am_b, "depth")

    # Previsão fechada: (1 − 1/20)/(1/20 − 1/40) = 0,95/0,025 = 38, e
    # (20 − 1)/(40 − 20) = 19/20 = 0,95.
    assert razao_u == pytest.approx(38.0, rel=0.02), f"razão em u = {razao_u:.3f}"
    assert razao_z == pytest.approx(0.95, rel=0.02), f"razão em z = {razao_z:.4f}"
    assert razao_u > 1.0 and razao_z < 1.0, (
        f"ordenação: em u o degrau perto domina ({razao_u:.1f}x); "
        f"em z o degrau longe domina ({1 / razao_z:.2f}x), que é a ordem errada"
    )


def test_field_muda_a_oclusao_mas_nao_o_canal_u():
    """Se os dois `field` dessem o mesmo, a escolha não seria ablacionável.
    E `u` não pode depender da escolha: ele é o dado, não uma derivada dele."""
    n = 96
    c = _consts(tau_occlusion=50.0)
    d01, am = G.codifica_release(G.rampa(n, 2.0, 6.0), focallength_px=800.0)
    a = S.pilha_geometrica(d01, am, c, field="inverse").canais
    b = S.pilha_geometrica(d01, am, c, field="depth").canais
    assert not np.allclose(a[CANAIS.index("O")], b[CANAIS.index("O")])
    assert np.allclose(a[CANAIS.index("u")], b[CANAIS.index("u")])


def test_field_invalido_levanta():
    d01, am = G.codifica_release(G.rampa(16, 2.0, 6.0))
    with pytest.raises(ValueError):
        S.pilha_geometrica(d01, am, _consts(), field="disparidade")


# =============================================================================
# Invariância a resolução — e o par que erra de propósito
# =============================================================================

def _esfera_em(n: int, fx: float, foco=True):
    return G.codifica_release(
        G.esfera(n, fx, 2.0, 10.0), focallength_px=(fx if foco else None)
    )


def test_resolucao_primeira_ordem_e_invariante_quando_L_acompanha():
    """Mesma cena a 96 e a 192 px, com `largura_px`/`altura_px` do próprio
    array: mesmo O e mesmo ‖∇u‖.

    Sem isso, treinar a 512² e inferir com tiling na resolução original mudaria
    o SIGNIFICADO do canal sem mudar nada que uma métrica de correlação
    detectasse. Note que a quantização uint16 não atrapalha aqui: o passo de
    1,4e-7 1/m contribui ~7e-6 para o gradiente contra um sinal de ~2e-2, três
    ordens de grandeza abaixo. É a derivada SEGUNDA que sofre, não a primeira.
    """
    c = _consts(tau_occlusion=0.1)
    o1 = S.pilha_geometrica(*_esfera_em(96, 800.0), c).canais[CANAIS.index("O")]
    o2 = S.pilha_geometrica(*_esfera_em(192, 1600.0), c).canais[CANAIS.index("O")]
    a, b = float(np.median(_miolo(o1))), float(np.median(_miolo(o2, 24)))
    assert abs(a - b) / a < 0.02, f"O a 96px={a:.6g} vs a 192px={b:.6g}"

    d01a, ama = _esfera_em(96, 800.0)
    d01b, amb = _esfera_em(192, 1600.0)
    g1 = np.hypot(*S.gradiente_normalizado(
        S.disparidade_metrica(d01a, ama), S.comprimento_de_normalizacao(ama)))
    g2 = np.hypot(*S.gradiente_normalizado(
        S.disparidade_metrica(d01b, amb), S.comprimento_de_normalizacao(amb)))
    ga, gb = float(np.median(_miolo(g1))), float(np.median(_miolo(g2, 24)))
    assert abs(ga - gb) / ga < 0.02, f"‖∇u‖ a 96px={ga:.6g} vs a 192px={gb:.6g}"


def test_resolucao_curvatura_e_invariante_quando_fx_acompanha():
    c = _consts()
    K1 = _K_do_canal(
        S.pilha_geometrica(*_esfera_em(96, 400.0), c).canais[CANAIS.index("K~")], c)
    K2 = _K_do_canal(
        S.pilha_geometrica(*_esfera_em(192, 800.0), c).canais[CANAIS.index("K~")], c)
    a, b = float(np.median(_miolo(K1))), float(np.median(_miolo(K2, 24)))
    assert abs(a - b) / a < 0.02, f"K a 96px={a:.4f} vs a 192px={b:.4f}"


def test_resolucao_SEM_corrigir_pelo_resize_da_errado_de_proposito():
    """O contraste dos dois testes acima, e a razão de eles existirem.

    Aqui o resize é feito e os escalares NÃO são corrigidos — o defeito que o
    plano descreve, e o mais fácil de cometer, porque o resultado continua sendo
    um mapa bonito em [0,1]. O teste EXIGE que dê errado; se um dia passar a
    "funcionar", é porque a normalização deixou de depender dos escalares e a
    invariância virou coincidência.

    Dois erros independentes, um por convenção de coordenada:
      (a) `largura_px`/`altura_px` herdados da resolução antiga  -> ∇ erra 2x;
      (b) `focallength_px` herdada da resolução antiga           -> K erra 93%.
    """
    c = _consts(tau_occlusion=0.1)
    d01, am = _esfera_em(192, 800.0)          # 192 px, fx=800 é o valor CERTO aqui

    # (a) escalares de imagem congelados em 96 px.
    am_L_errado = AmostraGeo(am.disparity_min, am.disparity_max, 96, 96, am.focallength_px)
    g_certo = np.hypot(*S.gradiente_normalizado(
        S.disparidade_metrica(d01, am), S.comprimento_de_normalizacao(am)))
    g_errado = np.hypot(*S.gradiente_normalizado(
        S.disparidade_metrica(d01, am_L_errado),
        S.comprimento_de_normalizacao(am_L_errado)))
    razao = float(np.median(_miolo(g_certo, 24)) / np.median(_miolo(g_errado, 24)))
    assert abs(razao - 2.0) < 0.01, (
        f"sem corrigir L, o gradiente tem de sair por um fator 2; deu {razao:.3f}"
    )
    o_certo = S.pilha_geometrica(d01, am, c).canais[CANAIS.index("O")]
    o_errado = S.pilha_geometrica(d01, am_L_errado, c).canais[CANAIS.index("O")]
    assert not np.allclose(o_certo, o_errado, rtol=0.05)

    # (b) focal congelada em 96 px, com o array a 192.
    am_fx_errado = AmostraGeo(am.disparity_min, am.disparity_max,
                              am.largura_px, am.altura_px, 400.0)
    K_certo = float(np.median(_miolo(_K_do_canal(
        S.pilha_geometrica(d01, am, c).canais[CANAIS.index("K~")], c), 24)))
    K_errado = float(np.median(_miolo(_K_do_canal(
        S.pilha_geometrica(d01, am_fx_errado, c).canais[CANAIS.index("K~")], c), 24)))
    assert abs(K_certo - 0.25) / 0.25 < 0.02, f"o caso certo deu K={K_certo:.4f}"
    assert abs(K_errado - 0.25) / 0.25 > 0.5, (
        f"com fx sem corrigir, K TEM de sair errado (medido: 0,0179, "
        f"93% abaixo de 0,25); deu {K_errado:.4f} "
        "(perto demais de 0,25 — a guarda deixou de guardar)"
    )


# =============================================================================
# Portões da segunda ordem
# =============================================================================

def test_sem_focal_a_curvatura_sai_neutra_e_a_flag_avisa():
    """O `meta/` do release NÃO traz `focallength_px` — conferido campo a campo.
    Nesse caso `K~` sai 0,5 em TODO lugar (K = 0 depois da normalização
    simétrica) e `segunda_ordem_valida` sai False. Nunca um `fx` inventado: o
    teste (b) acima mostra que errar `fx` por 2x move K em mais de 50%, ou seja
    um palpite produziria um canal plausível e errado."""
    c = _consts()
    d01, am = _esfera_em(96, 800.0, foco=False)
    assert am.focallength_px is None
    out = S.pilha_geometrica(d01, am, c)
    ch_K = out.canais[CANAIS.index("K~")]
    assert out.segunda_ordem_valida is False
    assert np.all(ch_K == np.float32(0.5)), "K~ tem de ser exatamente neutro"
    # Os cinco canais de 1ª ordem continuam válidos: a falta de fx não os afeta.
    assert out.diagnostico["quantizacao_suficiente"] is True
    assert float(out.canais[CANAIS.index("O")].max()) > 0.0


def test_quantizacao_pobre_neutraliza_a_curvatura():
    """Com poucos níveis, derivada segunda é ruído de quantização e não
    geometria, então `K~` sai neutro em vez de ruído — e a flag avisa."""
    c = _consts()
    d01, am = _esfera_em(96, 800.0)
    grosso = np.rint(d01 * 20) / 20.0                 # 21 níveis
    out = S.pilha_geometrica(grosso, am, c)
    assert out.diagnostico["niveis_quantizacao"] < 256
    assert out.segunda_ordem_valida is False
    assert np.all(out.canais[CANAIS.index("K~")] == np.float32(0.5))


def test_niveis_de_quantizacao_conta_o_que_promete():
    assert S.niveis_quantizacao(np.linspace(0, 1, 1000).reshape(1, -1)) == 1000
    assert S.niveis_quantizacao(np.full((8, 8), 0.5)) == 1


# =============================================================================
# Sanidade dos seis canais, em todas as geometrias
# =============================================================================

_CENAS = {
    "esfera": lambda: G.esfera(96, 400.0, 2.0, 10.0),
    "plano_inclinado": lambda: G.plano_inclinado(96, 800.0, (0.3, 0.1, 1.0), 5.0),
    "cilindro": lambda: G.cilindro(96, 1600.0, 1.0, 6.0),
    "rampa": lambda: G.rampa(96, 2.0, 6.0),
    "degrau": lambda: G.degrau(96, 1.0, 20.0),
}


@pytest.mark.parametrize("nome", sorted(_CENAS))
@pytest.mark.parametrize("field", ["inverse", "depth"])
def test_todos_os_canais_ficam_em_zero_um_e_finitos(nome, field):
    c = _consts(u_max=1.0, s_max=3.0, tau_occlusion=1.0)
    d01, am = G.codifica_release(_CENAS[nome](), focallength_px=400.0)
    out = S.pilha_geometrica(d01, am, c, field=field)
    assert out.canais.shape == (6, 96, 96)
    assert out.canais.dtype == np.float32
    assert np.all(np.isfinite(out.canais)), f"{nome}/{field}: canal não finito"
    assert float(out.canais.min()) >= 0.0 and float(out.canais.max()) <= 1.0


def test_frontoparalelo_tambem_fica_em_zero_um():
    """Fora do parametrize porque precisa de `faixa` explícita."""
    d01, am = G.codifica_release(G.frontoparalelo(96, 5.0), faixa=(0.1, 0.3),
                                 focallength_px=400.0)
    out = S.pilha_geometrica(d01, am, _consts())
    assert np.all(np.isfinite(out.canais))
    assert float(out.canais.min()) >= 0.0 and float(out.canais.max()) <= 1.0


def test_diagnostico_traz_o_que_o_contrato_promete():
    """`max_abs_por_canal` existe por um motivo específico: correlação NÃO
    detecta normalização por imagem, porque é invariante a escala — foi assim
    que o mapa de defocus quebrado passou despercebido. O teste que discrimina é
    comparar o MÁXIMO ABSOLUTO ENTRE AMOSTRAS, e para isso o número tem de estar
    no diagnóstico com o nome do canal junto."""
    c = _consts()
    d01, am = _esfera_em(96, 400.0)
    diag = S.pilha_geometrica(d01, am, c).diagnostico
    for chave in ("niveis_quantizacao", "frac_O_saturado", "max_abs_por_canal"):
        assert chave in diag, f"falta {chave} no diagnóstico"
    assert set(diag["max_abs_por_canal"]) == set(CANAIS)
    assert diag["shape_bate_com_amostra"] is True


def test_max_abs_entre_amostras_muda_com_a_cena_e_nao_e_sempre_um():
    """A guarda contra normalização por imagem: duas cenas de escala métrica
    MUITO diferente têm de produzir `u` com máximos diferentes. Se o canal
    fosse renormalizado por imagem, os dois máximos seriam 1,0 e a diferença
    física sumiria — invisível para correlação."""
    c = _consts(u_max=2.0)
    d01a, ama = G.codifica_release(G.rampa(64, 0.6, 1.0))    # cena perto: u ~ 1..1,67
    d01b, amb = G.codifica_release(G.rampa(64, 20.0, 40.0))  # cena longe: u ~ 0,025..0,05
    ua = S.pilha_geometrica(d01a, ama, c).diagnostico["max_abs_por_canal"]["u"]
    ub = S.pilha_geometrica(d01b, amb, c).diagnostico["max_abs_por_canal"]["u"]
    assert ua == pytest.approx(1.0 / 0.6 / 2.0, rel=1e-3)
    assert ub == pytest.approx(1.0 / 20.0 / 2.0, rel=1e-3)
    assert ua / ub > 20.0, "a escala métrica tem de sobreviver à normalização"


# =============================================================================
# Utilidades
# =============================================================================

@pytest.mark.parametrize("sigma,taps", [(0.5, 5), (1.0, 7), (2.0, 13), (3.0, 19)])
def test_raio_do_nucleo_segue_tres_sigmas(sigma, taps):
    """`riemann/losses.py:86` fixava radius=2 para qualquer sigma: com σ=2,0
    isso dá 5 taps cobrindo ±1σ, uma caixa truncada que perde ~32% da massa.
    Aqui o raio é round(3σ)."""
    k = S.nucleo_gaussiano_1d(sigma)
    assert len(k) == taps
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-12)


def test_nucleo_com_sigma_nao_positivo_levanta():
    with pytest.raises(ValueError):
        S.nucleo_gaussiano_1d(0.0)


def test_L_e_unico_para_os_dois_eixos():
    """Uma imagem 16:9 não pode ganhar anisotropia de ENQUADRAMENTO nos canais
    que medem anisotropia. O L é min(largura, altura) e é o mesmo nos dois
    eixos — ver `comprimento_de_normalizacao`."""
    am = AmostraGeo(0.1, 1.0, largura_px=1920, altura_px=1080)
    assert S.comprimento_de_normalizacao(am) == 1080.0
    campo = np.tile(np.linspace(0.0, 1.0, 32), (32, 1))
    gx, _ = S.gradiente_normalizado(campo, 32.0)
    _, gy = S.gradiente_normalizado(campo.T, 32.0)
    assert np.allclose(np.abs(gx), np.abs(gy.T)), (
        "o mesmo perfil em x e em y tem de dar o mesmo gradiente"
    )


def test_dimensoes_invalidas_levantam():
    with pytest.raises(ValueError):
        S.comprimento_de_normalizacao(AmostraGeo(0.1, 1.0, 0, 32))


def test_disp01_precisa_ser_2d():
    am = AmostraGeo(0.1, 1.0, 8, 8)
    with pytest.raises(ValueError):
        S.pilha_geometrica(np.zeros((3, 8, 8)), am, _consts())


# =============================================================================
# Constantes: ida e volta, e a guarda contra o YAML da campanha antiga
# =============================================================================

def test_geoconstants_ida_e_volta_no_yaml():
    c = _consts()
    assert GeoConstants.from_dict(c.to_dict()) == c
    assert set(c.to_dict()) == {
        "tau_occlusion", "u_max", "s_max", "k0_curvature", "kt_max",
        "smooth_sigma", "min_niveis_para_segunda_ordem",
    }


def test_geoconstants_recusa_yaml_da_campanha_antiga():
    """Um YAML da `geo_cond_v1` traz `z_percentile_max` e `min_quant_levels`.
    Aceitá-lo ignorando o extra carregaria constantes calibradas na distribuição
    errada para dentro do treino novo — o que `referencia_antiga/LEIA.md`
    proíbe. Tem de falhar ALTO."""
    antigo = {
        "tau_occlusion": 88.170, "u_max": 2.723, "s_max": 1.0,
        "k0_curvature": 1.0, "kt_max": 5.0, "smooth_sigma": 2.0,
        "z_percentile_max": 99.5, "min_quant_levels": 256,
    }
    with pytest.raises(ValueError, match="geo_cond_v1"):
        GeoConstants.from_dict(antigo)


def test_geoconstants_recusa_campo_faltando():
    d = _consts().to_dict()
    del d["u_max"]
    with pytest.raises(ValueError, match="faltando"):
        GeoConstants.from_dict(d)


def test_geoconstants_nao_tem_default():
    """Se alguém puser um default, este teste cai — e é para cair. O ponto do
    módulo é que a constante venha de uma calibração medida, não de um número
    plausível que atravessa o treino sem levantar exceção."""
    with pytest.raises(TypeError):
        GeoConstants()  # type: ignore[call-arg]
