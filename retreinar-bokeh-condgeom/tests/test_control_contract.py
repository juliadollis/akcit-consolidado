"""Invariantes do sinal de controle da BokehNet. Sem GPU, sem rede.

Cada teste aqui trava um dos defeitos da AUDITORIA_TREINO_BOKEHNET.md, e o nome
diz qual. Um teste que passa e não corresponde a um defeito real é decoração;
estes correspondem, e vários já pegaram regressão neste projeto.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from genfocus_train import control, data


# =============================================================================
# T4 — `MAX_COC` não pode ser parâmetro
# =============================================================================

def test_max_coc_nao_e_parametro_de_defocus_map():
    """O experimento `kfix` rodou duas convenções de normalizador no mesmo lote.

    Isso só foi possível porque `max_coc` era argumento com default: um override
    parcial normalizava a rota B por 10,5107 enquanto os metadados diziam 100,0.
    A defesa que funciona não é um comentário — é a ausência do parâmetro.
    """
    assinatura = inspect.signature(control.defocus_map)
    assert "max_coc" not in assinatura.parameters
    assert control.MAX_COC == 100.0   # Inference_bokehNet.py:20


def test_max_coc_divergente_no_release_e_recusado():
    registro = {
        "control_version": control.CONTROL_VERSION,
        "k_value": 15.0, "focus_disparity": 0.5,
        "disparity_min": 0.001, "disparity_max": 1.0,
        "image_h": 1500, "image_w": 2000,
        "max_coc": 10.5107,          # o valor do kfix
    }
    with pytest.raises(control.ControlContractError, match="max_coc"):
        control.validar_registro_controle(registro)


def test_control_version_divergente_e_recusado():
    registro = {
        "control_version": "normalized_depth_v0",
        "k_value": 15.0, "focus_disparity": 0.5,
        "disparity_min": 0.001, "disparity_max": 1.0,
        "image_h": 1500, "image_w": 2000,
    }
    with pytest.raises(control.ControlContractError, match="control_version"):
        control.validar_registro_controle(registro)


# =============================================================================
# T1 — a fórmula é a da inferência oficial, bit a bit
# =============================================================================

def test_defocus_map_replica_a_inferencia_oficial():
    """`Inference_bokehNet.py:138-140`, transcrito e comparado.

        disp_minus_focus = disp - disp_focus
        defocus_abs      = |k_value * disp_minus_focus|
        cond_map         = (defocus_abs / MAX_COC).clamp(0, 1)
    """
    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 50.0, size=(37, 41)).astype(np.float32)   # metros
    disp = 1.0 / z
    focus = float(np.median(disp))
    k = 15.0                                                        # o default oficial

    oficial = np.clip(np.abs(k * (disp - focus)) / 100.0, 0.0, 1.0)
    nosso = control.defocus_map(disp, focus, k)
    assert np.allclose(nosso, oficial, atol=0, rtol=0)


def test_focus_disparity_nao_e_um_sobre_a_mediana_da_profundidade():
    """Regra 1 do CONTRATO.md, e a diferença é real com contagem PAR.

    `np.median` faz a MÉDIA dos dois centrais quando a contagem é par, e a média
    de dois recíprocos não é o recíproco da média.
    """
    z = np.array([1.0, 2.0], dtype=np.float32)     # contagem PAR
    mediana_da_disparidade = float(np.median(1.0 / z))          # (1 + 0,5)/2 = 0,75
    um_sobre_mediana = 1.0 / float(np.median(z))                # 1/1,5 = 0,6667
    assert not np.isclose(mediana_da_disparidade, um_sobre_mediana)


# =============================================================================
# T2 — o K reescalado pela resolução do crop
# =============================================================================

def test_k_at_resolution_escala_com_o_lado_menor():
    # 1500x2000 -> lado menor 1500 -> 512: fator 0,3413
    k = control.k_at_resolution(15.0, (1500, 2000), 512)
    assert np.isclose(k, 15.0 * 512 / 1500)


def test_k_by_scale_bate_com_k_at_resolution_em_short_side():
    """As duas rotas de cálculo do fator TÊM que dar o mesmo número.

    `k_by_scale` existe porque `_plano_geometrico` já decidiu o resize e o fator
    depende do `scale_mode`; derivar de novo a partir de `dst_short_side` daria
    o número errado em `native` e `long_side`. Em `short_side` os dois coincidem,
    e é isso que este teste trava.
    """
    for (h, w) in [(1500, 2000), (2000, 1500), (683, 1024), (512, 512)]:
        novo_w, novo_h, _box, _flip, _seq = data._plano_geometrico(
            w, h, 512, "short_side", train=False, rng=None
        )
        escala = novo_w / float(w)
        assert np.isclose(
            control.k_by_scale(7.5, escala),
            control.k_at_resolution(7.5, (h, w), 512),
            rtol=2e-3,   # o resize arredonda para inteiro de pixel
        ), f"divergência em {h}x{w}"


def test_gate_mapa_pos_crop(tmp_path):
    """Gate NOVO do PLANO_REGERACAO: "mapa POS-CROP bate com k*|delta_disp|/100
    na escala do recorte".

    É o teste que o pipeline antigo não tinha e que teria pego o T2 no primeiro
    dia. Monta uma amostra sintética, passa pelo caminho REAL do dataloader, e
    confere o mapa contra a fórmula recalculada à mão na escala do crop.
    """
    from PIL import Image

    h, w = 600, 800
    # Disparidade que varia suavemente, sem borda (o resize não é o assunto aqui)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disp = 0.05 + 0.9 * (xx / (w - 1))                     # 1/m, de 0,05 a 0,95
    disp_min, disp_max = float(disp.min()), float(disp.max())
    u16 = np.round(
        (disp - disp_min) / (disp_max - disp_min) * 65535.0
    ).astype(np.uint16)

    aif = Image.fromarray(
        (np.stack([xx, yy, xx], -1) % 255).astype(np.uint8), "RGB"
    )
    bokeh = aif.copy()
    k_gravado, focus = 22.0, 0.4

    _a, _b, mapa, geo = data.prepare_aligned_bokeh_metric(
        aif, bokeh, u16,
        k_value=k_gravado, focus_disparity=focus,
        disparity_min=disp_min, disparity_max=disp_max,
        image_hw=(h, w), image_size=512, train=False, scale_mode="short_side",
    )

    # Recalcula à mão: disparidade reamostrada e recortada, K na escala do crop.
    novo_w, novo_h, box, _flip, _seq = data._plano_geometrico(
        w, h, 512, "short_side", train=False, rng=None
    )
    disp_dec = control.decode_disparity_u16(u16, disp_min, disp_max)
    disp_r = data.resize_nearest(disp_dec, novo_w, novo_h)
    x0, y0, x1, y1 = box
    k_esperado = k_gravado * (novo_w / w)
    esperado = np.clip(
        np.abs(k_esperado * (disp_r[y0:y1, x0:x1] - focus)) / control.MAX_COC, 0, 1
    )

    assert np.isclose(geo["k_efetivo"], k_esperado)
    assert np.allclose(mapa[0].numpy(), esperado, atol=1e-6)
    # e os 3 canais são idênticos, como `.repeat(3,1,1)` da inferência
    assert np.allclose(mapa[0].numpy(), mapa[2].numpy())


def test_k_cru_no_crop_erra_e_o_erro_varia_por_amostra():
    """Quantifica o T2, e o que importa nele.

    O nome anterior deste teste dizia "erra por mais de um terço", o que é FALSO
    para as duas resoluções realmente medidas no contrato (fatores 0,892 e 0,821
    = 12% e 22% de erro). A magnitude não é o argumento.

    O argumento é que o fator **varia por amostra**, de 1,12x a 2,93x. Um viés
    constante o modelo absorveria numa escala aprendida; um que muda a cada
    amostra não, e a saída racional passa a ser ignorar o K.
    """
    # (H, W) -> fator esperado, conferido à mão
    casos = {
        (1500, 2000): 512 / 1500,   # 0,341  -> K gravado 2,93x grande demais
        (683, 1024): 512 / 683,     # 0,750  -> 1,33x
        (624, 832): 512 / 624,      # 0,821  <- medido no CONTRATO.md
        (574, 766): 512 / 574,      # 0,892  <- medido no CONTRATO.md
    }
    fatores = []
    for (h, w), esperado in casos.items():
        fator = control.k_at_resolution(1.0, (h, w), 512)
        assert np.isclose(fator, esperado), f"{h}x{w}: {fator} != {esperado}"
        assert fator < 1.0, f"{h}x{w}: sem reescala o K fica grande demais"
        fatores.append(fator)

    # O QUE IMPORTA: a dispersão. Se fosse constante, seria absorvível.
    assert max(fatores) / min(fatores) > 2.5, (
        f"os fatores {fatores} deveriam variar por mais de 2,5x entre amostras"
    )


# =============================================================================
# T3 — NEAREST não inventa profundidade
# =============================================================================

def test_resize_nearest_nao_inventa_valor_intermediario():
    """Numa borda entre 1 m e 20 m, BILINEAR produz 10,5 m — uma superfície
    fantasma que o renderer depois borra como se fosse real."""
    disp = np.concatenate(
        [np.full((8, 4), 1.0, np.float32), np.full((8, 4), 0.05, np.float32)], axis=1
    )
    reduzido = data.resize_nearest(disp, 4, 4)
    assert set(np.unique(reduzido)) <= set(np.unique(disp))


def test_resize_nearest_e_identidade_no_mesmo_tamanho():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert data.resize_nearest(a, 4, 3) is a


def _resize_da_geracao(depth_m, long_side):
    """Cópia literal de `bokehnet-regen/src/dataio/encoding.py:resize_depth_nearest`."""
    h, w = depth_m.shape[:2]
    if max(h, w) <= long_side:
        return depth_m
    escala = long_side / float(max(h, w))
    nh, nw = max(1, round(h * escala)), max(1, round(w * escala))
    yi = np.minimum((np.arange(nh) / escala).astype(np.int64), h - 1)
    xi = np.minimum((np.arange(nw) / escala).astype(np.int64), w - 1)
    return depth_m[yi][:, xi]


def test_convencao_do_resize_diverge_da_geracao_so_com_arredondamento():
    """DIVERGÊNCIA CONHECIDA E MEDIDA, não bug — e por isso está num teste.

    A geração amostra em `i / escala`, com `escala = long_side / max(H,W)`; aqui
    amostramos em `i * H / novo_H`. Os dois coincidem sempre que
    `novo_H == H * escala` exatamente, que é o caso comum. Quando o `round()`
    muda o tamanho de saída, a grade da geração desliza até 1 pixel de origem.

    Por que a nossa é a que fica: `i*H/novo_H` mapeia a grade de saída
    uniformemente sobre a origem — é a definição padrão do vizinho mais próximo e
    é aritmética inteira exata. A da geração usa a escala PEDIDA, que não é a
    escala OBTIDA depois do arredondamento.

    Por que isso não é o defeito de "duas convenções na mesma cadeia": no treino
    o resize vai da grade da PROFUNDIDADE GRAVADA (lado longo 768) para a grade
    do crop — uma transformação que a geração nunca faz. Identidade bit a bit
    nunca foi possível. O que se exige é que a convenção seja UMA, declarada, e
    que a divergência esteja medida. Está aqui.

    AÇÃO PENDENTE: propor ao `bokehnet-regen` adotar `i*H/novo_H`. Enquanto não
    adotarem, este teste é o registro do tamanho da diferença.
    """
    rng = np.random.default_rng(0)
    for (h, w) in [(1500, 2000), (683, 1024), (1000, 1000)]:
        a = rng.random((h, w)).astype(np.float32)
        g = _resize_da_geracao(a, 768)
        nh, nw = g.shape
        assert np.array_equal(g, data.resize_nearest(a, nw, nh)), (
            f"{h}x{w}: as duas convenções deveriam coincidir (sem arredondamento)"
        )

    # Caso em que o arredondamento morde: aspecto não exato.
    a = np.arange(1503 * 2001, dtype=np.float32).reshape(1503, 2001)
    g = _resize_da_geracao(a, 768)
    nh, nw = g.shape
    meu = data.resize_nearest(a, nw, nh)
    fracao_diferente = float((g != meu).mean())
    # Medido: 18,0%. Travado com folga para não quebrar por versão de numpy,
    # mas apertado o bastante para denunciar se a divergência crescer.
    assert 0.0 < fracao_diferente < 0.25, f"divergência de {fracao_diferente:.1%}"


# =============================================================================
# T1 — decodificação da profundidade gravada
# =============================================================================

def test_decode_disparity_roundtrip():
    disp = np.linspace(0.01, 2.0, 256, dtype=np.float32).reshape(16, 16)
    dmin, dmax = float(disp.min()), float(disp.max())
    u16 = np.round((disp - dmin) / (dmax - dmin) * 65535.0).astype(np.uint16)
    voltou = control.decode_disparity_u16(u16, dmin, dmax)
    # erro de quantização: (dmax-dmin)/65535 por passo
    assert np.max(np.abs(voltou - disp)) < (dmax - dmin) / 65535.0


def test_decode_disparity_recusa_faixa_degenerada():
    with pytest.raises(control.ControlContractError):
        control.decode_disparity_u16(np.zeros((4, 4), np.uint16), 0.5, 0.5)


def test_defocus_recusa_focus_disparity_invalida():
    with pytest.raises(control.ControlContractError):
        control.defocus_map(np.ones((4, 4), np.float32), 0.0, 10.0)


# =============================================================================
# T2 — resolução heterogênea é recusada, não ajustada
# =============================================================================

def test_resolucao_divergente_do_metadado_e_recusada():
    from PIL import Image

    aif = Image.new("RGB", (800, 600))
    u16 = np.zeros((600, 800), np.uint16)
    with pytest.raises(control.ControlContractError, match="k_value"):
        data.prepare_aligned_bokeh_metric(
            aif, aif.copy(), u16,
            k_value=10.0, focus_disparity=0.5,
            disparity_min=0.01, disparity_max=1.0,
            image_hw=(1200, 1600),      # metadado diz outra resolução
            image_size=512, train=False,
        )


# =============================================================================
# T10 / T16 — proporção entre fontes é decisão, não acidente
# =============================================================================

def test_pesos_de_replay_realizam_a_fracao_exata():
    pesos = data.montar_pesos_de_replay(n_real=1000, n_sintetico=50, fracao_sintetica=0.15)
    massa_real = sum(pesos[:1000])
    massa_sint = sum(pesos[1000:])
    assert np.isclose(massa_sint / (massa_real + massa_sint), 0.15)
    # e o tamanho desproporcional das partes NÃO afeta a fração — que é o ponto
    pesos2 = data.montar_pesos_de_replay(n_real=100, n_sintetico=9000, fracao_sintetica=0.15)
    assert np.isclose(sum(pesos2[100:]) / sum(pesos2), 0.15)


def test_fracao_zero_zera_a_parte_sintetica():
    pesos = data.montar_pesos_de_replay(10, 5, 0.0)
    assert sum(pesos[10:]) == 0.0


def test_pesos_por_rota_normalizam_pelo_tamanho():
    rotas = ["b"] * 900 + ["c"] * 100
    pesos = data._pesos_por_rota(rotas, {"b": 0.5, "c": 0.5})
    assert np.isclose(sum(pesos[:900]), sum(pesos[900:]))


def test_peso_ausente_e_erro_e_nao_zero_silencioso():
    with pytest.raises(ValueError, match="route_weights"):
        data._pesos_por_rota(["b", "c"], {"b": 1.0})


# =============================================================================
# T11 — a métrica do probe
# =============================================================================

def test_lvcorr_e_positivo_quando_o_controle_funciona():
    from genfocus_train.probe import lvcorr

    ks = [0.0, 5.0, 10.0, 15.0]
    nitidez = [100.0, 60.0, 30.0, 10.0]      # cai quando K cresce = controle OK
    assert lvcorr(ks, nitidez) > 0.9


def test_lvcorr_e_nan_quando_o_modelo_ignora_o_k():
    from genfocus_train.probe import lvcorr

    assert np.isnan(lvcorr([0.0, 5.0, 10.0], [42.0, 42.0, 42.0]))


def test_lvcorr_agregado_ignora_nan():
    from genfocus_train.probe import lvcorr_agregado

    assert np.isclose(lvcorr_agregado({"a": 0.9, "b": float("nan"), "c": 0.7}), 0.8)


# =============================================================================
# A revisão externa: "o loader dividiria uma nova profundidade float por 65535"
# =============================================================================

def test_decode_disparity_distingue_por_dtype_e_nao_por_valor():
    """A divisão por 65535 é do dtype INTEIRO, nunca do float.

    A mina: um `depth` do contrato novo que chegue como float — PIL modo 'F',
    `datasets` devolvendo o PNG já convertido, ou disparidade crua — dividido
    cegamente por 65535 vira ~1e-5. O mapa sai praticamente zero, o modelo
    aprende "nunca borrar", e NADA levanta erro. Um treino inteiro rodaria assim.

    Regra: distinguir pelo dtype. Distinguir pelo VALOR seria errado — um mapa
    uint16 legítimo pode ter máximo 1.
    """
    disp = np.array([[0.5, 0.25], [0.1, 0.01]], dtype=np.float32)   # 2, 4, 10, 100 m
    dmin, dmax = float(disp.min()), float(disp.max())
    unit = ((disp - dmin) / (dmax - dmin)).astype(np.float32)
    u16 = np.round(unit * 65535).astype(np.uint16)

    # inteiro em qualquer largura -> divide
    for arr in (u16, u16.astype(np.int32), u16.astype(np.uint32)):
        assert np.allclose(
            control.decode_disparity_u16(arr, dmin, dmax), disp, atol=1e-4
        ), f"dtype {arr.dtype} deveria ser dividido por 65535"

    # float JÁ normalizado -> NÃO divide
    assert np.allclose(
        control.decode_disparity_u16(unit, dmin, dmax), disp, atol=1e-6
    )


def test_decode_disparity_recusa_float_fora_de_zero_um():
    """Float fora de [0,1] é recusado, nunca reescalado por adivinhação."""
    for arr in (
        np.array([[0.0, 65535.0]], dtype=np.float32),   # PIL 'F' com escala u16
        np.array([[2.0, 100.0]], dtype=np.float32),     # profundidade em METROS
    ):
        with pytest.raises(control.ControlContractError, match="fora de"):
            control.decode_disparity_u16(arr, 0.01, 0.5)


def test_caminho_aposentado_tambem_recusa_float_nao_normalizado():
    """A mesma mina no caminho APOSENTADO (`_defocus_to_float_pil`).

    Ele está atrás de `allow_retired_defocus_sources`, mas continuava dividindo
    incondicionalmente. Uma mina desarmada custa três linhas; uma mina gated
    ainda é uma mina.
    """
    from PIL import Image

    u16 = np.array([[0, 65535], [32768, 100]], dtype=np.uint16)
    saida = np.asarray(data._defocus_to_float_pil(u16), dtype=np.float32)
    assert np.isclose(saida.max(), 1.0) and np.isclose(saida.min(), 0.0)

    ja_float = (u16.astype(np.float32) / 65535.0)
    assert np.allclose(
        np.asarray(data._defocus_to_float_pil(ja_float), dtype=np.float32), ja_float
    ), "float já em [0,1] não pode ser dividido de novo"

    with pytest.raises(ValueError, match="fora de"):
        data._defocus_to_float_pil(u16.astype(np.float32))   # 0..65535 em float


def test_nenhuma_constante_K_converte_profundidade_normalizada_em_disparidade():
    """O núcleo do T1, medido em vez de argumentado.

    O caminho antigo montava `K·|D01 − s1|` com `D01` = profundidade métrica
    normalizada min-max; a inferência oficial monta `K·|1/z − 1/z_foco|`. É uma
    REPARAMETRIZAÇÃO NÃO LINEAR: nenhum escalar leva uma na outra, e por isso a
    busca binária por K da avaliação absorve a diferença de ESCALA mas não a de
    FORMA — que é a frase do `PLANO_REGERACAO_BOKEHNET.txt` §0, aqui virando
    número.
    """
    rng = np.random.default_rng(0)
    z = np.concatenate([
        rng.uniform(1.0, 20.0, 9000),        # o conteúdo
        rng.uniform(200.0, 10000.0, 1000),   # o fundo distante
    ]).astype(np.float32)

    z01 = (z - z.min()) / (z.max() - z.min())
    z_foco = 3.0
    K = 15.0
    mapa_disp = np.abs(K * (1.0 / z - 1.0 / z_foco))
    mapa_z01 = np.abs(K * (z01 - (z_foco - z.min()) / (z.max() - z.min())))

    # melhor constante possível, por mínimos quadrados
    alpha = float((mapa_z01 * mapa_disp).sum() / (mapa_z01 * mapa_z01).sum())
    residuo = np.linalg.norm(mapa_disp - alpha * mapa_z01) / np.linalg.norm(mapa_disp)

    # Medido: 0,926. Se algum dia isto cair perto de zero, a premissa desta
    # árvore inteira mudou e a regeração dos dados precisaria ser rediscutida.
    assert residuo > 0.5, (
        f"resíduo {residuo:.4f}: uma constante estaria quase corrigindo, o que "
        "contradiz a premissa do contrato metric_disparity_official_v1"
    )
    assert abs(np.corrcoef(mapa_z01, mapa_disp)[0, 1]) < 0.6


# =============================================================================
# Achados da auditoria adversarial (A1, A2, A5)
# =============================================================================

def test_A1_o_add_adapter_do_diffusers_desativa_os_outros():
    """A premissa do bug A1, travada contra a biblioteca instalada.

    `PeftAdapterMixin.add_adapter` termina em `set_adapter(nome)`, cujo docstring
    diz "forcing the model to only use that adapter and disables the other
    adapters". Por isso `_inject_shape_lora` PRECISA chamar
    `set_adapter([base, forma])` depois.

    Se uma versão futura do diffusers parar de desativar, este teste falha — e
    aí a chamada extra vira inofensiva, não errada. O teste existe para que a
    premissa fique visível, não para congelar a biblioteca.
    """
    import inspect

    peft_mod = pytest.importorskip(
        "diffusers.loaders.peft", reason="diffusers não instalado neste ambiente"
    )
    fonte = inspect.getsource(peft_mod.PeftAdapterMixin.add_adapter)
    assert "set_adapter" in fonte, (
        "add_adapter não chama mais set_adapter: reavalie _inject_shape_lora"
    )
    doc = peft_mod.PeftAdapterMixin.set_adapter.__doc__ or ""
    assert "disables the other adapters" in doc


def test_A1_backbone_reativa_os_dois_adapters_e_confere_active_adapters():
    """Sem GPU: confere o FONTE de `_inject_shape_lora`.

    Dois predicados distintos, e a versão com bug só tinha um:
      `requires_grad`   — quem TREINA
      `active_adapters` — quem AGE
    Um LoRA congelado que não está ativo não é congelado, é ausente.
    """
    import inspect
    import re

    fonte = pathlib_backbone_source()
    corpo = fonte.split("def _inject_shape_lora")[1].split("\n    def ")[0]

    assert re.search(
        r"set_adapter\(\s*\[\s*ADAPTER_NAME\s*,\s*SHAPE_ADAPTER_NAME\s*\]\s*\)", corpo
    ), "falta reativar os DOIS adapters depois do add_adapter (A1)"
    assert "active_adapters" in corpo, (
        "a trava do §3.3 tem que conferir active_adapters, e não só requires_grad: "
        "requires_grad é o mesmo predicado que o set_adapter do PEFT já ajustou, "
        "então uma trava baseada só nele CONCORDA com o bug"
    )
    # a ordem importa: reativar ANTES de congelar
    assert corpo.index("set_adapter") < corpo.index("requires_grad_(False)")


def pathlib_backbone_source() -> str:
    import pathlib

    return (pathlib.Path(__file__).resolve().parent.parent
            / "genfocus_train" / "backbone.py").read_text(encoding="utf-8")


def test_A5_validacao_nao_herda_replay_sintetico():
    """O loader de validação zera o replay — senão metade da val_loss é treino.

    `synthetic_replay_datasets` aponta para a rota A com `split: train`: é a
    fonte de TREINO do braço. E o braço afetado seria justamente o que existe
    para testar uma hipótese nossa.
    """
    import dataclasses
    import pathlib
    import re

    from genfocus_train.config import DatasetSourceConfig, StageConfig

    # Lê o FONTE em vez de importar: `trainer` puxa `backbone`, que exige
    # diffusers, e este teste roda em ambiente sem GPU e sem diffusers.
    texto = (pathlib.Path(__file__).resolve().parent.parent
             / "genfocus_train" / "trainer.py").read_text(encoding="utf-8")
    fonte = texto.split("def _build_val_loader")[1].split("\ndef ")[0]
    assert re.search(r"synthetic_replay_fraction\s*=\s*0\.0", fonte)
    assert re.search(r"synthetic_replay_datasets\s*=\s*\[\]", fonte)

    # e o `replace` realmente zera, sem o __post_init__ reclamar
    cfg = StageConfig(
        datasets=[DatasetSourceConfig("repo/treino", "train")],
        val_datasets=[DatasetSourceConfig("repo/val", "train")],
        steps=10,
        synthetic_replay_fraction=0.15,
        synthetic_replay_datasets=[DatasetSourceConfig("repo/rota-a", "train")],
    )
    val = dataclasses.replace(
        cfg, datasets=list(cfg.val_datasets),
        synthetic_replay_fraction=0.0, synthetic_replay_datasets=[],
    )
    assert val.synthetic_replay_fraction == 0.0
    assert val.synthetic_replay_datasets == []


def test_A2_bokeh_shape_falha_nomeando_a_lacuna():
    """O estágio §3.3 não tem dado. A falha tem que DIZER isso.

    Antes: `NotImplementedError("Stage 'bokeh_shape' não suportado.")` — genérico,
    e o `train check` dava luz verde, então o operador descobria na fila.
    """
    from genfocus_train.config import DatasetSourceConfig, StageConfig

    cfg = StageConfig(
        datasets=[DatasetSourceConfig("repo/x", "train")], steps=1,
        shape_column="aperture_shape",
    )
    with pytest.raises(NotImplementedError) as ctx:
        data.build_dataset(
            stage="bokeh_shape", stage_config=cfg,
            runtime=data.DatasetRuntimeConfig(image_size=512, train=False),
        )
    msg = str(ctx.value)
    assert "PointLight-1K" in msg and "Eq. 6" in msg, (
        "a mensagem tem que nomear a lacuna de DADOS, não só dizer 'não suportado'"
    )


def test_A4_full_seq_len_sai_da_imagem_de_origem():
    """O `mu` do treino tem que ser o que a inferência usaria para esta imagem.

    A inferência default (`--long_side 0`) NÃO redimensiona: o `image_seq_len`
    dela é o da imagem inteira nativa, alinhada a 16 (`flux.py:624`, antes do
    tiling). Antes, `full_seq_len` saía da imagem JÁ redimensionada, e
    `sigma_mu_source="full_image"` fechava só 3% do vão numa foto 3MP.
    """
    for (w, h) in [(1024, 688), (2000, 1500), (2048, 1536)]:
        _nw, _nh, _box, _flip, seq = data._plano_geometrico(
            w, h, 512, "short_side", train=False, rng=None
        )
        esperado = (((w + 15) // 16 * 16) // 16) * (((h + 15) // 16 * 16) // 16)
        assert seq == esperado, f"{w}x{h}: {seq} != {esperado} (tokens da ORIGEM)"
        # e é MUITO maior que os 1024 tokens do crop — é esse o ponto
        assert seq > 1024


def test_A7_route_weights_sobrevive_ao_replay():
    """Ligar T16 não pode desligar T10 em silêncio."""
    pesos_rota = [3.0] * 2 + [1.0] * 2        # rota B 3x mais provável que C
    combinado = data.montar_pesos_de_replay(
        n_real=4, n_sintetico=2, fracao_sintetica=0.2, pesos_reais=pesos_rota
    )
    assert np.isclose(sum(combinado[4:]), 0.2)          # a fração pedida
    assert np.isclose(sum(combinado[:4]), 0.8)          # o resto
    # e a proporção DENTRO da parte real sobreviveu
    assert np.isclose(combinado[0] / combinado[2], 3.0)

    # sem pesos por rota, cai no uniforme de antes (compatível)
    uniforme = data.montar_pesos_de_replay(4, 2, 0.2)
    assert len(set(uniforme[:4])) == 1


def test_A8_teto_por_cena_espalha_o_K_em_vez_de_pegar_as_primeiras():
    """A faixa de K DENTRO da cena é o sinal que ensina controle.

    Pegar as `teto` primeiras linhas de uma série ordenada por abertura trunca
    essa faixa sistematicamente. O corte tem que manter os extremos.
    """
    import re
    import pathlib as _pl

    fonte = (_pl.Path(__file__).resolve().parent.parent
             / "genfocus_train" / "data.py").read_text(encoding="utf-8")
    # a PRIMEIRA ocorrência de "scene_level_cap" é o vocabulário fechado do
    # RegistroDescartes; o filtro de fato vem depois do aviso de teto DESLIGADO.
    bloco = fonte.split("teto DESLIGADO")[1][:4000]
    assert "k_value" in bloco and "sorted" in bloco, (
        "o teto por cena tem que ordenar por k_value e espalhar"
    )

    # a aritmética do espaçamento, isolada: extremos sempre incluídos
    for n_linhas, teto in [(21, 4), (5, 4), (9, 3), (2, 4)]:
        if n_linhas <= teto:
            continue
        pos = [round(t * (n_linhas - 1) / (teto - 1)) for t in range(teto)]
        assert pos[0] == 0 and pos[-1] == n_linhas - 1, (
            f"n={n_linhas} teto={teto}: extremos de K perdidos ({pos})"
        )
        assert len(set(pos)) == teto


def test_A8_rotulo_do_teto_nao_afirma_que_e_do_paper():
    """Erro de categoria: decisão nossa apresentada como literal do supplement.

    O supplement B.2 descreve a composição dos 13K que os AUTORES coletaram, não
    um filtro a aplicar ao ITW, ao LFDOF ou à RealBokeh.
    """
    import pathlib as _pl

    # Formas aceitas de dizer "isto é decisão nossa". A lista é generosa de
    # propósito: o que o teste tem que impedir é a AUSÊNCIA da ressalva, não
    # impor uma redação. Impor redação transforma o teste num chato que as
    # pessoas contornam.
    RESSALVAS = (
        "DECISÃO NOSSA", "decisão nossa",
        "não um filtro", "NÃO é um filtro",
        "composição do dado deles",
    )
    raiz = _pl.Path(__file__).resolve().parent.parent
    alvos = [raiz / "genfocus_train" / "data.py",
             raiz / "genfocus_train" / "config.py"]
    alvos += sorted((raiz / "configs").glob("*.yaml"))

    for arq in alvos:
        texto = arq.read_text(encoding="utf-8")
        for pos, trecho in enumerate(texto.split("2 to 4 images per set")[:-1]):
            # janela AMPLA dos dois lados: a ressalva pode vir antes ou depois
            janela = trecho[-600:] + texto.split("2 to 4 images per set")[pos + 1][:900]
            assert any(r in janela for r in RESSALVAS), (
                f"{arq.name}: a citação do supplement B.2 aparece SEM a ressalva de "
                "que o teto por cena é decisão nossa. O supplement descreve a "
                "composição do dado dos autores, não um filtro para as outras rotas."
            )


# =============================================================================
# Revisão externa (2026-09-16) — os itens que não estavam cobertos
# =============================================================================

def _sem_comentarios(texto: str) -> str:
    """Remove comentários `#` para os testes casarem com CÓDIGO, não com a prosa
    que explica o defeito. Dois testes desta suíte já deram falso positivo por
    encontrar, num comentário, exatamente o padrão que eles proíbem."""
    saida = []
    for linha in texto.splitlines():
        corte = linha.find("#")
        saida.append(linha if corte < 0 else linha[:corte])
    return "\n".join(saida)

def _fonte(modulo: str) -> str:
    import pathlib as _pl

    return (_pl.Path(__file__).resolve().parent.parent
            / "genfocus_train" / f"{modulo}.py").read_text(encoding="utf-8")


def test_item11_seed_antes_da_inicializacao_do_lora():
    """`init_lora_weights="gaussian"` sorteia pesos dentro de `backbone.load()`.

    A seed era aplicada só em `_train_loop`, DEPOIS. Dois runs com a mesma seed
    começavam de pesos diferentes, e o `run_metadata` prometia reprodutibilidade
    que não existia.
    """
    fonte = _fonte("train")
    corpo = fonte.split("def _build_backbone_and_model")[1].split("\ndef ")[0]
    assert "_set_seed" in corpo, "a seed tem que ser aplicada antes do create_backbone"
    # compara com a CHAMADA, não com menções em docstring
    assert corpo.index("_set_seed(") < corpo.index("create_backbone(config"), (
        "a seed tem que vir ANTES da criação do backbone, não depois"
    )
    # e sem o offset por rank: os pesos iniciais têm que ser IGUAIS entre ranks
    assert "seed_per_rank" not in corpo


def test_item9_resume_recusa_fase_e_convencao_divergentes():
    """Retomar de um checkpoint de outra fase ou de outra convenção do mapa."""
    fonte = _fonte("trainer")
    corpo = fonte.split("def _verificar_compatibilidade_do_checkpoint")[1].split("\ndef ")[0]
    assert 'meta.get("stage")' in corpo, "o guard tem que comparar a FASE"
    assert 'meta.get("defocus_source")' in corpo, "o guard tem que comparar a CONVENÇÃO"
    assert "config_hash" in corpo
    # fase e convenção ABORTAM; config_hash só avisa
    assert corpo.count("raise RuntimeError") >= 2
    # e o chamador passa o config
    chamador = fonte.split("state = maybe_resume_checkpoint(")[1][:200]
    assert "config=config" in chamador


def test_item16_procedencia_registra_versoes_e_o_commit_do_forward():
    """"Treinado com que versão?" não pode depender de memória humana.

    O comportamento do `add_adapter` do PEFT quanto a adapters ativos já mudou
    entre versões — foi o defeito A1 — e `calculate_shift` é do diffusers.
    """
    fonte = _fonte("trainer")
    assert '"versoes": _versoes_do_ambiente()' in fonte
    assert '"genfocus_commit": _commit_do_forward()' in fonte

    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "_t_meta", __import__("pathlib").Path(__file__).resolve().parent.parent
        / "genfocus_train" / "trainer.py",
    )
    # não importa o módulo (puxa diffusers); confere só que os nomes das libs
    # que mudam o modelo estão listados
    corpo = fonte.split("def _versoes_do_ambiente")[1].split("\ndef ")[0]
    for lib in ("torch", "diffusers", "peft", "accelerate"):
        assert f'"{lib}"' in corpo, f"{lib} muda o modelo e tem que estar na procedência"


def test_item13_loss_agregada_entre_microbatches_e_gpus():
    """A loss do log era 1/32 do que o step otimizou (accum 8 × 4 GPUs)."""
    fonte = _fonte("trainer")
    assert "soma_loss += float(loss.detach().item())" in fonte
    assert "loss_val = soma_loss / max(1, n_micro)" in fonte
    assert "ReduceOp.AVG" in fonte.split("loss_val = soma_loss")[1][:600], (
        "além dos microbatches, a média entre GPUs"
    )


def test_item6_filtro_ssim_do_caminho_novo_e_por_amostra():
    """O caminho novo julga CADA amostra; o aposentado sonda 2.000 linhas.

    A sondagem existe lá só para responder "esta fonte tem SSIM preenchido?" —
    a rota B tem a coluna NULA em todas as linhas e seria descartada inteira por
    uma checagem de mera existência. Mas o LIMIAR, no caminho que importa, é
    aplicado linha a linha.
    """
    fonte = _fonte("data")
    # a ÚLTIMA ocorrência é a do caminho novo (`_selecionar_indices_metric`);
    # as anteriores são o dataclass e o caminho aposentado.
    bloco = fonte.split("runtime.min_calibration_ssim is not None")[-1][:1200]
    assert "for i, ssim in enumerate(col)" in bloco, (
        "o limiar de SSIM do §3.2(c) tem que ser avaliado POR AMOSTRA"
    )
    assert "2000" not in bloco, "nenhuma sondagem de N linhas decide o limiar"


# =============================================================================
# Terceira revisão externa (2026-09-16) — os seis pontos
# =============================================================================

def test_rev3_state_dict_inclui_o_base_congelado():
    """No §3.3 o base é congelado — e era EXCLUÍDO do checkpoint por isso.

    `if param.requires_grad` deixava o checkpoint da fase de forma sem o modelo
    que faz bokeh. Retomar dele produziria forma sobre ruído gaussiano.
    """
    fonte = _fonte("trainer")
    corpo = fonte.split("def _lora_state_dict")[1].split("\ndef ")[0]
    assert 'param.requires_grad or ".lora_" in name' in corpo, (
        "o checkpoint tem que gravar TODO parâmetro de LoRA, congelado inclusive"
    )


def test_rev3_cobertura_do_base_nao_usa_a_lista_de_treinaveis():
    """O defeito reproduzido: um checkpoint VAZIO era aceito.

    `missing` contém, por construção, só parâmetros com `requires_grad`. O base
    do §3.3 é congelado, então nunca aparecia lá e `base_faltando` vinha sempre
    vazio. A checagem tem que ser contra os parâmetros do MODELO.
    """
    fonte = _fonte("trainer")
    corpo = fonte.split("def load_lora_checkpoint_into_backbone")[1].split("\ndef ")[0]
    assert "_params_do_adapter(backbone, ADAPTER_NAME)" in corpo
    assert "[n for n in do_base if n not in lora_state]" in corpo
    assert "[n for n in missing" not in corpo, (
        "não dá para procurar o base congelado dentro da lista de treináveis"
    )

    # e a aritmética do defeito, isolada, para o motivo não virar folclore
    params = {"m.lora_A.default.weight": False, "m.lora_A.shape.weight": True}
    missing = [n for n, rg in params.items() if rg and n not in {}]
    assert [n for n in missing if ".default." in n] == [], (
        "é isto que tornava a checagem antiga vazia"
    )


def test_rev3_acumulacao_nao_sincroniza_no_fim_do_dataloader():
    """A sobra de época: um step por época com batch efetivo menor que 32."""
    fonte = _fonte("trainer")
    corpo = fonte.split("def _plugin_de_acumulacao")[1].split("\ndef ")[0]
    assert "sync_with_dataloader=False" in corpo
    # e o fallback é explícito, porque a API não pôde ser verificada aqui
    assert "except TypeError" in corpo and "AVISO GRAVE" in corpo


def test_rev3_probe_restaura_todos_os_rngs():
    """Medir não pode perturbar o que se mede.

    O probe chama `seed_everything` (torch + numpy) e gera na GPU, mas salvava
    só o RNG do torch na CPU.
    """
    fonte = _fonte("probe")
    corpo = fonte.split("def run_lvcorr_probe")[1]
    for estado in ("torch.get_rng_state()", "np.random.get_state()",
                   "torch.cuda.get_rng_state_all()"):
        assert estado in corpo, f"o probe tem que salvar {estado}"
    for restauro in ("torch.set_rng_state(", "np.random.set_state(",
                     "torch.cuda.set_rng_state_all("):
        assert restauro in corpo, f"o probe tem que restaurar via {restauro}"


def test_rev3_alarme_do_probe_para_de_verdade():
    """"O critério declarado é PARAR" — e o código só imprimia."""
    fonte = _fonte("trainer")
    assert "class ParadaPorControlabilidade" in fonte
    assert "raise ParadaPorControlabilidade(" in fonte
    assert "probe_alarmes_para_parar" in fonte
    assert "alarmes_seguidos = 0" in fonte, "o contador tem que zerar quando melhora"


def test_rev3_rng_de_augmentacao_com_num_workers_zero():
    """`num_workers=0` caía em `default_rng(None)` — entropia do SO."""
    fonte = _fonte("data")
    corpo = fonte.split("def _rng_do_worker")[1].split("\ndef ")[0]
    assert "SEED_DO_PROCESSO" in corpo
    assert "default_rng(None)" not in corpo


def test_rev3_resume_salva_e_restaura_rng():
    fonte = _fonte("trainer")
    assert '"rng": _estado_dos_rngs()' in fonte
    assert '"epoca": int(getattr(state, "epoca", 0))' in fonte
    corpo = fonte.split("def _estado_dos_rngs")[1].split("\ndef ")[0]
    for chave in ("torch_cpu", "numpy", "python", "torch_cuda"):
        assert chave in corpo
    # e a época volta para o _cycle
    assert "epoca_inicial=getattr(state" in fonte


def test_rev3_fingerprint_distingue_as_fases_que_o_stage_nao_distingue():
    """Fase 1 e fase 2 são as DUAS `stage=bokeh`. O que as separa são as fontes."""
    # lê o FONTE: `trainer` puxa `backbone`, que exige diffusers
    fonte = _fonte("trainer")
    corpo = fonte.split("def _fontes_fingerprint")[1].split("\ndef ")[0]
    assert "cfg.datasets" in corpo and "defocus_source" in corpo
    # NÃO pode incluir o que muda legitimamente num re-sbatch
    for proibido in ("steps", "upload", "wandb", "lr"):
        assert f"cfg.{proibido}" not in corpo, (
            f"`{proibido}` muda em re-sbatch legítimo e bloquearia a retomada"
        )


def test_rev3_init_lora_nao_cria_diretorio_ao_resolver():
    """`_checkpoint_dir` faz mkdir: usá-lo para PROCURAR criava o que procurava."""
    fonte = _fonte("train")
    corpo = fonte.split("def _resolve_init_lora")[1].split("\ndef ")[0]
    assert "_checkpoint_dir(" not in corpo, (
        "resolver --init-lora não pode chamar uma função que cria diretório"
    )
    assert "candidatos" in corpo and "is_dir()" in corpo


# =============================================================================
# Quarta revisão adversarial (2026-09-17)
# =============================================================================

def test_rev4_grad_accum_chega_ao_trainer():
    """C1 — era uma env var que NINGUÉM lia: efetivo 16 com banner dizendo 32."""
    import pathlib as _pl

    fonte = _fonte("train")
    assert "--grad-accum" in fonte and "_aplicar_grad_accum" in fonte
    assert "_aplicar_grad_accum(config, getattr(args" in fonte, (
        "o override tem que ser APLICADO, não só aceito"
    )
    slurm = (_pl.Path(__file__).resolve().parent.parent
             / "slurm" / "train_bokeh_fase1_h100n3.slurm").read_text(encoding="utf-8")
    assert "--grad-accum" in slurm, "o .slurm tem que PASSAR o valor"
    assert "--env GRAD_ACCUM" not in slurm, "a env var morta tem que ter saído"
    # e a conta inclui o batch_size, senão só acerta com batch_size=1
    assert "BATCH_SIZE" in slurm and "DIVISOR" in slurm


def test_rev4_ledger_indexa_por_sample_id_E_scene_id():
    """A1 — a rota A grava o sha da AIF por CENA; indexar só por sample_id
    deixava a fase 1 inteira SEM conferência de sha, com o YAML dizendo
    `verificar_sha256_da_origem: true`."""
    fonte = _fonte("release")
    corpo = fonte.split("def ledger_de_origem")[1].split("\n    def ")[0]
    assert '("sample_id", "scene_id")' in corpo
    # e o consumidor tenta a cena quando não há linha por amostra
    consumidor = fonte.split("linha_ledger = ledger.get(sample_id)")[1][:400]
    assert "ledger.get(cena)" in consumidor


def test_rev4_alvo_nao_e_recomprimido():
    """A2 — o ALVO do treino perdia ~37 dB de PSNR em JPEG q75, em silêncio."""
    fonte = _fonte("release")
    corpo = _sem_comentarios(
        fonte.split("def _bytes_de_linha_hf")[1].split("\n    def ")[0]
    )
    assert "decode=False" in corpo, "os bytes têm que vir crus da origem"
    # procura a CHAMADA, não a menção no comentário que explica o defeito
    import re as _re
    assert not _re.search(r'\.save\([^)]*format\s*=\s*["\']JPEG', corpo), (
        "nunca reencodar o alvo em JPEG"
    )
    # o fallback, se houver, é sem perda
    assert 'format="PNG"' in corpo


def test_rev4_epoca_avanca_de_verdade():
    """M4 — `epoca` era local ao `_cycle`: o checkpoint gravava sempre 0."""
    fonte = _fonte("trainer")
    corpo = fonte.split("def _cycle")[1].split("\ndef ")[0]
    assert "estado.epoca = epoca" in corpo, "a época tem que voltar para o TrainState"
    assert "estado=state" in fonte, "e o `_cycle` tem que receber o state"


def test_rev4_flip_funciona_em_todos_os_modos():
    """M1 — em `long_side` com rng=None o flip nunca acontecia (0/200)."""
    import numpy as _np

    for modo in ("short_side", "native", "long_side"):
        flips = sum(
            data._plano_geometrico(800, 600, 512, modo, True, None)[3]
            for _ in range(300)
        )
        assert 90 < flips < 210, f"{modo}: {flips}/300 flips — esperado ~metade"


def test_rev4_um_helper_de_rng_so():
    """M2 — havia duas cópias e só uma tinha o fallback de num_workers=0."""
    fonte = _fonte("data")
    assert fonte.count("def _rng_do_worker") == 1
    # o helper do deblur delega em vez de duplicar a lógica
    if "def _get_rng" in fonte:
        corpo = fonte.split("def _get_rng")[1].split("\n    def ")[0]
        assert "_rng_do_worker" in corpo, "o helper do deblur tem que DELEGAR"
    assert "default_rng(None)" not in _sem_comentarios(fonte), (
        "nenhum caminho pode cair na entropia do SO"
    )


def test_rev4_guarda_do_geo_nos_DOIS_leitores():
    """M3 — o mesmo YAML estourava num formato e ficava mudo no outro."""
    fonte = _fonte("data")
    for classe in ("class BokehMetricDataset", "class BokehReleaseTreeDataset"):
        corpo = fonte.split(classe)[1].split("\nclass ")[0]
        assert "runtime.geo_condition" in corpo, (
            f"{classe} precisa da guarda: ele não emite geo_map"
        )


def test_rev4_check_nao_exige_token_para_run_de_disco():
    """A3 — um run 100% de disco era reprovado por credencial que não usa,
    e o .slurm morria antes de treinar."""
    fonte = _fonte("train")
    assert "fontes_do_hub" in fonte
    assert "(problemas if fontes_do_hub else avisos).append(" in fonte
