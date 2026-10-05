"""Contrato do sinal de controle da BokehNet — lado CONSUMIDOR.

Este módulo é o espelho *read-side* de `bokehnet-regen/src/control/contract.py`.
A geração grava; aqui a gente lê e reconstrói o mapa. As duas pontas TÊM que usar
a mesma fórmula, e é por isso que este arquivo existe em vez de a conta estar
espalhada pelo `data.py`: o projeto já chegou a **quatro** interpretações de K
porque a fórmula tinha quatro cópias.

Versão do contrato: ``metric_disparity_official_v1``.

    z          = depth_pro(aif)                 # METROS
    disp       = 1 / z                          # 1/m
    focus_disp = median(disp[mask])             # calculado NA disparidade
    K          = k_eq3 / 1000                   # Eq. 3 é em mm, a inferência em 1/m
    MAX_COC    = 100.0                          # Inference_bokehNet.py:20, congelado
    defocus    = clip(|K * (disp - focus_disp)| / MAX_COC, 0, 1)

Autoridade, em ordem: (1) o paper, onde ele fala; (2) `Inference_bokehNet.py`,
onde o paper cala; (3) decisão nossa, DECLARADA como desvio, onde os dois calam.

O paper escreve `D` como mapa de PROFUNDIDADE (§3.2) e nunca usa a palavra
*disparity*. Operar em `1/z` é decisão por autoridade do código oficial
(`Inference_bokehNet.py:94` faz `disp = 1.0/depth` e `:118` tira a mediana disso),
e a Eq. 3 só fecha dimensionalmente assim. Está declarado, não presumido.

O que este módulo impede POR CONSTRUÇÃO:

  1. `MAX_COC` não é parâmetro de função nenhuma. Como parâmetro com default, um
     override parcial normaliza o mapa por um valor enquanto os metadados dizem
     outro — foi o experimento `kfix` (`max_coc = 10,5107`), que subiu o LF-Bokeh
     e derrubou RealBokeh e RealDOF no MESMO lote.
  2. Não há fallback numérico. Falta de dado levanta `ControlContractError`,
     nunca vira constante. O `k = 50,0` em 11.635 de 11.635 amostras da rota B
     nasceu de um `except: pass`.
  3. Toda quantidade em pixel carrega a resolução em que foi medida.
     `k_at_resolution` é obrigatória entre o `k_value` gravado (escala da imagem
     ORIGINAL) e o mapa que o modelo recebe (escala do crop de treino).
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Constantes CONGELADAS — não são parâmetros, e não têm setter
# =============================================================================

#: Versão do contrato. Um release que grave outra coisa tem que ser recusado, não
#: adaptado: adaptar em silêncio é como duas convenções entram no mesmo lote.
CONTROL_VERSION = "metric_disparity_official_v1"

#: Normalizador do mapa. Vem de `Inference_bokehNet.py:20`, NÃO do paper — a Eq. 2
#: é crua (`D_def = K·|D − D_focus|`). Mas o mapa entra no VAE, então algum
#: normalizador é obrigatório, e o único com autoridade é o do código oficial.
#:
#: GLOBAL e congelado: nunca por imagem, nunca por rota, nunca por fonte. A rota B
#: ocupar [0, 0,05] e a rota C [0, 0,4] é FÍSICO e é o que o modelo tem que
#: aprender. Reescalar uma rota para "ocupar [0,1] como a outra" é o defeito D1
#: com granularidade mais grossa.
MAX_COC = 100.0

#: Fator entre a convenção da Eq. 3 (milímetros) e a da inferência (1/m).
MM_PER_M = 1000.0

UINT16_MAX = 65535.0


class ControlContractError(ValueError):
    """Violação do contrato. Nunca é capturada para virar default."""


def _exigir(condicao: bool, mensagem: str) -> None:
    # `if`, não `assert`: `python -O` desliga assert, e aí a trava sumiria
    # exatamente no run de produção, que é onde ela importa.
    if not condicao:
        raise ControlContractError(mensagem)


# =============================================================================
# Decodificação da profundidade gravada
# =============================================================================

def decode_disparity_u16(
    encoded: np.ndarray, disparity_min: float, disparity_max: float
) -> np.ndarray:
    """uint16 linear em DISPARIDADE → disparidade em 1/m.

        disp = disparity_min + (u16 / 65535) * (disparity_max - disparity_min)

    A geração quantiza em disparidade, e não em profundidade métrica, porque é o
    espaço em que o CoC vive: o passo é uniforme em disparidade, logo o erro de
    CoC também é (`erro = K · passo`). Quantizar em profundidade linear gastava
    quase toda a resolução do uint16 no fundo distante — 24,7% das amostras da
    rota B tinham a cena útil em menos de 256 níveis de 65535.

    `encoded` pode vir como PIL 'I;16', array uint16, ou já em [0,1] float — o
    caso float existe porque `datasets` às vezes devolve o PNG já normalizado.
    A distinção é pelo dtype, nunca pelo valor: um mapa uint16 legítimo pode ter
    máximo 1.
    """
    arr = np.asarray(encoded)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.floating):
        unit = arr.astype(np.float32)
        _exigir(
            float(unit.min()) >= -1e-6 and float(unit.max()) <= 1.0 + 1e-6,
            f"profundidade float fora de [0,1]: [{unit.min()}, {unit.max()}]",
        )
    else:
        unit = arr.astype(np.float32) / UINT16_MAX

    _exigir(
        np.isfinite(disparity_min) and np.isfinite(disparity_max),
        f"limites de disparidade não finitos: {disparity_min}, {disparity_max}",
    )
    _exigir(
        disparity_min > 0.0,
        f"disparity_min={disparity_min} <= 0 (z_max infinito não é representável aqui)",
    )
    _exigir(
        disparity_max > disparity_min,
        f"faixa de disparidade degenerada: [{disparity_min}, {disparity_max}]",
    )
    return (
        float(disparity_min) + unit * (float(disparity_max) - float(disparity_min))
    ).astype(np.float32)


# =============================================================================
# Escala — a regra 3 do CONTRATO.md
# =============================================================================

def k_at_resolution(
    k_value: float, image_hw: tuple[int, int], dst_short_side: int
) -> float:
    """Reescala K para a resolução em que o modelo de fato vê a imagem.

        s  = dst_short_side / min(H, W)
        K' = K * s

    CoC em pixel escala com a resolução; disparidade não. `image_hw` é a resolução
    da IMAGEM em que o `k_value` foi medido — não a do array de profundidade, que
    é gravado com o lado longo limitado e é outra grade.

    ESTE É O FATOR QUE O DATALOADER ANTIGO NUNCA APLICOU. Ele reduzia o lado menor
    para 512, recortava, e usava o K da imagem original. Fatores medidos: 0,892 e
    0,821 — ou seja, o erro VARIA POR AMOSTRA, o que é pior que um viés constante:
    o modelo não consegue absorvê-lo numa constante aprendida e a saída racional
    passa a ser ignorar o K.

    O crop posterior não muda a escala, só a janela — por isso a função recebe o
    lado MENOR de destino e não a caixa de crop.

    Cuidado de ergonomia, herdado do `bokehnet-regen`: aqui o argumento é o lado
    MENOR; no `encode_depth` da geração é o lado MAIOR. Convenções opostas na
    mesma cadeia — sempre nomeie o argumento na chamada.
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    _exigir(height > 0 and width > 0, f"image_hw inválido: {image_hw!r}")
    _exigir(int(dst_short_side) > 0, f"dst_short_side inválido: {dst_short_side!r}")
    _exigir(np.isfinite(k_value) and k_value >= 0, f"k_value inválido: {k_value!r}")
    escala = float(dst_short_side) / float(min(height, width))
    return float(k_value) * escala


def k_by_scale(k_value: float, escala: float) -> float:
    """Variante para quando o fator de resize JÁ foi calculado pelo plano geométrico.

    Existe porque `_plano_geometrico` decide o resize (que depende de `scale_mode`:
    `short_side`, `native` ou `long_side`) e devolve o fator efetivo. Derivar o
    fator de novo aqui a partir de `dst_short_side` daria o número errado nos
    modos `native` (fator 1,0, ou upscale) e `long_side`.

    Invariante travado por teste: com `scale_mode="short_side"`,
    `k_by_scale(k, escala) == k_at_resolution(k, (h, w), image_size)`.
    """
    _exigir(np.isfinite(escala) and escala > 0, f"escala inválida: {escala!r}")
    _exigir(np.isfinite(k_value) and k_value >= 0, f"k_value inválido: {k_value!r}")
    return float(k_value) * float(escala)


# =============================================================================
# O mapa de defocus
# =============================================================================

def signed_coc_px(
    disparity: np.ndarray, focus_disparity: float, k_value: float
) -> np.ndarray:
    """CoC com SINAL, em pixels, na resolução de `disparity`.

        CoC = K * (disp - focus_disp)

    O sinal separa frente de fundo do plano focal. O mapa de condição usa o módulo
    — a inferência oficial faz `np.abs` (`Inference_bokehNet.py:138`) — mas manter
    o sinal disponível permite diagnóstico ("o modelo confunde frente com fundo?")
    sem recalcular nada.
    """
    _exigir(np.isfinite(k_value) and k_value >= 0, f"K inválido: {k_value!r}")
    _exigir(
        np.isfinite(focus_disparity) and focus_disparity > 0,
        f"focus_disparity inválida: {focus_disparity!r}",
    )
    disp = np.asarray(disparity, dtype=np.float32)
    return (float(k_value) * (disp - float(focus_disparity))).astype(np.float32)


def defocus_map(
    disparity: np.ndarray, focus_disparity: float, k_value: float
) -> np.ndarray:
    """A condição da BokehNet, em [0, 1]. Réplica de `Inference_bokehNet.py:138-140`.

    `MAX_COC` NÃO é parâmetro desta função, e isso é deliberado — ver o docstring
    do módulo. Se um dia houver motivo medido para mudá-lo, muda-se a constante,
    o `CONTROL_VERSION` sobe, e a geração inteira é refeita. Não se passa um
    argumento.
    """
    coc = np.abs(signed_coc_px(disparity, focus_disparity, k_value))
    return np.clip(coc / MAX_COC, 0.0, 1.0).astype(np.float32)


def defocus_from_encoded_depth(
    encoded_depth: np.ndarray,
    *,
    disparity_min: float,
    disparity_max: float,
    focus_disparity: float,
    k_value: float,
) -> np.ndarray:
    """Caminho completo: profundidade gravada → mapa em [0,1].

    `k_value` tem que chegar aqui JÁ na escala de pixel do destino (ver
    `k_at_resolution` / `k_by_scale`). A função não sabe a resolução e por isso
    não pode corrigir — de propósito: corrigir aqui esconderia de quem chama a
    responsabilidade de saber em que grade está.
    """
    disparity = decode_disparity_u16(encoded_depth, disparity_min, disparity_max)
    return defocus_map(disparity, focus_disparity, k_value)


# =============================================================================
# Validação de metadados vindos do release
# =============================================================================

#: Chaves escalares que TODA amostra do release novo tem que trazer. A ausência de
#: qualquer uma é rejeição, nunca default.
CAMPOS_CONTROLE_OBRIGATORIOS = (
    "control_version",
    "k_value",
    "focus_disparity",
    "disparity_min",
    "disparity_max",
    "image_h",
    "image_w",
)


def validar_registro_controle(registro: dict) -> None:
    """Confere que uma linha do dataset traz o contrato inteiro e coerente.

    Levanta `ControlContractError` com o campo faltante nomeado. Chamada uma vez
    por amostra no `__getitem__`: o custo é desprezível ao lado do VAE, e o
    benefício é que um release meio-de-uma-convenção-meio-de-outra morre na
    primeira amostra em vez de treinar 60K steps.
    """
    faltando = [c for c in CAMPOS_CONTROLE_OBRIGATORIOS if registro.get(c) is None]
    _exigir(not faltando, f"campos de controle ausentes: {faltando}")

    versao = str(registro["control_version"])
    _exigir(
        versao == CONTROL_VERSION,
        f"control_version={versao!r} != {CONTROL_VERSION!r}. "
        "Um release de outra convenção não é adaptado, é recusado.",
    )

    max_coc_gravado = registro.get("max_coc")
    if max_coc_gravado is not None:
        _exigir(
            abs(float(max_coc_gravado) - MAX_COC) < 1e-9,
            f"max_coc do release = {max_coc_gravado} != {MAX_COC}. "
            "Normalizador por fonte é o defeito D1 com granularidade grossa.",
        )

    _exigir(
        float(registro["k_value"]) >= 0 and np.isfinite(float(registro["k_value"])),
        f"k_value inválido: {registro['k_value']!r}",
    )
    _exigir(
        float(registro["focus_disparity"]) > 0,
        f"focus_disparity inválida: {registro['focus_disparity']!r}",
    )
