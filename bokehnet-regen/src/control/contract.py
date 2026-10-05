"""Contrato canônico do sinal de controle da BokehNet — `metric_disparity_official_v1`.

Esta é a ÚNICA implementação válida. Geração, dataloader, avaliação e inferência
importam daqui. Cópias divergem: foi assim que o projeto chegou a quatro
interpretações de K.

Autoridade, em ordem:
  1. o paper (arXiv:2512.16923v3), onde ele fala;
  2. `third_party/Genfocus/Inference_bokehNet.py`, onde o paper cala;
  3. decisão nossa, DECLARADA como desvio, onde os dois calam.

A fórmula:

    z          = depth_pro(aif)              # METROS, métrico, sem normalizar
    disp       = 1.0 / z                     # 1/m
    focus_disp = median(disp[mask])          # NA disparidade, não 1/median(z)
    K          = k_eq3 / 1000.0              # Eq. 3 é em mm; a inferência é em 1/m
    max_coc    = 100.0                       # MAX_COC oficial, congelado
    defocus    = clip(|K*(disp - focus_disp)| / max_coc, 0, 1)

Regras que este módulo impõe por construção:
  - sem fallback numérico: falta de dado levanta `SampleRejected`, nunca vira constante;
  - `max_coc` global e congelado, nunca por imagem nem por rota;
  - toda quantidade em pixel carrega a resolução em que foi medida;
  - `focus_disp` é mediana da disparidade e o mapa nunca usa `focus_depth_m` de volta.

Ver `reference/CONTRATO.md` para a derivação e as âncoras numéricas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------------
# Versão e constantes congeladas
# --------------------------------------------------------------------------------

CONTROL_VERSION = "metric_disparity_official_v1"

#: Normalizador do mapa de defocus. Vem de `Inference_bokehNet.py:20` (`MAX_COC = 100.0`),
#: NÃO do paper — a Eq. 2 é crua. Congelado por release e gravado em cada amostra.
#: Nunca recalibrar por imagem, por rota ou por fonte: a rota B ocupar [0, 0.05] e a
#: rota C [0, 0.4] é FÍSICO, e é o que o modelo precisa aprender.
MAX_COC = 100.0

#: Fator entre a convenção da Eq. 3 (milímetros) e a da inferência oficial (1/m).
#: `k_eq3` tem unidade px·mm e multiplica disparidade em 1/mm; a inferência usa
#: disparidade em 1/m. Ver a análise dimensional em `reference/CONTRATO.md`.
MM_PER_M = 1000.0

#: Largura do sensor full-frame, usada APENAS junto com um crop factor medido.
#: Nunca como default para amostra sem `focal_length_35`.
FULL_FRAME_WIDTH_MM = 36.0

#: Razão mínima z_max/z_min para a cena carregar informação de profundidade.
#: [A] ASSUMIDO, não medido. 1,02 é muito permissivo de propósito: rejeita só cena
#: praticamente frontoparalela. Calibrar no piloto pelo histograma de
#: `depth_range_degenerate`. Não confundir com a degeneração medida de faixa ÚTIL
#: (24,7% da rota B em menos de 256 níveis), que é um gate a montante, sobre a
#: quantização, e ainda não está implementado.
MIN_DEPTH_RANGE_RATIO = 1.02


# --------------------------------------------------------------------------------
# Rejeição — o oposto de fallback
# --------------------------------------------------------------------------------

class SampleRejected(Exception):
    """Amostra não pode receber rótulo de controle confiável.

    O `reason` é um slug estável e legível por máquina: é ele que alimenta o
    histograma de motivos de rejeição que todo run tem que imprimir no fim.
    Sem esse histograma não dá para calibrar limiar nenhum, e é ele que denuncia
    um fallback novo.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


#: Backends de profundidade registrados. Um só, de propósito: a cascata
#: `except Exception: pass` que trocava Depth Pro por Depth Anything devolvia
#: DISPARIDADE, gravada igual, deixando a amostra espelhada sem rastro.
DEPTH_BACKENDS = frozenset({"depth_pro"})

#: Slugs de rejeição vindos dos gates de qualidade. Sem esta ponte, `GateReport`
#: devolveria nomes de gate que não são slugs, e o histograma — o instrumento que
#: denuncia fallback novo — viraria vocabulário aberto.
GATE_REJECTION_REASONS = frozenset({
    "gate_mask_area_ratio",
    "gate_mask_border_coverage",
    "gate_mask_iou",
    "gate_focus_mask_not_sharpest",
    "gate_aif_sharpness",
    "gate_aif_aperture_wide",
    "gate_depth_useful_levels",
    "gate_focus_depth_implausible",
    "gate_pair_shape_mismatch",
    "gate_bokeh_not_blurrier",
    "gate_calibration_ssim_below_floor",
    #: A região que definiu `D_focus` reteve pouco detalhe da AIF — nem o refinamento
    #: automático do §3.2(c) achou plano de foco claro. É o único limiar que julga a
    #: região refinada com a grandeza certa (razão bokeh/AIF, normalizada pela textura
    #: da cena), e não com nitidez absoluta.
    "gate_focus_region_retention_low",
    # --- rota B: os três modos de falha da DeblurNet, que hoje compartilham slug ---
    #: A AIF gerada não é fiel à foto de entrada — estrutura perdida, cor deslocada,
    #: artefato. É a AIF "lavada" que o defeito B1 produzia ao rodar o nosso checkpoint
    #: sem `main_adapter="deblurring"`. Slug próprio porque ela pode ter variância de
    #: Laplaciano ALTA (artefato tem alta frequência) e escaparia por um gate de nitidez.
    "gate_deblur_aif_unfaithful",
    #: A AIF gerada é ~igual à entrada: a DeblurNet não deblurou nada. Difere do de cima
    #: como "não fez" difere de "fez errado", e o conserto é outro — lá é o adapter,
    #: aqui é o peso não ter carregado.
    "gate_deblur_aif_is_input",
    #: AIF e bokeh desalinhadas além do tolerado, por correlação de fase. Mede o
    #: deslocamento que o MODELO introduz, diferente do que o redimensionamento
    #: introduz — este último é determinístico e já está caracterizado.
    "gate_pair_registration_shift",
})

#: Slugs das FONTES — falhas em enumerar o par, antes de existir profundidade,
#: máscara ou K. Vivem aqui, e não em `sources/`, porque o vocabulário de rejeição é
#: fechado num lugar só: um slug que só existisse no adaptador não apareceria no
#: histograma agregado entre rotas, e é exatamente o histograma que denuncia fallback.
SOURCE_REJECTION_REASONS = frozenset({
    "source_name_unparseable",
    "source_metadata_missing",
    "source_level_out_of_range",
    "source_f_number_invalid",
    "source_metadata_field_invalid",
    "source_duplicate_sample",
    #: Os bytes da imagem no shard não decodificam, vêm vazios, ou a imagem
    #: chega numa resolução diferente da que o espelho declara. Resolução
    #: heterogênea muda K em pixel sem mudar nada visível no JSON.
    "source_image_unreadable",
    #: O `path` da célula do parquet contradiz o papel da coluna — o espelho do LFDOF
    #: nomeia cada célula (`focus`/`blur`/`pre-deblur`) e isso é conferível por linha.
    #: Colunas trocadas produziriam um dataset inteiro plausível, com AIF e alvo
    #: invertidos, e o sweep da Eq. 5 acharia um `K*` para ele sem denunciar nada.
    "source_image_role_mismatch",
    #: PNG RGBA com alpha < 255. `convert("RGB")` comporia sobre preto e INVENTARIA
    #: pixel na entrada do Depth Pro, do BiRefNet e do SSIM. Rejeita até que exista
    #: uma política de composição medida e declarada.
    "source_image_alpha_not_opaque",
    #: A origem oferece uma "AIF" que é produto de OUTRO modelo — o `image_pre_deblur`
    #: do LFDOF é pré-foco da DRB-Net. Usá-la seria o defeito D6 com outro nome: tratar
    #: saída de modelo como fotografia de referência.
    "source_pseudo_aif_excluded",
})

#: Slugs conhecidos. Manter fechado ajuda a agregar o histograma entre runs.
REJECTION_REASONS = GATE_REJECTION_REASONS | SOURCE_REJECTION_REASONS | frozenset({
    "depth_non_finite",
    "depth_non_positive",
    "depth_range_degenerate",
    "focus_mask_empty",
    "focus_disparity_invalid",
    "focus_depth_implausible",
    "exif_focal_length_missing",
    "exif_f_number_missing",
    "sensor_width_unresolvable",
    "focus_distance_below_focal_length",
    "k_non_finite",
    "k_non_positive",
    "k_out_of_configured_range",
    "max_coc_invalid",
    "resolution_invalid",
    "k_search_budget_exhausted",
    "depth_backend_unregistered",
    "depth_disparity_span_degenerate",
    "control_version_mismatch",
    #: O metadado afirma duas coisas incompatíveis sobre a região em foco — por
    #: exemplo `focus_source="retention_only"` com `mask_source="birefnet"`, que diria
    #: que a máscara gravada é a do BiRefNet quando o rótulo saiu da retenção. É
    #: proveniência que mente, e é o modo de falha que `focus_region.py` foi escrito
    #: para não repetir: a cascata antiga caía no GrabCut e seguia dizendo
    #: `"automatic"`.
    "focus_provenance_inconsistent",
})


def reject(reason: str, detail: str = "") -> None:
    """Único caminho de rejeição. Público: `qc` e `dataio` também rejeitam por aqui.

    `if`, não `assert`: `python -O` desliga assert, e aí o conjunto fechado deixaria
    de ser fechado exatamente no run de produção, que é onde importa.
    """
    if reason not in REJECTION_REASONS:
        raise KeyError(f"slug de rejeição não registrado: {reason!r}")
    raise SampleRejected(reason, detail)


_reject = reject          # alias interno


# --------------------------------------------------------------------------------
# Profundidade métrica
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricDepth:
    """Saída do Depth Pro preservada em METROS, com a proveniência do backend.

    `backend` existe porque o pipeline antigo tinha uma cascata de
    `except Exception: pass` que trocava Depth Pro por Depth Anything — que devolve
    DISPARIDADE, normalizada igual e gravada igual, deixando a amostra espelhada
    sem deixar rastro. Aqui só existe um backend, e ele é gravado.
    """

    values_m: np.ndarray
    min_m: float
    max_m: float
    backend: str            # SEM default: afirmar o backend por omissão é mentir

    @property
    def disparity(self) -> np.ndarray:
        """1/z em 1/m. É o espaço em que o controle vive."""
        return 1.0 / self.values_m

    @property
    def disparity_min(self) -> float:
        return 1.0 / self.max_m

    @property
    def disparity_max(self) -> float:
        return 1.0 / self.min_m


def validate_metric_depth(
    depth_m: np.ndarray,
    *,
    backend: str,
    min_finite_fraction: float = 0.999,
    min_range_ratio: float = MIN_DEPTH_RANGE_RATIO,
) -> MetricDepth:
    """Valida e embrulha a profundidade métrica. Não conserta nada: rejeita.

    `min_range_ratio` pega cena degenerada (praticamente frontoparalela), em que
    a disparidade não varia e o mapa de defocus não carrega informação.

    Nota deliberada: `z_max == 10000` (teto do Depth Pro) NÃO é motivo de rejeição.
    Em disparidade `1/10000 ≈ 0` e o fundo satura graciosamente — é exatamente a
    vantagem de sair da profundidade linear. Céu no infinito é comportamento
    correto. O que é defeito é foco no teto (ver `focus_disparity_from_mask`) e
    faixa útil degenerada, que é o que este gate pega.
    """
    if backend not in DEPTH_BACKENDS:
        _reject("depth_backend_unregistered", f"{backend!r} não está em {sorted(DEPTH_BACKENDS)}")
    depth_m = np.asarray(depth_m, dtype=np.float32)
    finite = np.isfinite(depth_m)
    if finite.mean() < min_finite_fraction:
        _reject("depth_non_finite", f"fração finita {finite.mean():.4f}")
    positive = finite & (depth_m > 0)
    if positive.mean() < min_finite_fraction:
        _reject("depth_non_positive", f"fração positiva {positive.mean():.4f}")

    z_min = float(depth_m[positive].min())
    z_max = float(depth_m[positive].max())
    if z_min <= 0 or not np.isfinite(z_max):
        _reject("depth_non_positive", f"z_min={z_min}, z_max={z_max}")
    if z_max / z_min < min_range_ratio:
        _reject("depth_range_degenerate", f"z_max/z_min={z_max / z_min:.4f}")

    return MetricDepth(depth_m, z_min, z_max, backend)


# --------------------------------------------------------------------------------
# Plano de foco — Eq. 4, na disparidade
# --------------------------------------------------------------------------------

#: Limites de plausibilidade do plano de foco. [A] ASSUMIDOS, não medidos.
#: Escolhidos para pegar a sentinela clara do Depth Pro (10.000 m) sem descartar
#: paisagem legitimamente distante. A faixa medida de `z_focus_m` na tabela kfix é
#: 0,2191 a 10.000 m, então o topo real da distribuição legítima ainda não é conhecido.
#: Calibrar no piloto pelo histograma de `focus_depth_implausible` e congelar.
FOCUS_DEPTH_MIN_M = 0.05
FOCUS_DEPTH_MAX_M = 1000.0


def focus_disparity_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    min_focus_depth_m: float = FOCUS_DEPTH_MIN_M,
    max_focus_depth_m: float = FOCUS_DEPTH_MAX_M,
) -> float:
    """Eq. 4 do paper, calculada NA DISPARIDADE — como a inferência oficial.

    `Inference_bokehNet.py:118` faz `disp_focus = median(disp[mask])`. Isso NÃO é
    equivalente a `1 / median(z[mask])`: a mediana só é invariante sob transformação
    monótona no caso de contagem ímpar, e `np.median` faz a MÉDIA dos dois centrais
    quando é par — e a média de dois recíprocos não é o recíproco da média.

    A diferença é minúscula em máscara grande, e o ponto do contrato é justamente
    não ter esse tipo de folga.

    Os limites de plausibilidade pegam o teto do Depth Pro: um `z_focus` de 10.000 m
    não é plano de foco, é sentinela.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    selected = np.asarray(mask) > 0.5
    if selected.shape != depth_m.shape:
        _reject("focus_mask_empty", f"shape da máscara {selected.shape} != depth {depth_m.shape}")
    values = depth_m[selected]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        _reject("focus_mask_empty", "nenhum pixel válido dentro da máscara")

    focus_disparity = float(np.median(1.0 / values))
    if not np.isfinite(focus_disparity) or focus_disparity <= 0:
        _reject("focus_disparity_invalid", f"focus_disp={focus_disparity!r}")

    focus_depth_m = 1.0 / focus_disparity
    if not (min_focus_depth_m <= focus_depth_m <= max_focus_depth_m):
        _reject("focus_depth_implausible", f"z_focus={focus_depth_m:.3f} m")

    return focus_disparity


# --------------------------------------------------------------------------------
# Eq. 3 — bokeh level a partir da óptica
# --------------------------------------------------------------------------------

def sensor_width_mm(
    focal_length_mm: float,
    focal_length_35mm: Optional[float],
) -> float:
    """Largura física do sensor, via crop factor do campo de 35 mm equivalente.

    SEM FALLBACK. O pipeline antigo assumia 36 mm quando `focal_length_35` faltava,
    o que subestima `pixel_ratio` — logo K — por até 5,6x num celular.

    Aviso medido, e não resolvido aqui: 30,33% das 13.800 amostras da rota B têm
    crop factor EXATAMENTE 1,0. Isso é ou full-frame de verdade, ou a assinatura de
    uma câmera que ecoa a focal no campo de 35 mm quando não sabe o valor. Presença
    não é correção. A auditoria desses 30% é por tabela `make/model` e acontece a
    montante deste módulo, marcando a amostra antes de chegar aqui.
    """
    if focal_length_mm is None or not np.isfinite(focal_length_mm) or focal_length_mm <= 0:
        _reject("exif_focal_length_missing", f"f={focal_length_mm!r}")
    if focal_length_35mm is None or not np.isfinite(focal_length_35mm) or focal_length_35mm <= 0:
        _reject("sensor_width_unresolvable", "focal_length_35 ausente ou inválido")

    crop_factor = float(focal_length_35mm) / float(focal_length_mm)
    if not np.isfinite(crop_factor) or crop_factor <= 0:
        _reject("sensor_width_unresolvable", f"crop_factor={crop_factor!r}")
    return FULL_FRAME_WIDTH_MM / crop_factor


def pixel_ratio(image_hw: tuple[int, int], sensor_mm: float) -> float:
    """px/mm — MAIOR LADO dividido pela largura física do sensor.

    A Fig. 16 do paper: "the image's largest edge length divided by the physical
    sensor width (px/mm)". Usar a largura erra por H/W em retrato, e 32 de 100
    amostras conferidas da rota B são retrato.

    Identidade verificada em 900/900 linhas da tabela kfix:
        fx_px = f_mm * pixel_ratio = max(W,H) * focal_length_35 / 36
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        _reject("resolution_invalid", f"hw={image_hw!r}")
    if not np.isfinite(sensor_mm) or sensor_mm <= 0:
        _reject("sensor_width_unresolvable", f"sensor_mm={sensor_mm!r}")
    return max(height, width) / float(sensor_mm)


def k_eq3_mm(
    focal_length_mm: float,
    f_number: float,
    focus_depth_m: float,
    px_per_mm: float,
) -> float:
    """Eq. 3 do paper, na convenção MILÍMETRO. Unidade de saída: px*mm.

        K = f^2 * D_focus / (2*F*(D_focus - f)) * pixel_ratio

    Só vale para a rota B / ITW. A rota C usa a Eq. 5 (sweep de SSIM).

    Análise dimensional:
        f^2 * z / (2F(z - f))  ->  mm^2
        pixel_ratio            ->  px/mm
        produto                ->  px*mm
        vezes |Delta(1/mm)|    ->  px    OK

    O divisor `2F` significa que isto é RAIO, não diâmetro — e o renderer clássico
    do BokehMe espalha com raio, então os dois concordam.
    """
    if f_number is None or not np.isfinite(f_number) or f_number <= 0:
        _reject("exif_f_number_missing", f"F={f_number!r}")
    f_mm = float(focal_length_mm)
    focus_mm = float(focus_depth_m) * MM_PER_M
    if focus_mm <= f_mm:
        _reject("focus_distance_below_focal_length", f"z_focus={focus_mm:.1f}mm <= f={f_mm:.1f}mm")

    k = (f_mm ** 2 * focus_mm) / (2.0 * float(f_number) * (focus_mm - f_mm)) * float(px_per_mm)
    if not np.isfinite(k):
        _reject("k_non_finite", f"k_eq3={k!r}")
    if k <= 0:
        _reject("k_non_positive", f"k_eq3={k!r}")
    return float(k)


def k_official(k_eq3_value: float) -> float:
    """Converte `k_eq3` (px*mm, disparidade em 1/mm) para a convenção da inferência
    oficial (disparidade em 1/m).

    Âncoras: a mediana do `k_eq3` da tabela `rota-b-kfix-eq3` é 16.553,9, que dá
    16,55 aqui; a mediana calculada da EXIF dá 20,1; e o default de
    `Inference_bokehNet.py:53` é 15,0. Três caminhos independentes na mesma faixa.
    """
    return float(k_eq3_value) / MM_PER_M


def k_from_exif(
    *,
    focal_length_mm: float,
    f_number: float,
    focal_length_35mm: Optional[float],
    focus_depth_m: float,
    image_hw: tuple[int, int],
) -> tuple[float, dict]:
    """Caminho completo da Eq. 3 para a rota B. Devolve (K oficial, diagnóstico).

    O diagnóstico vai inteiro para os metadados da amostra: sem ele não dá para
    auditar depois de qual sensor veio o K.
    """
    sensor = sensor_width_mm(focal_length_mm, focal_length_35mm)
    px_mm = pixel_ratio(image_hw, sensor)
    k_mm = k_eq3_mm(focal_length_mm, f_number, focus_depth_m, px_mm)
    k = k_official(k_mm)
    diagnostics = {
        "k_eq3_mm": k_mm,
        "k_value": k,
        "sensor_width_mm": sensor,
        "crop_factor": FULL_FRAME_WIDTH_MM / sensor,
        "pixel_ratio_px_per_mm": px_mm,
        "longest_edge_px": max(int(image_hw[0]), int(image_hw[1])),
        "focal_length_mm": float(focal_length_mm),
        "f_number": float(f_number),
        "focal_length_35mm": float(focal_length_35mm),
        "focus_depth_m": float(focus_depth_m),
    }
    return k, diagnostics


# --------------------------------------------------------------------------------
# O mapa de defocus
# --------------------------------------------------------------------------------

def signed_coc_px(depth_m: np.ndarray, focus_disparity: float, k_value: float) -> np.ndarray:
    """CoC com sinal, em PIXELS, na resolução de `depth_m`.

        CoC = K * (1/z - focus_disp)

    O sinal separa frente do fundo do plano focal; o mapa de condição usa o módulo.
    """
    if not np.isfinite(k_value) or k_value <= 0:
        _reject("k_non_positive", f"K={k_value!r}")
    if not np.isfinite(focus_disparity) or focus_disparity <= 0:
        _reject("focus_disparity_invalid", f"focus_disp={focus_disparity!r}")
    disparity = 1.0 / np.asarray(depth_m, dtype=np.float32)
    return (float(k_value) * (disparity - float(focus_disparity))).astype(np.float32)


def defocus_map(
    depth_m: np.ndarray,
    focus_disparity: float,
    k_value: float,
) -> np.ndarray:
    """Condição final da BokehNet, em [0, 1].

    `MAX_COC` é GLOBAL e CONGELADO **por construção**: não é parâmetro desta função.
    Como parâmetro com default, um override parcial gravaria `max_coc: 100.0` nos
    metadados ao lado de um mapa normalizado por outro valor — e a proveniência
    mentiria sem que nada denunciasse.

    Normalizar por imagem apaga o K algebricamente
    — foi o defeito que deixou `max(defocus) == 65535` em todas as amostras, com k
    variando de 33 a 195. Normalizar por rota é o mesmo defeito com granularidade
    mais grossa, e foi o que o `max_coc = 10,5107` do kfix fez.
    """
    coc = np.abs(signed_coc_px(depth_m, focus_disparity, k_value))
    return np.clip(coc / MAX_COC, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------
# Resolução — a quantidade em pixel carrega a escala em que foi medida
# --------------------------------------------------------------------------------

def k_at_resolution(k_value: float, src_hw: tuple[int, int], dst_short_side: int) -> float:
    """Reescala K para a resolução em que o modelo realmente vê a imagem.

    CoC em pixel escala com a resolução; a disparidade não. Então ao reduzir o lado
    MENOR para `dst_short_side`, o K tem que ser multiplicado pelo mesmo fator.

        s = dst_short_side / min(H, W)
        K' = K * s

    O crop posterior não muda a escala, só a janela.

    Este é o fator que o pipeline antigo nunca aplicou: o dataloader reescalava o
    lado menor para 512 e recortava, com fator variando por amostra (medidos 0,892
    e 0,821), enquanto o K gravado continuava na escala da imagem original.

    O paper **não publica a resolução de treino**. O tiling da §3.5 é explicitamente
    de INFERÊNCIA ("a tiling strategy inspired by [5] **during inference**",
    paper.txt:481-482). Treinar em crop de 512 é DECISÃO NOSSA, e é por isso que este
    fator existe — não porque o paper o dispense.
    """
    height, width = int(src_hw[0]), int(src_hw[1])
    if height <= 0 or width <= 0 or int(dst_short_side) <= 0:
        _reject("resolution_invalid", f"src={src_hw!r} dst_short={dst_short_side!r}")
    scale = float(dst_short_side) / float(min(height, width))
    return float(k_value) * scale


def k_for_bokehme(k_value: float, disparity_min: float, disparity_max: float) -> float:
    """Converte K (convenção 1/m) para o `--K` do BokehMe, que recebe disparidade
    normalizada em [0, 1].

    Contrato do `demo.py`, verificado:
        defocus = K * (disp_norm - disp_focus) / defocus_scale
    e dentro do pipeline:
        classical_renderer(image ** gamma, defocus * defocus_scale)

    Logo o raio de borrão é `K * Delta_disp_norm` em pixels, e `defocus_scale` só
    normaliza a entrada da rede — se cancela. Para preservar o CoC canônico:

        k_renderer = K * (disp_max - disp_min)
    """
    span = float(disparity_max) - float(disparity_min)
    if not np.isfinite(span) or span <= 0:
        _reject("depth_range_degenerate", f"span de disparidade={span!r}")
    return float(k_value) * span


# --------------------------------------------------------------------------------
# Codificação em disco — um lugar só
# --------------------------------------------------------------------------------

UINT16_MAX = 65535


def encode_defocus_uint16(defocus: np.ndarray) -> np.ndarray:
    """[0,1] -> uint16. NÃO normaliza: o mapa já vem dividido pelo `max_coc` global."""
    return (np.clip(np.asarray(defocus, dtype=np.float32), 0.0, 1.0) * UINT16_MAX).astype(np.uint16)


def decode_defocus_uint16(encoded: np.ndarray) -> np.ndarray:
    """uint16 -> [0,1]. A BokehNet consome o mapa em [0,1] CRU: a inferência oficial
    usa `No_preprocess=True` e pula o passo [0,1] -> [-1,1]. A assimetria com a AIF,
    que entra em [-1,1], é deliberada."""
    return (np.asarray(encoded, dtype=np.float32) / UINT16_MAX).astype(np.float32)


def control_metadata(
    *,
    k_value: float,
    focus_disparity: float,
    depth: MetricDepth,
    k_source: str,
    is_k_censored: bool,
) -> dict:
    """Os escalares sem os quais o mapa é irrecuperável. Vão em TODA amostra.

    `focus_depth_m` está aqui só para leitura humana. O mapa usa `focus_disparity`;
    reconstruir com `1/focus_depth_m` reintroduz a diferença que o contrato elimina.
    """
    return {
        "control_version": CONTROL_VERSION,
        "k_value": float(k_value),
        "k_source": k_source,
        "focus_disparity": float(focus_disparity),
        "focus_depth_m": 1.0 / float(focus_disparity),
        "z_min_m": float(depth.min_m),
        "z_max_m": float(depth.max_m),
        "max_coc": MAX_COC,
        "depth_backend": depth.backend,
        "is_k_censored": bool(is_k_censored),
    }
