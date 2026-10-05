"""Rota (c) do paper — LFDOF e RealBokeh, pares reais, K pela Eq. 5.

    K* = argmax_K  SSIM( R(I_aif, D; D_focus, K),  I_real )

Fluxo por par:

    AIF real  ->  Depth Pro       ->  z métrico em metros
              ->  BiRefNet        ->  máscara inicial M
    AIF+bokeh ->  retenção        ->  região em foco pela FÍSICA (qc.focus_region)
              ->  refinamento     ->  região final + `focus_source` por amostra
              ->  Eq. 4           ->  focus_disparity = median(1/z[região final])
              ->  Eq. 5 + BokehMe ->  K*, SSIM, censura
              ->  gates           ->  medem sempre, bloqueiam só com limiar congelado
              ->  writer          ->  disparidade uint16 + escalares + proveniência

## O que esta rota NÃO faz, e por quê

**Não grava as imagens.** A AIF e a bokeh são reais e já vivem em
`akcit-pixel/RealBokeh` e no LFDOF. Regravá-las custaria 365 GB contra 115 GB de cota
livre, e criaria uma segunda cópia que pode divergir da primeira.

**Não grava o mapa de defocus.** Ele é derivado no dataloader pela mesma função da
geração. Gravá-lo criaria uma segunda fonte de verdade — o defeito D1.

**Não escolhe a AIF dentro de `gt/`.** Foi o D6: a mediana do maior f-stop por cena é
f/14, e em 12,7% das cenas é f/5.6 ou mais aberto — nessas, profundidade, máscara e
sweep saíam todos de uma foto com bokeh forte. A AIF vem de `train/in/<cena>_f22.JPG`,
que é f/22 em 100% das cenas, e o espelho `akcit-pixel/RealBokeh` já a traz em
`image_focus` (confirmado byte a byte por sha256).

**Não descarta em silêncio.** Toda rejeição tem slug do conjunto fechado e entra no
histograma que o run imprime no fim. Sem esse histograma não dá para calibrar limiar
nenhum, e é ele que denuncia fallback novo.

## Substituição declarada — o refinamento da máscara

O §3.2(c) do paper faz refinamento MANUAL da máscara e diz textualmente: *"Rather than
simply verifying and discarding unreliable cases [...] preserves challenging samples
rather than excluding them"*.

Até a etapa 8 nós **descartávamos** onde o paper **corrigia**, e o piloto mediu o preço:
a máscara do BiRefNet acerta o plano de foco em **35,2%** das amostras (contra a distância
de foco que a RealBokeh publica medida a ±0,010 m), e **20,6%** eram descartadas com
`focus_mask_empty` porque o BiRefNet devolvia probabilidade exatamente 0 — ele é
segmentador de objeto saliente, e a RealBokeh é feita de cenas. Ver
`reference/MEDICAO_PLANO_FOCO.md`.

Agora a rota chama `qc.focus_region.refine_focus_mask`, que é o substituto **automático**
daquele passo manual (22.990 correções à mão não vão acontecer). O desvio que resta é
declarado e estreito: o paper corrige à mão, nós corrigimos por retenção de detalhe.

### Qual máscara vai para o disco, e por quê

`mask/<id>.png` guarda a **região refinada** — a mesma que produziu `focus_disparity` —
e `mask_source` diz qual das três origens ela tem (`birefnet`, `birefnet_refined`,
`retention_only`). A alternativa era gravar a máscara crua do BiRefNet; foi recusada
porque o arquivo passaria a ser uma máscara que **não corresponde ao rótulo**, e quem
auditasse `focus_disparity` olharia a máscara errada sem nada denunciar. A máscara crua
não se perde como medida: `focus_agreement`, `focus_initial_mask_area_ratio` e
`focus_initial_mask_was_empty` a descrevem no metadado, e a IoU AIF/bokeh continua sendo
calculada sobre ela.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import numpy as np

from control.contract import (
    SampleRejected, focus_disparity_from_mask, reject,
)
from dataio import (
    ControlLabel, DEPTH_LONG_SIDE, DepthLayout, FocusRegionRecord, KSource, Sample,
    SampleProvenance, SampleRefs, SceneSplit, FileSampleWriter, encode_depth,
)
from qc.focus_region import (
    DEFAULT_AGREEMENT_FLOOR, DEFAULT_TOP_FRACTION, DEFAULT_WINDOW_PX,
    DEFAULT_MIN_DETAIL_PERCENTILE, MIN_REGION_AREA_RATIO, FocusSource,
    detail_maps, refine_focus_mask,
)
from qc.gates import (
    GateReport, aif_aperture_is_narrow, aif_sharpness, bokeh_is_blurrier_than_aif,
    calibration_ssim_is_reliable, depth_useful_levels, focus_depth_plausible,
    focus_mask_is_sharpest, focus_region_retention, mask_area_ratio,
    mask_border_coverage, mask_iou, pair_shape_matches,
)
from qc.rejection import RejectionLog
from renderer.calibration import (
    K_ABSOLUTE_MAX_DEFAULT, K_MAX_DEFAULT, K_MIN_DEFAULT, calibrate_k,
    resize_area_for_photo, resize_nearest,
)


class PairSource(Protocol):
    """O que a rota C precisa saber de um par, e nada mais.

    Protocolo em vez de import concreto: `sources/realbokeh.py` e o adaptador do LFDOF
    entregam a mesma forma, e a rota não precisa conhecer nenhum dos dois.
    """

    scene_id: str
    sample_id: str
    source_dataset: str
    source_sample_id: str
    source_split: Optional[str]
    aif_ref: Optional[str]
    bokeh_ref: Optional[str]
    #: f-number do ALVO, quando a origem publica. `None` é legítimo (LFDOF não tem).
    f_number: Optional[float]
    #: os três termos da Eq. 3, quando a origem publica — usados como VALIDADOR do
    #: sweep, nunca como rótulo. O rótulo é a Eq. 5, fiel ao §3.2(c).
    focal_length_mm: Optional[float]
    focus_plane_distance_m: Optional[float]
    #: f-number da AIF. Na RealBokeh é 22,0 (`train/in/`); serve ao gate do D6.
    aif_f_number: Optional[float]


#: Carrega as duas imagens de um par, em BGR uint8. Fica fora da rota de propósito:
#: ler do espelho HF, de disco local ou de um mock de teste é decisão de quem chama.
LoadPair = Callable[[PairSource], tuple[np.ndarray, np.ndarray]]


@dataclass
class RouteCConfig:
    """Todos os botões. **Todo limiar é `None` por default** — mede no piloto,
    congela depois. É a instrução explícita: testar primeiro, filtros depois."""

    output_dir: Path
    depth_long_side: int = DEPTH_LONG_SIDE
    #: **Por cena.** Os níveis de uma cena são a MESMA AIF com aberturas diferentes — o
    #: `image_focus` é byte a byte idêntico entre os níveis de uma cena, medido em 90
    #: linhas de 9 cenas do espelho (`sources/lfdof.py:64-71`), e a máscara do §3.2(c)
    #: é uma por cena pela mesma razão (`sources/level_selection.py:26-29`). Com o teto
    #: de 4 níveis do paper são 3,51 amostras por cena no release publicado (15.423 em
    #: 4.399), então `depth/` cai de 15.423 para 4.399 arquivos.
    depth_layout: DepthLayout = DepthLayout.PER_SCENE
    calibration_long_side: int = 512
    k_min: float = K_MIN_DEFAULT
    k_max: float = K_MAX_DEFAULT
    k_absolute_max: float = K_ABSOLUTE_MAX_DEFAULT

    # --- limiares, todos [A] até o piloto ---
    min_calibration_ssim: Optional[float] = None
    min_focus_mask_sharpness_ratio: Optional[float] = None
    min_mask_area_ratio: Optional[float] = None
    max_mask_area_ratio: Optional[float] = None
    max_mask_border_coverage: Optional[float] = None
    min_aif_laplacian_variance: Optional[float] = None
    max_bokeh_over_aif_sharpness: Optional[float] = None
    min_depth_useful_levels: Optional[int] = None
    min_aif_f_number: Optional[float] = None
    min_mask_iou: Optional[float] = None
    #: Piso de retenção de detalhe DENTRO da região que definiu `D_focus`. É o limiar
    #: certo para julgar a região refinada — normalizado pela textura da cena, ao
    #: contrário da nitidez absoluta. `[A]` como todos os outros.
    min_focus_region_retention: Optional[float] = None

    # --- refinamento da região em foco (§3.2(c)) ---
    #: Resolução de trabalho da retenção, pelo lado longo, no **mesmo padrão de
    #: `calibrate_k(work_long_side=...)`**. `None` = resolução cheia, e é o default —
    #: **por medição, não por preferência**.
    #:
    #: A hipótese era que a retenção fosse caro em resolução cheia. Medido nesta
    #: máquina, a 1500x2000, que é a resolução do espelho da RealBokeh:
    #:
    #:     detail_retention a 1500x2000 .............. 291 ms
    #:     detail_retention a  384x 512 ...............  22 ms   (13x mais barato)
    #:     resize_area_for_photo de UMA foto de 3 MP . 185 ms
    #:
    #: São **duas** fotos a reduzir, então o caminho reduzido custa ~370 ms de redução
    #: para economizar ~270 ms de retenção. Ponta a ponta em `refine_focus_region`:
    #: **433 ms reduzido contra 391 ms em resolução cheia**. A resolução de trabalho
    #: seria uma perda de precisão paga com tempo a mais.
    #:
    #: O knob fica, e há dois motivos legítimos para usá-lo: uma origem de resolução
    #: maior que a RealBokeh, e o desalinhamento AIF/bokeh que a própria origem anota em
    #: 421 pares — reduzir atenua o efeito dele na razão bokeh/AIF (`[A]`, não medido).
    #: Qualquer que seja o valor, a grade usada vai para o metadado: quantidade em pixel
    #: sem a resolução ao lado não diz nada.
    focus_retention_long_side: Optional[int] = None
    focus_retention_window_px: int = DEFAULT_WINDOW_PX
    focus_top_fraction: float = DEFAULT_TOP_FRACTION
    #: Percentil de detalhe da AIF abaixo do qual o pixel não concorre à região.
    #: Relativo à própria imagem. Sem ele, a seleção escorregava para superfície
    #: lisa, onde a retenção vale ~1 por ser ruído sobre ruído — medido: 42,9% da
    #: região caindo no céu desfocado. `[A]`, e o primeiro a calibrar no piloto.
    focus_min_detail_percentile: float = DEFAULT_MIN_DETAIL_PERCENTILE
    focus_agreement_floor: float = DEFAULT_AGREEMENT_FLOOR
    focus_min_region_area_ratio: float = MIN_REGION_AREA_RATIO

    limit: Optional[int] = None
    seed: int = 0


@dataclass
class RouteCStats:
    processed: int = 0
    written: int = 0
    skipped_done: int = 0
    k_values: list[float] = field(default_factory=list)
    #: Eq. 3 sobre o metadata da origem. Acumulado separado de `k_values` de propósito:
    #: são grandezas DIFERENTES (absoluto x incremental), e misturá-las numa estatística
    #: só produziria um número que não descreve nem uma nem outra.
    k_analytic_values: list[float] = field(default_factory=list)
    ssim_values: list[float] = field(default_factory=list)
    censored: int = 0
    #: Quantas amostras aceitas por `focus_source`. Contado no laço a partir do
    #: metadado GRAVADO, não do objeto em memória: assim o número que o log imprime é
    #: o mesmo que está no disco.
    focus_sources: Counter = field(default_factory=Counter)
    #: Quantas entraram com máscara do BiRefNet vazia — as que antes eram DESCARTADAS.
    focus_initial_empty: int = 0

    def focus_summary(self) -> list[str]:
        """O bloco de `focus_source` do resumo, em contagem e percentual.

        Separado de `summary()` para poder ser testado sem montar um run inteiro, e
        porque este bloco é o instrumento que denuncia o refinamento carregando o lote:
        se `retention_only` for alto, isso significa que o BiRefNet está declinando na
        maioria das cenas, e a decisão de seguir ou parar é humana. Um número assim não
        pode ficar escondido num JSON que ninguém abre.
        """
        total = sum(self.focus_sources.values())
        if not total:
            return []
        lines = ["-" * 62,
                 "  região em foco (§3.2(c), substituto do refino manual):"]
        legenda = {
            FocusSource.BIREFNET.value: "máscara do BiRefNet aceita como veio",
            FocusSource.BIREFNET_REFINED.value: "BiRefNet ∩ retenção — refinada",
            FocusSource.RETENTION_ONLY.value: "só retenção — ANTES eram descartadas",
        }
        for source in (FocusSource.BIREFNET, FocusSource.BIREFNET_REFINED,
                       FocusSource.RETENTION_ONLY):
            n = self.focus_sources.get(source.value, 0)
            lines.append(f"    {source.value:<18} {n:>6}  ({100 * n / total:5.1f}%)  "
                         f"{legenda[source.value]}")
        # Uma fonte que não está no enum não pode ser silenciada pelo laço acima.
        for nome, n in sorted(self.focus_sources.items()):
            if nome not in legenda:
                lines.append(f"    {nome:<18} {n:>6}  ({100 * n / total:5.1f}%)  "
                             "FONTE DESCONHECIDA — vocabulário aberto é defeito")
        refinadas = total - self.focus_sources.get(FocusSource.BIREFNET.value, 0)
        lines.append(f"    refinadas no total {refinadas:>6}  "
                     f"({100 * refinadas / total:5.1f}%)  filtráveis por "
                     "`focus_was_refined` no manifesto")
        lines.append(f"    máscara inicial vazia {self.focus_initial_empty:>3}  "
                     f"({100 * self.focus_initial_empty / total:5.1f}%)  "
                     "no piloto anterior eram 20,6% DESCARTADAS")

        # Um lote em que a retenção decide sozinha na maioria das amostras é um lote em
        # que o BiRefNet praticamente não participou. Não é motivo para parar o run — o
        # §3.2(c) manda preservar a amostra —, é motivo para não publicar antes do laudo.
        so_retencao = self.focus_sources.get(FocusSource.RETENTION_ONLY.value, 0)
        if so_retencao > total / 2:
            lines.append(f"    >>> ATENÇÃO: em {100 * so_retencao / total:.1f}% das "
                         "amostras o BiRefNet NÃO participou da escolha da região.")
            lines.append("        Rode scripts/validate_focus_refinement.py antes de "
                         "usar este lote como treino.")
        return lines

    def summary(self) -> str:
        if not self.k_values:
            return "[rota-c] nenhuma amostra aceita."
        k = np.asarray(self.k_values)
        s = np.asarray([v for v in self.ssim_values if v is not None])
        lines = [
            "",
            "=" * 62,
            f"  aceitas          : {self.written}",
            f"  já feitas (skip) : {self.skipped_done}",
            f"  censuradas       : {self.censored}"
            f"  ({100 * self.censored / max(self.written, 1):.1f}%)",
            "-" * 62,
            "  k_value   p05 {:.2f}  mediana {:.2f}  p95 {:.2f}".format(
                *np.percentile(k, [5, 50, 95])),
            "            (sweep da Eq. 5 — borrão INCREMENTAL sobre a AIF)",
        ]
        # A âncora "3,6 a 36" é de `k_analytic`, que é a Eq. 3: borrão ABSOLUTO do
        # alvo. Colar as duas na mesma tabela convida a conclusão errada de que o
        # sweep está quebrado. Ver `_analytic_k` e a nota abaixo.
        if self.k_analytic_values:
            a = np.asarray(self.k_analytic_values)
            lines.append("  k_analytic p05 {:.2f}  mediana {:.2f}  p95 {:.2f}".format(
                *np.percentile(a, [5, 50, 95])))
            lines.append("            (Eq. 3 — borrão ABSOLUTO. NÃO é comparável com "
                         "k_value:")
            lines.append("             a AIF da RealBokeh é f/22, não all-in-focus, "
                         "então o sweep")
            lines.append("             mede só o que FALTA. Espera-se "
                         "k_value < k_analytic.)")
        if s.size:
            lines.append("  ssim      p05 {:.3f}  mediana {:.3f}  p95 {:.3f}".format(
                *np.percentile(s, [5, 50, 95])))
            lines.append("            o paper exige limiar (§3.2(c)) e não publica o valor")
        lines.extend(self.focus_summary())
        lines.append("=" * 62)
        return "\n".join(lines)


def build_gate_report(
    *, aif_bgr: np.ndarray, bokeh_bgr: np.ndarray, mask: np.ndarray,
    mask_initial: Optional[np.ndarray] = None,
    mask_bokeh: Optional[np.ndarray], disparity_u16: np.ndarray,
    focus_disparity: float, calibration_ssim: Optional[float],
    retention_in_region: Optional[float] = None,
    pair: PairSource, config: RouteCConfig,
) -> GateReport:
    """Todos os gates, na ordem em que valem a pena olhar no histograma.

    **Qual gate julga qual máscara**, e por quê — a distinção importa porque a rota
    passou a ter duas máscaras por amostra:

    * `mask` é a região FINAL de foco, a que produziu `focus_disparity` e a que vai
      para `mask/<id>.png`. Julgam-na `mask_area_ratio_*` ("máscara vazia não define
      plano de foco" — a afirmação é sobre a região que define o plano) e
      `focus_mask_sharpness_ratio`.
    * `mask_initial` e `mask_bokeh` são as máscaras cruas do BiRefNet na AIF e na
      bokeh. Julgam-nas `mask_border_coverage` — cuja heurística ("colada na borda
      costuma ser fundo, não objeto") só faz sentido para um segmentador de objeto, e
      penalizaria uma região de retenção que legitimamente toca a borda — e
      `mask_iou_aif_bokeh`, que por definição compara as duas máscaras do segmentador.
      Sem `mask_initial`, os dois caem na região final, para a rota A e a B seguirem
      valendo sem mudança.
    * `focus_region_retention` julga a região final com a grandeza **normalizada pela
      textura da cena** — o único limiar que não cai na armadilha da nitidez absoluta.
    """
    report = GateReport()
    inicial = mask if mask_initial is None else mask_initial
    report.add(pair_shape_matches(aif_bgr, bokeh_bgr))
    report.add(aif_aperture_is_narrow(pair.aif_f_number,
                                      min_f_number=config.min_aif_f_number))
    report.add(bokeh_is_blurrier_than_aif(
        aif_bgr, bokeh_bgr, max_ratio=config.max_bokeh_over_aif_sharpness))
    report.add(aif_sharpness(aif_bgr, min_variance=config.min_aif_laplacian_variance))

    for result in mask_area_ratio(mask, min_ratio=config.min_mask_area_ratio,
                                  max_ratio=config.max_mask_area_ratio):
        report.add(result)
    report.add(mask_border_coverage(inicial, max_ratio=config.max_mask_border_coverage))

    # O gate que confere a HIPÓTESE: a máscara marca mesmo a região em foco?
    report.add(focus_mask_is_sharpest(
        bokeh_bgr, mask, min_ratio=config.min_focus_mask_sharpness_ratio))
    if retention_in_region is not None:
        report.add(focus_region_retention(
            retention_in_region, min_retention=config.min_focus_region_retention))
    if mask_bokeh is not None:
        report.add(mask_iou(inicial, mask_bokeh, min_iou=config.min_mask_iou))

    report.add(depth_useful_levels(disparity_u16, min_levels=config.min_depth_useful_levels))
    for result in focus_depth_plausible(focus_disparity):
        report.add(result)
    report.add(calibration_ssim_is_reliable(calibration_ssim,
                                            min_ssim=config.min_calibration_ssim))
    return report


#: Nome de gate -> slug de rejeição do conjunto fechado. Sem esta ponte, o histograma
#: receberia nomes de gate que não são slugs e viraria vocabulário aberto.
GATE_TO_REASON = {
    "pair_shape_matches": "gate_pair_shape_mismatch",
    "aif_f_number": "gate_aif_aperture_wide",
    "bokeh_over_aif_sharpness": "gate_bokeh_not_blurrier",
    "aif_laplacian_variance": "gate_aif_sharpness",
    "mask_area_ratio_min": "gate_mask_area_ratio",
    "mask_area_ratio_max": "gate_mask_area_ratio",
    "mask_border_coverage": "gate_mask_border_coverage",
    "focus_mask_sharpness_ratio": "gate_focus_mask_not_sharpest",
    "focus_region_retention": "gate_focus_region_retention_low",
    "mask_iou_aif_bokeh": "gate_mask_iou",
    "depth_useful_levels": "gate_depth_useful_levels",
    "focus_depth_m_min": "gate_focus_depth_implausible",
    "focus_depth_m_max": "gate_focus_depth_implausible",
    "calibration_ssim": "gate_calibration_ssim_below_floor",
}


def enforce_gates(report: GateReport) -> None:
    blocked = report.blocked_by
    if blocked:
        reject(GATE_TO_REASON[blocked[0]], f"gates reprovados: {blocked}")


def _work_hw(image_hw: tuple[int, int], long_side: Optional[int]) -> tuple[int, int]:
    """Resolução de trabalho pelo lado longo. Mesma regra de `calibrate_k`."""
    h, w = image_hw
    if not long_side or max(h, w) <= long_side:
        return (int(h), int(w))
    escala = long_side / float(max(h, w))
    return (max(1, round(h * escala)), max(1, round(w * escala)))


def refine_focus_region(
    *,
    aif_bgr: np.ndarray,
    bokeh_bgr: np.ndarray,
    mask_initial: Optional[np.ndarray],
    depth_m: np.ndarray,
    config: RouteCConfig,
) -> tuple[np.ndarray, FocusRegionRecord]:
    """Substituto automático do refinamento manual do §3.2(c), plugado na rota C.

    Devolve `(região final na resolução da IMAGEM, marcação por amostra)`.

    Três coisas acontecem aqui, e nenhuma delas é decisão de `qc.focus_region`:

    1. **A resolução em que a retenção é medida.** Default é a resolução cheia, porque
       medir a 512 saiu **mais caro** (433 ms contra 391 ms: reduzir duas fotos de 3 MP
       custa mais que a retenção que a redução economiza — ver
       `RouteCConfig.focus_retention_long_side`). A grade usada vai para o metadado em
       `focus_retention_h/w` de qualquer jeito, porque `focus_retention_window_px` é
       quantidade em pixel e sem a resolução ao lado ela não significa nada: a 1500x2000
       uma janela de 33 px é 1,7% do lado longo, a 512x683 é 6,4%.
    2. **A volta para a resolução da imagem.** A região sai da grade de trabalho por
       vizinho mais próximo, nunca interpolada: a região é binária, e interpolar
       produziria valores intermediários que um `> 0.5` depois arredondaria de um jeito
       que ninguém declarou. `focus_disparity_from_mask` exige a mesma grade da
       profundidade, e a profundidade vem na resolução da imagem.
    3. **O diagnóstico pareado.** `disparity_from_initial_mask` é o que o rótulo teria
       sido SEM refinamento, na mesma amostra. É o que permite a
       `scripts/validate_focus_refinement.py` responder "melhorou?" comparando a
       amostra com ela mesma, em vez de comparar o subgrupo refinado contra o não
       refinado — que é um subgrupo selecionado por já concordar, e portanto mais fácil
       por construção. **Nunca é rótulo.**
    """
    image_hw = (int(aif_bgr.shape[0]), int(aif_bgr.shape[1]))
    work_hw = _work_hw(image_hw, config.focus_retention_long_side)

    if work_hw == image_hw:
        aif_work, bokeh_work = aif_bgr, bokeh_bgr
    else:
        # Foto reduz por média de bloco — a MESMA função do sweep da Eq. 5. Vizinho
        # mais próximo numa foto natural cria aliasing, e aliasing entra direto no
        # laplaciano, que é a grandeza que a retenção mede.
        aif_work = resize_area_for_photo(aif_bgr, work_hw)
        bokeh_work = resize_area_for_photo(bokeh_bgr, work_hw)

    # Os DOIS mapas: a retenção sozinha não distingue plano de foco de superfície lisa
    # (medido: 79,7% dos pixels empatando em 1,000, 42,9% da região indo para o céu).
    # O detalhe da AIF é o que diz onde a razão significa alguma coisa.
    retention, aif_detail = detail_maps(aif_work, bokeh_work,
                                        window=config.focus_retention_window_px)

    inicial = None if mask_initial is None else np.asarray(mask_initial) > 0.5
    inicial_work = (None if inicial is None
                    else (inicial if work_hw == image_hw
                          else resize_nearest(inicial, work_hw)))

    region = refine_focus_mask(
        inicial_work, retention, aif_detail,
        top_fraction=config.focus_top_fraction,
        min_detail_percentile=config.focus_min_detail_percentile,
        agreement_floor=config.focus_agreement_floor,
        min_area_ratio=config.focus_min_region_area_ratio,
    )

    focus_mask = (region.mask if work_hw == image_hw
                  else resize_nearest(region.mask, image_hw))
    if focus_mask.shape != image_hw:
        reject("resolution_invalid",
               f"região refinada {focus_mask.shape} != imagem {image_hw}")

    record = FocusRegionRecord.from_region(
        region,
        retention_hw=work_hw,
        retention_window_px=config.focus_retention_window_px,
        # `None`, e não 0,0, quando não houve máscara inicial: "o segmentador não
        # rodou" é diferente de "rodou e devolveu vazio", e o segundo caso é 20,6% do
        # piloto — o número que precisa continuar contável.
        initial_mask_area_ratio=(None if inicial_work is None
                                 else float(np.mean(inicial_work))),
        disparity_from_initial_mask=_baseline_focus_disparity(depth_m, inicial),
    )
    return focus_mask, record


def _focus_provenance(config: RouteCConfig, record: FocusRegionRecord) -> dict:
    """Os parâmetros `[A]` do refinamento, por amostra.

    Nenhum deles vem do paper, que não publica nada sobre o passo de refinamento — então
    são justamente eles que alguém vai querer conferir quando dois lotes discordarem.
    """
    return {
        "top_fraction": float(config.focus_top_fraction),
        "agreement_floor": float(config.focus_agreement_floor),
        "min_region_area_ratio": float(config.focus_min_region_area_ratio),
        "window_px": int(record.retention_window_px),
        "retention_h": int(record.retention_hw[0]),
        "retention_w": int(record.retention_hw[1]),
        "long_side_requested": (None if config.focus_retention_long_side is None
                                else int(config.focus_retention_long_side)),
        "evidence": "[A] nenhum destes valores vem do paper",
    }


def _baseline_focus_disparity(depth_m: np.ndarray,
                              mask_initial: Optional[np.ndarray]) -> Optional[float]:
    """A Eq. 4 sobre a máscara CRUA do BiRefNet — diagnóstico, nunca rótulo.

    `None` quando não existe valor: máscara vazia (o caso que antes era descartado, e
    onde por definição não há linha de base), ou a Eq. 4 rejeitaria o valor por
    implausível. Devolver `None` aqui não é fallback — é a ausência declarada de uma
    medida de diagnóstico, e quem consome (o script de validação) conta quantas
    amostras ficaram sem ela em vez de assumir um número.
    """
    if mask_initial is None or not mask_initial.any():
        return None
    if mask_initial.shape != depth_m.shape:
        # Grade diferente aqui é DEFEITO, não ausência de medida: devolver `None` em
        # silêncio esconderia uma máscara e uma profundidade em resoluções distintas,
        # que é a família do D12. `reject` para o slug entrar no histograma.
        reject("resolution_invalid",
               f"máscara inicial {mask_initial.shape} != profundidade {depth_m.shape} "
               "no diagnóstico pareado")
    try:
        return float(focus_disparity_from_mask(depth_m, mask_initial))
    except SampleRejected:
        return None


def process_pair(
    pair: PairSource,
    *,
    aif_bgr: np.ndarray,
    bokeh_bgr: np.ndarray,
    depth_runtime,
    mask_runtime,
    render_fn,
    config: RouteCConfig,
    provenance_base: dict,
) -> Sample:
    """Um par -> uma amostra. Levanta `SampleRejected` com slug em qualquer falha."""
    if aif_bgr.shape[:2] != bokeh_bgr.shape[:2]:
        reject("resolution_invalid",
               f"aif {aif_bgr.shape[:2]} != bokeh {bokeh_bgr.shape[:2]}")
    image_hw = (int(aif_bgr.shape[0]), int(aif_bgr.shape[1]))
    aif_rgb = np.ascontiguousarray(aif_bgr[..., ::-1])

    # Profundidade e máscara vêm da AIF — a Fig. 3(b) é explícita na ordem, e usar a
    # máscara da bokeh com a profundidade da AIF foi o defeito D12.
    depth = depth_runtime.infer(aif_rgb)
    mask_initial = mask_runtime.infer(aif_rgb)

    # O substituto automático do refinamento MANUAL do §3.2(c). Máscara vazia deixou de
    # ser descarte: ela vira `retention_only`, que é o que o paper manda fazer.
    focus_mask, focus_record = refine_focus_region(
        aif_bgr=aif_bgr, bokeh_bgr=bokeh_bgr, mask_initial=mask_initial,
        depth_m=depth.values_m, config=config)
    focus_disparity = focus_disparity_from_mask(depth.values_m, focus_mask)

    # A máscara da bokeh serve SÓ para medir divergência. Ela não define o foco.
    mask_bokeh = mask_runtime.infer(np.ascontiguousarray(bokeh_bgr[..., ::-1]))

    calibration = calibrate_k(
        render_fn, aif_bgr=aif_bgr, target_bgr=bokeh_bgr, depth_m=depth.values_m,
        focus_disparity=focus_disparity, k_min=config.k_min, k_max=config.k_max,
        k_absolute_max=config.k_absolute_max,
        work_long_side=config.calibration_long_side,
    )

    encoded = encode_depth(depth.values_m, image_hw=image_hw,
                           long_side=config.depth_long_side)

    report = build_gate_report(
        aif_bgr=aif_bgr, bokeh_bgr=bokeh_bgr, mask=focus_mask,
        mask_initial=mask_initial, mask_bokeh=mask_bokeh,
        disparity_u16=encoded.disparity_u16, focus_disparity=focus_disparity,
        calibration_ssim=calibration.calibration_ssim,
        retention_in_region=focus_record.retention_in_region,
        pair=pair, config=config,
    )
    enforce_gates(report)

    # K analítico pela Eq. 3, quando a origem publica os três termos. É VALIDADOR do
    # sweep, nunca rótulo: o rótulo é a Eq. 5, fiel ao §3.2(c).
    k_analytic = _analytic_k(pair, image_hw)

    return Sample(
        sample_id=pair.sample_id, route="c",
        refs=SampleRefs(pair.source_dataset, pair.source_sample_id, pair.scene_id,
                        aif_ref=pair.aif_ref, bokeh_ref=pair.bokeh_ref,
                        source_split=pair.source_split),
        control=ControlLabel(
            k_value=calibration.k_value, k_source=KSource.EQ5_SSIM_SWEEP,
            focus_disparity=focus_disparity, is_k_censored=calibration.is_censored,
            depth_backend=depth.backend, calibration_ssim=calibration.calibration_ssim,
            k_analytic=k_analytic,
            k_effective_factor=provenance_base.get("k_effective_factor"),
        ),
        # A máscara gravada é a região FINAL de foco, e `mask_source` sai do
        # `focus_source` pela ponte única de `dataio.FOCUS_SOURCE_TO_MASK_SOURCE`.
        # Gravar a máscara crua aqui faria de `mask/<id>.png` uma máscara que não
        # produziu `focus_disparity` — proveniência que mente.
        depth=encoded, mask=focus_mask, mask_source=focus_record.mask_source,
        focus=focus_record,
        provenance=SampleProvenance(
            pipeline_commit=provenance_base["pipeline_commit"],
            depth_model_sha256=provenance_base["depth_model_sha256"],
            mask_model_sha256=provenance_base["mask_model_sha256"],
            image_hw=image_hw, seed=config.seed,
            renderer=provenance_base.get("renderer"),
            source_license=provenance_base.get("source_license"),
            extra={"k_search": calibration.search,
                   "mask_backend": provenance_base.get("mask_backend"),
                   # Os parâmetros `[A]` que produziram a região desta amostra. Ficam
                   # NA AMOSTRA, e não só no `run_config.json` do run: dois runs com
                   # `focus_top_fraction` diferente produzem rótulos diferentes, e um
                   # release remontado de dois runs precisa poder ser separado.
                   "focus_refinement": _focus_provenance(config, focus_record)},
        ),
        quality=report.to_dict(),
    )


def _analytic_k(pair: PairSource, image_hw: tuple[int, int]) -> Optional[float]:
    """Eq. 3 sobre o `metadata/` da origem — validador independente do sweep.

    Falta de sensor não rejeita a amostra: o rótulo não depende disto. Devolve `None`
    e o histograma do run mostra quantas amostras ficaram sem validador.

    **`focus_plane_distance_m` é MEDIDO na cena real, e é de propósito.** Na rota B a
    Eq. 3 precisa de uma distância de foco que só existe estimada, e ali há uma escolha
    real a fazer entre `1/focus_disparity` e `median(z)` — que não são a mesma coisa,
    porque a mediana não comuta com a inversão. Aqui não há escolha a fazer: a origem
    publica a distância medida, e é ela que entra.

    É justamente isso que torna este validador independente. Alimentá-lo com
    `1/focus_disparity` o faria depender do Depth Pro e da máscara — os mesmos dois
    modelos que produzem o valor sendo validado —, e ele deixaria de validar coisa
    alguma. Não "harmonize" esta linha com a da rota B.
    """
    from control.contract import k_eq3_mm, k_official, pixel_ratio

    if not (pair.focal_length_mm and pair.f_number and pair.focus_plane_distance_m):
        return None
    sensor_mm = getattr(pair, "sensor_width_mm", None)
    if not sensor_mm:
        return None
    try:
        px_mm = pixel_ratio(image_hw, float(sensor_mm))
        return k_official(k_eq3_mm(float(pair.focal_length_mm), float(pair.f_number),
                                   float(pair.focus_plane_distance_m), px_mm))
    except SampleRejected:
        return None


def run_route_c(
    pairs: Iterable[PairSource],
    *,
    load_pair: LoadPair,
    depth_runtime,
    mask_runtime,
    render_fn,
    split: SceneSplit,
    config: RouteCConfig,
    provenance_base: dict,
) -> RouteCStats:
    """Loop principal. Imprime o histograma de motivos de rejeição no fim, sempre."""
    stats = RouteCStats()
    log = RejectionLog(Path(config.output_dir) / "rejections.jsonl")
    writer = FileSampleWriter(config.output_dir, split=split,
                              depth_layout=config.depth_layout)
    done = writer.completed_ids()
    if done:
        print(f"[rota-c] retomando: {len(done)} amostras já gravadas serão puladas.")

    try:
        for pair in pairs:
            if config.limit is not None and stats.written >= config.limit:
                break
            if pair.sample_id in done:
                stats.skipped_done += 1
                continue
            stats.processed += 1
            try:
                aif_bgr, bokeh_bgr = load_pair(pair)
                sample = process_pair(
                    pair, aif_bgr=aif_bgr, bokeh_bgr=bokeh_bgr,
                    depth_runtime=depth_runtime, mask_runtime=mask_runtime,
                    render_fn=render_fn, config=config,
                    provenance_base=provenance_base)
                meta = writer.write(sample)
            except SampleRejected as exc:
                log.reject_from(pair.sample_id, exc,
                                {"scene_id": pair.scene_id, "route": "c"})
                continue
            log.accept(pair.sample_id, {"scene_id": pair.scene_id,
                                        "k_value": meta["k_value"]})
            stats.written += 1
            stats.k_values.append(meta["k_value"])
            if meta.get("k_analytic") is not None:
                stats.k_analytic_values.append(meta["k_analytic"])
            stats.ssim_values.append(meta["calibration_ssim"])
            stats.censored += int(meta["is_k_censored"])
            # Contado do metadado GRAVADO, não do objeto em memória: o número que o log
            # imprime é o que está no disco, e o manifesto carrega o mesmo campo.
            stats.focus_sources[meta["focus_source"]] += 1
            stats.focus_initial_empty += int(meta["focus_initial_mask_was_empty"])
    finally:
        print(writer.stats.summary())
        print(log.summary())
        print(stats.summary())
        writer.close()
        log.close()
    return stats
