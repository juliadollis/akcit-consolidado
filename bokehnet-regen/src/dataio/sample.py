"""O que UMA amostra é — o contrato de dados, em código.

Regra que governa este módulo: **guardamos o que geramos, referenciamos o que já
existe.** Reescrever as imagens de origem custaria 365 GB contra 115 GB de cota livre,
e criaria uma segunda cópia que pode divergir da primeira.

| rota | bokeh | AIF | por quê |
|---|---|---|---|
| C | referência | referência | os dois são reais e vivem em `akcit-pixel/RealBokeh` e no LFDOF |
| B | referência | **gravada** | a AIF é produzida pela nossa DeblurNet |
| A | **gravado** | referência | o bokeh é renderizado por nós |

O mapa de defocus **não** é gravado. Ele é derivável de
`disparidade + k_value + focus_disparity + MAX_COC`, e derivar no dataloader com a
mesma função da geração é a única forma de garantir que treino e geração não divergem.
Gravá-lo criaria uma segunda fonte de verdade — que é o defeito D1.

## Três coisas que este módulo impede POR CONSTRUÇÃO

1. **`max_coc` não é campo.** Uma auditoria mediu que, quando era, uma amostra com
   `max_coc = 10.510746` — o valor exato do experimento `kfix` — atravessava o writer
   e aterrissava no disco sem erro. Isso é o D1 com granularidade de amostra, e é
   exatamente o mecanismo que fez o `kfix` subir o LF-Bokeh para +0,8288 enquanto
   derrubava RealBokeh para +0,4832 e RealDOF para −0,4599: duas convenções de
   normalizador no mesmo lote. O valor vem de `MAX_COC` e de lugar nenhum mais.

2. **`depth_backend` é obrigatório.** É o campo que prova que o D11 não disparou — a
   cascata que trocava Depth Pro por Depth Anything, que devolve disparidade e deixa a
   amostra espelhada. Antes ele era produzido pelo contrato e descartado uma camada
   abaixo.

3. **`provenance` não é dict livre e `mask_source` não é string livre.** Proveniência
   que pode mentir é pior que proveniência ausente: `mask_source="automatic"` depois de
   cair num fallback afirma algo falso, e nada denuncia.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from control.contract import (
    CONTROL_VERSION, DEPTH_BACKENDS, MAX_COC, k_at_resolution, reject,
)
from dataio.encoding import EncodedDepth, quantization_coc_error_px
from dataio.layout import depth_ref
# `FocusSource` é importado de `qc.focus_region` em vez de redefinido aqui de
# propósito: um vocabulário de proveniência com duas definições é a forma exata do
# defeito que este projeto persegue ("cópias divergem" — foi assim que se chegou a
# quatro interpretações de K). `qc` não importa `dataio`, então não há ciclo.
from qc.focus_region import FocusSource


class MaskSource(str, Enum):
    """De onde a máscara de foco veio. Enum fechado, não string livre.

    `AUTOMATIC` não existe de propósito: é vago demais para ser proveniência. Se um dia
    houver um segundo segmentador, ele ganha um valor próprio — nunca reusa o do
    BiRefNet.
    """

    BIREFNET = "birefnet"
    MANUAL = "manual"
    DEPTH_BAND = "depth_band"          # rota A: banda de profundidade sintética
    #: Rota A: **não houve máscara**. O plano de foco foi sorteado e o mapa de defocus
    #: sai direto da Eq. 2. `mask/<id>.png` grava a banda em foco derivada do plano
    #: sorteado, que é descritiva — nenhuma segmentação a produziu.
    SAMPLED_PLANE = "sampled_plane"
    #: Rota C: a máscara do BiRefNet CORRIGIDA pela região de retenção de detalhe —
    #: o substituto automático do refinamento manual do §3.2(c). O arquivo em
    #: `mask/<id>.png` é a região corrigida, que é a que produziu `focus_disparity`.
    BIREFNET_REFINED = "birefnet_refined"
    #: Rota C: o BiRefNet declinou (probabilidade 0 em 20,6% das amostras do piloto) e
    #: a região vem só da retenção. Valor próprio, nunca o do BiRefNet: dizer
    #: `birefnet` aqui afirmaria que um segmentador de saliência escolheu a região.
    RETENTION_ONLY = "retention_only"


class KSource(str, Enum):
    EQ3_EXIF = "eq3_exif"              # rota B, Eq. 3
    EQ5_SSIM_SWEEP = "eq5_ssim_sweep"  # rota C, Eq. 5
    SAMPLED_FROM_BC = "sampled_from_bc"  # rota A — decisão nossa, ver CONTRATO.md


@dataclass(frozen=True)
class SampleRefs:
    """Onde as imagens de origem vivem. Referência, não cópia."""

    source_dataset: str                 # ex.: "akcit-pixel/RealBokeh"
    source_sample_id: str               # ex.: "..._f_1038_level_3_aligned"
    scene_id: str                       # a chave do split — NUNCA a imagem
    aif_ref: Optional[str] = None
    bokeh_ref: Optional[str] = None
    source_split: Optional[str] = None


@dataclass(frozen=True)
class SampleProvenance:
    """Campos obrigatórios. Um dict vazio passava na versão anterior."""

    pipeline_commit: str
    depth_model_sha256: str
    mask_model_sha256: str
    image_hw: tuple[int, int]           # resolução da IMAGEM, não do array de depth
    seed: Optional[int] = None
    renderer: Optional[dict] = None     # `BokehMeRenderer.provenance()`, quando houver
    deblurnet: Optional[dict] = None    # rota B
    source_license: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["image_h"], out["image_w"] = int(self.image_hw[0]), int(self.image_hw[1])
        out.pop("image_hw")
        return out


@dataclass(frozen=True)
class ControlLabel:
    """O rótulo de controle e sua proveniência de medição.

    Sem `max_coc`: ele é `MAX_COC` e nada mais. Sem default em `is_k_censored` nem em
    `depth_backend`: quem constrói tem que decidir, porque afirmar por omissão é mentir.
    """

    k_value: float                      # NA ESCALA DE PIXEL DA IMAGEM ORIGINAL
    k_source: KSource
    focus_disparity: float              # 1/m — o mapa usa ISTO, nunca 1/focus_depth_m
    is_k_censored: bool
    depth_backend: str
    control_version: str = CONTROL_VERSION   # carimbado quando o rótulo foi CALCULADO
    calibration_ssim: Optional[float] = None
    k_analytic: Optional[float] = None
    k_effective_factor: Optional[float] = None

    @property
    def focus_depth_m(self) -> float:
        """Só leitura humana. O mapa NUNCA reconstrói a partir daqui."""
        return 1.0 / self.focus_disparity


#: `FocusSource` -> `MaskSource` do arquivo gravado em `mask/<id>.png`.
#:
#: A rota C grava a região **final** de foco, não a máscara crua do BiRefNet: é a
#: região final que produziu `focus_disparity`, e gravar outra coisa faria de
#: `mask/<id>.png` uma máscara que não corresponde ao rótulo — proveniência que mente.
#: Este dicionário é a única ponte entre os dois vocabulários, e
#: `test_dataio.test_toda_fonte_de_foco_tem_fonte_de_mascara` prova que ele é total:
#: um `FocusSource` novo sem entrada aqui levanta `KeyError`, não escolhe um default.
FOCUS_SOURCE_TO_MASK_SOURCE = {
    FocusSource.BIREFNET: MaskSource.BIREFNET,
    FocusSource.BIREFNET_REFINED: MaskSource.BIREFNET_REFINED,
    FocusSource.RETENTION_ONLY: MaskSource.RETENTION_ONLY,
    FocusSource.SAMPLED_PLANE: MaskSource.SAMPLED_PLANE,
}

#: Valores de `mask_source` que só existem porque houve refinamento. Se um deles
#: aparece, `focus_source` tem que dizer o mesmo — ver `validate_metadata`.
_REFINED_MASK_SOURCES = frozenset({MaskSource.BIREFNET_REFINED.value,
                                   MaskSource.RETENTION_ONLY.value})


@dataclass(frozen=True)
class FocusRegionRecord:
    """A marcação por amostra do substituto automático do §3.2(c).

    Existe por um requisito explícito: dar para **treinar com e sem** as amostras
    refinadas e medir a diferença. Sem esta marcação, "consertamos a máscara" seria uma
    afirmação sem teste — e o piloto mediu que a máscara do BiRefNet acerta o plano de
    foco em só **35,2%** das amostras, então a afirmação precisa de teste.

    Nada aqui é rótulo: o rótulo é `ControlLabel.focus_disparity`. Estes campos dizem
    **de onde ele veio** e **quanto confiar nele**.
    """

    source: FocusSource
    #: Fração da região de retenção que a máscara do BiRefNet cobria ANTES do refino.
    #: É a medida de quanto o segmentador concordava com a física. 0,0 = declinou.
    #: **É esta que decide o ramo** — ver `qc.focus_region.RefinedFocusRegion.agreement`
    #: para por que IoU não serve como critério aqui.
    agreement: float
    #: Quanto da máscara crua do BiRefNet estava DENTRO da região de retenção. É o
    #: contrapeso de `agreement`: uma máscara enorme alcança tudo e pontua 1,0 em
    #: `agreement`, e é aqui que ela se denuncia.
    precision: float
    #: IoU entre as duas regiões. Resumo simétrico; não decide nada.
    iou: float
    #: Retenção mediana dentro da região final, em [0, 1]. Perto de 1 = a região
    #: realmente preservou o detalhe da AIF; baixa = a amostra passou e merece
    #: desconfiança mesmo assim.
    retention_in_region: float
    #: Área da região final, como fração do quadro.
    area_ratio: float
    #: **A resolução em que a retenção foi medida.** Obrigatória: `retention_window_px`
    #: é uma quantidade em pixel, e a regra do projeto é que toda quantidade em pixel
    #: carrega a resolução em que foi medida. `agreement` e `area_ratio` são frações,
    #: mas as duas foram calculadas nesta grade — a 1500x2000 e a 512x683 a janela de
    #: 33 px cobre 1,7% e 6,4% do lado longo, que são escalas físicas diferentes.
    retention_hw: tuple[int, int]
    retention_window_px: int
    #: `True` quando o BiRefNet devolveu máscara vazia — o caso que antes virava
    #: `focus_mask_empty` e era DESCARTADO, contra o que o §3.2(c) manda fazer.
    initial_mask_was_empty: bool
    #: Área da máscara crua do BiRefNet, na grade da retenção. Continua medida mesmo
    #: quando ela não define mais o foco: sem ela, some a distribuição que denunciaria
    #: o segmentador degradando. `None` quando não houve máscara inicial nenhuma — o que
    #: é diferente de `0.0`, "o segmentador rodou e devolveu vazio".
    initial_mask_area_ratio: Optional[float]
    #: **Diagnóstico pareado, NUNCA rótulo**: a disparidade de foco que a máscara crua
    #: do BiRefNet teria produzido nesta mesma amostra. É o que permite responder "o
    #: refinamento melhorou?" sobre as MESMAS amostras, em vez de comparar o subgrupo
    #: refinado contra o não refinado — que é um subgrupo selecionado por concordar, e
    #: portanto mais fácil por construção. `None` quando não existe: máscara vazia, ou
    #: a Eq. 4 rejeitaria aquele valor por implausível.
    disparity_from_initial_mask: Optional[float] = None

    @property
    def was_refined(self) -> bool:
        """A amostra passou pelo substituto do passo manual do paper.

        Mesma regra de `qc.focus_region.RefinedFocusRegion.was_refined`, e
        `test_dataio.test_was_refined_bate_com_o_do_modulo_de_refino` prova que as duas
        não podem divergir.
        """
        return FocusSource(self.source) is not FocusSource.BIREFNET

    @classmethod
    def from_region(cls, region, *, retention_hw: tuple[int, int],
                    retention_window_px: int,
                    initial_mask_area_ratio: Optional[float],
                    disparity_from_initial_mask: Optional[float] = None,
                    ) -> "FocusRegionRecord":
        """Constrói a partir de um `qc.focus_region.RefinedFocusRegion`.

        Copia campo a campo em vez de guardar o objeto: o `RefinedFocusRegion` carrega
        a máscara inteira, e um registro de metadado não deve segurar um array de
        3 megapixels vivo até a serialização.
        """
        return cls(
            source=region.source, agreement=float(region.agreement),
            precision=float(region.precision), iou=float(region.iou),
            retention_in_region=float(region.retention_in_region),
            area_ratio=float(region.area_ratio),
            retention_hw=(int(retention_hw[0]), int(retention_hw[1])),
            retention_window_px=int(retention_window_px),
            initial_mask_was_empty=bool(region.initial_mask_was_empty),
            initial_mask_area_ratio=(None if initial_mask_area_ratio is None
                                     else float(initial_mask_area_ratio)),
            disparity_from_initial_mask=(
                None if disparity_from_initial_mask is None
                else float(disparity_from_initial_mask)),
        )

    @property
    def mask_source(self) -> MaskSource:
        """A `MaskSource` do arquivo que vai para `mask/<id>.png`."""
        return FOCUS_SOURCE_TO_MASK_SOURCE[FocusSource(self.source)]

    def to_metadata(self) -> dict:
        return {
            "focus_source": FocusSource(self.source).value,
            "focus_was_refined": bool(self.was_refined),
            "focus_agreement": float(self.agreement),
            "focus_retention_in_region": float(self.retention_in_region),
            "focus_region_area_ratio": float(self.area_ratio),
            "focus_retention_h": int(self.retention_hw[0]),
            "focus_retention_w": int(self.retention_hw[1]),
            "focus_retention_window_px": int(self.retention_window_px),
            "focus_initial_mask_was_empty": bool(self.initial_mask_was_empty),
            "focus_initial_mask_area_ratio": (
                None if self.initial_mask_area_ratio is None
                else float(self.initial_mask_area_ratio)),
            "focus_disparity_from_initial_mask": (
                None if self.disparity_from_initial_mask is None
                else float(self.disparity_from_initial_mask)),
        }


@dataclass
class Sample:
    sample_id: str
    route: str
    refs: SampleRefs
    control: ControlLabel
    depth: EncodedDepth
    mask: np.ndarray
    mask_source: MaskSource
    provenance: SampleProvenance
    #: De onde saiu a região que definiu `D_focus`. Obrigatório na prática: sem ele
    #: `validate_metadata` reprova, porque uma amostra que não diz se passou pelo
    #: refinamento não dá para incluir nem excluir de um treino com consciência.
    focus: Optional[FocusRegionRecord] = None
    quality: dict = field(default_factory=dict)
    generated_images: dict = field(default_factory=dict)
    is_valid_for_control: bool = True
    #: Ordem de canal das imagens em `generated_images`. Declarada, não assumida.
    channel_order: str = "bgr"

    def metadata(self) -> dict:
        meta: dict[str, Any] = {
            "sample_id": self.sample_id,
            "route": self.route,
            "control_version": self.control.control_version,
            **asdict(self.refs),
            "k_value": float(self.control.k_value),
            "k_source": KSource(self.control.k_source).value,
            "focus_disparity": float(self.control.focus_disparity),
            "focus_depth_m": float(self.control.focus_depth_m),
            "max_coc": MAX_COC,                     # da constante, nunca de um campo
            "is_k_censored": bool(self.control.is_k_censored),
            "is_valid_for_control": bool(self.is_valid_for_control
                                         and not self.control.is_k_censored),
            "depth_backend": self.control.depth_backend,
            "calibration_ssim": self.control.calibration_ssim,
            "k_analytic": self.control.k_analytic,
            "k_effective_factor": self.control.k_effective_factor,
            "mask_source": MaskSource(self.mask_source).value,
            "mask_area_ratio": float(np.mean(np.asarray(self.mask) > 0.5)),
            # K vive na escala de `image_hw`; a quantização vive em `depth_hw`.
            # Converter e nomear a resolução, em vez de gravar "px" solto.
            "quantization_coc_error_px_at_depth_hw": quantization_coc_error_px(
                self.depth,
                k_at_resolution(self.control.k_value, self.depth.image_hw,
                                min(self.depth.depth_hw))),
            **self.depth.to_metadata(),
            # Marcação do refinamento da região em foco. Emitida como campos de PRIMEIRO
            # NÍVEL, não dentro de um sub-dicionário, porque o manifesto espelha estas
            # chaves e filtrar 20 mil amostras por refinamento não pode exigir abrir
            # 20 mil JSONs. Ausente quando `focus` é `None` — e aí `validate_metadata`
            # reprova, em vez de gravar `None` que se confundiria com "medido como zero".
            **(self.focus.to_metadata() if self.focus is not None else {}),
            "provenance": self.provenance.to_dict(),
            "quality": self.quality,
            "generated_images": sorted(self.generated_images),
            "channel_order": self.channel_order,
        }
        return meta

    def metadata_json(self) -> str:
        return metadata_to_json(self.metadata())


def metadata_to_json(meta: dict) -> str:
    """Serializa um dicionário de metadados com a mesma regra de `Sample`.

    Público porque o writer **acrescenta** campos ao metadado antes de gravá-lo — a
    referência do arquivo de profundidade, que só o writer conhece porque só ele sabe em
    que layout está gravando. Sem isto, o writer teria que reimplementar o `default=`,
    e um serializador com duas implementações é a mesma família de defeito que uma
    fórmula com duas implementações.
    """
    return json.dumps(meta, indent=2, ensure_ascii=False, default=_jsonable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"não serializável: {type(value).__name__}")


#: Campos sem os quais o sinal de controle é irrecuperável, ou a proveniência é cega.
REQUIRED_METADATA_FIELDS = frozenset({
    "sample_id", "route", "control_version", "scene_id",
    "source_dataset", "source_sample_id",
    "k_value", "k_source", "focus_disparity", "max_coc",
    "is_k_censored", "is_valid_for_control",
    "depth_backend", "mask_source",
    "disparity_min", "disparity_max",
    "image_h", "image_w", "depth_h", "depth_w", "depth_encoding",
    "provenance",
    # --- região em foco (§3.2(c)): de onde veio `focus_disparity` ---------------
    # Obrigatórios porque o rótulo de controle inteiro pende de `focus_disparity`, e
    # `focus_disparity` sem a origem da região é um número sem proveniência. Medimos
    # que a máscara do BiRefNet acerta o plano de foco em 35,2% das amostras: uma
    # amostra que não diz se passou pelo refinamento não dá para incluir nem excluir
    # de um treino com consciência.
    "focus_source", "focus_was_refined", "focus_agreement",
    "focus_retention_in_region", "focus_region_area_ratio",
    # A resolução da retenção entra junto pela regra dura do projeto: toda quantidade
    # em pixel carrega a resolução em que foi medida, e `focus_retention_window_px` é
    # uma quantidade em pixel.
    "focus_retention_h", "focus_retention_w", "focus_retention_window_px",
})

_REQUIRED_PROVENANCE = ("pipeline_commit", "depth_model_sha256", "image_h", "image_w")

#: A máscara tem que ter procedência, mas nem toda rota usa um MODELO para obtê-la.
#: Nas rotas B e C ela sai do BiRefNet e o hash do checkpoint é a procedência. Na rota A
#: **não há segmentador**: o plano de foco é sorteado (`paper.txt:329-330`) e a banda em
#: foco é derivada dele por uma regra. Exigir `mask_model_sha256` ali obrigaria a
#: inventar um hash ou a afrouxar a checagem para todo mundo — as duas ruins.
#:
#: Então: **uma das duas**, nunca nenhuma. `mask_rule` é a descrição textual da regra
#: que produziu a banda, e ocupa o mesmo lugar de "como esta máscara veio a existir".
_REQUIRED_MASK_PROVENANCE = ("mask_model_sha256", "mask_rule")


def validate_metadata(meta: dict) -> None:
    """Falha alto se algo estiver ausente, inválido ou fora do vocabulário fechado.

    Cinco checagens, cada uma correspondendo a um defeito medido:

    1. campo obrigatório ausente — o writer antigo filtrava `None` do JSON, e a
       diferença entre "não medido" e "medido como zero" desaparecia;
    2. escalar inválido;
    3. **`max_coc != MAX_COC`** — o `kfix` chegando ao disco;
    4. `depth_backend` fora do conjunto registrado — o D11;
    5. `mask_source`/`k_source` fora do enum, e proveniência incompleta.
    """
    missing = sorted(REQUIRED_METADATA_FIELDS - set(meta))
    if missing:
        raise ValueError(f"metadados incompletos, faltam: {missing}")

    for name in ("k_value", "focus_disparity", "max_coc"):
        value = meta[name]
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"`{name}` inválido nos metadados: {value!r}")

    if float(meta["max_coc"]) != MAX_COC:
        reject("max_coc_invalid",
               f"{meta['max_coc']} != MAX_COC={MAX_COC}. Normalizador por amostra é o "
               "defeito D1 com granularidade fina — foi assim que o kfix rodou duas "
               "convenções no mesmo lote.")

    if meta["depth_backend"] not in DEPTH_BACKENDS:
        reject("depth_backend_unregistered",
               f"{meta['depth_backend']!r} não está em {sorted(DEPTH_BACKENDS)}")

    if meta["control_version"] != CONTROL_VERSION:
        reject("control_version_mismatch",
               f"rótulo {meta['control_version']!r} != contrato {CONTROL_VERSION!r}")

    MaskSource(meta["mask_source"])          # levanta se fora do enum
    KSource(meta["k_source"])
    _validate_focus_region(meta)
    _validate_depth_ref(meta)

    prov = meta["provenance"]
    if not isinstance(prov, dict):
        raise ValueError("`provenance` tem que ser dict")
    faltando = [f for f in _REQUIRED_PROVENANCE if not prov.get(f)]
    if faltando:
        raise ValueError(f"proveniência incompleta, faltam: {faltando}")
    # `mask_rule` vive em `extra`: `SampleProvenance` não tem campo próprio para ela, e
    # criar um que só a rota A preenche deixaria `None` em dois terços do dataset.
    # Procurar nos dois lugares é o que mantém a exigência real sem esse custo.
    extra_prov = prov.get("extra") or {}
    if not any(prov.get(f) or extra_prov.get(f) for f in _REQUIRED_MASK_PROVENANCE):
        raise ValueError(
            "proveniência da máscara ausente: é obrigatório um de "
            f"{list(_REQUIRED_MASK_PROVENANCE)}. `mask_model_sha256` quando um modelo a "
            "produziu; `mask_rule` quando foi uma regra, como na rota A, onde o plano de "
            "foco é sorteado e não há segmentador.")

    if meta["k_source"] == KSource.EQ5_SSIM_SWEEP.value and meta["calibration_ssim"] is None:
        raise ValueError("sweep da Eq. 5 sem `calibration_ssim`: censura não avaliada")


def _validate_depth_ref(meta: dict) -> None:
    """A referência do PNG de profundidade tem que ser coerente com o layout declarado.

    **Não é campo obrigatório**, e isso é deliberado: `bokehnet-regen-rota-c` e o
    `-rota-a` foram gravados antes dele existir, e `validate_metadata` é a mesma função
    que o `publish_release.py` roda sobre os `meta/` desses releases. Exigir o campo
    aqui reprovaria retroativamente um release correto — a compatibilidade de leitura é
    requisito, não cortesia.

    O que é exigido é que, **quando declarado**, o par (`depth_ref`, `depth_layout`)
    não minta:

    1. os dois vêm juntos. `depth_layout` sem `depth_ref` afirma um layout e não diz
       onde o arquivo está; `depth_ref` sem `depth_layout` tira da validação a única
       peça que permite conferir a referência contra a regra;
    2. `depth_layout` está no enum;
    3. `depth_ref` é **exatamente** o que o layout declarado manda. Uma referência
       `depth/<sample_id>.png` num release que diz `per_scene` é 41 arquivos onde
       deveria haver 1 — ou, pior, um caminho que não existe.
    """
    ref, layout = meta.get("depth_ref"), meta.get("depth_layout")
    if ref is None and layout is None:
        return

    if ref is None or layout is None:
        raise ValueError(
            f"`depth_ref`={ref!r} e `depth_layout`={layout!r}: os dois vêm juntos ou "
            "nenhum vem. Um sem o outro é layout sem endereço, ou endereço sem regra "
            "que o confira.")

    esperado = depth_ref(sample_id=meta["sample_id"], scene_id=meta["scene_id"],
                         layout=layout)      # levanta se `layout` fora do enum
    if ref != esperado:
        raise ValueError(
            f"`depth_ref`={ref!r} não é o que `depth_layout={layout!r}` manda "
            f"({esperado!r}) para sample_id={meta['sample_id']!r}, "
            f"scene_id={meta['scene_id']!r}.")


def _validate_focus_region(meta: dict) -> None:
    """A marcação do refinamento da região em foco tem que ser coerente.

    Três coisas, cada uma correspondendo a um jeito conhecido de a proveniência mentir:

    1. **`focus_source` fora do enum.** Vocabulário aberto aqui reabriria a porta do
       `mask_source="automatic"`, que afirmava algo falso e nada denunciava.
    2. **`focus_was_refined` em desacordo com `focus_source`.** O booleano é o campo por
       onde se filtra o dataset para treinar com e sem as amostras refinadas; se ele
       puder discordar da origem, o filtro seleciona outro conjunto que não o que diz
       selecionar.
    3. **`mask_source` em desacordo com `focus_source`.** `mask/<id>.png` guarda a
       região que produziu `focus_disparity`. Um metadado que diga
       `focus_source="retention_only"` e `mask_source="birefnet"` afirma que o arquivo
       no disco é a máscara do segmentador — e quem for auditar o rótulo vai olhar a
       máscara errada.
    """
    source = FocusSource(meta["focus_source"])       # levanta se fora do enum

    esperado = source is not FocusSource.BIREFNET
    if bool(meta["focus_was_refined"]) is not esperado:
        reject("focus_provenance_inconsistent",
               f"focus_was_refined={meta['focus_was_refined']!r} contradiz "
               f"focus_source={source.value!r}")

    mask_source = MaskSource(meta["mask_source"])
    if ({mask_source.value, source.value} & _REFINED_MASK_SOURCES
            and mask_source is not FOCUS_SOURCE_TO_MASK_SOURCE[source]):
        reject("focus_provenance_inconsistent",
               f"mask_source={mask_source.value!r} não é a máscara que produziu "
               f"focus_disparity (focus_source={source.value!r}). O arquivo em "
               "mask/<id>.png é a região final de foco.")

    for name, minimo, maximo in (("focus_agreement", 0.0, 1.0),
                                 ("focus_region_area_ratio", 0.0, 1.0)):
        value = meta[name]
        if value is None or not np.isfinite(value) or not (minimo <= float(value) <= maximo):
            raise ValueError(f"`{name}` fora de [{minimo}, {maximo}]: {value!r}")
    if float(meta["focus_region_area_ratio"]) <= 0.0:
        raise ValueError("`focus_region_area_ratio` == 0: região vazia não define plano "
                         "de foco, e a Eq. 4 não teria pixel para a mediana")

    # `focus_retention_in_region` pode ser NaN — significa "a região final não tinha
    # pixel com detalhe medível", que é diferente de "reteve zero". Não pode é estar
    # fora de [0, 1] quando é finito, porque a retenção é uma razão saturada em 1.
    retencao = meta["focus_retention_in_region"]
    if retencao is None or (np.isfinite(retencao) and not 0.0 <= float(retencao) <= 1.0):
        raise ValueError(f"`focus_retention_in_region` inválido: {retencao!r}")

    for name in ("focus_retention_h", "focus_retention_w", "focus_retention_window_px"):
        valor = meta[name]
        # `None` aqui é ausência disfarçada de presença: a chave existe, então o teste de
        # campo obrigatório passa, e a resolução da medida continua não existindo.
        if valor is None or int(valor) < 1:
            raise ValueError(f"`{name}` tem que ser um inteiro >= 1: {valor!r}")
