"""Rota (a) do paper — §3.2(a), pré-treino SINTÉTICO. O alvo é renderizado.

    D_def = K · |D − D_focus|                                            (Eq. 2)

Fluxo por imagem, e é o parágrafo inteiro do paper (`paper.txt:328-334`):

    AIF real  ──Depth Pro [7]──▶  D, profundidade MÉTRICA em metros
                                    │
        sorteio de (D_focus, K)  ◄───┴── D_focus da faixa de disparidade da PRÓPRIA
                                    │    imagem; K da distribuição medida em B e C
                                    │
                        Eq. 2  ──────┴──▶ D_def
                                    │
                    BokehMe [43] ───┴──▶ **bokeh renderizada = ALVO**
                                    │
                        gates  ─────┴──▶ medem sempre, bloqueiam só com limiar congelado
                        writer ─────┴──▶ disparidade uint16 + bokeh JPEG + escalares

> *"(a) Synthetic data. We pretrain with synthetic data: starting from real all-in-focus
> images and their estimated depth map D, we **randomly sample** a focus plane D_focus and
> a target bokeh level K (Fig. 3 (a)). We then construct D_def using Eq. 2 and use a
> **simulator [43]** to render the corresponding target bokeh image consistent with
> D_def."* — `paper.txt:328-332`

## O que esta rota NÃO tem, e é o paper que não tem — não economia nossa

* **Não tem máscara e não roda BiRefNet.** O plano de foco é **sorteado**
  (`paper.txt:329-330`), não estimado. A Eq. 4 (`paper.txt:352`) e o BiRefNet `[86]`
  pertencem a (b) e (c) (`paper.txt:348-350`, `361-362`). Qualquer máscara que esta rota
  grave é construção nossa e está declarada como tal — ver "A máscara" abaixo.
* **Não usa a Eq. 3.** Ela é da rota B / ITW (`paper.txt:336-338`), e aqui não há EXIF.
* **Não usa a Eq. 5 nem SSIM de calibração.** O sweep é da rota C
  (`paper.txt:369-370`); `calibration_ssim` é `None` e `validate_metadata` só o exige
  quando `k_source == EQ5_SSIM_SWEEP` (`dataio/sample.py:425`).
* **Não usa a DeblurNet.** A AIF já é all-in-focus; não há nada a deblurar.
* **Não tem censura de K.** Não existe varredura com teto, logo não existe "K no teto":
  `is_k_censored` é `False` sempre.

E uma coisa que só esta rota tem: **o alvo é renderizado.** Nas outras duas o alvo é
fotografia real. Por isso aqui `provenance.renderer` é o BokehMe com
`is_final_label_renderer: True` — e o paper autoriza isso de forma mais direta que na rota
C, com a citação literal duas vezes no corpo (`paper.txt:331` e `292`). A ressalva que o
próprio paper escreve fica registrada: a rota A *"is constrained by **renderer bias** and
may introduce unrealistic artifacts"* (`paper.txt:333-334`) — é pré-treino de **geometria
de CoC**, não de aparência, e o currículo confirma (40K passos em sintético, 60K em real,
`paper.txt:515-516`).

## Quantas variantes por imagem — 41, e é `[A]`

O paper **não publica** o número. Publica os dois lados da conta: ~70K pares sintéticos
(`paper.txt:527-528`) de um pool de ~1,7K AIFs nítidas (`paper.txt:998`). `70.000/1.700 =
41,2` `[I]`. O `41` que entra aqui é `[A]`, escolhido para fechar as duas âncoras, e é
flag (`samples_per_image`).

O defeito que isso conserta é o A2 da auditoria, e ele é pior que "40x menos dado": com
**uma** variante por imagem (`bokehnet-preprocessing/src/pipelines/route_a.py:138`), cada
conteúdo aparece com **um** K e **um** plano de foco, e o sinal que o pré-treino existe
para ensinar — *"a mesma cena, com K diferente, borra diferente"* (`paper.txt:332-333`) —
**não existe no dado**. K vira confundido com conteúdo.

## O sorteio — estratificado, determinístico por `sample_id`

`paper.txt:329-330` diz *"randomly sample"* e cala sobre a distribuição. Duas decisões
nossas, declaradas:

1. **K vem da distribuição empírica de B e C**, lida de um JSON
   (`sources/k_distribution.py`), na grandeza **livre de resolução**
   `k_per_long_side = k_value/max(H,W)`, remultiplicada pelo lado longo desta imagem. É o
   conserto do defeito A5: transportar K cru entre resoluções diferentes aplica um CoC em
   pixel que não corresponde a óptica nenhuma, e o `pixel_ratio` da rota B varia 12x
   (`ACHADOS.md:36` `[M]`). `k_source = SAMPLED_FROM_BC` (`CONTRATO.md:163-166`).
2. **41 sorteios i.i.d. por imagem deixariam imagens inteiras sem K alto.** O que ensina
   controle é a **cobertura por imagem**, então o sorteio é um **hipercubo latino** de
   dimensão 2 (K × plano de foco): com N variantes, cada uma cai num estrato diferente de
   K **e** num estrato diferente de plano de foco. Ver `draw_plan`.

A semente sai do `scene_id` por sha256, nunca de `hash()` — que é randomizado por
processo (mesma razão de `dataio/split.py:29-35`). Mesma imagem, mesma semente, mesmos K
e plano de foco, em qualquer máquina.

## O plano de foco cai DENTRO da faixa de disparidade da imagem — por construção

`D_focus` sorteado fora da faixa da própria imagem degenera o mapa: tudo saturado ou tudo
zero. A rota A resolve isso amostrando **um quantil da distribuição empírica de
disparidade da imagem**, e não um valor uniforme na faixa:

    focus_disparity = quantile( 1/z[finito e > 0],  q_low + u·(q_high − q_low) )

Três consequências, e a terceira é a que importa:

* um quantil dos dados **é** um valor entre o mínimo e o máximo observados — está dentro
  da faixa por definição, sem clamp;
* existe **massa de cena** naquele plano, então a banda em foco não sai vazia e o mapa não
  fica degenerado;
* o sorteio acompanha **o conteúdo**, não a geometria — e este é o ponto que a medição
  transformou de intuição em número. Uniforme na faixa de disparidade concentra os planos
  perto do pixel mais próximo, e a banda em foco sai **vazia** (nenhum pixel com `|CoC|`
  sub-pixel), o que rejeita a variante com `focus_mask_empty`. Medido em cena sintética
  de retrato (sujeito a 1,2 m, fundo de 8 a 30 m), 41 variantes:

        regra                              banda vazia        área mediana da banda
        uniforme na faixa (K=5)              56,1%                   0,0000
        uniforme na faixa (K=50)             80,5%                   0,0000
        quantil da disparidade (K=5)          0,0%                   0,8892
        quantil da disparidade (K=50)         0,0%                   0,1937

  e na cena natural com 10% de céu, uniforme leva a saturação mediana do mapa a **0,9320**
  com K = 300, contra **0,0326** por quantil. Ver `reference/ROTA_A_DECISOES.md`.

**Uma correção medida à auditoria.** O item A6 diz que sortear no quantil da profundidade
métrica é "o espaço errado". Medimos: **o quantil é invariante a transformação monótona**,
então `1/quantil_z(1−q) == quantil_disp(q)` — exato em estatística de ordem, e com erro
relativo de 1e-9 a 1e-7 quando `np.quantile` interpola entre dois vizinhos (interpolar
`1/z` não é o inverso de interpolar `z`). As duas regras produzem **os mesmos planos de
foco**: mediana 28,23 m nas duas, na cena natural medida. O que de fato
muda o resultado é quantil **contra uniforme**, não `z` contra disparidade. O que continua
valendo de A6 é a outra metade: a **unidade primária**. O campo gravado é
`focus_disparity`, como o contrato manda (`CONTRATO.md:36-40`), e `focus_depth_m` existe
só para leitura humana — o defeito antigo era produzir `focus_depth_m` e reconstruir a
disparidade duas vezes a jusante, em duas linhas diferentes.

**A população do sorteio são os planos de foco FISICAMENTE PLAUSÍVEIS.** Um pixel no teto
de 10.000 m do Depth Pro não é candidato a plano de foco — é sentinela (25,7% das amostras
medidas têm `z_max == 10.000`, `ACHADOS.md:54` `[M]`). Então `build_focus_sampling_pool`
restringe a população a `[FOCUS_DEPTH_MIN_M, FOCUS_DEPTH_MAX_M]` do contrato, e a fração
de pixels que sobrou vai gravada em `focus_sampling_pool_fraction`.

Isto não é clamp nem fallback: é a definição da população, e ela é medida. Sem a
restrição, uma cena com 10% de céu perde **4,9%** das variantes por
`focus_depth_implausible` com `q = [0,05, 0,95]` e **9,8%** com `q = [0, 1]` — perda
proporcional à ÁREA de céu, invisível no rótulo e impossível de recuperar depois. Com a
restrição, **0,0%**. O gate `focus_depth_plausible` continua ligado como arame de
tropeço: se ele passar a disparar, a regra de amostragem mudou.

`q_low`/`q_high` (`[A]`, 0,05 e 0,95) cortam os extremos amostrais **dentro** dessa
população — um pixel isolado a 0,3 m —, e o efeito medido está na tabela acima.

## A máscara — e o vocabulário que falta

A rota A **não tem máscara** (`paper.txt:329-330`), mas o writer grava
`mask/<id>.png` para toda amostra e `validate_metadata` exige `mask_source` e os campos de
região em foco (`dataio/sample.py:352-373`). A alternativa honesta seria **não gravar
campo de máscara** — e isso exige mudança em `dataio/`, que esta tarefa descreve e não
aplica.

O que esta rota grava enquanto isso é a **banda em foco derivada do próprio rótulo**:

    banda = |CoC| ≤ focus_band_coc_px              # 0,5 px por default

Ela não afirma nada sobre a cena: é uma função determinística de `(D, D_focus, K)`, e a
regra completa vai na proveniência em `mask_rule` — quantil, tolerância, grade, seed —, o
que permite reconstruí-la byte a byte. É o análogo exato de um hash de modelo: o que
permite refazer a máscara. Isso conserta o defeito A7, em que **um** array era gravado em
**três** chaves com semânticas diferentes (`foreground_mask`, `focus_mask_auto`,
`focus_mask_final`) com a única proveniência sendo a string `"depth_band_q12"`.

`focus_band_coc_px = 0,5` não é limiar de rejeição — é a definição da banda, e 0,5 px de
**raio** de CoC é sub-pixel, isto é, indistinguível de nítido. `[A]`, declarado.

**Vocabulário**: `MaskSource.DEPTH_BAND` existe e foi criado para a rota A antiga (a banda
no quantil 0,12 da profundidade), e `FocusSource` não tem valor para "sorteado". Reusar
qualquer valor existente seria proveniência que mente — `focus_source = "birefnet"` numa
rota que nunca roda BiRefNet é exatamente o `mask_source="automatic"` da cascata antiga.
Então este módulo **exige** dois valores novos e falha alto enquanto eles não existirem:
ver `resolve_sampled_vocabulary`.

## Os gates, e os dois que não existem aqui

Próprios da rota A, os três que a auditoria pede em F7 — `rendered_coc_p99_px`,
`defocus_saturation_ratio` e a razão de nitidez como "o render fez algo?" — todos com
limiar `None`: mede no piloto, congela depois.

**Não há gate de faixa de K**, e a ausência é decisão. Na rota B "K implausível" é uma
amostra a descartar porque o K vem da EXIF daquela foto. Aqui o K é **imposto** de um
suporte já validado uma vez, em voz alta, por `KDistribution.degenerate_reason()`: um K
fora do plausível é defeito da **distribuição**, e rejeitar amostra por isso esconderia um
erro de entrada atrás de um histograma de 70 mil linhas. O que se pergunta por amostra é
outra coisa — *"este K produziu um mapa utilizável NESTA cena?"* —, e é o que
`rendered_coc_p99_px` e `defocus_saturation_ratio` medem.

**Não há `focus_mask_is_sharpest`.** Ele seria tautológico: a banda é, por definição, onde
o CoC é sub-pixel, então o renderer a preservou nítida por construção. Um gate que não
pode reprovar é pior que gate nenhum (`REGISTRO.md:695-696`).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import numpy as np

from control.contract import (
    FOCUS_DEPTH_MAX_M, FOCUS_DEPTH_MIN_M, MAX_COC, SampleRejected, defocus_map, reject,
    signed_coc_px,
)
from dataio import (
    ControlLabel, DEPTH_LONG_SIDE, DepthLayout, EncodedDepth, FocusRegionRecord,
    KSource, MaskSource, Sample, SampleProvenance, SampleRefs, SceneSplit,
    FileSampleWriter, encode_depth,
)
from qc.focus_region import FocusSource
from qc.gates import (
    GateReport, GateResult, aif_sharpness, bokeh_is_blurrier_than_aif,
    depth_useful_levels, focus_depth_plausible, mask_area_ratio,
)
from qc.rejection import RejectionLog
from sources.k_distribution import DEFAULT_WIDEN_FRACTION, KDistribution, SampledK

#: Rótulo da rota no `Sample.route` e nas linhas do manifesto.
ROUTE = "a"

#: Nome do arquivo da bokeh em `generated/`: `<sample_id>_bokeh.jpg`.
BOKEH_IMAGE_NAME = "bokeh"

#: Variantes por imagem. `[A]` — ver o cabeçalho: `[I]` de 70K/1,7K, não número do paper.
DEFAULT_SAMPLES_PER_IMAGE = 41

#: Tolerância da banda em foco, em pixel de CoC (RAIO). `[A]`. Sub-pixel = nítido.
DEFAULT_FOCUS_BAND_COC_PX = 0.5

#: Faixa de quantis de disparidade em que o plano de foco é sorteado. `[A]`.
DEFAULT_FOCUS_QUANTILE_LOW = 0.05
DEFAULT_FOCUS_QUANTILE_HIGH = 0.95


# --------------------------------------------------------------------------------
# O vocabulário que falta — extensão, nunca reuso
# --------------------------------------------------------------------------------

#: Valor pedido para `qc.focus_region.FocusSource`: o plano de foco foi **SORTEADO**.
SAMPLED_FOCUS_SOURCE_VALUE = "sampled_plane"
#: Valor pedido para `dataio.sample.MaskSource`: a banda em foco DERIVADA do rótulo.
SAMPLED_MASK_SOURCE_VALUE = "sampled_plane"

_PATCH = f"""
A rota A precisa de dois valores de vocabulário que ainda não existem:

  1. qc/focus_region.py, em `FocusSource`:
         #: Rota A: o plano de foco foi SORTEADO (paper.txt:329-330). Não há máscara,
         #: não há BiRefNet e não há retenção — não existe região medida na cena.
         SAMPLED_PLANE = {SAMPLED_FOCUS_SOURCE_VALUE!r}

  2. dataio/sample.py, em `MaskSource`:
         #: Rota A: a banda |CoC| <= tolerância, DERIVADA do rótulo sorteado. Não é
         #: DEPTH_BAND, que era a banda no quantil 0,12 da rota A antiga (defeito A7).
         SAMPLED_PLANE = {SAMPLED_MASK_SOURCE_VALUE!r}

     e a entrada correspondente em `FOCUS_SOURCE_TO_MASK_SOURCE`, que
     `test_dataio.test_toda_fonte_de_foco_tem_fonte_de_mascara` exige ser total:
         FocusSource.SAMPLED_PLANE: MaskSource.SAMPLED_PLANE,

  3. dataio/sample.py, em `_REQUIRED_PROVENANCE`: `mask_model_sha256` não pode ser
     obrigatório numa rota que não roda segmentador nenhum. Exigir **um dos dois** —
     `mask_model_sha256` OU `mask_rule` (o dict com a regra sintética completa) — é o
     item F6 da auditoria, e não afrouxa a rota C.

Reusar `MaskSource.DEPTH_BAND` ou `FocusSource.BIREFNET` é proveniência que MENTE, e é
exatamente o `mask_source="automatic"` da cascata antiga com outra roupa. Ver
reference/ROTA_A_DECISOES.md.
""".strip()


class VocabularyExtensionRequired(RuntimeError):
    """O vocabulário fechado não tem valor para "plano de foco sorteado".

    **Não é `SampleRejected`.** Uma amostra rejeitada é um caso a calibrar; isto é um
    release que não pode existir, e um slug no histograma o disfarçaria de caso a
    calibrar. Falha no `main`, antes de qualquer GPU.
    """


def resolve_sampled_vocabulary() -> tuple[FocusSource, MaskSource]:
    """`(FocusSource, MaskSource)` do plano sorteado, ou explode com o patch exato.

    Existe como função, e não como import no topo, por uma razão prática: o módulo tem
    que ser **importável** para que os testes do sorteio, dos gates e da distribuição
    rodem antes de a extensão existir. O que não pode é **gravar** amostra sem ela.
    """
    try:
        return (FocusSource(SAMPLED_FOCUS_SOURCE_VALUE),
                MaskSource(SAMPLED_MASK_SOURCE_VALUE))
    except ValueError as exc:
        raise VocabularyExtensionRequired(_PATCH) from exc


# --------------------------------------------------------------------------------
# A fonte — protocolo, não import concreto
# --------------------------------------------------------------------------------

class AifSource(Protocol):
    """O que a rota A precisa saber de uma imagem AIF, e nada mais.

    Protocolo no molde do `PairSource` da rota C: `sources/genphoto_ebb.py` entrega esta
    forma, e a rota não conhece o adaptador. Escrever um adaptador novo — outro release
    de `[80]`, o outro lado da EBB! — não exige tocar neste arquivo.

    `scene_id` é o id da **imagem**: as N variantes o compartilham, e é a unidade do
    split. Sem isso, 41 variantes da mesma imagem cairiam dos dois lados e a validação
    mediria memorização (defeito A9).
    """

    scene_id: str
    source_dataset: str
    source_sample_id: str
    source_split: Optional[str]
    #: Onde a AIF vive. É REFERÊNCIA: a rota A não a regrava — o que ela grava é a bokeh.
    aif_ref: Optional[str]
    #: Revisão da fonte. `None` é legítimo e fica visível: o dataset histórico não gravou
    #: revisão nenhuma, e é um dos itens que ficaram impossíveis de certificar.
    source_revision: Optional[str]
    #: Variância do Laplaciano medida na enumeração, e a grade em que foi medida.
    laplacian_variance: Optional[float]
    sharpness_hw: Optional[tuple[int, int]]


#: Carrega a AIF em BGR uint8. Fora da rota de propósito, como nas outras duas.
LoadAif = Callable[[AifSource], np.ndarray]


# --------------------------------------------------------------------------------
# Config — todo limiar `None`
# --------------------------------------------------------------------------------

@dataclass
class RouteAConfig:
    """Todos os botões. **Todo limiar é `None` por default** — mede no piloto, congela
    depois.

    O que deliberadamente NÃO está aqui:

    * **`max_coc`.** É `MAX_COC = 100.0`, global e congelado (`contract.py:47`). O
      pipeline antigo o tinha como `--max-coc required=True` e o **gravava por amostra**
      (`route_a.py:207,177`) — o mecanismo exato do desastre do kfix, com LF-Bokeh +0,8288
      e RealDOF −0,4599 no mesmo lote (`ACHADOS.md:192`). `defocus_map` não aceita o
      parâmetro e esta config não expõe botão que encoste nisso (defeito A4).
    * **um renderer alternativo.** O antigo tinha
      `--renderer choices=["bokehme","smoke_gaussian"]` no caminho de produção, e os ~70K
      alvos publicados são **gaussiana de 16 camadas**, não bokeh (defeito A3). Renderer
      sintético vive no teste (`REGISTRO.md:301-303`); aqui entra o `render_fn` do BokehMe
      verificado, e o entrypoint exige o laudo.
    * **faixa de K.** Ver o cabeçalho: K é imposto de um suporte já validado; "K
      implausível" é defeito da distribuição, não da amostra.
    """

    output_dir: Path
    #: A distribuição de onde K é sorteado. **Sem default**: a rota A não tem equação
    #: para K, e um default aqui seria uma constante disfarçada de distribuição.
    k_distribution: KDistribution
    #: `[A]` = 41, de 70K/1,7K (`paper.txt:527-528`, `998`).
    samples_per_image: int = DEFAULT_SAMPLES_PER_IMAGE
    depth_long_side: int = DEPTH_LONG_SIDE
    #: **Por cena.** `PreparedImage` já calcula `encode_depth` uma vez por imagem e
    #: entrega o mesmo `EncodedDepth` às `samples_per_image` variantes; gravar por
    #: amostra guardava 41 cópias idênticas do mesmo PNG. Medido no `a_release`: 120 de
    #: 120 cenas com a profundidade byte a byte idêntica entre as 41 variantes, 31,7 GB
    #: contra 0,77 GB. É a rota em que a dedup vale mais, e por muito.
    depth_layout: DepthLayout = DepthLayout.PER_SCENE
    #: Alargamento da faixa de K, para cada lado. `[A]`, motivado pela regeração da rota B
    #: (`PLANO_EXECUCAO.md`) — se o K se mover, a cobertura já tem que existir.
    k_widen_fraction: float = DEFAULT_WIDEN_FRACTION
    #: Faixa de quantis de **disparidade** em que o plano de foco é sorteado. `[A]`.
    focus_quantile_low: float = DEFAULT_FOCUS_QUANTILE_LOW
    focus_quantile_high: float = DEFAULT_FOCUS_QUANTILE_HIGH
    #: Tolerância da banda em foco gravada em `mask/<id>.png`, em pixel de CoC. `[A]`.
    #: **Não é limiar de rejeição** — é a definição da banda.
    focus_band_coc_px: float = DEFAULT_FOCUS_BAND_COC_PX

    # --- limiares, todos [A] até o piloto ---
    #: Nitidez da AIF. Comparável só DENTRO da mesma fonte (`qc/gates.py:279-284`) — na
    #: rota A o corte principal já aconteceu na enumeração, por fonte.
    min_aif_laplacian_variance: Optional[float] = None
    min_depth_useful_levels: Optional[int] = None
    #: Razão de nitidez bokeh/AIF. Aqui ela responde *"o render fez alguma coisa?"*: com
    #: K pequeno e plano de foco cobrindo a cena, o BokehMe devolve quase a AIF.
    max_bokeh_over_aif_sharpness: Optional[float] = None
    #: p99 de `|CoC|` em pixel da resolução da IMAGEM. Âncora: o `coc_p99_px` mediano da
    #: rota B é **4,665 px** (`ACHADOS.md:37` `[M]`). Abaixo de ~1 px o alvo é
    #: indistinguível da entrada. `[A]`.
    min_rendered_coc_p99_px: Optional[float] = None
    #: Fração de pixels com `defocus == 1,0`, isto é `|CoC| ≥ MAX_COC = 100 px`. Medido na
    #: rota B com `max_coc = 10,5107`: 17,3% (`ACHADOS.md:57` `[M]`) — **com 100 é outro
    #: regime, e não foi medido**. Na rota A a saturação é escolha nossa, porque K é
    #: sorteado, e tem que ser vista no piloto. `[A]`.
    max_defocus_saturation_ratio: Optional[float] = None
    min_mask_area_ratio: Optional[float] = None
    max_mask_area_ratio: Optional[float] = None

    limit: Optional[int] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.samples_per_image) < 1:
            raise ValueError(
                f"samples_per_image = {self.samples_per_image}: uma variante por imagem é "
                "o defeito A2 e zero não gera dado. O paper implica ~41 "
                "(70K/1,7K, paper.txt:527-528 e 998).")
        if not 0.0 <= float(self.focus_quantile_low) < float(self.focus_quantile_high) <= 1.0:
            raise ValueError(
                f"faixa de quantil de foco inválida: "
                f"[{self.focus_quantile_low}, {self.focus_quantile_high}]")
        if float(self.focus_band_coc_px) <= 0:
            raise ValueError(f"focus_band_coc_px tem que ser > 0: {self.focus_band_coc_px}")

    @property
    def variant_id_width(self) -> int:
        """Largura do sufixo `_vNN`. Cresce com N para o id nunca colidir nem truncar."""
        return max(2, len(str(int(self.samples_per_image) - 1)))


# --------------------------------------------------------------------------------
# O sorteio — hipercubo latino, determinístico por `scene_id`
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantDraw:
    """Um par `(u_k, u_focus)` em [0,1), com o estrato de onde saiu.

    Os quantis vão para a proveniência de cada amostra. É o que o histórico não gravou —
    *"quais (K, D_focus) geraram cada variante"* ficou entre as coisas impossíveis de
    certificar no dataset publicado.
    """

    index: int
    u_k: float
    u_focus: float
    stratum_k: int
    stratum_focus: int
    n_strata: int


def variant_seed(scene_id: str, seed: int) -> int:
    """Semente estável derivada de `(seed, scene_id)` por sha256.

    `hash()` do Python **não** serve: é randomizado por processo (PYTHONHASHSEED), e o
    sorteio mudaria a cada execução — o release deixaria de ser reproduzível. É a mesma
    razão e a mesma construção de `dataio/split.py:29-35`.
    """
    digest = hashlib.sha256(f"{int(seed)}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def draw_plan(scene_id: str, *, n: int, seed: int = 0) -> list[VariantDraw]:
    """As N variantes de uma imagem, como **hipercubo latino** de dimensão 2.

    O problema que isto resolve: 41 sorteios i.i.d. por imagem deixam imagens inteiras sem
    K alto, por azar. Com 41 tiros independentes, a chance de nenhum cair no decil
    superior de K é `0,9^41 ≈ 1,3%` — uma imagem em cada 75, e são 1,7 mil imagens, logo
    ~23 imagens do pool inteiro sem nenhuma variante de borrão forte. O que o pré-treino
    existe para ensinar é *"a MESMA cena com K diferente borra diferente"*
    (`paper.txt:332-333`), e isso é **cobertura por imagem**, não cobertura do lote.

    No hipercubo latino, `[0,1)` é dividido em N estratos iguais e cada variante cai num
    estrato **diferente** de K e num estrato **diferente** de plano de foco — cobertura
    perfeita nas duas margens, com o pareamento entre elas sorteado. Custo declarado: as
    N variantes de uma imagem deixam de ser independentes entre si (é o ponto), e trocar
    `samples_per_image` muda todos os sorteios daquela imagem, porque os estratos mudam
    de largura. `samples_per_image` vai no `run_config.json` e na proveniência de cada
    amostra por isso.

    `[A]`: o paper diz *"randomly sample"* e cala. A estratificação é decisão nossa.
    """
    if int(n) < 1:
        raise ValueError(f"n tem que ser >= 1: {n!r}")
    n = int(n)
    rng = np.random.default_rng(variant_seed(scene_id, seed))
    estratos_k = rng.permutation(n)
    estratos_foco = rng.permutation(n)
    jitter_k = rng.random(n)
    jitter_foco = rng.random(n)
    return [
        VariantDraw(
            index=i,
            u_k=float((estratos_k[i] + jitter_k[i]) / n),
            u_focus=float((estratos_foco[i] + jitter_foco[i]) / n),
            stratum_k=int(estratos_k[i]),
            stratum_focus=int(estratos_foco[i]),
            n_strata=n,
        )
        for i in range(n)
    ]


def draw_for_variant(scene_id: str, index: int, *, n: int, seed: int = 0) -> VariantDraw:
    """O sorteio de UMA variante, sem processar a imagem inteira.

    Existe para que uma amostra isolada possa ser refeita a partir do `sample_id` — que é
    o que "reprodutível" significa em auditoria. O plano inteiro é recalculado porque o
    hipercubo latino é conjunto, não função de um ponto: isso é barato (uma permutação de
    41) e é a única forma de o número bater com o do run.
    """
    if not 0 <= int(index) < int(n):
        raise ValueError(f"índice {index!r} fora de [0, {n})")
    return draw_plan(scene_id, n=n, seed=seed)[int(index)]


# --------------------------------------------------------------------------------
# Plano de foco sorteado — e a validação que o contrato ainda não expõe
# --------------------------------------------------------------------------------

def validate_sampled_focus_disparity(
    focus_disparity: float, *,
    min_focus_depth_m: float = FOCUS_DEPTH_MIN_M,
    max_focus_depth_m: float = FOCUS_DEPTH_MAX_M,
) -> float:
    """As mesmas validações de `focus_disparity_from_mask`, para um plano **sorteado**.

    `control.contract` só tem `focus_disparity_from_mask` (`:286-323`), que **exige
    máscara** — e a rota A sorteia o plano (`paper.txt:329-330`). Este é o item F2 da
    auditoria: o corpo de `contract.py:316-321` extraído para uma função pública, com os
    dois slugs que **já estão registrados** (`focus_disparity_invalid` e
    `focus_depth_implausible`, `contract.py:163-164`). Nenhum slug novo, nenhuma constante
    nova — os limites vêm importados, nunca copiados, que é o defeito que
    `focus_depth_plausible` já teve uma vez (`qc/gates.py:316-319`).

    **O lugar certo desta função é `control/contract.py`**, e a mudança está descrita no
    relatório desta tarefa. Enquanto ela mora aqui, os limites são os do contrato por
    import, então não há como divergir em valor — só em endereço.
    """
    valor = float(focus_disparity)
    if not np.isfinite(valor) or valor <= 0:
        reject("focus_disparity_invalid", f"focus_disp sorteado = {focus_disparity!r}")
    profundidade = 1.0 / valor
    if not (float(min_focus_depth_m) <= profundidade <= float(max_focus_depth_m)):
        reject("focus_depth_implausible",
               f"z_focus sorteado = {profundidade:.3f} m, fora de "
               f"[{min_focus_depth_m}, {max_focus_depth_m}]. Com q_high próximo de 1 o "
               "quantil cai na sentinela de 10.000 m do Depth Pro (25,7% das amostras "
               "medidas, ACHADOS.md:54).")
    return valor


@dataclass(frozen=True)
class DisparityRange:
    """A faixa de disparidade da imagem e a população de planos de foco candidatos.

    Guardada porque ela é a resposta a *"o plano de foco caiu dentro da faixa da imagem?"*
    — e essa pergunta tem que ser verificável no metadado, não recomputável só com o
    código na mão.

    `disparity_min`/`max` descrevem a imagem **inteira**; `pool_fraction` diz quanta dela
    entrou na população de sorteio depois da restrição a profundidades plausíveis. Os dois
    juntos são o que permite auditar o sorteio: um `pool_fraction` de 0,4 significa que
    60% da cena está fora da faixa física — cena com muito céu —, e isso muda a
    interpretação de `z_focus` sem mudar nada visível no rótulo.
    """

    disparity_min: float
    disparity_max: float
    quantile_low: float
    quantile_high: float
    disparity_at_quantile_low: float
    disparity_at_quantile_high: float
    pool_fraction: float

    def to_metadata(self) -> dict:
        return {
            "focus_sampling_disparity_min": float(self.disparity_min),
            "focus_sampling_disparity_max": float(self.disparity_max),
            "focus_sampling_quantile_low": float(self.quantile_low),
            "focus_sampling_quantile_high": float(self.quantile_high),
            "focus_sampling_disparity_at_q_low": float(self.disparity_at_quantile_low),
            "focus_sampling_disparity_at_q_high": float(self.disparity_at_quantile_high),
            "focus_sampling_pool_fraction": float(self.pool_fraction),
            "focus_sampling_space": "disparity_1_over_m",
            "focus_sampling_rule": (
                "quantil da distribuição EMPÍRICA de 1/z da própria imagem, restrita aos "
                "pixels com z em [FOCUS_DEPTH_MIN_M, FOCUS_DEPTH_MAX_M], em "
                "[q_low, q_high]. Um quantil dos dados está dentro da faixa por "
                "definição, e existe massa de cena no plano — ver routes/route_a.py."),
        }


@dataclass(frozen=True)
class FocusSamplingPool:
    """A população de onde `D_focus` é sorteado, e a faixa da imagem em volta dela.

    Duas coisas diferentes, e separá-las é o ponto: `pool` são os **candidatos** a plano
    de foco (disparidade dos pixels com profundidade fisicamente plausível), e
    `disparity_full_min/max` é a faixa da imagem inteira, que é o que "dentro da faixa da
    imagem" quer dizer. O invariante que o sorteio confere é contra a faixa cheia; a
    população é um subconjunto dela.
    """

    pool: np.ndarray
    disparity_full_min: float
    disparity_full_max: float
    pool_fraction: float


def build_focus_sampling_pool(
    depth_m: np.ndarray, *,
    min_focus_depth_m: float = FOCUS_DEPTH_MIN_M,
    max_focus_depth_m: float = FOCUS_DEPTH_MAX_M,
) -> FocusSamplingPool:
    """Os planos de foco candidatos desta imagem, em disparidade.

    Duas filtragens, com razões diferentes:

    1. **não-finito e não-positivo** saem. `validate_metric_depth` já exige 99,9% de
       pixels finitos e positivos a montante, então isto é cinto de segurança: se sobrar
       um pixel inválido, ele não pode entrar num quantil que define o rótulo.
    2. **profundidade fora de `[FOCUS_DEPTH_MIN_M, FOCUS_DEPTH_MAX_M]`** sai da população.
       Um pixel no teto de 10.000 m do Depth Pro não é candidato a plano de foco — é
       sentinela, e 25,7% das amostras medidas têm `z_max == 10.000` (`ACHADOS.md:54`
       `[M]`). Medido: sem esta restrição, uma cena com 10% de céu perde 4,9% das
       variantes por `focus_depth_implausible` com `q = [0,05, 0,95]` e 9,8% com
       `q = [0, 1]`; com ela, 0,0%.

    **Não é clamp e não é fallback**: nenhum valor é substituído por constante. É a
    definição da população de sorteio, e a fração que sobrou vai para o metadado
    (`focus_sampling_pool_fraction`), onde uma cena de 95% de céu fica visível.

    Imagem sem nenhum plano plausível é rejeitada com `focus_depth_implausible` — e ela
    custa as N variantes, contadas em `RouteAStats.variants_lost_to_image`.
    """
    z = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    validos = z[np.isfinite(z) & (z > 0)]
    if validos.size == 0:
        reject("depth_non_positive", "nenhum pixel de profundidade válido na imagem")
    disparidade = 1.0 / validos
    finita = disparidade[np.isfinite(disparidade) & (disparidade > 0)]
    if finita.size == 0:
        reject("depth_non_finite", "nenhuma disparidade válida na imagem")

    plausivel = validos[(validos >= float(min_focus_depth_m))
                        & (validos <= float(max_focus_depth_m))]
    if plausivel.size == 0:
        reject("focus_depth_implausible",
               f"nenhum pixel com z em [{min_focus_depth_m}, {max_focus_depth_m}]: não "
               "existe plano de foco candidato nesta imagem. Cena inteira no teto do "
               "Depth Pro (ACHADOS.md:54) ou profundidade fora de escala.")
    return FocusSamplingPool(
        pool=1.0 / plausivel,
        disparity_full_min=float(finita.min()),
        disparity_full_max=float(finita.max()),
        pool_fraction=float(plausivel.size / validos.size),
    )


def sample_focus_disparity(
    pool: FocusSamplingPool, u: float, *, quantile_low: float, quantile_high: float,
) -> tuple[float, DisparityRange]:
    """Sorteia `focus_disparity` como quantil da população candidata **desta** imagem.

    Devolve o valor e a faixa, porque o metadado precisa dos dois: o valor é o rótulo, a
    faixa é o que prova que ele caiu dentro da imagem.
    """
    if not 0.0 <= float(u) <= 1.0:
        raise ValueError(f"u fora de [0, 1]: {u!r}")
    q = float(quantile_low) + float(u) * (float(quantile_high) - float(quantile_low))
    valor = float(np.quantile(pool.pool, q))
    faixa = DisparityRange(
        disparity_min=pool.disparity_full_min, disparity_max=pool.disparity_full_max,
        quantile_low=float(quantile_low), quantile_high=float(quantile_high),
        disparity_at_quantile_low=float(np.quantile(pool.pool, float(quantile_low))),
        disparity_at_quantile_high=float(np.quantile(pool.pool, float(quantile_high))),
        pool_fraction=pool.pool_fraction,
    )
    # Por construção `valor` está na faixa da imagem; conferir é barato e pega mudança
    # futura na regra de amostragem antes de ela virar 70 mil rótulos degenerados.
    if not (faixa.disparity_min <= valor <= faixa.disparity_max):
        reject("focus_disparity_invalid",
               f"plano sorteado {valor!r} fora da faixa da imagem "
               f"[{faixa.disparity_min}, {faixa.disparity_max}]")
    return valor, faixa


def focus_band_mask(coc_px: np.ndarray, *, band_px: float) -> np.ndarray:
    """A banda em foco: `|CoC| <= band_px`. **Derivada do rótulo, não medida na cena.**

    Rejeita banda vazia com `focus_mask_empty`: pode acontecer quando a disparidade tem
    um salto largo em torno do plano sorteado e K é alto — nenhum pixel fica a menos de
    meio pixel de CoC. Nesse caso não existe região em foco nesta cena com este par
    `(D_focus, K)`, e gravar `focus_region_area_ratio == 0` é reprovado por
    `validate_metadata` (`dataio/sample.py:467-469`) — corretamente: região vazia não
    define plano de foco.
    """
    banda = np.abs(np.asarray(coc_px, dtype=np.float64)) <= float(band_px)
    if not banda.any():
        reject("focus_mask_empty",
               f"nenhum pixel com |CoC| <= {band_px} px: não existe região em foco nesta "
               "cena com este par (D_focus, K) sorteado")
    return banda


# --------------------------------------------------------------------------------
# Gates próprios da rota A
# --------------------------------------------------------------------------------
# Vivem aqui, e não em `qc/gates.py`, pela mesma razão dos três gates da rota B: eles só
# fazem sentido quando o ALVO é renderizado a partir de um K imposto. Se passarem a valer
# para outra rota, migram para `qc/gates.py` — que é o lugar certo — e esta seção some.
# Ver o relatório desta tarefa, item "o que mudar em módulos alheios".

def rendered_coc_p99_px(
    coc_px: np.ndarray, *, min_p99: Optional[float] = None,
) -> GateResult:
    """p99 de `|CoC|`, em pixel da resolução da IMAGEM.

    É o gate que responde *"este rótulo pede borrão visível?"*. Abaixo de ~1 px o alvo
    renderizado é indistinguível da entrada e a amostra não carrega sinal de controle —
    ela ensina a rede a copiar a entrada quando `D_def` é pequeno, o que é verdade, mas
    41 variantes por imagem não podem ser todas assim.

    Âncora medida: o `coc_p99_px` mediano da rota B é **4,665 px** (`ACHADOS.md:37`
    `[M]`), com min 0,1002 e max 112,0. Default `None`: o número sai do piloto.

    O nome carrega a unidade **e** a grade, porque `K` vive na escala de pixel da imagem
    e a profundidade gravada vive em `depth_hw` — misturar as duas produz um "px" sem
    resolução, que é o que a regra 3 do contrato proíbe.
    """
    p99 = float(np.percentile(np.abs(np.asarray(coc_px, dtype=np.float64)), 99))
    return GateResult("rendered_coc_p99_px", p99, min_p99, True,
                      "p99 de |CoC| em px da resolução da imagem; âncora: mediana 4,665 "
                      "px na rota B (ACHADOS.md:37)")


def defocus_saturation_ratio(
    defocus: np.ndarray, *, max_ratio: Optional[float] = None,
) -> GateResult:
    """Fração de pixels com `defocus == 1,0`, isto é `|CoC| >= MAX_COC = 100 px`.

    Saturação alta significa que o mapa de condição perdeu a informação de **quanto**
    borrar numa parte grande da imagem: acima do teto, dois CoC diferentes viram o mesmo
    número. Na rota B isso foi medido com o normalizador errado do kfix
    (`max_coc = 10,5107` → 17,3%, `ACHADOS.md:57`); **com o `MAX_COC = 100` correto o
    regime é outro e não foi medido**.

    Na rota A a saturação é **escolha nossa**, porque K é sorteado — é o gate que diz se o
    suporte alargado da distribuição está pedindo borrão que o mapa não sabe representar.
    Default `None`: sai do piloto.
    """
    razao = float(np.mean(np.asarray(defocus, dtype=np.float64) >= 1.0))
    return GateResult("defocus_saturation_ratio", razao, max_ratio, False,
                      f"fração de pixels com |CoC| >= MAX_COC = {MAX_COC:.0f} px; acima "
                      "do teto o mapa não distingue mais quanto borrar")


# --------------------------------------------------------------------------------
# O relatório de gates
# --------------------------------------------------------------------------------

def build_gate_report(
    *, aif_bgr: np.ndarray, bokeh_bgr: np.ndarray, mask: np.ndarray,
    disparity_u16: np.ndarray, coc_px: np.ndarray, defocus: np.ndarray,
    focus_disparity: float, source: AifSource, config: RouteAConfig,
) -> GateReport:
    """Todos os gates da rota A, na ordem em que valem a pena olhar no histograma.

    Cinco gates das outras rotas **não entram**, e cada ausência é decisão:

    * `calibration_ssim_is_reliable` — não há Eq. 5 nesta rota (`CONTRATO.md:160`).
    * `aif_aperture_is_narrow` — nem `[80]` nem a EBB! publicam f-stop da AIF por imagem
      no que enumeramos; registrar `applicable=False` com a nota do gate (*"a origem
      correta é train/in/<id>_f22.JPG"*) poria uma frase sobre a RealBokeh no metadado de
      70 mil amostras.
    * `mask_iou` / `mask_border_coverage` — não há duas máscaras a comparar, e a heurística
      de borda (*"colada na borda costuma ser fundo, não objeto"*) descreve um segmentador
      de objeto saliente. A banda em foco legitimamente toca a borda quando o plano
      sorteado é o fundo.
    * `focus_mask_is_sharpest` — tautológico aqui: a banda **é** onde o CoC é sub-pixel.
      Ver o cabeçalho do módulo.
    * `pair_shape_matches` — `process_variant` já rejeita shape divergente com
      `resolution_invalid` **antes** de medir qualquer coisa, e com mensagem melhor. Manter
      o gate aqui o tornaria um gate que não pode reprovar, que é pior que gate nenhum
      (`REGISTRO.md:695-696`).

    `focus_depth_plausible` **entra, e por construção não deveria disparar**: a população
    de sorteio já é restrita a profundidades plausíveis (`build_focus_sampling_pool`). Ele
    fica como arame de tropeço entre módulos — se a regra de amostragem mudar, ou se o pool
    vier de outro lugar, é ele que acusa. A diferença em relação ao `pair_shape_matches`
    acima é que ali a redundância é dentro da mesma função, e aqui ela cruza uma fronteira.

    E `bokeh_is_blurrier_than_aif` muda de papel: nas outras rotas ele confere a sanidade
    de um par fotográfico; aqui ele pergunta *"o renderer fez alguma coisa?"*. É o gate que
    pegaria um BokehMe carregado com peso errado devolvendo a entrada — o defeito A3 pelo
    lado do silêncio.
    """
    report = GateReport()
    report.add(aif_sharpness(aif_bgr, min_variance=config.min_aif_laplacian_variance))
    report.add(bokeh_is_blurrier_than_aif(
        aif_bgr, bokeh_bgr, max_ratio=config.max_bokeh_over_aif_sharpness))

    report.add(rendered_coc_p99_px(coc_px, min_p99=config.min_rendered_coc_p99_px))
    report.add(defocus_saturation_ratio(
        defocus, max_ratio=config.max_defocus_saturation_ratio))

    for result in mask_area_ratio(mask, min_ratio=config.min_mask_area_ratio,
                                  max_ratio=config.max_mask_area_ratio):
        report.add(result)

    report.add(depth_useful_levels(disparity_u16,
                                   min_levels=config.min_depth_useful_levels))
    for result in focus_depth_plausible(focus_disparity):
        report.add(result)
    return report


#: Nome de gate -> slug de rejeição do conjunto FECHADO de `control.contract`. Sem esta
#: ponte o histograma receberia nomes de gate que não são slugs, e viraria vocabulário
#: aberto — e é o histograma que denuncia fallback novo.
#:
#: **Duas entradas são SLUGS PROVISÓRIOS**, marcadas abaixo. Os slugs próprios
#: (`gate_rendered_coc_too_small`, `gate_defocus_saturated`) exigem acrescentar valor a
#: `GATE_REJECTION_REASONS` (`control/contract.py:94-125`), que é `frozenset` fechado num
#: módulo que esta tarefa não pode editar. Enquanto isso: o **nome do gate** preserva a
#: distinção no `quality` de cada amostra, e o histograma agrega sob o slug existente mais
#: próximo. Ver o relatório.
GATE_TO_REASON = {
    "aif_laplacian_variance": "gate_aif_sharpness",
    "bokeh_over_aif_sharpness": "gate_bokeh_not_blurrier",
    # PROVISÓRIO: pede `gate_rendered_coc_too_small`. Fica em `gate_bokeh_not_blurrier`
    # porque é literalmente o mesmo fato — o alvo não ficou mais borrado que a entrada —
    # medido no CoC em vez de na razão de nitidez.
    "rendered_coc_p99_px": "gate_bokeh_not_blurrier",
    # PROVISÓRIO: pede `gate_defocus_saturated`. `k_out_of_configured_range` é o slug
    # registrado mais próximo: saturação alta é K grande demais para ESTA cena, que é a
    # forma que "K fora da faixa" assume quando o K é imposto e não medido.
    "defocus_saturation_ratio": "k_out_of_configured_range",
    "mask_area_ratio_min": "gate_mask_area_ratio",
    "mask_area_ratio_max": "gate_mask_area_ratio",
    "depth_useful_levels": "gate_depth_useful_levels",
    "focus_depth_m_min": "gate_focus_depth_implausible",
    "focus_depth_m_max": "gate_focus_depth_implausible",
}


def enforce_gates(report: GateReport) -> None:
    """Primeiro gate reprovado vira `SampleRejected` com slug do conjunto fechado."""
    blocked = report.blocked_by
    if blocked:
        reject(GATE_TO_REASON[blocked[0]], f"gates reprovados: {blocked}")


# --------------------------------------------------------------------------------
# Estatística do run
# --------------------------------------------------------------------------------

@dataclass
class RouteAStats:
    """O que o run mediu. Em **cenas e em amostras**, sempre (`CLAUDE.md:51-52`).

    70 mil amostras vindas de 1,7 mil imagens não são 70 mil unidades de diversidade, e
    na rota A essa distinção é a mais importante de todas: a razão amostras/cena **é** o
    parâmetro central do desenho (~41), e se ela cair sem ninguém ver, o dataset voltou a
    ser o de uma variante por imagem (defeito A2) com outro tamanho.
    """

    images_processed: int = 0
    written: int = 0
    skipped_done: int = 0
    #: Variantes perdidas porque a IMAGEM falhou (leitura, profundidade). Contadas
    #: separadas das rejeições por variante: uma imagem ruim custa N amostras, e somar as
    #: duas coisas num contador só esconderia qual das duas está governando o lote.
    variants_lost_to_image: int = 0
    images_lost: int = 0
    k_values: list = field(default_factory=list)
    k_per_long_side: list = field(default_factory=list)
    focus_depth_m: list = field(default_factory=list)
    coc_p99_values: list = field(default_factory=list)
    saturation_values: list = field(default_factory=list)
    band_area_values: list = field(default_factory=list)
    #: Quantas variantes tiveram K preso no piso positivo do alargamento.
    k_clamped: int = 0
    per_source: Counter = field(default_factory=Counter)
    per_scene: Counter = field(default_factory=Counter)

    def _percentis(self, valores: list, rotulo: str, fmt: str = "{:.4f}") -> list[str]:
        finitos = np.asarray([v for v in valores
                              if v is not None and np.isfinite(v)], dtype=np.float64)
        if not finitos.size:
            return [f"  {rotulo:<28} ausente em todas as amostras aceitas"]
        p05, p50, p95 = np.percentile(finitos, [5, 50, 95])
        return [f"  {rotulo:<28} p05 {fmt.format(p05)}  mediana {fmt.format(p50)}  "
                f"p95 {fmt.format(p95)}   (n={finitos.size})"]

    def coverage_summary(self) -> list[str]:
        """Cenas × amostras, e a razão que é o desenho da rota.

        Separado de `summary()` para poder ser testado sem montar um run inteiro, e
        porque é o número que denuncia o defeito A2 voltando: razão perto de 1 significa
        uma variante por imagem, e aí K está confundido com conteúdo.
        """
        if not self.per_scene:
            return []
        cenas = len(self.per_scene)
        amostras = sum(self.per_scene.values())
        razao = amostras / cenas
        por_imagem = Counter(self.per_scene.values())
        linhas = ["-" * 62,
                  f"  cenas (imagens)  : {cenas}",
                  f"  amostras         : {amostras}   ({razao:.1f} por imagem)",
                  "    variantes por imagem: "
                  + " · ".join(f"{k} -> {v}" for k, v in sorted(por_imagem.items()))]
        if razao < 2.0:
            linhas.append("    >>> ATENÇÃO: razão < 2. Uma variante por imagem é o "
                          "defeito A2 — cada cena com um K só,")
            linhas.append("        e o sinal 'a MESMA cena com K diferente borra "
                          "diferente' não existe no dado.")
        return linhas

    def summary(self) -> str:
        if not self.k_values:
            return "[rota-a] nenhuma amostra aceita."
        linhas = [
            "",
            "=" * 62,
            f"  aceitas          : {self.written}",
            f"  imagens vistas   : {self.images_processed}",
            f"  já feitas (skip) : {self.skipped_done}",
            f"  imagens perdidas : {self.images_lost}  "
            f"(custaram {self.variants_lost_to_image} variantes)",
            "-" * 62,
        ]
        linhas += self._percentis(self.k_per_long_side, "k_per_long_side (sorteado)",
                                  "{:.6f}")
        linhas.append("    a grandeza LIVRE DE RESOLUÇÃO em que a distribuição vive")
        linhas += self._percentis(self.k_values, "k_value (px da imagem)", "{:.2f}")
        linhas.append("    = k_per_long_side · max(H,W). Sem essa multiplicação o K "
                      "seria de outra resolução (A5).")
        if self.k_clamped:
            linhas.append(f"    K preso no piso positivo: {self.k_clamped} "
                          f"({100 * self.k_clamped / max(self.written, 1):.1f}%) — "
                          "cauda baixa do alargamento")
        linhas += self._percentis(self.focus_depth_m, "z_focus (m)", "{:.3f}")
        linhas.append("    sorteado como QUANTIL da disparidade da própria imagem")
        linhas += self._percentis(self.coc_p99_values, "coc_p99_px", "{:.3f}")
        linhas.append("    âncora medida: mediana 4,665 px na rota B (ACHADOS.md:37)")
        linhas += self._percentis(self.saturation_values, "defocus saturado", "{:.4f}")
        linhas.append(f"    fração com |CoC| >= MAX_COC = {MAX_COC:.0f}. Na rota B foram "
                      "17,3% com max_coc = 10,5107")
        linhas.append("    (ACHADOS.md:57) — com 100 é OUTRO regime e este é o primeiro "
                      "número dele.")
        linhas += self._percentis(self.band_area_values, "banda em foco (área)", "{:.4f}")
        linhas.append("    |CoC| <= tolerância. DERIVADA do rótulo, não medida na cena.")
        if self.per_source:
            linhas.append("-" * 62)
            linhas.append("  por fonte (paper.txt:996 — [80] e EBB!):")
            for fonte, n in sorted(self.per_source.items()):
                linhas.append(f"    {fonte:<26} {n:>7}")
        linhas += self.coverage_summary()
        linhas.append("=" * 62)
        return "\n".join(linhas)


# --------------------------------------------------------------------------------
# Uma imagem — o que é calculado UMA vez para as N variantes
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedImage:
    """Profundidade e codificação de UMA imagem, reusadas pelas N variantes.

    Existe por economia de GPU que muda a ordem de grandeza do run: a profundidade
    **não depende do sorteio**, e recomputá-la 41 vezes por imagem é 41x de Depth Pro
    jogado fora. `encode_depth` também roda uma vez.

    Consequência de armazenamento — o `[A]` A13 da auditoria, hoje **FECHADO**: as N
    variantes compartilham o mesmo array de profundidade, e o writer grava um PNG por
    CENA (`depth/<scene_id>.png`, `dataio.layout.DepthLayout.PER_SCENE`) em vez de 41
    cópias idênticas. Medido no `a_release`, em 120 cenas sorteadas com semente fixa:
    120/120 com a profundidade byte a byte idêntica entre as variantes, 0 divergentes.
    A coluna de controle das 69.700 amostras cai de **31,7 GB para 0,77 GB**.

    Cada `meta/` carrega `depth_ref` com o caminho — o `scene_id` não é dedutível do
    `sample_id`, então a referência é o contrato, não uma convenção a remontar.
    """

    source: AifSource
    aif_bgr: np.ndarray
    depth_m: np.ndarray
    depth_backend: str
    encoded: EncodedDepth
    focus_pool: FocusSamplingPool
    image_hw: tuple[int, int]


def prepare_image(
    source: AifSource, *, aif_bgr: np.ndarray, depth_runtime, config: RouteAConfig,
) -> PreparedImage:
    """Uma imagem -> profundidade validada + disparidade de sorteio. Uma vez por imagem.

    Rejeita, nunca conserta. Uma falha aqui custa **as N variantes** daquela imagem, e é
    por isso que `RouteAStats` conta `variants_lost_to_image` separado: somar essa perda
    às rejeições por variante esconderia qual das duas governa o lote.
    """
    aif_bgr = np.asarray(aif_bgr)
    if aif_bgr.ndim != 3 or aif_bgr.shape[2] != 3:
        reject("resolution_invalid", f"AIF com shape {aif_bgr.shape}")
    image_hw = (int(aif_bgr.shape[0]), int(aif_bgr.shape[1]))
    aif_rgb = np.ascontiguousarray(aif_bgr[..., ::-1])

    depth = depth_runtime.infer(aif_rgb)
    depth_m = np.asarray(depth.values_m)
    if depth_m.shape[:2] != image_hw:
        # K vive na escala de pixel da imagem, e o renderer recebe os dois arrays juntos
        # (`renderer/bokehme.py:324-327`). Grades diferentes aqui produziriam um CoC em
        # pixel de uma grade aplicado noutra — a família do defeito D12.
        reject("resolution_invalid",
               f"profundidade em {depth_m.shape[:2]} contra AIF em {image_hw}: o CoC em "
               "pixel tem que ser calculado na grade da imagem que vai ser renderizada")

    encoded = encode_depth(depth_m, image_hw=image_hw, long_side=config.depth_long_side)
    return PreparedImage(
        source=source, aif_bgr=aif_bgr, depth_m=depth_m, depth_backend=depth.backend,
        encoded=encoded, focus_pool=build_focus_sampling_pool(depth_m),
        image_hw=image_hw,
    )


# --------------------------------------------------------------------------------
# Uma variante -> uma amostra
# --------------------------------------------------------------------------------

def sample_id_for(scene_id: str, index: int, *, width: int = 2) -> str:
    """`<scene_id>_v<NN>` — o conserto do defeito A9.

    O antigo era `a_{...}_{index:06d}` com o **índice na lista** do manifesto
    (`bokehnet-preprocessing/src/pipelines/route_a.py:139`): reordenar o manifesto
    renomeava todas as amostras, e não havia `scene_id` em lugar nenhum. Aqui o id é
    função da cena e da variante, e sobrevive a qualquer reordenação.
    """
    return f"{scene_id}_v{int(index):0{int(width)}d}"


def _focus_record(
    mask: np.ndarray, *, focus_source: FocusSource, image_hw: tuple[int, int],
) -> FocusRegionRecord:
    """O `FocusRegionRecord` de um plano **sorteado** — e o que cada campo afirma aqui.

    `FocusRegionRecord` foi escrito para o refinamento do §3.2(c), em que há duas
    fotografias reais e portanto **retenção de detalhe**. A rota A não tem nenhuma das
    duas coisas: uma foto só, e o plano de foco vem de um sorteio. Então os campos são
    preenchidos assim, e cada escolha está declarada em `focus_region_semantics` na
    proveniência de cada amostra:

    * `agreement`, `precision`, `iou` = **1,0 por construção, não por medição.** A banda
      é definida pelo próprio plano de foco: comparar a região com o plano que a define
      dá 1,0 por identidade. É o mesmo raciocínio que a rota B registra para
      `precision`/`iou` (*"a máscara crua É a região; contra si mesma são 1,0 por
      definição"*, `routes/route_b.py:812-815`). Ler esses três como evidência de
      qualidade nesta rota é erro, e é por isso que a nota vai gravada.
    * `retention_in_region` = **NaN**, que `validate_metadata` aceita e significa "não
      havia detalhe medível". Aqui o significado exato é mais forte: **não existe
      retenção**, porque não existem duas fotos. `0,0` seria "reteve zero", que é falso.
    * `retention_hw` = a grade em que a banda foi calculada (a da imagem), e
      `retention_window_px` = **1**, que é literalmente verdade: não há filtro de média,
      a banda é por pixel. O nome do campo é que está errado para esta rota — deveria ser
      `region_hw` —, e a renomeação está descrita no relatório.
    * `initial_mask_was_empty` = `False` e `initial_mask_area_ratio` = `None`: não houve
      máscara inicial **nenhuma**, que o próprio docstring do campo distingue de "rodou e
      devolveu vazio" (`dataio/sample.py:204-206`).
    * `disparity_from_initial_mask` = `None`: não existe linha de base sem refinamento
      porque não existe refinamento.

    Um efeito colateral conhecido: `was_refined` é `FocusSource(...) is not BIREFNET`
    (`dataio/sample.py:223`), então `focus_was_refined` sai **True** para a rota A — o que
    é falso, porque não houve refinamento nenhum. A correção é mudar a propriedade para
    testar pertinência ao conjunto `{BIREFNET_REFINED, RETENTION_ONLY}`; está no relatório
    e **não** foi aplicada aqui.
    """
    binaria = np.asarray(mask) > 0.5
    return FocusRegionRecord(
        source=focus_source,
        agreement=1.0, precision=1.0, iou=1.0,
        retention_in_region=float("nan"),
        area_ratio=float(binaria.mean()),
        retention_hw=(int(image_hw[0]), int(image_hw[1])),
        retention_window_px=1,
        initial_mask_was_empty=False,
        initial_mask_area_ratio=None,
        disparity_from_initial_mask=None,
    )


def _mask_rule(
    *, draw: VariantDraw, sampled_k: SampledK, focus_disparity: float, k_value: float,
    band_px: float, image_hw: tuple[int, int], seed: int,
) -> dict:
    """A regra que RECONSTRÓI a banda em foco, byte a byte.

    É o item F6 da auditoria: `SampleProvenance.mask_model_sha256` é obrigatório e a rota
    A não roda modelo de máscara nenhum. A saída honesta é exigir **um dos dois** —
    o hash do modelo **ou** esta regra —, porque a regra é o análogo exato do hash: é o
    que permite refazer a máscara. Sem ela, `mask/<id>.png` seria um array sem origem
    declarada, que é o defeito A7 (um array em três chaves, proveniência
    `"depth_band_q12"` num dict livre).
    """
    return {
        "rule": "focus_band = abs(signed_coc_px(depth_m, focus_disparity, k_value)) "
                "<= focus_band_coc_px",
        "focus_band_coc_px": float(band_px),
        "grid": "image",
        "image_h": int(image_hw[0]), "image_w": int(image_hw[1]),
        "focus_disparity": float(focus_disparity),
        "k_value": float(k_value),
        "max_coc": MAX_COC,
        "seed": int(seed),
        "variant_index": int(draw.index),
        "u_focus": float(draw.u_focus),
        "u_k": float(draw.u_k),
        **sampled_k.to_metadata(),
        "is_derived_from_label": True,
        "is_measured_on_scene": False,
        "note": "A rota A NÃO tem máscara (paper.txt:329-330): o plano de foco é "
                "sorteado. Esta banda é função determinística de (D, D_focus, K) e não "
                "afirma nada sobre a cena. Não é o depth_band_q12 da rota A antiga.",
    }


def process_variant(
    prepared: PreparedImage,
    draw: VariantDraw,
    *,
    render_fn,
    config: RouteAConfig,
    provenance_base: dict,
    vocabulary: Optional[tuple[FocusSource, MaskSource]] = None,
) -> Sample:
    """Uma variante -> uma amostra. `SampleRejected` com slug em qualquer falha.

    A ordem é a do paper (`paper.txt:328-332`): profundidade já estimada, **sorteia**
    `D_focus` e `K`, Eq. 2, renderiza. O `render_fn` é o BokehMe verificado; o alvo que
    ele produz **é o rótulo** desta rota, e é o único lugar do projeto em que isso vale.
    """
    fonte_de_foco, fonte_de_mascara = vocabulary or resolve_sampled_vocabulary()
    image_hw = prepared.image_hw
    lado_longo = max(image_hw)

    # --- K: sorteio na grandeza livre de resolução, depois remultiplicado ---------
    sorteado = config.k_distribution.sample(
        draw.u_k, widen_fraction=config.k_widen_fraction)
    k_value = sorteado.at_long_side(lado_longo)

    # --- plano de foco: quantil da disparidade DESTA imagem ----------------------
    focus_disparity, faixa = sample_focus_disparity(
        prepared.focus_pool, draw.u_focus,
        quantile_low=config.focus_quantile_low,
        quantile_high=config.focus_quantile_high)
    validate_sampled_focus_disparity(focus_disparity)

    # --- Eq. 2, pela implementação única do contrato -----------------------------
    coc_px = signed_coc_px(prepared.depth_m, focus_disparity, k_value)
    defocus = defocus_map(prepared.depth_m, focus_disparity, k_value)
    mask = focus_band_mask(coc_px, band_px=config.focus_band_coc_px)

    # --- o alvo, renderizado -----------------------------------------------------
    bokeh_bgr = np.asarray(render_fn(prepared.aif_bgr, prepared.depth_m,
                                     focus_disparity, k_value))
    if bokeh_bgr.shape[:2] != image_hw:
        reject("resolution_invalid",
               f"bokeh renderizada em {bokeh_bgr.shape[:2]} contra AIF em {image_hw}")

    report = build_gate_report(
        aif_bgr=prepared.aif_bgr, bokeh_bgr=bokeh_bgr, mask=mask,
        disparity_u16=prepared.encoded.disparity_u16, coc_px=coc_px, defocus=defocus,
        focus_disparity=focus_disparity, source=prepared.source, config=config)
    enforce_gates(report)

    fonte = prepared.source
    return Sample(
        sample_id=sample_id_for(fonte.scene_id, draw.index,
                                width=config.variant_id_width),
        route=ROUTE,
        refs=SampleRefs(
            fonte.source_dataset, fonte.source_sample_id, fonte.scene_id,
            # A AIF é REAL e já existe na fonte: referência. A bokeh é PRODUTO desta
            # rota, então `bokeh_ref` é `None` — apontar para a origem afirmaria que ela
            # já existia lá. É o inverso exato da rota B.
            aif_ref=getattr(fonte, "aif_ref", None), bokeh_ref=None,
            source_split=getattr(fonte, "source_split", None)),
        control=ControlLabel(
            k_value=k_value, k_source=KSource.SAMPLED_FROM_BC,
            focus_disparity=focus_disparity,
            # Não há varredura, logo não há teto do qual censurar.
            is_k_censored=False,
            depth_backend=prepared.depth_backend,
            # Sem Eq. 5 nesta rota (`CONTRATO.md:160`).
            calibration_ssim=None,
            # Não há validador analítico: a Eq. 3 pede EXIF que esta fonte não publica, e
            # repetir o K sorteado aqui o faria passar por validador independente sem ser.
            k_analytic=None,
            # O BokehMe renderiza o alvo DESTA rota, então o fator medido no laudo
            # (0,9873, `ACHADOS.md:279-283`) descreve o renderizador que produziu o
            # rótulo. Gravado, **não corrigido** — corrigir exige medir com cena real.
            k_effective_factor=provenance_base.get("k_effective_factor"),
        ),
        depth=prepared.encoded, mask=mask, mask_source=fonte_de_mascara,
        focus=_focus_record(mask, focus_source=fonte_de_foco, image_hw=image_hw),
        provenance=SampleProvenance(
            pipeline_commit=provenance_base["pipeline_commit"],
            depth_model_sha256=provenance_base["depth_model_sha256"],
            # A rota A **não roda segmentador**. String vazia, e a regra completa da
            # banda vai em `mask_rule` — ver `_mask_rule` e o item F6 no relatório.
            mask_model_sha256=provenance_base.get("mask_model_sha256", ""),
            image_hw=image_hw, seed=config.seed,
            # O renderizador É o produtor do rótulo aqui (paper.txt:331, 292). Na rota B
            # este campo é `None` e na C ele calibra o K; esta é a única rota em que ele
            # gera o alvo.
            renderer=provenance_base.get("renderer"),
            source_license=provenance_base.get("source_license"),
            extra={
                "mask_rule": _mask_rule(
                    draw=draw, sampled_k=sorteado, focus_disparity=focus_disparity,
                    k_value=k_value, band_px=config.focus_band_coc_px,
                    image_hw=image_hw, seed=config.seed),
                "focus_region_semantics": (
                    "focus_agreement/precision/iou = 1,0 POR CONSTRUÇÃO, não por "
                    "medição: a banda é definida pelo plano de foco sorteado e "
                    "comparada com ele mesmo. focus_retention_in_region = NaN porque "
                    "não existe retenção — a rota A tem UMA foto, não um par. Ver "
                    "routes/route_a._focus_record."),
                "sampling": {
                    "k_source_rule": "quantil da distribuição empírica de B e C, na "
                                     "grandeza k_per_long_side, alargada em "
                                     f"{config.k_widen_fraction:.2f} para cada lado",
                    "k_distribution_schema": config.k_distribution.schema,
                    "k_distribution_n": int(config.k_distribution.n),
                    "k_distribution_control_version":
                        config.k_distribution.control_version,
                    "k_per_long_side_to_k_value": f"× max(H,W) = {lado_longo}",
                    "samples_per_image": int(config.samples_per_image),
                    "samples_per_image_evidence":
                        "[A] ~41 = 70K/1,7K (paper.txt:527-528, 998); o paper não "
                        "publica o número",
                    "stratification": "hipercubo latino 2D (K × plano de foco), um "
                                      "estrato de cada por variante",
                    "stratum_k": int(draw.stratum_k),
                    "stratum_focus": int(draw.stratum_focus),
                    "n_strata": int(draw.n_strata),
                    "variant_seed": int(variant_seed(fonte.scene_id, config.seed)),
                    **faixa.to_metadata(),
                },
                "source_revision": getattr(fonte, "source_revision", None),
                "aif_laplacian_variance": getattr(fonte, "laplacian_variance", None),
                "aif_sharpness_hw": list(getattr(fonte, "sharpness_hw", None) or ()),
                "route_a_decisions": {
                    "mask": "banda derivada do rótulo; a rota A NÃO tem máscara "
                            "(paper.txt:329-330)",
                    "birefnet": None,
                    "deblurnet": None,
                    "eq3": None, "eq4": None, "eq5": None,
                    "eq2": "paper.txt:312, aplicada — é a única equação desta rota",
                    "target": "RENDERIZADA pelo BokehMe [43] (paper.txt:331, 292)",
                    "renderer_bias_caveat": "paper.txt:333-334 — o próprio paper diz que "
                                            "a rota A é limitada por viés de renderer",
                },
            },
        ),
        quality=report.to_dict(),
        # A bokeh é GERADA: ela é gravada. `channel_order` DECLARADO, não assumido.
        generated_images={BOKEH_IMAGE_NAME: bokeh_bgr},
        channel_order="bgr",
    )


# --------------------------------------------------------------------------------
# O laço
# --------------------------------------------------------------------------------

def _ledger_line(root: Path, sample: Sample, meta: dict) -> dict:
    """Uma linha do ledger de bytes da rota A.

    Na rota B a AIF é gerada e a bokeh é referência; **aqui é o inverso**. O sha256 é do
    **JPEG em disco**, não do array em memória: JPEG q95 é lossy, e hashear o array
    provaria uma coisa enquanto o release contém outra.
    """
    caminho = root / "generated" / f"{sample.sample_id}_{BOKEH_IMAGE_NAME}.jpg"
    digest = None
    if caminho.is_file():
        h = hashlib.sha256()
        with caminho.open("rb") as fh:
            for bloco in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloco)
        digest = h.hexdigest()
    prov = sample.provenance.extra or {}
    return {
        "sample_id": sample.sample_id,
        "route": ROUTE,
        "scene_id": sample.refs.scene_id,
        "bokeh_role": "generated_by_bokehme",
        "bokeh_path": str(caminho.relative_to(root)) if caminho.is_file() else None,
        "bokeh_jpeg_sha256": digest,
        "bokeh_jpeg_bytes": caminho.stat().st_size if caminho.is_file() else None,
        "aif_role": "reference",
        "aif_ref": sample.refs.aif_ref,
        "source_dataset": sample.refs.source_dataset,
        "source_sample_id": sample.refs.source_sample_id,
        "source_revision": prov.get("source_revision"),
        "k_value": meta["k_value"],
        "focus_disparity": meta["focus_disparity"],
        "image_h": meta["image_h"], "image_w": meta["image_w"],
    }


def run_route_a(
    sources: Iterable[AifSource],
    *,
    load_aif: LoadAif,
    depth_runtime,
    render_fn,
    split: SceneSplit,
    config: RouteAConfig,
    provenance_base: dict,
) -> RouteAStats:
    """Laço principal. Imprime o histograma de motivos de rejeição no fim, **sempre**.

    Duas coisas diferentes da rota C, e as duas vêm de a rota A ter N variantes por
    imagem:

    1. **A retomada é conferida ANTES de carregar a imagem.** Se as N variantes já estão
       no manifesto, a imagem não é lida e o Depth Pro não roda. Num run de 70 mil
       amostras retomado no meio, a diferença é horas de GPU.
    2. **Uma falha de imagem custa N variantes**, e isso é contado separado das rejeições
       por variante (`variants_lost_to_image`). A rejeição da imagem entra no histograma
       **uma vez**, com `variants_lost` no payload: repeti-la N vezes inflaria o
       denominador do histograma e faria uma imagem ilegível parecer 41 amostras ruins.
    """
    # Falha aqui, antes de qualquer imagem: gravar amostra com vocabulário emprestado é
    # proveniência que mente, e é pior que não gravar.
    vocabulario = resolve_sampled_vocabulary()

    stats = RouteAStats()
    saida = Path(config.output_dir)
    log = RejectionLog(saida / "rejections.jsonl")
    writer = FileSampleWriter(saida, split=split,
                              depth_layout=config.depth_layout)
    ledger = (saida / "generated_images.jsonl").open("a", encoding="utf-8")
    done = writer.completed_ids()
    if done:
        print(f"[rota-a] retomando: {len(done)} amostras já gravadas serão puladas.")

    try:
        for fonte in sources:
            if config.limit is not None and stats.written >= config.limit:
                break
            plano = draw_plan(fonte.scene_id, n=config.samples_per_image,
                              seed=config.seed)
            ids = [sample_id_for(fonte.scene_id, d.index, width=config.variant_id_width)
                   for d in plano]
            faltando = [(d, sid) for d, sid in zip(plano, ids) if sid not in done]
            stats.skipped_done += len(plano) - len(faltando)
            if not faltando:
                continue

            stats.images_processed += 1
            try:
                aif_bgr = load_aif(fonte)
                preparada = prepare_image(fonte, aif_bgr=aif_bgr,
                                          depth_runtime=depth_runtime, config=config)
            except SampleRejected as exc:
                # UMA linha no histograma para a imagem inteira. Ver o docstring.
                log.reject_from(fonte.scene_id, exc,
                                {"scene_id": fonte.scene_id, "route": ROUTE,
                                 "variants_lost": len(faltando),
                                 "stage": "image"})
                stats.images_lost += 1
                stats.variants_lost_to_image += len(faltando)
                continue

            for sorteio, sample_id in faltando:
                if config.limit is not None and stats.written >= config.limit:
                    break
                try:
                    amostra = process_variant(
                        preparada, sorteio, render_fn=render_fn, config=config,
                        provenance_base=provenance_base, vocabulary=vocabulario)
                    if amostra.sample_id != sample_id:
                        raise AssertionError(
                            f"id divergente: {amostra.sample_id!r} != {sample_id!r}")
                    meta = writer.write(amostra)
                except SampleRejected as exc:
                    log.reject_from(sample_id, exc,
                                    {"scene_id": fonte.scene_id, "route": ROUTE,
                                     "variant_index": sorteio.index, "stage": "variant"})
                    continue

                ledger.write(json.dumps(_ledger_line(saida, amostra, meta),
                                        ensure_ascii=False) + "\n")
                ledger.flush()
                log.accept(sample_id, {"scene_id": fonte.scene_id,
                                       "k_value": meta["k_value"]})
                stats.written += 1
                _acumula(stats, amostra, meta)
    finally:
        print(writer.stats.summary())
        print(log.summary())
        print(stats.summary())
        writer.close()
        log.close()
        ledger.close()
    return stats


def _acumula(stats: RouteAStats, sample: Sample, meta: dict) -> None:
    """Acumula a estatística a partir do metadado **GRAVADO**.

    Do metadado, e não do objeto em memória, pela mesma razão das outras duas rotas: o
    número que o resumo imprime tem que ser o que está no disco. O que não está no
    metadado plano vem da proveniência da amostra, que é o objeto que o writer serializou.
    """
    stats.k_values.append(float(meta["k_value"]))
    stats.focus_depth_m.append(float(meta["focus_depth_m"]))
    stats.per_source[meta["source_dataset"]] += 1
    stats.per_scene[meta["scene_id"]] += 1

    regra = (sample.provenance.extra or {}).get("mask_rule") or {}
    if regra.get("k_per_long_side") is not None:
        stats.k_per_long_side.append(float(regra["k_per_long_side"]))
    stats.k_clamped += int(bool(regra.get("k_clamped_to_positive_floor")))

    qualidade = meta.get("quality") or {}
    for chave, destino in (("rendered_coc_p99_px", stats.coc_p99_values),
                           ("defocus_saturation_ratio", stats.saturation_values),
                           ("mask_area_ratio_min", stats.band_area_values)):
        medida = qualidade.get(chave)
        if medida is not None:
            destino.append(float(medida["value"]))
