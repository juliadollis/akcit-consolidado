"""Rota (b) do paper — ITW dataset [19], foto com bokeh real, K pela Eq. 3.

    K ≈ f² · D_focus / ( 2 · F · (D_focus − f) ) × pixel_ratio          (Eq. 3)

Fluxo por amostra:

    bokeh real  ──DeblurNet──▶  I_aif  (gerada, gravada em disco)
                                  ├── Depth Pro [7] ─▶ z métrico em metros
                                  └── BiRefNet [86] ─▶ máscara M em foco
                                                          │
                          Eq. 4 (paper.txt:352) ──────────┴─▶ focus_disparity
                          Eq. 3 (paper.txt:340) ─▶ K  (f, F, focal_length_35 da EXIF)
                          gates ─▶ medem sempre, bloqueiam só com limiar congelado
                          writer ─▶ disparidade uint16 + AIF em JPEG + escalares

As decisões desta rota, com a linha do paper e o gatilho de revisão de cada uma, estão em
`reference/ROTA_B_DECISOES.md`. Os defeitos que ela existe para não repetir estão em
`reference/ROTA_B_AUDITORIA.md`.

**O ALVO é a própria fotografia com bokeh. Não há renderizador nesta rota.**
Fechado contra o paper: a tupla de supervisão é `(I_aif, I_out, D_def)`
(paper.txt:321), a Fig. 3(b) alimenta a fileira "Bokeh images" com a foto real
(paper.txt:271, 283), e o renderizador [43] aparece só em (a) — *"feed it into a bokeh
renderer [43] to synthesize corresponding bokeh images"*, paper.txt:292 — e no sweep da
Eq. 5 da rota (c) (paper.txt:369-370). Consequências, todas ativas neste módulo:

* `provenance.renderer is None` — e não um dicionário com `is_final_label_renderer:
  True`, que é o que o pipeline antigo gravava (`route_b.py:121`, defeito B14);
* não há `calibration_ssim` (não há Eq. 5), não há `is_k_censored` verdadeiro (não há
  varredura com teto), e não há `k_effective_factor` — o fator de 0,9873 medido no
  BokehMe (`CONTRATO.md:114`) descreve um renderizador que esta rota não usa, então
  aplicá-lo aqui seria corrigir por um viés inexistente (`[A]` A12, recomendação da
  auditoria: **não aplicar**, e declarar).

## Os nove defeitos que este módulo existe para não repetir

Cada um com a evidência em `reference/ROTA_B_AUDITORIA.md`.

| # | defeito antigo | como fica impossível aqui |
|---|---|---|
| B1 | `generate()` sem `main_adapter` → AIF LAVADA | `DeblurVariant` amarra a tripla; este módulo não vê `main_adapter` |
| B2 | `k_eq3` em px·mm contra disparidade em 1/m (1000×) | só `k_from_exif`, nunca `k_eq3_mm` direto |
| B3 | `pixel_ratio` com a LARGURA | `k_from_exif(image_hw=...)` usa `max(H,W)` — paper.txt:1186-1187 |
| B4 | `--max-coc` por rota | `MAX_COC` é constante; `RouteBConfig` não tem o campo |
| B5 | `D_focus` mediana da profundidade | `focus_disparity_from_mask` faz `median(1/z)` |
| B6 | sensor 36 mm assumido | `sensor_width_mm` rejeita com `sensor_width_unresolvable` |
| B7 | cascata BiRefNet→RMBG→GrabCut dizendo "automatic" | um segmentador; falha = rejeição com slug |
| B9 | `import json` ausente, dentro de um `finally` | está importado, e há smoke de 2 amostras |
| B16 | `s1`, um segundo plano de foco na mesma amostra | não existe, nem por compatibilidade |

## Decisão 1 — qual `D_focus` alimenta a Eq. 3

O contrato produz `focus_disparity = median(1/z[M])` (`contract.py:297`), por autoridade
do código oficial (`Inference_bokehNet.py:118`, `CONTRATO.md:28-34`). Mas `k_eq3_mm` pede
uma **distância**, e `1 / median(1/z) ≠ median(z)`: a mediana não comuta com a inversão
(`np.median` faz a média dos dois centrais em contagem par, e a média de dois recíprocos
não é o recíproco da média).

**Escolhido: `focus_depth_m = 1 / focus_disparity`.** Três razões, em ordem de peso:

1. **É o mesmo plano de foco que gera o mapa.** `defocus_map` usa `focus_disparity`. Se a
   Eq. 3 usasse `median(z[M])`, a amostra teria **dois** planos de foco discordantes — que
   é literalmente o defeito B16 (`s1` gravado divergindo do implícito: mediana 0,00308,
   p90 0,07927, máximo 0,48821 — *"são planos de foco diferentes"*, `ACHADOS.md:58`).
2. **Regra 5 do contrato**: uma implementação só. `focus_disparity_from_mask` é a única
   porta para a Eq. 4, e ela devolve disparidade.
3. A diferença é da ordem da medida do defeito B5 — pequena e sistemática em máscara
   grande.

O desvio da literalidade do paper (que escreve `D_focus = median(D[M])` sobre um mapa de
profundidade, paper.txt:352) fica **declarado e medido**: `median(z[M])` é calculado ao
lado, e cada amostra grava as duas grandezas mais a razão entre elas, em
`focus_depth_m_from_disparity`, `focus_depth_m_median_z` e `focus_depth_ratio`. Quem
quiser refazer o rótulo pela leitura literal tem o número por amostra, e o piloto mede a
distribuição da divergência em vez de a supor desprezível. Ver `_focus_depth_diagnostics`.

## Decisão 2 — a rota B NÃO usa o refinamento da região em foco

O paper aplica o passo de refinamento **só em (c)**, e diz por quê:

> *"(c) LFDOF and RealBokeh. [...] Similar to (b), we employ BiRefNet [86] to obtain an
> initial in-focus mask M. However, **due to the increased diversity and complexity of
> the scenes in these datasets**, the initial estimate of M is sometimes unreliable.
> Rather than simply verifying and discarding unreliable cases, we introduce a manual
> refinement step."* — paper.txt:361-365

Duas coisas nessa frase governam esta decisão. O *"Similar to (b)"* diz que (b) é o
caminho **sem** refinamento; e o *"in these datasets"* é uma afirmação **comparativa** —
o refinamento existe porque LFDOF e RealBokeh são mais difíceis **que o ITW**. Para (b) o
paper especifica exatamente três passos, e nenhum é refinamento: BiRefNet, profundidade
monocular, mediana (paper.txt:348-352). Então esta rota faz esses três, e a máscara que
não serve **rejeita a amostra** com `focus_mask_empty`.

Isso é coerente com o domínio: o ITW é feito de fotos do Flickr em que o fotógrafo focou
um sujeito, e o BiRefNet é um segmentador de objeto **saliente** — as duas coisas
coincidem justamente aí. Os 35,2% de acerto medidos em `reference/MEDICAO_PLANO_FOCO.md`
são da **RealBokeh**, que é feita de cenas (um tronco num parque, um muro de pedra), e
não transferem para cá por analogia. Não temos gabarito de distância de foco no ITW, mas
temos duas medidas indiretas que o piloto produz e que decidem se esta escolha se
sustenta:

* a taxa de `focus_mask_empty` — se for da ordem dos 20,6% da rota C, o BiRefNet está
  declinando aqui também e a decisão precisa ser revista;
* `focus_agreement` — quanto a máscara do BiRefNet concorda com a região de maior
  retenção, medida e gravada em toda amostra **sem** alterar o rótulo.

Gatilho declarado para revisitar: `focus_mask_empty` acima de 5% das amostras
processadas, ou `focus_agreement` mediano abaixo de 0,30 (o
`DEFAULT_AGREEMENT_FLOOR` de `qc.focus_region`). Os dois números saem do resumo do run.

## O que `detail_retention` significa aqui — e por que ela NÃO pode virar rótulo

Isto muda de fato entre as rotas, e é o ponto mais fácil de errar por analogia.

Na rota C a retenção é `média_local(|∇²bokeh|) / média_local(|∇²aif|)` com **duas
fotografias reais e independentes**: onde a razão é alta, a óptica preservou o detalhe
que a AIF tem, logo aquele pixel estava no plano de foco. É evidência física.

Na rota B a AIF é **produzida pela DeblurNet a partir da própria bokeh**. A razão deixa
de medir a óptica e passa a medir **onde a DeblurNet acrescentou alta frequência**. Três
consequências:

1. **Não é evidência independente.** Usá-la para definir `D_focus` faria o rótulo depender
   da DeblurNet duas vezes — uma na AIF que alimenta o Depth Pro, outra no plano de foco —
   e o rótulo inteiro colapsaria na crença de um único modelo.
2. **Ela esconde exatamente o defeito de maior impacto.** Se o LoRA não carregar, a AIF
   sai ≈ igual à bokeh, a retenção fica ≈ 1 em todo lugar, e `sharpest_region_mask`
   devolve o topo de 5% de um empate numérico. O refinamento produziria uma região
   plausível, com `focus_source = "retention_only"` e aparência normal, para uma amostra
   **completamente quebrada**. Aplicar o refinamento aqui seria construir um mecanismo
   que mascara o defeito B1.
3. **Por causa de (2), ela é um bom diagnóstico.** É o segundo eixo que a auditoria pede
   em §4.3: uma razão de nitidez sozinha não separa *"não deblurou"* de *"lavou"*. A
   **dispersão** da retenção separa: concentrada perto de 1 = não fez nada; espalhada,
   baixa no fundo e alta no sujeito = deblurou; baixa em todo lugar = a AIF não tem
   relação estrutural com a entrada. Gravado por amostra em `deblur_retention_p05/p50/p95`
   e em `deblur_retention_spread`.

Limitação registrada: `detail_retention` satura em 1,0 (`focus_region.py:210`, *"acima de
1 é ruído ou desalinhamento"*). Na rota C isso é correto. Aqui, bokeh **mais** detalhada
que a AIF num pixel significa que a DeblurNet **removeu** detalhe ali — informação real
que o `clip` descarta. Por isso a dispersão é lida junto com
`deblur_structural_fidelity`, que não satura.

## O que esta rota grava, e o que referencia

A AIF é **gerada**, então é gravada (`generated/<id>_aif.jpg`, `channel_order` declarado).
A bokeh é **real** e fica como referência para `atfortes/BokehDiffusion` — regravá-la
duplicaria bytes que já existem (`dataio/sample.py:7-11`). O mapa de defocus não é
gravado: é derivado no dataloader pela mesma função da geração, e gravá-lo criaria uma
segunda fonte de verdade (defeito D1).
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
    SampleRejected, focus_disparity_from_mask, k_from_exif, reject,
)
from dataio import (
    ControlLabel, DEPTH_LONG_SIDE, DepthLayout, FocusRegionRecord, KSource,
    MaskSource, Sample, SampleProvenance, SampleRefs, SceneSplit, FileSampleWriter,
    encode_depth,
)
from qc.focus_region import (
    DEFAULT_TOP_FRACTION, DEFAULT_WINDOW_PX, MIN_REGION_AREA_RATIO, FocusSource,
    detail_maps, sharpest_region_mask,
)
from qc.gates import (
    GateReport, GateResult, aif_sharpness, bokeh_is_blurrier_than_aif,
    depth_useful_levels, focus_depth_plausible, focus_mask_is_sharpest,
    mask_area_ratio, mask_border_coverage, mask_iou, pair_shape_matches,
)
from qc.metrics import ssim, to_gray
from qc.rejection import RejectionLog

#: Rótulo da rota no `Sample.route` e nas linhas do manifesto.
ROUTE = "b"

#: Nome do arquivo da AIF em `generated/`. `<sample_id>_aif.jpg`.
AIF_IMAGE_NAME = "aif"


# --------------------------------------------------------------------------------
# A fonte — protocolo, não import concreto
# --------------------------------------------------------------------------------

class BokehSource(Protocol):
    """O que a rota B precisa saber de uma foto, e nada mais.

    Protocolo, no molde do `PairSource` da rota C: o adaptador do ITW entrega esta forma
    e a rota não precisa conhecer o dataset. `scripts/run_route_b.py` traz o adaptador
    concreto de `atfortes/BokehDiffusion` — `[I]` A1: o paper cita só a publicação de
    [19] (paper.txt:791-793), sem URL, e a identificação se sustenta no autor (*Fortes,
    A.*, handle `atfortes`) e no volume (13.800 linhas passam o filtro, `ACHADOS.md:151`,
    contra os "13K" de paper.txt:1000).

    **Os três termos da Eq. 3 são obrigatórios e não têm default.** Ausência rejeita a
    amostra (`exif_focal_length_missing`, `exif_f_number_missing`,
    `sensor_width_unresolvable`) — regra 4 do `CONTRATO.md:52-53`. É decisão nossa
    declarada, `[A]` A8: o paper cala sobre EXIF ausente.
    """

    scene_id: str
    sample_id: str
    source_dataset: str
    source_sample_id: str
    source_split: Optional[str]
    #: Onde a foto com bokeh vive. É REFERÊNCIA: a rota não a regrava.
    bokeh_ref: Optional[str]

    #: Eq. 3, da EXIF. `f` e `F` diretamente (paper.txt:342-343); `focal_length_35mm`
    #: é o que resolve a largura física do sensor via crop factor (`[A]` A7 — o paper
    #: não descreve como obteve a largura).
    focal_length_mm: Optional[float]
    f_number: Optional[float]
    focal_length_35mm: Optional[float]

    #: Distância de foco da EXIF, quando a câmera publica. **NUNCA rótulo**, por decisão
    #: EXPLÍCITA do paper: *"it is frequently missing or noisy; therefore, we do not rely
    #: on EXIF for D_focus"* (paper.txt:344-346). Serve de VALIDADOR independente — ver
    #: `_validator_k`.
    exif_focus_distance_m: Optional[float]

    #: Para auditar os 30,33% de crop factor exatamente 1,0 (`ACHADOS.md:165`). Marcam,
    #: não corrigem: não há lacuna a preencher (`ACHADOS.md:169-173`).
    camera_make: Optional[str]
    camera_model: Optional[str]


#: Carrega a foto com bokeh, em BGR uint8. Fica fora da rota de propósito: ler de um
#: parquet, de disco ou de um dublê de teste é decisão de quem chama.
LoadBokeh = Callable[[BokehSource], np.ndarray]


# --------------------------------------------------------------------------------
# Config — todo limiar `None`
# --------------------------------------------------------------------------------

@dataclass
class RouteBConfig:
    """Todos os botões. **Todo limiar é `None` por default** — mede no piloto, congela
    depois.

    O que deliberadamente NÃO está aqui:

    * **`max_coc`.** É `MAX_COC = 100.0`, global e congelado. Três `--max-coc`
      independentes (um por rota, `route_a.py:207`, `route_b.py:350-355`,
      `route_c.py:466`) são o mecanismo exato do resultado do kfix: LF-Bokeh +0,8288 com
      RealDOF −0,4599 (`ACHADOS.md:192`), com `max_coc = 10,510746` que é um percentil da
      própria rota B (`ACHADOS.md:39,44`). `defocus_map` não aceita o parâmetro
      (`contract.py:454-472`) e esta config não expõe botão que encoste nisso.
    * **a variante da DeblurNet.** Ela vive no `DeblurNetRuntime`, amarrada aos pesos.
      Um campo aqui seria um segundo lugar de onde ela pode vir.
    * **`focus_retention_long_side`.** A rota C tem esse knob porque lá a retenção
      produz o rótulo. Aqui ela é diagnóstico, e um knob que muda a grade de um
      diagnóstico só acrescenta um eixo de incomparabilidade entre metades do release.
      A retenção é medida na resolução da imagem, sempre, e a grade vai para o metadado.
    """

    output_dir: Path
    depth_long_side: int = DEPTH_LONG_SIDE
    #: **Por amostra, e é a resposta certa aqui.** A dedup de profundidade por cena
    #: economiza onde vários `sample_id` compartilham uma AIF. Nesta rota eles não
    #: compartilham: `scene_id` é o `flickr_photo_id` e `sample_id` é `b_<scene_id>`
    #: (`scripts/run_route_b.py:164-165`), uma função injetiva, e `run_route_b.py:289`
    #: ainda rejeita `flickr_photo_id` repetido com `source_duplicate_sample`. Medido
    #: sobre as 13.800 linhas que passam o filtro: **13.800 cenas, 1,00 amostra por
    #: cena, 0,0% de economia**. Ligar a dedup aqui renomearia 13.800 arquivos de
    #: `depth/b_<id>.png` para `depth/<id>.png` sem poupar um byte — mudança de
    #: contrato em troca de nada.
    depth_layout: DepthLayout = DepthLayout.PER_SAMPLE

    # --- limiares, todos [A] até o piloto ---
    min_mask_area_ratio: Optional[float] = None
    max_mask_area_ratio: Optional[float] = None
    max_mask_border_coverage: Optional[float] = None
    min_mask_iou: Optional[float] = None
    min_focus_mask_sharpness_ratio: Optional[float] = None
    min_aif_laplacian_variance: Optional[float] = None
    min_depth_useful_levels: Optional[int] = None

    #: Razão de nitidez bokeh/AIF. Perto de 1 = a DeblurNet não fez nada. É o gate B6 do
    #: `REGISTRO.md:187`, que no pipeline antigo tinha default 0,0 e o help dizia
    #: "0 logs only" (`route_b.py:405-410`) — desligado exatamente onde importava.
    max_bokeh_over_aif_sharpness: Optional[float] = None
    #: SSIM(AIF, bokeh) — o SEGUNDO eixo. Piso pega a AIF LAVADA (estrutura perdida);
    #: teto pega a AIF IDÊNTICA à entrada (LoRA não carregou). Uma razão de nitidez
    #: sozinha não separa os dois: artefato de difusão tem variância de Laplaciano ALTA.
    min_deblur_structural_ssim: Optional[float] = None
    max_deblur_structural_ssim: Optional[float] = None

    #: Deslocamento AIF↔bokeh por correlação de fase, em px da resolução da imagem.
    #: O pipeline antigo tinha `--max-pair-shift-px 6.0` (`route_b.py:411-422`); o
    #: código novo só compara shapes (`gates.py:321-325`). `[A]` A13: quanto a DeblurNet
    #: desloca de fato **não foi medido** — é o piloto que mede.
    max_pair_registration_shift_px: Optional[float] = None
    min_pair_registration_response: Optional[float] = None

    #: Faixa plausível de K. Âncoras: 16,6 (kfix ÷ 1000, `ACHADOS.md:229`), 20,1 (EXIF,
    #: `CONTRATO.md:77`), 15,0 (default oficial, `Inference_bokehNet.py:53`), e a Fig. 12
    #: com `K ∈ {0,5,10,15}` (paper.txt:1151-1156). `[A]` A10: o paper não publica
    #: `K_min`/`K_max` para a Eq. 3. Lembrar do `--k-max 300` da rota C, que virou 47,0%
    #: de amostras no teto exato (`ACHADOS.md:19`) — teto errado não aparece como erro.
    min_k_value: Optional[float] = None
    max_k_value: Optional[float] = None

    #: Janela do filtro de média da retenção, EM PIXEL da resolução da imagem. Vai para o
    #: metadado junto com a grade: quantidade em pixel sem a resolução ao lado não diz
    #: nada (a 1500x2000 uma janela de 33 px é 1,7% do lado longo; a 512x683, 6,4%).
    retention_window_px: int = DEFAULT_WINDOW_PX
    #: Fração do quadro que forma a região de maior retenção usada para medir
    #: `focus_agreement`. Diagnóstico; não entra no rótulo.
    retention_top_fraction: float = DEFAULT_TOP_FRACTION

    limit: Optional[int] = None
    seed: int = 0


# --------------------------------------------------------------------------------
# Estatística do run
# --------------------------------------------------------------------------------

@dataclass
class RouteBStats:
    processed: int = 0
    written: int = 0
    skipped_done: int = 0
    k_values: list[float] = field(default_factory=list)
    #: K pela Eq. 3 com a distância de foco da EXIF — validador INDEPENDENTE do rótulo,
    #: acumulado separado. Ver `_validator_k`.
    k_validator_values: list[float] = field(default_factory=list)
    focus_depth_ratios: list[float] = field(default_factory=list)
    focus_agreements: list[float] = field(default_factory=list)
    deblur_ssim_values: list[float] = field(default_factory=list)
    deblur_spreads: list[float] = field(default_factory=list)
    shift_values: list[float] = field(default_factory=list)
    #: Amostras aceitas com crop factor exatamente 1,0 — os 30,33% do `ACHADOS.md:165`.
    #: Marcadas, nunca rejeitadas.
    crop_factor_unity: int = 0
    crop_factor_unity_cameras: Counter = field(default_factory=Counter)

    def _percentis(self, valores: list[float], rotulo: str, fmt: str = "{:.2f}") -> list[str]:
        finitos = np.asarray([v for v in valores if v is not None and np.isfinite(v)])
        if not finitos.size:
            return [f"  {rotulo:<26} ausente em todas as amostras aceitas"]
        p05, p50, p95 = np.percentile(finitos, [5, 50, 95])
        return [f"  {rotulo:<26} p05 {fmt.format(p05)}  mediana {fmt.format(p50)}  "
                f"p95 {fmt.format(p95)}   (n={finitos.size})"]

    def deblur_summary(self) -> list[str]:
        """O bloco que diz se a DeblurNet trabalhou — e o que ela fez.

        Separado para poder ser testado sem montar um run inteiro, e porque é o
        instrumento que denuncia o defeito B1 em voz alta: `deblur_ssim` mediano perto
        de 1,0 com dispersão de retenção perto de 0 significa **AIF ≈ entrada**, e aí o
        rótulo inteiro é sobre uma imagem que ninguém deblurou.
        """
        linhas = ["-" * 62, "  DeblurNet — a AIF é PRODUTO de modelo, não fotografia:"]
        linhas += self._percentis(self.deblur_ssim_values, "SSIM(AIF, bokeh)", "{:.3f}")
        linhas += self._percentis(self.deblur_spreads, "dispersão da retenção", "{:.3f}")
        linhas += self._percentis(self.shift_values, "deslocamento AIF↔bokeh px", "{:.2f}")
        linhas.append("    SSIM→1 com dispersão→0 = a DeblurNet NÃO fez nada (defeito B1);")
        linhas.append("    SSIM muito baixo = AIF sem relação estrutural com a entrada.")
        linhas.append("    Deslocamento é o [A] A13, medido aqui pela primeira vez.")
        return linhas

    def exif_summary(self) -> list[str]:
        """Os 30,33% de crop factor exatamente 1,0 (`ACHADOS.md:160-173`).

        Marcação, não filtro: *"a tabela `make/model → sensor_width_mm` serve para
        auditar esses 30%, não para preencher lacuna — não há lacuna"*. O número aparece
        aqui para que a auditoria por câmera seja possível, não para justificar descarte.
        """
        if not self.written:
            return []
        share = 100.0 * self.crop_factor_unity / self.written
        linhas = ["-" * 62,
                  f"  crop factor == 1,0 exato : {self.crop_factor_unity} "
                  f"({share:.1f}%)  medido antes: 30,33%",
                  "    full-frame de verdade, ou câmera que ecoa a focal no campo de "
                  "35 mm.",
                  "    MARCADO, nunca rejeitado — errar aqui erra K na mesma proporção."]
        for camera, n in self.crop_factor_unity_cameras.most_common(5):
            linhas.append(f"      {camera:<40} {n:>6}")
        return linhas

    def summary(self) -> str:
        if not self.k_values:
            return "[rota-b] nenhuma amostra aceita."
        k = np.asarray(self.k_values)
        linhas = [
            "",
            "=" * 62,
            f"  aceitas          : {self.written}",
            f"  já feitas (skip) : {self.skipped_done}",
            "-" * 62,
        ]
        linhas += self._percentis(self.k_values, "k_value (Eq. 3)")
        linhas.append(f"    âncoras: 16,6 (kfix÷1000) · 20,1 (EXIF) · 15,0 (oficial) · "
                      f"Fig.12 {{0,5,10,15}}")
        linhas.append(f"    mediana medida agora: {np.median(k):.2f}")
        linhas += self._percentis(self.k_validator_values, "k pela dist. da EXIF")
        linhas.append("    VALIDADOR independente (não usa Depth Pro nem BiRefNet).")
        linhas.append("    O paper proíbe a EXIF como rótulo — paper.txt:344-346.")
        linhas += self._percentis(self.focus_depth_ratios,
                                  "1/med(1/z) ÷ med(z)", "{:.4f}")
        linhas.append("    a Decisão 1 em números: 1,0 = as duas leituras coincidem.")
        linhas += self._percentis(self.focus_agreements, "focus_agreement", "{:.3f}")
        linhas.append("    concordância BiRefNet × retenção. SEM refinamento nesta rota;")
        linhas.append(f"    revisitar a Decisão 2 se a mediana ficar abaixo de "
                      f"{0.30:.2f}.")
        linhas += self.deblur_summary()
        linhas += self.exif_summary()
        linhas.append("=" * 62)
        return "\n".join(linhas)


# --------------------------------------------------------------------------------
# Gates específicos da rota B
# --------------------------------------------------------------------------------
# Vivem aqui, e não em `qc/gates.py`, por duas razões: `qc/gates.py` está sob edição de
# outro agente, e estes três só fazem sentido quando a AIF é PRODUTO de modelo. Se
# passarem a valer para outra rota, migram para `qc/gates.py` — que é o lugar certo — e
# esta seção some. Ver o relatório desta tarefa, item "o que mudar em módulos alheios".

def deblur_structural_fidelity(
    bokeh_bgr: np.ndarray, aif_bgr: np.ndarray, *,
    min_ssim: Optional[float] = None, max_ssim: Optional[float] = None,
) -> list[GateResult]:
    """SSIM(AIF, bokeh) — o SEGUNDO eixo do gate de qualidade da AIF.

    O primeiro eixo é `bokeh_is_blurrier_than_aif`, que mede razão de nitidez. Ele não
    basta, e a auditoria diz por quê (§4.3, item 1): *"o caso 'saída LAVADA' pode ter
    variância de Laplaciano ALTA por artefato e ainda assim ser lixo. Uma razão de
    nitidez sozinha não separa os dois."*

    SSIM contra a **entrada** separa, porque ela mede estrutura e não energia:

    | caso | razão de nitidez | SSIM(AIF, bokeh) |
    |---|---|---|
    | deblur bem-feito | baixa | intermediário — nítido, mesma cena |
    | LoRA não carregou (saída ≈ entrada) | **≈ 1** | **≈ 1** |
    | AIF lavada / alucinada | baixa (ou alta por artefato) | **baixo** |

    Daí os dois limiares. `max_ssim` é o teto que pega a identidade, e ele existe
    separado de `max_bokeh_over_aif_sharpness` de propósito: uma difusão pode devolver
    algo com a mesma energia de alta frequência da entrada sem ser a entrada.

    Os dois `None` por default. O valor não vem do paper — que não publica nada sobre
    controle de qualidade da AIF da rota B —, vem do histograma do piloto.
    """
    valor = ssim(aif_bgr, bokeh_bgr)
    return [
        GateResult("deblur_structural_ssim_min", valor, min_ssim, True,
                   "SSIM(AIF, bokeh): piso pega a AIF LAVADA — estrutura da entrada "
                   "perdida. Alta variância de Laplaciano NÃO desmente isto"),
        GateResult("deblur_structural_ssim_max", valor, max_ssim, False,
                   "SSIM(AIF, bokeh): teto pega a AIF IDÊNTICA à entrada, que é o LoRA "
                   "não tendo carregado (defeito B1 pelo outro lado)"),
    ]


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    """Janela de Hann 2D. Sem ela a correlação de fase mede a moldura da imagem.

    A descontinuidade de borda de um recorte retangular é energia de alta frequência
    perfeitamente alinhada nas duas imagens, e ela produz um pico em (0,0) que não vem do
    conteúdo — um gate que sempre reporta deslocamento zero é pior que gate nenhum.
    """
    h, w = int(shape[0]), int(shape[1])
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.outer(wy, wx).astype(np.float32)


def phase_correlation_shift(
    reference_bgr: np.ndarray, moved_bgr: np.ndarray,
) -> tuple[float, float, float]:
    """`(dy, dx, resposta)` — deslocamento global por correlação de fase.

    A resposta é o pico da superfície de correlação normalizada dividido pela média
    dela: com duas imagens idênticas dá um número grande; com duas imagens sem relação,
    ≈ 1. Ela é obrigatória ao lado do deslocamento, porque um `argmax` sobre ruído
    devolve uma coordenada qualquer com toda a cara de medida.

    Deslocamento inteiro, sem sub-pixel: o gate do pipeline antigo era 6,0 px
    (`route_b.py:411-422`) e o recorte medido do caminho `long_side > 0` chega a 30,86 px
    (`model_runtime/deblurnet.py`, tabela do docstring). Meio pixel não muda nenhuma
    dessas decisões, e interpolação de sub-pixel esconderia o caso em que a superfície
    de correlação não tem pico nenhum.
    """
    a = np.asarray(reference_bgr)
    b = np.asarray(moved_bgr)
    if a.shape[:2] != b.shape[:2]:
        reject("resolution_invalid",
               f"correlação de fase entre {a.shape[:2]} e {b.shape[:2]}")
    janela = _hann2d(a.shape[:2])
    cinza_a, cinza_b = to_gray(a), to_gray(b)
    # Remover a média antes da janela: uma componente DC diferente entre as duas imagens
    # aparece como energia em frequência zero e domina a normalização de fase.
    fa = np.fft.rfft2((cinza_a - float(np.mean(cinza_a))) * janela)
    fb = np.fft.rfft2((cinza_b - float(np.mean(cinza_b))) * janela)
    produto = fa * np.conj(fb)
    magnitude = np.abs(produto)
    # Onde a magnitude é zero não há fase a comparar; dividir daria NaN e o `argmax`
    # devolveria a primeira posição — deslocamento (0,0) por acidente aritmético.
    normalizado = np.divide(produto, magnitude, out=np.zeros_like(produto),
                            where=magnitude > 0)
    superficie = np.fft.irfft2(normalizado, s=a.shape[:2])

    pico = int(np.argmax(superficie))
    dy, dx = np.unravel_index(pico, superficie.shape)
    h, w = superficie.shape
    dy = dy - h if dy > h // 2 else dy          # a FFT enrola: metade superior é negativa
    dx = dx - w if dx > w // 2 else dx
    media = float(np.mean(np.abs(superficie)))
    resposta = float(superficie[np.unravel_index(pico, superficie.shape)] / media) \
        if media > 0 else 0.0
    return float(dy), float(dx), resposta


def pair_registration(
    bokeh_bgr: np.ndarray, aif_bgr: np.ndarray, *,
    max_shift_px: Optional[float] = None,
    min_response: Optional[float] = None,
) -> list[GateResult]:
    """A AIF está registrada com a bokeh? O defeito B17, e o `[A]` A13.

    A geometria já é garantida por construção quando `ResizePolicy` é
    `NO_CROP_MULTIPLE_OF_16` com `long_side = 0` — o round-trip é identidade exata,
    `max_registration_shift_px == 0,00` por construção. Mas a DeblurNet é um modelo de
    **difusão**: geometria idêntica não impede que o conteúdo saia deslocado ou
    deformado. `ProcessingPlan` mede o que o redimensionamento faz; só a correlação de
    fase mede o que o modelo faz.

    Se houver deslocamento, `D`, `M` e o alvo vivem em geometrias diferentes — e o alvo
    da rota B é a própria bokeh, então o erro entra direto na supervisão.
    """
    dy, dx, resposta = phase_correlation_shift(bokeh_bgr, aif_bgr)
    deslocamento = float(np.hypot(dy, dx))
    return [
        GateResult("pair_registration_shift_px", deslocamento, max_shift_px, False,
                   f"deslocamento AIF↔bokeh (dy={dy:+.0f}, dx={dx:+.0f}) em px da "
                   "resolução da imagem; o gate antigo era 6,0 px"),
        GateResult("pair_registration_response", resposta, min_response, True,
                   "pico da correlação sobre a média: baixo significa que o "
                   "deslocamento medido NÃO é confiável, não que ele é zero"),
    ]


def exif_crop_factor_suspect(
    crop_factor: float, *, make: Optional[str] = None, model: Optional[str] = None,
) -> GateResult:
    """Marca crop factor exatamente 1,0. **Nunca bloqueia — por construção.**

    Medido: 4.185 de 13.800 amostras, **30,33%**, com crop factor exatamente 1,0
    (`ACHADOS.md:165`). *"Trinta por cento de full-frame num dataset do Flickr é alto"* —
    é ou full-frame de verdade, ou a assinatura de uma câmera que ecoa a focal no campo
    de 35 mm quando não sabe o valor.

    Não recebe limiar, e isso é deliberado: a auditoria é explícita em que a tabela
    `make/model` serve para **auditar** esses 30%, não para preencher lacuna — não há
    lacuna, `focal_length_35` está presente em 13.800/13.800 (`ACHADOS.md:155-157`).
    Um parâmetro de limiar aqui permitiria que alguém, num run futuro, transformasse a
    marcação em descarte de 30% do dataset por um valor que ninguém mediu.
    """
    suspeito = abs(float(crop_factor) - 1.0) < 1e-9
    camera = " ".join(p for p in (make, model) if p) or "câmera não publicada"
    return GateResult("exif_crop_factor_unity", 1.0 if suspeito else 0.0, None, False,
                      f"crop factor {'EXATAMENTE 1,0' if suspeito else 'medido'} — "
                      f"{camera}. MARCAÇÃO, nunca rejeição: errar o sensor erra K na "
                      "mesma proporção (5,6x no pior caso medido)")


def k_in_range(k_value: float, *, min_k: Optional[float] = None,
               max_k: Optional[float] = None) -> list[GateResult]:
    """Faixa plausível de K para a Eq. 3. `[A]` A10 — o paper não publica os limites.

    O pipeline antigo tinha `--min-physical-k 1e-6` e `--max-physical-k None`
    (`route_b.py:356-363`), isto é, nenhum limite útil. E o lado oposto tem uma lição
    medida: o `--k-max 300` da rota C produziu **47,0%** de amostras censuradas no teto
    exato (`ACHADOS.md:19`) sem que nada denunciasse.

    Por isso os dois defaults são `None` e a rota **rejeita** em vez de censurar: não há
    varredura aqui, então não existe "K no teto" — existe K implausível, que é uma
    amostra a descartar com slug e a contar no histograma.
    """
    return [
        GateResult("k_value_min", float(k_value), min_k, True,
                   "K abaixo do plausível: âncoras 15,0 / 16,6 / 20,1"),
        GateResult("k_value_max", float(k_value), max_k, False,
                   "K acima do plausível. NÃO censura no teto — rejeita, porque não há "
                   "varredura da qual o teto seria uma borda"),
    ]


# --------------------------------------------------------------------------------
# O relatório de gates
# --------------------------------------------------------------------------------

def build_gate_report(
    *, aif_bgr: np.ndarray, bokeh_bgr: np.ndarray, mask: np.ndarray,
    mask_bokeh: Optional[np.ndarray], disparity_u16: np.ndarray,
    focus_disparity: float, k_value: float, k_diagnostics: dict,
    source: BokehSource, config: RouteBConfig,
) -> GateReport:
    """Todos os gates da rota B, na ordem em que valem a pena olhar no histograma.

    Dois gates da rota C **não entram**, e a ausência é decisão, não esquecimento:

    * `aif_aperture_is_narrow` — a AIF da rota B é **gerada**, não fotografada. Ela não
      tem f-stop, e a nota do gate (*"a origem correta é `train/in/<id>_f22.JPG`"*)
      descreve a RealBokeh. Registrar `applicable=False` com aquela nota poria uma frase
      falsa no metadado de 13 mil amostras.
    * `calibration_ssim_is_reliable` — não há Eq. 5 nesta rota (`CONTRATO.md:160`), logo
      não há SSIM de calibração a julgar.

    E `focus_mask_is_sharpest` é aqui **mais forte** que na rota C: ele exige a imagem
    com bokeh, e a rota B **tem** a bokeh real — é ela o alvo. É o único gate que testa a
    hipótese física de que a máscara marca a região em foco, e o único que pega o caso em
    que o fotógrafo focou o fundo e o BiRefNet marcou o sujeito em primeiro plano.
    """
    report = GateReport()
    report.add(pair_shape_matches(aif_bgr, bokeh_bgr))

    # --- a AIF é produto de modelo: os dois eixos, juntos --------------------
    report.add(bokeh_is_blurrier_than_aif(
        aif_bgr, bokeh_bgr, max_ratio=config.max_bokeh_over_aif_sharpness))
    for result in deblur_structural_fidelity(
            bokeh_bgr, aif_bgr, min_ssim=config.min_deblur_structural_ssim,
            max_ssim=config.max_deblur_structural_ssim):
        report.add(result)
    report.add(aif_sharpness(aif_bgr, min_variance=config.min_aif_laplacian_variance))
    for result in pair_registration(
            bokeh_bgr, aif_bgr,
            max_shift_px=config.max_pair_registration_shift_px,
            min_response=config.min_pair_registration_response):
        report.add(result)

    # --- a máscara, que define D_focus, que multiplica no K ------------------
    for result in mask_area_ratio(mask, min_ratio=config.min_mask_area_ratio,
                                  max_ratio=config.max_mask_area_ratio):
        report.add(result)
    report.add(mask_border_coverage(mask, max_ratio=config.max_mask_border_coverage))
    report.add(focus_mask_is_sharpest(
        bokeh_bgr, mask, min_ratio=config.min_focus_mask_sharpness_ratio))
    if mask_bokeh is not None:
        report.add(mask_iou(mask, mask_bokeh, min_iou=config.min_mask_iou))

    # --- profundidade, plano de foco e K ------------------------------------
    report.add(depth_useful_levels(disparity_u16,
                                   min_levels=config.min_depth_useful_levels))
    for result in focus_depth_plausible(focus_disparity):
        report.add(result)
    for result in k_in_range(k_value, min_k=config.min_k_value,
                             max_k=config.max_k_value):
        report.add(result)
    report.add(exif_crop_factor_suspect(
        k_diagnostics["crop_factor"],
        make=getattr(source, "camera_make", None),
        model=getattr(source, "camera_model", None)))
    return report


#: Nome de gate -> slug de rejeição do conjunto FECHADO de `control.contract`. Sem esta
#: ponte, o histograma receberia nomes de gate que não são slugs e viraria vocabulário
#: aberto — e é o histograma que denuncia fallback novo.
#:
#: **Três entradas são SLUGS PROVISÓRIOS**, marcadas abaixo. Os slugs próprios
#: (`gate_deblur_aif_unfaithful`, `gate_pair_registration_shift`) exigem acrescentar
#: valor a `GATE_REJECTION_REASONS` (`control/contract.py:94-111`), que é `frozenset`
#: fechado num módulo que esta tarefa não pode editar. Enquanto isso: o **nome do gate**
#: preserva a distinção no `quality` de cada amostra, e o histograma agrega sob o slug
#: existente mais próximo. Ver o relatório.
GATE_TO_REASON = {
    "pair_shape_matches": "gate_pair_shape_mismatch",
    "bokeh_over_aif_sharpness": "gate_bokeh_not_blurrier",
    # PROVISÓRIO: pede `gate_deblur_aif_unfaithful`. Fica em `gate_aif_sharpness` porque
    # é uma reprovação de QUALIDADE DA AIF — o mesmo objeto que aquele slug julga.
    "deblur_structural_ssim_min": "gate_aif_sharpness",
    # PROVISÓRIO: a saída ≈ entrada é, literalmente, "a bokeh não é mais borrada que a
    # AIF" — o mesmo fato que `gate_bokeh_not_blurrier` nomeia, medido por outro eixo.
    "deblur_structural_ssim_max": "gate_bokeh_not_blurrier",
    "aif_laplacian_variance": "gate_aif_sharpness",
    # PROVISÓRIO: pede `gate_pair_registration_shift`. `gate_pair_shape_mismatch` é o
    # slug de "o par não está alinhado"; deslocamento é a generalização de shape errado.
    "pair_registration_shift_px": "gate_pair_shape_mismatch",
    "pair_registration_response": "gate_pair_shape_mismatch",
    "mask_area_ratio_min": "gate_mask_area_ratio",
    "mask_area_ratio_max": "gate_mask_area_ratio",
    "mask_border_coverage": "gate_mask_border_coverage",
    "focus_mask_sharpness_ratio": "gate_focus_mask_not_sharpest",
    "mask_iou_aif_bokeh": "gate_mask_iou",
    "depth_useful_levels": "gate_depth_useful_levels",
    "focus_depth_m_min": "gate_focus_depth_implausible",
    "focus_depth_m_max": "gate_focus_depth_implausible",
    # `k_out_of_configured_range` é slug REGISTRADO (`contract.py:153`) e é exatamente
    # isto: K fora da faixa que a config declarou.
    "k_value_min": "k_out_of_configured_range",
    "k_value_max": "k_out_of_configured_range",
    # `exif_crop_factor_unity` NÃO aparece aqui de propósito: ele não tem limiar, então
    # nunca reprova. Se um dia reprovasse, o `KeyError` desta tabela seria o alarme
    # correto — marcação virando descarte de 30% do dataset tem que doer.
}


def enforce_gates(report: GateReport) -> None:
    """Primeiro gate reprovado vira `SampleRejected` com slug do conjunto fechado."""
    blocked = report.blocked_by
    if blocked:
        reject(GATE_TO_REASON[blocked[0]], f"gates reprovados: {blocked}")


# --------------------------------------------------------------------------------
# Plano de foco — Eq. 4, sem refinamento, com o desvio medido
# --------------------------------------------------------------------------------

def _focus_depth_diagnostics(depth_m: np.ndarray, mask: np.ndarray,
                             focus_disparity: float) -> dict:
    """As DUAS leituras de `D_focus`, lado a lado. A Decisão 1, auditável por amostra.

    `1 / median(1/z[M])` é o que alimenta a Eq. 3 e o que gera o mapa. `median(z[M])` é a
    leitura literal do paper (paper.txt:352). Gravar as duas e a razão entre elas é o que
    transforma "a diferença é pequena" de suposição em medida: o piloto imprime a
    distribuição de `focus_depth_ratio`, e quem quiser refazer o rótulo pela leitura
    literal tem o número sem reprocessar imagem.

    `median(z[M])` **nunca** entra em conta nenhuma. É por isso que ele mora num
    diagnóstico e não num campo do `ControlLabel`: um segundo plano de foco num campo de
    rótulo é o defeito B16 (`ACHADOS.md:58`: divergência p90 de 0,079 e máxima de 0,488
    entre dois campos que descreviam a mesma grandeza).
    """
    valores = np.asarray(depth_m, dtype=np.float32)[np.asarray(mask) > 0.5]
    valores = valores[np.isfinite(valores) & (valores > 0)]
    from_disparity = 1.0 / float(focus_disparity)
    median_z = float(np.median(valores)) if valores.size else float("nan")
    return {
        "focus_depth_m_from_disparity": from_disparity,
        "focus_depth_m_median_z": median_z,
        "focus_depth_ratio": (from_disparity / median_z
                              if median_z and np.isfinite(median_z) else float("nan")),
        "focus_depth_convention": "1/median(1/z[M])",
        "focus_depth_convention_note": (
            "Decisão 1 da rota B: a Eq. 3 recebe 1/focus_disparity, não median(z[M]). "
            "Desvio declarado da literalidade de paper.txt:352; o mesmo plano que gera "
            "o mapa. Ver o docstring de routes/route_b.py."),
    }


def focus_region_record(
    *, aif_bgr: np.ndarray, bokeh_bgr: np.ndarray, mask: np.ndarray,
    config: RouteBConfig,
) -> tuple[FocusRegionRecord, dict]:
    """A marcação da região em foco da rota B: **BiRefNet, sem refinamento**.

    Devolve `(registro, diagnóstico da DeblurNet)`. A região é a máscara do BiRefNet como
    ela veio — `FocusSource.BIREFNET`, `focus_was_refined = False` — porque o paper aplica
    o refinamento só em (c) (paper.txt:361-365; ver a Decisão 2 no docstring do módulo).

    A retenção é medida **de qualquer forma**, e nunca vira rótulo. Ela serve a duas
    coisas:

    1. `focus_agreement` — quanto o BiRefNet concorda com a região de maior retenção.
       É o número que decide, no piloto, se a Decisão 2 se sustenta.
    2. o diagnóstico da DeblurNet — a **dispersão** da retenção é o eixo que separa
       "não deblurou" de "deblurou" de "lavou", e é a informação que o gate de SSIM
       complementa. Ver a seção sobre `detail_retention` no docstring do módulo.

    Atenção ao que `agreement` mede aqui, que **não** é o que mede na rota C: lá os dois
    lados são fotografias reais, e a concordância é entre saliência e física. Aqui um dos
    lados saiu da DeblurNet, então é concordância entre saliência e **a crença da
    DeblurNet sobre onde havia borrão**. Continua sendo informativo — as duas discordam
    quando alguma das duas está errada —, mas não é evidência independente.
    """
    binaria = np.asarray(mask) > 0.5
    if not binaria.any():
        # Um segmentador só. Ele falhou: a amostra é rejeitada com slug, e não há cascata
        # para RMBG nem GrabCut (defeito B7, `genrefocus.py:420-428`). O histograma conta
        # quantas — é ele que diz se a Decisão 2 precisa de revisão.
        reject("focus_mask_empty",
               "BiRefNet devolveu máscara vazia. A rota B segue paper.txt:348-352 (mask "
               "-> mediana) e NÃO aplica o refinamento do §3.2(c), que o paper reserva a "
               "(c). Ver a Decisão 2 em routes/route_b.py")

    retencao, detalhe_aif = detail_maps(aif_bgr, bokeh_bgr,
                                        window=config.retention_window_px)
    validos = np.isfinite(retencao)
    # `detalhe_aif` é obrigatório desde o conserto de `sharpest_region_mask`: sem ele a
    # seleção escorregava para superfícies lisas, onde a razão vale ~1 por ser ruído
    # sobre ruído. Vale aqui igual, ainda que esta região seja só diagnóstico.
    regiao = sharpest_region_mask(retencao, detalhe_aif,
                                  top_fraction=config.retention_top_fraction,
                                  min_area_ratio=MIN_REGION_AREA_RATIO)

    # `agreement` é a fração da região de maior retenção que a máscara cobre — a mesma
    # definição de `qc.focus_region.refine_focus_mask` (:275), para os dois números serem
    # comparáveis entre rotas. `0.0` quando a cena não tem detalhe onde medir; não é
    # "discordam totalmente", é "não deu para comparar", e o `[0,1]` exigido por
    # `_validate_focus_region` não tem como expressar a diferença — daí a nota abaixo.
    agreement = (0.0 if regiao is None
                 else float((binaria & regiao).sum() / max(int(regiao.sum()), 1)))
    dentro = retencao[binaria & validos]
    retention_in_region = float(np.median(dentro)) if dentro.size else float("nan")

    amostra_valida = retencao[validos]
    if amostra_valida.size:
        p05, p50, p95 = (float(v) for v in np.percentile(amostra_valida, [5, 50, 95]))
    else:
        p05 = p50 = p95 = float("nan")

    registro = FocusRegionRecord(
        source=FocusSource.BIREFNET,
        agreement=agreement,
        # Na rota B a máscara crua **é** a região que define o foco — não há uma segunda
        # região a comparar com ela. Precisão e IoU contra si mesma são 1,0 por
        # definição, e é isso que estes campos afirmam: não houve refinamento, logo não
        # houve desacordo. Copiar `agreement` aqui é que seria mentira.
        precision=1.0,
        iou=1.0,
        retention_in_region=retention_in_region,
        area_ratio=float(binaria.mean()),
        retention_hw=(int(retencao.shape[0]), int(retencao.shape[1])),
        retention_window_px=int(config.retention_window_px),
        initial_mask_was_empty=False,
        initial_mask_area_ratio=float(binaria.mean()),
        # Não existe "linha de base sem refinamento" nesta rota: a máscara crua **é** o
        # rótulo. Gravar `focus_disparity` aqui faria o campo de diagnóstico pareado
        # repetir o rótulo e parecer uma comparação que não aconteceu.
        disparity_from_initial_mask=None,
    )
    diagnostico = {
        "deblur_retention_p05": p05,
        "deblur_retention_p50": p50,
        "deblur_retention_p95": p95,
        "deblur_retention_spread": p95 - p05 if np.isfinite(p95) and np.isfinite(p05)
        else float("nan"),
        "deblur_retention_measurable_fraction": float(validos.mean()),
        "retention_region_found": regiao is not None,
        "retention_semantics": (
            "AIF GERADA pela DeblurNet a partir da própria bokeh: esta razão mede onde a "
            "DeblurNet acrescentou alta frequência, NÃO onde a óptica preservou detalhe. "
            "Diagnóstico, nunca rótulo — ver routes/route_b.py."),
    }
    return registro, diagnostico


# --------------------------------------------------------------------------------
# K — Eq. 3, e o validador independente
# --------------------------------------------------------------------------------

def _validator_k(source: BokehSource, image_hw: tuple[int, int]) -> Optional[float]:
    """Eq. 3 com a distância de foco da **EXIF** — validador independente do rótulo.

    O paper proíbe a EXIF como fonte de `D_focus`, com razão declarada: *"Although some
    devices may provide a focus-distance field in EXIF, it is frequently missing or
    noisy; therefore, we do not rely on EXIF for D_focus"* (paper.txt:344-346). Este
    número **não é rótulo** e nunca entra em `k_value`.

    Ele é o validador certo por ser **independente**: não passa pelo Depth Pro nem pelo
    BiRefNet, que são os dois modelos que produzem o valor sendo validado. É a mesma
    lógica de `route_c._analytic_k`, com os papéis invertidos — lá o validador é a Eq. 3
    e o rótulo é a Eq. 5; aqui o rótulo é a Eq. 3 sobre um `D_focus` estimado e o
    validador é a Eq. 3 sobre um `D_focus` medido pela câmera.

    Âncora: a mediana de `k_value` calculada da EXIF em 318 amostras com distância de
    foco é **20,1** (`CONTRATO.md:77`), contra 16,6 da tabela kfix. Se este validador
    sair sistematicamente longe do rótulo no piloto, o suspeito é a escala do Depth Pro —
    e o número existe para poder acusar.

    Ausência não rejeita: o rótulo não depende disto, e o resumo do run diz em quantas
    amostras o validador ficou ausente.
    """
    distancia = getattr(source, "exif_focus_distance_m", None)
    if not distancia or not np.isfinite(distancia) or distancia <= 0:
        return None
    try:
        k, _ = k_from_exif(
            focal_length_mm=float(source.focal_length_mm),
            f_number=float(source.f_number),
            focal_length_35mm=source.focal_length_35mm,
            focus_depth_m=float(distancia),
            image_hw=image_hw,
        )
    except SampleRejected:
        return None
    return k


# --------------------------------------------------------------------------------
# Uma amostra
# --------------------------------------------------------------------------------

def process_sample(
    source: BokehSource,
    *,
    bokeh_bgr: np.ndarray,
    deblur_runtime,
    depth_runtime,
    mask_runtime,
    config: RouteBConfig,
    provenance_base: dict,
) -> Sample:
    """Uma foto com bokeh -> uma amostra. `SampleRejected` com slug em qualquer falha.

    A ordem é a do paper, e ela importa: *"Given real bokeh images, DeblurNet recovers an
    AIF image. We then estimate depth and extract a foreground mask [86]"*
    (paper.txt:293-294). A profundidade sai da **AIF**, não da bokeh: *"where D is the
    monocular depth map estimated from I_aif"* (paper.txt:314).

    De qual imagem sai a **máscara** o paper não afirma literalmente — o *"We then"* vem
    depois do passo da DeblurNet, o que sugere AIF. É o `[A]` A2, e ele é **médio**, não
    baixo: numa foto com bokeh, a região nítida *é* a região em foco, então a máscara da
    bokeh pode ser melhor que a da AIF. Escolha desta rota: a máscara do rótulo sai da
    **AIF**, seguindo a sugestão do texto e a simetria com a rota C; e a máscara da
    **bokeh** é calculada de qualquer forma, só para medir a divergência
    (`mask_iou_aif_bokeh`). No piloto, essa IoU é o número que diz se A2 merece ser
    revisitado — e a revisão custaria trocar duas linhas, não reescrever a rota.
    """
    bokeh_bgr = np.asarray(bokeh_bgr)
    if bokeh_bgr.ndim != 3 or bokeh_bgr.shape[2] != 3:
        reject("resolution_invalid", f"bokeh com shape {bokeh_bgr.shape}")
    image_hw = (int(bokeh_bgr.shape[0]), int(bokeh_bgr.shape[1]))
    bokeh_rgb = np.ascontiguousarray(bokeh_bgr[..., ::-1])

    # 1. A AIF. `main_adapter` vem da variante, dentro do runtime — nunca daqui.
    deblurred = deblur_runtime.infer(bokeh_rgb)
    aif_rgb = np.asarray(deblurred.aif_rgb)
    if aif_rgb.shape[:2] != image_hw:
        reject("resolution_invalid",
               f"AIF em {aif_rgb.shape[:2]} contra bokeh em {image_hw}: K vive na escala "
               "de pixel da imagem fonte e as duas têm que coincidir")
    aif_bgr = np.ascontiguousarray(aif_rgb[..., ::-1])

    # 2. Profundidade e máscara, as duas a partir da AIF.
    depth = depth_runtime.infer(aif_rgb)
    mask = mask_runtime.infer(aif_rgb)
    # A máscara da bokeh serve SÓ para medir divergência (o `[A]` A2). Ela não define
    # o foco, e é por isso que ela não entra em `focus_disparity_from_mask`.
    mask_bokeh = mask_runtime.infer(bokeh_rgb)

    # 3. Eq. 4, na disparidade, sobre a máscara do BiRefNet — sem refinamento.
    focus_record, deblur_diag = focus_region_record(
        aif_bgr=aif_bgr, bokeh_bgr=bokeh_bgr, mask=mask, config=config)
    focus_disparity = focus_disparity_from_mask(depth.values_m, mask)
    focus_diag = _focus_depth_diagnostics(depth.values_m, mask, focus_disparity)

    # 4. Eq. 3. `k_from_exif` compõe sensor -> pixel_ratio -> k_eq3_mm -> k_official num
    #    caminho só: chamar `k_eq3_mm` direto é o fator 1000x do defeito B2, e passar
    #    `shape[1]` em vez de `image_hw` é o defeito B3.
    k_value, k_diag = k_from_exif(
        focal_length_mm=source.focal_length_mm,
        f_number=source.f_number,
        focal_length_35mm=source.focal_length_35mm,
        focus_depth_m=focus_diag["focus_depth_m_from_disparity"],
        image_hw=image_hw,
    )

    encoded = encode_depth(depth.values_m, image_hw=image_hw,
                           long_side=config.depth_long_side)

    report = build_gate_report(
        aif_bgr=aif_bgr, bokeh_bgr=bokeh_bgr, mask=mask, mask_bokeh=mask_bokeh,
        disparity_u16=encoded.disparity_u16, focus_disparity=focus_disparity,
        k_value=k_value, k_diagnostics=k_diag, source=source, config=config)
    enforce_gates(report)

    return Sample(
        sample_id=source.sample_id, route=ROUTE,
        refs=SampleRefs(source.source_dataset, source.source_sample_id, source.scene_id,
                        # A AIF é PRODUTO desta rota, então `aif_ref` é `None`: apontar
                        # para a origem afirmaria que ela já existia lá.
                        aif_ref=None, bokeh_ref=source.bokeh_ref,
                        source_split=source.source_split),
        control=ControlLabel(
            k_value=k_value, k_source=KSource.EQ3_EXIF,
            focus_disparity=focus_disparity,
            # Não há varredura, logo não há teto do qual censurar. K implausível é
            # REJEIÇÃO com slug (`k_in_range`), não uma amostra marcada e mantida.
            is_k_censored=False,
            depth_backend=depth.backend,
            # Sem Eq. 5 nesta rota (`CONTRATO.md:160`).
            calibration_ssim=None,
            # `k_analytic` na rota C é a Eq. 3 validando o sweep. Aqui a Eq. 3 **é** o
            # rótulo: repetir o mesmo número neste campo o faria passar por validador
            # independente sem ser um. O validador de verdade é a distância de foco da
            # EXIF, e ele vai na proveniência — ver `_validator_k`.
            k_analytic=None,
            # `k_effective_factor` descreve o BokehMe, que esta rota não usa (`[A]` A12).
            k_effective_factor=None,
        ),
        depth=encoded, mask=mask, mask_source=MaskSource.BIREFNET,
        focus=focus_record,
        provenance=SampleProvenance(
            pipeline_commit=provenance_base["pipeline_commit"],
            depth_model_sha256=provenance_base["depth_model_sha256"],
            mask_model_sha256=provenance_base["mask_model_sha256"],
            image_hw=image_hw, seed=config.seed,
            # NENHUM renderizador na rota B (paper.txt:271,283,292,321). O antigo gravava
            # `{"name": "not_applicable_route_b", "is_final_label_renderer": True}` —
            # proveniência que mente na direção tranquilizadora (defeito B14).
            renderer=None,
            deblurnet=deblurred.provenance,
            source_license=provenance_base.get("source_license"),
            extra={
                # O diagnóstico da Eq. 3 — `sensor_width_mm`, `crop_factor`,
                # `pixel_ratio_px_per_mm`, `longest_edge_px`, os três termos da EXIF.
                # Sem ele, K é um número sem como auditar de qual sensor veio. O lugar
                # certo é um campo em `ControlLabel`, exigido quando
                # `k_source == EQ3_EXIF` — mudança em `dataio/`, descrita no relatório.
                "k_eq3_diagnostics": k_diag,
                "focus_depth_diagnostics": focus_diag,
                "deblur_diagnostics": deblur_diag,
                # O K independente, da distância de foco da EXIF. NUNCA rótulo.
                "k_validator_exif_focus_distance": _validator_k(source, image_hw),
                "mask_backend": provenance_base.get("mask_backend"),
                "mask_image": "aif",
                # A câmera, para a auditoria dos 30,33% de crop factor 1,0
                # (`ACHADOS.md:160-173`). Gravada em TODA amostra, não só nas suspeitas:
                # a pergunta "quais modelos ecoam a focal?" só se responde tendo o
                # denominador — quantas amostras cada modelo produziu no total.
                "camera_make": getattr(source, "camera_make", None),
                "camera_model": getattr(source, "camera_model", None),
                "route_b_decisions": {
                    "focus_refinement_applied": False,
                    "focus_refinement_evidence": "paper.txt:361-365 — o §3.2(c) "
                                                 "introduz o refino para LFDOF/RealBokeh "
                                                 "e diz 'Similar to (b)' para o passo "
                                                 "sem refino",
                    "d_focus_for_eq3": "1/focus_disparity",
                    "renderer": None,
                },
            },
        ),
        quality=report.to_dict(),
        # A AIF é gerada: ela é gravada. `channel_order` DECLARADO, não assumido — a
        # inversão cega `[..., ::-1]` transformava RGB [200,0,0] em [0,0,200] sem
        # registro (`dataio/writer.py:116-119`).
        generated_images={AIF_IMAGE_NAME: aif_bgr},
        channel_order="bgr",
    )


# --------------------------------------------------------------------------------
# O laço
# --------------------------------------------------------------------------------

def _ledger_line(root: Path, sample: Sample, meta: dict) -> dict:
    """Uma linha do ledger de bytes da rota B.

    `publish_release.py:125-137` exige `source_images.jsonl` com sha256 da AIF e da
    bokeh — escrito para a rota C, onde **as duas** são referência. Aqui a AIF é
    **gerada**, então o que existe para hashear é o arquivo que nós escrevemos. O sha256
    é do **JPEG em disco**, não do array em memória: JPEG q95 é lossy, e hashear o array
    provaria uma coisa e o release conteria outra.
    """
    caminho = root / "generated" / f"{sample.sample_id}_{AIF_IMAGE_NAME}.jpg"
    digest = None
    if caminho.is_file():
        h = hashlib.sha256()
        with caminho.open("rb") as fh:
            for bloco in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloco)
        digest = h.hexdigest()
    return {
        "sample_id": sample.sample_id,
        "route": ROUTE,
        "aif_role": "generated_by_deblurnet",
        "aif_path": str(caminho.relative_to(root)) if caminho.is_file() else None,
        "aif_jpeg_sha256": digest,
        "aif_jpeg_bytes": caminho.stat().st_size if caminho.is_file() else None,
        "bokeh_role": "reference",
        "bokeh_ref": sample.refs.bokeh_ref,
        "source_dataset": sample.refs.source_dataset,
        "source_sample_id": sample.refs.source_sample_id,
        "deblur_variant": (sample.provenance.deblurnet or {}).get("deblur_variant"),
        "deblur_lora_sha256": (sample.provenance.deblurnet or {}).get("deblur_lora_sha256"),
        "image_h": meta["image_h"], "image_w": meta["image_w"],
    }


def run_route_b(
    sources: Iterable[BokehSource],
    *,
    load_bokeh: LoadBokeh,
    deblur_runtime,
    depth_runtime,
    mask_runtime,
    split: SceneSplit,
    config: RouteBConfig,
    provenance_base: dict,
) -> RouteBStats:
    """Laço principal. Imprime o histograma de motivos de rejeição no fim, **sempre**.

    O `finally` é o mesmo padrão da rota C, e existe pela razão do `CLAUDE.md`: sem o
    histograma não dá para calibrar limiar nenhum, e é ele que denuncia fallback novo. E
    `json` está importado no topo deste módulo — o `NameError` de `route_b.py:298` do
    pipeline antigo era numa linha **dentro de um `finally`**, então matava o run na
    amostra 0 e nenhum `except Exception` a montante o pegava.
    """
    stats = RouteBStats()
    saida = Path(config.output_dir)
    log = RejectionLog(saida / "rejections.jsonl")
    writer = FileSampleWriter(saida, split=split,
                              depth_layout=config.depth_layout)
    ledger = (saida / "generated_images.jsonl").open("a", encoding="utf-8")
    done = writer.completed_ids()
    if done:
        print(f"[rota-b] retomando: {len(done)} amostras já gravadas serão puladas.")

    try:
        for source in sources:
            if config.limit is not None and stats.written >= config.limit:
                break
            if source.sample_id in done:
                stats.skipped_done += 1
                continue
            stats.processed += 1
            try:
                bokeh_bgr = load_bokeh(source)
                sample = process_sample(
                    source, bokeh_bgr=bokeh_bgr, deblur_runtime=deblur_runtime,
                    depth_runtime=depth_runtime, mask_runtime=mask_runtime,
                    config=config, provenance_base=provenance_base)
                meta = writer.write(sample)
            except SampleRejected as exc:
                log.reject_from(source.sample_id, exc,
                                {"scene_id": source.scene_id, "route": ROUTE})
                continue

            ledger.write(json.dumps(_ledger_line(saida, sample, meta),
                                    ensure_ascii=False) + "\n")
            ledger.flush()
            log.accept(source.sample_id, {"scene_id": source.scene_id,
                                          "k_value": meta["k_value"]})
            stats.written += 1
            _acumula(stats, sample, meta)
    finally:
        print(writer.stats.summary())
        print(log.summary())
        print(stats.summary())
        writer.close()
        log.close()
        ledger.close()
    return stats


def _acumula(stats: RouteBStats, sample: Sample, meta: dict) -> None:
    """Acumula a estatística do run a partir do metadado **GRAVADO**.

    Do metadado, e não do objeto em memória, pela mesma razão da rota C: o número que o
    resumo imprime tem que ser o que está no disco. O que não está no metadado plano —
    os diagnósticos — vem da proveniência da amostra, que é o objeto que o writer
    serializou.
    """
    stats.k_values.append(float(meta["k_value"]))
    stats.focus_agreements.append(float(meta["focus_agreement"]))

    extra = sample.provenance.extra or {}
    validador = extra.get("k_validator_exif_focus_distance")
    if validador is not None:
        stats.k_validator_values.append(float(validador))

    foco = extra.get("focus_depth_diagnostics") or {}
    razao = foco.get("focus_depth_ratio")
    if razao is not None:
        stats.focus_depth_ratios.append(float(razao))

    deblur = extra.get("deblur_diagnostics") or {}
    espalhamento = deblur.get("deblur_retention_spread")
    if espalhamento is not None:
        stats.deblur_spreads.append(float(espalhamento))

    qualidade = meta.get("quality") or {}
    ssim_gate = qualidade.get("deblur_structural_ssim_min")
    if ssim_gate is not None:
        stats.deblur_ssim_values.append(float(ssim_gate["value"]))
    deslocamento = qualidade.get("pair_registration_shift_px")
    if deslocamento is not None:
        stats.shift_values.append(float(deslocamento["value"]))

    crop = qualidade.get("exif_crop_factor_unity")
    if crop is not None and float(crop["value"]) > 0.5:
        stats.crop_factor_unity += 1
        camera = " ".join(str(p) for p in (extra.get("camera_make"),
                                          extra.get("camera_model")) if p)
        stats.crop_factor_unity_cameras[camera or "câmera não publicada"] += 1
