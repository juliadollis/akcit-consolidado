"""Gates de qualidade — MEDEM sempre, BLOQUEIAM só quando há limiar congelado.

Decisão que governa o módulo: **limiar não medido não bloqueia.** Todo gate calcula
seu valor e o grava; só rejeita se um limiar tiver sido passado explicitamente. Um
limiar escolhido a priori é um chute com cara de rigor, e o piloto existe justamente
para transformar histograma em número.

Isso separa duas coisas que o pipeline antigo misturava:

- **estruturalmente certo** — unidade, schema, proveniência, flag de censura. Bloqueia
  desde já, porque ou está certo ou o dado é inauditável.
- **limiar de qualidade** — SSIM, nitidez, IoU, faixa de profundidade. Mede no piloto,
  congela depois.

Sobre a máscara ruim, o §3.2(c) (paper.txt:364-368) diz textualmente: *"**Rather than
simply verifying and discarding unreliable cases**, we introduce a manual refinement
step [...] This strategy **preserves challenging samples rather than excluding them**"*.

Até a etapa 8 este módulo era a resposta errada a isso: o paper **corrigia** a máscara e
nossos gates **descartavam** a amostra. Desde a etapa 9 a correção existe, automática, em
`qc.focus_region`, e a rota C a aplica antes dos gates. O que sobrou aqui é o papel certo
de um gate — medir e, com limiar congelado, bloquear —, e duas consequências que valem
ler antes de mexer:

1. **Gate que não conseguiu medir não reprova.** `GateResult.passed` reprova valor
   não-finito antes de olhar o limiar, então um NaN vira rejeição com um slug que afirma
   mérito. Isso desfazia o refinamento pela porta de trás em dois lugares: `mask_iou`
   com as duas máscaras vazias (a IoU não existe) e `focus_mask_is_sharpest` com região
   pontilhada apagada pela erosão. Os dois devolvem `applicable=False`.
2. **Qual gate julga qual máscara é decisão da rota**, e está documentada em
   `routes.route_c.build_gate_report`: os que afirmam algo sobre o plano de foco julgam a
   região FINAL, e os que comparam AIF com bokeh julgam as máscaras cruas do BiRefNet.

Dois gates conferem a hipótese FÍSICA de que a máscara marca a região em foco, e a
diferença entre eles importa: `focus_mask_is_sharpest` mede nitidez **absoluta** na
bokeh, e `focus_region_retention` mede a razão bokeh/AIF, **normalizada pela textura da
cena**. Só o segundo acerta numa parede lisa legitimamente em foco. Nenhum dos dois é a
IoU entre máscaras AIF/bokeh, que não pega o modo de falha dominante: quando o fotógrafo
focou o fundo, o BiRefNet acha o objeto saliente e as duas máscaras CONCORDAM, com IoU
alta, enquanto o plano de foco está errado e o sweep inteiro otimiza para o plano errado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from control.contract import FOCUS_DEPTH_MAX_M, FOCUS_DEPTH_MIN_M
from qc.metrics import laplacian_variance, to_gray


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    threshold: Optional[float] = None
    higher_is_better: bool = True
    note: str = ""
    #: `False` quando a grandeza **não existe para esta fonte** — não quando ela existe
    #: e deu ruim. São coisas diferentes, e colapsar as duas em NaN custou caro:
    #: `aif_f_number` é `None` no LFDOF, que não publica f-number, e o NaN resultante
    #: reprovava 100% das amostras com o slug `gate_aif_aperture_wide` — motivo
    #: registrado, histograma bonito, conclusão errada.
    #:
    #: Gate inaplicável **não bloqueia** e **aparece no metadado** como inaplicável, de
    #: modo que "não medimos" nunca se disfarce de "medimos e passou".
    applicable: bool = True

    @property
    def measured_only(self) -> bool:
        return self.threshold is None

    @property
    def passed(self) -> bool:
        if not self.applicable:
            return True                       # não julga o que não pode medir
        if not np.isfinite(self.value):
            return False                      # medido e indefinido: nunca passa
        if self.threshold is None:
            return True                       # sem limiar congelado, não bloqueia
        return (self.value >= self.threshold) if self.higher_is_better else (self.value <= self.threshold)

    def to_dict(self) -> dict:
        return {
            "value": float(self.value),
            "threshold": None if self.threshold is None else float(self.threshold),
            "higher_is_better": self.higher_is_better,
            "measured_only": self.measured_only,
            "applicable": self.applicable,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    def add(self, result: GateResult) -> "GateReport":
        self.results.append(result)
        return self

    @property
    def blocked_by(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    @property
    def accepted(self) -> bool:
        return not self.blocked_by

    def to_dict(self) -> dict:
        return {r.name: r.to_dict() for r in self.results}

    def values(self) -> dict:
        return {r.name: float(r.value) for r in self.results}


# --------------------------------------------------------------------------------
# Máscara
# --------------------------------------------------------------------------------

def mask_area_ratio(mask: np.ndarray, *, min_ratio: Optional[float] = None,
                    max_ratio: Optional[float] = None) -> list[GateResult]:
    binary = np.asarray(mask) > 0.5
    ratio = float(binary.mean())
    return [
        GateResult("mask_area_ratio_min", ratio, min_ratio, True,
                   "máscara vazia ou minúscula não define plano de foco"),
        GateResult("mask_area_ratio_max", ratio, max_ratio, False,
                   "máscara quase integral não distingue foco de fundo"),
    ]


def mask_border_coverage(mask: np.ndarray, *, max_ratio: Optional[float] = None) -> GateResult:
    binary = np.asarray(mask) > 0.5
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    return GateResult("mask_border_coverage", float(border.mean()), max_ratio, False,
                      "máscara colada na borda costuma ser fundo, não objeto")


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, *,
             min_iou: Optional[float] = None) -> GateResult:
    a, b = np.asarray(mask_a) > 0.5, np.asarray(mask_b) > 0.5
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        # Duas máscaras VAZIAS davam 1.0 — o pior caso possível recebendo a nota
        # máxima, e indistinguível de concordância real no relatório.
        #
        # `applicable=False`, e não reprovação: quando o BiRefNet declina nas duas
        # imagens, a IoU entre elas **não existe** — não é "existe e deu ruim". Enquanto
        # a máscara vazia rejeitava a amostra antes dos gates, este ramo era inalcançável
        # na rota C; com o refinamento de `qc.focus_region` a amostra chega até aqui, e
        # reprovar por NaN devolveria pela porta do gate os 20,6% de descarte que o
        # §3.2(c) proíbe explicitamente. O sinal não se perde: ele está em
        # `focus_initial_mask_was_empty` e em `focus_agreement == 0` no metadado.
        return GateResult("mask_iou_aif_bokeh", float("nan"), min_iou, True,
                          "ambas as máscaras do BiRefNet vazias: IoU INEXISTENTE, não "
                          "é concordância e não é reprovação — ver "
                          "focus_initial_mask_was_empty",
                          applicable=False)
    value = float(np.logical_and(a, b).sum() / union)
    return GateResult("mask_iou_aif_bokeh", value, min_iou, True,
                      "concordância entre a máscara da AIF e a da bokeh; NÃO prova que "
                      "a região está em foco")


# --------------------------------------------------------------------------------
# O gate que confere a hipótese física
# --------------------------------------------------------------------------------

def focus_mask_is_sharpest(
    bokeh_image: np.ndarray,
    mask: np.ndarray,
    *,
    min_ratio: Optional[float] = None,
    erode_px: int = 3,
) -> GateResult:
    """Nitidez DENTRO sobre nitidez FORA da máscara, medida na imagem BOKEH.

    É o único gate que testa a hipótese que o rótulo inteiro assume: *esta máscara
    marca a região em foco*. Numa foto com bokeh, a região em foco é, por definição, a
    mais nítida — então a razão tem que ser bem maior que 1.

    Quando o fotógrafo focou o fundo e o BiRefNet marcou o objeto saliente em primeiro
    plano, a razão cai **abaixo de 1** e o caso é pego. Nenhum gate de IoU pega isso,
    porque as duas máscaras automáticas concordam no objeto errado.

    A erosão tira a borda da máscara do cálculo: numa borda de objeto desfocado existe
    gradiente alto que não é nitidez, e ele contaminaria os dois lados.
    """
    gray = to_gray(bokeh_image)
    inside = _erode(np.asarray(mask) > 0.5, erode_px)
    outside = _erode(~(np.asarray(mask) > 0.5), erode_px)

    if inside.sum() < 64 or outside.sum() < 64:
        # `applicable=False`: a razão não pôde ser MEDIDA, e reprovar aqui afirmaria
        # "esta máscara não é a região mais nítida" sem ter medido nitidez nenhuma —
        # um slug com conteúdo inventado, que é pior que gate ausente.
        #
        # O ramo passou a ser comum quando a rota C começou a julgar a REGIÃO REFINADA:
        # a região de retenção é o topo de um quantil e pode sair pontilhada, e três
        # passos de erosão a apagam. Sem isto, cada região pontilhada saía rejeitada
        # como se a máscara estivesse no lugar errado.
        return GateResult("focus_mask_sharpness_ratio", float("nan"), min_ratio, True,
                          "área insuficiente dentro ou fora da máscara após erosão: "
                          "razão NÃO medida, não reprovada",
                          applicable=False)

    sharp_in = _local_sharpness(gray, inside)
    sharp_out = _local_sharpness(gray, outside)
    ratio = float(sharp_in / sharp_out) if sharp_out > 0 else float("inf")
    return GateResult("focus_mask_sharpness_ratio", ratio, min_ratio, True,
                      "nitidez dentro/fora da máscara NA BOKEH. < 1 sugere foco no fundo "
                      "com máscara no primeiro plano")


def focus_region_retention(
    retention_in_region: float, *, min_retention: Optional[float] = None,
) -> GateResult:
    """Retenção mediana DENTRO da região que definiu `D_focus`, em [0, 1].

    É o gate certo para julgar a região em foco depois do refinamento automático do
    §3.2(c), e existe porque `focus_mask_is_sharpest` não serve para esse papel:

    * `focus_mask_is_sharpest` mede nitidez **absoluta** na bokeh. Uma parede lisa
      legitimamente em foco tem pouca energia de alta frequência e reprovaria; uma
      folhagem desfocada tem muita e passaria. É exatamente a armadilha que
      `qc.focus_region` foi escrito para não cair.
    * esta razão é **normalizada pela textura da própria cena** (bokeh ÷ AIF no mesmo
      pixel), então parede lisa em foco dá ~1 e folhagem desfocada dá ~0.

    Perto de 1: a região preservou o detalhe da AIF, logo estava no plano de foco.
    Baixa: nem a retenção achou plano de foco claro, e o `D_focus` da amostra merece
    desconfiança mesmo tendo sido produzido.

    Default `None` como todo gate deste módulo: mede no piloto, congela depois. O valor
    não vem do paper, que não publica nada sobre o passo de refinamento.
    """
    return GateResult("focus_region_retention", float(retention_in_region),
                      min_retention, True,
                      "retenção mediana de detalhe dentro da região que produziu "
                      "D_focus; normalizada pela textura da cena")


def _erode(binary: np.ndarray, px: int) -> np.ndarray:
    """Erosão por mínimo em janela quadrada, sem depender de cv2."""
    if px <= 0:
        return binary
    out = binary.copy()
    for _ in range(px):
        shifted = out.copy()
        shifted[1:, :] &= out[:-1, :]
        shifted[:-1, :] &= out[1:, :]
        shifted[:, 1:] &= out[:, :-1]
        shifted[:, :-1] &= out[:, 1:]
        out = shifted
    return out


def _local_sharpness(gray: np.ndarray, region: np.ndarray) -> float:
    """Energia do Laplaciano dentro da região. Comparável entre regiões da MESMA imagem."""
    lap = (-4.0 * gray
           + np.pad(gray, ((1, 0), (0, 0)), mode="edge")[:-1, :]
           + np.pad(gray, ((0, 1), (0, 0)), mode="edge")[1:, :]
           + np.pad(gray, ((0, 0), (1, 0)), mode="edge")[:, :-1]
           + np.pad(gray, ((0, 0), (0, 1)), mode="edge")[:, 1:])
    return float(np.mean(lap[region] ** 2))


# --------------------------------------------------------------------------------
# Nitidez da AIF
# --------------------------------------------------------------------------------

def aif_sharpness(aif_image: np.ndarray, *, min_variance: Optional[float] = None) -> GateResult:
    """Variância do Laplaciano — a medida de nitidez do paper (supp. B.1 e B.2).

    Só é comparável DENTRO de uma mesma fonte: a variância escala com resolução e com
    compressão, então ranquear EBB! contra Generative Photography num ranking único
    enviesa a seleção para a fonte de maior resolução.
    """
    return GateResult("aif_laplacian_variance", laplacian_variance(aif_image),
                      min_variance, True, "comparável só dentro da mesma fonte")


# --------------------------------------------------------------------------------
# Profundidade
# --------------------------------------------------------------------------------

def depth_useful_levels(disparity_u16: np.ndarray, *, min_levels: Optional[int] = None) -> GateResult:
    """Quantos níveis distintos a cena ocupa de fato.

    Mede o defeito real por trás do `z_max == 10000`: não é o teto em si — em
    disparidade `1/10000 ≈ 0` é inofensivo — é a cena útil colapsar em poucos níveis.
    Medido no dataset antigo: 24,7% das amostras com menos de 256 níveis de 65535, e
    mediana de 23 no subgrupo `z_max >= 1000 m`.
    """
    levels = int(len(np.unique(np.asarray(disparity_u16))))
    return GateResult("depth_useful_levels", float(levels),
                      None if min_levels is None else float(min_levels), True,
                      "níveis distintos de disparidade ocupados pela cena")


def focus_depth_plausible(
    focus_disparity: float, *,
    min_m: float = FOCUS_DEPTH_MIN_M,
    max_m: float = FOCUS_DEPTH_MAX_M,
) -> list[GateResult]:
    """Plano de foco em faixa fisicamente plausível. Pega a sentinela de 10.000 m.

    A primeira versão deste gate **calculava o veredito e o jogava fora**: computava
    `inside` e o usava só para concatenar "FORA DA FAIXA" numa string, com
    `threshold=None`. Ou seja, `passed` era `True` SEMPRE, e um `z_focus` de 10.000 m
    — a sentinela do Depth Pro, 25,7% das amostras medidas — passava com `passed:true`.

    Agravante da versão antiga: `0.05` e `1000.0` eram literais aqui, cópia das
    constantes de `control.contract`. Mudar o contrato não mudava o gate — a forma
    "cópias divergem" que o projeto inteiro existe para impedir. Agora são importadas.
    """
    depth_m = 1.0 / float(focus_disparity) if focus_disparity > 0 else float("inf")
    return [
        GateResult("focus_depth_m_min", depth_m, min_m, True,
                   "plano de foco raso demais para ser físico"),
        GateResult("focus_depth_m_max", depth_m, max_m, False,
                   "z_focus no teto do Depth Pro é sentinela, não medida"),
    ]


# --------------------------------------------------------------------------------
# Par AIF / bokeh
# --------------------------------------------------------------------------------

def pair_shape_matches(aif: np.ndarray, bokeh: np.ndarray) -> GateResult:
    """Estruturalmente certo, não limiar: shapes diferentes tornam tudo sem sentido."""
    same = aif.shape[:2] == bokeh.shape[:2]
    return GateResult("pair_shape_matches", 1.0 if same else 0.0, 1.0, True,
                      f"aif {aif.shape[:2]} vs bokeh {bokeh.shape[:2]}")


def bokeh_is_blurrier_than_aif(aif: np.ndarray, bokeh: np.ndarray, *,
                               max_ratio: Optional[float] = None) -> GateResult:
    """Sanidade do par: a bokeh tem que ser MENOS nítida que a AIF.

    O que este gate pega, com precisão: o caso em que a "AIF" é **quase tão borrada
    quanto o alvo**, e a razão vai para perto de 1 ou acima. É o extremo do D6 — as
    3,0% de cenas da RealBokeh cujo maior f-stop em `gt/` é f/2.8 ou mais aberto.

    O que ele **não** pega: um par f/14 contra f/2.0. Ali a razão continua baixa,
    porque o alvo é de fato muito mais borrado — e ainda assim a "AIF" tem borrão
    residual que contamina depth, máscara e sweep. Para esse caso o gate certo é o
    `aif_aperture_is_narrow`, que olha o f-stop, não o pixel.
    """
    v_aif = laplacian_variance(aif)
    v_bokeh = laplacian_variance(bokeh)
    ratio = float(v_bokeh / v_aif) if v_aif > 0 else float("inf")
    return GateResult("bokeh_over_aif_sharpness", ratio, max_ratio, False,
                      "razão de nitidez bokeh/AIF; perto de 1 significa que a 'AIF' "
                      "não é mais nítida que o alvo")


def aif_aperture_is_narrow(f_number: Optional[float], *,
                           min_f_number: Optional[float] = None) -> GateResult:
    """O gate DIRETO do D6: a AIF tem que vir de abertura fechada.

    Na rota C antiga a "AIF" era o maior f-stop dentro de `gt/`, cuja mediana por cena
    é **f/14** e que em **12,7%** das cenas é f/5.6 ou mais aberto — em 3,0%, f/2.8. Ou
    seja: em uma cena a cada oito, a profundidade, a máscara e o sweep de K saíam todos
    de uma foto com bokeh forte.

    A correção estrutural é ler `train/in/<id>_f22.JPG`, que é f/22 em 100% das cenas.
    Este gate existe para provar que a correção está de pé, e para pegar qualquer
    caminho novo que volte a tirar a AIF de dentro de `gt/`.

    Valor sugerido `[A]`: `min_f_number=16.0`. Não vem do paper, que cala sobre
    f-stop mínimo de AIF. Default `None` como todo gate deste módulo — medir no
    piloto, congelar depois.
    """
    if f_number is None:
        # A origem não publica f-number. É o caso do LFDOF, onde a AIF vem de um light
        # field e o defeito D6 não pode ocorrer por construção — não há "escolher o
        # maior f-stop de dentro de gt/" ali. Inaplicável, não reprovado.
        return GateResult("aif_f_number", float("nan"), min_f_number, True,
                          "a origem não publica f-stop da AIF; gate inaplicável",
                          applicable=False)
    if not np.isfinite(f_number) or f_number <= 0:
        # Aqui a origem AFIRMA um f-number e ele é lixo. Isso é medição ruim, não
        # ausência de medição, e reprova.
        return GateResult("aif_f_number", float("nan"), min_f_number, True,
                          f"f-stop da AIF inválido: {f_number!r}")
    return GateResult("aif_f_number", float(f_number), min_f_number, True,
                      "AIF tem que vir de abertura fechada; a origem correta é "
                      "`train/in/<id>_f22.JPG`")


def calibration_ssim_is_reliable(
    calibration_ssim: Optional[float], *, min_ssim: Optional[float] = None,
) -> GateResult:
    """O limiar de SSIM da Eq. 5. **O paper afirma executar este passo, duas vezes.**

    §3.2(c), paper.txt:397-398: o `K*` selecionado vira rótulo *"provided that its
    corresponding SSIM exceeds a **predefined threshold** to ensure reliable
    supervision"*. E supp. B.2, paper.txt:1008: *"we applied a Structural Similarity
    (SSIM) threshold to filter out sub-optimal results"*.

    O valor do limiar o paper **não** publica. Default `None`, como todo gate deste
    módulo: mede no piloto, congela depois, contra painel revisado.

    Sem este gate a rota C não implementa o §3.2(c) inteiro — foi o defeito P1-C5 no
    dataset antigo, em que `calibration_ssim` era gravado para todas as amostras e
    nenhuma era filtrada por ele.
    """
    if calibration_ssim is None or not np.isfinite(calibration_ssim):
        return GateResult("calibration_ssim", float("nan"), min_ssim, True,
                          "SSIM da Eq. 5 ausente ou não-finito")
    return GateResult("calibration_ssim", float(calibration_ssim), min_ssim, True,
                      "limiar exigido pelo §3.2(c); o valor não é publicado")
