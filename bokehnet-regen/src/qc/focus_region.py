"""Refinamento automático da região em foco — o substituto do passo manual do paper.

## O que o paper faz, e por que precisamos de um substituto

§3.2(c), `paper.txt:359-368`, sobre LFDOF e RealBokeh:

    "due to the increased diversity and complexity of the scenes in these datasets,
    the initial estimate of M is sometimes unreliable. **Rather than simply verifying
    and discarding unreliable cases**, we introduce a manual refinement step.
    Specifically, we re-select a small yet reliable in-focus region to correct M,
    thereby accurately extracting D_focus. This strategy preserves challenging samples
    rather than excluding them."

Medimos o "sometimes unreliable" no piloto de 162 amostras: a disparidade de foco tirada
da máscara do BiRefNet fica dentro de ±25% da distância de foco **medida** que a
RealBokeh publica em apenas **35,2%** dos casos. A razão mediana é 0,579 — a máscara
escolhe um plano cerca de 1,7× mais longe. No pior caso, `train_91`: foco medido a
10,23 m, máscara em 1,65 m. E a origem mede isso com incerteza mediana de **±0,010 m**.

A causa está nas imagens: a RealBokeh é feita de **cenas**, não de fotos de objeto — um
tronco de árvore num parque, um muro de pedra. O BiRefNet segmenta objeto **saliente**,
e fora desse domínio ele ou devolve probabilidade exatamente 0 (declinando, o que
descartava 20,6% das amostras) ou segmenta algo que não é o que estava em foco.

Descartar é justamente a estratégia que o paper diz **não** ter adotado. Então o que
falta é o refinamento — automático, já que 22.990 correções à mão não vão acontecer.

## Como este módulo acha "uma região pequena porém confiável em foco"

Pela definição física de estar em foco, e não por saliência.

Temos a AIF **e** a bokeh da mesma cena. No plano de foco, a bokeh **preservou** o
detalhe da AIF; fora dele, perdeu. Então a grandeza que interessa é a **retenção**:

    retencao(x) = media_local(|laplaciano(bokeh)|) / media_local(|laplaciano(aif)|)

Perto de 1 no plano de foco, perto de 0 no que borrou. O denominador é o que faz isso
funcionar: ele **normaliza pela textura da própria cena**. Medida de nitidez absoluta
falha exatamente aqui — uma folhagem desfocada tem mais energia de alta frequência que
uma parede lisa em foco, e um `argmax` de nitidez escolheria a folhagem. A razão cancela
isso, porque a folhagem é texturizada nas duas imagens.

É a mesma grandeza que o gate `focus_mask_is_sharpest` já usava para testar a hipótese
da máscara. Aqui ela deixa de só julgar e passa a também propor.

## O que este módulo NÃO faz

Não decide sozinho o que vira rótulo. Ele devolve a região refinada **e** de onde ela
veio, e quem chama grava as duas coisas. Amostra refinada fica marcada, para que dê
para treinar com e sem elas e medir a diferença — que é a única forma de saber se o
refinamento ajudou.

Também não usa a distância de foco medida como rótulo. Ela é o **gabarito** contra o
qual este método é validado (`scripts/validate_focus_refinement.py`), e usá-la como
rótulo resolveria a RealBokeh e deixaria o LFDOF — que não publica distância nenhuma —
com um rótulo de outra qualidade na mesma rota. Duas qualidades de rótulo no mesmo lote
foi o erro do kfix, com outra roupa.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from control.contract import reject

#: Janela do filtro de média, em pixels, na resolução em que a retenção é medida.
#: Grande o bastante para o laplaciano local ter estatística, pequena o bastante para
#: não misturar plano de foco com fundo. `[A]` — o paper não publica nada disto.
DEFAULT_WINDOW_PX = 33

#: Fração da imagem que forma a "região pequena porém confiável". `[A]`.
#: Pequena de propósito: o paper pede *small yet reliable*, e a mediana da disparidade
#: só precisa de amostra suficiente, não da região inteira.
DEFAULT_TOP_FRACTION = 0.05

#: Percentil de DETALHE DA AIF abaixo do qual o pixel não concorre. Relativo à própria
#: imagem, de propósito: um limiar absoluto depende de exposição, ISO e conteúdo, e o
#: 1e-3 da primeira versão nunca disparava — céu real dá |lap| ≈ 2,5, 2.500x acima.
#:
#: **30 e não 60**, e o valor foi medido, não escolhido. Dois cenários opostos:
#:
#:   * foco numa região COM textura, fora de foco um céu liso — o caso que motivou o
#:     piso, porque a razão vale ~1 no céu por ser ruído sobre ruído;
#:   * foco numa região com POUCA textura, fora de foco uma folhagem fina — o caso em
#:     que o piso atrapalha, porque a região certa é a menos texturizada.
#:
#: Varrendo o percentil, a região cai 100% no lado correto **nos dois** até 40, e a
#: partir de 50 o segundo cenário desaba para 4% e depois 0%. 30 fica com folga dentro
#: da faixa segura.
#:
#: A medição também mostrou o que de fato conserta o problema do céu, e **não é o
#: piso**: é o desempate por detalhe da AIF em `sharpest_region_mask`. Mesmo com o piso
#: em 0 os dois cenários acertam. O piso é rede de segurança para quando retenção
#: **e** detalhe empatam; por isso pode ser modesto. `[A]`, calibrável no piloto.
DEFAULT_MIN_DETAIL_PERCENTILE = 30.0

#: Piso de área para a região refinada ser utilizável. `[A]`.
MIN_REGION_AREA_RATIO = 0.001

#: Acima disto a máscara inicial e a região de retenção concordam, e a máscara inicial
#: é mantida — o caminho do paper, sem refinamento. `[A]`.
DEFAULT_AGREEMENT_FLOOR = 0.30


class FocusSource(str, Enum):
    """De onde saiu a região que define `D_focus`. Gravado por amostra.

    Enum fechado, no molde de `MaskSource`: uma fonte nova ganha valor próprio e nunca
    reusa o de outra. Proveniência que mente é pior que proveniência ausente — a cascata
    antiga do pipeline caía no GrabCut e continuava dizendo `"automatic"`.
    """

    #: Máscara do BiRefNet, aceita como veio. O caminho do paper quando M é confiável.
    BIREFNET = "birefnet"
    #: Máscara do BiRefNet **corrigida** pela região de retenção — o passo que o paper
    #: faz à mão. A amostra fica marcada.
    BIREFNET_REFINED = "birefnet_refined"
    #: Não havia máscara utilizável (BiRefNet vazio ou em desacordo total). A região vem
    #: só da retenção. É o caso que antes virava `focus_mask_empty` e era descartado.
    RETENTION_ONLY = "retention_only"
    #: Rota A: o plano de foco é **sorteado**, não estimado — `paper.txt:329-330`,
    #: *"we randomly sample a focus plane D_focus and a target bokeh level K"*.
    #:
    #: Valor próprio, e não `DEPTH_BAND` ou `BIREFNET`: ali houve uma máscara e alguém
    #: a mediu; aqui não houve máscara nenhuma. Reusar um valor existente seria
    #: proveniência que mente, que é o defeito que este enum existe para impedir — o
    #: pipeline antigo caía no GrabCut e continuava dizendo `"automatic"`.
    SAMPLED_PLANE = "sampled_plane"


@dataclass(frozen=True)
class RefinedFocusRegion:
    """A região em foco, e a história de como se chegou nela."""

    mask: np.ndarray
    source: FocusSource
    #: `|inicial ∩ retenção| / |retenção|` — quanto da evidência física a máscara
    #: inicial alcança. **É esta que decide o ramo**, e a escolha merece explicação.
    #:
    #: A crítica óbvia é que ela premia máscara grande: uma máscara cobrindo metade do
    #: quadro alcança tudo e pontua 1,000. A resposta seria IoU — mas IoU está errado
    #: aqui, por um motivo estrutural: a região de retenção é **fixada** no topo de 5%
    #: do quadro, enquanto a máscara é do tamanho do objeto. Uma máscara perfeita de um
    #: objeto que ocupa 20% da imagem e contém toda a região de retenção tem IoU de
    #: 0,25 — abaixo do piso — e seria refinada sem necessidade. IoU pune a diferença
    #: de tamanho, que aqui é de construção e não defeito.
    #:
    #: O tamanho excessivo é tratado onde ele de fato importa: pelo gate
    #: `max_mask_area_ratio`, e por `precision` e `iou`, gravados ao lado para que o
    #: caso "grande demais" seja visível em vez de invisível.
    agreement: float
    #: `|inicial ∩ retenção| / |inicial|` — quanto da máscara inicial está DENTRO da
    #: evidência física. É esta que denuncia a máscara grande demais.
    precision: float
    #: IoU, simétrico. Resumo único; não decide nada, e por isso não esconde de que
    #: lado foi o desacordo — para isso existem `agreement` e `precision`.
    iou: float
    #: Retenção mediana dentro da região final. Perto de 1 = a região realmente
    #: preservou o detalhe da AIF. Baixa = nem a retenção achou plano de foco claro,
    #: e a amostra merece desconfiança mesmo tendo passado.
    retention_in_region: float
    #: Área da região final, como fração do quadro.
    area_ratio: float
    #: `True` quando a máscara inicial estava vazia — o caso que antes era descartado.
    initial_mask_was_empty: bool

    @property
    def was_refined(self) -> bool:
        """A amostra passou pelo substituto do passo manual do paper.

        É por este campo que dá para treinar com e sem essas amostras e medir se o
        refinamento ajudou. Sem ele, "consertamos" seria uma afirmação sem teste.
        """
        return self.source is not FocusSource.BIREFNET


# --------------------------------------------------------------------------------
# Primitivas — numpy puro, sem cv2
# --------------------------------------------------------------------------------

def _luma(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> luminância float32. Coeficientes ITU-R BT.601."""
    a = np.asarray(image_bgr, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.ndim != 3 or a.shape[2] != 3:
        reject("resolution_invalid", f"esperado HxWx3 ou HxW, recebido {a.shape}")
    return 0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]


def _laplacian_abs(gray: np.ndarray) -> np.ndarray:
    """|∇²I| pelo estêncil de 5 pontos, com borda replicada.

    Borda replicada e não zero: preencher com zero cria uma borda artificial de alta
    frequência, e a moldura da imagem inteira competiria pela região "mais nítida".
    """
    p = np.pad(gray, 1, mode="edge")
    lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
           - 4.0 * p[1:-1, 1:-1])
    return np.abs(lap)


def _box_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Média em janela quadrada, por imagem integral. O(1) por pixel.

    A janela encolhe nas bordas em vez de assumir valores fora do quadro, e a divisão
    usa a contagem real de pixels — média sobre pixel que não existe seria invenção.
    """
    if window < 1:
        raise ValueError(f"janela tem que ser >= 1: {window}")
    h, w = values.shape
    r = int(window) // 2
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(np.cumsum(values.astype(np.float64), axis=0), axis=1,
              out=integral[1:, 1:])

    linhas = np.arange(h)
    colunas = np.arange(w)
    y0 = np.maximum(linhas - r, 0)
    y1 = np.minimum(linhas + r + 1, h)
    x0 = np.maximum(colunas - r, 0)
    x1 = np.minimum(colunas + r + 1, w)

    soma = (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
            - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)])
    contagem = np.outer(y1 - y0, x1 - x0).astype(np.float64)
    return (soma / contagem).astype(np.float32)


def detail_maps(aif_bgr: np.ndarray, bokeh_bgr: np.ndarray, *,
               window: int = DEFAULT_WINDOW_PX,
               min_aif_detail: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """`(retenção, detalhe_da_AIF)` — os dois mapas, porque um sem o outro engana.

    A retenção sozinha **não** distingue "em foco" de "liso demais para saber". Uma
    região lisa que borra continua lisa: `|lap(bokeh)| ≈ |lap(aif)|`, retenção ≈ 1, o
    mesmo valor que o plano de foco produz. Medido numa cena de objeto texturizado em
    foco contra céu liso desfocado: **79,7%** dos pixels válidos empatavam em 1,000, e
    **42,9%** da região selecionada caía no céu.

    O detalhe da AIF é o que separa os dois casos, e por isso viaja junto: ele diz
    **quanta evidência** existe naquele pixel. Onde a AIF é lisa, a razão é ruído
    dividido por ruído, e o número não significa nada por mais alto que seja.
    """
    aif = np.asarray(aif_bgr)
    bokeh = np.asarray(bokeh_bgr)
    if aif.shape[:2] != bokeh.shape[:2]:
        reject("resolution_invalid",
               f"aif {aif.shape[:2]} != bokeh {bokeh.shape[:2]} na retenção")

    detalhe_aif = _box_mean(_laplacian_abs(_luma(aif)), window)
    detalhe_bokeh = _box_mean(_laplacian_abs(_luma(bokeh)), window)

    # `min_aif_detail` continua aqui só para pegar o caso degenerado de imagem
    # exatamente constante. Ele NÃO serve como critério de confiabilidade: medido, um
    # gradiente de céu real dá |lap| ≈ 2,5 — 2.500x acima deste limiar. Quem decide
    # confiabilidade é o percentil em `sharpest_region_mask`, que é relativo à própria
    # imagem e por isso não depende de escala, exposição ou ruído da câmera.
    com_detalhe = detalhe_aif > min_aif_detail
    retencao = np.full(detalhe_aif.shape, np.nan, dtype=np.float32)
    np.divide(detalhe_bokeh, detalhe_aif, out=retencao, where=com_detalhe)
    # Acima de 1 é ruído ou desalinhamento, não "mais nítida que a AIF".
    np.clip(retencao, 0.0, 1.0, out=retencao)
    return retencao, detalhe_aif


def detail_retention(aif_bgr: np.ndarray, bokeh_bgr: np.ndarray, *,
                     window: int = DEFAULT_WINDOW_PX,
                     min_aif_detail: float = 1e-3) -> np.ndarray:
    """Só o mapa de retenção. Para DIAGNÓSTICO — ver o aviso.

    Quem for **selecionar região** precisa de `detail_maps`, não disto: a retenção
    sozinha não distingue plano de foco de superfície lisa, e escolher por ela leva a
    região para o céu. A rota B usa esta função como diagnóstico da DeblurNet, onde a
    pergunta é outra (a dispersão dos valores, não onde está o máximo).
    """
    return detail_maps(aif_bgr, bokeh_bgr, window=window,
                       min_aif_detail=min_aif_detail)[0]


def sharpest_region_mask(retention: np.ndarray, aif_detail: np.ndarray, *,
                         top_fraction: float = DEFAULT_TOP_FRACTION,
                         min_detail_percentile: float = DEFAULT_MIN_DETAIL_PERCENTILE,
                         min_area_ratio: float = MIN_REGION_AREA_RATIO,
                         ) -> Optional[np.ndarray]:
    """A "região pequena porém confiável": o topo da retenção, **onde há textura**.

    Dois consertos sobre a primeira versão, os dois medidos:

    1. **Piso de textura relativo.** Só concorrem pixels cujo detalhe na AIF está acima
       do percentil `min_detail_percentile` do detalhe da própria imagem. Relativo, e
       não absoluto, porque o valor absoluto depende de exposição, ISO e conteúdo — o
       limiar fixo de 1e-3 da primeira versão nunca disparava (céu real dá 2,5).
    2. **Seleção por POSTO, não por `>= quantil`.** A retenção satura em 1,0 e empata em
       massa: medido, **79,7%** dos pixels válidos empatavam no teto, e `top_fraction`
       ficava inerte — variá-la de 0,20 a 0,001, fator 200, devolvia exatamente a mesma
       região de 77,7% do quadro. Agora se ordena e se pega os N primeiros, com o
       desempate pelo **detalhe da AIF**: entre pixels igualmente retentivos, ganha
       aquele onde a evidência é mais forte.

    Devolve `None` quando não sobra pixel confiável — cena lisa demais para localizar
    plano de foco. `None` é resposta; inventar região aqui seria fallback com outro nome.
    """
    if not 0.0 < top_fraction < 1.0:
        raise ValueError(f"top_fraction fora de (0,1): {top_fraction}")
    if not 0.0 <= min_detail_percentile < 100.0:
        raise ValueError(f"min_detail_percentile fora de [0,100): {min_detail_percentile}")
    if retention.shape != aif_detail.shape:
        raise ValueError(f"mapas com shapes diferentes: {retention.shape} e "
                         f"{aif_detail.shape}")

    finitos = np.isfinite(retention)
    if not finitos.any():
        return None

    corte_textura = float(np.percentile(aif_detail[finitos], min_detail_percentile))
    elegiveis = finitos & (aif_detail >= corte_textura)

    piso = max(int(min_area_ratio * retention.size), 1)
    n_elegiveis = int(elegiveis.sum())
    if n_elegiveis < piso:
        return None

    alvo = min(max(int(round(top_fraction * retention.size)), piso), n_elegiveis)

    idx = np.flatnonzero(elegiveis)
    # `lexsort` usa a ÚLTIMA chave como primária: retenção manda, detalhe desempata.
    ordem = np.lexsort((-aif_detail.ravel()[idx], -retention.ravel()[idx]))
    escolhidos = idx[ordem[:alvo]]

    regiao = np.zeros(retention.shape, dtype=bool)
    regiao.ravel()[escolhidos] = True
    return regiao


def refine_focus_mask(initial_mask: Optional[np.ndarray], retention: np.ndarray,
                      aif_detail: np.ndarray, *,
                      top_fraction: float = DEFAULT_TOP_FRACTION,
                      min_detail_percentile: float = DEFAULT_MIN_DETAIL_PERCENTILE,
                      agreement_floor: float = DEFAULT_AGREEMENT_FLOOR,
                      min_area_ratio: float = MIN_REGION_AREA_RATIO,
                      ) -> RefinedFocusRegion:
    """Máscara inicial + retenção -> região em foco, com a origem declarada.

    A política, e a razão de cada ramo:

    1. **A inicial concorda com a retenção** (alcança pelo menos `agreement_floor`
       dela — ver a nota em `RefinedFocusRegion.agreement` sobre por que não é IoU):
       mantém a inicial. É o caminho do paper, e não há por que mexer no que está certo.
    2. **A inicial existe mas discorda**: interseca as duas. Preserva a intenção
       semântica do BiRefNet — o objeto — e restringe à parte dele que a física diz
       estar em foco. Se a interseção ficar pequena demais, cai no ramo 3.
    3. **Não há inicial utilizável**: usa só a retenção. É o caso que virava
       `focus_mask_empty` e era **descartado**, contra o que o paper manda fazer.

    Levanta `SampleRejected("focus_mask_empty")` só quando nem a retenção decide — cena
    sem detalhe em lugar nenhum. Aí não há plano de foco a extrair, e descartar não é
    escolha, é consequência.
    """
    regiao_retencao = sharpest_region_mask(
        retention, aif_detail, top_fraction=top_fraction,
        min_detail_percentile=min_detail_percentile, min_area_ratio=min_area_ratio)
    if regiao_retencao is None:
        reject("focus_mask_empty",
               "nem BiRefNet nem retenção acharam região em foco: a cena não tem "
               "detalhe suficiente para localizar o plano de foco")

    total = float(retention.size)
    piso_area = max(int(min_area_ratio * total), 1)

    inicial = None if initial_mask is None else np.asarray(initial_mask, dtype=bool)
    inicial_vazia = inicial is None or not inicial.any()

    if inicial_vazia:
        acordo = precisao = iou = 0.0
    else:
        intersecao = float((inicial & regiao_retencao).sum())
        acordo = intersecao / max(float(regiao_retencao.sum()), 1.0)
        precisao = intersecao / max(float(inicial.sum()), 1.0)
        iou = intersecao / max(float((inicial | regiao_retencao).sum()), 1.0)

    def _empacota(mask, source):
        dentro = retention[mask & np.isfinite(retention)]
        return RefinedFocusRegion(
            mask=mask, source=source, agreement=acordo,
            precision=precisao, iou=iou,
            retention_in_region=float(np.median(dentro)) if dentro.size else float("nan"),
            area_ratio=float(mask.sum() / total),
            initial_mask_was_empty=bool(inicial_vazia))

    if not inicial_vazia and acordo >= agreement_floor:
        return _empacota(inicial, FocusSource.BIREFNET)

    if not inicial_vazia:
        intersecao = inicial & regiao_retencao
        if intersecao.sum() >= piso_area:
            return _empacota(intersecao, FocusSource.BIREFNET_REFINED)

    return _empacota(regiao_retencao, FocusSource.RETENTION_ONLY)
