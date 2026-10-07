"""Alinhamento afim da profundidade de um braço contra a do Depth Pro BASE.

## Por que isto existe

A pergunta que o Caminho A faz é sobre a **forma** da profundidade e a qualidade da
borda: o Depth Pro afinado por nós descreve a cena melhor que o base, a ponto de o bokeh
renderizado a partir dele ficar mais parecido com a fotografia real?

Os nossos pesos **não são métricos**. O treino alinhava por transformação afim dentro da
própria loss, então a escala absoluta nunca entrou no gradiente e não há razão para ela
estar certa. Medido nas 50 imagens do RealDOF (`caminho_b_analise/saida/depth_stats.json`):
a profundidade sai em ~57-61% da base, com fator **variável por imagem** (0,27 a 0,84).

Sem alinhar, o que o experimento mediria seria isto:

    D_focus = mediana(1/z na região em foco)   ->  muda ~1,7x
    CoC     = K * (1/z - D_focus)              ->  toda a escala do desfoque muda junto

O `calibrate_k` absorveria parte disso ajustando K, mas não tudo, porque o deslocamento
(o `b` da afim) **não** é absorvível por um K escalar: ele desloca o plano focal relativo
a cada profundidade da cena. O resultado seria uma diferença de SSIM que vem de deriva de
escala, não de borda melhor — e nada no pipeline reclamaria, porque os valores continuam
fisicamente plausíveis o tempo todo. É o formato exato de defeito que este projeto já
levou três vezes.

## O que exatamente é alinhado

Mínimos quadrados de UM grau, **por amostra**, resolvido em forma fechada:

    minimiza_{a,b}  || a * x_braço + b - x_base ||²

`x` é a **disparidade**, `1/z`, e não a profundidade em metros. A escolha não é
estilística — é o domínio em que o resto do pipeline vive:

    contract.focus_disparity_from_mask  ->  mediana(1/z) na região
    contract.signed_coc_px              ->  K * (1/z - D_focus)

Alinhar em disparidade faz `D_focus` e o gradiente de CoC saírem na MESMA escala da base,
que é precisamente a deriva que se quer remover. Alinhar em metros deixaria a relação
entre os dois braços não-afim justamente na grandeza que o renderizador consome.

O domínio é `--dominio-alinhamento` e não constante: `profundidade` existe para quem
quiser medir a diferença, e o domínio usado vai para o metadado de toda amostra. Um
resultado sem essa marca é ilegível seis meses depois.

## O que este módulo NÃO faz

**Não conserta profundidade ruim.** Se o ajuste sair com inclinação `a <= 0`, o braço
inverteu perto e longe em relação à base, e isso **rejeita** a amostra com slug em vez de
virar um mapa espelhado — que foi o defeito D11 (Depth Anything devolvendo disparidade
normalizada como se fosse profundidade, amostra espelhada, nenhum gate cego a inversão
pegando).

**Não esconde o que fez.** `ultimo_ajuste` guarda `a`, `b`, `r2` e a fração de pixels que
o piso positivo tocou, e o chamador grava isso num JSONL por amostra. R² baixo é um
resultado — quer dizer que a relação entre os dois braços não é afim, e aí "alinhar por
afim" é uma aproximação que precisa aparecer no relatório, não uma limpeza silenciosa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from control.contract import MetricDepth, reject, validate_metric_depth

#: Domínios em que a afim pode ser resolvida. Conjunto fechado: um domínio novo muda o
#: significado de todo número do relatório, então ele entra aqui e no metadado junto.
DOMINIOS = ("disparidade", "profundidade")

#: Piso de disparidade positiva, em 1/m. `1e-4` é `z = 10.000 m`, que é exatamente o teto
#: do Depth Pro (`validate_metric_depth` documenta `z_max == 10000` como comportamento
#: correto, não defeito). Um pixel que o ajuste empurraria para disparidade <= 0 é um
#: pixel mais distante que o teto do modelo; prendê-lo no teto é a leitura certa, e a
#: FRAÇÃO de pixels presos vai para o metadado em vez de sumir.
PISO_DISPARIDADE = 1e-4

#: Idem em metros, para o domínio `profundidade`.
PISO_PROFUNDIDADE_M = 1.0 / 10_000.0


@dataclass(frozen=True)
class AjusteAfim:
    """O que o ajuste desta amostra afirma. Vai inteiro para o metadado."""

    a: float
    b: float
    r2: float
    dominio: str
    #: Fração de pixels que o piso positivo tocou depois de aplicar a afim.
    fracao_no_piso: float
    #: Razão entre o `D_focus` que a amostra teria antes e depois do alinhamento, quando
    #: o chamador informa a região. `None` quando não informa — é diagnóstico, não rótulo.
    razao_mediana_disparidade: Optional[float] = None

    def como_dict(self) -> dict:
        return {"a": float(self.a), "b": float(self.b), "r2": float(self.r2),
                "dominio": self.dominio,
                "fracao_no_piso": float(self.fracao_no_piso),
                "razao_mediana_disparidade": (
                    None if self.razao_mediana_disparidade is None
                    else float(self.razao_mediana_disparidade))}


def ajusta_afim(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """`(a, b, r2)` que minimiza `|| a*x + b - y ||²`. Forma fechada, float64.

    Forma fechada e não `np.polyfit`: são 3,5 milhões de pixels por imagem e a fórmula
    de dois momentos é uma passada de soma, sem montar matriz de projeto.

    R² é devolvido junto e não é enfeite: ele é a medida de quanto "a relação entre os
    dois braços é afim" é verdade nesta amostra. R² baixo não invalida o alinhamento —
    invalida a leitura de que só a escala mudou.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        reject("depth_non_finite",
               f"ajuste afim com {x.size} e {y.size} amostras — não há reta a ajustar")
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    sxx = float(dx @ dx)
    if sxx <= 0.0:
        reject("depth_range_degenerate",
               "o braço devolveu disparidade constante — não há inclinação a estimar")
    a = float(dx @ dy) / sxx
    b = my - a * mx
    syy = float(dy @ dy)
    residuo = float(np.sum((a * x + b - y) ** 2))
    r2 = 1.0 - residuo / syy if syy > 0 else 0.0
    return a, b, float(r2)


@dataclass
class DepthProAlinhado:
    """Embrulha um `DepthProRuntime` e alinha a saída dele à do Depth Pro BASE.

    Satisfaz o mesmo protocolo que `routes/route_c.py` exige de `depth_runtime`:
    `.infer(image_rgb) -> MetricDepth` e `.provenance() -> dict`. A rota não sabe que
    existe alinhamento; quem decide é o entrypoint, por flag.

    **Roda os DOIS modelos em cada imagem.** É o dobro do custo de profundidade, e é
    inerente: a referência de escala é a predição da base NAQUELA imagem, porque o fator
    varia de 0,27 a 0,84 entre imagens. Um fator global — a mediana das 50, digamos —
    seria mais barato e estaria errado em quase toda amostra.
    """

    alvo: Any
    base: Any
    dominio: str = "disparidade"
    #: Últimos ajustes, em ordem de chamada. O entrypoint drena isto para um JSONL.
    ajustes: list[AjusteAfim] = field(default_factory=list)
    calls: int = 0

    def __post_init__(self) -> None:
        if self.dominio not in DOMINIOS:
            raise ValueError(f"domínio {self.dominio!r} fora de {list(DOMINIOS)}")

    # -- proveniência ----------------------------------------------------------

    def provenance(self) -> dict:
        """A do braço ALVO, mais a marcação do alinhamento.

        O `depth_model_sha256` continua sendo o do checkpoint que produziu a forma — é a
        resposta certa para "de qual peso veio esta profundidade". O que muda é que agora
        há um segundo peso envolvido, e ele entra como campo próprio em vez de substituir
        o primeiro.
        """
        proc = dict(self.alvo.provenance())
        base_proc = self.base.provenance()
        proc["depth_alignment"] = {
            "solicitado": True,
            "aplicado": True,
            "metodo": "minimos_quadrados_afim_por_amostra",
            "dominio": self.dominio,
            "referencia_checkpoint": base_proc.get("depth_checkpoint"),
            "referencia_sha256": base_proc.get("depth_model_sha256"),
            "piso_positivo": (PISO_DISPARIDADE if self.dominio == "disparidade"
                              else PISO_PROFUNDIDADE_M),
            "motivo": ("os pesos afinados não são métricos — o treino alinhava por afim "
                       "dentro da loss, então a escala absoluta nunca entrou no "
                       "gradiente. Sem isto a diferença medida no bokeh seria deriva de "
                       "escala (D_focus muda ~1,7x), não borda melhor."),
        }
        return proc

    # -- inferência ------------------------------------------------------------

    def infer(self, image_rgb: np.ndarray) -> MetricDepth:
        alvo = self.alvo.infer(image_rgb)
        base = self.base.infer(image_rgb)
        if alvo.values_m.shape != base.values_m.shape:
            reject("resolution_invalid",
                   f"o braço devolveu {alvo.values_m.shape} e a referência "
                   f"{base.values_m.shape} na MESMA imagem")

        z_alvo = np.asarray(alvo.values_m, dtype=np.float64)
        z_base = np.asarray(base.values_m, dtype=np.float64)
        validos = (np.isfinite(z_alvo) & (z_alvo > 0)
                   & np.isfinite(z_base) & (z_base > 0))
        if validos.mean() < 0.999:
            reject("depth_non_finite",
                   f"só {validos.mean():.4f} dos pixels são positivos e finitos nos dois "
                   "braços — não há pareamento a ajustar")

        if self.dominio == "disparidade":
            x_todos, y_todos = 1.0 / z_alvo, 1.0 / z_base
            piso = PISO_DISPARIDADE
        else:
            x_todos, y_todos = z_alvo, z_base
            piso = PISO_PROFUNDIDADE_M

        a, b, r2 = ajusta_afim(x_todos[validos], y_todos[validos])
        if not np.isfinite(a) or a <= 0.0:
            # Inclinação <= 0 significa que o braço ordenou a cena ao contrário da base.
            # Isso não é escala a corrigir: é o mapa espelhado do D11, e aplicar a afim
            # produziria uma profundidade plausível e invertida que nenhum gate pega.
            reject("depth_range_degenerate",
                   f"ajuste afim com inclinação a={a:.6g} <= 0 no domínio "
                   f"{self.dominio!r}: o braço inverteu perto e longe em relação à base. "
                   "Rejeita em vez de gravar um mapa espelhado — ver o defeito D11.")

        ajustado = a * x_todos + b
        no_piso = ~(ajustado > piso)
        fracao_no_piso = float(np.mean(no_piso))
        ajustado = np.where(no_piso, piso, ajustado)

        z = (1.0 / ajustado) if self.dominio == "disparidade" else ajustado

        # Diagnóstico pareado, na grandeza que o renderizador consome. Não é rótulo:
        # é o número que responde "de quanto foi a correção de escala nesta imagem".
        disp_antes = float(np.median(1.0 / z_alvo[validos]))
        disp_depois = float(np.median(1.0 / z[validos]))
        razao = (disp_depois / disp_antes) if disp_antes > 0 else None

        self.ajustes.append(AjusteAfim(a=a, b=b, r2=r2, dominio=self.dominio,
                                       fracao_no_piso=fracao_no_piso,
                                       razao_mediana_disparidade=razao))
        self.calls += 1
        # Revalida: o alinhamento é uma transformação, e transformação passa pelo mesmo
        # portão que a saída crua do modelo. `backend` vem do braço alvo porque é dele
        # que a forma veio.
        return validate_metric_depth(z.astype(np.float32), backend=alvo.backend)


def marca_sem_alinhamento(motivo: str, *, solicitado: bool) -> dict:
    """A marcação para quem NÃO alinhou. Existe para que a ausência seja legível.

    `solicitado=False` (a flag não foi passada) e `solicitado=True, aplicado=False` (a
    flag foi passada mas o braço É a referência) são situações diferentes, e um campo
    booleano só não as separa. Um artefato em que as duas aparecem como "sem
    alinhamento" não diz se o experimento testou a hipótese ou se esqueceu a flag.
    """
    return {"solicitado": bool(solicitado), "aplicado": False, "motivo": motivo}
