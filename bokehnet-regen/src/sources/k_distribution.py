"""A distribuição de onde a rota A sorteia K — lida de um JSON, nunca de constante.

A rota A **não tem equação para K**. O paper diz só *"we randomly sample a focus plane
D_focus and a target bokeh level K"* (`paper.txt:329-330`) e **cala sobre a
distribuição**. Amostrar da distribuição empírica das rotas B e C
(`k_source = "sampled_from_bc"`) é decisão nossa, já registrada em `CONTRATO.md:163-166`
e continua `[A]`.

Este módulo é a **fonte** dessa distribuição, no mesmo sentido em que `realbokeh.py` é a
fonte dos pares: ele enumera o que existe, afirma o que a origem afirma, valida, e não
produz rótulo nenhum. Quem sorteia é `routes/route_a.py`; quem produz o JSON é
`scripts/build_k_distribution.py`, lendo `manifest.jsonl` de um ou mais diretórios de
release.

## A grandeza é `k_per_long_side_px`, e isso é obrigatório — não estilo

O K vive **em pixel**. `K = k_eq3/1000` e `k_eq3 ∝ pixel_ratio = max(H,W)/sensor_mm`
(`CONTRATO.md:22-23`, `control/contract.py:356-371`), logo **K escala linearmente com a
resolução da imagem**. O `pixel_ratio` medido na rota B vai de 22,2 a 277,3, mediana 42,6
(`ACHADOS.md:36` `[M]`) — uma faixa de 12x. Transportar o K cru de uma imagem da rota B
para uma imagem da rota A, que tem outra resolução, aplica um CoC em pixel que não
corresponde a óptica nenhuma. É o defeito A5 da auditoria, e é a terceira aparição do
mesmo erro de convenção de escala neste projeto.

Então a distribuição é armazenada na grandeza **livre de resolução**

    k_per_long_side = k_value / max(image_h, image_w)          [1/px · px = adimensional]

e a rota A remultiplica pelo `max(H, W)` da imagem dela. A regra 3 do contrato —
*"toda quantidade em pixel carrega a resolução em que foi medida"* (`CONTRATO.md:47-50`)
— é o que obriga isso, e o teste que a trava é: a mesma imagem em duas resoluções tem
que dar o mesmo CoC **relativo**.

Isso é possível hoje porque a linha do manifesto **já carrega** `image_h` e `image_w`
(`dataio/writer.py:150`). O bloqueador que a auditoria registrava em F3 ("a linha do
manifesto não carrega image_h/image_w") **não existe mais** — foi resolvido quando o
writer passou a gravar a resolução ao lado do K.

## O alargamento de 25% para cada lado

A rota B vai ser **regerada** com a nossa DeblurNet (`PLANO_EXECUCAO.md`, passo 5), e a
hipótese declarada do plano é que a distribuição de K não muda muito entre as duas
variantes — *"se não vale, a rota A precisa ser regerada e o pré-treino refeito"*. Se o K
se mover um pouco, sortear exatamente `[p01, p99]` observado deixaria a cobertura no
limite: a faixa nova cairia fora do que o pré-treino viu.

Então o sorteio alarga a faixa em `widen_fraction` **para cada lado**, esticando o desvio
em relação à **mediana**:

    v' = p50 + (1 + f)·(v − p50)        # f = 0,25

o que leva `p01 ↦ p50 − 1,25·(p50 − p01)` e `p99 ↦ p50 + 1,25·(p99 − p50)` — cada lado
25% mais longe — e **deixa a mediana onde ela estava**. A alternativa de esticar em torno
do centro de `[p01, p99]` foi medida e recusada: numa distribuição assimétrica à direita
ela desloca a mediana em −37,7%. Ver `KDistribution.widen`. `[A]`: a fração 0,25 é
escolha nossa, e a razão é a regeração da rota B, não medição.

**Positividade.** K tem que ser > 0 (`control/contract.py:464-465`). Se
`c − s·(c − p01)` cai em zero ou abaixo, o valor é preso num piso **derivado da própria
distribuição** — `min_observado · (1 − widen_fraction)` — e o evento é **contado** por
amostra e no resumo do run. Nem clamp silencioso, nem rejeição em massa da cauda baixa:
truncar a cauda sem contar seria uma seleção invisível.

## Recusar de plano uma distribuição degenerada

Este módulo tem `degenerate_reason()`, e existe por um fato medido: a rota B publicada
hoje tem `k = 50,0` em 11.635/11.635 amostras (`ACHADOS.md:14` `[M]`) e a rota C tem
47,0% no teto exato de 300 (`ACHADOS.md:19` `[M]`). **Amostrar dessa "distribuição" hoje
é amostrar de duas constantes** — a auditoria chama isso de pré-requisito duro, não de
preferência: a rota A não pode rodar antes de B e C regeradas.

O guarda é explícito, os limites são constantes nomeadas e `[A]`, e o entrypoint só passa
por cima com `--allow-degenerate-k-distribution` e um aviso em voz alta. Um pipeline que
aceita em silêncio uma distribuição de um valor só produz 70 mil amostras em que K é
confundido com conteúdo, e isso não aparece como erro em lugar nenhum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

#: Nome do schema do JSON. Versionado: um consumidor que não reconheça o nome tem que
#: parar, não adivinhar o layout.
SCHEMA = "bokehnet_k_distribution_v1"

#: A grandeza armazenada. **Não é `k_value`** — ver o cabeçalho do módulo.
QUANTITY = "k_per_long_side_px"

#: Fração de alargamento da faixa, para cada lado. `[A]`, motivada pela regeração da
#: rota B (`PLANO_EXECUCAO.md`). Não vem do paper, que cala sobre a distribuição.
DEFAULT_WIDEN_FRACTION = 0.25

#: Quantos nós de quantil o JSON carrega. 101 = passo de 1%, que resolve p01 e p99 sem
#: interpolar entre extremos distantes.
DEFAULT_QUANTILE_KNOTS = 101

# --- guardas de degeneração, todos `[A]` -----------------------------------------
#: Menos valores distintos que isto e a "distribuição" é um punhado de constantes.
MIN_DISTINCT_VALUES = 32
#: Por rota contribuinte. A rota B publicada tem **1** (`ACHADOS.md:14`).
MIN_DISTINCT_PER_ROUTE = 8
#: Razão mínima p99/p01. Abaixo disto o sorteio não produz variação de borrão.
MIN_P99_OVER_P01 = 1.05
#: Fração máxima de amostras censuradas numa rota contribuinte. A rota C publicada tem
#: 47,0% no teto exato (`ACHADOS.md:19`) — censura alta significa que o teto da busca
#: estava errado, e a distribuição sobrevivente é a cauda do teto, não a do fenômeno.
MAX_CENSORED_SHARE = 0.25


# --------------------------------------------------------------------------------
# Um sorteio
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class SampledK:
    """Um valor sorteado, com tudo que permite refazê-lo.

    `raw_per_long_side` é o quantil da distribuição **observada**;
    `per_long_side` é ele depois do alargamento. Os dois vão para a proveniência da
    amostra: sem o primeiro não dá para responder "este K existia nos dados ou saiu do
    alargamento?", que é justamente a pergunta que o alargamento cria.
    """

    per_long_side: float
    raw_per_long_side: float
    quantile: float
    widen_fraction: float
    clamped_to_floor: bool

    def at_long_side(self, long_side: int) -> float:
        """K na escala de pixel de uma imagem com este lado longo.

        É a **única** porta de volta para a convenção do contrato. Multiplicar por
        `max(H, W)` em qualquer outro lugar criaria uma segunda definição de escala,
        que é como o projeto chegou a quatro interpretações de K.
        """
        if int(long_side) <= 0:
            raise ValueError(f"long_side inválido: {long_side!r}")
        return float(self.per_long_side) * float(long_side)

    def to_metadata(self) -> dict:
        return {
            "k_per_long_side": float(self.per_long_side),
            "k_per_long_side_observed": float(self.raw_per_long_side),
            "k_quantile": float(self.quantile),
            "k_widen_fraction": float(self.widen_fraction),
            "k_clamped_to_positive_floor": bool(self.clamped_to_floor),
        }


# --------------------------------------------------------------------------------
# A distribuição
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class KDistribution:
    """Função quantil empírica de `k_per_long_side_px`, com a proveniência de origem.

    Guardada como nós de quantil, e não como a amostra inteira: 70 mil linhas de K não
    precisam viajar num arquivo de configuração, e a função quantil é tudo que o sorteio
    consome. O custo declarado é a interpolação linear entre nós — com 101 nós, ela é
    menor que a largura de um passo de 1% da distribuição.
    """

    quantile_q: np.ndarray
    quantile_value: np.ndarray
    n: int
    control_version: str
    created_utc: str
    #: Um dict por diretório de release lido, com contagens de exclusão e sha256 do
    #: manifesto. É o que prova de onde a distribuição veio.
    sources: tuple[dict, ...] = ()
    #: Por rota contribuinte: n, distintos, censuradas, percentis. Alimenta
    #: `degenerate_reason`.
    per_route: dict = None                     # type: ignore[assignment]
    schema: str = SCHEMA
    quantity: str = QUANTITY

    # -- construção -------------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantile_q",
                           np.asarray(self.quantile_q, dtype=np.float64))
        object.__setattr__(self, "quantile_value",
                           np.asarray(self.quantile_value, dtype=np.float64))
        object.__setattr__(self, "per_route", dict(self.per_route or {}))
        self._validate()

    def _validate(self) -> None:
        """Falha alto. Um JSON de distribuição inválido não é caso a resolver em runtime.

        `ValueError`, e não `SampleRejected`: isto é **erro de configuração** do run, no
        mesmo molde de "fonte fora do conjunto" da rota C. Uma distribuição quebrada não
        é uma amostra ruim — ela envenena as 70 mil — e não pode aparecer no histograma
        de rejeição como se fosse um caso a calibrar.
        """
        if self.schema != SCHEMA:
            raise ValueError(f"schema desconhecido: {self.schema!r}; esperado {SCHEMA!r}")
        if self.quantity != QUANTITY:
            raise ValueError(
                f"grandeza {self.quantity!r} != {QUANTITY!r}. K cru, sem a resolução em "
                "que foi medido, é o defeito A5 — ver o cabeçalho deste módulo.")
        q, v = self.quantile_q, self.quantile_value
        if q.ndim != 1 or v.ndim != 1 or q.size != v.size or q.size < 2:
            raise ValueError(f"nós de quantil inválidos: q={q.shape}, value={v.shape}")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
            raise ValueError("nó de quantil não-finito")
        if q[0] != 0.0 or q[-1] != 1.0:
            raise ValueError(f"os nós têm que cobrir [0, 1]; vieram [{q[0]}, {q[-1]}]")
        if np.any(np.diff(q) <= 0):
            raise ValueError("`quantile_q` tem que ser estritamente crescente")
        if np.any(np.diff(v) < 0):
            raise ValueError("`quantile_value` tem que ser não-decrescente")
        if v[0] <= 0:
            raise ValueError(
                f"k_per_long_side mínimo {v[0]!r} <= 0. K não-positivo não é nível de "
                "bokeh — `signed_coc_px` rejeita com `k_non_positive`.")
        if int(self.n) <= 0:
            raise ValueError(f"n inválido: {self.n!r}")

    # -- leitura ---------------------------------------------------------------

    @property
    def minimum(self) -> float:
        return float(self.quantile_value[0])

    @property
    def maximum(self) -> float:
        return float(self.quantile_value[-1])

    def quantile(self, u: float) -> float:
        """Quantil `u` da distribuição **observada**, sem alargamento."""
        if not np.isfinite(u) or not 0.0 <= float(u) <= 1.0:
            raise ValueError(f"quantil fora de [0, 1]: {u!r}")
        return float(np.interp(float(u), self.quantile_q, self.quantile_value))

    @property
    def p01(self) -> float:
        return self.quantile(0.01)

    @property
    def p50(self) -> float:
        return self.quantile(0.50)

    @property
    def p99(self) -> float:
        return self.quantile(0.99)

    def distinct_values(self) -> int:
        return int(np.unique(self.quantile_value).size)

    # -- alargamento ------------------------------------------------------------

    def positive_floor(self, widen_fraction: float = DEFAULT_WIDEN_FRACTION) -> float:
        """Piso positivo do sorteio, derivado da própria distribuição.

        Não é constante mágica: é `min_observado · (1 − f)`. Fica **abaixo** de tudo que
        foi observado, portanto amplia a cobertura para baixo, e é estritamente positivo
        porque `min_observado > 0` (garantido em `_validate`) e `f < 1`.

        Ele só entra em ação quando o alargamento levaria K a zero ou abaixo — o que
        acontece quando a mediana é maior que `(1+f)/f` vezes o desvio até o mínimo. O
        evento é **contado** por amostra (`k_clamped_to_positive_floor`) e no resumo do
        run: truncar a cauda baixa em silêncio seria uma seleção invisível.
        """
        if not 0.0 <= float(widen_fraction) < 1.0:
            raise ValueError(f"widen_fraction fora de [0, 1): {widen_fraction!r}")
        return self.minimum * (1.0 - float(widen_fraction))

    def widened_support(
        self, widen_fraction: float = DEFAULT_WIDEN_FRACTION,
    ) -> tuple[float, float]:
        """`(lo, hi)` do suporte alargado, já com o piso positivo aplicado no lo."""
        return (max(self.widen(self.p01, widen_fraction),
                    self.positive_floor(widen_fraction)),
                self.widen(self.p99, widen_fraction))

    def widen(self, value: float, widen_fraction: float = DEFAULT_WIDEN_FRACTION) -> float:
        """Estica o desvio em relação à MEDIANA por `1 + f`. A mediana fica parada.

            v' = p50 + (1 + f)·(v − p50)

        Com `f = 0,25`, cada lado da faixa observada `[p01, p99]` se estende 25% em
        torno da mediana: `p01 ↦ p50 − 1,25·(p50 − p01)` e
        `p99 ↦ p50 + 1,25·(p99 − p50)`.

        **A alternativa foi recusada por medição.** O mapa afim em torno do centro de
        `[p01, p99]` (`v' = c + (1+2f)·(v − c)`) também produz a faixa alargada pedida,
        mas numa distribuição assimétrica à direita — que é o que uma razão de óptica
        produz — ele **desloca a mediana**. Medido numa lognormal com σ = 0,5 e 5.000
        pontos: a mediana cai de 0,019725 para 0,012286, **−37,7%**, e o p25 de 0,014201
        para 0,004000, **−71,8%**. A rota A passaria a ter um K típico que não é o K típico do real,
        contra a razão de existir do `sampled_from_bc` — *"existe para manter o sintético
        na mesma escala física do real"* (`CONTRATO.md:163-166`).

        Ancorar na mediana preserva a localização e a forma relativa, e mesmo assim
        alarga os dois extremos. É monótono em `v`, então continua sendo uma função
        quantil válida. `[A]`: nada disto vem do paper, que cala sobre a distribuição.
        """
        if not 0.0 <= float(widen_fraction) < 1.0:
            raise ValueError(f"widen_fraction fora de [0, 1): {widen_fraction!r}")
        centro = self.p50
        return float(centro + (1.0 + float(widen_fraction)) * (float(value) - centro))

    def sample(
        self, u: float, *, widen_fraction: float = DEFAULT_WIDEN_FRACTION,
    ) -> SampledK:
        """Sorteia pelo quantil `u`, alarga, e devolve os dois valores.

        `u` vem de fora — a rota A o produz de forma determinística a partir do
        `scene_id` e do índice da variante, e é por isso que este módulo não tem RNG
        nenhum: uma semente aqui seria um segundo lugar de onde a aleatoriedade pode
        vir, e o release deixaria de ser reproduzível por `sample_id`.
        """
        bruto = self.quantile(u)
        alargado = self.widen(bruto, widen_fraction)
        piso = self.positive_floor(widen_fraction)
        preso = alargado < piso
        if preso:
            alargado = piso
        return SampledK(per_long_side=float(alargado), raw_per_long_side=float(bruto),
                        quantile=float(u), widen_fraction=float(widen_fraction),
                        clamped_to_floor=bool(preso))

    # -- o guarda ---------------------------------------------------------------

    def degenerate_reason(self) -> Optional[str]:
        """Por que esta distribuição não serve para sortear — ou `None` se serve.

        Cada ramo corresponde a um fato medido no dado publicado hoje. Ver o cabeçalho
        do módulo: a rota B com `k = 50,0` em 11.635/11.635 (`ACHADOS.md:14`) e a rota C
        com 47,0% no teto exato (`ACHADOS.md:19`).
        """
        if self.n < MIN_DISTINCT_VALUES:
            return (f"n = {self.n} amostras, abaixo de {MIN_DISTINCT_VALUES}: não é "
                    "distribuição, é um punhado de pontos")
        distintos = self.distinct_values()
        if distintos < MIN_DISTINCT_VALUES:
            return (f"{distintos} valores distintos de {QUANTITY}, abaixo de "
                    f"{MIN_DISTINCT_VALUES}. Sortear daqui é sortear de constantes — "
                    "é o caso da rota B publicada, com k = 50,0 em 11.635/11.635 "
                    "(ACHADOS.md:14).")
        p01, p99 = self.p01, self.p99
        if p01 <= 0 or p99 / p01 < MIN_P99_OVER_P01:
            return (f"p99/p01 = {p99 / p01:.4f} < {MIN_P99_OVER_P01}: a faixa é "
                    "praticamente constante e o sorteio não produz variação de borrão")
        for rota, info in sorted(self.per_route.items()):
            d = int(info.get("distinct", 0))
            if d < MIN_DISTINCT_PER_ROUTE:
                return (f"rota {rota!r} contribui com {d} valores distintos, abaixo de "
                        f"{MIN_DISTINCT_PER_ROUTE}. Uma rota constante dentro da mistura "
                        "não deixa de ser constante.")
            censuradas = info.get("censored_share")
            if censuradas is not None and float(censuradas) > MAX_CENSORED_SHARE:
                return (f"rota {rota!r} tem {100 * float(censuradas):.1f}% de amostras "
                        f"censuradas, acima de {100 * MAX_CENSORED_SHARE:.0f}%. O teto da "
                        "busca estava errado (a rota C publicada tem 47,0% no teto exato "
                        "de 300, ACHADOS.md:19), então a cauda alta desta distribuição é "
                        "a do teto, não a do fenômeno.")
        return None

    # -- serialização -----------------------------------------------------------

    def to_dict(self) -> dict:
        lo, hi = self.widened_support(DEFAULT_WIDEN_FRACTION)
        return {
            "schema": self.schema,
            "quantity": self.quantity,
            "quantity_definition": (
                "k_value / max(image_h, image_w) — K por pixel de lado longo, livre de "
                "resolução. Ver src/sources/k_distribution.py e CONTRATO.md:47-50."),
            "control_version": self.control_version,
            "created_utc": self.created_utc,
            "n": int(self.n),
            "distinct_values": self.distinct_values(),
            "widen_fraction_recommended": DEFAULT_WIDEN_FRACTION,
            "stats": {
                "min": self.minimum, "p01": self.p01, "p50": self.p50,
                "p99": self.p99, "max": self.maximum,
                "widened_support_min": lo, "widened_support_max": hi,
            },
            "quantiles": {"q": [float(x) for x in self.quantile_q],
                          "value": [float(x) for x in self.quantile_value]},
            "per_route": self.per_route,
            "sources": list(self.sources),
            "degenerate_reason": self.degenerate_reason(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "KDistribution":
        faltando = [k for k in ("schema", "quantity", "quantiles", "n",
                                "control_version") if k not in data]
        if faltando:
            raise ValueError(f"JSON de distribuição incompleto, faltam: {faltando}")
        nos = data["quantiles"]
        if "q" not in nos or "value" not in nos:
            raise ValueError("`quantiles` precisa de `q` e `value`")
        return cls(
            quantile_q=np.asarray(nos["q"], dtype=np.float64),
            quantile_value=np.asarray(nos["value"], dtype=np.float64),
            n=int(data["n"]),
            control_version=str(data["control_version"]),
            created_utc=str(data.get("created_utc", "")),
            sources=tuple(data.get("sources") or ()),
            per_route=dict(data.get("per_route") or {}),
            schema=str(data["schema"]),
            quantity=str(data["quantity"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "KDistribution":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- resumo -----------------------------------------------------------------

    def summary(self, widen_fraction: float = DEFAULT_WIDEN_FRACTION) -> str:
        lo, hi = self.widened_support(widen_fraction)
        linhas = [
            "",
            "=" * 62,
            f"  distribuição de K  : {self.quantity}  (schema {self.schema})",
            f"  contrato           : {self.control_version}",
            f"  n                  : {self.n}  ({self.distinct_values()} valores "
            f"distintos nos nós)",
            f"  observado          : p01 {self.p01:.6f}  mediana {self.p50:.6f}  "
            f"p99 {self.p99:.6f}",
            f"  suporte alargado   : [{lo:.6f}, {hi:.6f}]  "
            f"(+{100 * widen_fraction:.0f}% de cada lado)",
            "    o alargamento existe porque a rota B vai ser REGERADA com outra "
            "DeblurNet",
            "    (PLANO_EXECUCAO.md). Sortear [p01,p99] exato deixaria a cobertura no "
            "limite.",
        ]
        for rota, info in sorted(self.per_route.items()):
            linhas.append(
                f"  rota {rota:<3}          : n={info.get('n')}  "
                f"distintos={info.get('distinct')}  "
                f"censuradas={info.get('censored_share')}")
        motivo = self.degenerate_reason()
        linhas.append("-" * 62)
        if motivo:
            linhas.append(f"  >>> DEGENERADA: {motivo}")
            linhas.append("      A rota A NÃO pode rodar com esta distribuição sem "
                          "--allow-degenerate-k-distribution.")
        else:
            linhas.append("  distribuição utilizável pelos guardas de degeneração.")
        linhas.append("=" * 62)
        return "\n".join(linhas)


# --------------------------------------------------------------------------------
# Construção a partir dos manifestos das rotas B e C
# --------------------------------------------------------------------------------

@dataclass
class ManifestScan:
    """O que uma varredura de manifesto encontrou, **e o que ela jogou fora**.

    Os contadores de exclusão não são decoração: sem eles, uma distribuição montada de
    3% das linhas pareceria idêntica a uma montada de 100%.
    """

    release_dir: str
    lines_total: int = 0
    lines_used: int = 0
    excluded_not_valid_for_control: int = 0
    excluded_censored: int = 0
    excluded_missing_resolution: int = 0
    excluded_k_invalid: int = 0
    routes: dict = None                        # type: ignore[assignment]
    values: list = None                        # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.routes = dict(self.routes or {})
        self.values = list(self.values or [])

    def to_dict(self) -> dict:
        return {
            "release_dir": self.release_dir,
            "lines_total": self.lines_total,
            "lines_used": self.lines_used,
            "excluded": {
                "not_valid_for_control": self.excluded_not_valid_for_control,
                "is_k_censored": self.excluded_censored,
                "missing_image_hw": self.excluded_missing_resolution,
                "k_value_invalid": self.excluded_k_invalid,
            },
            "routes": self.routes,
        }


def scan_manifest_rows(rows: Iterable[dict], *, release_dir: str) -> ManifestScan:
    """Converte linhas de manifesto em `k_per_long_side`, contando cada exclusão.

    Três exclusões, e cada uma tem um motivo escrito:

    * **`is_valid_for_control == False`** — a amostra não serve de rótulo, então o K dela
      não descreve o fenômeno.
    * **`is_k_censored == True`** — o K bateu no teto da busca da Eq. 5. Medido: 47,0% da
      rota C publicada está no teto exato de 300 (`ACHADOS.md:19`). Um valor de teto é
      uma borda de configuração, não uma medida; incluí-lo criaria uma moda artificial
      exatamente no limite.
    * **resolução ausente** — sem `image_h`/`image_w` não há como tirar a resolução do K,
      e um K sem resolução é o defeito A5. Excluir é a única saída honesta: dividir por
      um lado longo assumido seria inventar a escala.

    A contagem por rota é feita **antes** das exclusões (`seen`), e a censura é contada
    junto. Isso é obrigatório e não detalhe: se `censored_share` fosse calculado só sobre
    as linhas SOBREVIVENTES, ele seria 0,0 por construção — e o guarda que existe para
    pegar os 47,0% de censura da rota C (`ACHADOS.md:19`) nunca dispararia. Um guarda que
    não pode disparar é pior que guarda nenhum.
    """
    scan = ManifestScan(release_dir=release_dir)
    for row in rows:
        scan.lines_total += 1
        rota = str(row.get("route", "?"))
        info = scan.routes.setdefault(
            rota, {"values": [], "censored": 0, "seen": 0, "total": 0})
        info["seen"] += 1
        if row.get("is_k_censored", False):
            info["censored"] += 1

        if not row.get("is_valid_for_control", False):
            scan.excluded_not_valid_for_control += 1
            continue
        if row.get("is_k_censored", False):
            scan.excluded_censored += 1
            continue
        h, w = row.get("image_h"), row.get("image_w")
        if not h or not w or int(h) <= 0 or int(w) <= 0:
            scan.excluded_missing_resolution += 1
            continue
        k = row.get("k_value")
        if k is None or not np.isfinite(k) or float(k) <= 0:
            scan.excluded_k_invalid += 1
            continue

        info["values"].append(float(k) / float(max(int(h), int(w))))
        scan.values.append(info["values"][-1])
        scan.lines_used += 1

    for info in scan.routes.values():
        info["total"] = len(info["values"])
    return scan


def build_distribution(
    scans: Iterable[ManifestScan], *, control_version: str,
    knots: int = DEFAULT_QUANTILE_KNOTS, created_utc: str = "",
) -> KDistribution:
    """Junta as varreduras numa função quantil. Sem ponderação entre rotas.

    **Sem ponderação, e isso é decisão declarada**: cada amostra de B e de C pesa igual.
    A alternativa — equalizar as duas rotas — mudaria a distribuição em favor da rota
    menor, e nada no paper autoriza isso. O que a mistura pesa está visível em
    `per_route`, e a rota A grava por amostra o quantil sorteado, então dá para refazer
    com outra ponderação sem reprocessar imagem.
    """
    scans = list(scans)
    valores = np.asarray([v for s in scans for v in s.values], dtype=np.float64)
    if valores.size == 0:
        raise ValueError(
            "nenhuma linha de manifesto sobreviveu aos filtros. Veja os contadores de "
            "exclusão: uma distribuição vazia é erro de entrada, não caso a resolver.")
    q = np.linspace(0.0, 1.0, int(knots))
    v = np.quantile(valores, q)
    # `np.quantile` é monótona por construção, mas ponto flutuante pode devolver um
    # passo de -1e-18 entre nós iguais. `maximum.accumulate` fixa a monotonicidade sem
    # mexer em valor nenhum de forma perceptível — e `_validate` exige monotonicidade.
    v = np.maximum.accumulate(v)

    por_rota: dict = {}
    for s in scans:
        for rota, info in s.routes.items():
            alvo = por_rota.setdefault(rota, {"values": []})
            alvo["values"].extend(info["values"])
    censura_por_rota = _censored_share_per_route(scans)
    for rota, info in por_rota.items():
        vals = np.asarray(info.pop("values"), dtype=np.float64)
        info["n"] = int(vals.size)
        info["distinct"] = int(np.unique(vals).size)
        info["p01"] = float(np.quantile(vals, 0.01))
        info["p50"] = float(np.quantile(vals, 0.50))
        info["p99"] = float(np.quantile(vals, 0.99))
        info["censored_share"] = censura_por_rota.get(rota)

    return KDistribution(
        quantile_q=q, quantile_value=v, n=int(valores.size),
        control_version=control_version, created_utc=created_utc,
        sources=tuple(s.to_dict() for s in scans), per_route=por_rota,
    )


def _censored_share_per_route(scans: Iterable[ManifestScan]) -> dict:
    """Fração censurada por rota, sobre TODAS as linhas vistas daquela rota.

    O denominador é `seen` — as linhas do manifesto —, não as sobreviventes. Ver a nota
    em `scan_manifest_rows`: com o denominador errado esta fração é 0,0 sempre.

    Devolve `None` para a rota em que o número não existe, em vez de 0,0: *"não medimos
    a censura desta rota"* e *"esta rota não tem amostra censurada"* são afirmações
    diferentes, e a segunda é a que autoriza a rota A a rodar.
    """
    agregado: dict = {}
    for scan in scans:
        for rota, info in scan.routes.items():
            if "censored" not in info or "seen" not in info:
                continue
            alvo = agregado.setdefault(rota, {"censored": 0, "seen": 0})
            alvo["censored"] += int(info.get("censored", 0))
            alvo["seen"] += int(info.get("seen", 0))
    return {rota: (v["censored"] / v["seen"] if v["seen"] else None)
            for rota, v in agregado.items()}
