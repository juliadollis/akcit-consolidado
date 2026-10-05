"""Teto de níveis por cena — reproduz o tamanho de dataset que o paper publica.

## O número do paper, e como ele fecha

Suplemento B.2, `paper.txt:998-1006`:

    "we utilize a comprehensive dataset of 26K real bokeh images. This collection
    comprises 13K previously filtered and verified images from the ITW dataset [19],
    alongside **13K images newly curated for this work**. The newly collected data
    consists of focus-consistent series captured with varying apertures, **containing
    2 to 4 images per set**."

E, logo abaixo, a pista que confirma a leitura:

    "This annotation step required **4 to 8 seconds per image**, amounting to
    approximately **8 hours** of manual effort in total."

Duas contas independentes fecham `[I do M]`:

* **As máscaras.** 8 h = 28.800 s; a 4–8 s por imagem dá **3.600 a 7.200** anotações. A
  RealBokeh tem **4.399 cenas**, e a máscara é uma por cena — a AIF é a mesma em todos
  os níveis. Cai no meio da faixa.
* **O tamanho.** Com teto de 4 sobre o histograma medido do split `train`
  (1→2, 2→705, 3→621, 4→1, 5→2341, 6→1, 7→9, 9→34, 12→1, 21→244 cenas):

      1·2 + 2·705 + 3·621 + 4·1 + 4·2341 + 4·1 + 4·9 + 4·34 + 4·1 + 4·244 = **13.799**

  ≈ "13K", com a folga certa para o limiar de SSIM que eles aplicam depois.

**A comparação é contra o split `train`, e isso é essencial.** O enumerador percorre os
três splits do espelho, e o total com teto 4 é **15.427** — mas `test` (1.257 linhas) e
`validation` (1.238) são **retidos**, não treinados. Os "13K" do paper descrevem dado de
treino: *"we utilize a comprehensive dataset of 26K real bokeh images ... alongside 13K
images newly curated for this work"*, na seção de **treino** da BokehNet. Comparar 15.427
contra 13K mistura treino com conjunto retido.

Os quatro números, medidos:

| teto | treino | vs 13K | total (3 splits) | retido |
|---|---|---|---|---|
| nenhum | 20.495 | +57,7% | 22.990 | 2.495 |
| **4** | **13.799** | **+6,1%** | 15.427 | 1.628 |
| 3 | 11.168 | −14,1% | 12.439 | 1.271 |

## O que este módulo faz, e o que ele não decide

Ele corta níveis, não cenas: **toda cena continua representada**. Uma cena com 21
aberturas contribui com 4 em vez de 21; uma cena com 2 contribui com as 2.

Isso importa porque a distribuição atual é desequilibrada de um jeito que a contagem de
amostras esconde: **244 cenas (6,2%) geravam 5.124 amostras (25%)**. Um quarto do sinal
de treino vinha de 6,2% do conteúdo, repetido 21 vezes. O teto não joga cena fora — ele
tira o peso excessivo de umas poucas.

## `[A]` — quais níveis, quando sobram

O paper diz *quantos* e não diz *quais*. Escolhemos **espaçados uniformemente** pela
lista de níveis da cena, sempre incluindo os extremos.

A alternativa óbvia — pegar as aberturas mais abertas, que é onde o bokeh é mais forte —
foi rejeitada: enviesaria K para cima e daria à rede menos variedade de borrão **dentro
da mesma cena**, que é exatamente o sinal que ela precisa aprender. Com o mesmo conteúdo,
mesma profundidade e mesmo plano de foco, a única coisa que varia entre níveis é K; jogar
fora essa amplitude é jogar fora a supervisão de K.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

#: O teto do paper. Ver o cabeçalho para a aritmética que o sustenta.
PAPER_MAX_LEVELS_PER_SCENE = 4


def _uniform_indices(total: int, keep: int) -> list[int]:
    """`keep` índices espaçados por igual em `range(total)`, com os extremos incluídos.

    `total=21, keep=4` -> `[0, 6, 13, 20]`. Determinístico e sem sorteio: a mesma cena
    dá sempre os mesmos níveis, então o release é reproduzível e o `sample_id` de uma
    amostra selecionada não muda entre runs.
    """
    if keep >= total:
        return list(range(total))
    if keep == 1:
        return [0]
    passo = (total - 1) / (keep - 1)
    return sorted({int(round(i * passo)) for i in range(keep)})


def cap_levels_per_scene(pairs: Iterable, *, max_levels: int = PAPER_MAX_LEVELS_PER_SCENE
                         ) -> list:
    """Mantém no máximo `max_levels` níveis por cena, espaçados uniformemente.

    Preserva a ordem de entrada dos pares que sobrevivem — quem chamou já pode ter
    ordenado para leitura sequencial de parquet, e reordenar aqui desfaria isso.

    Não rejeita nem registra motivo: um nível não selecionado **não é uma amostra
    recusada**, é uma amostra que o desenho do dataset não pediu. Misturá-la ao
    histograma de rejeição poluiria justamente o instrumento que denuncia defeito.
    A contagem do que foi cortado sai em `cap_summary`.
    """
    if max_levels < 1:
        raise ValueError(f"max_levels tem que ser >= 1: {max_levels}")

    pares = list(pairs)
    por_cena: dict[str, list] = {}
    for par in pares:
        por_cena.setdefault(par.scene_id, []).append(par)

    mantidos: set[int] = set()
    for cena, do_grupo in por_cena.items():
        # ordena por nível para que "espaçado uniformemente" seja pela abertura, e não
        # pela ordem em que os pares apareceram no shard
        ordenados = sorted(do_grupo, key=lambda p: p.level)
        for i in _uniform_indices(len(ordenados), max_levels):
            mantidos.add(id(ordenados[i]))

    return [p for p in pares if id(p) in mantidos]


def cap_summary(before: Sequence, after: Sequence, *, max_levels: int) -> str:
    """O que o teto fez, em cenas E em amostras.

    Em cenas **e** em amostras porque "13.799 amostras" sozinho esconde de quantas
    unidades independentes elas vêm — a regra do projeto, e o que tornava "20.554
    amostras" uma afirmação enganosa no release antigo.
    """
    cenas_antes = {p.scene_id for p in before}
    cenas_depois = {p.scene_id for p in after}
    niveis_antes = Counter(len([q for q in before if q.scene_id == c]) for c in cenas_antes)

    linhas = [
        "",
        "=" * 62,
        f"  teto de níveis por cena: {max_levels}  (paper.txt:1001-1003)",
        f"  amostras : {len(before):>6}  ->  {len(after):>6}   "
        f"({100 * len(after) / max(len(before), 1):.1f}%)",
        f"  cenas    : {len(cenas_antes):>6}  ->  {len(cenas_depois):>6}   "
        f"(nenhuma cena é descartada pelo teto)",
    ]
    acima = sum(n for k, n in niveis_antes.items() if k > max_levels)
    if acima:
        amostras_dessas = sum(k * n for k, n in niveis_antes.items() if k > max_levels)
        linhas.append(
            f"  cenas acima do teto: {acima} ({100 * acima / max(len(cenas_antes), 1):.1f}% "
            f"das cenas) geravam {amostras_dessas} amostras "
            f"({100 * amostras_dessas / max(len(before), 1):.1f}%)")
    linhas.append("=" * 62)
    return "\n".join(linhas)
