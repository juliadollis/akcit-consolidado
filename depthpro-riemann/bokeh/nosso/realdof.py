"""Adaptador de fonte do RealDOF — a terceira origem de pares reais da rota C.

Irmão de `sources/realbokeh.py` e `sources/lfdof.py`, e escrito para responder UMA
pergunta experimental: a profundidade dos nossos Depth Pro afinados produz bokeh melhor
que a do Depth Pro base, medido contra fotografia real? O RealDOF é o conjunto certo
para isso por ser pequeno (50 pares) e inteiramente fotográfico — cada `image_blur` é
uma foto tirada com diafragma aberto, não um desfoque sintetizado.

Uma origem só, já pareada:

| origem | o que traz | acesso |
|---|---|---|
| `akcit-pixel/RealDOF` (privado) | `image_focus` (AIF), `image_blur` (alvo real) e `file_name_base` | snapshot JÁ EM DISCO |

O snapshot está em `hf-cache-julia/hub/datasets--akcit-pixel--RealDOF/snapshots/<rev>/`.
**Não há download a fazer**, e isso é regra e não conveniência.

## A estrutura REAL, medida em 2026-09-18

Lendo os 4 shards parquet do snapshot local:

```
shards                     : 4  (validation-0000{0..3}-of-00004)
linhas                     : 50 = 13 + 13 + 12 + 12
splits publicados          : 1  — `validation`, e só
colunas                    : image_pre_deblur, image_blur, image_focus, file_name_base
modo/formato das imagens   : RGB / PNG em 100/100 células (sem alpha a decidir)
focus e blur com o mesmo HW: 50/50 pares
resoluções DISTINTAS       : 5 — (1536,2320) x25, (1536,2336) x13, (1520,2320) x9,
                                 (1536,2304) x2, e mais uma
EXIF                       : NENHUMA das 50 tem (medido em `caminho_b_analise/`)
```

### Duas consequências estruturais, e as duas são declaradas

**1. Resolução heterogênea.** Ao contrário do LFDOF (688×1008 em 100% das linhas) e do
espelho da RealBokeh (1500×2000), aqui há cinco resoluções. Por isso
`REALDOF_IMAGE_HW` é `None` e o carregador **não** fixa `expected_hw`: rejeitar por
resolução aqui descartaria 25 das 50 amostras.

O preço é real e é o motivo de estar escrito aqui: `k_value` é medido em PIXEL, e duas
amostras em resoluções diferentes têm K em escalas ligeiramente diferentes (2304 a
2336 de lado longo — 1,4% de espalhamento). Para a pergunta deste experimento isso não
contamina nada, porque a comparação é **pareada por imagem** entre braços de
profundidade: a mesma foto, a mesma resolução, três profundidades. Para virar rótulo de
release, teria que ser resolvido.

**2. Sem óptica.** Não há f-number, focal nem distância de foco — as 50 imagens vieram
sem EXIF, confirmado. Igual ao LFDOF: os quatro campos são `init=False`,
`aif_aperture_is_narrow` sai `applicable=False` e `_analytic_k` devolve `None`. A rota C
não precisa deles; a rota B precisaria.

## O nome, e o que ele afirma

`realdof_<numero>_<alinhamento>`, com `<alinhamento>` em `aligned`, `misaligned` ou
`shift_<X.Y>px` — o **mesmo vocabulário** do LFDOF, o que é esperado: os dois espelhos
saíram do mesmo gerador. Medido nas 50 linhas: 16 `aligned`, 25 `misaligned`, 9
`shift_*px` (2,2 a 4,9 px).

Os 25 `misaligned` são metade do conjunto e não são descartados — o paper §3.2(c) manda
preservar amostra difícil, e o desalinhamento entra como METADADO (`alignment`), legível
por quem for cortar depois. Descartá-los aqui seria decidir em silêncio que metade do
RealDOF não existe.

Cada linha é a sua própria cena: 50 nomes únicos, 50 números de cena distintos, um par
por cena. Não há `level` — o RealDOF não tem série de aberturas por cena, ao contrário
das outras duas fontes. `scene_level_count` seria 1 em 50 de 50 e por isso não existe
como campo: um número que é constante não descreve nada.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from control.contract import SampleRejected
from control.contract import reject as _contract_reject
from qc.rejection import RejectionLog

# --------------------------------------------------------------------------------
# Identidade da origem
# --------------------------------------------------------------------------------

#: Espelho privado, já pareado. `image_focus` é a AIF, `image_blur` é o alvo real.
REALDOF_DATASET = "akcit-pixel/RealDOF"

#: Fonte original pública: o conjunto de teste real do IFAN (Lee et al., "Iterative
#: Filter Adaptive Network for Single Image Defocus Deblurring", CVPR 2021), 50 pares
#: fotografados com a mesma câmera em duas aberturas. `[M]` quanto ao pareamento;
#: `[A]` quanto à licença, que não foi conferida aqui.
UPSTREAM_URL = "https://github.com/codeslake/IFAN"

#: Colunas do espelho que a rota C lê. NOMES, não caminhos — quem carrega decide de
#: onde lê (ver `LoadPair` em `routes/route_c.py`).
REALDOF_AIF_COLUMN = "image_focus"
REALDOF_BOKEH_COLUMN = "image_blur"

#: A quarta coluna do schema, que este módulo **não** referencia em lugar nenhum.
#: Saída de MODELO (pré-deblur), não imagem de origem. Existe como constante para que a
#: decisão de não usá-la esteja escrita, e não apenas ausente.
REALDOF_UNUSED_COLUMN = "image_pre_deblur"

#: **`None` de propósito.** Cinco resoluções distintas nas 50 linhas — ver o cabeçalho.
#: O carregador não fixa resolução esperada, e a consequência (K em pixel com 1,4% de
#: espalhamento entre amostras) está declarada lá.
REALDOF_IMAGE_HW: Optional[tuple[int, int]] = None

#: O único split que a origem publica. Enumerado, e não `[a-z]+`, pela mesma razão do
#: LFDOF: um split novo aparece como `source_name_unparseable` no histograma, alto e
#: claro, em vez de virar um `scene_id` que ninguém pediu.
SOURCE_SPLITS = frozenset({"validation"})

#: O split é constante — não está no nome, está no arquivo. Os 4 shards são
#: `validation-*`, e as 50 linhas vêm todas deles.
SOURCE_SPLIT = "validation"


# --------------------------------------------------------------------------------
# Rejeição — vocabulário fechado, herdado do contrato
# --------------------------------------------------------------------------------

#: Nenhum slug novo nasce nesta fonte: ela falha das mesmas maneiras que o LFDOF.
ALL_REJECTION_REASONS = frozenset({
    "source_name_unparseable",
    "source_duplicate_sample",
    "source_image_unreadable",
    "source_image_role_mismatch",
    "source_image_alpha_not_opaque",
})


def reject_source(reason: str, detail: str = "") -> None:
    """Único caminho de rejeição desta FONTE — o enumerador e o carregador usam este.

    Público de propósito: `sources/realdof_images.py` levanta os mesmos slugs, e um
    segundo `_reject` privado lá seria a semente de dois vocabulários divergentes para a
    mesma fonte.
    """
    _contract_reject(reason, detail)


_reject = reject_source


# --------------------------------------------------------------------------------
# O nome do espelho
# --------------------------------------------------------------------------------

#: `realdof_<numero>_<alinhamento>` — medido em 50/50 linhas.
_NAME_RE = re.compile(
    r"^realdof"
    r"_(?P<scene>[0-9]+)"
    r"_(?P<alignment>aligned|misaligned|shift_(?P<shift_px>[0-9]+(?:\.[0-9]+)?)px)$"
)

ALIGNMENT_ALIGNED = "aligned"
ALIGNMENT_MISALIGNED = "misaligned"
ALIGNMENT_SHIFT = "shift"


def scene_key(split: str, scene_number: str | int) -> str:
    """`(split, numero)` -> chave de cena. **Única definição** desta chave no RealDOF.

    Mesma forma que `realbokeh.scene_key` e `lfdof.scene_key`, de propósito:
    `dataio.split` vê um `scene_id` só, com uma regra só. Aqui o split é constante, então
    o prefixo não desambigua nada hoje — ele existe para que a chave tenha a mesma forma
    das outras duas fontes, que escrevem no mesmo release.
    """
    return f"{split}_{scene_number}"


@dataclass(frozen=True)
class ParsedName:
    """Tudo que o `file_name_base` afirma, e nada além disso."""

    scene_id: str
    scene_number: str
    source_split: str
    alignment: str                              # "aligned" | "misaligned" | "shift"
    #: Deslocamento residual anotado pela origem, em pixels da resolução NATIVA daquela
    #: linha (que varia — ver o cabeçalho). `None` quando o nome não anota, o que é
    #: **diferente de zero**: `misaligned` é exatamente o caso em que a origem diz que
    #: não fechou e NÃO diz quanto, e gravar 0,0 seria o valor de "perfeito".
    alignment_shift_px_at_source_hw: Optional[float]
    raw: str


def _parse_name(name: str) -> ParsedName:
    if not isinstance(name, str) or not name:
        _reject("source_name_unparseable", f"nome vazio ou não-string: {name!r}")
    match = _NAME_RE.match(name)
    if match is None:
        _reject("source_name_unparseable",
                f"{name!r} fora do padrão realdof_<numero>_<alinhamento>, com "
                "<alinhamento> em aligned|misaligned|shift_<X.Y>px")

    shift_text = match.group("shift_px")
    if shift_text is None:
        alignment = match.group("alignment")     # "aligned" ou "misaligned"
        shift_px: Optional[float] = None
    else:
        alignment = ALIGNMENT_SHIFT
        shift_px = float(shift_text)

    return ParsedName(
        scene_id=scene_key(SOURCE_SPLIT, match.group("scene")),
        scene_number=match.group("scene"),
        source_split=SOURCE_SPLIT,
        alignment=alignment,
        alignment_shift_px_at_source_hw=shift_px,
        raw=name,
    )


def parse_full_name(name: str) -> ParsedName:
    """`file_name_base` -> o que o nome afirma. **Rejeita** nome fora do padrão."""
    return _parse_name(name)


# --------------------------------------------------------------------------------
# O par
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class RealDOFPair:
    """Um par (AIF real, bokeh real) do RealDOF.

    Satisfaz o `PairSource` de `routes/route_c.py` — `scene_id`, `sample_id`,
    `source_dataset`, `source_sample_id`, `source_split`, `aif_ref`, `bokeh_ref`,
    `f_number`, `focal_length_mm`, `focus_plane_distance_m`, `aif_f_number`.

    Como no `LFDOFPair`, **quatro campos são estruturalmente `None`** e `init=False`
    para que ninguém possa injetá-los: as 50 imagens chegaram sem EXIF, medido. Não há
    campo `level` nem `scene_level_count` — o RealDOF tem um par por cena.
    """

    scene_id: str
    sample_id: str
    source_dataset: str
    source_sample_id: str
    source_split: str

    aif_ref: str
    bokeh_ref: str

    #: Anotação de alinhamento da própria origem. Metade das linhas é `misaligned`, e
    #: elas ficam — ver o cabeçalho do módulo.
    alignment: str
    alignment_shift_px_at_source_hw: Optional[float]

    # -- o que o RealDOF NÃO publica -------------------------------------------
    f_number: Optional[float] = field(default=None, init=False)
    aif_f_number: Optional[float] = field(default=None, init=False)
    focal_length_mm: Optional[float] = field(default=None, init=False)
    focus_plane_distance_m: Optional[float] = field(default=None, init=False)

    @property
    def is_aligned(self) -> bool:
        """`True` só quando a origem AFIRMA alinhamento."""
        return self.alignment == ALIGNMENT_ALIGNED

    @property
    def has_analytic_validator(self) -> bool:
        """Sempre `False` no RealDOF, e é afirmação, não acidente: sem EXIF não há
        Eq. 3 a fechar, então o sweep da Eq. 5 roda sem validador independente."""
        return False


def _sample_id(scene_id: str) -> str:
    """`c_realdof_<cena>`.

    Não usa o `file_name_base` cru de propósito, pela mesma razão do LFDOF: ele carrega
    a anotação de alinhamento, então re-anotar um par mudaria o `sample_id` e a retomada
    por `completed_ids()` reprocessaria a amostra como se fosse nova. O `file_name_base`
    original fica inteiro em `source_sample_id`.
    """
    return f"c_realdof_{scene_id}"


def pair_from_name(file_name_base: str, *,
                   source_dataset: str = REALDOF_DATASET) -> RealDOFPair:
    """Constrói UM par, ou levanta `SampleRejected` com o motivo."""
    parsed = _parse_name(file_name_base)
    return RealDOFPair(
        scene_id=parsed.scene_id,
        sample_id=_sample_id(parsed.scene_id),
        source_dataset=source_dataset,
        source_sample_id=file_name_base,
        source_split=parsed.source_split,
        aif_ref=REALDOF_AIF_COLUMN,
        bokeh_ref=REALDOF_BOKEH_COLUMN,
        alignment=parsed.alignment,
        alignment_shift_px_at_source_hw=parsed.alignment_shift_px_at_source_hw,
    )


def enumerate_pairs(file_name_bases: Iterable[str], *, log: RejectionLog,
                    source_dataset: str = REALDOF_DATASET) -> list[RealDOFPair]:
    """Enumera os pares do RealDOF a partir dos `file_name_base` do espelho.

    Uma passagem só, ao contrário do LFDOF: lá a passagem 1 existia para contar
    `scene_level_count`, que aqui não existe.

    `log` é **obrigatório e sem default**: rejeição sem motivo registrado não existe
    neste projeto, e um default `None` transformaria a chamada curta em descarte
    silencioso.
    """
    pares: list[RealDOFPair] = []
    vistos: dict[str, str] = {}                # sample_id -> file_name_base que o criou
    for nome in file_name_bases:
        try:
            par = pair_from_name(nome, source_dataset=source_dataset)
        except SampleRejected as exc:
            # `sample_id` do nome cru: um nome que não parseia não tem `sample_id`
            # nosso, e inventar um esconderia a linha no JSONL.
            log.reject_from(str(nome), exc, {"source_dataset": source_dataset})
            continue
        if par.sample_id in vistos:
            try:
                _reject("source_duplicate_sample",
                        f"{nome!r} e {vistos[par.sample_id]!r} produzem o mesmo "
                        f"sample_id {par.sample_id!r}. Medido: 0 colisões em 50 "
                        "linhas — se isto disparou, o espelho mudou.")
            except SampleRejected as exc:
                log.reject_from(str(nome), exc, {"scene_id": par.scene_id})
            continue
        vistos[par.sample_id] = nome
        pares.append(par)
        log.accept(par.sample_id, {"scene_id": par.scene_id,
                                   "alignment": par.alignment,
                                   "source_split": par.source_split})
    return pares


def scene_source_splits(pairs: Iterable[RealDOFPair]) -> dict[str, str]:
    """`{scene_id: source_split}`, pronto para `dataio.split.split_from_source`.

    Todas as cenas caem em `validation`, que `split_from_source` mapeia para `val`. O
    resultado é um `SceneSplit` com **train vazio**, e isso é o correto e não um
    defeito: o RealDOF é conjunto de teste de 50 pares, não material de treino. Quem
    quiser usá-lo como treino tem que dizer isso em voz alta, não herdar por omissão.
    """
    return {p.scene_id: p.source_split for p in pairs}


def enumeration_summary(pairs: list[RealDOFPair], log: RejectionLog) -> str:
    """O que a enumeração viu. Impresso pelo entrypoint, sempre."""
    alinhamentos = Counter(p.alignment for p in pairs)
    linhas = [
        f"[fonte] RealDOF: {len(pairs)} pares em {len({p.scene_id for p in pairs})} cenas "
        f"(um par por cena).",
        f"[fonte] alinhamento declarado pela origem: {dict(alinhamentos)} "
        "— nenhum é descartado; a anotação vai para o metadado.",
    ]
    resumo = log.summary() if hasattr(log, "summary") else ""
    if resumo:
        linhas.append(resumo)
    return "\n".join(linhas)
