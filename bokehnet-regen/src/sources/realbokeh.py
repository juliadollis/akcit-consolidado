"""Adaptador de fonte da RealBokeh — enumera os pares (AIF real, bokeh real) da rota C.

Duas origens, e as duas são necessárias:

| origem | o que traz | acesso |
|---|---|---|
| `akcit-pixel/RealBokeh` (privado) | as IMAGENS já pareadas: `image_focus`, `image_blur`, e o `file_name_base` que identifica cena e **nível** | token HF |
| `timseizinger/RealBokeh_3MP` (público) | `train/metadata/<cena>.json`: `focal_length`, `target_avs`, `focus_plane_distance`, `focus_plane_uncertainty` | anônimo |

O espelho resolve os defeitos D5 e D6 de uma vez — `image_focus` é o
`train/in/<cena>_f22.JPG` **byte a byte** (sha256 `59d8e910ca69…` conferido na cena
1038, `reference/ACHADOS.md`), então a AIF nunca mais sai de dentro de `gt/`, onde a
mediana do maior f-stop por cena é f/14 e 12,7% das cenas têm f/5.6 ou mais aberto.

O que o espelho **não** traz é o f-number: `file_name_base` carrega o **nível
ordinal**, não a abertura. Recuperá-la é o join deste módulo — `target_avs[level - 1]`
do JSON da cena.

## O que este módulo NÃO faz

Não abre imagem, não estima profundidade, não calcula K e não decide split. Ele
enumera e afirma o que a origem afirma, com a proveniência de onde cada campo veio.
O rótulo de K da rota C é a Eq. 5 (sweep de SSIM); `focal_length_mm` e
`focus_plane_distance_m` viajam junto como **validador** desse sweep, nunca como
rótulo — é `routes/route_c.py` quem decide isso, e este módulo só entrega os números.

## `level` é 1-BASED — medido, não suposto

Contagem sobre as **20.495** linhas do split `train` do espelho (nomes lidos por
projeção de coluna do parquet, `reference/ACHADOS.md`):

```
cenas distintas                                   : 3.959
cenas cujos níveis são EXATAMENTE 1..n, contíguos : 3.959 / 3.959   (100,0%)
menor nível observado em qualquer cena            : 1
cenas com nível 0                                 : 0
cenas com nível repetido                          : 0
```

Se a construção do espelho fosse 0-based, `level_0` apareceria — e ele não aparece em
nenhuma das 3.959 cenas. O discriminador mais forte é outro: **3.949 pares usam
`level == len(target_avs)`**, que sob indexação 0-based (`target_avs[level]`) seria
`IndexError` em todos os 3.949. Logo o índice é `level - 1`, e `f_number_for_level`
**rejeita** índice fora da lista em vez de fazer clamp: um clamp casaria o nível 6 com
o f-number do nível 5 e gravaria a abertura errada sem que nada denunciasse.

Contra as 20.495 linhas, com os 3.960 JSONs de `train/metadata` na mão:
`level > len(target_avs)` em **0** linhas, `level < 1` em **0** linhas.

O que continua **[I], não [M]**: que o nível `i` corresponda ao `i`-ésimo elemento de
`target_avs` na ORDEM em que o JSON os lista. A favor, medido sobre os **4.400** JSONs
(train + test + validation):

```
len(target_avs) == len(target_images)                        : 4.400 / 4.400
`target_avs` em ordem CRESCENTE                              : 4.400 / 4.400
target_avs[i] == f-number no nome de target_images[i]        : 23.051 / 23.051
```

Este módulo **confere a terceira linha par a par**, em runtime. Contra: nada.
Confirmação byte a byte exigiria baixar `image_blur` do espelho privado e comparar com
o `gt/<cena>/<cena>_f<av>.JPG` correspondente — não feito.

## O sufixo do nome é ANOTAÇÃO DE ALINHAMENTO, e não é sempre `aligned`

`reference/ACHADOS.md` registra o padrão como `…_level_<N>_aligned`. Medido sobre as
20.495 linhas, o sufixo tem **32 formas distintas**:

```
aligned          20.074   (97,94%)
misaligned          183   ( 0,89%)
shift_<X.Y>px       238   ( 1,16%)   30 valores distintos, de 2,0 a 4,9 px
```

São 421 linhas em que a origem **declara** que o registro geométrico entre `image_focus`
e `image_blur` não fechou. Descartar isso ao parsear seria jogar fora o único sinal de
qualidade de par que a origem publica — então ele vira campo, e o gate que o consome
(se houver) é decisão da rota, não daqui. O deslocamento vem em PIXELS; a resolução em
que foi medido é `MIRROR_IMAGE_HW` — ver a nota `[A]` na constante.

## As 59 imagens que faltam — identificadas

`reference/ACHADOS.md` registrava "faltam 59 imagens e 1 cena inteira contra o `gt/`
bruto (0,3%) — **não identificadas** [A]". Com o join deste módulo elas são **[M]**:
**59 imagens em 11 cenas**, e a "cena inteira" é uma das 11, não uma décima segunda.

```
cena  918: 21 níveis declarados, faltam 20 (níveis 2..21)
cena  938: 21 níveis, faltam 12 (10..21)      cena 2255:  2 níveis, faltam 2 (1..2)  <- cena inteira
cena  939: 21 níveis, faltam  9 (13..21)      cena   91:  5 níveis, faltam 2 (4..5)
cena  913:  5 níveis, faltam  4 ( 2.. 5)      cena  916:  5 níveis, faltam 2 (4..5)
cena   92:  9 níveis, faltam  3 ( 7.. 9)      cena  915:  5 níveis, falta  1 (5)
cena  925:  5 níveis, faltam  3 ( 3.. 5)      cena  919:  3 níveis, falta  1 (3)
```

Os níveis ausentes são **sempre a cauda** (11/11 cenas) — nunca um buraco no meio, o
que é o que mantém a contiguidade `1..k` intacta e o que torna a leitura 1-based
verificável. `2255` é a única cena com metadata e zero linhas no espelho; nenhuma cena
do espelho está sem metadata (**0** de 3.959).

Consequência para quem consome: `scene_level_count` (o que a cena declara) pode ser
maior que quantos pares a enumeração devolve para aquela cena. São 43 pares em 11
cenas — 0,21%. Não é rejeição: o par que existe é válido; o que falta simplesmente não
existe no espelho.

## Sem fallback

Toda ausência levanta `SampleRejected` com slug do vocabulário fechado. Nenhum campo
tem default numérico, nenhum `.get(chave, <numero>)`, nenhum `except: pass`. Um par
sem metadata da cena é rejeitado com motivo e entra no histograma; não some.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from control.contract import REJECTION_REASONS, SampleRejected
from control.contract import reject as _contract_reject
from qc.rejection import RejectionLog

# --------------------------------------------------------------------------------
# Identidade das origens
# --------------------------------------------------------------------------------

#: Espelho privado, já pareado. `image_focus` é a AIF, `image_blur` é o alvo.
MIRROR_DATASET = "akcit-pixel/RealBokeh"

#: Dataset bruto público. É de onde vem `metadata/<cena>.json` — e só ele.
RAW_DATASET = "timseizinger/RealBokeh_3MP"

#: Colunas do espelho que a rota C lê. Nomes, não caminhos: quem carrega a imagem
#: (`LoadPair` em `routes/route_c.py`) decide se lê do parquet, de disco ou de um mock.
MIRROR_AIF_COLUMN = "image_focus"
MIRROR_BOKEH_COLUMN = "image_blur"

#: (H, W) das imagens do espelho. **[M]**: 2000x1500 preservado sem reencode, medido
#: na cena 1038 (`reference/ACHADOS.md`). Existe aqui por uma razão só: o contrato
#: manda que toda quantidade em pixel carregue a resolução em que foi medida, e
#: `alignment_shift_px_at_mirror_hw` é uma quantidade em pixel.
#:
#: **[A]** que o deslocamento anotado no nome tenha sido medido NESTA resolução. É a
#: leitura mais plausível — a anotação está no nome do próprio espelho — mas não foi
#: confirmada. Se um dia for confirmada em outra escala, o fator entra aqui e em
#: lugar nenhum mais.
MIRROR_IMAGE_HW: tuple[int, int] = (1500, 2000)

#: Splits que a origem publica. `split_from_source` já sabe mapear estes nomes.
SOURCE_SPLITS = frozenset({"train", "test", "validation"})


# --------------------------------------------------------------------------------
# Rejeição — slugs que este módulo precisa e que `contract.REJECTION_REASONS` ainda
# não registra
# --------------------------------------------------------------------------------

#: Slugs de rejeição desta camada de FONTE. Nenhum deles existe hoje em
#: `control.contract.REJECTION_REASONS`, e `contract.py` **não foi editado** —
#: registrá-los é decisão de quem mantém o contrato, não deste módulo.
#:
#: Enquanto não estiverem lá, `_reject` levanta `SampleRejected` com o slug direto.
#: O histograma de `qc.rejection.RejectionLog` agrega por `exc.reason`, então nada se
#: perde; o que falta é o conjunto ser fechado de novo. Assim que forem registrados,
#: `_reject` passa a usar `contract.reject` sozinho — o `if` abaixo já trata os dois
#: casos, e nenhuma chamada muda.
#:
#: Por que cada um é um slug separado, e não um "erro de fonte" genérico: o histograma
#: é o instrumento que denuncia fallback novo, e ele só serve se distinguir "a cena não
#: tem metadata" (falta um arquivo no repo bruto) de "o nível não existe em
#: `target_avs`" (o join está errado) de "o f-number do nome não bate com o
#: `target_avs`" (a ordem da lista não é a que assumimos). São três consertos
#: diferentes.
#: Os seis slugs desta fonte. Espelham `control.contract.SOURCE_REJECTION_REASONS`,
#: que é o registro autoritativo; o teste `test_slugs_registrados_no_contrato` prova
#: que os dois conjuntos são iguais, para que um slug novo aqui não escape do
#: vocabulário fechado.
SOURCE_REJECTION_REASONS = frozenset({
    #: `file_name_base` fora do padrão do espelho. Não é "nome feio": é um nome de
    #: que não dá para extrair cena nem nível, logo não dá para fazer o join.
    "source_name_unparseable",
    #: A cena do espelho não tem `metadata/<cena>.json` no dataset bruto. Sem ele não
    #: há f-number, nem os três termos da Eq. 3.
    "source_metadata_missing",
    #: O nível do nome está fora de `target_avs`. NUNCA vira clamp: clamp casaria o
    #: nível 6 com a abertura do nível 5 e gravaria a abertura errada em silêncio.
    "source_level_out_of_range",
    #: `target_avs[level-1]` ausente, não finito, não positivo — ou divergente do
    #: f-number que está no nome de `target_images[level-1]`, que é a checagem
    #: cruzada que valida a ORDEM da lista.
    "source_f_number_invalid",
    #: `focal_length`, `focus_plane_distance`, `focus_plane_uncertainty` ou
    #: `source_av` ausente ou inválido no JSON da cena.
    "source_metadata_field_invalid",
    #: Duas linhas do espelho com o mesmo (cena, nível). Sem este gate elas viram dois
    #: `sample_id` iguais, e o writer/retomada trata a segunda como a primeira.
    "source_duplicate_sample",
})

#: Nome antigo, mantido porque `__init__` e os testes já o importam.
PENDING_REJECTION_REASONS = SOURCE_REJECTION_REASONS

def _reject(reason: str, detail: str = "") -> None:
    """Único caminho de rejeição deste módulo — delega ao contrato.

    Os seis slugs `source_*` já estão registrados em
    `control.contract.SOURCE_REJECTION_REASONS`, então este módulo não mantém
    vocabulário próprio: slug desconhecido levanta `KeyError` lá, e o histograma
    agrega junto com o das rotas.
    """
    _contract_reject(reason, detail)


# --------------------------------------------------------------------------------
# O nome do espelho
# --------------------------------------------------------------------------------

#: `timseizinger_realbokeh_3mp_<split>_f_<cena>_level_<N>_<alinhamento>`
#:
#: `_f_` é literal e vem do gerador do espelho; não é f-number. O `<cena>` é o id
#: numérico do dataset bruto e não contém `_` (medido: 3.959 de 3.959).
#: `<alinhamento>` é `aligned`, `misaligned` ou `shift_<X.Y>px` (medido: 32 formas,
#: 20.074 + 183 + 238 = 20.495 linhas).
_NAME_RE = re.compile(
    r"^timseizinger_realbokeh_3mp"
    r"_(?P<split>train|test|validation)"
    r"_f_(?P<scene>[0-9]+)"
    r"_level_(?P<level>[0-9]+)"
    r"_(?P<alignment>aligned|misaligned|shift_(?P<shift_px>[0-9]+(?:\.[0-9]+)?)px)$"
)

def scene_key(split: str, scene_number: str | int) -> str:
    """`(split, numero)` -> chave de cena. **Única definição** desta chave.

    Existe porque a numeração de cena do espelho reinicia em cada split: sem o prefixo,
    a cena 1 de `train`, de `test` e de `validation` viram uma só. Toda coisa que
    agrupa por cena — split, `sample_id`, contagem de diversidade — usa esta função.
    """
    return f"{split}_{scene_number}"


ALIGNMENT_ALIGNED = "aligned"
ALIGNMENT_MISALIGNED = "misaligned"
ALIGNMENT_SHIFT = "shift"


@dataclass(frozen=True)
class ParsedName:
    """Tudo que o `file_name_base` afirma, e nada além disso."""

    #: **Qualificado pelo split**: `train_1`, `test_1` e `validation_1` são TRÊS cenas
    #: físicas diferentes. Medido no espelho: a numeração REINICIA em cada split, e as
    #: 220 cenas de `test` reusam os números 1..220 que `train` também usa. Um
    #: `scene_id` cru colidiria: mesmo id de cena para cenas distintas (split furado) e
    #: mesmo `sample_id` para pares distintos (o gate de duplicata descartaria os dois
    #: últimos em silêncio, perdendo 2.495 amostras).
    scene_id: str
    #: O número cru, como aparece no nome e no caminho de `metadata/`. É por ele que se
    #: encontra o arquivo; é pelo `scene_id` que se agrupa.
    scene_number: str
    level: int
    source_split: str
    alignment: str                              # "aligned" | "misaligned" | "shift"
    #: Deslocamento residual anotado pela origem, em PIXELS de `MIRROR_IMAGE_HW`.
    #: `None` quando o nome não anota deslocamento — o que é diferente de "zero".
    #: `alignment == "misaligned"` é exatamente esse caso: a origem diz que não fechou
    #: e NÃO diz quanto. Gravar 0,0 aqui seria fallback numérico.
    alignment_shift_px_at_mirror_hw: Optional[float]
    raw: str


def _parse_name(name: str) -> ParsedName:
    if not isinstance(name, str) or not name:
        _reject("source_name_unparseable", f"nome vazio ou não-string: {name!r}")
    match = _NAME_RE.match(name)
    if match is None:
        _reject("source_name_unparseable",
                f"{name!r} fora do padrão "
                "timseizinger_realbokeh_3mp_<split>_f_<cena>_level_<N>_<alinhamento>")
    level = int(match.group("level"))
    if level < 1:
        # Redundante com o regex de hoje (`[0-9]+` aceita "0"), e de propósito: é a
        # afirmação de que o índice é 1-based, no lugar onde ela é usada.
        _reject("source_level_out_of_range",
                f"{name!r}: nível {level} < 1. O nível do espelho é 1-based "
                "(3.959/3.959 cenas com níveis contíguos 1..n, zero cenas com nível 0)")

    alignment_group = match.group("alignment")
    shift_text = match.group("shift_px")
    if shift_text is None:
        alignment = alignment_group                  # "aligned" ou "misaligned"
        shift_px: Optional[float] = None
    else:
        alignment = ALIGNMENT_SHIFT
        shift_px = float(shift_text)

    return ParsedName(
        scene_id=scene_key(match.group("split"), match.group("scene")),
        scene_number=match.group("scene"),
        level=level,
        source_split=match.group("split"),
        alignment=alignment,
        alignment_shift_px_at_mirror_hw=shift_px,
        raw=name,
    )


def parse_file_name_base(name: str) -> tuple[str, int]:
    """`file_name_base` -> `(scene_id, level)`.

    **Rejeita** nome fora do padrão com `SampleRejected("source_name_unparseable")`.
    Não devolve `None`: um `None` silencioso vira `scene_id=None` no manifesto, e uma
    amostra sem cena é uma amostra fora do split — que é vazamento com outro nome.

    Para o resto do que o nome afirma (split e anotação de alinhamento), use
    `parse_full_name`.
    """
    parsed = _parse_name(name)
    return parsed.scene_id, parsed.level


def parse_full_name(name: str) -> ParsedName:
    """Como `parse_file_name_base`, mas devolve split e alinhamento também."""
    return _parse_name(name)


# --------------------------------------------------------------------------------
# O JSON da cena, do dataset bruto público
# --------------------------------------------------------------------------------

#: `train/gt/1000/1000_f3.2.JPG` -> 3.2. O `parse_aperture` da rota C antiga usava
#: exatamente `split("_f")[-1]`, e ele casa com 20.554 de 20.554 nomes de `gt/`
#: (`reference/ACHADOS.md`) — ou seja, as cenas perdidas nunca foram falha de parser.
#: Aqui ele serve só de CHECAGEM CRUZADA contra `target_avs`, nunca de fonte primária.
_APERTURE_IN_PATH_RE = re.compile(r"_f(?P<av>[0-9]+(?:\.[0-9]+)?)\.[A-Za-z]+$")


def _aperture_from_path(path: str) -> Optional[float]:
    match = _APERTURE_IN_PATH_RE.search(str(path))
    if match is None:
        return None
    return float(match.group("av"))


def load_split_metadata(metadata_dir: str | Path, split: str) -> dict[str, dict]:
    """Lê `<metadata_dir>/*.json` de UM split e devolve `{scene_key: metadata}`.

    A chave é `scene_key(split, stem_do_arquivo)`, e o stem é o `<cena>` do
    `file_name_base` — é por ele que o join acontece. Quando o JSON traz `id`, ele é
    conferido contra o stem: divergência é `ValueError`, não rejeição de amostra.
    Rejeitar amostra é para dado faltando; arquivo cujo nome contradiz o conteúdo é
    fonte corrompida, e processar meio dataset corrompido é pior que parar.

    Diretório vazio também é `ValueError`: devolver `{}` faria `enumerate_pairs`
    rejeitar tudo por `source_metadata_missing`, e o histograma diria "o repo bruto
    perdeu todas as cenas" quando o que houve foi caminho errado.
    """
    directory = Path(metadata_dir)
    if not directory.is_dir():
        raise ValueError(f"não é diretório: {directory}")
    if split not in SOURCE_SPLITS:
        raise ValueError(f"split {split!r} fora de {sorted(SOURCE_SPLITS)}")

    out: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"{path}: JSON não é objeto, é {type(meta).__name__}")
        if "id" in meta and str(meta["id"]) != path.stem:
            raise ValueError(
                f"{path}: campo `id` = {meta['id']!r} contradiz o nome do arquivo "
                f"({path.stem!r}). O join da rota C é por este nome."
            )
        out[scene_key(split, path.stem)] = meta

    if not out:
        raise ValueError(f"nenhum JSON de metadata em {directory}")
    return out


def load_scene_metadata(raw_root: str | Path) -> dict[str, dict]:
    """Lê `<raw_root>/<split>/metadata/*.json` dos três splits, com chave qualificada.

    A RealBokeh_3MP guarda um `metadata/` POR SPLIT, e os números de cena se repetem
    entre eles. Ler um diretório só e chavear pelo número cru fundiria as três cenas
    de número 1 numa só — e o par de `test` receberia a distância de foco da cena de
    `train`, sem nada denunciar, porque o JSON tem todos os campos e é válido.

    Split cujo `metadata/` não existe é ignorado com aviso, não é erro: dá para rodar
    só com `train` se for isso que estiver em disco.
    """
    root = Path(raw_root)
    if not root.is_dir():
        raise ValueError(f"não é diretório: {root}")

    out: dict[str, dict] = {}
    encontrados = []
    for split in sorted(SOURCE_SPLITS):
        directory = root / split / "metadata"
        if not directory.is_dir():
            continue
        out.update(load_split_metadata(directory, split))
        encontrados.append(split)

    if not out:
        raise ValueError(
            f"nenhum <split>/metadata/ em {root}. Esperado o layout do {RAW_DATASET}: "
            f"{root}/train/metadata/<cena>.json"
        )
    if len(encontrados) < len(SOURCE_SPLITS):
        faltando = sorted(SOURCE_SPLITS - set(encontrados))
        print(f"[realbokeh] aviso: sem metadata para {faltando}; "
              f"as cenas desses splits serão rejeitadas por source_metadata_missing.")
    return out


def _required_positive_float(meta: Mapping[str, Any], key: str, scene_id: str) -> float:
    """Campo obrigatório do JSON da cena. Ausente ou inválido REJEITA.

    Sem `.get(key, <numero>)`: substituir por constante é exatamente o fallback que o
    contrato proíbe, e num campo da Eq. 3 ele mentiria sobre a óptica da cena.
    """
    if key not in meta:
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `{key}` ausente em metadata/{scene_id}.json")
    value = meta[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `{key}` = {value!r} não é número")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        _reject("source_metadata_field_invalid", f"cena {scene_id}: `{key}` = {value!r}")
    if value <= 0.0:
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `{key}` = {value!r} não é positivo")
    return value


def _required_non_negative_float(meta: Mapping[str, Any], key: str, scene_id: str) -> float:
    """Como `_required_positive_float`, mas aceita zero.

    Só `focus_plane_uncertainty` usa isto: incerteza zero é uma afirmação legítima
    ("medi e não sobrou dúvida"), enquanto distância focal zero ou f-number zero são
    impossibilidades físicas.
    """
    if key not in meta:
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `{key}` ausente em metadata/{scene_id}.json")
    value = meta[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `{key}` = {value!r} não é número")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
        _reject("source_metadata_field_invalid", f"cena {scene_id}: `{key}` = {value!r}")
    return value


def f_number_for_level(meta: Mapping[str, Any], level: int) -> float:
    """f-number do ALVO daquele nível — `target_avs[level - 1]`.

    **1-based**, medido: as 3.959 cenas do espelho têm níveis exatamente `1..n`,
    contíguos, sem repetição, e nenhuma tem nível 0. Se a construção fosse 0-based,
    `level_0` apareceria; ele não aparece em nenhuma linha das 20.495.

    Índice fora da lista **rejeita** (`source_level_out_of_range`). Não faz clamp: com
    clamp, um nível 6 numa cena de 5 níveis receberia a abertura do nível 5 e a
    amostra iria para o disco com o f-number errado, sem nada denunciar — que é o
    formato exato dos defeitos que este projeto está desfazendo.

    Confere ainda o f-number contra o NOME em `target_images[level - 1]`
    (`gt/1000/1000_f3.2.JPG` -> 3.2). É a checagem que valida a ORDEM de `target_avs`:
    se a lista estivesse fora de ordem em relação aos arquivos, esta comparação
    quebraria. Quando `target_images` não existe no JSON, a checagem é pulada e isso
    fica declarado no diagnóstico de quem chamou — não é motivo de rejeição, porque a
    fonte primária do f-number é `target_avs`.
    """
    scene_id = str(meta["id"]) if "id" in meta else "?"
    if "target_avs" not in meta:
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `target_avs` ausente")
    target_avs = meta["target_avs"]
    if not isinstance(target_avs, (list, tuple)):
        _reject("source_metadata_field_invalid",
                f"cena {scene_id}: `target_avs` é {type(target_avs).__name__}, não lista")
    if len(target_avs) == 0:
        _reject("source_metadata_field_invalid", f"cena {scene_id}: `target_avs` vazio")

    level = int(level)
    if level < 1:
        _reject("source_level_out_of_range",
                f"cena {scene_id}: nível {level} < 1 (o nível do espelho é 1-based)")
    index = level - 1
    if index >= len(target_avs):
        _reject("source_level_out_of_range",
                f"cena {scene_id}: nível {level} (índice {index}) fora de "
                f"`target_avs` de {len(target_avs)} elementos. SEM clamp: o nível "
                f"{len(target_avs)} tem outra abertura e gravá-la aqui seria mentira.")

    value = target_avs[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("source_f_number_invalid",
                f"cena {scene_id}: target_avs[{index}] = {value!r} não é número")
    f_number = float(value)
    if f_number != f_number or f_number in (float("inf"), float("-inf")) or f_number <= 0.0:
        _reject("source_f_number_invalid",
                f"cena {scene_id}: target_avs[{index}] = {f_number!r}")

    target_images = meta.get("target_images")
    if isinstance(target_images, (list, tuple)) and index < len(target_images):
        from_name = _aperture_from_path(target_images[index])
        if from_name is not None and abs(from_name - f_number) > 1e-6 * max(1.0, f_number):
            _reject("source_f_number_invalid",
                    f"cena {scene_id}: target_avs[{index}] = {f_number} diverge do "
                    f"f-number do nome target_images[{index}] = "
                    f"{target_images[index]!r} -> {from_name}. A ORDEM de `target_avs` "
                    "é o que o join por nível assume; se ela não bate com os arquivos, "
                    "o join está errado.")
    return f_number


def level_count(meta: Mapping[str, Any]) -> int:
    """Quantos níveis a cena declara. É `len(target_avs)`, e nada mais."""
    if "target_avs" not in meta or not isinstance(meta["target_avs"], (list, tuple)):
        scene_id = str(meta["id"]) if "id" in meta else "?"
        _reject("source_metadata_field_invalid", f"cena {scene_id}: `target_avs` ausente")
    return len(meta["target_avs"])


# --------------------------------------------------------------------------------
# O par
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class RealBokehPair:
    """Um par (AIF real, bokeh real) da rota C, com o que a origem afirma sobre ele.

    Satisfaz o `PairSource` de `routes/route_c.py` — `scene_id`, `sample_id`,
    `source_dataset`, `source_sample_id`, `source_split`, `aif_ref`, `bokeh_ref`,
    `f_number`, `focal_length_mm`, `focus_plane_distance_m`, `aif_f_number` — e
    acrescenta o que só a RealBokeh tem.

    Nenhum campo é `Optional` nesta fonte, de propósito: a RealBokeh publica todos, e
    aceitar `None` aqui abriria a porta para um default numérico uma camada abaixo.
    Onde o dado falta, a construção REJEITA; não existe par meio construído.

    `focus_plane_distance_m` é profundidade métrica MEDIDA na cena real — não estimada
    por modelo. Ela é o validador do sweep da Eq. 5 e permite auditar a escala do
    Depth Pro por cena. `focus_plane_uncertainty_m` viaja junto porque uma distância
    de foco sem barra de erro não dá para usar como validador: `1,92 ± 0,12 m` e
    `1,92 ± 0,00 m` autorizam conclusões diferentes.
    """

    scene_id: str
    level: int
    sample_id: str
    source_dataset: str
    source_sample_id: str
    source_split: str

    #: f-number do ALVO (`image_blur`), de `target_avs[level - 1]`.
    f_number: float
    #: f-number da AIF (`image_focus`), de `source_av`. É f/22 na RealBokeh, e é o que
    #: alimenta o gate do D6 (`aif_aperture_is_narrow`) — lido, nunca assumido.
    aif_f_number: float

    #: Os três termos da Eq. 3, anotados de fábrica pela origem.
    focal_length_mm: float
    focus_plane_distance_m: float
    focus_plane_uncertainty_m: float

    #: Onde as imagens estão, no espelho. São nomes de COLUNA — quem carrega decide de
    #: onde lê. Ver `LoadPair` em `routes/route_c.py`.
    aif_ref: str
    bokeh_ref: str

    #: Os mesmos dois arquivos no dataset bruto público, para auditoria byte a byte.
    #: `image_focus` já foi conferido contra `raw_aif_path` por sha256 na cena 1038.
    raw_aif_path: str
    raw_bokeh_path: str

    #: Anotação de alinhamento da própria origem. Ver o cabeçalho do módulo.
    alignment: str
    alignment_shift_px_at_mirror_hw: Optional[float]

    #: Quantos níveis a cena tem. Sem ele, "2.932 amostras" esconde que elas vêm de
    #: bem menos cenas — e é essa razão que diz qual é a diversidade efetiva.
    scene_level_count: int

    @property
    def is_aligned(self) -> bool:
        """`True` só quando a origem AFIRMA alinhamento. `misaligned` e `shift_*px`
        são `False`, e a diferença entre eles fica em `alignment`."""
        return self.alignment == ALIGNMENT_ALIGNED


def _sample_id(scene_id: str, level: int) -> str:
    """`c_realbokeh_<cena>_l<nivel>`.

    Não usa o `file_name_base` cru de propósito: ele carrega a anotação de
    alinhamento, então re-anotar um par mudaria o `sample_id`, e a retomada por
    `RejectionLog.completed_ids()` reprocessaria a amostra como se fosse nova. O
    `file_name_base` original fica inteiro em `source_sample_id`.
    """
    return f"c_realbokeh_{scene_id}_l{level}"


def pair_from_name(
    file_name_base: str,
    scene_metadata: Mapping[str, Mapping[str, Any]],
    *,
    source_dataset: str = MIRROR_DATASET,
) -> RealBokehPair:
    """Constrói UM par, ou levanta `SampleRejected` com o motivo.

    É o caminho unitário: `enumerate_pairs` é só o laço com o histograma em volta.
    Separado para poder ser testado por caso de falha, um a um.
    """
    parsed = _parse_name(file_name_base)

    if parsed.scene_id not in scene_metadata:
        _reject("source_metadata_missing",
                f"cena {parsed.scene_id} (de {file_name_base!r}) não tem "
                f"{parsed.source_split}/metadata/{parsed.scene_number}.json em {RAW_DATASET}")
    meta = scene_metadata[parsed.scene_id]

    f_number = f_number_for_level(meta, parsed.level)
    aif_f_number = _required_positive_float(meta, "source_av", parsed.scene_id)
    focal_length_mm = _required_positive_float(meta, "focal_length", parsed.scene_id)
    focus_distance_m = _required_positive_float(meta, "focus_plane_distance", parsed.scene_id)
    focus_uncertainty_m = _required_non_negative_float(
        meta, "focus_plane_uncertainty", parsed.scene_id)

    if "source_image" not in meta:
        _reject("source_metadata_field_invalid",
                f"cena {parsed.scene_id}: `source_image` ausente — sem ele não há como "
                "auditar a AIF do espelho contra o arquivo bruto")
    n_levels = level_count(meta)

    target_images = meta.get("target_images")
    if not isinstance(target_images, (list, tuple)) or parsed.level - 1 >= len(target_images):
        _reject("source_metadata_field_invalid",
                f"cena {parsed.scene_id}: `target_images` não cobre o nível "
                f"{parsed.level} (tem {0 if not isinstance(target_images, (list, tuple)) else len(target_images)})")

    prefix = f"{parsed.source_split}/"
    return RealBokehPair(
        scene_id=parsed.scene_id,
        level=parsed.level,
        sample_id=_sample_id(parsed.scene_id, parsed.level),
        source_dataset=source_dataset,
        source_sample_id=file_name_base,
        source_split=parsed.source_split,
        f_number=f_number,
        aif_f_number=aif_f_number,
        focal_length_mm=focal_length_mm,
        focus_plane_distance_m=focus_distance_m,
        focus_plane_uncertainty_m=focus_uncertainty_m,
        aif_ref=MIRROR_AIF_COLUMN,
        bokeh_ref=MIRROR_BOKEH_COLUMN,
        raw_aif_path=prefix + str(meta["source_image"]),
        raw_bokeh_path=prefix + str(target_images[parsed.level - 1]),
        alignment=parsed.alignment,
        alignment_shift_px_at_mirror_hw=parsed.alignment_shift_px_at_mirror_hw,
        scene_level_count=n_levels,
    )


def enumerate_pairs(
    file_name_bases: Iterable[str],
    scene_metadata: Mapping[str, Mapping[str, Any]],
    *,
    log: RejectionLog,
    source_dataset: str = MIRROR_DATASET,
) -> list[RealBokehPair]:
    """Junta espelho + metadata e devolve os pares da rota C.

    `log` é **obrigatório e sem default**. Um default `None` transformaria a chamada
    curta — que é a que todo mundo escreve — em descarte silencioso, e a regra do
    projeto é que rejeição sem motivo registrado não existe. Um `RejectionLog()` sem
    `path` custa nada e já agrega o histograma em memória.

    Cada linha rejeitada entra em `log` com `sample_id`, slug e detalhe. `log.summary()`
    imprime o histograma que todo run tem que terminar imprimindo.
    """
    pairs: list[RealBokehPair] = []
    seen: dict[str, str] = {}                 # sample_id -> file_name_base que o criou

    for name in file_name_bases:
        try:
            pair = pair_from_name(name, scene_metadata, source_dataset=source_dataset)
        except SampleRejected as exc:
            # `sample_id` do nome cru: um nome que não parseia não tem `sample_id`
            # nosso, e inventar um esconderia a linha no JSONL.
            log.reject_from(str(name), exc, {"source_dataset": source_dataset})
            continue

        if pair.sample_id in seen:
            try:
                _reject("source_duplicate_sample",
                        f"{name!r} e {seen[pair.sample_id]!r} produzem o mesmo "
                        f"sample_id {pair.sample_id!r}")
            except SampleRejected as exc:
                log.reject_from(str(name), exc, {"scene_id": pair.scene_id})
            continue

        seen[pair.sample_id] = name
        pairs.append(pair)
        log.accept(pair.sample_id, {"scene_id": pair.scene_id, "level": pair.level,
                                    "source_split": pair.source_split})
    return pairs


# --------------------------------------------------------------------------------
# Split — o que alimenta `dataio.split.split_from_source`
# --------------------------------------------------------------------------------

def scene_source_splits(pairs: Iterable[RealBokehPair]) -> dict[str, str]:
    """`{scene_id: source_split}`, pronto para `dataio.split.split_from_source`.

    A RealBokeh_3MP já separa `train` / `test` / `validation` **por cena** (220 cenas
    em cada um dos dois últimos), e reusar isso é melhor que sortear: preserva a
    intenção de quem montou o dataset e mantém comparabilidade com quem já publicou
    número nele.

    Uma cena que aparecer em dois splits é `ValueError`, não "o último ganha": cena nos
    dois lados é vazamento, e a RealBokeh entrega de 2 a 21 aberturas da MESMA cena —
    é o caso em que um split por imagem infla a validação sem que nada denuncie.

    Aviso de uso, medido: o espelho `akcit-pixel/RealBokeh` publica **só o split
    `train`** (20.495 de 20.495 linhas). Alimentar `split_from_source` só com os pares
    do espelho produz um `SceneSplit` com `val_fraction == 0,0` — tecnicamente válido,
    inútil na prática. Quem quiser validação ou junta os pares dos splits `test` /
    `validation` do bruto, ou usa `dataio.split.build_scene_split`. Este módulo não
    escolhe por ninguém, e não inventa um lado.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        if pair.source_split not in SOURCE_SPLITS:
            raise ValueError(
                f"cena {pair.scene_id!r}: split de origem {pair.source_split!r} fora de "
                f"{sorted(SOURCE_SPLITS)}"
            )
        previous = out.get(pair.scene_id)
        if previous is not None and previous != pair.source_split:
            raise ValueError(
                f"cena {pair.scene_id!r} aparece em dois splits de origem: "
                f"{previous!r} e {pair.source_split!r}. Cena nos dois lados é "
                "vazamento — a unidade do split é a CENA, nunca a amostra."
            )
        out[pair.scene_id] = pair.source_split
    if not out:
        raise ValueError("nenhum par: não há split a montar")
    return out


# --------------------------------------------------------------------------------
# Contagens — em cenas E em amostras
# --------------------------------------------------------------------------------

def enumeration_summary(pairs: list[RealBokehPair], log: RejectionLog) -> str:
    """O relatório que fecha a enumeração.

    Imprime cena E amostra, porque 20.495 amostras vindas de 3.959 cenas não são
    20.495 unidades de diversidade — e imprime a distribuição de níveis por cena, que
    é o discriminador que provou de onde o `image_focus` do espelho vem.
    """
    per_scene = Counter(p.scene_id for p in pairs)
    lines = [
        "",
        "=" * 62,
        f"  fonte     : {MIRROR_DATASET}  x  {RAW_DATASET}/metadata",
        f"  pares     : {len(pairs)}",
        f"  cenas     : {len(per_scene)}",
        f"  níveis/cena: "
        + " · ".join(f"{k} -> {v}" for k, v in sorted(Counter(per_scene.values()).items())),
    ]
    alignment = Counter(p.alignment for p in pairs)
    if alignment:
        lines.append("  alinhamento: "
                     + " · ".join(f"{k} -> {v}" for k, v in alignment.most_common()))
    splits = Counter(p.source_split for p in pairs)
    lines.append("  split de origem: "
                 + " · ".join(f"{k} -> {v}" for k, v in splits.most_common()))
    lines.append("=" * 62)
    return "\n".join(lines) + log.summary()
