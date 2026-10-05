"""Onde a profundidade de uma amostra mora — e por que ela não mora com a amostra.

## O fato medido

A profundidade é função da **cena**, não da amostra. Na rota A as 41 variantes de uma
imagem compartilham a mesma AIF, logo o mesmo Depth Pro, logo o mesmo PNG:
`routes/route_a.py:PreparedImage` já calcula `encode_depth` **uma vez por imagem** e
entrega o mesmo `EncodedDepth` às 41. Medido sobre o release `a_release` (69.700
amostras, 1.700 cenas), em 120 cenas sorteadas com semente fixa: **120/120** com a
profundidade byte a byte idêntica em todas as variantes, 0 divergentes, 0 ausentes.

Gravar `depth/<sample_id>.png` guardava 41 cópias do mesmo arquivo: 31,7 GB onde 0,77 GB
bastam. Este módulo é o contrato do layout que grava uma vez por cena.

## Dois layouts, um contrato

`per_sample` é o layout dos releases já publicados (`bokehnet-regen-rota-c`, e o
`-rota-a` que está subindo). Ele **não vira ilegal**: quem lê tem que continuar lendo os
dois, e é `depth_ref_of` que resolve os dois com uma regra só.

O que muda é o **contrato de leitura**. No layout por amostra o caminho era uma
convenção (`depth/` + `sample_id` + `.png`) que cada leitor remontava sozinho; no layout
por cena ele deixa de ser adivinhável, porque `scene_id` não é dedutível de `sample_id`
em rota nenhuma (`c_realbokeh_train_12_l3` -> `train_12`; `gp_ab…_v07` -> `gp_ab…`;
`b_9001` -> `9001`). Então a referência passa a ser **declarada em cada `meta/`**, no
campo `depth_ref`, e o layout do release inteiro em `depth_layout`.

## Por que o campo se chama `depth_ref`

1. **O sufixo `_ref` já é o vocabulário deste projeto** para "onde os pixels disto
   moram": `SampleRefs.aif_ref` e `SampleRefs.bokeh_ref` (`dataio/sample.py`). Um nome
   novo para a mesma ideia é a forma de nome do defeito que o CLAUDE.md persegue —
   cópias divergem.
2. **O valor é um caminho relativo ao release** (`depth/train_1275.png`), não um id nu.
   Um id nu (`depth_key = "train_1275"`) obrigaria quem lê a saber o diretório e a
   extensão — que é exatamente a convenção que a dedup torna indecifrável.
3. **Não `depth_path`**: "path" lê como caminho de sistema de arquivos, e um caminho
   absoluto gravado num release morre no primeiro `rsync`, no primeiro repack e no
   primeiro upload. `ref` diz o que é: uma referência interna ao release, resolvida
   contra a raiz de quem o abriu.

## A armadilha que este módulo fecha

`scene_id` vira **nome de arquivo**, e nome de arquivo colide. Duas colisões reais:

* `sources/realbokeh.scene_key` e `sources/lfdof.scene_key` produzem **a mesma string**
  — as duas são `f"{split}_{numero}"`. A cena `train_1275` da RealBokeh e a `train_1275`
  do LFDOF são cenas físicas diferentes. Hoje isso não faz dano porque o `sample_id`
  carrega a fonte (`c_realbokeh_…` contra `c_lfdof_…`) e o arquivo é por amostra; com
  `depth/<scene_id>.png` as duas escreveriam o **mesmo arquivo**. Quem detecta é o
  writer, que recusa alto — ver `dataio/writer.py`.
* `scene_id` com `/`, espaço ou vazio viraria caminho, e não nome.
  `require_filename_safe_scene_id` é o portão, e ele roda em toda gravação por cena.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

#: O diretório da profundidade, nos dois layouts. Único lugar onde o nome existe.
DEPTH_DIR = "depth"


class DepthLayout(str, Enum):
    """Vocabulário fechado, como `MaskSource` e `KSource`.

    String livre aqui reabriria a porta do `mask_source="automatic"`: um release que
    dissesse `depth_layout="scene"` passaria por um leitor e falharia noutro, e a
    diferença apareceria como arquivo faltando, não como layout desconhecido.
    """

    #: Um PNG por amostra — `depth/<sample_id>.png`. O layout dos releases já
    #: publicados, e o correto onde a dedup não se aplica (rota B: 1 amostra por cena).
    PER_SAMPLE = "per_sample"
    #: Um PNG por cena — `depth/<scene_id>.png`. O layout das rotas em que várias
    #: amostras compartilham a mesma AIF (A: 41 variantes; C: até 4 níveis).
    PER_SCENE = "per_scene"


#: O default de quem grava daqui para a frente. Quem não quer dedup **declara**
#: `PER_SAMPLE`, e a declaração fica no `run_config.json` do release.
DEFAULT_DEPTH_LAYOUT = DepthLayout.PER_SCENE

#: `scene_id` seguro como nome de arquivo em POSIX, em APFS/NTFS e no Hub.
#:
#: Fechado em vez de aberto de propósito: a lista de caracteres que *quebram* varia por
#: sistema (`:` no HFS, `\` no Windows, `?`/`*` no S3), e enumerar o que quebra deixa de
#: fora o próximo. Todas as fontes deste projeto já cabem aqui — `gp_a1b2c3d4e5f6`,
#: `ebb_…`, `train_1275`, `9001` —, então o conjunto restrito não custa nada e o portão
#: dispara antes de o arquivo existir.
_SCENE_ID_SEGURO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")


def require_filename_safe_scene_id(scene_id) -> str:
    """`scene_id`, ou `ValueError` dizendo por que ele não pode virar nome de arquivo.

    Levanta em vez de sanear. Sanear (`scene_id.replace("/", "_")`) é um fallback: duas
    cenas diferentes podem sanear para o mesmo nome, e aí a dedup funde duas cenas
    físicas num PNG só — sem deixar rastro, que é a assinatura de todo defeito deste
    projeto.
    """
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError(
            f"scene_id inválido para virar nome de arquivo: {scene_id!r}. No layout "
            "`per_scene` o `scene_id` É o nome do PNG de profundidade.")
    if not _SCENE_ID_SEGURO.match(scene_id):
        raise ValueError(
            f"scene_id {scene_id!r} não é seguro como nome de arquivo. Permitido: "
            "letras, dígitos, `.`, `_` e `-`, começando por letra ou dígito, até 127 "
            "caracteres. No layout `per_scene` ele vira `depth/<scene_id>.png` — um "
            "`/` viraria diretório, um espaço quebra o upload, e sanear fundiria duas "
            "cenas físicas num arquivo só.")
    return scene_id


def depth_ref(*, sample_id: str, scene_id: str, layout) -> str:
    """O caminho, relativo à raiz do release, do PNG de profundidade desta amostra.

    **Uma implementação só**, chamada pelo writer e pela validação. Duas cópias desta
    regra é como se chega a dois layouts que se parecem e não são o mesmo.
    """
    layout = DepthLayout(layout)
    if layout is DepthLayout.PER_SCENE:
        return f"{DEPTH_DIR}/{require_filename_safe_scene_id(scene_id)}.png"
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"sample_id inválido: {sample_id!r}")
    return f"{DEPTH_DIR}/{sample_id}.png"


def depth_ref_of(registro: Mapping) -> str:
    """A referência de profundidade de um `meta/<id>.json` **ou** de uma linha do
    manifesto — que carregam as mesmas chaves para isto.

    Três casos, e nenhum deles é adivinhação:

    1. **`depth_ref` presente** — é o contrato, e é o que vale. Ponto.
    2. **nem `depth_ref` nem `depth_layout`** — release anterior ao campo
       (`bokehnet-regen-rota-c`, `-rota-a`). Ali o layout era `per_sample` e só, então
       `depth/<sample_id>.png` é a leitura correta e é a única possível.
    3. **`depth_layout` presente sem `depth_ref`** — erro. O release afirma um layout e
       não diz onde o arquivo está; num release `per_scene` a convenção do caso 2
       apontaria para um arquivo que não existe, e devolver esse caminho seria dar uma
       resposta errada em vez de nenhuma.
    """
    ref = registro.get("depth_ref")
    if ref:
        if not isinstance(ref, str):
            raise ValueError(f"`depth_ref` tem que ser string: {ref!r}")
        return ref

    layout = registro.get("depth_layout")
    if layout is not None and DepthLayout(layout) is not DepthLayout.PER_SAMPLE:
        raise ValueError(
            f"`depth_layout={layout!r}` sem `depth_ref` em "
            f"{registro.get('sample_id')!r}. Num release por cena o caminho NÃO é "
            "dedutível do `sample_id`, e a referência é o contrato — não há convenção "
            "para cair de volta.")

    sample_id = registro.get("sample_id")
    if not sample_id:
        raise ValueError("registro sem `sample_id` e sem `depth_ref`: não há "
                         "profundidade alcançável a partir dele")
    return f"{DEPTH_DIR}/{sample_id}.png"


def resolve_depth_path(release_dir, registro: Mapping) -> Path:
    """O arquivo que **existe**, a partir de um `meta/` ou de uma linha do manifesto.

    Resolve os dois layouts (pela `depth_ref_of`) e as duas formas de empacotamento: o
    release plano do disco e o release **reagrupado por cena** que o
    `scripts/publish_release.py` produz para caber no limite de 10.000 arquivos por
    pasta do Hub (`depth/<scene_id>/<arquivo>`).

    Levanta `FileNotFoundError` nomeando os caminhos tentados. Nunca devolve um caminho
    que não existe: um `Path` inexistente devolvido aqui vira `FileNotFoundError` três
    camadas acima, sem dizer qual layout foi tentado.
    """
    raiz = Path(release_dir)
    ref = depth_ref_of(registro)
    tentativas = [raiz / ref]

    cena = registro.get("scene_id")
    if cena:
        # O nível a mais que o upload insere. O nome do arquivo não muda.
        tentativas.append(raiz / DEPTH_DIR / str(cena) / Path(ref).name)

    for caminho in tentativas:
        if caminho.is_file():
            return caminho
    raise FileNotFoundError(
        f"profundidade de {registro.get('sample_id')!r} não encontrada. Tentados: "
        + ", ".join(str(c) for c in tentativas))


def depth_layout_of_run_config(run_config: Optional[Mapping]) -> DepthLayout:
    """O layout que o release DECLARA, do `run_config.json`.

    Ausente significa `per_sample`, e não "descubra contando arquivos": os releases sem
    o campo são exatamente os anteriores a ele, e todos são por amostra. Inferir pelo
    número de arquivos confundiria um release por cena com um release por amostra
    incompleto — que é a diferença entre publicar e não publicar.
    """
    if not run_config:
        return DepthLayout.PER_SAMPLE
    declarado = run_config.get("depth_layout")
    if declarado is None:
        return DepthLayout.PER_SAMPLE
    return DepthLayout(declarado)
