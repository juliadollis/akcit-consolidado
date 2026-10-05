"""Leitor do release do `bokehnet-regen` — a ÁRVORE DE ARQUIVOS e o resolvedor
de referências de pixel.

Este módulo fecha a lacuna **L1** da `AUDITORIA_TREINO_BOKEHNET.md` §5.

## O problema que ele resolve

`BokehMetricDataset` (`data.py`) chama `load_dataset(repo)` e lê
`registro["aif"]`, `registro["bokeh"]`, `registro["depth"]` — ou seja, assume
que o release é uma **tabela** com os pixels dentro. O `bokehnet-regen` publica
outra coisa (`src/dataio/writer.py:FileSampleWriter`,
`scripts/publish_release.py`):

    <release>/depth/<id>.png          disparidade uint16, lado longo 768
    <release>/mask/<id>.png           região FINAL de foco
    <release>/meta/<id>.json          TUDO que não é pixel
    <release>/generated/<id>_*.jpg    SÓ o que a rota GERA
    <release>/manifest.jsonl          uma linha por amostra
    <release>/split.json              split POR CENA, materializado
    <release>/source_images.jsonl     sha256 dos bytes de ORIGEM

E, por decisão de cota (365 GB contra 115 GB livres), **os pixels podem não
estar no release** (`bokehnet-regen/src/dataio/sample.py`, docstring — *"guardamos
o que geramos, referenciamos o que já existe"*):

| rota | bokeh | AIF |
|---|---|---|
| C | referência | referência |
| B | referência | **gravada** em `generated/<id>_aif.jpg` |
| A | **gravado** em `generated/<id>_bokeh.jpg` | referência |

`publish_release.py` chama um release com `source/` de **autocontido**, e o modo
é escolhido em tempo de geração com `--store-source-images`.

**O ponto deste módulo é que o modo do release deixe de ser decisão bloqueante.**
Ele aceita as duas formas, e o caminho de tabela achatada de `data.py` continua
existindo intacto para um release que saia assim.

## As três origens de pixel, nesta ordem

1. `generated/` — o que ESTA rota gerou. É a origem mais forte: são bytes do
   próprio release, e `meta["generated_images"]` diz quais papéis estão lá.
2. `source/` — release **autocontido**. Os bytes são os ORIGINAIS da origem,
   não recomprimidos, justamente para que o sha256 do ledger bata com o arquivo
   ao lado (`mirror_images.MirrorImageLoader._store`).
3. o **espelho de origem**, por `source_dataset` + `source_sample_id` +
   `aif_ref`/`bokeh_ref`. É aqui que entra o índice.

Em 2 e 3 o sha256 é conferido contra `source_images.jsonl` quando a linha existe
— **o release existe para que esse join seja verificável**, e conferir é a única
coisa que transforma "aponta para o espelho" em evidência.

## Reuso do índice do `bokehnet-regen`

`IndiceEspelho` é o espelho *read-side* de
`bokehnet-regen/src/sources/mirror_images.py:MirrorIndex`, com a MESMA chave
(`file_name_base`, que é o `source_sample_id`), o MESMO cache
(`_mirror_index.json`, no formato `{"shards": [...], "locations": {nome: [shard,
linha]}}`) e o mesmo `locate()`. Isso é deliberado: um índice já construído pela
geração é lido aqui sem reconstruir, e o caminho de leitura do cache **não
importa pyarrow** — inspecionar o índice não deveria exigir o stack de parquet.

Duas cópias da mesma lógica seria exatamente o defeito que este projeto persegue
("cópias divergem — foi assim que se chegou a quatro interpretações de K"), e o
que impede a divergência aqui é o formato de cache compartilhado, travado por
teste.

## Decisões nossas, declaradas (o paper e a inferência oficial calam)

- **D-R1.** Escalares vêm do `manifest.jsonl` para a varredura (filtros, split,
  contagem por rota) e do `meta/<id>.json` para a amostra. O manifesto **não
  traz `disparity_min`/`disparity_max`** (writer.py), que são obrigatórios para
  decodificar a profundidade, nem `aif_ref`/`bokeh_ref`. Ler os 20 mil JSONs só
  para filtrar seria desperdício; ler só os que sobreviveram, sob demanda, não.
  Onde os dois trazem a mesma chave, o `meta/` vence — ele é a fonte, o
  manifesto é a cópia para varredura.
- **D-R2.** O layout por amostra é aceito **plano** (`depth/<id>.png`) e
  **agrupado por cena** (`depth/<scene_id>/<id>.png`). Os dois existem de
  verdade: `publish_release.py:_agrupa_por_cena` reagrupa no push porque o Hub
  recusa mais de 10.000 arquivos por diretório, e o release em disco fica plano.
  A detecção é por tentativa, e o resultado é memorizado por subpasta.
- **D-R3.** `snapshot_download` roda com `local_files_only=True` por default.
  Baixar é opt-in explícito (`permitir_download=True`). Motivo registrado: uma
  auditoria deste projeto mediu 552 GB de egress inexplicado vindo de download
  automático; ler do disco é o default, e quem quiser baixar pede.
- **D-R4.** Pixel que não resolve é **erro com motivo nomeado**, nunca amostra
  descartada em silêncio e nunca imagem substituta. É a regra 4 do
  `CONTRATO.md` aplicada ao lado de leitura.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

from . import control

# =============================================================================
# Nomes do layout — vindos do `bokehnet-regen`, não inventados aqui
# =============================================================================

#: `dataio/writer.py:FileSampleWriter` e `scripts/publish_release.py:_POR_AMOSTRA`.
SUBPASTAS_POR_AMOSTRA = ("depth", "mask", "meta", "generated")

NOME_MANIFESTO = "manifest.jsonl"
NOME_SPLIT = "split.json"
#: Ledger dos bytes de ORIGEM: sha256, shard e linha. `publish_release.py` REPROVA
#: um release da rota C sem ele — é o que torna o join verificável.
NOME_LEDGER_ORIGEM = "source_images.jsonl"
#: Pasta do release **autocontido** (`--store-source-images`).
PASTA_AUTOCONTIDA = "source"

#: Coluna de texto que identifica a linha no espelho parquet. É o
#: `source_sample_id`. Igual a `mirror_images.NAME_COLUMN`.
COLUNA_NOME_NO_ESPELHO = "file_name_base"
#: Cache do índice, DENTRO do snapshot. Mesmo nome e mesmo formato do
#: `mirror_images.INDEX_FILENAME`, para que um índice já construído pela geração
#: seja lido aqui sem reconstruir.
NOME_CACHE_INDICE = "_mirror_index.json"

#: Papéis de pixel que uma amostra pode ter. Vocabulário FECHADO: um papel novo
#: sem entrada aqui levanta, em vez de virar uma chave que ninguém agrega.
PAPEIS = ("aif", "bokeh")


class ReleaseError(control.ControlContractError):
    """Defeito no release, ou release numa forma que este leitor não lê.

    Herda de `ControlContractError` de propósito: a política de "sem fallback
    numérico" é a mesma dos dois lados, e quem já captura o erro do contrato
    captura este. Nunca é capturado para virar default.
    """


def _exigir(condicao: bool, mensagem: str) -> None:
    # `if`, não `assert`: `python -O` desliga assert, e a trava sumiria no run
    # de produção, que é onde ela importa.
    if not condicao:
        raise ReleaseError(mensagem)


# =============================================================================
# Índice do espelho — read-side de `mirror_images.MirrorIndex`
# =============================================================================

@dataclass(frozen=True)
class LocalizacaoNoEspelho:
    """Onde uma linha do espelho mora. Espelha `mirror_images.MirrorLocation`."""

    shard: str
    row: int


class IndiceEspelho:
    """`file_name_base -> (shard, linha)`, lido do cache ou construído.

    Ver o cabeçalho do módulo para por que o formato de cache é compartilhado
    com o `bokehnet-regen` em vez de reimplementado.
    """

    def __init__(
        self, locations: Mapping[str, LocalizacaoNoEspelho], *, shards: Sequence[str]
    ) -> None:
        self._locations = dict(locations)
        self._shard_rank = {name: i for i, name in enumerate(shards)}
        self.shards = list(shards)

    def __len__(self) -> int:
        return len(self._locations)

    def __contains__(self, name: object) -> bool:
        return name in self._locations

    def locate(self, name: str) -> LocalizacaoNoEspelho:
        try:
            return self._locations[name]
        except KeyError:
            raise KeyError(
                f"{name!r} não está no índice do espelho ({len(self._locations)} "
                "linhas). Índice desatualizado ou snapshot incompleto — reconstrua "
                "com IndiceEspelho.carregar(..., forcar=True) antes de culpar o "
                "release."
            ) from None

    def sort_key(self, name: str) -> tuple[int, int]:
        """Ordem de leitura `(shard, linha)`.

        Parquet lê por row group: pedir a linha 40.000 do shard 3, depois a 12 do
        shard 70, depois a 40.001 do shard 3 relê o mesmo row group duas vezes.
        O dataloader **não** pode reordenar as amostras (a ordem é do sampler),
        mas quem varrer o release em lote deve.
        """
        loc = self.locate(name)
        return (self._shard_rank[loc.shard], loc.row)

    # -- construção ------------------------------------------------------------

    @classmethod
    def carregar(cls, snapshot_dir: str | Path, *, forcar: bool = False) -> "IndiceEspelho":
        """Lê `_mirror_index.json`; constrói (e cacheia) se não existir.

        O caminho do cache NÃO importa pyarrow, de propósito — reler um índice já
        construído é só JSON, e vale poder fazer isso numa máquina sem o stack de
        parquet.
        """
        raiz = Path(snapshot_dir)
        _exigir(raiz.is_dir(), f"snapshot do espelho não é um diretório: {raiz}")
        cache = raiz / NOME_CACHE_INDICE
        if cache.is_file() and not forcar:
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                locations = {
                    nome: LocalizacaoNoEspelho(shard, int(row))
                    for nome, (shard, row) in payload["locations"].items()
                }
                return cls(locations, shards=list(payload["shards"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                # Cache truncado por uma escrita não-atômica de versão anterior.
                # Reconstruir é caro mas correto; seguir com índice pela metade
                # apontaria amostras para a linha errada do espelho, que é o
                # defeito mais silencioso que existe aqui.
                print(
                    f"[release] cache do índice em {cache} ilegível ({exc}); "
                    "reconstruindo a partir dos parquets.",
                    flush=True,
                )
        return cls._construir(raiz, cache)

    @classmethod
    def _construir(cls, raiz: Path, cache: Path) -> "IndiceEspelho":
        import pyarrow.parquet as pq   # import tardio: só quem constrói precisa

        shards = sorted(str(p.relative_to(raiz)) for p in raiz.rglob("*.parquet"))
        if not shards:
            raise ReleaseError(
                f"nenhum .parquet em {raiz}, e nenhum {NOME_CACHE_INDICE} para ler. "
                "Este é o snapshot do espelho de origem, não o release: aponte-o "
                "com `mirror_roots`, ou publique o release AUTOCONTIDO "
                "(--store-source-images) e a resolução por espelho deixa de ser "
                "necessária."
            )

        locations: dict[str, LocalizacaoNoEspelho] = {}
        duplicados: list[str] = []
        for shard in shards:
            tabela = pq.read_table(raiz / shard, columns=[COLUNA_NOME_NO_ESPELHO])
            for row, nome in enumerate(
                tabela.column(COLUNA_NOME_NO_ESPELHO).to_pylist()
            ):
                if nome in locations:
                    duplicados.append(str(nome))
                    continue
                locations[str(nome)] = LocalizacaoNoEspelho(shard, row)
        if duplicados:
            # Escolher a primeira ocorrência em silêncio é o tipo de decisão que
            # some no meio de 20.495 linhas.
            raise ReleaseError(
                f"{len(duplicados)} `{COLUNA_NOME_NO_ESPELHO}` repetidos no espelho "
                f"(ex.: {duplicados[:3]}). O índice seria ambíguo — resolva na origem."
            )

        # ESCRITA ATÔMICA — temp + rename, com nome único por processo.
        #
        # Em multi-GPU os N ranks constroem o índice ao mesmo tempo e gravariam o
        # MESMO arquivo. `write_text` trunca e depois escreve: outro rank lendo no
        # meio pega JSON cortado (`JSONDecodeError`), e dois escrevendo ao mesmo
        # tempo deixam o cache corrompido para sempre — um defeito intermitente,
        # que aparece no arranque de um run de 60K steps e some quando se tenta
        # reproduzir. `os.replace` é atômico no mesmo sistema de arquivos: quem
        # lê vê o arquivo antigo inteiro ou o novo inteiro, nunca metade.
        #
        # O conteúdo é idêntico entre os ranks (mesmo snapshot, mesma ordem de
        # shards), então quem ganha a corrida não importa.
        try:
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "shards": shards,
                        "locations": {
                            n: [l.shard, l.row] for n, l in locations.items()
                        },
                    },
                    indent=0,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, cache)
        except OSError as exc:      # snapshot só-leitura: o índice ainda serve
            print(
                f"[release] AVISO: não consegui cachear o índice em {cache}: {exc}. "
                "Ele será reconstruído no próximo run.",
                flush=True,
            )
        return cls(locations, shards=shards)


# =============================================================================
# Gramática das referências de pixel
# =============================================================================

#: `repo/nome#split[123].coluna` — a forma que a rota B usa para a bokeh do ITW
#: (`scripts/run_route_b.py:132`: `f"{source_dataset}#{ITW_SPLIT}[{index}].image"`).
_REF_LINHA_HF = re.compile(r"^(?P<ds>[^#]+)#(?P<split>[^\[]+)\[(?P<row>\d+)\]\.(?P<col>.+)$")
#: Nome de coluna parquet: um identificador simples, sem separador de caminho.
#: É a forma das rotas C (`image_focus` / `image_blur`).
_REF_COLUNA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Referencia:
    """Uma referência de pixel, já decomposta.

    `tipo` é vocabulário FECHADO:

    - `"coluna"`   — nome de coluna no espelho parquet; a LINHA vem do
      `source_sample_id` via `IndiceEspelho`. Rotas C (RealBokeh, LFDOF).
    - `"linha_hf"` — `repo#split[linha].coluna`. Rota B (ITW/BokehDiffusion),
      cuja unidade de origem é o índice da linha, não um nome.
    - `"caminho"`  — caminho relativo dentro da raiz da fonte. Rota A
      (`sources/genphoto_ebb.py:374`, `aif_ref=relativo`).
    """

    tipo: str
    bruta: str
    coluna: str | None = None
    dataset: str | None = None
    split: str | None = None
    linha: int | None = None
    caminho: str | None = None


def parse_referencia(ref: str) -> Referencia:
    """Decompõe `aif_ref`/`bokeh_ref`. Forma desconhecida é ERRO, não palpite.

    As três gramáticas convivem porque as três rotas têm origens de natureza
    diferente, e o `bokehnet-regen` gravou cada uma na forma que a origem
    entende. Adivinhar aqui reintroduziria o problema que o campo existe para
    evitar.
    """
    texto = str(ref).strip()
    _exigir(bool(texto), "referência de pixel vazia")

    casamento = _REF_LINHA_HF.match(texto)
    if casamento is not None:
        return Referencia(
            tipo="linha_hf",
            bruta=texto,
            dataset=casamento.group("ds"),
            split=casamento.group("split"),
            linha=int(casamento.group("row")),
            coluna=casamento.group("col"),
        )
    if _REF_COLUNA.match(texto):
        return Referencia(tipo="coluna", bruta=texto, coluna=texto)
    if "#" in texto or "[" in texto:
        raise ReleaseError(
            f"referência {texto!r} parece a forma `repo#split[linha].coluna` mas não "
            "casa com ela. Referência malformada é erro: adivinhar aqui é como uma "
            "amostra acaba apontando para o pixel errado sem que nada denuncie."
        )
    return Referencia(tipo="caminho", bruta=texto, caminho=texto)


# =============================================================================
# A árvore de arquivos
# =============================================================================

def _linhas_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    saida: list[dict] = []
    with path.open(encoding="utf-8") as h:
        for numero, linha in enumerate(h, 1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                saida.append(json.loads(linha))
            except json.JSONDecodeError as exc:
                raise ReleaseError(
                    f"{path}:{numero} não é JSON: {exc}. Manifesto truncado é "
                    "release pela metade — a contagem mentiria."
                ) from exc
    return saida


def resolver_raiz_do_release(
    origem: str,
    *,
    permitir_download: bool = False,
    baixador: Callable[..., str] | None = None,
) -> Path:
    """Caminho local do release: um diretório em disco ou um snapshot do Hub.

    **D-R3.** O default é `local_files_only=True`: o snapshot é lido do cache do
    `huggingface_hub` que já estiver em disco, e baixar é opt-in
    (`permitir_download=True`). O motivo está no cabeçalho do módulo — 552 GB de
    egress inexplicado numa auditoria deste projeto. Um download que acontece
    por default é um download que ninguém decidiu.
    """
    caminho = Path(os.path.expanduser(str(origem)))
    if caminho.is_dir():
        return caminho

    if baixador is None:
        from huggingface_hub import snapshot_download as baixador  # type: ignore

    try:
        local = baixador(
            repo_id=str(origem),
            repo_type="dataset",
            local_files_only=not permitir_download,
        )
    except Exception as exc:
        raise ReleaseError(
            f"não consegui obter o release {origem!r} localmente: "
            f"{type(exc).__name__}: {exc}.\n"
            + (
                ""
                if permitir_download
                else "O download está DESLIGADO por default (ver D-R3 em "
                "`release.py`). Para baixar, declare "
                "`permitir_download_do_release: true` no YAML — "
                "explicitamente, e o valor vai para o run_metadata."
            )
        ) from exc
    return Path(local)


class ArvoreDeRelease:
    """Um release do `bokehnet-regen` em disco, lido como árvore de arquivos.

    Não carrega pixel nenhum na construção: lê o `manifest.jsonl` (texto) e o
    `split.json`. Os `meta/<id>.json` são lidos sob demanda — ver D-R1.
    """

    def __init__(self, raiz: str | Path, *, rotulo: str | None = None) -> None:
        self.raiz = Path(raiz)
        self.rotulo = rotulo or str(raiz)
        _exigir(
            self.raiz.is_dir(), f"release {self.rotulo!r}: {self.raiz} não é diretório."
        )

        caminho_manifesto = self.raiz / NOME_MANIFESTO
        _exigir(
            caminho_manifesto.is_file(),
            f"release {self.rotulo!r}: {NOME_MANIFESTO} ausente em {self.raiz}. "
            "Sem manifesto não há release — é o arquivo que define o que existe.",
        )
        self.linhas = _linhas_jsonl(caminho_manifesto)
        _exigir(
            bool(self.linhas),
            f"release {self.rotulo!r}: {NOME_MANIFESTO} vazio.",
        )

        self.por_id: dict[str, dict] = {}
        repetidos: list[str] = []
        for linha in self.linhas:
            sid = str(linha.get("sample_id", ""))
            _exigir(bool(sid), f"release {self.rotulo!r}: linha sem `sample_id`.")
            if sid in self.por_id:
                repetidos.append(sid)
            self.por_id[sid] = linha
        _exigir(
            not repetidos,
            f"release {self.rotulo!r}: {len(repetidos)} `sample_id` repetidos no "
            f"manifesto (ex.: {repetidos[:3]}). A contagem estaria inflada e um "
            "arquivo sobrescreveria o outro em silêncio.",
        )

        self._layout: dict[str, bool] = {}       # subpasta -> agrupada por cena
        self._meta_cache: dict[str, dict] = {}
        self._ledger: dict[str, dict] | None = None

    # -- construção alternativa ------------------------------------------------

    @classmethod
    def abrir(
        cls,
        origem: str,
        *,
        permitir_download: bool = False,
        baixador: Callable[..., str] | None = None,
    ) -> "ArvoreDeRelease":
        raiz = resolver_raiz_do_release(
            origem, permitir_download=permitir_download, baixador=baixador
        )
        return cls(raiz, rotulo=str(origem))

    @staticmethod
    def parece_arvore(origem: str) -> bool:
        """`True` se `origem` é um diretório local com `manifest.jsonl`.

        Só decide sobre o que dá para decidir **sem rede**. Para um repo do Hub,
        quem decide é `release_format` no YAML (ou o snapshot já em disco).
        """
        caminho = Path(os.path.expanduser(str(origem)))
        return caminho.is_dir() and (caminho / NOME_MANIFESTO).is_file()

    # -- layout ----------------------------------------------------------------

    def caminho_por_amostra(self, sub: str, nome_do_arquivo: str, scene_id: str) -> Path:
        """Caminho de um arquivo por amostra, aceitando os DOIS layouts (D-R2).

        Plano (`depth/<id>.png`) é o que o `FileSampleWriter` escreve em disco;
        agrupado por cena (`depth/<scene_id>/<id>.png`) é o que
        `publish_release.py` publica quando a pasta passa de 10.000 arquivos, que
        é o limite do Hub. Os dois existem de verdade, então os dois são lidos.
        """
        _exigir(
            sub in SUBPASTAS_POR_AMOSTRA,
            f"subpasta {sub!r} não é uma das do release: {SUBPASTAS_POR_AMOSTRA}",
        )
        plano = self.raiz / sub / nome_do_arquivo
        agrupado = self.raiz / sub / str(scene_id) / nome_do_arquivo

        lembrado = self._layout.get(sub)
        if lembrado is True and agrupado.is_file():
            return agrupado
        if lembrado is False and plano.is_file():
            return plano

        if plano.is_file():
            self._layout[sub] = False
            return plano
        if agrupado.is_file():
            self._layout[sub] = True
            return agrupado
        return plano if lembrado is not True else agrupado

    # -- escalares -------------------------------------------------------------

    def metadados(self, sample_id: str) -> dict:
        """`meta/<id>.json`, memorizado. É a FONTE dos escalares (D-R1)."""
        em_cache = self._meta_cache.get(sample_id)
        if em_cache is not None:
            return em_cache
        linha = self.por_id.get(sample_id)
        _exigir(
            linha is not None,
            f"release {self.rotulo!r}: {sample_id!r} não está no manifesto.",
        )
        caminho = self.caminho_por_amostra(
            "meta", f"{sample_id}.json", str(linha.get("scene_id", ""))
        )
        _exigir(
            caminho.is_file(),
            f"release {self.rotulo!r}: {sample_id!r} está no manifesto e não tem "
            f"`meta/`. Esperado em {caminho}. Release incompleto — "
            "`publish_release.py` reprova exatamente isto.",
        )
        meta = json.loads(caminho.read_text(encoding="utf-8"))
        self._meta_cache[sample_id] = meta
        return meta

    def registro(self, sample_id: str) -> dict:
        """Escalares da amostra: manifesto + `meta/`, com o `meta/` vencendo.

        O manifesto é a cópia para varredura rápida e o `meta/` é a fonte; quando
        os dois trazem a mesma chave, quem vale é a fonte. Divergência entre eles
        seria defeito do release, e quem a detecta é `publish_release.py`.
        """
        linha = self.por_id.get(sample_id)
        _exigir(
            linha is not None,
            f"release {self.rotulo!r}: {sample_id!r} não está no manifesto.",
        )
        return {**linha, **self.metadados(sample_id)}

    # -- split -----------------------------------------------------------------

    @property
    def caminho_do_split(self) -> Path:
        return self.raiz / NOME_SPLIT

    @property
    def tem_split(self) -> bool:
        return self.caminho_do_split.is_file()

    def split_por_cena(self, particao: str) -> frozenset[str]:
        """Os `scene_id` de uma partição do `split.json` MATERIALIZADO.

        O split é lido do release, nunca sorteado aqui. Sorteado, ele muda a cada
        run e "o benchmark está fora do treino" deixa de ser verificável — e a
        mesma cena aparece com 2 a 21 aberturas, então split por linha vaza a
        cena inteira.
        """
        _exigir(
            self.tem_split,
            f"release {self.rotulo!r}: {NOME_SPLIT} ausente. O split materializado "
            "é requisito do release (`dataio/split.py`), e sem ele o split por "
            "linha vazaria a cena.",
        )
        return carregar_assignment_de_split(self.caminho_do_split, particao)

    def verificar_split(self) -> None:
        """O `split` gravado em cada linha tem que bater com o `split.json`.

        É a versão de leitura do `dataio/split.py:check_no_leak`. O que ela pega:
        o release foi gerado com um split e alguém trocou o arquivo depois; duas
        execuções usaram `salt` diferente; uma cena mudou de lado entre lotes.
        Qualquer um dos três faz a validação medir memorização, e nenhum aparece
        na loss.
        """
        if not self.tem_split:
            return
        atribuicao = json.loads(self.caminho_do_split.read_text(encoding="utf-8"))
        mapa = atribuicao.get("assignment") if isinstance(atribuicao, dict) else None
        if not isinstance(mapa, dict):
            return
        divergentes: list[str] = []
        sem_split: list[str] = []
        for linha in self.linhas:
            gravado = linha.get("split")
            cena = str(linha.get("scene_id", ""))
            if gravado is None:
                sem_split.append(str(linha.get("sample_id", "?")))
                continue
            esperado = mapa.get(cena)
            if esperado is not None and str(gravado) != str(esperado):
                divergentes.append(cena)
        if divergentes:
            raise ReleaseError(
                f"release {self.rotulo!r}: {len(set(divergentes))} cenas cujo `split` "
                f"gravado no manifesto diverge do {NOME_SPLIT} "
                f"(ex.: {sorted(set(divergentes))[:3]}). Um dos dois está errado, e "
                "treinar assim mede memorização."
            )
        if sem_split:
            print(
                f"[release/{self.rotulo}] AVISO: {len(sem_split)} linhas do manifesto "
                "sem campo `split`; a checagem de coerência ficou parcial.",
                flush=True,
            )

    # -- pixels ----------------------------------------------------------------

    @property
    def autocontido(self) -> bool:
        """`source/` existe e tem arquivo. Mesma definição do `publish_release.py`."""
        pasta = self.raiz / PASTA_AUTOCONTIDA
        return pasta.is_dir() and any(pasta.iterdir())

    def ledger_de_origem(self) -> dict[str, dict]:
        """`source_images.jsonl` indexado pela UNIDADE que a rota usou.

        DEFEITO CORRIGIDO (revisão adversarial, 2026-09-17): indexava só por
        `sample_id`, e a **rota A grava por `scene_id`** — porque as 41 variantes
        de uma cena saem dos MESMOS bytes de AIF, então um sha por amostra seria
        41 cópias do mesmo hash.

        `publish_release.py` documenta isso literalmente:
        `unidade_do_ledger_de_origem = "scene_id"` para a rota A, `"sample_id"`
        para a C — *"Ler as duas com a mesma chave é o `KeyError: 'sample_id'`
        que derrubava a rota A."*

        Consequência do defeito: na rota A — **que é o dataset da fase 1** — o
        ledger vinha VAZIO, `esperado_sha` era sempre `None`, e a AIF era lida do
        espelho **sem nenhuma conferência**. Um espelho trocado, truncado ou de
        outra revisão treinava em silêncio, com `verificar_sha256_da_origem: true`
        no YAML dando falsa sensação de verificação.

        Agora indexa pelas DUAS chaves. Elas não colidem: uma amostra é
        `<scene>_vNN` e a cena é `<scene>`; e mesmo que colidissem, o consumidor
        pergunta primeiro pelo `sample_id`.
        """
        if self._ledger is None:
            ledger: dict[str, dict] = {}
            for linha in _linhas_jsonl(self.raiz / NOME_LEDGER_ORIGEM):
                for chave in ("sample_id", "scene_id"):
                    if chave in linha and linha[chave] is not None:
                        ledger.setdefault(str(linha[chave]), linha)
            self._ledger = ledger
        return self._ledger

    def caminho_da_profundidade(self, sample_id: str) -> Path:
        """Caminho do PNG de profundidade, honrando `depth_ref` quando existe.

        MUDANÇA DO RELEASE (medida em 2026-09-19, rota A): a profundidade deixou
        de ser POR AMOSTRA e passou a ser POR CENA — `depth/<scene_id>.png`, uma
        só para as 41 variantes. Faz sentido: as variantes de uma cena saem da
        MESMA AIF, logo do mesmo Depth Pro. Antes eram 41 cópias idênticas, e o
        release completo teria ~32 GB de duplicação.

        O release não deixa isso implícito: cada amostra traz
        `depth_ref: "depth/<scene>.png"` e `depth_layout: "per_scene"`. Então a
        regra aqui é: **o que o metadado diz ganha**; a construção por convenção
        é só o fallback para releases antigos, que endereçavam por `sample_id`.
        Adivinhar o layout quando o dado o declara é a classe de defeito que
        este projeto persegue.
        """
        linha = self.por_id[sample_id]
        cena = str(linha.get("scene_id", ""))

        # 1) o que o metadado DIZ — manifesto primeiro, meta/ depois
        ref = linha.get("depth_ref")
        if not ref:
            try:
                ref = self.metadados(sample_id).get("depth_ref")
            except Exception:  # noqa: BLE001 — meta/ ausente cai no fallback
                ref = None
        if ref:
            caminho = self.raiz / str(ref)
            _exigir(
                caminho.is_file(),
                f"release {self.rotulo!r}: `depth_ref`={ref!r} para {sample_id!r} "
                f"não existe em disco ({caminho}). O metadado e o release "
                "discordam — não vou adivinhar outro caminho.",
            )
            return caminho

        # 2) fallback: convenção. Por CENA e depois por AMOSTRA, porque o
        #    release novo é por cena e o antigo era por amostra.
        for nome in (f"{cena}.png", f"{sample_id}.png"):
            if not nome.startswith(".png"):
                caminho = self.caminho_por_amostra("depth", nome, cena)
                if caminho.is_file():
                    return caminho
        _exigir(
            False,
            f"release {self.rotulo!r}: profundidade ausente para {sample_id!r}. "
            f"Procurei por `depth_ref` no manifesto e no meta/, e por convenção "
            f"em depth/{cena}.png e depth/{sample_id}.png.",
        )
        raise AssertionError("inalcançável")

    def caminho_gerado(self, sample_id: str, papel: str) -> Path | None:
        """`generated/<id>_<papel>.jpg`, ou `None` se esta rota não gerou o papel.

        Quem declara o que foi gerado é `meta["generated_images"]`
        (`dataio/sample.py:metadata`), não a existência do arquivo: perguntar ao
        metadado é o que separa "a rota não gera isso" de "o arquivo sumiu" — e a
        segunda tem que ser erro.
        """
        _exigir(papel in PAPEIS, f"papel desconhecido: {papel!r} (esperado {PAPEIS})")
        meta = self.metadados(sample_id)
        gerados = meta.get("generated_images")
        if not isinstance(gerados, (list, tuple)) or papel not in gerados:
            return None
        caminho = self.caminho_por_amostra(
            "generated", f"{sample_id}_{papel}.jpg", str(meta.get("scene_id", ""))
        )
        _exigir(
            caminho.is_file(),
            f"release {self.rotulo!r}: {sample_id!r} declara ter gerado {papel!r} "
            f"mas {caminho} não existe. Release incompleto.",
        )
        return caminho


def carregar_assignment_de_split(caminho: str | Path, particao: str) -> frozenset[str]:
    """Lê o `split.json` do `dataio/split.py:SceneSplit.to_json`.

        {"salt": ..., "val_fraction": ..., "counts": {...},
         "assignment": {"<scene_id>": "train" | "val"}}

    Aceita também um mapa direto `{"train": [...], "val": [...]}`, que é a forma
    que `data.carregar_split_por_cena` já lia. As duas convivem porque o release
    grava a primeira e alguns manifestos internos usam a segunda.
    """
    caminho = Path(os.path.expanduser(str(caminho)))
    _exigir(
        caminho.is_file(),
        f"manifesto de split por cena não encontrado: {caminho}. Ele é gerado junto "
        "do release, por `dataio/split.py`.",
    )
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    _exigir(isinstance(payload, dict), f"{caminho}: split.json não é um objeto.")

    atribuicao = payload.get("assignment")
    if isinstance(atribuicao, dict):
        cenas = frozenset(
            str(cena) for cena, lado in atribuicao.items() if str(lado) == str(particao)
        )
        _exigir(
            bool(cenas),
            f"{caminho}: nenhuma cena na partição {particao!r} "
            f"(lados presentes: {sorted(set(map(str, atribuicao.values())))}). "
            "Partição vazia é o jeito mais silencioso de não ter validação.",
        )
        return cenas

    if particao in payload and isinstance(payload[particao], (list, tuple)):
        return frozenset(str(c) for c in payload[particao])

    raise ReleaseError(
        f"{caminho}: não reconheci o formato do split. Esperado "
        '`{"assignment": {cena: lado}}` (o do release) ou `{"train": [...]}`.'
    )


# =============================================================================
# O resolvedor de referências
# =============================================================================

@dataclass(frozen=True)
class PixelResolvido:
    """Uma imagem e de ONDE ela veio. A proveniência não é decoração.

    `origem` é vocabulário FECHADO: `"generated"`, `"source"`, `"espelho"`.
    `sha256_conferido` é `None` quando não havia sha no ledger para conferir —
    diferente de `False`, que seria "conferi e não bateu" (esse caso levanta).
    """

    imagem: Image.Image
    origem: str
    caminho: str
    sha256_conferido: bool | None


class ResolvedorDePixels:
    """Dado `source_dataset` + `source_sample_id` + `aif_ref`/`bokeh_ref`, acha
    os bytes. Nesta ordem: `generated/` → `source/` → espelho de origem.

    `espelhos` mapeia `source_dataset` -> diretório do snapshot local. A chave é
    o nome do dataset de origem exatamente como o release o grava
    (ex.: `akcit-pixel/RealBokeh`), porque é o único identificador que a amostra
    carrega.

    **Nada aqui inventa pixel.** Se as três origens falham, levanta
    `ReleaseError` dizendo qual das três faltou e o que destravaria — que é a
    regra 4 do `CONTRATO.md` do lado de leitura.
    """

    def __init__(
        self,
        arvore: ArvoreDeRelease,
        *,
        espelhos: Mapping[str, str] | None = None,
        verificar_sha256: bool = True,
        permitir_download: bool = False,
    ) -> None:
        self.arvore = arvore
        self.espelhos = {str(k): str(v) for k, v in (espelhos or {}).items()}
        self.verificar_sha256 = bool(verificar_sha256)
        self.permitir_download = bool(permitir_download)
        self._indices: dict[str, IndiceEspelho] = {}
        self._tabelas: dict[str, Any] = {}
        self._datasets_hf: dict[str, Any] = {}
        #: contagem por origem, para o log do início do run
        self.contagem_por_origem: dict[str, int] = {
            "generated": 0, "source": 0, "espelho": 0
        }
        self.sem_sha_para_conferir = 0

    # -- API -------------------------------------------------------------------

    def resolver(self, sample_id: str, papel: str) -> PixelResolvido:
        _exigir(papel in PAPEIS, f"papel desconhecido: {papel!r} (esperado {PAPEIS})")
        meta = self.arvore.metadados(sample_id)

        gerado = self.arvore.caminho_gerado(sample_id, papel)
        if gerado is not None:
            self.contagem_por_origem["generated"] += 1
            return PixelResolvido(
                _abrir_rgb(gerado.read_bytes(), sample_id=sample_id, papel=papel,
                           de=str(gerado)),
                "generated", str(gerado), None,
            )

        # A1 — a rota A registra o sha da AIF por CENA (as 41 variantes saem dos
        # mesmos bytes), a rota C por amostra. Perguntamos pelo `sample_id` e,
        # se não houver linha, pela cena — nessa ordem, porque o mais específico
        # ganha.
        ledger = self.arvore.ledger_de_origem()
        linha_ledger = ledger.get(sample_id)
        if linha_ledger is None:
            cena = str(meta.get("scene_id", "") or "")
            if cena:
                linha_ledger = ledger.get(cena)
        esperado_sha = (linha_ledger or {}).get(f"{papel}_sha256")

        do_source = self._caminho_no_source(sample_id, papel, meta, linha_ledger)
        if do_source is not None:
            dados = do_source.read_bytes()
            conferido = self._conferir(dados, esperado_sha, sample_id, papel,
                                       str(do_source))
            self.contagem_por_origem["source"] += 1
            return PixelResolvido(
                _abrir_rgb(dados, sample_id=sample_id, papel=papel, de=str(do_source)),
                "source", str(do_source), conferido,
            )

        dados, de = self._bytes_do_espelho(sample_id, papel, meta)
        conferido = self._conferir(dados, esperado_sha, sample_id, papel, de)
        self.contagem_por_origem["espelho"] += 1
        return PixelResolvido(
            _abrir_rgb(dados, sample_id=sample_id, papel=papel, de=de),
            "espelho", de, conferido,
        )

    def resumo(self) -> str:
        total = sum(self.contagem_por_origem.values())
        partes = " · ".join(
            f"{k}={v}" for k, v in self.contagem_por_origem.items() if v
        )
        extra = (
            f" · {self.sem_sha_para_conferir} sem sha no ledger"
            if self.sem_sha_para_conferir else ""
        )
        return f"[release/{self.arvore.rotulo}] {total} pixels resolvidos: {partes}{extra}"

    # -- `source/` -------------------------------------------------------------

    def _caminho_no_source(
        self, sample_id: str, papel: str, meta: dict, linha_ledger: dict | None
    ) -> Path | None:
        """O arquivo do release autocontido, ou `None` se não é autocontido.

        O NOME do arquivo vem do ledger (`aif_file`/`bokeh_file`), que é quem o
        gravou. As convenções são duas e diferem entre fontes: a RealBokeh grava
        `<sample_id>_<papel>`, e o LFDOF grava a AIF UMA vez por cena
        (`<scene_id>_aif`), porque ela é byte a byte idêntica nos N níveis e
        gravá-la por amostra custaria ~13 GB contra ~1 GB. Adivinhar o nome
        acertaria numa fonte e erraria na outra — por isso o ledger manda, e os
        palpites só existem para um release sem ledger.
        """
        pasta = self.arvore.raiz / PASTA_AUTOCONTIDA
        if not pasta.is_dir():
            return None

        nome = (linha_ledger or {}).get(f"{papel}_file")
        if nome:
            caminho = pasta / str(nome)
            _exigir(
                caminho.is_file(),
                f"release autocontido {self.arvore.rotulo!r}: o ledger diz que "
                f"{sample_id!r}/{papel} está em {caminho}, e o arquivo não existe.",
            )
            return caminho

        cena = str(meta.get("scene_id", ""))
        for base in (f"{sample_id}_{papel}", f"{cena}_{papel}"):
            for ext in (".jpg", ".png", ".bin"):
                candidato = pasta / f"{base}{ext}"
                if candidato.is_file():
                    return candidato
        return None

    # -- espelho ---------------------------------------------------------------

    def _bytes_do_espelho(
        self, sample_id: str, papel: str, meta: dict
    ) -> tuple[bytes, str]:
        ref_bruta = meta.get(f"{papel}_ref")
        source_dataset = str(meta.get("source_dataset") or "")
        source_sample_id = str(meta.get("source_sample_id") or "")
        _exigir(
            bool(ref_bruta),
            f"amostra {sample_id!r}: sem `{papel}_ref` e sem os pixels no release. "
            "O pixel é irrecuperável — a amostra não é 'incompleta', o release é.",
        )
        _exigir(
            bool(source_dataset),
            f"amostra {sample_id!r}: `{papel}_ref`={ref_bruta!r} sem `source_dataset`. "
            "Uma referência sem a origem não aponta para lugar nenhum.",
        )

        raiz = self.espelhos.get(source_dataset)
        ref = parse_referencia(str(ref_bruta))

        if ref.tipo == "linha_hf":
            return self._bytes_de_linha_hf(sample_id, papel, ref, raiz)

        _exigir(
            raiz is not None,
            f"amostra {sample_id!r}: os pixels de {papel!r} são REFERÊNCIA a "
            f"{source_dataset!r} ({ref_bruta!r}) e este release não é autocontido.\n"
            "Duas saídas, as duas explícitas:\n"
            f"  (a) aponte o espelho local com `mirror_roots: {{{source_dataset!r}: "
            "'/caminho/do/snapshot'}}` no YAML;\n"
            "  (b) republique o release AUTOCONTIDO (--store-source-images), e a "
            "resolução por espelho deixa de ser necessária.\n"
            "O que NÃO é saída: substituir por outra imagem.",
        )
        caminho_raiz = Path(os.path.expanduser(str(raiz)))

        if ref.tipo == "caminho":
            arquivo = caminho_raiz / str(ref.caminho)
            _exigir(
                arquivo.is_file(),
                f"amostra {sample_id!r}: `{papel}_ref`={ref.bruta!r} aponta para "
                f"{arquivo}, que não existe no espelho de {source_dataset!r}.",
            )
            return arquivo.read_bytes(), str(arquivo)

        # ref.tipo == "coluna": a linha vem do `source_sample_id`, via índice.
        _exigir(
            bool(source_sample_id),
            f"amostra {sample_id!r}: `{papel}_ref`={ref.bruta!r} é um NOME DE COLUNA "
            "e a linha é identificada por `source_sample_id`, que está ausente.",
        )
        indice = self._indice(source_dataset, caminho_raiz)
        local = indice.locate(source_sample_id)
        tabela = self._tabela(caminho_raiz, local.shard)

        # A checagem que fecha o buraco do índice cacheado de OUTRO snapshot: ali
        # o índice aponta para a linha errada de um shard remontado e TODOS os
        # pares saem trocados, com o histograma limpo. É uma comparação de string.
        na_linha = tabela.column(COLUNA_NOME_NO_ESPELHO)[local.row].as_py()
        _exigir(
            str(na_linha) == source_sample_id,
            f"amostra {sample_id!r}: o índice aponta {local.shard} linha {local.row}, "
            f"que contém {na_linha!r} e não {source_sample_id!r}. Índice cacheado de "
            "outro snapshot — reconstrua com IndiceEspelho.carregar(..., forcar=True).",
        )
        _exigir(
            str(ref.coluna) in tabela.column_names,
            f"amostra {sample_id!r}: coluna {ref.coluna!r} não existe no shard "
            f"{local.shard} (tem {list(tabela.column_names)[:8]}…).",
        )
        celula = tabela.column(str(ref.coluna))[local.row].as_py()
        dados = _bytes_da_celula(
            celula, raiz=caminho_raiz, sample_id=sample_id, coluna=str(ref.coluna)
        )
        return dados, f"{source_dataset}:{local.shard}[{local.row}].{ref.coluna}"

    def _bytes_de_linha_hf(
        self, sample_id: str, papel: str, ref: Referencia, raiz: str | None
    ) -> tuple[bytes, str]:
        """`repo#split[linha].coluna` — a forma da rota B (bokeh do ITW).

        A unidade de origem ali é o ÍNDICE da linha, não um nome, então o índice
        do espelho não se aplica: quem endereça é o próprio `datasets`.
        """
        import datasets as _datasets

        chave = f"{ref.dataset}#{ref.split}"
        ds = self._datasets_hf.get(chave)
        if ds is None:
            if raiz:
                ds = _datasets.load_dataset(
                    str(Path(os.path.expanduser(raiz))), split=str(ref.split)
                )
            else:
                _exigir(
                    self.permitir_download,
                    f"amostra {sample_id!r}: `{papel}_ref`={ref.bruta!r} aponta para a "
                    f"linha {ref.linha} de {ref.dataset!r}, que não está em disco. "
                    "Aponte o snapshot com `mirror_roots`, ou declare "
                    "`permitir_download_do_release: true` — baixar não acontece por "
                    "default (ver D-R3 em `release.py`).",
                )
                ds = _datasets.load_dataset(str(ref.dataset), split=str(ref.split))
            # ── A2 — os BYTES ORIGINAIS, sem redecodificar ───────────────────
            # `datasets` decodifica a coluna `Image` para PIL. Reencodar o PIL
            # com `.save(format="JPEG")` usa quality=75 por default, e MEDIDO:
            # 20.784 -> 10.537 bytes, PSNR 36,9 dB, maxabs 19. Na rota B a bokeh
            # é a FOTOGRAFIA REAL, ou seja, o ALVO do treino — perder 37 dB do
            # ground truth em silêncio é inaceitável.
            # Com `decode=False` a célula vira {"bytes","path"} e o sha bate byte
            # a byte com a origem, que é o que torna o ledger uma evidência.
            try:
                ds = ds.cast_column(str(ref.coluna), _datasets.Image(decode=False))
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[release] AVISO: não consegui pedir os bytes crus de "
                    f"{ref.coluna!r} ({exc}). Vou cair na decodificação, e aí o "
                    "ALVO pode ser recomprimido. Confira a versão do `datasets`.",
                    flush=True,
                )
            self._datasets_hf[chave] = ds

        # `ref.linha is None`, e NÃO `ref.linha or -1`: em Python `0 or -1` é
        # `-1`, porque zero é falsy. A guarda rejeitava a LINHA 0, que é
        # perfeitamente válida — e a mensagem saía se contradizendo ("pede a
        # linha 0 de ..., que tem 15305 linhas"). Custou 1.764 steps de smoke
        # para aparecer, porque só UMA amostra em 13.765 endereça a linha 0.
        _exigir(
            ref.linha is not None and 0 <= int(ref.linha) < len(ds),
            f"amostra {sample_id!r}: `{papel}_ref` pede a linha {ref.linha} de "
            f"{chave!r}, que tem {len(ds)} linhas. Snapshot de outra revisão — a "
            "linha não é o mesmo pixel.",
        )
        celula = ds[int(ref.linha)][str(ref.coluna)]
        de = f"{chave}[{ref.linha}].{ref.coluna}"
        # caminho normal: `decode=False` devolve os bytes do arquivo de origem
        if isinstance(celula, dict) and celula.get("bytes") is not None:
            return bytes(celula["bytes"]), de
        if isinstance(celula, (bytes, bytearray)):
            return bytes(celula), de
        if isinstance(celula, Image.Image):
            # fallback: o `cast_column` falhou. Reencodar é LOSSY para JPEG, e o
            # alvo do treino não pode perder qualidade em silêncio — então
            # gravamos em PNG (sem perda) e avisamos que o sha não vai bater.
            print(
                f"[release] AVISO: {sample_id!r}/{papel} veio DECODIFICADO; "
                "reencodando sem perda (PNG). O sha256 não baterá com a origem.",
                flush=True,
            )
            buffer = io.BytesIO()
            celula.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue(), de
        dados = _bytes_da_celula(
            celula,
            raiz=Path(os.path.expanduser(str(raiz))) if raiz else Path("."),
            sample_id=sample_id,
            coluna=str(ref.coluna),
        )
        return dados, f"{chave}[{ref.linha}].{ref.coluna}"

    def _indice(self, source_dataset: str, raiz: Path) -> IndiceEspelho:
        indice = self._indices.get(source_dataset)
        if indice is None:
            indice = IndiceEspelho.carregar(raiz)
            self._indices[source_dataset] = indice
        return indice

    def _tabela(self, raiz: Path, shard: str):
        """Uma tabela parquet aberta por vez.

        O dataloader não controla a ordem em que as amostras chegam (quem a
        define é o sampler), então um cache maior não ajudaria de forma previsível
        e custaria memória por worker. Quem varre o release em lote deve ordenar
        com `IndiceEspelho.sort_key` antes.
        """
        em_cache = self._tabelas.get(shard)
        if em_cache is not None:
            return em_cache
        import pyarrow.parquet as pq

        tabela = pq.read_table(raiz / shard)
        self._tabelas = {shard: tabela}
        return tabela

    # -- sha256 ----------------------------------------------------------------

    def _conferir(
        self, dados: bytes, esperado: str | None, sample_id: str, papel: str, de: str
    ) -> bool | None:
        """Confere os bytes contra o `source_images.jsonl`.

        É o que transforma "o release aponta para um espelho privado" em
        evidência: sem isso, ninguém consegue provar que o K foi calibrado contra
        ESTES bytes — que é a razão de o ledger existir
        (`mirror_images.MirrorImageLoader`, comentário do construtor).
        """
        if not esperado:
            self.sem_sha_para_conferir += 1
            return None
        if not self.verificar_sha256:
            return None
        obtido = hashlib.sha256(dados).hexdigest()
        _exigir(
            obtido == str(esperado),
            f"amostra {sample_id!r}, papel {papel!r}: sha256 {obtido[:16]}… lido de "
            f"{de} não bate com {str(esperado)[:16]}… do {NOME_LEDGER_ORIGEM}. Os "
            "bytes não são os que rotularam esta amostra — o K foi calibrado contra "
            "OUTRA imagem. Amostra recusada, não ajustada.",
        )
        return True


def _abrir_rgb(dados: bytes, *, sample_id: str, papel: str, de: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(dados)) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise ReleaseError(
            f"amostra {sample_id!r}, papel {papel!r}: os bytes de {de} não "
            f"decodificam como imagem ({type(exc).__name__}: {exc})."
        ) from exc


def _bytes_da_celula(celula: Any, *, raiz: Path, sample_id: str, coluna: str) -> bytes:
    """A célula `{bytes, path}` do `datasets`/parquet, ou bytes crus."""
    if isinstance(celula, (bytes, bytearray)):
        return bytes(celula)
    if isinstance(celula, dict):
        dados = celula.get("bytes")
        if dados:
            return bytes(dados)
        caminho = celula.get("path")
        _exigir(
            bool(caminho),
            f"amostra {sample_id!r}: coluna {coluna!r} sem `bytes` nem `path`.",
        )
        arquivo = Path(str(caminho))
        if not arquivo.is_absolute():
            arquivo = raiz / arquivo
        _exigir(
            arquivo.is_file(),
            f"amostra {sample_id!r}: coluna {coluna!r} aponta para {arquivo}, que "
            "não existe.",
        )
        return arquivo.read_bytes()
    raise ReleaseError(
        f"amostra {sample_id!r}: coluna {coluna!r} veio como "
        f"{type(celula).__name__}, que não é imagem nem célula {{bytes, path}}."
    )


# =============================================================================
# Pré-voo
# =============================================================================

def conferir_cobertura_do_espelho(
    arvore: ArvoreDeRelease,
    resolvedor: ResolvedorDePixels,
    sample_ids: Iterable[str],
) -> None:
    """Confere TODOS os nomes contra o índice do espelho, antes do treino.

    O `preflight_de_pixels` resolve 8 amostras espalhadas — suficiente para pegar
    um espelho que não foi montado, e insuficiente para pegar um espelho
    INCOMPLETO. Medido em 2026-09-25: o snapshot local da RealBokeh tinha só o
    split `test` (1.257 de 15.423 nomes, cobertura 5,3%); ali a falta era tão
    grande que as 8 amostras a encontraram, mas um buraco de 5% passaria pelas
    oito e apareceria no step 8.000.

    Esta checagem é de CONJUNTO: nenhum pixel é decodificado, nenhum parquet é
    lido além do índice que já está em memória. Custa um passe pelo manifesto e
    responde a pergunta certa — "o espelho cobre este release?" — em vez de
    "estas oito amostras carregam?".

    Só vale para a forma `coluna` (rotas C), que é a que passa pelo índice. A
    forma `linha_hf` (rota B) usa o número da linha direto, e `caminho` (rota A)
    resolve no sistema de arquivos: nas duas, uma ausência já falha nomeada.
    """
    faltando: dict[str, list[str]] = {}
    total = 0
    for sid in sample_ids:
        try:
            meta = arvore.metadados(sid)
        except ReleaseError:
            continue
        fonte = str(meta.get("source_dataset") or "")
        nome = str(meta.get("source_sample_id") or "")
        refs = [meta.get("aif_ref"), meta.get("bokeh_ref")]
        if not any(isinstance(r, str) and parse_referencia(r).tipo == "coluna"
                   for r in refs if r):
            continue
        raiz = resolvedor.espelhos.get(fonte)
        if raiz is None:
            continue
        total += 1
        indice = resolvedor._indice(fonte, Path(os.path.expanduser(str(raiz))))
        if nome not in indice:
            faltando.setdefault(fonte, []).append(nome)

    if not faltando:
        if total:
            print(f"[release] espelho cobre {total}/{total} nomes.", flush=True)
        return

    linhas = []
    for fonte, nomes in sorted(faltando.items()):
        linhas.append(
            f"  {fonte}: {len(nomes)} de {total} nomes ausentes "
            f"(ex.: {nomes[:2]})"
        )
    raise ReleaseError(
        "o espelho de origem NÃO cobre este release:\n"
        + "\n".join(linhas)
        + "\n\nO índice do espelho é construído dos parquets em `mirror_roots`. "
        "Nome ausente significa snapshot incompleto — baixe os shards que faltam, "
        "ou aponte `mirror_roots` para um snapshot completo. Seguir daqui daria "
        "`KeyError` no meio do treino, numa amostra sorteada, depois de horas de "
        "GPU."
    )


def preflight_de_pixels(
    arvore: ArvoreDeRelease,
    resolvedor: ResolvedorDePixels,
    sample_ids: Iterable[str],
    *,
    n: int = 8,
) -> None:
    """Resolve alguns pixels ANTES do treino, para a falha ser no minuto 0.

    Um run que morre no step 8.000 porque o espelho não estava montado gastou
    8.000 steps de fila para descobrir uma linha de YAML. Isto resolve `n`
    amostras espalhadas pela lista e deixa o erro nomeado aparecer na partida.

    Espalhadas, e não as `n` primeiras: o manifesto sai agrupado por cena, e as
    8 primeiras amostras vêm de uma ou duas cenas — um prefixo testaria aquele
    canto do release, não o release.
    """
    ids = list(sample_ids)
    if not ids or n <= 0:
        return
    passo = max(1, len(ids) // int(n))
    for sid in ids[::passo][:int(n)]:
        for papel in PAPEIS:
            resolvedor.resolver(sid, papel)
    print(resolvedor.resumo(), flush=True)
