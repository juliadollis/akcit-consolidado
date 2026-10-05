"""Lê os pixels do espelho `akcit-pixel/RealBokeh`.

`sources/realbokeh.py` responde *quais pares existem*; este módulo responde *onde
estão os bytes*. A separação é de propósito: enumerar 20.495 pares é barato e roda
em qualquer máquina, decodificar 41.000 imagens de 1500×2000 não é.

## Por que existe um índice

O espelho são 85 shards parquet. `aif_ref` e `bokeh_ref` do `RealBokehPair` são
**nomes de coluna** (`image_focus`, `image_blur`), não caminhos — a linha é
identificada por `file_name_base`, que é o `source_sample_id`. Alguém tem que
traduzir `file_name_base -> (shard, linha)`, e esse alguém é o `MirrorIndex`.

O índice é construído lendo **só a coluna `file_name_base`** de cada shard, que é
texto: uns segundos e alguns MB, contra dezenas de GB se lêssemos as imagens. Fica
cacheado em JSON ao lado do snapshot.

## Por que a ordem importa

Parquet lê por row group. Pedir a linha 40.000 do shard 3, depois a linha 12 do
shard 70, depois a 40.001 do shard 3 de novo relê o mesmo row group duas vezes e joga
fora a página no meio. `order_pairs_for_sequential_read` reordena os pares por
`(shard, linha)` **antes** do laço da rota, e o loader mantém um shard aberto por vez.
Com isso a leitura vira quase sequencial e a memória fica constante.

Reordenar é seguro porque nada na rota C depende da ordem: `sample_id` é determinístico,
o split é por cena e materializado, e a retomada é por id. A única coisa que a ordem
afetaria é *quais* amostras entram num `--limit` de piloto — e por isso o piloto usa
`sample_pairs_for_pilot`, que sorteia por cena com seed em vez de pegar o prefixo de
uma ordem de disco.

## Decodificação

PIL, não OpenCV: o `cv2` do container falha com `GLIBC_2.38 not found`, e o renderer
já convive com isso (ver `renderer/bokehme.py`). Saída BGR uint8 HxWx3, que é o que
`routes/route_c.py` espera.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from control.contract import SampleRejected, reject

#: Coluna de texto que identifica a linha. É o `source_sample_id` do par.
NAME_COLUMN = "file_name_base"

#: Nome do cache do índice, dentro do próprio snapshot.
INDEX_FILENAME = "_mirror_index.json"


@dataclass(frozen=True)
class MirrorLocation:
    """Onde uma linha do espelho mora."""

    shard: str
    row: int


class MirrorIndex:
    """`file_name_base -> (shard, linha)`, construído uma vez e cacheado."""

    def __init__(self, locations: Mapping[str, MirrorLocation], *, shards: Sequence[str]):
        self._locations = dict(locations)
        #: ordem canônica dos shards, para ordenar leitura
        self._shard_rank = {name: i for i, name in enumerate(shards)}

    def __len__(self) -> int:
        return len(self._locations)

    def __contains__(self, name: object) -> bool:
        return name in self._locations

    def locate(self, name: str) -> MirrorLocation:
        try:
            return self._locations[name]
        except KeyError:
            raise KeyError(
                f"{name!r} não está no índice do espelho ({len(self._locations)} linhas). "
                "Índice desatualizado ou snapshot incompleto — reconstrua com "
                "MirrorIndex.build(..., force=True) antes de culpar o enumerador."
            ) from None

    def names(self) -> list[str]:
        """Nomes em ordem de leitura `(shard, linha)` — a ordem que o enumerador deve
        receber para que a leitura de parquet fique quase sequencial."""
        return sorted(self._locations, key=self.sort_key)

    def sort_key(self, name: str) -> tuple[int, int]:
        loc = self.locate(name)
        return (self._shard_rank[loc.shard], loc.row)

    # -- construção ------------------------------------------------------------

    @classmethod
    def build(cls, snapshot_dir: str | Path, *, force: bool = False) -> "MirrorIndex":
        """Lê só a coluna de nome de cada shard. Cacheia em `_mirror_index.json`.

        O caminho do cache não importa `pyarrow` de propósito: reler um índice já
        construído é só JSON, e vale poder fazer isso numa máquina sem o stack de
        parquet — inspecionar o índice não deveria exigir o ambiente do cluster.
        """
        root = Path(snapshot_dir)
        cache = root / INDEX_FILENAME
        if cache.is_file() and not force:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            locations = {name: MirrorLocation(shard, row)
                         for name, (shard, row) in payload["locations"].items()}
            return cls(locations, shards=payload["shards"])

        import pyarrow.parquet as pq

        shards = sorted(str(p.relative_to(root)) for p in root.rglob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"nenhum .parquet em {root}. Baixe o espelho antes:\n"
                "  huggingface-cli download akcit-pixel/RealBokeh --repo-type dataset "
                "--local-dir <dir>"
            )

        locations: dict[str, MirrorLocation] = {}
        duplicados: list[str] = []
        for shard in shards:
            table = pq.read_table(root / shard, columns=[NAME_COLUMN])
            for row, name in enumerate(table.column(NAME_COLUMN).to_pylist()):
                if name in locations:
                    duplicados.append(str(name))
                    continue
                locations[str(name)] = MirrorLocation(shard, row)
        if duplicados:
            raise ValueError(
                f"{len(duplicados)} file_name_base repetidos no espelho "
                f"(ex.: {duplicados[:3]}). O índice seria ambíguo — resolva na origem; "
                "escolher a primeira ocorrência em silêncio é o tipo de decisão que "
                "some no meio de 20.495 linhas."
            )

        cache.write_text(json.dumps(
            {"shards": shards,
             "locations": {n: [l.shard, l.row] for n, l in locations.items()}},
            indent=0), encoding="utf-8")
        return cls(locations, shards=shards)


class MirrorImageLoader:
    """`LoadPair` da rota C sobre o snapshot local do espelho.

    Mantém `cache_shards` tabelas abertas (default 1). Com os pares reordenados por
    `order_pairs_for_sequential_read`, 1 basta e a memória fica constante.
    """

    def __init__(self, snapshot_dir: str | Path, index: MirrorIndex, *,
                 cache_shards: int = 1, expected_hw: Optional[tuple[int, int]] = None,
                 ledger_path: Optional[str | Path] = None,
                 store_dir: Optional[str | Path] = None):
        self._root = Path(snapshot_dir)
        self._index = index
        self._cache_shards = max(1, int(cache_shards))
        self._expected_hw = expected_hw
        self._open: "OrderedDict[str, object]" = OrderedDict()

        # -- o que torna o release auditável -----------------------------------
        # A rota C não GERA pixel: a AIF e a bokeh vêm da origem. Sem registrar nada,
        # o release publicado seria rótulo solto apontando para um espelho privado
        # que pode mudar — e ninguém conseguiria provar que o K foi calibrado contra
        # ESTES bytes. O ledger grava o sha256 de cada imagem lida, com shard e linha.
        self._ledger_path = Path(ledger_path) if ledger_path else None
        self._ledger = None
        if self._ledger_path is not None:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._ledger = self._ledger_path.open("a", encoding="utf-8")

        # `store_dir` copia os bytes ORIGINAIS (sem recomprimir) para dentro do
        # release, tornando-o autocontido. Recomprimir seria falsificar a evidência:
        # o sha256 do ledger deixaria de bater com o arquivo ao lado dele.
        self._store_dir = Path(store_dir) if store_dir else None
        if self._store_dir is not None:
            self._store_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self._ledger is not None:
            self._ledger.close()
            self._ledger = None

    @staticmethod
    def _ext(data: bytes) -> str:
        """Extensão pelo magic number — o `path` da célula pode mentir."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:2] == b"\xff\xd8":
            return ".jpg"
        return ".bin"

    def _table(self, shard: str):
        import pyarrow.parquet as pq

        table = self._open.get(shard)
        if table is None:
            table = pq.read_table(self._root / shard)
            self._open[shard] = table
            while len(self._open) > self._cache_shards:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(shard)
        return table

    def _decode(self, data, *, sample_id: str, column: str) -> np.ndarray:
        """Bytes de imagem -> BGR uint8.

        Aceita também a célula `{bytes, path}` crua, para os testes e para quem
        chamar direto.
        """
        from PIL import Image

        if isinstance(data, dict):
            data = self._raw_bytes(data, sample_id=sample_id, column=column)
        try:
            with Image.open(io.BytesIO(data)) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        except Exception as exc:                        # bytes corrompidos no shard
            reject("source_image_unreadable",
                   f"{sample_id}: coluna {column!r} não decodifica ({exc})")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            reject("source_image_unreadable",
                   f"{sample_id}: coluna {column!r} decodificou com shape {rgb.shape}")
        if self._expected_hw is not None and rgb.shape[:2] != self._expected_hw:
            reject("source_image_unreadable",
                   f"{sample_id}: coluna {column!r} é {rgb.shape[:2]}, esperado "
                   f"{self._expected_hw}. Resolução heterogênea muda o K em pixel — "
                   "ver a regra de resolução em reference/CONTRATO.md")
        return np.ascontiguousarray(rgb[:, :, ::-1])     # RGB -> BGR, sem cv2

    def _raw_bytes(self, cell, *, sample_id: str, column: str) -> bytes:
        data = cell.get("bytes") if isinstance(cell, dict) else None
        if data is None:
            path = cell.get("path") if isinstance(cell, dict) else None
            if not path:
                reject("source_image_unreadable",
                       f"{sample_id}: coluna {column!r} sem bytes nem path")
            data = (self._root / path).read_bytes()
        return data

    def __call__(self, pair) -> tuple[np.ndarray, np.ndarray]:
        loc = self._index.locate(pair.source_sample_id)
        table = self._table(loc.shard)
        saida, registro = [], {"sample_id": pair.sample_id,
                               "source_sample_id": pair.source_sample_id,
                               "shard": loc.shard, "row": loc.row}
        for papel, coluna in (("aif", pair.aif_ref), ("bokeh", pair.bokeh_ref)):
            cell = table.column(coluna)[loc.row].as_py()
            data = self._raw_bytes(cell, sample_id=pair.sample_id, column=coluna)
            saida.append(self._decode(data, sample_id=pair.sample_id, column=coluna))
            registro[f"{papel}_sha256"] = hashlib.sha256(data).hexdigest()
            registro[f"{papel}_bytes"] = len(data)
            registro[f"{papel}_column"] = coluna
            if self._store_dir is not None:
                destino = self._store_dir / f"{pair.sample_id}_{papel}{self._ext(data)}"
                destino.write_bytes(data)          # bytes ORIGINAIS, sem recomprimir
                registro[f"{papel}_file"] = destino.name
        if self._ledger is not None:
            self._ledger.write(json.dumps(registro, ensure_ascii=False) + "\n")
            self._ledger.flush()
        return saida[0], saida[1]


def order_pairs_for_sequential_read(pairs: Iterable, index: MirrorIndex) -> list:
    """Ordena por `(shard, linha)`. Ver o cabeçalho do módulo."""
    return sorted(pairs, key=lambda p: index.sort_key(p.source_sample_id))


def sample_pairs_for_pilot(pairs: Sequence, *, limit: int, seed: int = 0) -> list:
    """Amostra ~`limit` pares **espalhados por cena**, com seed.

    O prefixo da ordem de disco não serve para piloto: os shards do espelho estão
    agrupados por cena, então os 200 primeiros pares saem de uma dúzia de cenas e o
    histograma de rejeição mede aquela dúzia, não o dataset. Como a rota C conta o
    `--limit` por amostra ACEITA, um piloto enviesado congelaria limiar em cima de um
    canto do dataset.

    Sorteia cenas inteiras, não pares soltos: uma cena com 21 aberturas contribui com
    as 21 ou com nenhuma, que é a mesma unidade do split.
    """
    if limit <= 0:
        raise ValueError("limit tem que ser positivo")
    por_cena: dict[str, list] = {}
    for pair in pairs:
        por_cena.setdefault(pair.scene_id, []).append(pair)

    cenas = sorted(por_cena)
    random.Random(seed).shuffle(cenas)
    escolhidos: list = []
    for cena in cenas:
        if len(escolhidos) >= limit:
            break
        escolhidos.extend(por_cena[cena])
    return escolhidos
