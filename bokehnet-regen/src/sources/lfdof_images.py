"""Lê os pixels do espelho `akcit-pixel/LFDOF`.

`sources/lfdof.py` responde *quais pares existem*; este módulo responde *onde estão os
bytes*. A separação é a mesma de `sources/mirror_images.py` e pelo mesmo motivo:
enumerar 11.972 pares é barato e roda em qualquer máquina, decodificar 24.000 PNGs de
688×1008 não é.

## O que este módulo REUSA em vez de copiar

`MirrorIndex`, `order_pairs_for_sequential_read` e `sample_pairs_for_pilot` vêm de
`sources/mirror_images.py` **importados, não copiados**. Os três são genéricos: o índice
mapeia `file_name_base -> (shard, linha)` lendo só a coluna de nome de cada
`*.parquet` do snapshot, a ordenação usa `source_sample_id`, e a amostragem de piloto
usa `scene_id`. Nada neles é da RealBokeh.

Copiar seria pior do que verboso: este projeto chegou a quatro interpretações de K
porque havia quatro cópias do cálculo (`CLAUDE.md`, "O contrato"). Um segundo índice
divergiria do primeiro exatamente quando alguém consertasse um só.

O índice cacheia em `_mirror_index.json` **dentro do snapshot do LFDOF**, então não há
colisão com o cache da RealBokeh: são diretórios diferentes.

Medido no espelho do LFDOF, o que faz a ordenação valer a pena:

```
shards                                          : 83 (78 train + 5 test)
row groups por shard                            : 5 em 83/83
cenas espalhadas por mais de um shard            : 75 de 840
blocos contíguos de cena na ordem (shard, linha) : 840 para 840 cenas
```

A última linha é a que importa: na ordem de disco, **cada cena é um bloco contíguo**.
É o que permite o cache de AIF abaixo funcionar com uma entrada só.

## O que este módulo NÃO reusa, e por quê

`MirrorImageLoader` quase serve. Três coisas do LFDOF ele não faz, e cada uma delas é
um caminho por onde entraria dado inventado ou dado errado em silêncio:

1. **Alpha.** As duas colunas do LFDOF são PNG **RGBA** (medido: 90/90 linhas), com
   alpha ≡ 255. `Image.convert("RGB")` sobre RGBA compõe sobre **preto**. Com alpha
   opaco isso é descartar um plano constante, sem perda; com alpha < 255 é inventar
   pixel onde a origem declarou transparência — e esse pixel entraria no Depth Pro, no
   BiRefNet e no SSIM da Eq. 5. Aqui a opacidade é **verificada** e o contrário
   **rejeita**, em vez de compor.

2. **O `path` da célula, que nomeia o papel.** Cada célula do parquet do LFDOF traz
   `LFDOF_<split>_data_<cena>_<papel>_level_<N>_<alinhamento>.png`, com `<papel>` em
   `focus` / `blur` / `pre-deblur` (medido: bate em 180/180 células conferidas, em 3
   row groups de 3 shards). Isso dá uma checagem cruzada por LINHA de que a coluna
   `image_focus` traz mesmo a AIF — a única que este módulo tem, já que não há
   `target_avs` contra o qual conferir como na RealBokeh.

   O defeito que ela pega é o pior possível: colunas trocadas produzem um dataset
   inteiro plausível com AIF e alvo invertidos. O sweep da Eq. 5 encontraria um `K*`,
   o SSIM até seria alto, e nada mais no pipeline denunciaria.

3. **A AIF é a MESMA em todos os níveis da cena.** Medido byte a byte: o `image_focus`
   dos 15 níveis da cena 1275 tem um sha256 só (`d358a3847630…`), e em 9 cenas / 90
   linhas conferidas nenhuma cena tem mais de um sha de `image_focus`. É o esperado —
   há uma all-in-focus por light field —, e significa que decodificar a AIF 15 vezes
   por cena é 15× trabalho jogado fora.

   O cache é um **memo por sha256, não por cena**: os bytes são sempre lidos da tabela
   (que já está em RAM) e sempre hasheados; o hash é que decide se o array decodificado
   pode ser reusado. Assim o cache não pode servir pixel errado — se a AIF de um nível
   for diferente da do anterior, o sha difere e ele decodifica. Um cache por `scene_id`
   **assumiria** a igualdade em vez de conferi-la, e a economia é a mesma.

## Decodificação

PIL, não OpenCV: o `cv2` do container falha com `GLIBC_2.38 not found`, e o renderer já
convive com isso (ver `renderer/bokehme.py`). Saída BGR uint8 HxWx3, que é o que
`routes/route_c.py` espera.

## Ledger

Mesmo contrato de auditoria do `MirrorImageLoader`: a rota C não GERA pixel, então sem
registrar nada o release seria rótulo solto apontando para um espelho privado que pode
mudar. O ledger grava sha256, tamanho, shard, linha e o `path` original de cada imagem
lida — e, no LFDOF, também se a AIF veio do cache, para que a contagem de decodificações
seja auditável.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from control.contract import SampleRejected
from sources.lfdof import (
    LFDOF_AIF_COLUMN, LFDOF_BOKEH_COLUMN, LFDOF_IMAGE_HW, LFDOF_UNUSED_COLUMN,
    reject_source as _reject,
)
# Importados, NÃO copiados — ver o cabeçalho.
from sources.mirror_images import (  # noqa: F401  (reexportados de propósito)
    INDEX_FILENAME, MirrorIndex, MirrorLocation, NAME_COLUMN,
    order_pairs_for_sequential_read, sample_pairs_for_pilot,
)

#: `image_focus` -> `focus`, `image_blur` -> `blur`. É o `<papel>` que aparece no `path`
#: de cada célula do parquet. `LFDOF_UNUSED_COLUMN` está aqui só para que a checagem de
#: papel saiba reconhecê-lo e dizer o nome certo se alguém apontar uma coluna para ela.
COLUMN_ROLE = {
    LFDOF_AIF_COLUMN: "focus",
    LFDOF_BOKEH_COLUMN: "blur",
    LFDOF_UNUSED_COLUMN: "pre-deblur",
}


def expected_cell_path(source_sample_id: str, column: str) -> Optional[str]:
    """`file_name_base` + coluna -> o `path` que a célula deve trazer.

    `lfdof_train_data_1275_level_3_aligned` + `image_focus`
        -> `LFDOF_train_data_1275_focus_level_3_aligned.png`

    Medido em 180 células (3 row groups, 3 shards, 2 colunas): 180/180 batem. Devolve
    `None` quando a coluna não é uma das três do schema — aí não há papel a esperar, e
    inventar um seria pior que não conferir.
    """
    role = COLUMN_ROLE.get(column)
    if role is None:
        return None
    if not source_sample_id.startswith("lfdof_") or "_level_" not in source_sample_id:
        return None
    cabeca, cauda = source_sample_id.rsplit("_level_", 1)
    return f"LFDOF{cabeca[len('lfdof'):]}_{role}_level_{cauda}.png"


class LFDOFImageLoader:
    """`LoadPair` da rota C sobre o snapshot local do espelho do LFDOF.

    Mantém `cache_shards` tabelas parquet abertas (default 1). Com os pares reordenados
    por `order_pairs_for_sequential_read`, 1 basta e a memória fica constante — as cenas
    são blocos contíguos na ordem de disco (medido: 840 blocos para 840 cenas).

    `expected_hw` default `LFDOF_IMAGE_HW = (688, 1008)`, medido. Resolução diferente
    **rejeita**; não redimensiona. K vive em pixel, e redimensionar em silêncio mudaria
    o significado do rótulo sem mudar nada no JSON (`reference/CONTRATO.md`).
    """

    def __init__(self, snapshot_dir: str | Path, index: MirrorIndex, *,
                 cache_shards: int = 1,
                 expected_hw: Optional[tuple[int, int]] = LFDOF_IMAGE_HW,
                 ledger_path: Optional[str | Path] = None,
                 store_dir: Optional[str | Path] = None,
                 cache_aif: bool = True):
        self._root = Path(snapshot_dir)
        self._index = index
        self._cache_shards = max(1, int(cache_shards))
        self._expected_hw = expected_hw
        self._open: "OrderedDict[str, object]" = OrderedDict()

        self._ledger_path = Path(ledger_path) if ledger_path else None
        self._ledger = None
        if self._ledger_path is not None:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._ledger = self._ledger_path.open("a", encoding="utf-8")

        # Copia os bytes ORIGINAIS (sem recomprimir) para dentro do release, tornando-o
        # autocontido. Recomprimir seria falsificar a evidência: o sha256 do ledger
        # deixaria de bater com o arquivo ao lado dele.
        #
        # ATENÇÃO à licença: o LFDOF não tem licença explícita na página upstream `[M]`
        # (`reference/ACHADOS.md`). Guardar os pixels dentro do release é o que exige
        # resolver isso; o rótulo sozinho não exige
        # (`reference/ROTA_C_AUDITORIA.md:553-556`).
        self._store_dir = Path(store_dir) if store_dir else None
        if self._store_dir is not None:
            self._store_dir.mkdir(parents=True, exist_ok=True)

        #: memo de decodificação da AIF, chaveado pelo sha256 dos BYTES — ver o
        #: cabeçalho. Uma entrada: a cena corrente.
        self._cache_aif = bool(cache_aif)
        self._aif_memo: Optional[tuple[str, np.ndarray]] = None
        #: contadores, para o relatório do run poder afirmar quanto o cache economizou
        #: em vez de "acho que economizou".
        self.aif_decodes = 0
        self.aif_cache_hits = 0

    # -- ciclo de vida ---------------------------------------------------------

    def close(self) -> None:
        if self._ledger is not None:
            self._ledger.close()
            self._ledger = None

    def __enter__(self) -> "LFDOFImageLoader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def cache_summary(self) -> str:
        total = self.aif_decodes + self.aif_cache_hits
        if total == 0:
            return "[lfdof] nenhuma AIF lida."
        return (f"[lfdof] AIF: {self.aif_decodes} decodificações, "
                f"{self.aif_cache_hits} reusos do memo por sha256 "
                f"({100 * self.aif_cache_hits / total:.1f}% de {total} leituras)")

    # -- parquet ---------------------------------------------------------------

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
            # Só as colunas que a rota C usa. `image_pre_deblur` fica de FORA de
            # propósito: é saída de modelo (DRB-Net, variante §3.4), não imagem de
            # origem, e carregá-la gastaria um terço da banda e da RAM para nada. Ver
            # o cabeçalho de `sources/lfdof.py`.
            table = pq.read_table(
                self._root / shard,
                columns=[NAME_COLUMN, LFDOF_AIF_COLUMN, LFDOF_BOKEH_COLUMN])
            self._open[shard] = table
            while len(self._open) > self._cache_shards:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(shard)
        return table

    # -- bytes -> array --------------------------------------------------------

    def _raw_bytes(self, cell, *, sample_id: str, column: str) -> tuple[bytes, Optional[str]]:
        """Bytes da célula e o `path` que ela declara. Célula sem bytes nem path rejeita."""
        if not isinstance(cell, dict):
            _reject("source_image_unreadable",
                    f"{sample_id}: célula da coluna {column!r} é "
                    f"{type(cell).__name__}, esperado {{bytes, path}}")
        path = cell.get("path")
        data = cell.get("bytes")
        if data is None:
            if not path:
                _reject("source_image_unreadable",
                        f"{sample_id}: coluna {column!r} sem bytes nem path")
            data = (self._root / path).read_bytes()
        if not data:
            _reject("source_image_unreadable",
                    f"{sample_id}: coluna {column!r} tem 0 bytes")
        return data, (str(path) if path else None)

    def _check_role(self, *, sample_id: str, source_sample_id: str, column: str,
                    path: Optional[str]) -> None:
        """A checagem cruzada por linha: o `path` da célula nomeia o papel da coluna.

        Célula sem `path` **não** rejeita: a checagem é um bônus que a origem oferece,
        não a fonte primária do papel (que é o nome da coluna). Rejeitar por ausência
        de `path` inviabilizaria qualquer mock e qualquer parquet remontado sem esse
        campo. `path` presente e CONTRADIZENDO o papel rejeita, porque aí a origem está
        afirmando duas coisas incompatíveis e adivinhar qual vale é justamente o que
        não se faz aqui.
        """
        if path is None:
            return
        esperado = expected_cell_path(source_sample_id, column)
        if esperado is None:
            return
        if Path(path).name != esperado:
            _reject("source_image_role_mismatch",
                    f"{sample_id}: a coluna {column!r} (papel "
                    f"{COLUMN_ROLE.get(column)!r}) traz path {Path(path).name!r}, "
                    f"esperado {esperado!r}. Colunas trocadas produzem um dataset "
                    "inteiro plausível com AIF e alvo invertidos, e o sweep da Eq. 5 "
                    "não denuncia isso — por isso rejeita em vez de confiar na coluna.")

    def _decode(self, data: bytes, *, sample_id: str, column: str) -> np.ndarray:
        """Bytes de PNG/JPEG -> BGR uint8 HxWx3, sem inventar pixel."""
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as img:
                modo = img.mode
                if modo in ("RGBA", "LA", "PA") or (
                        modo == "P" and "transparency" in img.info):
                    # Compor sobre preto seria inventar pixel. Confere a opacidade e
                    # só então descarta o canal — ver o cabeçalho do módulo.
                    rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
                    alpha_min = int(rgba[:, :, 3].min())
                    if alpha_min != 255:
                        _reject("source_image_alpha_not_opaque",
                                f"{sample_id}: coluna {column!r} é {modo} com alpha "
                                f"mínimo {alpha_min} (< 255). `convert('RGB')` "
                                "comporia sobre preto, que é inventar pixel onde a "
                                "origem declarou transparência — e esse pixel entraria "
                                "no Depth Pro, no BiRefNet e no SSIM da Eq. 5. Medido: "
                                "alpha ≡ 255 em 90/90 linhas do espelho.")
                    rgb = rgba[:, :, :3]
                else:
                    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        except SampleRejected:
            raise
        except Exception as exc:                        # bytes corrompidos no shard
            _reject("source_image_unreadable",
                    f"{sample_id}: coluna {column!r} não decodifica ({exc})")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            _reject("source_image_unreadable",
                    f"{sample_id}: coluna {column!r} decodificou com shape {rgb.shape}")
        if self._expected_hw is not None and rgb.shape[:2] != tuple(self._expected_hw):
            _reject("source_image_unreadable",
                    f"{sample_id}: coluna {column!r} é {rgb.shape[:2]}, esperado "
                    f"{tuple(self._expected_hw)}. Resolução heterogênea muda o K em "
                    "pixel — ver a regra de resolução em reference/CONTRATO.md")
        return np.ascontiguousarray(rgb[:, :, ::-1])     # RGB -> BGR, sem cv2

    def _aif_array(self, data: bytes, sha: str, *,
                   sample_id: str, column: str) -> np.ndarray:
        """Decodifica a AIF, ou reusa o memo se os BYTES forem os mesmos."""
        if self._cache_aif and self._aif_memo is not None and self._aif_memo[0] == sha:
            self.aif_cache_hits += 1
            return self._aif_memo[1]
        array = self._decode(data, sample_id=sample_id, column=column)
        self.aif_decodes += 1
        if self._cache_aif:
            self._aif_memo = (sha, array)
        return array

    # -- LoadPair --------------------------------------------------------------

    def __call__(self, pair) -> tuple[np.ndarray, np.ndarray]:
        """`PairSource` -> (AIF BGR, bokeh BGR). Levanta `SampleRejected` com slug."""
        loc = self._index.locate(pair.source_sample_id)
        table = self._table(loc.shard)

        # O índice diz (shard, linha); a linha diz qual `file_name_base` ela é. Conferir
        # é uma comparação de string e fecha o buraco em que um índice cacheado de um
        # snapshot antigo aponta para a linha errada de um shard remontado — aí TODOS os
        # pares saem trocados, e o histograma fica limpo.
        na_linha = table.column(NAME_COLUMN)[loc.row].as_py()
        if na_linha != pair.source_sample_id:
            _reject("source_image_unreadable",
                    f"{pair.sample_id}: o índice aponta {loc.shard} linha {loc.row}, "
                    f"que contém {na_linha!r} e não {pair.source_sample_id!r}. Índice "
                    "cacheado de outro snapshot — reconstrua com "
                    "MirrorIndex.build(..., force=True).")

        saida: list[np.ndarray] = []
        registro = {"sample_id": pair.sample_id,
                    "source_sample_id": pair.source_sample_id,
                    "scene_id": pair.scene_id,
                    "shard": loc.shard, "row": loc.row}
        for papel, coluna in (("aif", pair.aif_ref), ("bokeh", pair.bokeh_ref)):
            cell = table.column(coluna)[loc.row].as_py()
            data, path = self._raw_bytes(cell, sample_id=pair.sample_id, column=coluna)
            self._check_role(sample_id=pair.sample_id,
                             source_sample_id=pair.source_sample_id,
                             column=coluna, path=path)
            sha = hashlib.sha256(data).hexdigest()
            if papel == "aif":
                antes = self.aif_cache_hits
                array = self._aif_array(data, sha, sample_id=pair.sample_id,
                                        column=coluna)
                registro["aif_from_cache"] = self.aif_cache_hits > antes
            else:
                array = self._decode(data, sample_id=pair.sample_id, column=coluna)
            saida.append(array)
            registro[f"{papel}_sha256"] = sha
            registro[f"{papel}_bytes"] = len(data)
            registro[f"{papel}_column"] = coluna
            registro[f"{papel}_path"] = path
            if self._store_dir is not None:
                registro[f"{papel}_file"] = self._store(
                    data, papel=papel, pair=pair)

        if self._ledger is not None:
            self._ledger.write(json.dumps(registro, ensure_ascii=False) + "\n")
            self._ledger.flush()
        return saida[0], saida[1]

    def _store(self, data: bytes, *, papel: str, pair) -> str:
        """Grava os bytes ORIGINAIS no release. A AIF vai UMA vez por cena.

        A AIF é byte a byte idêntica nos N níveis da cena (medido), então gravá-la por
        `sample_id` escreveria a mesma imagem 15 vezes em 682 das 840 cenas: ~13 GB
        contra ~1 GB. Ela vai por `scene_id`, e o ledger grava o nome do arquivo em cada
        linha, então o join amostra -> arquivo continua explícito.

        Não sobrescreve arquivo já gravado com o mesmo nome: se ele existe, os bytes são
        os mesmos (o sha está no ledger de cada linha e é conferível). Reescrever seria
        I/O jogado fora; sobrescrever com bytes diferentes seria o defeito, e é por isso
        que o sha vai para o ledger em TODA linha, não só na primeira.
        """
        nome = (f"{pair.scene_id}_aif{self._ext(data)}" if papel == "aif"
                else f"{pair.sample_id}_bokeh{self._ext(data)}")
        destino = self._store_dir / nome
        if not destino.exists():
            destino.write_bytes(data)          # bytes ORIGINAIS, sem recomprimir
        return nome
