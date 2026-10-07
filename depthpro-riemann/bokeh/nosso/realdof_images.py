"""Lê os pixels do espelho `akcit-pixel/RealDOF`.

`sources/realdof.py` responde *quais pares existem*; este módulo responde *onde estão os
bytes*. Mesma separação de `sources/mirror_images.py` e `sources/lfdof_images.py`.

## O que este módulo REUSA em vez de copiar

`MirrorIndex`, `order_pairs_for_sequential_read` e `sample_pairs_for_pilot` vêm de
`sources/mirror_images.py` **importados, não copiados** — os três são genéricos. Copiar
seria pior que verboso: este projeto chegou a quatro interpretações de K porque havia
quatro cópias do cálculo.

## O que ele NÃO reusa do `MirrorImageLoader`, e por quê

Duas coisas, e as duas são checagens que não existem lá:

1. **O `path` da célula, que nomeia o papel.** Cada célula traz
   `RealDOF_<numero>_<papel>_<alinhamento>.png`, com `<papel>` em `focus` / `blur` /
   `pre-deblur` — medido, bate em 100/100 células das 50 linhas. É a checagem cruzada
   por LINHA de que `image_focus` traz mesmo a AIF.

   O defeito que ela pega é o pior possível: colunas trocadas produzem um lote inteiro
   plausível com AIF e alvo invertidos. O sweep da Eq. 5 acharia um `K*`, o SSIM até
   seria alto, e nada mais denunciaria — e num experimento que compara profundidades
   pelo SSIM do bokeh, isso passaria como resultado.

   Note a ordem: no LFDOF o papel vem antes do `level` (`..._focus_level_3_aligned`);
   aqui vem antes do alinhamento (`RealDOF_10_focus_aligned`). São gramáticas
   diferentes, e por isso `expected_cell_path` é desta fonte e não importada de lá.

2. **Alpha.** Medido: 100/100 células são PNG modo `RGB`, sem canal alpha. O
   `convert("RGB")` do `MirrorImageLoader` seria seguro hoje. A verificação fica assim
   mesmo, no molde do LFDOF: se um dia o espelho for remontado com RGBA,
   `convert("RGB")` comporia sobre PRETO em silêncio — inventar pixel onde a origem
   declarou transparência, e esse pixel entraria no Depth Pro, no BiRefNet e no SSIM.

## Resolução

**Não há `expected_hw`.** São cinco resoluções nas 50 linhas (ver o cabeçalho de
`sources/realdof.py`), então fixar uma descartaria metade do conjunto. O que continua
sendo verificado é que a AIF e o alvo do MESMO par têm a mesma resolução — e isso já é
feito por `routes/route_c.process_pair`, que rejeita com `resolution_invalid`.

## Decodificação

PIL, não OpenCV: o `cv2` do container falha com `GLIBC_2.38 not found`. Saída BGR uint8
HxWx3, que é o que `routes/route_c.py` espera.

## Ledger

Mesmo contrato de auditoria dos outros dois carregadores: a rota C não GERA pixel, então
sem registrar nada o resultado seria número solto apontando para um espelho privado que
pode mudar. O ledger grava sha256, tamanho, shard, linha e o `path` original de cada
imagem lida.
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
from sources.realdof import (
    REALDOF_AIF_COLUMN, REALDOF_BOKEH_COLUMN, REALDOF_IMAGE_HW, REALDOF_UNUSED_COLUMN,
    reject_source as _reject,
)
# Importados, NÃO copiados — ver o cabeçalho.
from sources.mirror_images import (  # noqa: F401  (reexportados de propósito)
    INDEX_FILENAME, MirrorIndex, MirrorLocation, NAME_COLUMN,
    order_pairs_for_sequential_read, sample_pairs_for_pilot,
)

#: `image_focus` -> `focus`, `image_blur` -> `blur`. É o `<papel>` que aparece no `path`
#: de cada célula. `REALDOF_UNUSED_COLUMN` está aqui só para que a checagem saiba
#: reconhecê-lo e dizer o nome certo se alguém apontar uma coluna para ela.
COLUMN_ROLE = {
    REALDOF_AIF_COLUMN: "focus",
    REALDOF_BOKEH_COLUMN: "blur",
    REALDOF_UNUSED_COLUMN: "pre-deblur",
}


def expected_cell_path(source_sample_id: str, column: str) -> Optional[str]:
    """`file_name_base` + coluna -> o `path` que a célula deve trazer.

    `realdof_10_aligned` + `image_focus` -> `RealDOF_10_focus_aligned.png`

    Medido em 100 células (50 linhas, 2 colunas): 100/100 batem. Devolve `None` quando a
    coluna não é uma das três do schema, ou quando o nome não tem a forma esperada — aí
    não há papel a esperar, e inventar um seria pior que não conferir.
    """
    papel = COLUMN_ROLE.get(column)
    if papel is None:
        return None
    partes = source_sample_id.split("_", 2)
    if len(partes) != 3 or partes[0] != "realdof":
        return None
    _, numero, alinhamento = partes
    return f"RealDOF_{numero}_{papel}_{alinhamento}.png"


class RealDOFImageLoader:
    """`LoadPair` da rota C sobre o snapshot local do espelho do RealDOF.

    Mantém `cache_shards` tabelas parquet abertas (default 1). São 4 shards de 12 a 13
    linhas; com os pares reordenados por `order_pairs_for_sequential_read`, 1 basta.

    **Sem cache de AIF**, ao contrário do `LFDOFImageLoader`: lá a mesma all-in-focus
    serve 2 a 17 níveis da cena e o memo economiza 15 decodificações; aqui é um par por
    cena e o memo nunca acertaria. Um cache que nunca acerta é código que só pode
    quebrar.
    """

    def __init__(self, snapshot_dir: str | Path, index: MirrorIndex, *,
                 cache_shards: int = 1,
                 expected_hw: Optional[tuple[int, int]] = REALDOF_IMAGE_HW,
                 ledger_path: Optional[str | Path] = None,
                 store_dir: Optional[str | Path] = None):
        self._root = Path(snapshot_dir)
        self._index = index
        self._cache_shards = max(1, int(cache_shards))
        #: `None` por default, e é o certo aqui — ver "Resolução" no cabeçalho.
        self._expected_hw = expected_hw
        self._open: "OrderedDict[str, object]" = OrderedDict()

        self._ledger_path = Path(ledger_path) if ledger_path else None
        self._ledger = None
        if self._ledger_path is not None:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._ledger = self._ledger_path.open("a", encoding="utf-8")

        # Copia os bytes ORIGINAIS (sem recomprimir). Recomprimir falsificaria a
        # evidência: o sha256 do ledger deixaria de bater com o arquivo ao lado dele.
        self._store_dir = Path(store_dir) if store_dir else None
        if self._store_dir is not None:
            self._store_dir.mkdir(parents=True, exist_ok=True)

    # -- ciclo de vida ---------------------------------------------------------

    def close(self) -> None:
        if self._ledger is not None:
            self._ledger.close()
            self._ledger = None

    def __enter__(self) -> "RealDOFImageLoader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

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
            # propósito: é saída de modelo, não imagem de origem, e carregá-la gastaria
            # um terço da banda e da RAM para nada.
            table = pq.read_table(
                self._root / shard,
                columns=[NAME_COLUMN, REALDOF_AIF_COLUMN, REALDOF_BOKEH_COLUMN])
            self._open[shard] = table
            while len(self._open) > self._cache_shards:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(shard)
        return table

    # -- bytes -> array --------------------------------------------------------

    def _raw_bytes(self, cell, *, sample_id: str,
                   column: str) -> tuple[bytes, Optional[str]]:
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
        não a fonte primária do papel (que é o nome da coluna). `path` presente e
        CONTRADIZENDO o papel rejeita — aí a origem afirma duas coisas incompatíveis, e
        adivinhar qual vale é justamente o que não se faz aqui.
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
                    f"esperado {esperado!r}. Colunas trocadas produzem um lote inteiro "
                    "plausível com AIF e alvo invertidos, e o sweep da Eq. 5 não "
                    "denuncia isso — por isso rejeita em vez de confiar na coluna.")

    def _decode(self, data: bytes, *, sample_id: str, column: str) -> np.ndarray:
        """Bytes de PNG/JPEG -> BGR uint8 HxWx3, sem inventar pixel."""
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as img:
                modo = img.mode
                if modo in ("RGBA", "LA", "PA") or (
                        modo == "P" and "transparency" in img.info):
                    # Compor sobre preto seria inventar pixel. Confere a opacidade e só
                    # então descarta o canal — ver o cabeçalho do módulo.
                    rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
                    alpha_min = int(rgba[:, :, 3].min())
                    if alpha_min != 255:
                        _reject("source_image_alpha_not_opaque",
                                f"{sample_id}: coluna {column!r} é {modo} com alpha "
                                f"mínimo {alpha_min} (< 255). `convert('RGB')` comporia "
                                "sobre preto, que é inventar pixel onde a origem "
                                "declarou transparência. Medido: modo RGB, sem alpha, "
                                "em 100/100 células do espelho.")
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
                    f"{tuple(self._expected_hw)}")
        return np.ascontiguousarray(rgb[:, :, ::-1])     # RGB -> BGR, sem cv2

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
                    "alignment": getattr(pair, "alignment", None),
                    "shard": loc.shard, "row": loc.row}
        for papel, coluna in (("aif", pair.aif_ref), ("bokeh", pair.bokeh_ref)):
            cell = table.column(coluna)[loc.row].as_py()
            data, path = self._raw_bytes(cell, sample_id=pair.sample_id, column=coluna)
            self._check_role(sample_id=pair.sample_id,
                             source_sample_id=pair.source_sample_id,
                             column=coluna, path=path)
            array = self._decode(data, sample_id=pair.sample_id, column=coluna)
            saida.append(array)
            registro[f"{papel}_sha256"] = hashlib.sha256(data).hexdigest()
            registro[f"{papel}_bytes"] = len(data)
            registro[f"{papel}_column"] = coluna
            registro[f"{papel}_path"] = path
            registro[f"{papel}_hw"] = [int(array.shape[0]), int(array.shape[1])]
            if self._store_dir is not None:
                destino = self._store_dir / f"{pair.sample_id}_{papel}{self._ext(data)}"
                if not destino.exists():
                    destino.write_bytes(data)      # bytes ORIGINAIS, sem recomprimir
                registro[f"{papel}_file"] = destino.name

        if self._ledger is not None:
            self._ledger.write(json.dumps(registro, ensure_ascii=False) + "\n")
            self._ledger.flush()
        return saida[0], saida[1]
