#!/usr/bin/env python3
"""Lista os `scene_id` de um release que aparecem nos benches de avaliação.

POR QUE ISTO EXISTE
-------------------
A rota C sai do split **test** da RealBokeh (`source_split: "test"` em todo
meta), e os benches `juliadollis/bokeh-bench-realbokeh-test` e `-v2` saem do
MESMO lugar. Medido em 2026-09-25: **220 de 220** cenas do bench estão na rota
C, com 162 `file_name_base` batendo exatamente.

Treinar nelas e depois avaliar nelas não produz um número otimista: produz um
número que mede memorização, e que sobe justamente quando o modelo está pior.

Tirar custa 814 de 15.423 amostras — 5,3% da rota C. A lista sai daqui, fica
versionada em disco, e o dataloader a consome por `excluir_cenas_de_avaliacao`.
Nada é inferido em tempo de treino.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def cenas_do_bench(padrao: str) -> set[str]:
    """`cena_id`/`file_name_base` dos shards -> `test_<n>`, a chave da rota C."""
    nomes: set[str] = set()
    for shard in sorted(glob.glob(padrao, recursive=True)):
        tabela = pq.read_table(shard)
        for coluna in ("cena_id", "file_name_base"):
            if coluna in tabela.column_names:
                nomes |= {str(v) for v in tabela.column(coluna).to_pylist()}
    cenas = set()
    for n in nomes:
        if re.fullmatch(r"\d+", n):
            cenas.add(f"test_{n}")
            continue
        m = re.search(r"test_f?_?(\d+)", n)
        if m:
            cenas.add(f"test_{m.group(1)}")
    return cenas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.jsonl do release")
    ap.add_argument("--bench", action="append", required=True,
                    help="glob dos parquets de um bench (repetível)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    linhas = [json.loads(l) for l in Path(args.manifest).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    por_cena = Counter(str(x["scene_id"]) for x in linhas)
    total = len(linhas)

    uniao: set[str] = set()
    for padrao in args.bench:
        cb = cenas_do_bench(padrao)
        comuns = cb & set(por_cena)
        n = sum(por_cena[c] for c in comuns)
        print(f"{padrao.split('/')[-4] if '/' in padrao else padrao}: "
              f"bench={len(cb)} em_comum={len(comuns)} amostras={n} "
              f"({100 * n / max(total, 1):.1f}%)", flush=True)
        uniao |= comuns

    n = sum(por_cena[c] for c in uniao)
    print(f"\nUNIÃO: {len(uniao)} cenas, {n}/{total} amostras = "
          f"{100 * n / max(total, 1):.1f}%")
    print(f"sobram {total - n} amostras de {len(por_cena) - len(uniao)} cenas")
    Path(args.out).write_text(json.dumps(sorted(uniao), indent=1), encoding="utf-8")
    print(f"escrito: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
