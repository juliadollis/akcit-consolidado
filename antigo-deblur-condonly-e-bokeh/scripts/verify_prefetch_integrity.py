#!/usr/bin/env python3
"""Varre uma pasta do prefetch procurando PNGs corrompidos/truncados.

Por que existe: o job 29267 (fase 1 do BokehNet, 2026-08-06) morreu no step
3380 com `OSError: image file is truncated` num arquivo de `rota_a/aif/`. A
causa foi a corrida do prefetch — o processo do login node e o job SLURM 29258
escreveram a MESMA pasta em paralelo, e dois escritores no mesmo arquivo deixam
o conteúdo cortado. O `metadata.jsonl` com 136575 linhas para 68000 idx únicos
é a assinatura dessa corrida.

Este script dá a lista EXATA dos arquivos ruins, para decidir se vale re-baixar
só esses índices ou se o estrago é grande demais.

Duas passadas, da mais barata para a mais cara:

  1. `--quick` (padrão): abre cada arquivo, lê os 8 bytes da assinatura PNG e os
     12 bytes finais, e confere o chunk IEND (`\\x00\\x00\\x00\\x00IEND\\xaeB\\x60\\x82`).
     Um arquivo truncado no meio NÃO tem o IEND. Custo: 2 seeks por arquivo.
  2. `--full`: decodifica a imagem inteira com PIL (`.load()`), que também pega
     corrupção NO MEIO do arquivo com o IEND intacto (bytes intercalados pelos
     dois escritores). Bem mais caro, mas é o mesmo trabalho que o DataLoader
     faz, então é o teste definitivo.

Uso (sempre via SLURM sem GPU — ver scripts/verify_prefetch_integrity.slurm):
    python3 scripts/verify_prefetch_integrity.py --root /workspace/data-bokeh/rota_a
    python3 scripts/verify_prefetch_integrity.py --root ... --full --workers 8

Saída: um relatório no stdout e um JSON com a lista de arquivos ruins em
`<root>/integrity_report.json`. NÃO apaga nem modifica nada.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
SUBDIRS = ("aif", "bokeh", "depth")


def check_quick(path: str) -> str | None:
    """Retorna None se OK, ou uma string descrevendo o problema."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return "arquivo vazio (0 bytes)"
        if size < len(PNG_MAGIC) + len(PNG_IEND):
            return f"arquivo curto demais para ser um PNG ({size} bytes)"
        with open(path, "rb") as fh:
            if fh.read(len(PNG_MAGIC)) != PNG_MAGIC:
                return "assinatura PNG ausente ou corrompida"
            fh.seek(-len(PNG_IEND), os.SEEK_END)
            if fh.read(len(PNG_IEND)) != PNG_IEND:
                return f"chunk IEND ausente — arquivo TRUNCADO ({size} bytes)"
    except OSError as exc:
        return f"erro de I/O: {exc}"
    return None


def check_full(path: str) -> str | None:
    """Decodifica a imagem toda — o mesmo que o DataLoader faz."""
    quick = check_quick(path)
    if quick is not None:
        return quick
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.load()
    except Exception as exc:  # noqa: BLE001 - qualquer falha aqui é arquivo ruim
        return f"falha ao decodificar ({type(exc).__name__}: {exc})"
    return None


def iter_files(root: str) -> Iterator[str]:
    for sub in SUBDIRS:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            print(f"[verify] AVISO: subpasta ausente: {d}", flush=True)
            continue
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".png"):
                    yield entry.path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pasta do prefetch (ex.: /workspace/data-bokeh/rota_a)")
    ap.add_argument("--full", action="store_true", help="decodifica a imagem inteira (mais lento, definitivo)")
    ap.add_argument("--workers", type=int, default=8, help="threads de I/O")
    ap.add_argument("--report", default=None, help="caminho do JSON de saída (default: <root>/integrity_report.json)")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[verify] ERRO: pasta não existe: {root}", file=sys.stderr)
        return 2

    check = check_full if args.full else check_quick
    mode = "FULL (decodifica tudo)" if args.full else "QUICK (assinatura + IEND)"
    print(f"[verify] root={root}", flush=True)
    print(f"[verify] modo={mode} workers={args.workers}", flush=True)

    files = list(iter_files(root))
    total = len(files)
    print(f"[verify] {total} arquivos .png a conferir", flush=True)

    bad: list[dict[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, problem in zip(files, pool.map(check, files)):
            done += 1
            if problem is not None:
                rel = os.path.relpath(path, root)
                bad.append({"file": rel, "problem": problem})
                print(f"[verify] RUIM  {rel}  ->  {problem}", flush=True)
            if done % 20000 == 0:
                print(f"[verify] progresso: {done}/{total} ({len(bad)} ruins até aqui)", flush=True)

    # Quais índices do metadata.jsonl foram afetados (para o re-download seletivo).
    # O nome do arquivo é NNNNNNN_stem.png, então o índice é o prefixo.
    idx_ruins = sorted({os.path.basename(b["file"]).split("_", 1)[0] for b in bad})

    report = {
        "root": root,
        "mode": "full" if args.full else "quick",
        "total_files": total,
        "bad_count": len(bad),
        "bad_files": bad,
        "affected_indices": idx_ruins,
    }
    out = args.report or os.path.join(root, "integrity_report.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[verify] relatório salvo em {out}", flush=True)
    except OSError as exc:
        print(f"[verify] AVISO: não consegui salvar o relatório: {exc}", flush=True)

    print("=" * 60, flush=True)
    print(f"[verify] TOTAL:   {total} arquivos", flush=True)
    print(f"[verify] RUINS:   {len(bad)}", flush=True)
    print(f"[verify] ÍNDICES AFETADOS: {len(idx_ruins)} de 68000", flush=True)
    if bad:
        print(f"[verify] índices: {', '.join(idx_ruins[:50])}{' ...' if len(idx_ruins) > 50 else ''}", flush=True)
    print("=" * 60, flush=True)
    print("[verify] FIM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
