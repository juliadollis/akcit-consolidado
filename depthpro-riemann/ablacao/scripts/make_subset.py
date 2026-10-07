#!/usr/bin/env python3
"""
scripts/make_subset.py
======================
Cria subconjuntos ("batches") REPRODUTÍVEIS do dataset já preparado, para as três fases:
  - smoke  : validação rápida do pipeline (poucas dezenas de imagens)
  - ablation: escolha da melhor loss (alguns milhares)
  - full   : treino final (dezenas de milhares, ou tudo)

Funciona SOBRE um dataset já convertido para o layout rgb/ + depth/ (ou seja, rode o
prepare_hypersim.py primeiro, OU aponte para qualquer dataset nesse layout).

O subconjunto é criado por SYMLINK (padrão) — não duplica os arquivos, economiza disco.
Use --copy para copiar de fato (ex.: se for mover para outra máquina).

A seleção é DETERMINÍSTICA (ordena por nome e amostra com passo fixo), então o mesmo
--n gera sempre o mesmo subconjunto — reprodutível entre execuções e máquinas.

Exemplos:
    # 60 imagens para smoke test
    python scripts/make_subset.py --src /data/hypersim_full/train \\
        --dst /data/hypersim/smoke/train --n 60

    # 3000 imagens para ablação
    python scripts/make_subset.py --src /data/hypersim_full/train \\
        --dst /data/hypersim/ablation/train --n 3000
"""

import argparse
import os
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEPTH_EXTS = {".png", ".npy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dataset origem no layout rgb/ + depth/")
    ap.add_argument("--dst", required=True,
                    help="destino do subconjunto (será criado)")
    ap.add_argument("--n", type=int, required=True,
                    help="nº de pares a incluir (-1 = todos)")
    ap.add_argument("--copy", action="store_true",
                    help="copiar arquivos em vez de criar symlinks")
    ap.add_argument("--seed-stride", action="store_true",
                    help="amostrar com passo uniforme (default: primeiros N)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    rgb_src, depth_src = src / "rgb", src / "depth"
    if not rgb_src.exists() or not depth_src.exists():
        raise FileNotFoundError(f"Origem inválida: esperado {rgb_src} e {depth_src}")

    # Pares casados por nome (stem)
    rgb = {p.stem: p for p in rgb_src.iterdir() if p.suffix.lower() in IMG_EXTS}
    depth = {p.stem: p for p in depth_src.iterdir() if p.suffix.lower() in DEPTH_EXTS}
    keys = sorted(set(rgb) & set(depth))
    total = len(keys)
    if total == 0:
        raise RuntimeError("Nenhum par (rgb, depth) casado na origem.")

    n = total if args.n < 0 else min(args.n, total)

    # Seleção determinística
    if args.seed_stride and n < total:
        stride = total / n
        idx = [int(i * stride) for i in range(n)]
        chosen = [keys[i] for i in idx]
    else:
        chosen = keys[:n]

    rgb_dst, depth_dst = dst / "rgb", dst / "depth"
    rgb_dst.mkdir(parents=True, exist_ok=True)
    depth_dst.mkdir(parents=True, exist_ok=True)

    def place(src_path: Path, dst_path: Path):
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        if args.copy:
            shutil.copy2(src_path, dst_path)
        else:
            os.symlink(src_path.resolve(), dst_path)

    for k in chosen:
        place(rgb[k], rgb_dst / rgb[k].name)
        place(depth[k], depth_dst / depth[k].name)

    mode = "copiados" if args.copy else "linkados (symlink)"
    print(f"Subconjunto criado: {len(chosen)}/{total} pares {mode}")
    print(f"  origem : {src}")
    print(f"  destino: {dst}")
    print(f"  rgb/   : {rgb_dst}")
    print(f"  depth/ : {depth_dst}")


if __name__ == "__main__":
    main()
