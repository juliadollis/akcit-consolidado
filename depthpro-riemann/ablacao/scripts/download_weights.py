#!/usr/bin/env python3
"""
scripts/download_weights.py
===========================
Baixa os pesos do DepthPro (depth_pro.pt) para o diretório de modelos.

Dentro do container, o diretório /models é montado como volume — então os pesos
baixados persistem no host e não precisam ser rebaixados a cada execução.

Uso (dentro do container):
    python scripts/download_weights.py --out-dir /models
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEPTH_PRO_URL = (
    "https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/"
    "checkpoints/depth_pro.pt"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/models")
    args = ap.parse_args()

    ckpt_dir = Path(args.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / "depth_pro.pt"

    if dest.exists():
        print(f"[download] já existe: {dest}")
        return

    print(f"[download] baixando depth_pro.pt para {dest} ...")
    r = subprocess.run(
        ["wget", "-q", "--show-progress", DEPTH_PRO_URL, "-O", str(dest)]
    )
    if r.returncode != 0:
        print("[download] FALHA. Verifique a rede/URL.", file=sys.stderr)
        sys.exit(1)
    print(f"[download] OK: {dest}")


if __name__ == "__main__":
    main()
