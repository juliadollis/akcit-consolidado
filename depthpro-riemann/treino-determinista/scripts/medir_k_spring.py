#!/usr/bin/env python3
"""
scripts/medir_k_spring.py
=========================
Passo 3 do reteste de curvatura: mede a distribuicao de |K| com a formulacao
metrica correta (riemann.geometry.surface_curvatures).

Diferenca em relacao ao snippet original: o Spring e redimensionado para um
QUADRADO (dataset._resize faz cv2.resize para (size,size), sem preservar aspect),
mas o original e 1920x1080. Logo fx e fy escalam por fatores DIFERENTES:

    fx_trab = fx_orig * size / W_orig      (largura)
    fy_trab = fy_orig * size / H_orig      (altura)

Assumindo pixels quadrados no original (fy_orig = fx_orig), passar apenas fx
(deixando fy=fx) erra a curvatura. Aqui os dois eixos sao tratados.

Uso:
    python scripts/medir_k_spring.py --raiz /data/spring_prep/test --fx-orig 1234.0
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.dataset import HighQualityDepthDataset  # noqa: E402
from riemann.geometry import surface_curvatures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default="/data/spring_prep/test")
    ap.add_argument("--fx-orig", type=float, required=True,
                    help="focal em pixels na resolucao ORIGINAL (mediana do prepare_spring)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--w-orig", type=int, default=1920)
    ap.add_argument("--h-orig", type=int, default=1080)
    ap.add_argument("--n-lotes", type=int, default=10)
    ap.add_argument("--smooth-sigma", type=float, default=0.5)
    args = ap.parse_args()

    fx = args.fx_orig * args.size / args.w_orig
    fy = args.fx_orig * args.size / args.h_orig  # assume pixel quadrado no original
    print(f"focal de trabalho: fx={fx:.1f} px  fy={fy:.1f} px  ({args.size}x{args.size})")

    ds = HighQualityDepthDataset(args.raiz, size=(args.size, args.size))
    dl = DataLoader(ds, batch_size=2, num_workers=0)

    Ks = []
    for i, b in enumerate(dl):
        if i >= args.n_lotes:
            break
        r = surface_curvatures(b["depth"].float(), fx=fx, fy=fy,
                               smooth_sigma=args.smooth_sigma, clamp_val=None)
        Ks.append(r["K"].flatten())

    K = torch.cat(Ks).abs()
    K = K[torch.isfinite(K)]

    print("percentis de |K| (1/m^2):")
    for q in [50, 90, 99, 99.5, 99.9]:
        print(f"  p{q}: {float(torch.quantile(K, q / 100)):12.4f}")
    print("fracao acima de cada teto:")
    for teto in [1, 5, 20, 50]:
        print(f"  > {teto:3d}: {100 * float((K > teto).float().mean()):5.2f}%")


if __name__ == "__main__":
    main()
