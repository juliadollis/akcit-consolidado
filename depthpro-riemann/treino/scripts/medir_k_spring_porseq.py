#!/usr/bin/env python3
"""Passo 3 do reteste, com a focal DE CADA SEQUENCIA.

Por que existe: o medir_k_spring.py recebe um unico --fx-orig e aplica a todas as
imagens. No Spring o fx varia 4.7x entre as 37 sequencias (1292.9 a 6060.6, 13
valores distintos). Como a curvatura depende do fx, usar uma unica mediana injeta
uma dispersao artificial em |K| -- medido numa esfera sintetica de K conhecido:
8.3x para cima na sequencia mais aberta e 25x para baixo na mais fechada.

Isso importa porque o passo 3 existe para ESCOLHER O TETO a partir dos percentis.
Um teto escolhido sobre uma cauda fabricada seria arbitrario.

Roda as duas versoes lado a lado para mostrar o tamanho do efeito no dado real.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace")
from riemann.dataset import HighQualityDepthDataset
from riemann.geometry import surface_curvatures


def fx_por_sequencia(raiz_spring):
    """Mesma logica do ler_fx() do prepare_spring: 1o numero de cada linha, mediana."""
    out = {}
    for p in sorted(Path(raiz_spring).glob("*/cam_data/intrinsics.txt")):
        vals = []
        for linha in p.read_text().strip().splitlines():
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", linha)]
            if nums:
                vals.append(nums[0])
        if vals:
            out[p.parts[-3]] = float(np.median(vals))
    return out


def percentis(K, rotulo):
    # numpy em vez de torch.quantile: o torch limita o tensor de entrada a ~16M
    # elementos e aqui sao ~29M pixels (111 imagens de 512x512).
    K = K[torch.isfinite(K)]
    arr = K.float().cpu().numpy()
    print(f"\n=== {rotulo} ===  (n = {arr.size:,} pixels)")
    print("percentis de |K| (1/m^2):")
    for q in (50, 90, 99, 99.5, 99.9):
        print(f"  p{q}: {float(np.percentile(arr, q)):14.4f}")
    print("fracao acima de cada teto:")
    for teto in (1, 5, 20, 50):
        print(f"  > {teto:3d}: {100*float((K > teto).float().mean()):6.2f}%")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default="/data/spring_prep/test")
    ap.add_argument("--spring-root", default="/data/spring_amostra/spring/train")
    ap.add_argument("--fx-mediana", type=float, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--w-orig", type=int, default=1920)
    ap.add_argument("--h-orig", type=int, default=1080)
    ap.add_argument("--smooth-sigma", type=float, default=0.5)
    a = ap.parse_args()

    fxs = fx_por_sequencia(a.spring_root)
    print(f"fx lido de {len(fxs)} sequencias: "
          f"min {min(fxs.values()):.1f}  mediana {np.median(list(fxs.values())):.1f}  "
          f"max {max(fxs.values()):.1f}")

    ds = HighQualityDepthDataset(a.raiz, size=(a.size, a.size))
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    print(f"amostras: {len(ds)}  (usando TODAS, nao so 10 lotes)")

    K_med, K_seq = [], []
    fx_m = a.fx_mediana * a.size / a.w_orig
    fy_m = a.fx_mediana * a.size / a.h_orig
    faltando = set()
    for i, b in enumerate(dl):
        chave = ds.keys[i]
        m = re.match(r"seq(\w+?)__", chave)
        seq = m.group(1) if m else None
        d = b["depth"].float()
        K_med.append(surface_curvatures(d, fx=fx_m, fy=fy_m,
                                        smooth_sigma=a.smooth_sigma, clamp_val=None)["K"].flatten())
        f_orig = fxs.get(seq)
        if f_orig is None:
            faltando.add(seq); continue
        K_seq.append(surface_curvatures(d, fx=f_orig*a.size/a.w_orig, fy=f_orig*a.size/a.h_orig,
                                        smooth_sigma=a.smooth_sigma, clamp_val=None)["K"].flatten())
    if faltando:
        print(f"[aviso] sem fx para: {sorted(faltando)}")

    A = percentis(torch.cat(K_med).abs(), "COM UMA UNICA MEDIANA DE fx (como o script atual)")
    B = percentis(torch.cat(K_seq).abs(), "COM O fx DE CADA SEQUENCIA (correto)")

    print("\n=== o que muda ===")
    for q in (50, 90, 99, 99.9):
        va = float(np.percentile(A, q)); vb = float(np.percentile(B, q))
        print(f"  p{q}: {va:14.4f} -> {vb:14.4f}   ({vb/max(va,1e-12):6.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
