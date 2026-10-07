#!/usr/bin/env python3
"""Segunda hipotese para a cauda de |K|: quantizacao da disparidade no fundo.

A primeira hipotese (bordas) foi REFUTADA: os pixels de degrau forte tem |K|
minusculo (p50 = 0.003) e a cauda vive justamente nas regioes MAIS PLANAS.

Hipotese agora: o Spring nao distribui profundidade, e sim disparidade. Como
Z = fx*B/d, um quantum de disparidade vira um degrau de profundidade que CRESCE
COM O QUADRADO DA DISTANCIA:

    dZ = Z^2 / (fx*B) * dd

Com fx=2586 e B=0.065, a 45 m um quantum de 0.01 px de disparidade ja vira ~12 cm
de degrau. Numa parede plana e distante isso e uma escada, e a segunda derivada de
uma escada e enorme. Se for isso, |K| deve crescer com a profundidade.
"""
import sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace")
from riemann.dataset import HighQualityDepthDataset
from riemann.geometry import surface_curvatures

SIZE, W_ORIG, H_ORIG, FX, B = 512, 1920, 1080, 2585.859, 0.065
ds = HighQualityDepthDataset("/data/spring_prep/test", size=(SIZE, SIZE))
dl = DataLoader(ds, batch_size=1, num_workers=0)
fx, fy = FX*SIZE/W_ORIG, FX*SIZE/H_ORIG

Ks, Ds = [], []
for b in dl:
    d = b["depth"].float()
    K = surface_curvatures(d, fx=fx, fy=fy, smooth_sigma=0.5, clamp_val=None)["K"]
    Ks.append(K.flatten()); Ds.append(d.flatten())
K = torch.cat(Ks).abs().numpy(); D = torch.cat(Ds).numpy()
ok = np.isfinite(K) & np.isfinite(D) & (D > 0); K, D = K[ok], D[ok]
print(f"pixels validos: {K.size:,}")

print("\n|K| por faixa de PROFUNDIDADE")
print(f"{'faixa (m)':>16s} {'% pixels':>10s} {'p50':>11s} {'p99':>13s} {'p99.9':>14s} {'>5':>8s}")
bordas = [0,5,10,20,40,80,np.inf]
for lo,hi in zip(bordas[:-1],bordas[1:]):
    m=(D>=lo)&(D<hi)
    if m.sum()<1000: continue
    k=K[m]
    rot=f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
    print(f"{rot:>16s} {100*m.mean():9.2f}% {np.percentile(k,50):11.3f} "
          f"{np.percentile(k,99):13.1f} {np.percentile(k,99.9):14.1f} {100*(k>5).mean():7.2f}%")

print("\nDegrau de profundidade por quantum de disparidade (dZ = Z^2/(fx*B) * dd):")
for Z in (5,10,20,40,80):
    print(f"  a {Z:3d} m, dd=0.01 px  ->  dZ = {Z**2/(FX*B)*0.01*100:8.2f} cm")

print("\nSe eu limitar a profundidade, o que sobra:")
for teto in (80, 40, 20, 10):
    m = D < teto; k = K[m]
    print(f"  mantendo D < {teto:3d} m ({100*m.mean():5.1f}% dos pixels): "
          f"p50={np.percentile(k,50):8.3f}  p99={np.percentile(k,99):11.2f}  "
          f"p99.9={np.percentile(k,99.9):12.2f}  >5: {100*(k>5).mean():5.2f}%")
