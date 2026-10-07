#!/usr/bin/env python3
"""Diagnostico: de onde vem a cauda de |K| no Spring.

O passo 3 mostrou mediana fisicamente sensata (~4.8) mas p99.9 ~ 3.6 milhoes.
K = 1/R^2, entao 3.6e6 equivale a um raio de curvatura de 0.53 MILIMETRO. Isso
nao e geometria de cena: e o que a diferenca finita produz numa DESCONTINUIDADE
de profundidade (borda de objeto), onde a superficie nem e diferenciavel.

Este script testa essa hipotese: separa os pixels por quao forte e o degrau de
profundidade na vizinhanca e mostra a distribuicao de |K| em cada grupo.
"""
import sys, re
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace")
from riemann.dataset import HighQualityDepthDataset
from riemann.geometry import surface_curvatures

RAIZ = "/data/spring_prep/test"
SIZE, W_ORIG, H_ORIG, FX = 512, 1920, 1080, 2585.859

ds = HighQualityDepthDataset(RAIZ, size=(SIZE, SIZE))
dl = DataLoader(ds, batch_size=1, num_workers=0)
fx, fy = FX*SIZE/W_ORIG, FX*SIZE/H_ORIG

Ks, degraus = [], []
for b in dl:
    d = b["depth"].float()
    K = surface_curvatures(d, fx=fx, fy=fy, smooth_sigma=0.5, clamp_val=None)["K"]
    # degrau RELATIVO de profundidade: |grad D| / D. Adimensional, comparavel
    # entre cenas perto e longe.
    gy, gx = torch.gradient(d[0,0], dim=(0,1))
    rel = (torch.sqrt(gx**2 + gy**2) / d[0,0].clamp(min=1e-6))
    Ks.append(K.flatten()); degraus.append(rel.flatten())

K = torch.cat(Ks).abs().numpy(); S = torch.cat(degraus).numpy()
ok = np.isfinite(K) & np.isfinite(S); K, S = K[ok], S[ok]
print(f"pixels: {K.size:,}")

print("\n|K| por faixa de degrau relativo de profundidade")
print(f"{'faixa de |grad D|/D':>26s} {'% dos pixels':>13s} {'p50':>10s} {'p99':>13s} {'p99.9':>14s}")
cortes = [(0,0.001),(0.001,0.01),(0.01,0.05),(0.05,0.2),(0.2,np.inf)]
for lo,hi in cortes:
    m = (S>=lo)&(S<hi)
    if m.sum()==0: continue
    k=K[m]
    rot = f"{lo:g} a {hi:g}" if np.isfinite(hi) else f"acima de {lo:g}"
    print(f"{rot:>26s} {100*m.mean():12.2f}% {np.percentile(k,50):10.3f} "
          f"{np.percentile(k,99):13.1f} {np.percentile(k,99.9):14.1f}")

print("\nSe eu remover os pixels de borda, o que sobra:")
for corte in (0.2, 0.05, 0.01):
    m = S < corte
    k = K[m]
    print(f"  mantendo |grad D|/D < {corte:<5g}  ({100*m.mean():5.1f}% dos pixels): "
          f"p50={np.percentile(k,50):8.3f}  p99={np.percentile(k,99):10.2f}  "
          f"p99.9={np.percentile(k,99.9):11.2f}  >5: {100*(k>5).mean():5.2f}%")

print("\nRaio de curvatura equivalente (R = 1/sqrt(K)) nos percentis altos:")
for q in (99, 99.9):
    v = np.percentile(K, q)
    print(f"  p{q}: K={v:12.1f}  ->  R = {1/np.sqrt(v)*1000:8.3f} mm")
