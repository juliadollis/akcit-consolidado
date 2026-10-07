#!/usr/bin/env python3
"""Quanto da PERDA vem de quantos pixels, em funcao do teto.

A gauss_loss_metrica e  mean(|K_pred_clamp - K_alvo_clamp|)  sobre a mascara.
Duas consequencias mecanicas do clamp ser aplicado nos DOIS lados antes da
subtracao:

  - teto BAIXO: onde pred e alvo saturam do mesmo lado, a diferenca e
    exatamente zero -> o pixel nao gera gradiente nenhum;
  - teto ALTO: como |K| varre 6 ordens de grandeza, um L1 vira media dominada
    por um punhado de pixels extremos.

Este script mede os dois lados, usando o alvo como proxy (nao temos predicao
ainda): a fracao de pixels saturados e a fracao da soma de |K| que vem do topo.
"""
import sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace")
from riemann.dataset import HighQualityDepthDataset
from riemann.geometry import surface_curvatures

SIZE, W, H, FX = 512, 1920, 1080, 2585.859
ds = HighQualityDepthDataset("/data/spring_prep/full", size=(SIZE, SIZE))
dl = DataLoader(ds, batch_size=4, num_workers=0)
fx, fy = FX*SIZE/W, FX*SIZE/H

Ks = []
for b in dl:
    K = surface_curvatures(b["depth"].float(), fx=fx, fy=fy, smooth_sigma=0.5, clamp_val=None)["K"]
    Ks.append(K.flatten())
K = torch.cat(Ks).abs().numpy()
K = K[np.isfinite(K)]
print(f"pixels: {K.size:,}")

print("\nCONCENTRACAO: quanto da soma de |K| vem do topo da distribuicao")
tot = K.sum()
ordenado = np.sort(K)[::-1]
for frac in (0.0001, 0.001, 0.01, 0.1):
    n = max(1, int(frac*K.size))
    print(f"  top {100*frac:7.3f}% dos pixels ({n:>9,}) carregam "
          f"{100*ordenado[:n].sum()/tot:6.2f}% da soma de |K|")

print("\nO DILEMA DO TETO")
print(f"{'teto':>10s} {'% saturado':>12s} {'% da soma que sobra':>21s} {'top 0.1% carrega':>18s}")
for teto in (1, 5, 20, 50, 100, 500, 1000, 5000, 20000):
    Kc = np.minimum(K, teto)
    sat = 100*(K > teto).mean()
    o = np.sort(Kc)[::-1]; n = max(1, int(0.001*Kc.size))
    print(f"{teto:10d} {sat:11.2f}% {100*Kc.sum()/tot:20.2f}% {100*o[:n].sum()/Kc.sum():17.2f}%")

print("\nAmplitude: p50 = %.2f   p99.9 = %.0f   razao = %.0fx (%.1f ordens de grandeza)"
      % (np.percentile(K,50), np.percentile(K,99.9),
         np.percentile(K,99.9)/max(np.percentile(K,50),1e-9),
         np.log10(np.percentile(K,99.9)/max(np.percentile(K,50),1e-9))))

print("\nE se a perda fosse em log(1+|K|) em vez de |K| direto:")
L = np.log1p(K); o = np.sort(L)[::-1]
for frac in (0.001, 0.01):
    n = max(1, int(frac*L.size))
    print(f"  top {100*frac:6.2f}% carregam {100*o[:n].sum()/L.sum():6.2f}% da soma")
print("  p50 = %.3f   p99.9 = %.3f   razao = %.1fx" % (np.percentile(L,50), np.percentile(L,99.9),
      np.percentile(L,99.9)/max(np.percentile(L,50),1e-9)))
