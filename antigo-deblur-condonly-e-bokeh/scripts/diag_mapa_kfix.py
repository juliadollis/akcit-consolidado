#!/usr/bin/env python3
"""Confere se os mapas de defocus do modo `kfix` saem SAUDAVEIS.

Roda ANTES do treino de 60K steps. Motivo: a Etapa A.2 mostrou que a escolha da
unidade/normalizador decide entre mapa util e mapa inutil —
  - MAX_COC herdado (100) com disparidade em 1/m  -> 89% dos pixels em 1.0
  - o mesmo com 1/mm                              -> map_max mediano 0.032
Um mapa saturado ensina "borre tudo"; um morto ensina "nao borre nada". Os dois
arruinam o treino, e o prejuizo so apareceria dias depois.

Criterio de aprovacao (mesmo da Etapa A.2):
  - fracao de pixels saturados (>=0.999) mediana  < 0.50
  - map_max mediano                              > 0.05
  - E o mapa tem de VARIAR entre amostras: se todas derem o mesmo max, o K
    voltou a ser constante e o conserto nao surtiu efeito.

Sai com codigo != 0 se reprovar, para a cadeia nao disparar o treino.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/workspace/genrefocus_deblurnet_paper")

from genfocus_train.config import load_config
from genfocus_train.data import DatasetRuntimeConfig, build_dataset

N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
cfg = load_config(sys.argv[1])
stage = cfg.data.bokeh

ds = build_dataset(
    stage="bokeh",
    stage_config=stage,
    runtime=DatasetRuntimeConfig(
        image_size=stage.image_size, train=False,
        defocus_source=stage.defocus_source, max_coc=cfg.model.max_coc,
        min_calibration_ssim=stage.min_calibration_ssim,
        kfix_repo=stage.kfix_repo,
    ),
)
print(f"[diag] dataset: {len(ds)} amostras | defocus_source={stage.defocus_source}", flush=True)

passo = max(1, len(ds) // N)
maxs, sats, means = [], [], []
for i in range(0, min(len(ds), passo * N), passo):
    m = ds[i]["defocus_map"][0].numpy()
    maxs.append(float(m.max())); means.append(float(m.mean()))
    sats.append(float((m >= 0.999).mean()))

maxs, sats, means = np.array(maxs), np.array(sats), np.array(means)
print(f"\n[diag] {len(maxs)} amostras inspecionadas")
print(f"[diag] map_max      mediana {np.median(maxs):.4f}   min {maxs.min():.4f}   max {maxs.max():.4f}")
print(f"[diag] map_mean     mediana {np.median(means):.4f}")
print(f"[diag] frac saturada mediana {np.median(sats):.4f}")
print(f"[diag] DESVIO do map_max entre amostras: {maxs.std():.4f}")

reprovas = []
if np.median(sats) >= 0.50: reprovas.append(f"saturado (frac={np.median(sats):.2f} >= 0.50)")
if np.median(maxs) <= 0.05: reprovas.append(f"morto (map_max={np.median(maxs):.3f} <= 0.05)")
if maxs.std() < 1e-3:       reprovas.append("map_max identico entre amostras — o K voltou a ser constante")

print("\n" + "=" * 62)
if reprovas:
    print("REPROVADO: " + "; ".join(reprovas))
    print("NAO dispare o treino — reveja a calibracao do MAX_COC.")
    raise SystemExit(1)
print("APROVADO: mapas usam a faixa [0,1] e VARIAM entre amostras.")
raise SystemExit(0)
