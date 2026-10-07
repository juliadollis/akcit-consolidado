#!/usr/bin/env python3
"""Por que a seq0020 e a seq0043 pontuam tao mal?

Hipotese: o Spring distribui DISPARIDADE, e o prepare_spring converte com
Z = fx*B/d usando o fx daquela sequencia. O fx varia 4,7x entre sequencias
(1292,9 a 6060,6). Uma sequencia de fx alto gera profundidades grandes, que o
--max-depth 100 recorta. O recorte e NAO-LINEAR, entao o alinhamento afim da
metrica nao o desfaz, e a mascara passa a esconder o fundo inteiro.
"""
import glob, os, re
import numpy as np

RAIZ = "/data/spring_split/test"
SPRING = "/data/spring/train/spring/train"

def fx_da_seq(seq):
    for nome in ("cam_data/intrinsics.txt", "intrinsics.txt"):
        p = os.path.join(SPRING, seq, nome)
        if os.path.exists(p):
            vals = [float(l.split()[0]) for l in open(p) if l.strip()]
            return float(np.median(vals))
    return float("nan")

seqs = sorted({os.path.basename(f).split("__")[0]
               for f in glob.glob(f"{RAIZ}/depth/*.npy")})
print(f"{'cena':<10s} {'fx':>9s} {'n':>4s} {'p5':>8s} {'p50':>8s} {'p95':>8s} "
      f"{'%mascarado':>11s} {'%>100m':>8s}")
linhas = []
for s in seqs:
    fs = sorted(glob.glob(f"{RAIZ}/depth/{s}__*.npy"))
    amostra = fs[::max(1, len(fs)//10)]
    d_all, mask_frac, acima = [], [], []
    for f in amostra:
        d = np.load(f).astype(np.float32)
        invalido = ~np.isfinite(d) | (d <= 0)
        mask_frac.append(invalido.mean())
        v = d[np.isfinite(d) & (d > 0)]
        if v.size:
            d_all.append(np.random.default_rng(0).choice(v, size=min(50000, v.size), replace=False))
        acima.append((v >= 99.0).mean() if v.size else np.nan)
    v = np.concatenate(d_all) if d_all else np.array([np.nan])
    linhas.append((s, fx_da_seq(s), len(fs), np.percentile(v,5), np.percentile(v,50),
                   np.percentile(v,95), 100*np.mean(mask_frac), 100*np.nanmean(acima)))
for l in sorted(linhas, key=lambda x: -x[6]):
    print(f"{l[0]:<10s} {l[1]:9.1f} {l[2]:4d} {l[3]:8.2f} {l[4]:8.2f} {l[5]:8.2f} "
          f"{l[6]:11.2f} {l[7]:8.2f}")
