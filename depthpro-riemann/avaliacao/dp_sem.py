#!/usr/bin/env python3
"""Confere o erro padrao da media por cena do boundary_fscore."""
import csv, statistics as st
from pathlib import Path
AV = Path("/host/avaliacoes")
vals = []
for rot in sorted(p for p in AV.iterdir() if p.is_dir()):
    f = rot / "spring_test" / "por_imagem.csv"
    if not f.is_file():
        continue
    porcena = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            porcena.setdefault(r["cena"], []).append(float(r["boundary_fscore"]))
    medias = [st.mean(v) for v in porcena.values()]
    n = len(medias)
    sem = st.stdev(medias) / (n ** 0.5)
    vals.append((rot.name, n, st.mean(medias), sem))
sems = [v[3] for v in vals]
print(f"modelos: {len(vals)}  n_cenas distintos: {sorted({v[1] for v in vals})}")
print(f"SEM por cena do boundary_fscore: min={min(sems):.4f} mediana={st.median(sems):.4f} max={max(sems):.4f}")
for nome, n, m, s in vals[:3] + vals[-3:]:
    print(f"  {nome:48s} n={n} media={m:.4f} sem={s:.4f}")
