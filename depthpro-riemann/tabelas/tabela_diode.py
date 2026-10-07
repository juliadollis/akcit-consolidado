import glob
import json
import statistics as st
from collections import defaultdict

ROT = {"B0_berhu": "B0 berHu (controle)", "Bgrad": "Bgrad", "Bgeod": "Bgeod",
       "Bmetric": "Bmetric", "B1_gauss_metrica_teto5": "B1 curv, teto 5",
       "B1_gauss_metrica_teto1000": "B1 curv, teto 1000",
       "B3_gauss_metrica_teto5": "B3 normal+curv, teto 5",
       "B3_gauss_metrica_teto50": "B3 normal+curv, teto 50",
       "B3_gauss_metrica_teto1000": "B3 normal+curv, teto 1000",
       "B0_berhu_size768": "B0 berHu a 768 px"}

por_braco = defaultdict(dict)
for q in glob.glob("/host/avaliacoes/*/diode_val/test_metrics.json"):
    rot = q.split("/")[-3]
    partes = rot.split("__")
    if len(partes) != 3:
        continue
    _, braco, seed = partes
    s = int(seed.replace("seed_", ""))
    if s < 6:
        por_braco[braco][s] = json.load(open(q))

print(f"braços com pelo menos uma seed: {len(por_braco)}\n")
print("| braço | n | AbsRel | delta1 | RMSE | F-borda |")
print("|---|---|---|---|---|---|")
linhas = []
for b, ss in por_braco.items():
    ms = list(ss.values())
    linhas.append((st.mean(m["boundary_fscore"] for m in ms), b, ms))
for _, b, ms in sorted(linhas, reverse=True):
    print(f"| {ROT.get(b, b)} | {len(ms)} | "
          f"{st.mean(m['abs_rel'] for m in ms):.4f} | "
          f"{st.mean(m['d1'] for m in ms):.4f} | "
          f"{st.mean(m['rmse'] for m in ms):.4f} | "
          f"{st.mean(m['boundary_fscore'] for m in ms):.4f} |")
