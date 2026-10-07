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
MET = [("abs_rel", "AbsRel"), ("d1", "delta1"), ("rmse", "RMSE"),
       ("boundary_fscore", "F-borda"), ("boundary_fmax", "fmax")]

por = defaultdict(dict)
zs = None
for q in glob.glob("/host/avaliacoes/*/diode_val/test_metrics.json"):
    rot = q.split("/")[-3]
    if rot == "zero_shot":
        zs = json.load(open(q))
        continue
    p = rot.split("__")
    if len(p) != 3:
        continue
    s = int(p[2].replace("seed_", ""))
    if s < 6:
        por[p[1]][s] = json.load(open(q))

def fmt(m, k):
    return f"{m[k]:.4f}" if k in m else "n/d"

print("| braço | n | " + " | ".join(r for _, r in MET) + " |")
print("|---|---|" + "---|" * len(MET))
if zs:
    print("| zero-shot | - | " + " | ".join(fmt(zs, k) for k, _ in MET) + " |")
ordem = sorted(por, key=lambda b: -st.mean(
    m["boundary_fscore"] for m in por[b].values()))
for b in ordem:
    ms = list(por[b].values())
    print(f"| {ROT.get(b, b)} | {len(ms)} | " + " | ".join(
        f"{st.mean(m[k] for m in ms):.4f}" if k in ms[0] else "n/d"
        for k, _ in MET) + " |")

ctrl = por.get("B0_berhu", {})
print("\n### pareado contra o controle (mesma seed)\n")
print("| braço | n | delta F-borda | desvio | seeds a favor |")
print("|---|---|---|---|---|")
for b in ordem:
    if b == "B0_berhu":
        continue
    com = sorted(set(por[b]) & set(ctrl))
    if not com:
        continue
    d = [por[b][s]["boundary_fscore"] - ctrl[s]["boundary_fscore"] for s in com]
    dp = f"{st.stdev(d):.4f}" if len(d) > 1 else "n/d"
    print(f"| {ROT.get(b, b)} | {len(d)} | {st.mean(d):+.4f} | {dp} | "
          f"{sum(1 for x in d if x > 0)}/{len(d)} |")
