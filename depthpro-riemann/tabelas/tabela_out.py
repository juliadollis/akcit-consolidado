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
ABL = {"heads__B0_berhu": "abl B0 controle", "heads__B1_berhu+grad": "abl +grad",
       "heads__B1_berhu+metric": "abl +metric", "heads__B1_berhu+geod": "abl +geod",
       "heads__B1_berhu+normal": "abl +normal", "heads__B1_berhu+gauss": "abl +gauss",
       "heads__B7_gauss_dom": "abl gauss dom", "heads__B7_gaussheavy": "abl gauss pesado",
       "heads__B7_normal_dom": "abl normal dom", "heads__B7_champion_prev": "abl campeao"}
MET = [("abs_rel", "AbsRel"), ("d1", "delta1"), ("boundary_fscore", "F-borda")]


def carrega(mesa):
    seeds, abl, zs = defaultdict(dict), {}, None
    for q in glob.glob("/host/avaliacoes/*/" + mesa + "/test_metrics.json"):
        rot = q.split("/")[-3]
        if rot == "zero_shot":
            zs = json.load(open(q))
            continue
        if rot.startswith("runs_ablacao_spring__"):
            abl[rot.split("__", 2)[2]] = json.load(open(q))
            continue
        p = rot.split("__")
        if len(p) == 3:
            s = int(p[2].replace("seed_", ""))
            if s < 6:
                seeds[p[1]][s] = json.load(open(q))
    return seeds, abl, zs


def mf(d, k):
    return st.mean(m[k] for m in d.values())


sIN, aIN, zIN = carrega("diode_val")
sOU, aOU, zOU = carrega("diode_outdoor")

print("## campanha de seeds -- outdoor (n=6)\n")
print("| braco | " + " | ".join(r for _, r in MET) + " | F-borda indoors | vs ctrl |")
print("|---|---|---|---|---|---|")
ctrlO = sOU.get("B0_berhu", {})
cO = mf(ctrlO, "boundary_fscore") if ctrlO else 0.0
if zOU:
    print("| zero-shot | " + " | ".join("%.4f" % zOU[k] for k, _ in MET)
          + " | %.4f | -- |" % zIN["boundary_fscore"])
for b in sorted(sOU, key=lambda x: -mf(sOU[x], "boundary_fscore")):
    f = mf(sOU[b], "boundary_fscore")
    ins = mf(sIN[b], "boundary_fscore") if b in sIN else 0.0
    d = "--" if b == "B0_berhu" else "%+.4f" % (f - cO)
    print("| " + ROT.get(b, b) + " | "
          + " | ".join("%.4f" % mf(sOU[b], k) for k, _ in MET)
          + " | %.4f | %s |" % (ins, d))

print("\n## ablacao -- outdoor (n=1)\n")
print("| braco | AbsRel | F-borda out | F-borda indoors | vs ctrl |")
print("|---|---|---|---|---|")
ca = aOU.get("heads__B0_berhu", {}).get("boundary_fscore", 0.0)
for b in sorted(aOU, key=lambda x: -aOU[x]["boundary_fscore"]):
    ins = aIN[b]["boundary_fscore"] if b in aIN else 0.0
    d = "--" if b == "heads__B0_berhu" else "%+.4f" % (aOU[b]["boundary_fscore"] - ca)
    print("| %s | %.4f | %.4f | %.4f | %s |"
          % (ABL.get(b, b), aOU[b]["abs_rel"], aOU[b]["boundary_fscore"], ins, d))

print("\n## pareado por seed contra o controle -- outdoor\n")
print("| braco | n | delta F-borda | desvio | a favor |")
print("|---|---|---|---|---|")
for b in sorted(sOU, key=lambda x: -mf(sOU[x], "boundary_fscore")):
    if b == "B0_berhu":
        continue
    com = sorted(set(sOU[b]) & set(ctrlO))
    if not com:
        continue
    d = [sOU[b][s]["boundary_fscore"] - ctrlO[s]["boundary_fscore"] for s in com]
    dp = "%.4f" % st.stdev(d) if len(d) > 1 else "n/d"
    print("| %s | %d | %+.4f | %s | %d/%d |"
          % (ROT.get(b, b), len(d), st.mean(d), dp,
             sum(1 for x in d if x > 0), len(d)))
