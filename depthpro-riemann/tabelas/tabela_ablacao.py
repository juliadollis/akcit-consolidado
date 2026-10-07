import glob
import json

ROT = {"heads__B0_berhu": ("B0 berHu (controle)", "berHu sozinho"),
       "heads__B1_berhu+grad": ("B1 +grad", "1a ordem"),
       "heads__B1_berhu+metric": ("B1 +metric", "1a ordem, quadratico"),
       "heads__B1_berhu+geod": ("B1 +geod", "1a ordem, elemento de arco"),
       "heads__B1_berhu+normal": ("B1 +normal", "1a ordem, normalizado"),
       "heads__B1_berhu+gauss": ("B1 +gauss", "2a ordem, curvatura"),
       "heads__B7_gauss_dom": ("B7 gauss dominante", "curvatura com peso alto"),
       "heads__B7_gaussheavy": ("B7 gauss pesado", "curvatura com peso muito alto"),
       "heads__B7_normal_dom": ("B7 normal dominante", "normal com peso alto"),
       "heads__B7_champion_prev": ("B7 campeao anterior", "melhor combinacao previa")}
MESAS = ["spring_test", "spring_val", "diode_val"]

d = {}
for q in glob.glob("/host/avaliacoes/runs_ablacao_spring__*/*/test_metrics.json"):
    rot = q.split("/")[-3]
    mesa = q.split("/")[-2]
    braco = rot.split("__", 2)[2]
    d.setdefault(braco, {})[mesa] = json.load(open(q))

ctrl = d.get("heads__B0_berhu", {})
for met, nome, maior_melhor in [("boundary_fscore", "F-borda", True),
                                ("abs_rel", "AbsRel", False)]:
    print(f"\n## {nome}\n")
    print("| braço | o que é | " + " | ".join(MESAS) + " | vs controle (diode) |")
    print("|---|---|" + "---|" * (len(MESAS) + 1))
    ordem = sorted(d, key=lambda b: -d[b].get("diode_val", {}).get(met, 0)
                   if maior_melhor else d[b].get("diode_val", {}).get(met, 9))
    for b in ordem:
        nm, oq = ROT.get(b, (b, "?"))
        vals = " | ".join(f"{d[b][m][met]:.4f}" if m in d[b] else "n/d"
                          for m in MESAS)
        if "diode_val" in d[b] and "diode_val" in ctrl and b != "heads__B0_berhu":
            dlt = d[b]["diode_val"][met] - ctrl["diode_val"][met]
            delta = f"{dlt:+.4f}"
        else:
            delta = "—"
        print(f"| {nm} | {oq} | {vals} | {delta} |")
