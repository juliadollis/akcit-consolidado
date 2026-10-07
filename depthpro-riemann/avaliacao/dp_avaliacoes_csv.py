#!/usr/bin/env python3
"""Monta avaliacoes.csv: uma linha por (modelo, mesa). So le, nao apaga nada."""
import csv, json
from pathlib import Path

AV = Path("/host/avaliacoes")
SAIDA = Path("/host/_hub_avaliacoes.csv")

FIXAS = ["rotulo", "origem", "braco", "seed", "mesa", "n_imagens", "n_cenas"]

linhas, chaves_metrica, avisos = [], [], []

for rot in sorted(p for p in AV.iterdir() if p.is_dir()):
    for mesa in sorted(p for p in rot.iterdir() if p.is_dir()):
        tm, mt = mesa / "test_metrics.json", mesa / "meta.json"
        if not tm.is_file():
            avisos.append(f"{rot.name}/{mesa.name}: sem test_metrics.json"); continue
        met = json.loads(tm.read_text())
        meta = json.loads(mt.read_text()) if mt.is_file() else {}
        for k in met:
            if k not in chaves_metrica:
                chaves_metrica.append(k)
        # rotulo = <origem>__<braco>__seed_N; zero_shot nao segue o formato
        partes = rot.name.split("__")
        if len(partes) == 3 and partes[2].startswith("seed_"):
            origem, braco, seed = partes[0], partes[1], partes[2][len("seed_"):]
        else:
            origem = braco = seed = ""
            avisos.append(f"{rot.name}: rotulo fora do formato <origem>__<braco>__seed_N; "
                          "origem/braco/seed deixados vazios")
        lin = {"rotulo": rot.name, "origem": origem, "braco": braco, "seed": seed,
               "mesa": mesa.name,
               "n_imagens": meta.get("n_imagens", ""),
               "n_cenas": meta.get("n_cenas", ""),
               "checkpoint": meta.get("checkpoint", "")}
        lin.update(met)
        linhas.append(lin)

faltando = [l["rotulo"] for l in linhas if any(k not in l for k in chaves_metrica)]
if faltando:
    avisos.append("linhas com metricas faltando: " + ", ".join(faltando))

cols = FIXAS + chaves_metrica + ["checkpoint"]
with open(SAIDA, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for l in sorted(linhas, key=lambda x: x["rotulo"]):
        w.writerow(l)

print(f"linhas (sem cabecalho): {len(linhas)}")
print(f"colunas ({len(cols)}): {', '.join(cols)}")
print("mesas:", sorted({l['mesa'] for l in linhas}))
print("origens:", sorted({l['origem'] for l in linhas}))
print("bracos:", sorted({l['braco'] for l in linhas}))
print("avisos:")
for a in avisos:
    print("  -", a)
print("\n--- 2 primeiras linhas ---")
print(open(SAIDA).readline().strip())
