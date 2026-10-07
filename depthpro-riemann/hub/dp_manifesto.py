#!/usr/bin/env python3
"""Enumera o que sera publicado. NAO sobe nada, NAO apaga nada.

Regra: so entra seed que tenha test_metrics.json ao lado do best.pt.
"""
import json, os, sys
from pathlib import Path

H = Path("/host")
JSONS = ("test_metrics.json", "summary.json", "history.json")

FONTES_SEED = [
    ("originais", H / "runs_riemann",      "pesos_originais"),
    ("retreino",  H / "runs_retreino",     "pesos_retreino"),
    ("retreino",  H / "runs_b0_retreino",  "pesos_retreino"),
    ("confirma",  H / "runs_confirma",     "pesos_confirma"),
]

itens = []        # (categoria, origem_abs, destino_no_repo, tamanho)
pulados = []      # (categoria, caminho, motivo)
destinos = {}     # destino -> origem, para detectar colisao

def add(cat, src: Path, dest: str):
    if dest in destinos:
        print(f"!! COLISAO: {dest} <- {destinos[dest]} e {src}", file=sys.stderr)
        sys.exit(2)
    destinos[dest] = str(src)
    itens.append((cat, str(src), dest, src.stat().st_size))

for cat, raiz, prefixo in FONTES_SEED:
    if not raiz.is_dir():
        pulados.append((cat, str(raiz), "diretorio inexistente")); continue
    for braco in sorted(p for p in raiz.iterdir() if p.is_dir()):
        for seed in sorted(p for p in braco.iterdir() if p.is_dir()):
            tm = seed / "test_metrics.json"
            best = seed / "best.pt"
            if not tm.is_file():
                pulados.append((cat, f"{raiz.name}/{braco.name}/{seed.name}",
                                "sem test_metrics.json"))
                continue
            if not best.is_file():
                pulados.append((cat, f"{raiz.name}/{braco.name}/{seed.name}",
                                "sem best.pt")); continue
            base = f"{prefixo}/{braco.name}/{seed.name}"
            add(cat, best, f"{base}/best.pt")
            for j in JSONS:
                f = seed / j
                if f.is_file():
                    add(cat, f, f"{base}/{j}")

# ablacao: configs SEM test_metrics.json por design; numeros vivem em
# ablation_results.csv na raiz. So entra config com summary.json (= run fechado).
abl = H / "runs_ablacao_spring" / "ablacao_spring"
if abl.is_dir():
    csv = abl / "ablation_results.csv"
    if csv.is_file():
        add("ablacao", csv, "ablacao_spring/ablation_results.csv")
    for cfg in sorted(p for p in abl.iterdir() if p.is_dir()):
        if not (cfg / "summary.json").is_file():
            pulados.append(("ablacao", f"ablacao_spring/{cfg.name}",
                            "sem summary.json (treino em andamento)"))
            continue
        for f in sorted(cfg.iterdir()):
            if f.is_file():
                add("ablacao", f, f"ablacao_spring/{cfg.name}/{f.name}")

# metricas preservadas: so json/csv
mp = H / "runs_riemann_metricas_preservadas"
if mp.is_dir():
    for f in sorted(mp.rglob("*")):
        if f.is_file():
            add("metricas", f, f"metricas_campanha_completa/{f.relative_to(mp)}")

# avaliacoes (repo dataset, manifesto separado)
av_itens = []
av = H / "avaliacoes"
for rot in sorted(p for p in av.iterdir() if p.is_dir()):
    for mesa in sorted(p for p in rot.iterdir() if p.is_dir()):
        for f in sorted(mesa.iterdir()):
            if f.is_file():
                av_itens.append(("avaliacao", str(f),
                                 f"{rot.name}/{mesa.name}/{f.name}", f.stat().st_size))

def resumo(lst, titulo):
    print(f"\n===== {titulo} =====")
    porcat = {}
    for cat, _s, _d, sz in lst:
        c = porcat.setdefault(cat, [0, 0])
        c[0] += 1; c[1] += sz
    tot = 0
    for cat, (n, sz) in sorted(porcat.items()):
        print(f"  {cat:12s} {n:5d} arquivos  {sz/2**30:8.2f} GiB")
        tot += sz
    print(f"  {'TOTAL':12s} {len(lst):5d} arquivos  {tot/2**30:8.2f} GiB")

resumo(itens, "MODELOS")
n_ckpt = {}
for cat, _s, d, _z in itens:
    if d.endswith("best.pt"):
        n_ckpt[cat] = n_ckpt.get(cat, 0) + 1
print("  checkpoints (best.pt) por categoria:", n_ckpt)

resumo(av_itens, "AVALIACOES")
print("  rotulos:", len({d.split('/')[0] for _c, _s, d, _z in av_itens}))

print("\n===== PULADOS =====")
for cat, p, m in pulados:
    print(f"  [{cat}] {p}: {m}")
if not pulados:
    print("  (nenhum)")

json.dump([{"cat": c, "src": s, "dest": d, "size": z} for c, s, d, z in itens],
          open("/host/_hub_manifesto_modelos.json", "w"), indent=1)
json.dump([{"cat": c, "src": s, "dest": d, "size": z} for c, s, d, z in av_itens],
          open("/host/_hub_manifesto_avaliacoes.json", "w"), indent=1)
json.dump([{"cat": c, "caminho": p, "motivo": m} for c, p, m in pulados],
          open("/host/_hub_pulados.json", "w"), indent=1)
print("\nmanifestos escritos.")
