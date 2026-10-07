#!/usr/bin/env python3
"""Sobe o retreino do B0 para o Hub, e compara com o numero original.

A comparacao existe porque o retreino NAO e bit a bit igual: o codigo original
nao chama torch.use_deterministic_algorithms, entao duas execucoes da mesma seed
podem divergir, e o early stop amplifica isso. A diferenca medida aqui e uma
estimativa util do piso de ruido do protocolo.
"""
import glob
import json
import os

from huggingface_hub import HfApi

REPO = "juliadollis/depthpro-spring-ft"
NOVO = "/host/runs_b0_retreino/B0_berhu"
ORIG = "/host/runs_riemann_metricas_preservadas/B0_berhu"

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

subidos = 0
for c in sorted(glob.glob(f"{NOVO}/seed_*/best.pt")):
    rel = os.path.relpath(c, os.path.dirname(NOVO))
    api.upload_file(path_or_fileobj=c, path_in_repo=f"pesos_retreino/{rel}",
                    repo_id=REPO, repo_type="model")
    subidos += 1
    print(f"  subiu {rel}", flush=True)
for j in sorted(glob.glob(f"{NOVO}/seed_*/*.json")):
    rel = os.path.relpath(j, os.path.dirname(NOVO))
    api.upload_file(path_or_fileobj=j, path_in_repo=f"pesos_retreino/{rel}",
                    repo_id=REPO, repo_type="model")
print(f"checkpoints subidos: {subidos}", flush=True)

print("\nretreino contra o original, mesma seed:", flush=True)
print(f"{'seed':>4s} {'F-borda novo':>13s} {'orig':>8s} {'delta':>8s} "
      f"{'AbsRel novo':>12s} {'orig':>8s} {'delta':>8s}", flush=True)
for q in sorted(glob.glob(f"{NOVO}/seed_*/test_metrics.json")):
    s = int(q.split("seed_")[1].split("/")[0])
    o = f"{ORIG}/seed_{s}/test_metrics.json"
    if not os.path.exists(o):
        continue
    a, b = json.loads(open(q).read()), json.loads(open(o).read())
    print(f"{s:>4d} {a['boundary_fscore']:13.4f} {b['boundary_fscore']:8.4f} "
          f"{a['boundary_fscore']-b['boundary_fscore']:+8.4f} "
          f"{a['abs_rel']:12.4f} {b['abs_rel']:8.4f} "
          f"{a['abs_rel']-b['abs_rel']:+8.4f}", flush=True)
print("FIM", flush=True)
