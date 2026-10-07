#!/usr/bin/env python3
"""Sobe uma seed da ablacao para o Hub assim que o treino dela fecha.

Regra da casa: checkpoint que nao esta no Hub nao existe. O /raid e area de
trabalho, nao armazenamento duravel -- em 10/09/2026 perdemos 37 de 47 best.pt
numa liberacao de quota, e as metricas sobreviveram mas os pesos nao.

Sobe para o mesmo repo onde ja esta a seed 42 da ablacao, com prefixo proprio,
para que as seis seeds fiquem lado a lado.
"""
import os
import sys

from huggingface_hub import HfApi

REPO = "akcit-dephpro/depthpro-riemann-modelos"
SEED = sys.argv[1]
RAIZ = f"/host/runs_ablacao_seeds/seed_{SEED}"
ARQS = ("best.pt", "summary.json", "history.json", "test_metrics.json")

api = HfApi(token=os.environ["HF_TOKEN"])
enviados = 0
for braco in sorted(os.listdir(RAIZ)):
    d = os.path.join(RAIZ, braco)
    if not os.path.isdir(d):
        continue
    for nome in ARQS:
        local = os.path.join(d, nome)
        if not os.path.exists(local):
            continue
        destino = f"ablacao_seeds/seed_{SEED}/{braco}/{nome}"
        api.upload_file(path_or_fileobj=local, path_in_repo=destino,
                        repo_id=REPO, repo_type="model")
        print(f"  {destino}", flush=True)
        enviados += 1

print(f"[hub] ablacao seed {SEED}: {enviados} arquivos", flush=True)
