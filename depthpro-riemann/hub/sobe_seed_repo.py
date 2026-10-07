#!/usr/bin/env python3
"""Sobe uma seed da ablacao para um repo escolhido.

A org akcit-dephpro bateu o teto de 100 GB de storage privado e devolve 403 no
endpoint de LFS, foi isso que impediu o upload automatico das seeds 45 e 46.
Destino alternativo: akcit-h100n1, que tem folga.

Uso: sobe_seed_repo.py <seed> <repo_id>
"""
import os
import sys

from huggingface_hub import HfApi

SEED, REPO = sys.argv[1], sys.argv[2]
RAIZ = f"/host/runs_ablacao_seeds/seed_{SEED}"
ARQS = ("best.pt", "summary.json", "history.json", "test_metrics.json")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

# nao sobrescreve o que ja esta la: outra sessao pode estar subindo em paralelo
ja = set(api.list_repo_files(REPO, repo_type="model"))
print(f"ja no repo: {len(ja)} arquivos", flush=True)

enviados = pulados = 0
for braco in sorted(os.listdir(RAIZ)):
    d = os.path.join(RAIZ, braco)
    if not os.path.isdir(d):
        continue
    for nome in ARQS:
        local = os.path.join(d, nome)
        if not os.path.exists(local):
            continue
        destino = f"seed_{SEED}/{braco}/{nome}"
        if destino in ja:
            pulados += 1
            continue
        api.upload_file(path_or_fileobj=local, path_in_repo=destino,
                        repo_id=REPO, repo_type="model")
        print(f"  {destino}", flush=True)
        enviados += 1

print(f"[hub] seed {SEED} -> {REPO}: {enviados} enviados, {pulados} ja existiam",
      flush=True)
