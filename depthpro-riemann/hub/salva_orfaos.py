#!/usr/bin/env python3
"""Salva as seeds 45 e 46 num destino que ainda tenha espaco.

O akcit-dephpro bateu o limite de armazenamento privado (403 Forbidden no LFS),
e por isso as duas seeds existem SO em disco -- que esta sob pressao de quota e
ja comeu 2 checkpoints da ablacao original. Tenta destinos em ordem e para no
primeiro que aceitar.
"""
import os
import sys

from huggingface_hub import HfApi

CANDIDATOS = ["akcit-h100n1/depthpro-ablacao-seeds",
              "juliadollis/depthpro-ablacao-seeds"]
SEEDS = ["45", "46"]
ARQS = ("best.pt", "summary.json", "history.json")

api = HfApi(token=os.environ["HF_TOKEN"])

for repo in CANDIDATOS:
    print(f"=== tentando {repo} ===", flush=True)
    try:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    except Exception as e:
        print(f"  nao consegui criar: {str(e)[:160]}", flush=True)
        continue

    enviados, falhou = 0, None
    for seed in SEEDS:
        raiz = f"/host/runs_ablacao_seeds/seed_{seed}"
        if not os.path.isdir(raiz):
            continue
        for braco in sorted(os.listdir(raiz)):
            d = os.path.join(raiz, braco)
            if not os.path.isdir(d):
                continue
            for nome in ARQS:
                local = os.path.join(d, nome)
                if not os.path.exists(local):
                    continue
                destino = f"seed_{seed}/{braco}/{nome}"
                try:
                    api.upload_file(path_or_fileobj=local, path_in_repo=destino,
                                    repo_id=repo, repo_type="model")
                    print(f"  ok {destino}", flush=True)
                    enviados += 1
                except Exception as e:
                    falhou = str(e)[:200]
                    print(f"  FALHOU {destino}: {falhou}", flush=True)
                    break
            if falhou:
                break
        if falhou:
            break

    print(f"  enviados: {enviados}", flush=True)
    if not falhou and enviados > 0:
        print(f"\nDESTINO OK: {repo} ({enviados} arquivos)", flush=True)
        sys.exit(0)

print("\nNENHUM DESTINO ACEITOU -- os pesos seguem so em disco", flush=True)
sys.exit(1)
