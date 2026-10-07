#!/usr/bin/env python3
"""Sobe uma seed recem-treinada para o Hub. Generico: a raiz e o destino vem do
ambiente, para servir a qualquer fila.

Regra: checkpoint que nao esta no Hub nao existe.
"""
import glob
import os

from huggingface_hub import HfApi

REPO = "juliadollis/depthpro-spring-ft"
RAIZ = os.environ["FR_RAIZ"]
NOME = os.environ["FR_NOME"]
SEED = os.environ["FR_SEED"]
DEST = os.environ.get("FR_DEST", "pesos_retreino")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

sd = f"{RAIZ}/{NOME}/seed_{SEED}"
# so sobe se a seed FECHOU: sem test_metrics.json o best.pt e um parcial, e peso
# sem o numero dele engana quem baixar. Foi um defeito real da fila anterior.
if not os.path.exists(f"{sd}/test_metrics.json"):
    print(f"[hub] {NOME} seed {SEED}: sem test_metrics.json, NAO subindo parcial",
          flush=True)
    raise SystemExit(0)

enviados = []
for f in sorted(glob.glob(f"{sd}/best.pt") + glob.glob(f"{sd}/*.json")):
    api.upload_file(path_or_fileobj=f,
                    path_in_repo=f"{DEST}/{NOME}/seed_{SEED}/{os.path.basename(f)}",
                    repo_id=REPO, repo_type="model")
    enviados.append(os.path.basename(f))
print(f"[hub] {NOME} seed {SEED}: {', '.join(enviados)}", flush=True)
