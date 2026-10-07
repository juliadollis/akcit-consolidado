#!/usr/bin/env python3
"""Sobe checkpoints do reteste da curvatura para o Hub, na org akcit-pixel.

REGRA: checkpoint que nao esta no Hub nao existe. Ja perdemos 37 de 47 numa
liberacao de quota no cluster. O raid NAO e armazenamento duravel.
"""
import os
import glob
from huggingface_hub import HfApi

REPO = "juliadollis/depthpro-spring-ft"
RAIZ = "/host/runs_riemann"
PRESERVADAS = "/host/runs_riemann_metricas_preservadas"

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
print(f"repo: {REPO} (privado)", flush=True)

ckpts = sorted(glob.glob(f"{RAIZ}/*/seed_*/best.pt"))
print(f"checkpoints a subir: {len(ckpts)}", flush=True)
for c in ckpts:
    rel = os.path.relpath(c, RAIZ)
    print(f"  {rel}  {os.path.getsize(c)/1e9:.2f} GB", flush=True)
    api.upload_file(path_or_fileobj=c, path_in_repo=f"pesos/{rel}",
                    repo_id=REPO, repo_type="model")

for j in sorted(glob.glob(f"{RAIZ}/*/seed_*/*.json")):
    api.upload_file(path_or_fileobj=j,
                    path_in_repo=f"pesos/{os.path.relpath(j, RAIZ)}",
                    repo_id=REPO, repo_type="model")
print("metricas dos treinos com peso anexadas", flush=True)

for j in sorted(glob.glob(f"{PRESERVADAS}/*/seed_*/*.json")):
    rel = os.path.relpath(j, PRESERVADAS)
    api.upload_file(path_or_fileobj=j,
                    path_in_repo=f"metricas_campanha_completa/{rel}",
                    repo_id=REPO, repo_type="model")
print("metricas da campanha completa anexadas", flush=True)

api.upload_file(path_or_fileobj="/host/README_HF.md", path_in_repo="README.md",
                repo_id=REPO, repo_type="model")
print("README enviado", flush=True)

# remove a copia de teste de cota que EU subi minutos atras. E duplicata exata do
# arquivo que agora esta em pesos/B3_gauss_metrica_teto50/seed_0/best.pt, e a cota
# privada esta apertada.
try:
    api.delete_file("teste_de_cota/best.pt", repo_id=REPO, repo_type="model")
    print("duplicata de teste removida", flush=True)
except Exception as e:
    print("nao removi a duplicata de teste:", e, flush=True)
print("FIM", flush=True)
