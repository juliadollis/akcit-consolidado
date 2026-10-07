#!/usr/bin/env python3
"""Sobe os resultados da ablacao do Wallisson no Spring para o Hub."""
import glob
import os

from huggingface_hub import HfApi

REPO = "juliadollis/depthpro-spring-ft"
RAIZ = "/host/runs_ablacao_spring"

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

n = 0
for f in sorted(glob.glob(f"{RAIZ}/**/*", recursive=True)):
    if not os.path.isfile(f):
        continue
    # pesos so quando a config fechou (tem o CSV ou o summary ao lado)
    rel = os.path.relpath(f, RAIZ)
    api.upload_file(path_or_fileobj=f, path_in_repo=f"ablacao_spring/{rel}",
                    repo_id=REPO, repo_type="model")
    n += 1
    print(f"  {rel}", flush=True)
print(f"[hub] ablacao: {n} arquivos enviados", flush=True)
