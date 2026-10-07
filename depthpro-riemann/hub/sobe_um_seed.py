#!/usr/bin/env python3
"""Sobe UMA seed recem-treinada para o Hub, e compara com o numero original.

Chamado pela fila logo depois que a seed fecha. A comparacao contra o original
existe porque o retreino nao e bit a bit igual: o codigo nao chama
torch.use_deterministic_algorithms, e o early stop amplifica qualquer diferenca.
A diferenca medida aqui e uma estimativa do piso de ruido do protocolo.
"""
import glob
import json
import os

from huggingface_hub import HfApi

REPO = "juliadollis/depthpro-spring-ft"
NOME = os.environ["FR_NOME"]
SEED = os.environ["FR_SEED"]
SD = f"/host/runs_retreino/{NOME}/seed_{SEED}"
ORIG = f"/host/runs_riemann_metricas_preservadas/{NOME}/seed_{SEED}/test_metrics.json"

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

enviados = []
for f in sorted(glob.glob(f"{SD}/best.pt") + glob.glob(f"{SD}/*.json")):
    destino = f"pesos_retreino/{NOME}/seed_{SEED}/{os.path.basename(f)}"
    api.upload_file(path_or_fileobj=f, path_in_repo=destino,
                    repo_id=REPO, repo_type="model")
    enviados.append(os.path.basename(f))
print(f"[hub] {NOME} seed {SEED}: {', '.join(enviados)}", flush=True)

novo = f"{SD}/test_metrics.json"
if os.path.exists(novo) and os.path.exists(ORIG):
    a, b = json.loads(open(novo).read()), json.loads(open(ORIG).read())
    linha = []
    for k, r in (("boundary_fscore", "F-borda"), ("boundary_fmax", "fmax"),
                 ("abs_rel", "AbsRel")):
        linha.append(f"{r} {a[k]:.4f} vs {b[k]:.4f} ({a[k]-b[k]:+.4f})")
    print(f"[hub] retreino x original, {NOME} seed {SEED}: " + " | ".join(linha),
          flush=True)
else:
    print(f"[hub] sem original para comparar: {NOME} seed {SEED}", flush=True)
