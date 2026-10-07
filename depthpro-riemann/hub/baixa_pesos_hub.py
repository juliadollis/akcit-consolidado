#!/usr/bin/env python3
"""Baixa do Hub os pesos que nao estao mais em disco, para os fiscais avaliarem.

Os 47 best.pt da campanha foram apagados do /raid depois de publicados, para
liberar quota. Estao em akcit-dephpro/depthpro-riemann-modelos. Este script os
traz de volta no MESMO layout que o avalia_modelos.py espera, ou seja
runs_<origem>/<braco>/seed_N/best.pt, para o fiscal reconhecer sem alteracao.
"""
import os

from huggingface_hub import HfApi, hf_hub_download

REPO = "akcit-dephpro/depthpro-riemann-modelos"
# prefixo no Hub -> raiz local que o fiscal varre
DESTINO = {"pesos_originais": "/host/runs_riemann",
           "pesos_retreino": "/host/runs_retreino_hub"}

api = HfApi(token=os.environ["HF_TOKEN"])
fs = [f for f in api.list_repo_files(REPO, repo_type="model")
      if f.endswith("best.pt")]
print(f"best.pt no Hub: {len(fs)}", flush=True)

baixados = pulados = 0
for f in sorted(fs):
    pref = f.split("/")[0]
    if pref not in DESTINO:
        continue
    rel = "/".join(f.split("/")[1:])          # <braco>/seed_N/best.pt
    alvo = os.path.join(DESTINO[pref], rel)
    if os.path.exists(alvo):
        pulados += 1
        continue
    os.makedirs(os.path.dirname(alvo), exist_ok=True)
    print(f"  baixando {pref}/{rel}", flush=True)
    p = hf_hub_download(REPO, f, repo_type="model",
                        token=os.environ["HF_TOKEN"])
    # link duro quando possivel, para nao duplicar no disco
    try:
        os.link(p, alvo)
    except OSError:
        import shutil
        shutil.copy2(p, alvo)
    # as metricas junto, para o peso nunca ficar sem o numero dele
    for extra in ("test_metrics.json", "summary.json", "history.json"):
        origem = f.rsplit("/", 1)[0] + "/" + extra
        destino = os.path.join(os.path.dirname(alvo), extra)
        if origem in api.list_repo_files(REPO, repo_type="model") and \
                not os.path.exists(destino):
            try:
                pj = hf_hub_download(REPO, origem, repo_type="model",
                                     token=os.environ["HF_TOKEN"])
                os.link(pj, destino)
            except Exception:
                pass
    baixados += 1

print(f"\nbaixados: {baixados}   ja existiam: {pulados}", flush=True)
print("FIM", flush=True)
