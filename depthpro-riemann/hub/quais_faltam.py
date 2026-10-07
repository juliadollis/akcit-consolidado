import os
from collections import Counter
from huggingface_hub import HfApi

REPO = "akcit-dephpro/depthpro-riemann-modelos"
DESTINO = {"pesos_originais": "/host/runs_riemann",
           "pesos_retreino": "/host/runs_retreino_hub"}

api = HfApi(token=os.environ["HF_TOKEN"])
fs = [f for f in api.list_repo_files(REPO, repo_type="model") if f.endswith("best.pt")]
print(f"best.pt no Hub: {len(fs)}")
print("prefixos:", dict(Counter(f.split("/")[0] for f in fs)))
print("\nnao cobertos pelo mapa de destino:")
for f in sorted(fs):
    if f.split("/")[0] not in DESTINO:
        print("  ", f)
print("\ncobertos mas ausentes em disco:")
for f in sorted(fs):
    p = f.split("/")[0]
    if p in DESTINO:
        alvo = os.path.join(DESTINO[p], "/".join(f.split("/")[1:]))
        if not os.path.exists(alvo):
            print("  ", f)
