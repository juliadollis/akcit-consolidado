import glob
import os
from collections import Counter

from huggingface_hub import HfApi

REPO = "akcit-dephpro/depthpro-riemann-modelos"
api = HfApi(token=os.environ["HF_TOKEN"])
pts = [f for f in api.list_repo_files(REPO, repo_type="model") if f.endswith("best.pt")]

print("=== ablacao_seeds no Hub, por seed ===")
print(dict(Counter(f.split("/")[1] for f in pts if f.startswith("ablacao_seeds/"))))

print("\n=== ablacao_spring no Hub (seed 42) ===")
for f in sorted(f for f in pts if f.startswith("ablacao_spring/")):
    print("  ", f.split("/")[1])

print("\n=== em disco AGORA ===")
loc = sorted(glob.glob("/host/runs_*/**/best.pt", recursive=True))
print("total:", len(loc))
for c in loc:
    print("  ", c.replace("/host/", ""))

print("\n=== ORFAOS: em disco e NAO no Hub ===")
nomes_hub = set()
for f in pts:
    nomes_hub.add("/".join(f.split("/")[1:]))
orf = []
for c in loc:
    rel = "/".join(c.replace("/host/", "").split("/")[1:])
    if rel not in nomes_hub:
        orf.append((c, os.path.getsize(c)))
if not orf:
    print("  (nenhum)")
for c, t in orf:
    print("  %6.2f GB  %s" % (t / 1e9, c.replace("/host/", "")))
