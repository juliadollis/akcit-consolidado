import os
from collections import Counter

from huggingface_hub import HfApi

REPO = "akcit-dephpro/depthpro-riemann-modelos"
api = HfApi(token=os.environ["HF_TOKEN"])
fs = api.list_repo_files(REPO, repo_type="model")
pts = [f for f in fs if f.endswith("best.pt")]
print("repo:", REPO)
print("arquivos totais:", len(fs))
print("best.pt:", len(pts))
print("por prefixo:", dict(Counter(f.split("/")[0] for f in pts)))
print()
for pref in sorted(set(f.split("/")[0] for f in pts)):
    sub = sorted(f for f in pts if f.startswith(pref + "/"))
    print(f"{pref}: {len(sub)}")
    for f in sub[:3]:
        print("   ", f)
    if len(sub) > 3:
        print("    ...")
