import os
from collections import Counter

from huggingface_hub import HfApi

REPO = "akcit-dephpro/depthpro-riemann-modelos"
api = HfApi(token=os.environ["HF_TOKEN"])
fs = api.list_repo_files(REPO, repo_type="model")
pt = [f for f in fs if f.endswith("best.pt")]
print("arquivos no repo:", len(fs))
print("best.pt no repo :", len(pt))
print("\npor prefixo de topo:")
for k, v in sorted(Counter(f.split("/")[0] for f in pt).items()):
    print("  %-22s %d" % (k, v))
print("\nablacao_seeds por seed:")
for k, v in sorted(Counter(f.split("/")[1] for f in pt
                           if f.startswith("ablacao_seeds/")).items()):
    print("  %-10s %d" % (k, v))
print("\nablacao_spring (seed 42):")
for f in sorted(f for f in pt if f.startswith("ablacao_spring/")):
    print("  " + f)
