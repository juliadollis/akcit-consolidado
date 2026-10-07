import os
from collections import Counter
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "juliadollis/depthpro-spring-ft"
fs = api.list_repo_files(REPO, repo_type="model")
print("repo:", REPO, "| arquivos:", len(fs))
print("  best.pt        :", sum(1 for f in fs if f.endswith("best.pt")))
print("  test_metrics   :", sum(1 for f in fs if f.endswith("test_metrics.json")))
print("  README.md      :", "README.md" in fs)
pref = Counter(f.split("/")[0] for f in fs)
print("  por pasta      :", dict(pref))
print("\n  pesos por braco:")
for b in sorted({f.split("/")[1] for f in fs if f.startswith("pesos/")}):
    n = sum(1 for f in fs if f.startswith("pesos/" + b + "/") and f.endswith("best.pt"))
    print("    %-28s %d checkpoints" % (b, n))
info = api.repo_info(REPO, repo_type="model")
print("\n  privado:", info.private)
