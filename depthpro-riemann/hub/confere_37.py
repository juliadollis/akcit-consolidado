import os
from collections import Counter
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "juliadollis/depthpro-spring-ft"
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
tam = {s.rfilename: (s.size or 0) for s in info.siblings}
pesos = [f for f in tam if f.endswith("best.pt")]
print("repo:", REPO, "| privado:", info.private)
print("best.pt no Hub:", len(pesos))
por = Counter(f.split("/")[0] for f in pesos)
print("por pasta:", dict(por))
print()
for pasta in sorted(por):
    bracos = Counter(f.split("/")[1] for f in pesos if f.startswith(pasta + "/"))
    for b in sorted(bracos):
        print("  %-18s %-28s %d" % (pasta, b, bracos[b]))
gb = sum(tam[f] for f in pesos) / 1e9
print("\ntotal de pesos no Hub: %.1f GB" % gb)
