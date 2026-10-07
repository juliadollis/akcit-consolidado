import glob
import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "juliadollis/depthpro-spring-ft"
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
no_hub = {s.rfilename for s in info.siblings if s.rfilename.endswith("best.pt")}
print(f"best.pt no Hub: {len(no_hub)}")
for pref in sorted({f.split('/')[0] for f in no_hub}):
    print(f"  {pref}/: {sum(1 for f in no_hub if f.startswith(pref + '/'))}")

# mapeia cada checkpoint local para onde ele DEVERIA estar no Hub
locais = {}
for raiz, pref in (("/host/runs_riemann", "pesos"),
                   ("/host/runs_retreino", "pesos_retreino"),
                   ("/host/runs_b0_retreino", "pesos_retreino")):
    for c in glob.glob(f"{raiz}/*/seed_*/best.pt"):
        p = c.split("/")
        locais[c] = f"{pref}/{p[-3]}/{p[-2]}/best.pt"

print(f"\ncheckpoints locais: {len(locais)}")
faltam = {c: d for c, d in locais.items() if d not in no_hub}
print(f"FALTAM no Hub: {len(faltam)}")
for c, d in sorted(faltam.items(), key=lambda x: x[1]):
    print(f"  {d}")
