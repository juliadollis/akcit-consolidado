import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
alvos = []
for m in api.list_models(author="juliadollis", limit=1000):
    n = m.id.lower()
    if "depthpro" in n or "depth-pro" in n or "akcit" in n or "dephpro" in n:
        alvos.append((m.id, m.lastModified))
for d in api.list_datasets(author="juliadollis", limit=1000):
    n = d.id.lower()
    if "depthpro" in n or "depth-pro" in n or "akcit" in n or "dephpro" in n:
        alvos.append((d.id + "  [dataset]", d.lastModified))
print("candidatos em juliadollis:")
for i, (n, t) in enumerate(sorted(alvos, key=lambda x: str(x[1]), reverse=True)):
    print(f"  {n}   (mod. {t})")
if not alvos:
    print("  (nenhum)")
