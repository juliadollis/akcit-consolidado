import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
w = api.whoami()
orgs = [o["name"] for o in w.get("orgs", [])]
print("procurando 'depth' ou 'pro' nas orgs:", ", ".join(orgs), "\n")
achou = False
for org in orgs:
    for fn, tipo in ((api.list_models, "model"), (api.list_datasets, "dataset")):
        try:
            for r in fn(author=org, limit=500):
                n = r.id.lower()
                if "depth" in n or "dephpro" in n or "pro" in n:
                    print(f"  [{tipo}] {r.id}")
                    achou = True
        except Exception as e:
            print(f"  ({org} {tipo}: {str(e)[:60]})")
if not achou:
    print("  (nada com 'depth' nas orgs)")

print("\nrepos MAIS RECENTES de juliadollis (models):")
ms = list(api.list_models(author="juliadollis", limit=1000, sort="lastModified",
                          direction=-1))
for m in ms[:12]:
    print(f"  {m.id}")
