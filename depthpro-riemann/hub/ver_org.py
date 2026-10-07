import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
w = api.whoami()
orgs = {o["name"]: o.get("roleInOrg") for o in w.get("orgs", [])}
print("akcit-dephpro nas minhas orgs?", "akcit-dephpro" in orgs)
if "akcit-dephpro" in orgs:
    print("  papel:", orgs["akcit-dephpro"])
else:
    print("  orgs com 'akcit' ou 'dep':",
          [o for o in orgs if "akcit" in o.lower() or "dep" in o.lower()])

for fn, tipo in ((api.list_models, "model"), (api.list_datasets, "dataset")):
    try:
        rs = list(fn(author="akcit-dephpro", limit=100))
        print(f"  {tipo}s em akcit-dephpro: {len(rs)}")
        for r in rs:
            print(f"     {r.id}")
    except Exception as e:
        print(f"  {tipo}s: erro {str(e)[:90]}")

# teste de escrita real, com um repo descartavel
try:
    rid = api.create_repo("akcit-dephpro/_teste_de_escrita", repo_type="model",
                          private=True, exist_ok=True)
    print("\nESCRITA OK: consigo criar repo em akcit-dephpro ->", rid)
except Exception as e:
    print("\nESCRITA FALHOU:", str(e)[:200])
