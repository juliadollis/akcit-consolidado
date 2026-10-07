import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
w = api.whoami()
orgs = {o["name"]: o.get("roleInOrg") for o in w.get("orgs", [])}
for cand in ("akcit-h100n1", "akcit-h100-n1", "akcith100n1"):
    if cand in orgs:
        print(f"org '{cand}' JA EXISTE, papel={orgs[cand]}")
        break
else:
    print("nenhuma org akcit-h100n1 encontrada nas minhas orgs")
    print("orgs com 'akcit':", [o for o in orgs if "akcit" in o.lower()])
    # tenta escrever para descobrir se existe e eu tenho acesso
    try:
        r = api.create_repo("akcit-h100n1/_teste", repo_type="model",
                            private=True, exist_ok=True)
        print("ESCRITA OK em akcit-h100n1 ->", r)
        api.delete_repo("akcit-h100n1/_teste", repo_type="model")
        print("(repo de teste removido)")
    except Exception as e:
        print("ESCRITA FALHOU:", str(e)[:250])
