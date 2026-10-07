import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
try:
    eu = api.whoami()
    print("conta:", eu.get("name"))
    print("orgs:", [o.get("name") for o in eu.get("orgs", [])])
except Exception as e:
    print("whoami:", str(e)[:120])

for org in ("akcit-dephpro", "akcit-h100n1", "akcit-pixel", "juliadollis"):
    print("\n=== " + org + " ===")
    try:
        ms = list(api.list_models(author=org))
        for m in ms[:8]:
            try:
                info = api.repo_info(m.id, repo_type="model", files_metadata=True)
                tot = sum(s.size or 0 for s in info.siblings)
                print("  %-55s privado=%-5s %.1f GB" % (m.id, info.private, tot / 1e9))
            except Exception as e:
                print("  %-55s erro %s" % (m.id, str(e)[:40]))
        if not ms:
            print("  (nenhum modelo)")
    except Exception as e:
        print("  erro:", str(e)[:120])
