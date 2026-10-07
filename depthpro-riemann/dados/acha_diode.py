import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
print("datasets com 'diode' no Hub:")
vistos = []
for d in api.list_datasets(search="diode", limit=40):
    vistos.append(d.id)
    print(f"  {d.id}")
if not vistos:
    print("  (nenhum)")

print("\ndetalhe dos mais promissores:")
for rid in vistos[:6]:
    try:
        info = api.repo_info(rid, repo_type="dataset", files_metadata=True)
        tot = sum(s.size or 0 for s in info.siblings)
        exts = {}
        for s in info.siblings:
            e = s.rfilename.rsplit(".", 1)[-1] if "." in s.rfilename else "-"
            exts[e] = exts.get(e, 0) + 1
        print(f"  {rid}: {len(info.siblings)} arquivos, {tot/1e9:.2f} GB, {exts}")
    except Exception as e:
        print(f"  {rid}: erro {str(e)[:70]}")
