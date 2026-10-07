from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
tot = 0.0
linhas = []
for m in api.list_models(author="juliadollis", limit=200):
    try:
        info = api.model_info(m.id, files_metadata=True)
        sz = sum((s.size or 0) for s in info.siblings)
        n = len([s for s in info.siblings if s.rfilename.endswith(".safetensors")])
        vis = "PRIV" if info.private else "pub "
        if sz > 1e8:
            linhas.append((sz, n, m.id, vis))
        tot += sz
    except Exception:
        pass
linhas.sort(reverse=True)
print("  TOTAL nos seus repos de modelo: %.1f GB" % (tot / 1e9))
print()
for sz, n, rid, vis in linhas[:10]:
    print("   %7.1f GB  %3d pesos  %s  %s" % (sz / 1e9, n, vis, rid))
