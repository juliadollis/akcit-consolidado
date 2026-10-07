from huggingface_hub import HfApi
a = HfApi()
try:
    fs = a.list_repo_files("juliadollis/depthpro-spring-ft", repo_type="model")
    print("depthpro-spring-ft: %d arquivos" % len(fs))
    pref = {}
    for f in fs:
        k = "/".join(f.split("/")[:1])
        pref[k] = pref.get(k, 0) + 1
    print(pref)
    print("exemplo:", [f for f in fs if f.endswith("best.pt")][:3])
except Exception as e:
    print("err:", type(e).__name__, e)
import huggingface_hub as h
print("tem upload_large_folder:", hasattr(a, "upload_large_folder"))
print("hub", h.__version__)
