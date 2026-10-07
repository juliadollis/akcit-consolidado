import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
for rid, tipo in (("AKCITPixel3/depthpro-riemann-spring-eval", "dataset"),
                  ("akcit-pixel/depthpro-spring-ft", "model"),
                  ("AKCITPixel3/depthpro-spring-ft", "model")):
    try:
        info = api.repo_info(rid, repo_type=tipo, files_metadata=True)
        fs = [s.rfilename for s in info.siblings]
        print(f"{rid}  [{tipo}]")
        print(f"   privado={info.private}  arquivos={len(fs)}  mod={info.lastModified}")
        for f in sorted(fs)[:8]:
            print(f"     {f}")
        if len(fs) > 8:
            print(f"     ... e mais {len(fs)-8}")
    except Exception as e:
        print(f"{rid} [{tipo}] -> {str(e)[:90]}")
    print()
