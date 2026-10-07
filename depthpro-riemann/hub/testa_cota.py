import os
import glob
from huggingface_hub import HfApi

tok = os.environ["HF_TOKEN"]
api = HfApi(token=tok)
CAND = ["AKCITPixel3", "AkcitPixel2", "julia-se"]
alvo = glob.glob("/host/runs_riemann/B3_gauss_metrica_teto50/seed_0/best.pt")[0]
print("teste com %s (%.2f GB)" % (alvo, os.path.getsize(alvo) / 1e9), flush=True)

for org in CAND:
    repo = org + "/depthpro-spring-ft"
    try:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=alvo, path_in_repo="teste_de_cota/best.pt",
                        repo_id=repo, repo_type="model")
        print("OK  %s aceitou o upload privado" % org, flush=True)
        break
    except Exception as e:
        msg = str(e)
        curto = "storage limit" if "storage limit" in msg else msg[:160].replace("\n", " ")
        print("FALHOU %-14s -> %s" % (org, curto), flush=True)
