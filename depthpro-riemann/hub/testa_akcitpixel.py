import os
import glob
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "akcit-pixel/depthpro-spring-ft"
alvo = sorted(glob.glob("/host/runs_riemann/B3_gauss_metrica_teto50/seed_*/best.pt"))[1]
print("testando %s com %s (%.2f GB)" % (REPO, os.path.basename(os.path.dirname(alvo)),
                                        os.path.getsize(alvo) / 1e9), flush=True)
try:
    api.upload_file(path_or_fileobj=alvo,
                    path_in_repo="pesos/B3_gauss_metrica_teto50/seed_1/best.pt",
                    repo_id=REPO, repo_type="model")
    print("OK  a cota de akcit-pixel foi liberada", flush=True)
except Exception as e:
    msg = str(e)
    print("AINDA BLOQUEADO ->",
          "storage limit" if "storage limit" in msg else msg[:200].replace("\n", " "),
          flush=True)
fs = api.list_repo_files(REPO, repo_type="model")
print("arquivos agora em %s: %d" % (REPO, len(fs)), flush=True)
