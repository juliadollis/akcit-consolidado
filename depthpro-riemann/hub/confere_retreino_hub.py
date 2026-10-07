import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "juliadollis/depthpro-spring-ft"
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
alvo = [s for s in info.siblings if s.rfilename.startswith("pesos_retreino/")]
print(f"arquivos em pesos_retreino/: {len(alvo)}\n")
por_seed = {}
for s in alvo:
    partes = s.rfilename.split("/")
    seed = partes[2] if len(partes) > 3 else "?"
    por_seed.setdefault(seed, []).append((partes[-1], s.size))
for seed in sorted(por_seed):
    arqs = sorted(por_seed[seed])
    nomes = ", ".join(n for n, _ in arqs)
    tem_metrica = any(n == "test_metrics.json" for n, _ in arqs)
    print(f"  {seed}: {nomes}")
    if not tem_metrica:
        print(f"      ^^ SEM test_metrics.json: peso publicado sem o numero dele")
