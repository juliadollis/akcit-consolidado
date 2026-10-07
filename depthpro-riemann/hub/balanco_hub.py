import os
import glob
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "juliadollis/depthpro-spring-ft"
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
fs = {s.rfilename for s in info.siblings}

pesos = sorted(f for f in fs if f.startswith("pesos/") and f.endswith("best.pt"))
retre = sorted(f for f in fs if f.startswith("pesos_retreino/") and f.endswith("best.pt"))
met = [f for f in fs if f.startswith("metricas_campanha_completa/")
       and f.endswith("test_metrics.json")]
print("=== juliadollis/depthpro-spring-ft ===")
print(f"  pesos originais que sobreviveram : {len(pesos)}")
print(f"  pesos do retreino                : {len(retre)}")
print(f"  metricas da campanha (47 treinos): {len(met)}")

# quais das 47 rodadas tem peso no Hub hoje
def chave(f):
    p = f.split("/")
    return (p[1], p[2])  # braco, seed_N
tem = {chave(f) for f in pesos} | {chave(f) for f in retre}
todas = {chave(f) for f in met}
print(f"\n  rodadas da campanha com PESO no Hub: {len(tem & todas)} de {len(todas)}")
faltam = sorted(todas - tem)
print(f"  rodadas SEM peso no Hub            : {len(faltam)}")
from collections import Counter
for b, c in sorted(Counter(b for b, _ in faltam).items()):
    print(f"      {b:<28s} {c}")

# no cluster, o que existe de peso e ainda nao esta no Hub
print("\n=== pesos no /raid ainda nao publicados ===")
local = sorted(glob.glob("/host/runs_retreino/*/seed_*/best.pt")
               + glob.glob("/host/runs_b0_retreino/*/seed_*/best.pt"))
for c in local:
    p = c.split("/")
    k = (p[-3], p[-2])
    marca = "no Hub" if k in tem else "FALTA"
    tm = os.path.exists(os.path.join(os.path.dirname(c), "test_metrics.json"))
    print(f"  {p[-3]}/{p[-2]}: {marca}"
          + ("" if tm else "   (ainda treinando: sem test_metrics.json)"))
