#!/usr/bin/env python3
"""Monta as listas de repos da campanha Eq. 4 para a passada por imagem.

Inclui as linhas de IDENTIDADE da campanha antiga de proposito: a identidade
devolve a entrada sem tocar, entao ela nao depende do plano de foco e o numero
dela vale igual nas duas campanhas. Sem ela nao existe o piso de cada mesa, e a
margem sobre a identidade e a unica comparacao valida ENTRE mesas.
"""
import os
from datasets import load_dataset

tok = os.environ["HF_TOKEN"]
bru = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok,
                   download_mode="force_redownload").to_pandas()
todos = list(dict.fromkeys(bru["Dataset"].astype(str)))

eq4 = [r for r in todos if "bokeh-eq4-" in r]
ident = [r for r in todos if "identidade" in r and "bokeh-eq4-" not in r]
alvo = eq4 + ident
print("repos da campanha Eq.4:", len(eq4))
print("linhas de identidade herdadas:", len(ident), ident)
print("repos no total:", len(alvo))

gpus = [0, 1, 2, 3, 5, 6, 7]
for i, g in enumerate(gpus):
    with open("/host/lista_eq4_gpu%d.txt" % g, "w") as f:
        f.write("\n".join(alvo[i::len(gpus)]) + "\n")
with open("/host/lista_eq4_todos.txt", "w") as f:
    f.write("\n".join(alvo) + "\n")
print("listas escritas")
