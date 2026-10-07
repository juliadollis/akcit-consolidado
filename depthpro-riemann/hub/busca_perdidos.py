#!/usr/bin/env python3
"""Procura os dois pesos da ablacao original que nao estao em akcit-dephpro.

B7_champion_prev e B7_normal_dom existiam em disco e o disco foi varrido. Antes
de declarar perda, varrer TODOS os repos das orgs onde arquivamos coisa.
"""
import os

from huggingface_hub import HfApi

ALVOS = ("B7_champion_prev", "B7_normal_dom")
api = HfApi(token=os.environ["HF_TOKEN"])

repos = []
for org in ("akcit-dephpro", "akcit-h100n1", "akcit-pixel", "juliadollis"):
    for tipo, lista in (("model", api.list_models), ("dataset", api.list_datasets)):
        try:
            for r in lista(author=org):
                repos.append((r.id, tipo))
        except Exception as e:
            print(f"  {org}/{tipo}: {str(e)[:90]}")

print(f"repos a varrer: {len(repos)}\n")
achou = []
for rid, tipo in sorted(set(repos)):
    try:
        fs = api.list_repo_files(rid, repo_type=tipo)
    except Exception:
        continue
    hits = [f for f in fs if any(a in f for a in ALVOS)]
    if hits:
        print(f"ACHEI em {rid} ({tipo}):")
        for f in hits:
            print("   ", f)
        achou.extend(hits)

if not achou:
    print("NAO ACHEI em nenhum repo das orgs varridas.")
    print("Repos varridos:")
    for rid, tipo in sorted(set(repos)):
        print(f"   {rid} ({tipo})")
