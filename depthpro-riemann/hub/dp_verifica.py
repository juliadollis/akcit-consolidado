#!/usr/bin/env python3
"""Confere, por tamanho, que cada arquivo do manifesto esta no repo."""
import json, os, sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
ALVOS = [
    ("akcit-dephpro/depthpro-riemann-modelos", "model",
     "/host/_hub_manifesto_modelos.json", "/host/_hub_stage_modelos"),
    ("akcit-dephpro/depthpro-riemann-avaliacoes", "dataset",
     "/host/_hub_manifesto_avaliacoes.json", "/host/_hub_stage_avaliacoes"),
]
for repo, tipo, man, stage in ALVOS:
    print(f"\n===== {repo} ({tipo}) =====")
    try:
        info = api.repo_info(repo, repo_type=tipo, files_metadata=True)
    except Exception as e:
        print("  ERRO:", type(e).__name__, e); continue
    remoto = {s.rfilename: s.size for s in info.siblings}
    print(f"  privado={info.private}  arquivos no repo={len(remoto)}")
    print(f"  total no repo: {sum(v or 0 for v in remoto.values())/2**30:.2f} GiB")
    esperado = {i["dest"]: i["size"] for i in json.load(open(man))}
    # extras do staging que nao estao no manifesto (README, csv consolidado)
    for raiz, _d, arqs in os.walk(stage):
        for a in arqs:
            p = os.path.join(raiz, a)
            rel = os.path.relpath(p, stage)
            if rel.startswith(".cache"):
                continue
            esperado.setdefault(rel, os.path.getsize(p))
    faltando = [d for d in esperado if d not in remoto]
    difere = [(d, esperado[d], remoto[d]) for d in esperado
              if d in remoto and remoto[d] is not None and remoto[d] != esperado[d]]
    extra = [d for d in remoto if d not in esperado and d != ".gitattributes"]
    print(f"  esperados={len(esperado)}  faltando={len(faltando)}  "
          f"tamanho_divergente={len(difere)}  extras_no_repo={len(extra)}")
    for d in faltando[:20]:
        print("    FALTA:", d)
    for d, a, b in difere[:20]:
        print(f"    DIFERE: {d} local={a} remoto={b}")
    for d in extra[:20]:
        print("    EXTRA:", d)
    n_ckpt = {}
    for d in remoto:
        if d.endswith("best.pt"):
            n_ckpt[d.split("/")[0]] = n_ckpt.get(d.split("/")[0], 0) + 1
    if n_ckpt:
        print("  best.pt por categoria:", n_ckpt)
