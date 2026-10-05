#!/usr/bin/env python3
"""Baixa o release da rota B — SEM `mask/`, que o dataloader nao usa.

Licao da rota A: paralelizar em 7 downloaders estourou a cota do Hub (2500
requisicoes por 5 min) e matou 6 deles; e `snapshot_download` sem filtro travou
listando 210.809 arquivos. Aqui sao 55.069, dos quais 13.765 sao `mask/` — que
o loader nunca abre. Um downloader so, com `max_workers` moderado, filtrando a
mask fora: 41.304 arquivos.
"""
import os, time
from huggingface_hub import snapshot_download

REPO = "juliadollis/bokehnet-regen-rota-b-nossa"
DEST = "/workspace/releases/rota_b"
t0 = time.time()
p = snapshot_download(
    REPO, repo_type="dataset", local_dir=DEST,
    allow_patterns=["generated/**", "depth/**", "meta/**", "*.jsonl", "*.json", "README.md"],
    ignore_patterns=["mask/**"],
    max_workers=8,
)
n = sum(len(f) for _, _, f in os.walk(DEST))
print(f"PRONTO: {n} arquivos em {(time.time()-t0)/60:.1f} min -> {p}", flush=True)
