#!/usr/bin/env python3
"""
Downloader robusto do depth_pro.pt via API do huggingface_hub (sem depender de
wget/curl nem do CLI, que muda de nome entre versoes). Usado como fallback do
download_weights.py do Francisco (que usa wget, ausente no nosso container).

Coloca o arquivo em <DL_OUT>/checkpoints/depth_pro.pt.
"""
import os
from huggingface_hub import hf_hub_download

repo = os.environ.get("DL_REPO", "nycu-cplab/Genfocus-Model")
fname = os.environ.get("DL_FILE", "checkpoints/depth_pro.pt")
out = os.environ.get("DL_OUT", "/workspace/models")
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

p = hf_hub_download(repo_id=repo, filename=fname, local_dir=out, token=tok)
print("[ok] pesos baixados:", p)
