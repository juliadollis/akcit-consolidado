#!/usr/bin/env python3
"""Sobe os pesos para akcit-dephpro/depthpro-riemann-modelos.
upload_large_folder: resumivel, com retry, aguenta os 70 GiB."""
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "akcit-dephpro/depthpro-riemann-modelos"
info = api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
print("repo:", getattr(info, "url", info), flush=True)
api.upload_large_folder(
    folder_path="/host/_hub_stage_modelos",
    repo_id=REPO, repo_type="model",
    num_workers=8, print_report=True, print_report_every=60,
)
fs = api.list_repo_files(REPO, repo_type="model")
print("FIM. arquivos no repo:", len(fs), flush=True)
