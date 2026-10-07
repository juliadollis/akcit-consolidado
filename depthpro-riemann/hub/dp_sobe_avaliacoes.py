#!/usr/bin/env python3
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "akcit-dephpro/depthpro-riemann-avaliacoes"
info = api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
print("repo:", info.url if hasattr(info, "url") else info)
api.upload_folder(
    folder_path="/host/_hub_stage_avaliacoes",
    repo_id=REPO, repo_type="dataset",
    commit_message="Avaliacoes depth-riemannian: brutos por modelo/mesa + avaliacoes.csv + README",
)
fs = api.list_repo_files(REPO, repo_type="dataset")
print("arquivos no repo:", len(fs))
