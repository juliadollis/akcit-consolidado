"""Sobe os best.pt do reteste riemanniano para juliadollis/depth-riemannian-checkpoints.

37 arquivos, ~1,37 GB cada, ~48 GB no total. Preserva o layout
`<condicao>/seed_<N>/best.pt`, que e o que da procedencia: condicao e semente
sao a identidade da execucao.

Idempotente: `upload_large_folder` retoma e nao reenvia o que ja esta la.
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

RAIZ = Path("/workspace/runs_retreino_hub")
REPO = "juliadollis/depth-riemannian-checkpoints"

tok = os.environ.get("HF_TOKEN")
if not tok:
    sys.exit("[erro] HF_TOKEN ausente")

api = HfApi(token=tok)
arquivos = sorted(RAIZ.rglob("best.pt"))
total = sum(f.stat().st_size for f in arquivos)
print(f"[sobe] {len(arquivos)} arquivos, {total/1e9:.1f} GB -> {REPO}", flush=True)
for f in arquivos:
    print(f"   {f.relative_to(RAIZ)}  {f.stat().st_size/1e9:.2f} GB", flush=True)

api.upload_large_folder(
    folder_path=str(RAIZ),
    repo_id=REPO,
    repo_type="model",
    allow_patterns=["*/seed_*/best.pt"],
    print_report=True,
)
print("[sobe] CONCLUIDO", flush=True)
