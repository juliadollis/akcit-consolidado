#!/usr/bin/env bash
# =============================================================================
# Sincroniza CÓDIGO do Mac -> cluster H100 (Mac -> dgx).
# RODE NO SEU MAC (o cluster não alcança o laptop).
#
#   bash scripts/sync_to_cluster.sh           # dry-run (mostra o que MUDARIA)
#   bash scripts/sync_to_cluster.sh --go      # aplica de verdade
#
# NAO usa --delete: so ADICIONA/ATUALIZA arquivos no destino, nunca apaga nada
# no cluster. (Excludes abaixo evitam ate ENVIAR third_party/outputs/caches.)
# =============================================================================
set -euo pipefail

# --- ajuste aqui se mudar de máquina/pasta ---
MAC_DIR="/Users/juliadollis/Projects_Code/repositorio_ref/genrefocus_deblurnet_paper/"
# Destino default = H100-02 (SLURM + singularity). Sobrescreva com REMOTE=... para
# outro no, ex.: a H100-01, que NAO tem SLURM e roda por docker:
#   REMOTE=dgx-H100-01:/raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper/ \
#     bash scripts/sync_to_cluster.sh --go
REMOTE="${REMOTE:-dgx-H100-02:/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/}"

# Coisas que vivem SÓ no cluster (ou que não devem ir): nunca enviadas.
EXCLUDES=(
    --exclude 'third_party/'        # clone do Genfocus (feito no cluster)
    --exclude 'outputs/'            # checkpoints, .safetensors, samples
    --exclude 'hf-cache/'           # cache HuggingFace
    --exclude 'pip-cache/'
    --exclude 'python-packages/'    # pip --user do container
    --exclude 'logs/'               # logs de SLURM
    --exclude 'wandb/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.pytest_cache/'      # cache do pytest (gerado ao rodar testes local)
    --exclude '.venv/'              # virtualenvs locais
    --exclude 'venv/'
    --exclude '.git/'
    --exclude '.DS_Store'
    --exclude 'sample_dump/'
)

MODE="--dry-run"
NOTE="(DRY-RUN — nada foi alterado. Rode com --go pra aplicar.)"
if [[ "${1:-}" == "--go" ]]; then
    MODE=""
    NOTE="(APLICADO.)"
fi

echo ">>> Mac -> cluster  ${MODE:-(REAL)}"
echo "    de:   ${MAC_DIR}"
echo "    para: ${REMOTE}"
echo ""

rsync -ah ${MODE} --itemize-changes "${EXCLUDES[@]}" "${MAC_DIR}" "${REMOTE}"

echo ""
echo ">>> ${NOTE}"
