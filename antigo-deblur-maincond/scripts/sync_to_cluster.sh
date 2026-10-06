#!/usr/bin/env bash
# =============================================================================
# Sincroniza CÓDIGO do Mac -> cluster H100 (Mac -> dgx).
# RODE NO SEU MAC (o cluster não alcança o laptop).
#
#   bash scripts/sync_to_cluster.sh           # dry-run (mostra o que MUDARIA)
#   bash scripts/sync_to_cluster.sh --go      # aplica de verdade
#
# É SEGURO: usa --delete pra manter o código idêntico, MAS protege (via
# --exclude) tudo que é gerado SÓ no cluster — third_party/ (clone do Genfocus),
# outputs/ (checkpoints/modelos), caches e logs NUNCA são apagados.
# =============================================================================
set -euo pipefail

# --- ajuste aqui se mudar de máquina/pasta ---
MAC_DIR="/Users/juliadollis/Projects_Code/repositorio_ref/genrefocus_deblurnet/"
REMOTE="dgx-H100-02:/raid/user_juliadollis/projects/genrefocus_deblurnet/"

# Coisas que vivem SÓ no cluster (ou que não devem ir): protegidas do --delete.
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

rsync -ah --delete ${MODE} --itemize-changes "${EXCLUDES[@]}" "${MAC_DIR}" "${REMOTE}"

echo ""
echo ">>> ${NOTE}"
