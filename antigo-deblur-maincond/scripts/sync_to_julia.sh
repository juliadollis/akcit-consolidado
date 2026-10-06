#!/usr/bin/env bash
# =============================================================================
# Sincroniza CÓDIGO do Mac -> /raid do user_danielpedrozo, pasta projects/julia
# (ambiente do treino 2-GPU / h100n3). RODE NO SEU MAC.
#
#   bash scripts/sync_to_julia.sh           # dry-run (mostra o que MUDARIA)
#   bash scripts/sync_to_julia.sh --go      # aplica de verdade
#
# Mesma rede de segurança do sync_to_cluster.sh: --delete só pra manter o código
# idêntico, MAS protege third_party/ outputs/ caches e logs (nunca são apagados).
#
# Host SSH e destino podem ser sobrescritos por env:
#   REMOTE_HOST=outro-alias bash scripts/sync_to_julia.sh --go
# =============================================================================
set -euo pipefail

MAC_DIR="/Users/juliadollis/Projects_Code/repositorio_ref/genrefocus_deblurnet/"
REMOTE_HOST="${REMOTE_HOST:-dgx-H100-02}"
REMOTE_DIR="${REMOTE_DIR:-/raid/user_danielpedrozo/projects/julia/}"
REMOTE="${REMOTE_HOST}:${REMOTE_DIR}"

EXCLUDES=(
    --exclude 'third_party/'
    --exclude 'outputs/'
    --exclude 'hf-cache/'
    --exclude 'pip-cache/'
    --exclude 'python-packages/'
    --exclude 'logs/'
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

echo ">>> Mac -> julia (danielpedrozo raid)  ${MODE:-(REAL)}"
echo "    de:   ${MAC_DIR}"
echo "    para: ${REMOTE}"
echo ""

rsync -ah --delete ${MODE} --itemize-changes "${EXCLUDES[@]}" "${MAC_DIR}" "${REMOTE}"

echo ""
echo ">>> ${NOTE}"
