#!/usr/bin/env bash
# =============================================================================
# Baixa RESULTADOS do cluster H100 -> Mac (dgx -> Mac).
# RODE NO SEU MAC.
#
#   bash scripts/pull_from_cluster.sh           # baixa outputs/ (checkpoints, samples, infer)
#   bash scripts/pull_from_cluster.sh logs      # baixa também os logs de SLURM
#
# NUNCA usa --delete: só traz/atualiza arquivos, nunca apaga nada do seu Mac.
# =============================================================================
set -euo pipefail

REMOTE_ROOT="dgx-H100-02:/raid/user_juliadollis/projects/genrefocus_deblurnet"
MAC_ROOT="/Users/juliadollis/Projects_Code/repositorio_ref/genrefocus_deblurnet"

pull() {
    local sub="$1"
    echo ">>> baixando ${sub}/ ..."
    mkdir -p "${MAC_ROOT}/${sub}"
    rsync -ah --info=progress2 "${REMOTE_ROOT}/${sub}/" "${MAC_ROOT}/${sub}/"
}

pull outputs
if [[ "${1:-}" == "logs" ]]; then
    pull logs
fi

echo ">>> pronto."
