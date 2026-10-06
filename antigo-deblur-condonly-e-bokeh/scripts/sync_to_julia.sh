#!/usr/bin/env bash
# =============================================================================
# Sincroniza CÓDIGO do Mac -> /raid do user_danielpedrozo, pasta projects/julia
# (ambiente do treino 2-GPU / h100n3, host dgx-H100-03). RODE NO SEU MAC.
#
#   bash scripts/sync_to_julia.sh           # dry-run (mostra o que MUDARIA)
#   bash scripts/sync_to_julia.sh --go      # aplica de verdade
#
# ATENÇÃO: a pasta julia/ e COMPARTILHADA (cemig RAG, eval, etc.). Este sync
# NAO usa --delete de proposito: ele so ADICIONA/ATUALIZA os arquivos do bokeh,
# nunca apaga o que e de outros projetos. (Um --delete aqui ja apagou o cemig
# uma vez.) Se um dia precisar limpar so o bokeh, faca a mao, com cuidado.
#
# Host SSH e destino podem ser sobrescritos por env:
#   REMOTE_HOST=outro-alias bash scripts/sync_to_julia.sh --go
# =============================================================================
set -euo pipefail

MAC_DIR="/Users/juliadollis/Projects_Code/repositorio_ref/genrefocus_deblurnet_paper/"
REMOTE_HOST="${REMOTE_HOST:-dgx-H100-03}"
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

rsync -ah ${MODE} --itemize-changes "${EXCLUDES[@]}" "${MAC_DIR}" "${REMOTE}"

echo ""
echo ">>> ${NOTE}"
