#!/bin/bash
#SBATCH --job-name=bokeh-prefetch
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# PREFETCH da rota a (ou b/c) para pasta local, VIA SLURM e SEM GPU.
#
# Mesma coisa que rodar scripts/prefetch_bokeh_local.py no login node, mas
# contabilizado pelo escalonador. Use esta versao se a politica do cluster
# desencoraja processo longo no no de login. NAO pede --gres, entao nao ocupa
# GPU e costuma entrar rapido na fila (so CPU+RAM).
#
# O script Python e RESUMIVEL (conta as linhas do metadata.jsonl e da skip),
# entao e seguro:
#   - re-submeter apos preempcao/timeout: continua de onde parou;
#   - migrar do login node para o SLURM no meio do caminho (mate o processo do
#     login antes, para os dois nao escreverem no mesmo metadata.jsonl).
#
# Uso:
#   sbatch scripts/prefetch_bokeh.slurm                      # rota a (default)
#   DATASET=AKCITPixel3/BKXcuVXCmeRvN OUT=/workspace/data-bokeh/rota_b \
#     sbatch scripts/prefetch_bokeh.slurm                    # rota b
#   LIMIT=200 sbatch scripts/prefetch_bokeh.slurm            # so 200 amostras
# =============================================================================

export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper
# --- Segredos (NUNCA hardcode) ----------------------------------------------
for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT:-}/.env" "${PROJECT_DIR:-}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env da raiz do projeto (cp .env.example .env)}"
export HF_TOKEN

export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}
export DATASET=${DATASET:-AKCITPixel3/AfONERuvNmglv}
export OUT=${OUT:-/workspace/data-bokeh/rota_a}
export MIN_SIDE=${MIN_SIDE:-512}
export LIMIT=${LIMIT:-0}

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: sif nao encontrado em ${IMAGE_PATH}"; exit 1; }

echo "=========================================="
echo "Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host:     $(hostname)   [SEM GPU: so CPU+rede+disco]"
echo "Dataset:  ${DATASET}"
echo "Saida:    ${OUT}"
echo "=========================================="

APP_CMD='
set -euo pipefail
cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:${PYTHONPATH:-}"
LIMIT_ARG=""
[[ "${LIMIT}" != "0" ]] && LIMIT_ARG="--limit ${LIMIT}"
PYTHONUNBUFFERED=1 python3 scripts/prefetch_bokeh_local.py \
    --dataset "${DATASET}" --out "${OUT}" --min-side "${MIN_SIDE}" ${LIMIT_ARG}
'

singularity exec \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env DATASET=${DATASET} \
    --env OUT=${OUT} \
    --env MIN_SIDE=${MIN_SIDE} \
    --env LIMIT=${LIMIT} \
    --env TOKENIZERS_PARALLELISM=false \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
