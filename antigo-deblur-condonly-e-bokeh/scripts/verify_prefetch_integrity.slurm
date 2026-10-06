#!/bin/bash
#SBATCH --job-name=bokeh-verify
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# VERIFICACAO DE INTEGRIDADE da pasta do prefetch, VIA SLURM e SEM GPU.
#
# Motivo: o job 29267 (fase 1 do BokehNet) morreu no step 3380 com
# "OSError: image file is truncated" num png de rota_a/aif/. A causa foi a
# corrida do prefetch (processo do login node + job SLURM 29258 escrevendo a
# MESMA pasta em paralelo). Este job da a lista EXATA dos arquivos ruins.
#
# NAO pede --gres, entao nao ocupa GPU e entra rapido na fila. SO LEITURA:
# o script nunca apaga nem reescreve imagem, so escreve o integrity_report.json.
#
# Uso:
#   sbatch scripts/verify_prefetch_integrity.slurm                   # rota a, quick
#   FULL=1 sbatch scripts/verify_prefetch_integrity.slurm            # decodifica tudo
#   ROOT=/workspace/data-bokeh/rota_b sbatch scripts/verify_prefetch_integrity.slurm
# =============================================================================

export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper

export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}
export ROOT=${ROOT:-/workspace/data-bokeh/rota_a}
export FULL=${FULL:-0}
export WORKERS=${WORKERS:-8}

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: sif nao encontrado em ${IMAGE_PATH}"; exit 1; }

echo "=========================================="
echo "Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host:     $(hostname)   [SEM GPU: so CPU+disco]"
echo "Pasta:    ${ROOT}"
echo "Modo:     $([[ "${FULL}" == "1" ]] && echo 'FULL (decodifica tudo)' || echo 'QUICK (assinatura + IEND)')"
echo "=========================================="

APP_CMD='
set -euo pipefail
cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:${PYTHONPATH:-}"
FULL_ARG=""
[[ "${FULL}" == "1" ]] && FULL_ARG="--full"
PYTHONUNBUFFERED=1 python3 scripts/verify_prefetch_integrity.py \
    --root "${ROOT}" --workers "${WORKERS}" ${FULL_ARG}
'

singularity exec \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env ROOT=${ROOT} \
    --env FULL=${FULL} \
    --env WORKERS=${WORKERS} \
    --env TOKENIZERS_PARALLELISM=false \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
