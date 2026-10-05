#!/bin/bash
#SBATCH --job-name=deblur-prefetch
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=20-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# =============================================================================
# Pré-aquece o cache do treino da DeblurNet — SEM GPU.
# =============================================================================
# POR QUE EXISTE: os datasets (DDPD 9,6 GB + RealBokeh 46,9 GB) não estão no
# cache deste nó, e o filtro de nitidez do RealBokeh decodifica 20.495 imagens.
# Fazer isso dentro do job de 4 GPUs seguraria as GPUs ociosas por ~1-2 h e
# ainda arriscaria o barrier do NCCL com 4 ranks medindo em paralelo.
#
# SEM `--gres` de propósito: o INSTRUCOES_H100.md pede que trabalho pesado de
# CPU vá para o SLURM sem GPU. E baixar aqui respeita a regra "nunca baixe
# coisa fora do SLURM".
#
# O QUE GERA: entradas em $HF_HOME e o cache de índices do filtro.
# NÃO apaga nada, não toca em job de ninguém.
#
# ENCADEAMENTO: o treino é submetido com --dependency=afterok:<este_job>.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_danielpedrozo}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/julia/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_deblur_paper.yaml}"

cd "${PROJECT_ROOT}"
mkdir -p logs

for _env in "${ENV_FILE:-}" "${PROJECT_ROOT}/.env" "${BIND_ROOT}/projects/julia/.env"; do
    if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; echo "[env] carregado: ${_env}"; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN ausente}"

export HF_HOME="${HF_HOME:-${BIND_ROOT}/projects/julia/hf-cache}"

echo "=============================================================="
echo " prefetch | job=${SLURM_JOB_ID:-none} | $(hostname) | $(date)"
echo " HF_HOME: ${HF_HOME}"
echo " Cota antes:"; quota -s 2>/dev/null | tail -2 || true
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
python3 scripts/prefetch_datasets.py --config '"${TRAIN_CONFIG}"'
'

srun singularity exec \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "--- cota depois ---"; quota -s 2>/dev/null | tail -2 || true
echo "=== FIM (rc=$?) ==="
