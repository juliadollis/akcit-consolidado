#!/bin/bash
#SBATCH --job-name=inspect-datasets
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Inspeciona os datasets HF (DDPD, RealBokeh): splits, colunas, tipos e 1 amostra.
# NÃO treina nada e NÃO baixa as imagens todas (usa streaming p/ a amostra).
# =============================================================================

# =============================================================================
# >>> SEGREDO — PREENCHA AQUI <<<
# =============================================================================
HF_TOKEN="${HF_TOKEN:?defina HF_TOKEN no ambiente}"
export HF_TOKEN

if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERRO: preencha HF_TOKEN no topo do script (linha HF_TOKEN=\"...\")."
    exit 1
fi

# =============================================================================
# Caminhos — AJUSTE se necessário
# =============================================================================
export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet
export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif

# Variáveis internas do Singularity
export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUSERBASE=/workspace/python-packages

mkdir -p "${PROJECT_ROOT}/logs"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "ERRO: imagem Singularity não encontrada em ${IMAGE_PATH}"
    exit 1
fi

echo "=========================================="
echo "Job:  ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host: $(hostname)"
echo "=========================================="

# =============================================================================
# Comando dentro do container
# =============================================================================
APP_CMD='
set -euo pipefail

# datasets já deve estar no --user de uma rodada anterior; garante que existe.
python3 -c "import datasets" 2>/dev/null || python3 -m pip install --user --quiet datasets

cd /workspace/projects/genrefocus_deblurnet
python3 scripts/inspect_datasets.py
'

singularity exec \
    --nvccli \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env PYTHONUSERBASE=/workspace/python-packages \
    --env PYTHONPATH=/workspace/python-packages/lib/python3.10/site-packages:/workspace/python-packages/lib/python3.11/site-packages \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env NVIDIA_VISIBLE_DEVICES=all \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
