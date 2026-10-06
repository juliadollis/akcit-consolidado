#!/bin/bash
#SBATCH --job-name=dump-samples
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Salva 1+ amostra de CADA coluna de imagem (image_pre_deblur, image_blur,
# image_focus) dos datasets HF como PNG, pra você abrir e confirmar
# visualmente qual é a BORRADA (entrada) e qual é a NÍTIDA (alvo AIF).
# NÃO treina nada e NÃO baixa o dataset inteiro (usa streaming).
#
# Os PNGs caem em:
#   /raid/user_juliadollis/projects/genrefocus_deblurnet_paper/sample_dump/
# Para abrir no seu computador, depois do job:
#   rsync -avz <cluster>:/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/sample_dump ./
# =============================================================================

export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper
# --- Segredos (NUNCA hardcode) ----------------------------------------------
# Ordem de precedencia: ja exportado no ambiente > $ENV_FILE > .env na raiz do
# projeto. Crie o .env a partir de .env.example (ele e git-ignored):
#   cp .env.example .env && ${EDITOR:-nano} .env
for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT:-}/.env" "${PROJECT_DIR:-}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env da raiz do projeto (cp .env.example .env)}"
export HF_TOKEN

if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERRO: preencha HF_TOKEN no topo do script (linha HF_TOKEN=\"...\")."
    exit 1
fi

# =============================================================================
# Caminhos — AJUSTE se necessário
# =============================================================================
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

# datasets/numpy/Pillow já devem estar no --user de uma rodada anterior; garante.
python3 -c "import datasets, numpy, PIL" 2>/dev/null \
    || python3 -m pip install --user --quiet datasets numpy Pillow

cd /workspace/projects/genrefocus_deblurnet_paper
PYTHONUNBUFFERED=1 python3 scripts/dump_samples.py
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

echo ""
echo "PNGs salvos em: ${PROJECT_ROOT}/sample_dump/"
echo "Baixe pro seu PC com rsync e abra pra comparar borrada vs nítida."
