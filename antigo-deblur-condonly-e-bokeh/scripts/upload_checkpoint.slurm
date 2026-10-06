#!/bin/bash
#SBATCH --job-name=deblurnet-upload-ckpt
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Exporta o ÚLTIMO checkpoint de um treino e sobe pro HuggingFace — SEM parar o
# treino (roda numa GPU separada; só LÊ o checkpoint do disco).
#
# Uso típico (pegar o estado atual do treino combinado e subir pro HF):
#   TRAIN_CONFIG=configs/train_base.yaml \
#   HF_REPO_ID=genrefocus-deblurnet-condlora \
#   sbatch scripts/upload_checkpoint.slurm
#
# Opcional: escolher um checkpoint específico em vez do último:
#   CKPT=outputs/deblurnet_base/deblur/checkpoints/step_1500.pt ...
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

export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif

export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_base.yaml}
export HF_REPO_ID=${HF_REPO_ID:-genrefocus-deblurnet-condlora}
export CKPT=${CKPT:-}                          # vazio = último checkpoint
export PATH_IN_REPO=${PATH_IN_REPO:-deblur.safetensors}

export HF_HOME=/workspace/hf-cache
export PYTHONUSERBASE=/workspace/python-packages

mkdir -p "${PROJECT_ROOT}/logs"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "ERRO: imagem Singularity não encontrada em ${IMAGE_PATH}"; exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ]]; then
    echo "ERRO: Genfocus não encontrado. Rode antes: bash scripts/setup_genfocus.sh"; exit 1
fi

echo "=========================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Train cfg:  ${TRAIN_CONFIG}"
echo "Checkpoint: ${CKPT:-<último>}"
echo "HF repo:    ${HF_REPO_ID}  (arquivo: ${PATH_IN_REPO})"
echo "=========================================="

APP_CMD='
set -euo pipefail
cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

OUT=/tmp/deblur_export.safetensors

echo ">>> EXPORTANDO checkpoint -> ${OUT}"
if [[ -n "${CKPT}" ]]; then
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train export \
        --config "${TRAIN_CONFIG}" --output "${OUT}" --checkpoint "${CKPT}"
else
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train export \
        --config "${TRAIN_CONFIG}" --output "${OUT}"
fi

echo ">>> UPLOAD pro HF (${HF_REPO_ID})"
PYTHONUNBUFFERED=1 python3 scripts/hf_upload.py \
    --file "${OUT}" --repo-id "${HF_REPO_ID}" --path-in-repo "${PATH_IN_REPO}"

echo ">>> PRONTO."
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env PYTHONUSERBASE=/workspace/python-packages \
    --env PYTHONPATH=/workspace/python-packages/lib/python3.10/site-packages:/workspace/python-packages/lib/python3.11/site-packages \
    --env TRAIN_CONFIG=${TRAIN_CONFIG} \
    --env HF_REPO_ID=${HF_REPO_ID} \
    --env CKPT=${CKPT} \
    --env PATH_IN_REPO=${PATH_IN_REPO} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
