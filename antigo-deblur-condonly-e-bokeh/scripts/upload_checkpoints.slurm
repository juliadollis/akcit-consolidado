#!/bin/bash
#SBATCH --job-name=deblur-uploadckpts
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Sobe os checkpoints que JÁ estão no disco (step_*.pt) pro HF, um repo POR-STEP
# (<base>-step<N>). Carrega o FLUX UMA vez e itera os checkpoints.
# Roda 1 GPU na h100n2 (mesma raid do treino), numa GPU VAZIA.
#
# Ex.: só alguns steps ->  ONLY=5000,10000,13500 sbatch scripts/upload_checkpoints.slurm
#      1 a cada 4       ->  EVERY=4              sbatch scripts/upload_checkpoints.slurm
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

export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}

export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_base_4gpu.yaml}
export REPO_BASE=${REPO_BASE:-genrefocus-deblurnet-condlora}   # vira <REPO_BASE>-step<N>
export EVERY=${EVERY:-1}                                       # sobe 1 a cada N checkpoints
export ONLY=${ONLY:-}                                          # csv de steps, ex.: 5000,10000
export CKPTS_DIR=${CKPTS_DIR:-}                                # default: do config

export HF_HOME=/workspace/hf-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTHONUSERBASE=/workspace/python-packages

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: SIF não encontrado em ${IMAGE_PATH}"; exit 1; }

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) | $(hostname) | repo_base=${REPO_BASE} every=${EVERY} only=${ONLY:-<todos>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail
# GPU VAZIA (sem NINGUÉM); se a mais livre tiver >500 MiB em uso, ABORTA.
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t"," -k2 -n | head -1)
GPU_IDX=$(echo "${GPU_INFO}" | cut -d"," -f1 | tr -d " ")
GPU_USED=$(echo "${GPU_INFO}" | cut -d"," -f2 | tr -d " ")
if [[ "${GPU_USED}" -gt 500 ]]; then
    echo "ERRO: nenhuma GPU VAZIA (a mais livre, GPU ${GPU_IDX}, tem ${GPU_USED} MiB). Abortando."
    exit 1
fi
export CUDA_VISIBLE_DEVICES=${GPU_IDX}
echo ">>> GPU VAZIA: CUDA_VISIBLE_DEVICES=${GPU_IDX} (${GPU_USED} MiB em uso)"

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"

EXTRA=""
[[ -n "${ONLY}" ]]      && EXTRA="${EXTRA} --only ${ONLY}"
[[ "${EVERY}" != "1" ]] && EXTRA="${EXTRA} --every ${EVERY}"
[[ -n "${CKPTS_DIR}" ]] && EXTRA="${EXTRA} --ckpts-dir ${CKPTS_DIR}"

PYTHONUNBUFFERED=1 python3 -m scripts.upload_all_checkpoints \
    --config "${TRAIN_CONFIG}" --repo-base "${REPO_BASE}" ${EXTRA}
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    --env PYTHONUSERBASE=/workspace/python-packages \
    --env PYTHONPATH=/workspace/python-packages/lib/python3.10/site-packages:/workspace/python-packages/lib/python3.11/site-packages \
    --env TRAIN_CONFIG=${TRAIN_CONFIG} \
    --env REPO_BASE=${REPO_BASE} \
    --env EVERY=${EVERY} \
    --env ONLY=${ONLY} \
    --env CKPTS_DIR=${CKPTS_DIR} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
