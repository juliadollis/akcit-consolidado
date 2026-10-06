#!/bin/bash
#SBATCH --job-name=deblur-examples
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Gera N exemplos antes->depois [borrada | nosso deblur | ground-truth] a partir
# de um dataset HF, usando a inferência CORRETA (main_adapter="deblurring").
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
export CKPT=${CKPT:-}                          # vazio = último checkpoint (step_60000)
export DATASET=${DATASET:-akcit-pixel/DDPD}
export SPLIT=${SPLIT:-validation}
export N=${N:-5}

export HF_HOME=/workspace/hf-cache
export PYTHONUSERBASE=/workspace/python-packages

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: sif não encontrado"; exit 1; }
[[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ]] || { echo "ERRO: rode setup_genfocus.sh"; exit 1; }

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) | dataset=${DATASET}:${SPLIT} n=${N} ckpt=${CKPT:-<último>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail
# Escolhe a GPU realmente mais livre (o SLURM deste cluster aloca GPU ocupada).
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t"," -k2 -nr | head -1 | cut -d"," -f1 | tr -d " ")
echo ">>> GPU escolhida (mais livre): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

EXPORT=/tmp/deblur_export.safetensors
echo ">>> EXPORT ${CKPT:-<último>} -> ${EXPORT}"
if [[ -n "${CKPT}" ]]; then
    python3 -m genfocus_train.train export --config "${TRAIN_CONFIG}" --output "${EXPORT}" --checkpoint "${CKPT}"
else
    python3 -m genfocus_train.train export --config "${TRAIN_CONFIG}" --output "${EXPORT}"
fi

OUT_DIR=/workspace/projects/genrefocus_deblurnet_paper/outputs/examples
echo ">>> GERANDO ${N} exemplos de ${DATASET}:${SPLIT}"
python3 scripts/make_examples.py \
    --dataset "${DATASET}" --split "${SPLIT}" --n ${N} \
    --weight-dir /tmp --weight-name deblur_export.safetensors \
    --out-dir "${OUT_DIR}"

echo ">>> PRONTO:"
ls -la "${OUT_DIR}"
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
    --env CKPT=${CKPT} \
    --env DATASET=${DATASET} \
    --env SPLIT=${SPLIT} \
    --env N=${N} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
