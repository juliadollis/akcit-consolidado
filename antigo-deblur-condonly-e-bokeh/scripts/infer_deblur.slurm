#!/bin/bash
#SBATCH --job-name=deblurnet-infer
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# INFERÊNCIA da DeblurNet a partir de um checkpoint de TREINO (step_*.pt).
# Roda numa GPU SEPARADA — pode rodar com o treino ainda em andamento.
#
#   1) exporta o checkpoint -> third_party/Genfocus/deblurNet.safetensors
#   2) roda o Inference_deblurNet.py OFICIAL (mesma pipeline do paper)
#   3) salva o PNG em outputs/deblurnet_ddpd/infer/
#
# Por padrão usa o ÚLTIMO checkpoint salvo e a imagem de exemplo do repo.
# Pra escolher outro checkpoint/imagem:
#   CKPT=outputs/deblurnet_ddpd/deblur/checkpoints/step_4000.pt \
#   INPUT=/workspace/.../minha_blur.jpg \
#   sbatch scripts/infer_deblur.slurm
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

# =============================================================================
export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif

export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_ddpd.yaml}   # localiza os checkpoints
export CKPT=${CKPT:-}                                          # vazio = último checkpoint
export INPUT=${INPUT:-third_party/Genfocus/inference_example/Blurry_example.jpg}
export STEPS=${STEPS:-28}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-1}

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
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
echo "Host:       $(hostname)"
echo "Train cfg:  ${TRAIN_CONFIG}"
echo "Checkpoint: ${CKPT:-<último>}"
echo "Input:      ${INPUT}"
echo "Steps:      ${STEPS}"
echo "=========================================="

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    echo "Instalando requirements..."
    python3 -m pip install --upgrade pip --user
    python3 -m pip install --user -r /workspace/projects/genrefocus_deblurnet_paper/requirements.txt
    python3 -m pip install --user peft accelerate
fi

PROJ=/workspace/projects/genrefocus_deblurnet_paper
cd "${PROJ}"
export PYTHONPATH="${PROJ}:${PROJ}/third_party/Genfocus:${PYTHONPATH:-}"

OUT_DIR="${PROJ}/outputs/deblurnet_ddpd/infer"
mkdir -p "${OUT_DIR}"

# ----- 1) exporta o checkpoint -> deblurNet.safetensors (na pasta do Genfocus,
#          que é onde o Inference_deblurNet.py procura por padrão) -----
EXPORT_TARGET="${PROJ}/third_party/Genfocus/deblurNet.safetensors"
echo ">>> EXPORTANDO checkpoint -> ${EXPORT_TARGET} <<<"
if [[ -n "${CKPT}" ]]; then
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train export \
        --config "${TRAIN_CONFIG}" --output "${EXPORT_TARGET}" --checkpoint "${CKPT}"
else
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train export \
        --config "${TRAIN_CONFIG}" --output "${EXPORT_TARGET}"
fi

# ----- 2) inferência com a pipeline OFICIAL do paper -----
# SEMPRE resolve pra caminho ABSOLUTO (logo abaixo damos cd pra outra pasta,
# então um caminho relativo apontaria pro lugar errado).
case "${INPUT}" in
    /*) INPUT_ABS="${INPUT}" ;;            # já é absoluto
    *)  INPUT_ABS="${PROJ}/${INPUT}" ;;    # relativo à raiz do projeto
esac
if [[ ! -f "${INPUT_ABS}" ]]; then
    echo "ERRO: imagem de entrada não encontrada: ${INPUT_ABS}"; exit 1
fi
OUT_PNG="${OUT_DIR}/deblurred_$(basename "${INPUT_ABS%.*}").png"
echo ">>> INFERÊNCIA (Inference_deblurNet.py, steps=${STEPS}) input=${INPUT_ABS} <<<"
cd "${PROJ}/third_party/Genfocus"
PYTHONUNBUFFERED=1 python3 Inference_deblurNet.py \
    --input "${INPUT_ABS}" \
    --output "${OUT_PNG}" \
    --steps "${STEPS}"

echo ">>> PRONTO. Saída: ${OUT_PNG}"
ls -la "${OUT_PNG}"
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env PYTHONUSERBASE=/workspace/python-packages \
    --env PYTHONPATH=/workspace/python-packages/lib/python3.10/site-packages:/workspace/python-packages/lib/python3.11/site-packages \
    --env INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS} \
    --env TRAIN_CONFIG=${TRAIN_CONFIG} \
    --env CKPT=${CKPT} \
    --env INPUT=${INPUT} \
    --env STEPS=${STEPS} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
