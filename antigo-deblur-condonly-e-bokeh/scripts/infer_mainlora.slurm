#!/bin/bash
#SBATCH --job-name=deblur-mainlora
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TESTE DE DIAGNÓSTICO: exporta um checkpoint e roda a inferência gerando 2 PNGs
#   - _mainlora.png : LoRA no main+cond (como o 21K foi treinado)
#   - _official.png : LoRA só na cond (inferência oficial → deu lavado)
# Se o _mainlora deblurar, o bug do descasamento de branch está confirmado.
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
export CKPT=${CKPT:-}   # vazio = usa o ÚLTIMO checkpoint (o step_21000 pode já ter sido rotacionado)
export INPUT=${INPUT:-third_party/Genfocus/inference_example/Blurry_example.jpg}
export STEPS=${STEPS:-28}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}

export HF_HOME=/workspace/hf-cache
export PYTHONUSERBASE=/workspace/python-packages

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: sif não encontrado"; exit 1; }
[[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ]] || { echo "ERRO: rode setup_genfocus.sh"; exit 1; }

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) | cfg=${TRAIN_CONFIG} ckpt=${CKPT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail
# NOTA: neste cluster o SLURM aloca GPUs que já estão OCUPADAS (o probe
# --gres=gpu:1 devolveu a GPU 6, que tinha 75GB em uso). Respeitar a alocação
# do SLURM causa OOM. Então, como o container vê todas as GPUs (NVIDIA_VISIBLE_
# DEVICES=all), escolhemos aqui a GPU REALMENTE mais livre. Não é o ideal, é
# contorno de um cluster com SLURM não-isolando GPU.
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t"," -k2 -nr | head -1 | cut -d"," -f1 | tr -d " ")
echo ">>> GPU escolhida (mais livre): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    python3 -m pip install --user -r requirements.txt >/dev/null
    python3 -m pip install --user peft accelerate huggingface_hub >/dev/null
fi

OUT_DIR=/workspace/projects/genrefocus_deblurnet_paper/outputs/infer_mainlora
mkdir -p "${OUT_DIR}"
EXPORT=/tmp/deblur_export.safetensors

echo ">>> EXPORT ${CKPT:-<último>} -> ${EXPORT}"
if [[ -n "${CKPT}" ]]; then
    python3 -m genfocus_train.train export --config "${TRAIN_CONFIG}" --output "${EXPORT}" --checkpoint "${CKPT}"
else
    python3 -m genfocus_train.train export --config "${TRAIN_CONFIG}" --output "${EXPORT}"
fi

case "${INPUT}" in
    /*) INPUT_ABS="${INPUT}" ;;
    *)  INPUT_ABS="/workspace/projects/genrefocus_deblurnet_paper/${INPUT}" ;;
esac
[[ -f "${INPUT_ABS}" ]] || { echo "ERRO: entrada não encontrada: ${INPUT_ABS}"; exit 1; }

PREFIX="${OUT_DIR}/$(basename "${INPUT_ABS%.*}")"
echo ">>> INFER (2 modos) prefix=${PREFIX}"
python3 scripts/infer_mainlora.py \
    --input "${INPUT_ABS}" --out-prefix "${PREFIX}" \
    --weight-dir /tmp --weight-name deblur_export.safetensors --steps ${STEPS}

echo ">>> PRONTO:"
ls -la "${OUT_DIR}"
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
# =all: o container vê todas as GPUs; a escolha da GPU livre é feita DENTRO do
# APP_CMD (necessário porque o SLURM deste cluster aloca GPU já ocupada).
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
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
