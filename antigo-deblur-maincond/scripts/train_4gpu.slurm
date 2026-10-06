#!/bin/bash
#SBATCH --job-name=deblurnet-4gpu
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TREINO DO PAPER em 4 GPUs (DDP-free + all-reduce manual via accelerate),
# na h100n2, /raid/user_juliadollis. Config padrão: train_base_4gpu.yaml
# (grad_accum 8 → batch efetivo 32; gradient_checkpointing OFF = mais rápido).
#
# Tempo limite folgado (14 dias). resume:true + save a cada 250 → re-`sbatch`
# retoma do último checkpoint.
# =============================================================================

# >>> TOKENS: exporte no ambiente ANTES do sbatch (já vêm via --export=ALL se
#     você os exportou no shell). Hardcode local abaixo p/ conveniência. <<<
HF_TOKEN="${HF_TOKEN:?defina HF_TOKEN no ambiente}"
WANDB_API_KEY="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"
export HF_TOKEN WANDB_API_KEY

export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet
export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}

export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_base_4gpu.yaml}
export SMOKE_CONFIG=${SMOKE_CONFIG:-configs/train_smoke.yaml}
export RUN_SMOKE=${RUN_SMOKE:-0}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}
export NUM_GPUS=${NUM_GPUS:-4}
export UPLOAD_TO_HF=${UPLOAD_TO_HF:-1}
export HF_REPO_ID=${HF_REPO_ID:-genrefocus-deblurnet-paper-4gpu}

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUSERBASE=/workspace/python-packages
# Modo ONLINE (0). O travamento antigo era o FLUX INCOMPLETO no cache (resolvido
# por download_flux.slurm). Com o cache completo o modelo carrega rápido, e o
# dataset (akcit-pixel/*) PRECISA do hub/cache — offline=1 quebrava o load_dataset.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false   # silencia o warning de fork dos tokenizers

export MAIN_PORT=$((20000 + SLURM_JOB_ID % 20000))

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
echo "GPUs:       ${NUM_GPUS}   (porta accelerate: ${MAIN_PORT})"
echo "Train cfg:  ${TRAIN_CONFIG}"
echo "Upload HF:  ${UPLOAD_TO_HF}  (repo: ${HF_REPO_ID})"
echo "=========================================="

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    echo "Instalando requirements..."
    python3 -m pip install --upgrade pip --user
    python3 -m pip install --user -r /workspace/projects/genrefocus_deblurnet/requirements.txt
    python3 -m pip install --user peft accelerate wandb huggingface_hub
fi

cd /workspace/projects/genrefocus_deblurnet
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet:/workspace/projects/genrefocus_deblurnet/third_party/Genfocus:${PYTHONPATH:-}"

python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"
python3 -c "import wandb, os; wandb.login(key=os.environ[\"WANDB_API_KEY\"])"

if [[ "${RUN_SMOKE}" == "1" ]]; then
    echo ">>> SMOKE TEST (1 GPU, sanidade) <<<"
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"'
fi

echo ">>> TREINO ${NUM_GPUS} GPUs (accelerate) — ${TRAIN_CONFIG} <<<"
PYTHONUNBUFFERED=1 accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port ${MAIN_PORT} \
    -m genfocus_train.train deblur --config '"${TRAIN_CONFIG}"'

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo ">>> UPLOAD do modelo pro HuggingFace (repo: ${HF_REPO_ID}) <<<"
    python3 scripts/hf_upload.py --config '"${TRAIN_CONFIG}"' --repo-id "${HF_REPO_ID}" \
        || echo "WARN: upload pro HF falhou; modelo salvo em outputs/<run>/deblur/deblur.safetensors"
fi
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    --env TOKENIZERS_PARALLELISM=false \
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env PYTHONUSERBASE=/workspace/python-packages \
    --env PYTHONPATH=/workspace/python-packages/lib/python3.10/site-packages:/workspace/python-packages/lib/python3.11/site-packages \
    --env INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS} \
    --env RUN_SMOKE=${RUN_SMOKE} \
    --env SMOKE_CONFIG=${SMOKE_CONFIG} \
    --env TRAIN_CONFIG=${TRAIN_CONFIG} \
    --env NUM_GPUS=${NUM_GPUS} \
    --env MAIN_PORT=${MAIN_PORT} \
    --env UPLOAD_TO_HF=${UPLOAD_TO_HF} \
    --env HF_REPO_ID=${HF_REPO_ID} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
