#!/bin/bash
#SBATCH --job-name=deblurnet-2gpu
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TREINO DO PAPER em 2 GPUs (DDP via accelerate), na h100n2, no seu próprio
# /raid/user_juliadollis (mesmo ambiente dos runs single-GPU — código, SIF,
# Genfocus e libs já estão lá).
#
# Tempo limite BEM folgado (14 dias). resume:true + save a cada 250 → se cair,
# é só re-`sbatch` que retoma do último checkpoint.
#
# Config padrão: configs/train_base_2gpu.yaml (grad_accum=16 → batch efetivo 32
# com 2 GPUs). Lança 1 task que abre 2 processos (1 por GPU) via accelerate.
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
: "${WANDB_API_KEY:?defina WANDB_API_KEY no .env da raiz do projeto}"
export HF_TOKEN WANDB_API_KEY

# =============================================================================
export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}

export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_base_2gpu.yaml}
export SMOKE_CONFIG=${SMOKE_CONFIG:-configs/train_smoke.yaml}
export RUN_SMOKE=${RUN_SMOKE:-0}                 # pipeline já validado; pula smoke
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}   # libs já instaladas no /raid
export NUM_GPUS=${NUM_GPUS:-2}
export UPLOAD_TO_HF=${UPLOAD_TO_HF:-1}
export HF_REPO_ID=${HF_REPO_ID:-genrefocus-deblurnet-condlora-2gpu}

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUSERBASE=/workspace/python-packages
# Modo ONLINE (0). O travamento antigo no "Fetching" era o FLUX INCOMPLETO no
# cache (faltava um shard do T5) — resolvido pelo scripts/download_flux.slurm.
# Com o cache completo, o load do modelo é rápido (só verifica), e o dataset
# (akcit-pixel/*) PRECISA do hub/cache — offline=1 quebrava o load_dataset.
# Se o load do modelo travar de novo, rode download_flux.slurm (NÃO ligue offline).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false   # silencia o warning de fork dos tokenizers

# porta única do accelerate por job (evita conflito se 2 jobs no mesmo nó)
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
echo "Project:    ${PROJECT_ROOT}"
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
    python3 -m pip install --user -r /workspace/projects/genrefocus_deblurnet_paper/requirements.txt
    python3 -m pip install --user peft accelerate wandb huggingface_hub
fi

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"
python3 -c "import wandb, os; wandb.login(key=os.environ[\"WANDB_API_KEY\"])"

# smoke roda em 1 processo (NÃO sob accelerate) — só sanidade
if [[ "${RUN_SMOKE}" == "1" ]]; then
    echo ">>> SMOKE TEST (1 GPU, sanidade) <<<"
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"'
fi

echo ">>> TREINO 2 GPUs (accelerate, ${NUM_GPUS} processos) — ${TRAIN_CONFIG} <<<"
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
