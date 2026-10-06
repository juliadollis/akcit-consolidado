#!/bin/bash
#SBATCH --job-name=deblurnet-debug-smoke
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# DEBUG do "CUDA error: illegal memory access" do smoke.
#   - CUDA_LAUNCH_BLOCKING=1  -> kernels síncronos: o stacktrace aponta a
#                               LINHA REAL que estoura (o erro original era
#                               assíncrono e mostrava um ponto enganoso).
#   - config com gradient_checkpointing=false -> sem "recompute" no backward,
#                               trace limpo. Se PASSAR aqui, o bug é o GC.
# Roda SÓ o smoke (3 steps, 256²) — rápido. NÃO faz overfit nem treino.
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
# Caminhos
# =============================================================================
export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif
export DEBUG_CONFIG=${DEBUG_CONFIG:-configs/train_smoke_debug.yaml}
# Reinstalar deps? 0 = pula (já instaladas em rodada anterior). 1 = instala.
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}

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
echo "Job:  ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host: $(hostname)"
echo "Cfg:  ${DEBUG_CONFIG}  (gradient_checkpointing=false)"
echo "CUDA_LAUNCH_BLOCKING=1"
echo "=========================================="

APP_CMD='
set -euo pipefail

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    python3 -m pip install --upgrade pip --user
    python3 -m pip install --user -r /workspace/projects/genrefocus_deblurnet_paper/requirements.txt
    python3 -m pip install --user peft accelerate wandb
else
    # Garante o mínimo p/ rodar caso falte algo.
    python3 -c "import peft, accelerate" 2>/dev/null \
        || python3 -m pip install --user --quiet peft accelerate
fi

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"

echo ">>> SMOKE DEBUG (CUDA_LAUNCH_BLOCKING=1, GC off) <<<"
CUDA_LAUNCH_BLOCKING=1 PYTHONUNBUFFERED=1 \
    python3 -m genfocus_train.train smoke --config '"${DEBUG_CONFIG}"'
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
    --env DEBUG_CONFIG=${DEBUG_CONFIG} \
    --env CUDA_LAUNCH_BLOCKING=1 \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
