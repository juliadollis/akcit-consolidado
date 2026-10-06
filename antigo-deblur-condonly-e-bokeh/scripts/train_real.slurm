#!/bin/bash
#SBATCH --job-name=deblurnet-train
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TREINO REAL da DeblurNet (Stage 1). Mesma "casca" Singularity do overfit (que
# já rodou ok). Roda smoke (sanidade) -> treino real.
#
# Config padrão: configs/train_ddpd.yaml (SÓ DDPD = etapa 1 do plano).
# Para a etapa 2 (DDPD + subset RealBokeh), rode com:
#   TRAIN_CONFIG=configs/train_base.yaml sbatch scripts/train_real.slurm
#
# RESUME: a config tem runtime.resume=true e salva checkpoint a cada 1000 steps.
# Se o job bater no limite de tempo, é só dar `sbatch` de novo que ele RETOMA
# do último checkpoint (outputs/<run>/deblur/checkpoints/).
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

if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERRO: preencha HF_TOKEN no topo do script."; exit 1
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERRO: preencha WANDB_API_KEY no topo do script."; exit 1
fi

# =============================================================================
# Caminhos e parâmetros
# =============================================================================
export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif

export SMOKE_CONFIG=${SMOKE_CONFIG:-configs/train_smoke.yaml}
export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_ddpd.yaml}   # etapa 1: só DDPD
export RUN_SMOKE=${RUN_SMOKE:-1}                               # 1 = roda smoke antes
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-1}
export UPLOAD_TO_HF=${UPLOAD_TO_HF:-0}                         # 1 = sobe o modelo pro HF no fim do treino
export HF_REPO_ID=${HF_REPO_ID:-genrefocus-deblurnet}         # repo destino (sem '/' = seu namespace HF)

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
echo "Smoke cfg:  ${SMOKE_CONFIG}  (RUN_SMOKE=${RUN_SMOKE})"
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

if [[ "${RUN_SMOKE}" == "1" ]]; then
    echo ">>> SMOKE TEST (sanidade) <<<"
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"'
fi

echo ">>> TREINO REAL (${TRAIN_CONFIG}) <<<"
PYTHONUNBUFFERED=1 python3 -m genfocus_train.train deblur --config '"${TRAIN_CONFIG}"'

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo ">>> UPLOAD do modelo pro HuggingFace (repo: ${HF_REPO_ID}) <<<"
    # resolve outputs/<run>/deblur/deblur.safetensors a partir do config e sobe.
    # NÃO-FATAL: se falhar, o modelo continua salvo localmente em outputs/.
    python3 scripts/hf_upload.py --config "${TRAIN_CONFIG}" --repo-id "${HF_REPO_ID}" \
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
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env PYTHONUSERBASE=/workspace/python-packages \
    --env PYTHONPATH=/workspace/python-packages/lib/python3.10/site-packages:/workspace/python-packages/lib/python3.11/site-packages \
    --env INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS} \
    --env RUN_SMOKE=${RUN_SMOKE} \
    --env SMOKE_CONFIG=${SMOKE_CONFIG} \
    --env TRAIN_CONFIG=${TRAIN_CONFIG} \
    --env UPLOAD_TO_HF=${UPLOAD_TO_HF} \
    --env HF_REPO_ID=${HF_REPO_ID} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
