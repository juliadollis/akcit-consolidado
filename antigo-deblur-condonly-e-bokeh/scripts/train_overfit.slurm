#!/bin/bash
#SBATCH --job-name=deblurnet-smoke-overfit
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# SMOKE (3 steps, poucos min) -> OVERFIT (~400 steps em 4 amostras).
# Objetivo: provar que o pipeline real (FLUX + LoRA + transformer_forward) roda
# sem erro e que a loss_ema CAI no Wandb. NÃO é o treino de 60K do paper.
# Mesma "casca" Singularity do seu exemplo_slurm.sh.
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
    echo "ERRO: preencha HF_TOKEN no topo do script (linha HF_TOKEN=\"...\")."
    exit 1
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERRO: preencha WANDB_API_KEY no topo do script (linha WANDB_API_KEY=\"...\")."
    exit 1
fi

# =============================================================================
# Caminhos — AJUSTE se necessário
# =============================================================================
export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif
export HOST_HF_CACHE_DIR=/raid/user_juliadollis/hf-cache
export HOST_PIP_CACHE_DIR=/raid/user_juliadollis/pip-cache
export HOST_PYTHON_PACKAGES=/raid/user_juliadollis/python-packages

# =============================================================================
# Parâmetros
# =============================================================================
export SMOKE_CONFIG=${SMOKE_CONFIG:-configs/train_smoke.yaml}
export OVERFIT_CONFIG=${OVERFIT_CONFIG:-configs/train_overfit.yaml}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-1}

# =============================================================================
# Variáveis internas do Singularity
# =============================================================================
export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export PYTHONUSERBASE=/workspace/python-packages

# =============================================================================
# Cria diretórios no host
# =============================================================================
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${HOST_HF_CACHE_DIR}"
mkdir -p "${HOST_PIP_CACHE_DIR}"
mkdir -p "${HOST_PYTHON_PACKAGES}"

# =============================================================================
# Validações
# =============================================================================
if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "ERRO: imagem Singularity não encontrada em ${IMAGE_PATH}"
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "ERRO: projeto não encontrado em ${PROJECT_ROOT}"
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ]]; then
    echo "ERRO: Genfocus não encontrado em ${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py"
    echo "Rode antes: bash scripts/setup_genfocus.sh (clona o repo oficial)"
    exit 1
fi

# =============================================================================
# Log de contexto
# =============================================================================
echo "=========================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host:       $(hostname)"
echo "Projeto:    ${PROJECT_ROOT}"
echo "Imagem:     ${IMAGE_PATH}"
echo "Smoke cfg:  ${SMOKE_CONFIG}"
echo "Overfit cfg:${OVERFIT_CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-não definido}"
echo "=========================================="

nvidia-smi --query-gpu=index,name,memory.used,memory.total \
           --format=csv,noheader,nounits || true

# =============================================================================
# Comando executado DENTRO do container Singularity
# =============================================================================
APP_CMD='
set -euo pipefail

# ── Dependências (modo --user, persistem em /workspace/python-packages) ──────
if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    echo "Instalando requirements..."
    python3 -m pip install --upgrade pip --user
    python3 -m pip install --user -r /workspace/projects/genrefocus_deblurnet_paper/requirements.txt
    # peft + accelerate (treino) e wandb (logging) podem não estar no requirements.
    python3 -m pip install --user peft accelerate wandb
else
    echo "Pulando install (INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS})"
fi

cd /workspace/projects/genrefocus_deblurnet_paper

# Genfocus (transformer_forward) precisa ser importável. O pacote `Genfocus`
# fica DENTRO do clone (third_party/Genfocus/Genfocus/), por isso apontamos
# para third_party/Genfocus (a raiz do clone), e não só para third_party.
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

# Sanity check: Genfocus importável (falha cedo com mensagem clara).
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"

# Login no wandb via API key (não-interativo).
python3 -c "import wandb, os; wandb.login(key=os.environ[\"WANDB_API_KEY\"])"

# ── 1) SMOKE TEST (3 steps, valida o pipeline em poucos min) ─────────────────
echo ">>> [1/2] SMOKE TEST <<<"
python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"'

# ── 2) OVERFIT (~400 steps, loss_ema tem que cair no Wandb) ──────────────────
echo ">>> [2/2] OVERFIT <<<"
python3 -m genfocus_train.train deblur --config '"${OVERFIT_CONFIG}"'
'

# =============================================================================
# Executa no container Singularity com GPU
# =============================================================================
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
    --env SMOKE_CONFIG=${SMOKE_CONFIG} \
    --env OVERFIT_CONFIG=${OVERFIT_CONFIG} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
