#!/bin/bash
#SBATCH --job-name=bokehnet-smoke
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# SMOKE TEST DO BOKEHNET (Stage 2) — 3 steps no caminho REAL de 2 condições
# (FLUX + LoRA + transformer_forward com AIF + defocus map), ANTES de gastar
# 40K/60K steps. Valida group_mask (diag entre condições), dataloader de bokeh
# e o forward de 2 conds. Config: train_bokeh_smoke.yaml (rota a, 8 samples,
# 256², rank 16). 1 GPU, sem accelerate.
#
# NOTA: a 1ª execução baixa a rota a (= dado da FASE 1) pro cache — não é
# desperdício, a fase 1 usa o mesmo dado. Por isso --time=02:00:00 (folga p/
# download); o compute em si (3 steps 256²) é rápido.
#
# GPU: como neste cluster o SLURM às vezes aloca GPU ocupada, escolhemos DENTRO
# do container a GPU realmente mais livre (mesmo contorno dos scripts de infer).
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
export SMOKE_CONFIG=${SMOKE_CONFIG:-configs/train_bokeh_smoke.yaml}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}
# PEFT_PIN: NÃO use por default. O trio ML limpo (transformers 5.0.0.dev0 /
# diffusers 0.37.1 / peft 0.18.1) e o datasets vivem no ~/.local (user-site do
# home), que o Python ja usa por default. NAO sobrescrevemos PYTHONUSERBASE:
# se apontar pra outro dir, o ~/.local sai do sys.path e o peft/diffusers
# "somem" (ModuleNotFoundError: No module named 'peft').
# Extras que faltam no ~/.local (wandb, python-dotenv): instale UMA vez com
#   singularity exec --bind /raid/user_juliadollis:/workspace \
#     /raid/user_juliadollis/images/transformers-pytorch-gpu.sif \
#     python3 -m pip install --user wandb python-dotenv
# ATENÇÃO: 'peft==0.13.2' (handoff antigo) quebra o diffusers 0.37.1
# (ImportError: peft>=0.17.0 is required). Deixe PEFT_PIN vazio.
export PEFT_PIN=${PEFT_PIN:-}

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: sif não encontrado em ${IMAGE_PATH}"; exit 1; }
[[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ]] || { echo "ERRO: rode setup_genfocus.sh"; exit 1; }

echo "=========================================="
echo "Job:  ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host: $(hostname)"
echo "Cfg:  ${SMOKE_CONFIG}  (smoke bokeh, 2 condições)"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail
# GPU realmente mais livre (SLURM deste cluster às vezes aloca GPU ocupada).
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t"," -k2 -nr | head -1 | cut -d"," -f1 | tr -d " ")
echo ">>> GPU escolhida (mais livre): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    python3 -m pip install --upgrade pip --user
    python3 -m pip install --user -r requirements.txt
    python3 -m pip install --user peft accelerate wandb huggingface_hub
fi
if [[ -n "${PEFT_PIN}" ]]; then
    echo ">>> pin do peft (fix EncoderDecoderCache): ${PEFT_PIN}"
    python3 -m pip install --user "${PEFT_PIN}"
fi
python3 -c "import peft, transformers; print(\"[ok] peft\", peft.__version__, \"transformers\", transformers.__version__)"
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"

echo ">>> SMOKE BOKEHNET (3 steps, 2 condições) — ${SMOKE_CONFIG} <<<"
PYTHONUNBUFFERED=1 python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"' --stage bokeh
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
# GPUS: quais GPUs o container enxerga. Default all. Para pular GPU com defeito
# (ex.: nvml error na H100n3), passe GPUS=0,1,2,3,5,6,7 sbatch ...
export NVIDIA_VISIBLE_DEVICES=${GPUS:-all}

# NV_FLAG: mecanismo de GPU do singularity. Default --nvccli. Se o no tiver GPU
# com defeito e o nvidia-container-cli falhar (nvml error), use NV_FLAG=--nv.
export NV_FLAG=${NV_FLAG:---nvccli}
singularity exec \
    ${NV_FLAG} \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS} \
    --env PEFT_PIN=${PEFT_PIN} \
    --env SMOKE_CONFIG=${SMOKE_CONFIG} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
