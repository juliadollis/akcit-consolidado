#!/bin/bash
#SBATCH --job-name=bokehnet-4gpu
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TREINO DO BOKEHNET (Stage 2) em 4 GPUs, na h100n2, /raid/user_juliadollis.
# Config: train_bokeh_synth_4gpu.yaml (fase 1) / train_bokeh_real_4gpu.yaml (fase 2) (rank 64, batch efetivo 32, 2 condições:
# AIF + defocus_map pré-computado). wandb no projeto genrefocus-bokehnet.
# resume:true + save a cada 250 → re-`sbatch` retoma do último checkpoint.
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

export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}

# Currículo do paper em 2 fases (ver bloco de uso no fim do arquivo):
#   FASE 1 (sintético): TRAIN_CONFIG=configs/train_bokeh_synth_4gpu.yaml  (default)
#   FASE 2 (real):      TRAIN_CONFIG=configs/train_bokeh_real_4gpu.yaml INIT_LORA=outputs/bokehnet_synth_4gpu
export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_bokeh_synth_4gpu.yaml}
export INIT_LORA=${INIT_LORA:-}   # fase 2: output_dir da fase 1 (ou um step_*.pt)
# OUTPUT_DIR: override do output_dir do config. Vazio = usa o do YAML.
# Serve para CONTINUAR um run começado em 2 GPUs agora em 4 GPUs (aponte para o
# MESMO dir do run de 2 GPUs; o resume carrega LoRA+optimizer+scheduler+step, que
# são portáveis entre contagens de GPU pois o batch efetivo é 32 nos dois).
export OUTPUT_DIR=${OUTPUT_DIR:-}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}
# PEFT_PIN: NÃO use por default. O trio ML limpo (transformers 5.0.0.dev0 /
# diffusers 0.37.1 / peft 0.18.1) e o datasets vivem no ~/.local (user-site do
# home), que o Python ja usa por default. NAO sobrescrevemos PYTHONUSERBASE:
# se apontar pra outro dir, o ~/.local sai do sys.path e o peft/diffusers
# "somem" (ModuleNotFoundError). Extras que faltam no ~/.local (wandb,
# python-dotenv): instale UMA vez com  python3 -m pip install --user wandb
# python-dotenv  dentro do container. ATENÇÃO: 'peft==0.13.2' (handoff antigo)
# quebra o diffusers 0.37.1 (ImportError: peft>=0.17.0 is required). PEFT_PIN vazio.
export PEFT_PIN=${PEFT_PIN:-}
export NUM_GPUS=${NUM_GPUS:-4}
export UPLOAD_TO_HF=${UPLOAD_TO_HF:-1}
export HF_REPO_ID=${HF_REPO_ID:-genrefocus-bokehnet-synth-4gpu}

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false

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
    python3 -m pip install --user -r /workspace/projects/genrefocus_deblurnet_paper/requirements.txt
    python3 -m pip install --user peft accelerate wandb huggingface_hub
fi

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

if [[ -n "${PEFT_PIN}" ]]; then
    echo ">>> pin do peft (fix EncoderDecoderCache): ${PEFT_PIN}"
    python3 -m pip install --user "${PEFT_PIN}"
fi
python3 -c "import peft, transformers; print(\"[ok] peft\", peft.__version__, \"transformers\", transformers.__version__)"
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importável\")"
python3 -c "import wandb, os; wandb.login(key=os.environ[\"WANDB_API_KEY\"])"

echo ">>> TREINO BOKEHNET ${NUM_GPUS} GPUs (accelerate) — ${TRAIN_CONFIG} <<<"
INIT_LORA_ARG=""
[[ -n "${INIT_LORA}" ]] && INIT_LORA_ARG="--init-lora ${INIT_LORA}"
PYTHONUNBUFFERED=1 accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port ${MAIN_PORT} \
    -m genfocus_train.train bokeh --config '"${TRAIN_CONFIG}"' ${INIT_LORA_ARG}

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo ">>> UPLOAD do modelo pro HuggingFace (repo: ${HF_REPO_ID}) <<<"
    python3 scripts/hf_upload.py --config '"${TRAIN_CONFIG}"' --stage bokeh --repo-id "${HF_REPO_ID}" \
        || echo "WARN: upload pro HF falhou; modelo salvo em outputs/<run>/bokeh/bokeh.safetensors"
fi
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
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    --env TOKENIZERS_PARALLELISM=false \
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS} \
    --env PEFT_PIN=${PEFT_PIN} \
    --env TRAIN_CONFIG=${TRAIN_CONFIG} \
    --env INIT_LORA=${INIT_LORA} \
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

# =============================================================================
# CURRÍCULO EM 2 FASES (paper §4.1) — como rodar:
#
#   # FASE 1 (sintético, 40K):
#   sbatch scripts/train_bokeh_4gpu.slurm
#
#   # FASE 2 (real, 60K — paper §4.1) — só depois da fase 1 terminar. Inicia o LoRA da fase 1:
#   TRAIN_CONFIG=configs/train_bokeh_real_4gpu.yaml \
#   INIT_LORA=outputs/bokehnet_synth_4gpu \
#   HF_REPO_ID=genrefocus-bokehnet-real-4gpu \
#   sbatch scripts/train_bokeh_4gpu.slurm
#
# Para encadear automático (fase 2 começa quando a 1 terminar OK):
#   JID=$(sbatch --parsable scripts/train_bokeh_4gpu.slurm)
#   TRAIN_CONFIG=configs/train_bokeh_real_4gpu.yaml INIT_LORA=outputs/bokehnet_synth_4gpu \
#     HF_REPO_ID=genrefocus-bokehnet-real-4gpu \
#     sbatch --dependency=afterok:${JID} scripts/train_bokeh_4gpu.slurm
# =============================================================================
