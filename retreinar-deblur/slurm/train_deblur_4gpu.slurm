#!/bin/bash
#SBATCH --job-name=deblur-paper-4gpu
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# O QUE FAZ
#   Treina a DeblurNet (Stage 1) em 4 GPUs com `configs/train_deblur_paper.yaml`.
#   Batch efetivo 32 = batch_size 1 x grad_accum 8 x 4 GPUs (paper §4.1).
#
# QUANTO DEMORA
#   60.000 steps. O ritmo real depende do no; com gradient_checkpointing ligado
#   a ordem de grandeza medida no projeto foi ~30 s/step em 1 H100. O --time
#   esta em 14 dias de proposito (regra do projeto: limite BEM ALTO; job morto
#   por timeout e retrabalho caro). O job termina sozinho ao chegar em 60k.
#
# O QUE GERA
#   outputs/deblur_paper/deblur/checkpoints/step_*.pt   (5 ultimos + o melhor)
#   outputs/deblur_paper/deblur/deblur.safetensors      (LoRA final, p/ inferencia)
#   outputs/deblur_paper/deblur/metrics.jsonl           (log local)
#   outputs/deblur_paper/effective_config.yaml          (config efetiva do run)
#   HF: snapshot a cada 1000 steps em `genrefocus-deblurnet-v2-paper`
#   wandb: projeto `genrefocus-deblurnet-v2`, run `deblur-paper-60k`
#
# COMO RODAR  (crie o logs/ ANTES do primeiro sbatch: o SLURM nao cria sozinho)
#   ssh dgx-H100-02
#   cd /raid/user_juliadollis/projects/retreinar-deblur
#   mkdir -p logs
#   sbatch slurm/train_deblur_4gpu.slurm
#
# COMO ACOMPANHAR
#   tail -n +1 -f logs/deblur-paper-4gpu-<JOBID>.out logs/deblur-paper-4gpu-<JOBID>.err
#   wandb: https://wandb.ai/<entity>/genrefocus-deblurnet-v2
#   squeue -u $USER -o "%.8i %.10P %.20j %.2t %.12M %R"
#
# RE-SBATCH E SEGURO
#   resume=true no YAML + checkpoint a cada 250 steps. Submeter de novo retoma
#   do ultimo checkpoint e CONTINUA o mesmo grafico do wandb (wandb_resume).
#   Nada e apagado por este script.
#
# VARIAVEIS (todas opcionais)
#   TRAIN_CONFIG=configs/train_deblur_maincond.yaml   # troca de config
#   PROJECT_ROOT=...  BIND_ROOT=...  SIF=...  NUM_GPUS=...
#   RUN_SMOKE=1                                       # smoke antes do treino
# =============================================================================

set -euo pipefail

# --- Localizacao ------------------------------------------------------------
# Default: h100n2, /raid/user_juliadollis (INSTRUCOES_H100.md).
# Na h100n3 a usuaria nao tem pasta propria; use
#   BIND_ROOT=/raid/user_danielpedrozo PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/retreinar-deblur
#   SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
#   e troque --partition para h100n3.
BIND_ROOT="${BIND_ROOT:-/raid/user_juliadollis}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_deblur_paper.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/train_smoke.yaml}"
RUN_SMOKE="${RUN_SMOKE:-0}"

# --nv e o default do template atual do cluster. A h100n2 historicamente usava
# --nvccli; se o container nao enxergar as GPUs, tente NV_FLAG=--nvccli.
NV_FLAG="${NV_FLAG:---nv}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

# --- Segredos (NUNCA hardcode) ----------------------------------------------
# Precedencia: ja exportado > $ENV_FILE > .env na raiz do projeto.
for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. Faca: cp .env.example .env && edite (chmod 600 .env)}"
: "${WANDB_API_KEY:?ERRO: WANDB_API_KEY nao definido. Faca: cp .env.example .env && edite}"
export HF_TOKEN WANDB_API_KEY
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

export HF_HOME="${HF_HOME:-${BIND_ROOT}/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # o load_dataset dos akcit-pixel/* precisa do hub
export TOKENIZERS_PARALLELISM=false
export MAIN_PORT=$((20000 + SLURM_JOB_ID % 20000))

# --- Checagens que falham CEDO ----------------------------------------------
[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }
[ -f "${PROJECT_ROOT}/${TRAIN_CONFIG}" ] || { echo "ERRO: config nao encontrado: ${TRAIN_CONFIG}"; exit 1; }
[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ] || {
  echo "ERRO: third_party/Genfocus ausente. Ele vem junto na pasta; confira o rsync."; exit 1; }

echo "=============================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID:-local})"
echo "No:         $(hostname)"
echo "Projeto:    ${PROJECT_ROOT}"
echo "Config:     ${TRAIN_CONFIG}"
echo "GPUs:       ${NUM_GPUS}  (porta accelerate ${MAIN_PORT})"
echo "HF_HOME:    ${HF_HOME}"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | grep -E "^[0-9]+," || true

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

# PYTHONPATH so com CODIGO DO PROJETO, montado AQUI DENTRO e nunca na chamada
# do singularity (INSTRUCOES_H100.md: passar site-packages por --env PYTHONPATH
# tira o ~/.local do sys.path e some com diffusers/peft).
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "--- versoes ---"
python3 - <<PYV
import torch, transformers, diffusers, peft, accelerate, datasets
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("diffusers   ", diffusers.__version__)
print("peft        ", peft.__version__)
print("accelerate  ", accelerate.__version__)
print("datasets    ", datasets.__version__)
PYV
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importavel\")"
python3 -c "import wandb, os; wandb.login(key=os.environ[\"WANDB_API_KEY\"]); print(\"[ok] wandb\")"
python3 -c "
from huggingface_hub import HfApi
import os
print(\"[ok] HF como\", HfApi(token=os.environ[\"HF_TOKEN\"]).whoami()[\"name\"])
"

if [ "'"${RUN_SMOKE}"'" = "1" ]; then
  echo ">>> SMOKE antes do treino <<<"
  python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"'
fi

echo ">>> TREINO '"${NUM_GPUS}"' GPUs — '"${TRAIN_CONFIG}"' <<<"
accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes '"${NUM_GPUS}"' \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port '"${MAIN_PORT}"' \
  -m genfocus_train.train deblur --config '"${TRAIN_CONFIG}"'
'

srun singularity exec ${NV_FLAG} \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  --env TOKENIZERS_PARALLELISM=false \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=== FIM (rc=$?) ==="
