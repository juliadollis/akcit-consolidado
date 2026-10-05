#!/bin/bash
#SBATCH --job-name=deblur-smoke
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# O QUE FAZ
#   3 steps de treino no caminho REAL (FLUX + LoRA + transformer_forward dos
#   autores). Valida, nesta ordem: import do Genfocus, carga do FLUX, injecao
#   da LoRA com a trava dos 343 modulos (C1), montagem do batch, forward,
#   backward, optimizer.step mexendo os pesos, e export do .safetensors.
#   Nao treina nada — serve para nao descobrir um ImportError depois de 6h
#   de fila.
#
# QUANTO DEMORA
#   Dominado pelo download/carga do FLUX. Primeira vez (cache frio) pode levar
#   dezenas de minutos; com o cache quente, poucos minutos. --time em 4h por
#   folga.
#
# O QUE GERA
#   outputs/smoke/smoke.safetensors  — e o log com "[smoke] PASSOU".
#   Nao escreve no wandb nem no HF (desligados no train_smoke.yaml).
#
# COMO RODAR (crie o logs/ antes do primeiro sbatch)
#   cd /raid/user_juliadollis/projects/retreinar-deblur && mkdir -p logs
#   sbatch slurm/smoke.slurm
#
# COMO ACOMPANHAR
#   tail -n +1 -f logs/deblur-smoke-<JOBID>.out logs/deblur-smoke-<JOBID>.err
#
# CRITERIO DE SUCESSO
#   O log tem que terminar com "[smoke] PASSOU" e um "Param diff apos 3 steps"
#   maior que 1e-8. Se a trava do C1 disparar, a mensagem diz quantos modulos
#   LoRA foram injetados e quais sao os de topo.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_juliadollis}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/train_smoke.yaml}"
NV_FLAG="${NV_FLAG:---nv}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
# O smoke nao usa wandb nem sobe nada, mas PRECISA do HF_TOKEN: os datasets
# akcit-pixel/* sao privados e o FLUX.1-dev e gated.
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. cp .env.example .env && edite}"
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${BIND_ROOT}/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }
[ -f "${PROJECT_ROOT}/${SMOKE_CONFIG}" ] || { echo "ERRO: config nao encontrado: ${SMOKE_CONFIG}"; exit 1; }

echo "=== smoke @ $(hostname) job=${SLURM_JOB_ID:-local} | config ${SMOKE_CONFIG} ==="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -z "$free_gpu" ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi
gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
gpu_mem=$(echo  "$free_gpu" | cut -d, -f3 | tr -d " ")
echo "[gpu] mais livre: ${gpu_mem} MiB em uso"
if [ "$gpu_mem" -gt 500 ]; then echo "[erro] GPU ocupada (${gpu_mem} MiB). Abortando."; exit 1; fi
export CUDA_VISIBLE_DEVICES=$gpu_uuid

echo "--- versoes ---"
python3 - <<PYV
import torch, transformers, diffusers, peft, accelerate, datasets
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("diffusers   ", diffusers.__version__)
print("peft        ", peft.__version__)
print("accelerate  ", accelerate.__version__)
print("datasets    ", datasets.__version__)
PYV
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importavel\")"

python3 -m genfocus_train.train smoke --config '"${SMOKE_CONFIG}"'
'

srun singularity exec ${NV_FLAG} \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  --env TOKENIZERS_PARALLELISM=false \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=== FIM (rc=$?) ==="
