#!/bin/bash
#SBATCH --job-name=ab15k
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --time=20-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# =============================================================================
# A/B de 15.000 steps na dgx-H100-02 — por que perdemos no DPDD?
# =============================================================================
# USO:  CFG=configs/ab15k_controle.yaml sbatch --job-name=ab15k-controle slurm/train_ab15k_n2.slurm
#       CFG=configs/ab15k_scene.yaml    sbatch --job-name=ab15k-scene    slurm/train_ab15k_n2.slurm
#       CFG=configs/ab15k_rank64.yaml   sbatch --job-name=ab15k-rank64   slurm/train_ab15k_n2.slurm
#
# QUANTO DEMORA: ~8,5 s/step extrapolado de medicoes em 4 GPUs. 15.000 steps
# ~ 1,5 dia. --time alto assim mesmo (regra do INSTRUCOES_H100.md).
#
# ATENCAO AOS CAMINHOS: o /raid e LOCAL de cada no. Aqui e a pasta da usuaria
# (/raid/user_juliadollis), NAO a do Daniel — aquela e da h100n3.
#
# O FLUX e os datasets NAO estao em cache neste no (o hf-cache foi limpo para
# abrir espaco). O primeiro job baixa ~110 GB; os seguintes reusam. Por isso
# submeta o CONTROLE primeiro e espere ele passar do step 1 antes dos outros.
#
# NUNCA: cancelar job, apagar nada, sobrescrever PYTHONUSERBASE, passar
# --env PYTHONPATH ao singularity, usar PEFT_PIN.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_juliadollis}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
CFG="${CFG:-configs/ab15k_controle.yaml}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
BATCH_EFETIVO_ALVO="${BATCH_EFETIVO_ALVO:-32}"
NV_FLAG="${NV_FLAG:---nv}"
MAIN_PORT="${MAIN_PORT:-29${SLURM_JOB_ID: -3}}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

for _env in "${ENV_FILE:-}" "${PROJECT_ROOT}/.env" "${BIND_ROOT}/.env" "${BIND_ROOT}/projects/.env"; do
    if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; echo "[env] ${_env}"; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN ausente}"
: "${WANDB_API_KEY:?ERRO: WANDB_API_KEY ausente}"

export HF_HOME="${HF_HOME:-${BIND_ROOT}/hf-cache}"
mkdir -p "${HF_HOME}"

[ -f "${PROJECT_ROOT}/${CFG}" ] || { echo "ERRO: config nao encontrado: ${CFG}"; exit 1; }
[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ] || {
    echo "ERRO: third_party/Genfocus ausente"; exit 1; }

echo "=============================================================="
echo " ${SLURM_JOB_NAME:-ab15k} | job=${SLURM_JOB_ID:-none} | $(hostname) | $(date)"
echo " config:  ${CFG}"
echo " HF_HOME: ${HF_HOME}  (cota: $(quota -s 2>/dev/null | tail -1 | awk '{print $2"/"$3}'))"
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

echo "--- GPUs ---"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | grep -E "^[0-9]+," || true
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<nao definido>}"

export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=$(python3 -c "import torch;print(torch.cuda.device_count())")
MENOR=$(python3 -c "
import torch
print(min(torch.cuda.mem_get_info(i)[0]//(1024*1024) for i in range(torch.cuda.device_count())))
")
echo "[gpu] ${NUM_GPUS} visivel(is) | menor memoria livre ${MENOR} MiB (minimo '"${MIN_FREE_MIB}"')"
if [ "${MENOR}" -lt '"${MIN_FREE_MIB}"' ]; then
  echo "[erro] GPU alocada sem memoria suficiente. Nao vou usar GPU de terceiro."; exit 1
fi
if   [ "${NUM_GPUS}" -ge 4 ]; then NUM_GPUS=4
elif [ "${NUM_GPUS}" -ge 2 ]; then NUM_GPUS=2
else                               NUM_GPUS=1; fi
GRAD_ACCUM=$(( '"${BATCH_EFETIVO_ALVO}"' / NUM_GPUS ))
echo "[gpu] ${NUM_GPUS} GPU(s) | grad_accum=${GRAD_ACCUM} -> efetivo $(( GRAD_ACCUM * NUM_GPUS ))"

python3 - <<PYV
import torch, transformers, diffusers, peft, accelerate, datasets
print("torch", torch.__version__, "| diffusers", diffusers.__version__, "| peft", peft.__version__)
PYV
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus\")"

CFG_EF="configs/_efetivo_${SLURM_JOB_ID:-manual}.yaml"
python3 - <<PYC
import yaml
cfg = yaml.safe_load(open("'"${CFG}"'"))
cfg.setdefault("runtime", {})["gradient_accumulation_steps"] = int(${GRAD_ACCUM})
yaml.safe_dump(cfg, open("${CFG_EF}", "w"), sort_keys=False, allow_unicode=True)
print("[cfg]", "${CFG_EF}", "grad_accum =", ${GRAD_ACCUM})
PYC

python3 -m genfocus_train.train check --config "${CFG_EF}"

echo ">>> TREINO ${NUM_GPUS} GPU(s) <<<"
if [ "${NUM_GPUS}" -gt 1 ]; then
  exec accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" \
       --mixed_precision bf16 --dynamo_backend no --main_process_port '"${MAIN_PORT}"' \
       -m genfocus_train.train deblur --config "${CFG_EF}"
else
  exec python3 -m genfocus_train.train deblur --config "${CFG_EF}"
fi
'

srun singularity exec ${NV_FLAG} \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=== FIM (rc=$?) ==="
