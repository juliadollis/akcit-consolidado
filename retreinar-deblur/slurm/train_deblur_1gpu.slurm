#!/bin/bash
#SBATCH --job-name=deblur-paper-1gpu
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# O QUE FAZ
#   Mesmo treino do train_deblur_4gpu.slurm, em 1 GPU. Use quando as 4 nao
#   estiverem disponiveis, ou para depurar o caminho longo sem ocupar o no.
#
# ATENCAO — BATCH EFETIVO
#   O paper usa batch efetivo 32 (1 x grad_accum 8 x 4 GPUs). Com 1 GPU, o
#   `gradient_accumulation_steps: 8` do YAML da batch efetivo 8, NAO 32.
#   Para manter 32 numa GPU so, suba a acumulacao na hora de submeter:
#       GRAD_ACCUM=32 sbatch slurm/train_deblur_1gpu.slurm
#   Isso escreve um override do YAML em configs/_override_1gpu.yaml (arquivo
#   novo; o config original nao e tocado). Sem GRAD_ACCUM o script usa o YAML
#   como esta e AVISA no log que o batch efetivo ficou 8.
#
# QUANTO DEMORA
#   ~4x o tempo do run de 4 GPUs para o mesmo numero de steps (e 4x de novo se
#   voce subir grad_accum para 32). --time em 14 dias de proposito.
#
# O QUE GERA / COMO ACOMPANHAR / RE-SBATCH
#   Identico ao train_deblur_4gpu.slurm — mesmos caminhos de outputs, mesmo
#   repo do HF, mesmo run do wandb. Ver o cabecalho daquele arquivo.
#
# COMO RODAR (crie o logs/ antes do primeiro sbatch)
#   cd /raid/user_juliadollis/projects/retreinar-deblur && mkdir -p logs
#   sbatch slurm/train_deblur_1gpu.slurm
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_juliadollis}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_deblur_paper.yaml}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
NV_FLAG="${NV_FLAG:---nv}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. cp .env.example .env && edite}"
: "${WANDB_API_KEY:?ERRO: WANDB_API_KEY nao definido. cp .env.example .env && edite}"
export HF_TOKEN WANDB_API_KEY
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${BIND_ROOT}/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }
[ -f "${PROJECT_ROOT}/${TRAIN_CONFIG}" ] || { echo "ERRO: config nao encontrado: ${TRAIN_CONFIG}"; exit 1; }

# --- Override opcional de grad_accum (cria arquivo NOVO, nao edita o original)
if [ -n "${GRAD_ACCUM}" ]; then
  OVERRIDE="configs/_override_1gpu.yaml"
  python3 - "$TRAIN_CONFIG" "$OVERRIDE" "$GRAD_ACCUM" <<'PY'
import sys, yaml
src, dst, ga = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = yaml.safe_load(open(src, encoding="utf-8"))
d.setdefault("runtime", {})["gradient_accumulation_steps"] = ga
d["runtime"]["output_dir"] = d["runtime"].get("output_dir", "outputs/run") + "_1gpu"
d.setdefault("logging", {})
if d["logging"].get("run_name"):
    d["logging"]["run_name"] += "-1gpu"
with open(dst, "w", encoding="utf-8") as h:
    yaml.safe_dump(d, h, sort_keys=False, allow_unicode=True)
print(f"[override] {dst}: gradient_accumulation_steps={ga}, output_dir={d['runtime']['output_dir']}")
PY
  TRAIN_CONFIG="${OVERRIDE}"
else
  echo "[aviso] sem GRAD_ACCUM: o batch efetivo sera grad_accum x 1 GPU, NAO os 32 do paper."
  echo "        Para bater com o paper: GRAD_ACCUM=32 sbatch slurm/train_deblur_1gpu.slurm"
fi

echo "=============================================================="
echo "Job:     ${SLURM_JOB_NAME} (${SLURM_JOB_ID:-local})  no $(hostname)"
echo "Config:  ${TRAIN_CONFIG}"
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

# 1 GPU: escolher uma GPU VAZIA por UUID. O SLURM deste cluster pode alocar uma
# GPU ja ocupada, e na h100n3 os indices do nvidia-smi nao batem com os do CUDA
# por causa da GPU com defeito (INSTRUCOES_H100.md).
free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -z "$free_gpu" ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi
gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
gpu_mem=$(echo  "$free_gpu" | cut -d, -f3 | tr -d " ")
echo "[gpu] mais livre: ${gpu_mem} MiB em uso"
if [ "$gpu_mem" -gt 500 ]; then echo "[erro] GPU ocupada (${gpu_mem} MiB). Abortando para nao tomar OOM."; exit 1; fi
export CUDA_VISIBLE_DEVICES=$gpu_uuid

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

echo ">>> TREINO 1 GPU — '"${TRAIN_CONFIG}"' <<<"
python3 -m genfocus_train.train deblur --config '"${TRAIN_CONFIG}"'
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
