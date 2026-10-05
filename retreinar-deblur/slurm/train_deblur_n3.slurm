#!/bin/bash
#SBATCH --job-name=deblur-paper-n3
#SBATCH --partition=h100n3
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
# DeblurNet (GenRefocus Stage 1) — replicacao do paper — dgx-H100-03 (h100n3)
# =============================================================================
#
# O QUE FAZ: treina o LoRA da DeblurNet sobre FLUX.1-dev, 60K steps, batch
# efetivo 32, seguindo o paper §4.1 e §B.1.
#
# POR QUE 2 GPUS E NAO 4 (medido em 2026-09-10):
#   scontrol show node dgx-H100-03 -> Gres=gpu:h100:3
#   O SLURM da h100n3 so gerencia 3 das 8 GPUs fisicas; as outras 5 estao fora
#   do controle dele (e onde rodam os VLLM de outros usuarios). Com 1 ja
#   alocada, sobram 2. Um --gres=gpu:4 aqui ficaria PENDING para sempre.
#   h100n2 (8 GPUs) e b200n1 (8) estavam com AllocTRES=gpu=8, ou seja, 0 livres.
#   Batch efetivo e PRESERVADO: 1 x 16 x 2 = 32, exatamente o paper §4.1.
#   Se um dia houver 4 livres, basta --gres=gpu:4: a acumulacao se ajusta sozinha.
#
# QUANTO DEMORA: ~31 s/step em 1 H100 no treino anterior; com 2 GPUs e batch
# efetivo 32, na ordem de 15-20 s/step. 60K steps ~ 11-14 dias. Por isso
# --time=20-00:00:00 (regra do INSTRUCOES_H100.md: tempo BEM alto).
#
# O QUE GERA:
#   outputs/deblur_paper/deblur/checkpoints/step_*.pt   (a cada 250 steps)
#   outputs/deblur_paper/deblur/checkpoints/best.pt     (melhor val_loss, nunca podado)
#   outputs/deblur_paper/deblur/deblur.safetensors      (+ deblur.json) no fim
#   wandb: projeto genrefocus-deblurnet-v2, run retomavel
#   HF:    um repo, um arquivo POR STEP (nada se sobrescreve)
#
# COMO ACOMPANHAR:
#   ssh dgx-H100-03 'cd /raid/user_danielpedrozo/projects/julia/retreinar-deblur && tail -n +1 -f logs/deblur-paper-n3-<JOBID>.out'
#
# RE-SBATCH E SEGURO: `resume: true` no config retoma do ultimo checkpoint, e o
# id do wandb e estavel, entao a curva continua no MESMO grafico.
#
# NUNCA: cancelar job (nem meu, nem de terceiro), apagar nada, baixar fora do
# SLURM. O FLUX ja esta no hf-cache; HF_HUB_OFFLINE=1 garante que nada e baixado.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_danielpedrozo}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/julia/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_deblur_paper.yaml}"

# Memoria livre minima exigida por GPU. FLUX bf16 (~24 GB) + LoRA fp32 + estados
# do Adam + ativacoes com gradient checkpointing cabem folgado em 45 GB.
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
# Batch efetivo do paper (§4.1: 1 x 8 x 4 GPUs). A acumulacao e derivada do
# numero de GPUs REALMENTE usaveis, para o batch efetivo nao mudar.
BATCH_EFETIVO_ALVO="${BATCH_EFETIVO_ALVO:-32}"

# NV_FLAG: mecanismo de GPU do singularity. Na h100n3 o default historico e
# --nvccli; se o nvidia-container-cli falhar (nvml error por causa da GPU com
# defeito), re-submeta com NV_FLAG=--nv.
NV_FLAG="${NV_FLAG:---nvccli}"
MAIN_PORT="${MAIN_PORT:-29571}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

# Segredos: .env git-ignored, nunca hardcoded (INSTRUCOES_H100.md regra 4).
for _env in "${ENV_FILE:-}" "${PROJECT_ROOT}/.env" "${BIND_ROOT}/projects/julia/.env"; do
    if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; echo "[env] carregado: ${_env}"; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN ausente. Confira o .env}"
: "${WANDB_API_KEY:?ERRO: WANDB_API_KEY ausente. Confira o .env}"

# O FLUX e os datasets ja estao neste cache. OFFLINE=1 impede qualquer download
# (regra: nunca baixar fora do SLURM — e aqui nem dentro, ja esta tudo local).
export HF_HOME="${HF_HOME:-${BIND_ROOT}/projects/julia/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

[ -f "${PROJECT_ROOT}/${TRAIN_CONFIG}" ] || { echo "ERRO: config nao encontrado: ${TRAIN_CONFIG}"; exit 1; }
[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ] || {
    echo "ERRO: third_party/Genfocus ausente — o treino usa o transformer_forward dos autores."; exit 1; }

echo "=============================================================="
echo " DeblurNet paper | job=${SLURM_JOB_ID:-none} | $(hostname) | $(date)"
echo " Projeto:  ${PROJECT_ROOT}"
echo " Config:   ${TRAIN_CONFIG}"
echo " HF_HOME:  ${HF_HOME}"
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

# ── GPUs: usar o que o SLURM alocou, e SO isso ──────────────────────────────
# O SLURM da h100n3 gerencia 3 das 8 GPUs fisicas. As outras 5 tem processos de
# terceiros (VLLM) FORA do SLURM. Portanto:
#   - NAO escolhemos GPU a mao aqui. Tomar uma GPU que o SLURM nao nos deu pode
#     colidir com o job de outra pessoa. Usamos exatamente CUDA_VISIBLE_DEVICES.
#   - Mas CONFERIMOS a memoria livre das GPUs que recebemos e ABORTAMOS se nao
#     couber, em vez de morrer de OOM depois de carregar o FLUX.
# O INSTRUCOES_H100.md permite escolher GPU livre a mao no caso de 1 GPU; aqui
# preferimos abortar a arriscar atrapalhar terceiro.
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
  | grep -E "^[0-9]+," || true
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<nao definido>}"

NUM_GPUS=$(python3 -c "import torch;print(torch.cuda.device_count())")
if [ "${NUM_GPUS}" -lt 1 ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi

# Memoria livre da GPU mais apertada entre as VISIVEIS (indices ja remapeados
# pelo CUDA_VISIBLE_DEVICES, entao consultamos via torch, nao via nvidia-smi —
# na h100n3 os indices do nvidia-smi nao batem com os do CUDA).
MENOR_LIVRE=$(python3 -c "
import torch
livres = []
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    livres.append(free // (1024*1024))
print(min(livres))
")
echo "[gpu] ${NUM_GPUS} visivel(is); menor memoria livre: ${MENOR_LIVRE} MiB (minimo exigido: '"${MIN_FREE_MIB}"')"
if [ "${MENOR_LIVRE}" -lt '"${MIN_FREE_MIB}"' ]; then
  echo "[erro] GPU alocada sem memoria suficiente (${MENOR_LIVRE} MiB livres)."
  echo "[erro] Nao vou usar GPU de outra pessoa. Re-submeta mais tarde."
  exit 1
fi

# So 1, 2 ou 4: a acumulacao tem que dividir o batch efetivo alvo exato.
if   [ "${NUM_GPUS}" -ge 4 ]; then NUM_GPUS=4
elif [ "${NUM_GPUS}" -ge 2 ]; then NUM_GPUS=2
else                               NUM_GPUS=1
fi
GRAD_ACCUM=$(( '"${BATCH_EFETIVO_ALVO}"' / NUM_GPUS ))
echo "[gpu] usando ${NUM_GPUS} GPU(s) | grad_accum=${GRAD_ACCUM} -> batch efetivo = 1 x ${GRAD_ACCUM} x ${NUM_GPUS} = $(( GRAD_ACCUM * NUM_GPUS ))"

# PYTHONPATH so com CODIGO DO PROJETO, montado AQUI DENTRO e nunca na chamada do
# singularity (INSTRUCOES_H100.md: --env PYTHONPATH tira o ~/.local do sys.path
# e some com diffusers/peft).
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "--- versoes ---"
python3 - <<PYV
import torch, transformers, diffusers, peft, accelerate, datasets
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| gpus visiveis", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("diffusers   ", diffusers.__version__)
print("peft        ", peft.__version__)
print("accelerate  ", accelerate.__version__)
print("datasets    ", datasets.__version__)
PYV
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importavel\")"

# Config efetivo: mesma base, com a acumulacao derivada do nº real de GPUs.
# NAO altera o config original — escreve um derivado, que fica como trilha.
CFG_EFETIVO="configs/_efetivo_${SLURM_JOB_ID:-manual}.yaml"
python3 - <<PYC
import yaml
base = "'"${TRAIN_CONFIG}"'"
cfg = yaml.safe_load(open(base))
cfg.setdefault("runtime", {})["gradient_accumulation_steps"] = int(${GRAD_ACCUM})
with open("${CFG_EFETIVO}", "w") as h:
    yaml.safe_dump(cfg, h, sort_keys=False, allow_unicode=True)
print("[cfg] derivado:", "${CFG_EFETIVO}", "grad_accum =", ${GRAD_ACCUM})
PYC

echo "--- check (nao treina, valida config e credenciais) ---"
python3 -m genfocus_train.train check --config "${CFG_EFETIVO}"

echo ">>> TREINO ${NUM_GPUS} GPU(s) <<<"
if [ "${NUM_GPUS}" -gt 1 ]; then
  exec accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" \
       --mixed_precision bf16 --dynamo_backend no --main_process_port '"${MAIN_PORT}"' \
       -m genfocus_train.train deblur --config "${CFG_EFETIVO}"
else
  exec python3 -m genfocus_train.train deblur --config "${CFG_EFETIVO}"
fi
'

srun singularity exec ${NV_FLAG} \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=== FIM (rc=$?) ==="
