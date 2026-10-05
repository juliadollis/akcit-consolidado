#!/bin/bash
#SBATCH --job-name=deblur-medicoes-c0
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# O QUE FAZ
#   Roda as medicoes C0 do PLANO_CORRECOES_DEBLURNET.md. Sao diagnosticos: nao
#   treinam, nao escrevem no HF, nao mudam config nenhum. Servem para trocar
#   suposicao por numero medido antes de o treino longo comecar.
#
#     scripts/c0_1_resolucao_dfs.py    resolucao REALMENTE armazenada nos dfs
#                                      (decide se o LPIPS e comparavel ao
#                                      numero publicado na Tabela 2)
#     scripts/c0_2_gts_distintas.py    quantas GTs distintas o top-3000 do
#                                      RealBokeh de fato seleciona
#     scripts/c0_3_modulos_lora.py     contagem de modulos LoRA injetados
#                                      (tem que dar 343 depois do C1) — este e
#                                      o unico que precisa de GPU, porque
#                                      carrega o FLUX
#
#   Roda cada um de forma TOLERANTE: se um script ainda nao existe ou falha, o
#   job registra e segue para o proximo, em vez de morrer no primeiro.
#
# QUANTO DEMORA
#   Dominado pelo download dos datasets (o RealBokeh train tem ~20k linhas) e
#   pela carga do FLUX no c0_3. Com cache frio, algumas horas. --time em 1 dia.
#
# O QUE GERA
#   outputs/medicoes_c0/*.json e o proprio log. Nada mais.
#
# COMO RODAR (crie o logs/ antes do primeiro sbatch)
#   cd /raid/user_juliadollis/projects/retreinar-deblur && mkdir -p logs
#   sbatch slurm/medicoes_c0.slurm
#
# COMO ACOMPANHAR
#   tail -n +1 -f logs/deblur-medicoes-c0-<JOBID>.out logs/deblur-medicoes-c0-<JOBID>.err
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_juliadollis}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/retreinar-deblur}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
NV_FLAG="${NV_FLAG:---nv}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs/medicoes_c0

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. cp .env.example .env && edite}"
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${BIND_ROOT}/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }

echo "=== medicoes C0 @ $(hostname) job=${SLURM_JOB_ID:-local} ==="

APP_CMD='
set -uo pipefail   # SEM -e: uma medicao que falha nao pode derrubar as outras
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -n "$free_gpu" ]; then
  gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
  gpu_mem=$(echo  "$free_gpu" | cut -d, -f3 | tr -d " ")
  echo "[gpu] mais livre: ${gpu_mem} MiB em uso"
  if [ "$gpu_mem" -le 500 ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_uuid
  else
    echo "[aviso] nenhuma GPU vazia; o c0_3 (que carrega o FLUX) provavelmente vai falhar."
  fi
fi

rc_total=0
for script in scripts/c0_1_resolucao_dfs.py scripts/c0_2_gts_distintas.py scripts/c0_3_modulos_lora.py; do
  echo ""
  echo "==================================================================="
  echo ">>> ${script}"
  echo "==================================================================="
  if [ ! -f "${script}" ]; then
    echo "[pulado] ${script} ainda nao existe nesta arvore."
    continue
  fi
  if python3 "${script}" --out outputs/medicoes_c0; then
    echo "[ok] ${script}"
  else
    rc=$?
    echo "[FALHOU rc=${rc}] ${script} — seguindo para a proxima medicao."
    rc_total=1
  fi
done

echo ""
echo "=== resumo das medicoes ==="
ls -la outputs/medicoes_c0/ 2>/dev/null || echo "(nada gerado)"
for f in outputs/medicoes_c0/*.json; do
  [ -f "$f" ] || continue
  echo "--- $f"
  python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2, ensure_ascii=False))" "$f"
done
exit ${rc_total}
'

srun singularity exec ${NV_FLAG} \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  --env TOKENIZERS_PARALLELISM=false \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  "${SIF}" bash -lc "${APP_CMD}" || echo "[aviso] alguma medicao falhou; veja o log acima."

echo "=== FIM ==="
