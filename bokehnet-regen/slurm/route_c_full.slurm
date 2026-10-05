#!/bin/bash
#SBATCH --job-name=bokeh-c-full
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Rota C completa — 20.495 pares da RealBokeh, K pelo sweep da Eq. 5.
#
# PRÉ-REQUISITO: o piloto rodou e os limiares foram ESCOLHIDOS por alguém que
# olhou output/thresholds_proposta.json. Este job recusa começar com todos os
# limiares vazios — rodar 20 mil amostras sem filtro é gerar o dataset de novo
# com o mesmo problema que estamos consertando.
#
# NÃO cancela job nenhum. Nada é apagado: a rota retoma por sample_id, então
# relançar continua de onde parou em vez de reescrever.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

DATA_ROOT=${DATA_ROOT:-/raid/user_danielpedrozo/projects/julia/data}
MIRROR_DIR=${MIRROR_DIR:-${DATA_ROOT}/RealBokeh_mirror}
RAW_DIR=${RAW_DIR:-${DATA_ROOT}/RealBokeh_3MP}
MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/c_realbokeh}

# --- os limiares congelados, vindos do piloto --------------------------------
# Preencha a partir de output/thresholds_proposta.json. Cada linha que ficar
# vazia é um gate que MEDE e não bloqueia — o que é uma escolha legítima, desde
# que seja escolha. Registre os valores usados em reference/ACHADOS.md.
THRESHOLDS=${THRESHOLDS:-""}
# exemplo:
# THRESHOLDS="--min-calibration-ssim 0.72 --min-mask-area-ratio 0.01 \
#             --max-mask-area-ratio 0.85 --min-aif-f-number 8.0"

if [[ -z "${THRESHOLDS}" ]]; then
    echo "ERRO: THRESHOLDS vazio."
    echo "Rode o piloto, leia output/thresholds_proposta.json, escolha os números e"
    echo "relance com:  sbatch --export=ALL,THRESHOLDS='--min-... 0.7 ...' $0"
    echo "Se a intenção for MESMO gerar tudo sem filtro para calibrar depois,"
    echo "passe THRESHOLDS='--seed 0' explicitamente."
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:    $(hostname)"
echo " Saída:   ${OUTPUT_DIR}"
echo " Limiares:${THRESHOLDS}"
echo " Data:    $(date)"
echo "=========================================="

for caminho in "${SIF}" "${PROJECT_ROOT}/third_party/BokehMe/demo.py" \
               "${PROJECT_ROOT}/output/renderer_verification.json"; do
    [[ -e "${caminho}" ]] || { echo "ERRO: não encontrado: ${caminho}"; exit 1; }
done
for dir in "${MIRROR_DIR}" "${RAW_DIR}" "${MODELS_DIR}"; do
    [[ -d "${dir}" ]] || { echo "ERRO: diretório não encontrado: ${dir}"; exit 1; }
done

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -z "$free_gpu" ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi
gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
gpu_mem=$(echo "$free_gpu"  | cut -d, -f3 | tr -d " ")
echo "[gpu] mais livre: memory.used=${gpu_mem} MiB"
if [ "$gpu_mem" -gt 2000 ]; then
  echo "[erro] GPU mais livre com ${gpu_mem} MiB em uso. Abortando em vez de tomar OOM."
  exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu_uuid

export PYTHONUNBUFFERED=1
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

python3 -c "import torch, numpy, cupy; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| cupy\", cupy.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> rota C completa <<<"
python3 scripts/run_route_c.py --source realbokeh \
    --output-dir '"${OUTPUT_DIR}"' \
    --mirror-dir '"${MIRROR_DIR}"' --raw-dir '"${RAW_DIR}"' \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda --seed 0 '"${THRESHOLDS}"'

echo
echo ">>> validação do release (NÃO sobe nada; --yes é manual) <<<"
python3 scripts/publish_release.py --release-dir '"${OUTPUT_DIR}"'
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo "=========================================="
