#!/bin/bash
#SBATCH --job-name=bokeh-c-pilot
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:3
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# PILOTO da rota C — ~200 amostras, TODOS os dez gates em modo medir.
#
# O produto deste job NÃO é dataset: é o histograma de rejeição e a distribuição
# de cada métrica. `scripts/calibrate_thresholds.py` lê a saída e propõe os
# limiares; o run completo (`route_c_full.slurm`) recebe os números escolhidos.
#
# NÃO cancela nem toca em nenhum outro job. A QOS `onejob` permite 2 rodando;
# este ocupa a segunda vaga, ao lado do treino que já está lá.
# Nada é apagado: `--output-dir` é destino novo, e a rota retoma por sample_id.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

DATA_ROOT=${DATA_ROOT:-/raid/user_danielpedrozo/projects/julia/data}
MIRROR_DIR=${MIRROR_DIR:-${DATA_ROOT}/RealBokeh_mirror}
RAW_DIR=${RAW_DIR:-${DATA_ROOT}/RealBokeh_3MP}
MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/c_pilot}
PILOT_N=${PILOT_N:-200}

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

echo "=========================================="
echo " Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:   $(hostname)"
echo " Saída:  ${OUTPUT_DIR}"
echo " Data:   $(date)"
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

# --- seleção de GPU por UUID, nunca por índice -------------------------------
# A GPU4 da h100n3 já apareceu com defeito, imprimindo "Unable to determine the
# device handle for GPU4" NO MEIO do stdout e desalinhando os índices do
# nvidia-smi em relação aos do CUDA. O grep filtra isso.
# As N GPUs mais livres, por UUID. Nunca por índice: a GPU4 da h100n3 já apareceu com
# defeito, imprimindo "Unable to determine the device handle for GPU4" NO MEIO do stdout
# e desalinhando os índices do nvidia-smi em relação aos do CUDA. O grep filtra isso.
N_GPUS=${N_GPUS:-3}
livres=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
         | grep -E "^[0-9]+," | awk -F, "\$3+0 < 2000" | sort -t, -k3 -n | head -${N_GPUS})
n_livres=$(echo "$livres" | grep -c . || true)
if [ "$n_livres" -lt "$N_GPUS" ]; then
  echo "[erro] só ${n_livres} GPU(s) com <2000 MiB em uso; pedi ${N_GPUS}. Abortando em vez de tomar OOM."
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E "^[0-9]+," || true
  exit 1
fi
gpu_uuids=$(echo "$livres" | cut -d, -f2 | tr -d " " | paste -sd, -)
echo "[gpu] usando ${N_GPUS}: $(echo "$livres" | cut -d, -f1 | tr -d " " | paste -sd" " -)"
export CUDA_VISIBLE_DEVICES=$gpu_uuids

export PYTHONUNBUFFERED=1
# PREFIXA, nunca substitui: substituir tiraria o ~/.local do sys.path e sumiria
# com peft/diffusers. Nunca sobrescrever PYTHONUSERBASE pelo mesmo motivo.
#
# `.pydeps-clean` e nao `.pydeps`: o `.pydeps` recebeu um `pip install kornia` SEM
# --no-deps que arrastou torch 2.14+cu130 junto. Como o PYTHONPATH prefixa, aquele
# torch SOMBREARIA o 2.9.0+cu126 do container — que e exatamente a versao contra a
# qual o renderer foi verificado. O `.pydeps-clean` tem cupy 12.3 + einops + kornia
# instalados com --no-deps, e nada mais.
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

python3 -c "import torch, numpy, cupy; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| cupy\", cupy.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local (compileall e AST não pegam NameError; a suíte pega) <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> smoke de 2 amostras antes de qualquer coisa <<<"
python3 scripts/run_route_c.py --source realbokeh --pilot 2 \
    --output-dir '"${OUTPUT_DIR}"'_smoke \
    --mirror-dir '"${MIRROR_DIR}"' --raw-dir '"${RAW_DIR}"' \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda

echo
echo ">>> piloto de '"${PILOT_N}"' amostras, todos os gates em modo medir <<<"
python3 scripts/run_route_c.py --source realbokeh --pilot '"${PILOT_N}"' \
    --output-dir '"${OUTPUT_DIR}"' \
    --mirror-dir '"${MIRROR_DIR}"' --raw-dir '"${RAW_DIR}"' \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda --seed 0

echo
echo ">>> distribuições e cortes propostos <<<"
python3 scripts/calibrate_thresholds.py \
    --pilot-dir '"${OUTPUT_DIR}"' \
    --output-json '"${PROJECT_ROOT}"'/output/thresholds_proposta.json
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo " Próximo passo: ler output/thresholds_proposta.json, escolher os números,"
echo " e passá-los em route_c_full.slurm. Nenhum limiar foi aplicado aqui."
echo "=========================================="
