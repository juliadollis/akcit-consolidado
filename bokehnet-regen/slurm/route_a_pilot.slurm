#!/bin/bash
#SBATCH --job-name=bokeh-a-pilot
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# PILOTO da rota A — §3.2(a), pré-treino sintético. ~20 imagens x N variantes,
# TODOS os sete gates em modo medir.
#
# O produto deste job NÃO é dataset. São quatro números que não existem ainda:
#
#   1. o histograma de rejeição da rota A (nenhuma versão dela jamais teve um);
#   2. a distribuição de `coc_p99_px` e de `defocus_saturation_ratio` com
#      MAX_COC = 100 — a medição da rota B foi com max_coc = 10,5107 (17,3%
#      saturado, ACHADOS.md:57), e com 100 é OUTRO regime, não medido;
#   3. a área da banda em foco, que decide os limiares de `mask_area_ratio`;
#   4. **o custo de GPU por render do BokehMe.** O job 32212 mediu o CONTRATO do
#      renderer (ACHADOS.md:245-290), não o throughput. 70 mil renders é a única
#      etapa do projeto sem estimativa. Cronometre aqui antes de dimensionar o
#      run completo.
#
# DEPENDÊNCIA DE ORDEM, não preferência: a rota A sorteia K da distribuição
# medida em B e C (PLANO_EXECUCAO.md). A rota B publicada tem k = 50,0 em
# 11.635/11.635 (ACHADOS.md:14) e a rota C 47,0% no teto exato de 300
# (ACHADOS.md:19) — sortear daí é sortear de duas constantes. O entrypoint
# RECUSA uma distribuição degenerada, e é para recusar.
#
# NÃO cancela nem toca em nenhum outro job. A QOS `onejob` permite 2 rodando.
# Nada é apagado: `--output-dir` é destino novo, e a rota retoma por sample_id.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

DATA_ROOT=${DATA_ROOT:-/raid/user_danielpedrozo/projects/julia/data}
# As DUAS fontes do §3.2(a): [80] = Generative Photography (paper.txt:941) e
# [27] = EBB! (paper.txt:814). DiffCamera é [69], BASELINE, e não é fonte.
GENPHOTO_DIR=${GENPHOTO_DIR:-${DATA_ROOT}/GenerativePhotography}
EBB_DIR=${EBB_DIR:-${DATA_ROOT}/EBB_sharp}
MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/a_pilot}
K_DIST=${K_DIST:-${PROJECT_ROOT}/manifests/k_distribution_bc.json}
PILOT_N=${PILOT_N:-20}
# [A] 41 = 70K/1,7K (paper.txt:527-528, 998). O paper NÃO publica o número.
SAMPLES_PER_IMAGE=${SAMPLES_PER_IMAGE:-41}
# [A] a divisão do pool de ~1,7K entre as duas fontes não é publicada.
TOP_N_GENPHOTO=${TOP_N_GENPHOTO:-850}
TOP_N_EBB=${TOP_N_EBB:-850}

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

echo "=========================================="
echo " Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:   $(hostname)"
echo " Saída:  ${OUTPUT_DIR}"
echo " K dist: ${K_DIST}"
echo " N/img:  ${SAMPLES_PER_IMAGE}   imagens: ${PILOT_N}"
echo " Data:   $(date)"
echo "=========================================="

for caminho in "${SIF}" "${PROJECT_ROOT}/third_party/BokehMe/demo.py" \
               "${PROJECT_ROOT}/output/renderer_verification.json" "${K_DIST}"; do
    [[ -e "${caminho}" ]] || { echo "ERRO: não encontrado: ${caminho}"; exit 1; }
done
for dir in "${GENPHOTO_DIR}" "${EBB_DIR}" "${MODELS_DIR}"; do
    [[ -d "${dir}" ]] || { echo "ERRO: diretório não encontrado: ${dir}"; exit 1; }
done

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

# --- seleção de GPU por UUID, nunca por índice -------------------------------
# A GPU4 da h100n3 já apareceu com defeito, imprimindo "Unable to determine the
# device handle for GPU4" NO MEIO do stdout e desalinhando os índices do
# nvidia-smi em relação aos do CUDA. O grep filtra isso.
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
# PREFIXA, nunca substitui. `.pydeps-clean` e nao `.pydeps`: o `.pydeps` tem um
# torch 2.14+cu130 que SOMBREARIA o 2.9.0+cu126 do container — a versao contra a
# qual o renderer foi verificado.
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

python3 -c "import torch, numpy, cupy; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| cupy\", cupy.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local (compileall e AST não pegam NameError; a suíte pega) <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> smoke de 2 amostras antes de qualquer coisa <<<"
# `--samples-per-image 2 --pilot 1` = uma imagem, duas variantes. É o smoke que
# pega NameError e caminho de escrita; não é medição de nada.
python3 scripts/run_route_a.py \
    --output-dir '"${OUTPUT_DIR}"'_smoke --pilot 1 --samples-per-image 2 \
    --k-distribution '"${K_DIST}"' \
    --source GenerativePhotography='"${GENPHOTO_DIR}"' \
    --source "EBB!"='"${EBB_DIR}"' \
    --top-n-per-source GenerativePhotography=2 --top-n-per-source "EBB!"=2 \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda

echo
echo ">>> piloto: '"${PILOT_N}"' imagens x '"${SAMPLES_PER_IMAGE}"' variantes, gates em modo medir <<<"
echo ">>> CRONOMETRE: o tempo abaixo dividido pelo nº de amostras é o custo/render <<<"
/usr/bin/time -v python3 scripts/run_route_a.py \
    --output-dir '"${OUTPUT_DIR}"' --pilot '"${PILOT_N}"' \
    --samples-per-image '"${SAMPLES_PER_IMAGE}"' \
    --k-distribution '"${K_DIST}"' \
    --source GenerativePhotography='"${GENPHOTO_DIR}"' \
    --source "EBB!"='"${EBB_DIR}"' \
    --top-n-per-source GenerativePhotography='"${TOP_N_GENPHOTO}"' \
    --top-n-per-source "EBB!"='"${TOP_N_EBB}"' \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda --seed 0

echo
echo ">>> distribuições e cortes propostos <<<"
python3 scripts/calibrate_thresholds.py \
    --pilot-dir '"${OUTPUT_DIR}"' \
    --output-json '"${PROJECT_ROOT}"'/output/thresholds_rota_a_proposta.json
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo " Próximo passo: ler output/thresholds_rota_a_proposta.json, escolher os"
echo " números, anotar o custo/render medido em reference/ACHADOS.md, e passar"
echo " os limiares em route_a_full.slurm. Nenhum limiar foi aplicado aqui."
echo "=========================================="
