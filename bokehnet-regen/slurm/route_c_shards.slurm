#!/bin/bash
#SBATCH --job-name=bokeh-c-full3
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
# Rota C completa, 3 fatias em paralelo, uma GPU cada.
#
# MODO MEDIR: nenhum limiar bloqueia. O produto é o dataset BRUTO, com toda
# métrica gravada por amostra. O dataset final de treino sai dele por FILTRO
# sobre o manifesto — sem reprocessar nada, já que `sample_id` é determinístico.
# Foi a recomendação da auditoria: dois produtos, um bruto e auditável, um final.
#
# Fatiamento é POR CENA: todos os níveis de uma cena caem na mesma fatia, que é
# o que permite reusar AIF, profundidade e máscara entre eles.
#
# PER_GPU > 1 porque o trabalho NÃO é limitado por GPU. Medido no run anterior:
# utilização média de 4,7% / 8,2% / 8,8% em 20 amostras de 1 s, com cada processo
# consumindo 1,6 núcleos de 224 e 11,8 GB de 80 GB. O laço é serial — lê parquet,
# decodifica PNG, calcula a retenção em numpy, e só então manda uma rajada curta
# para a GPU. Entre rajadas ela espera. Empilhar processos por GPU é o que preenche
# esse tempo ocioso; mais GPUs não resolveria nada.
#
# NÃO cancela job nenhum. Nada é apagado: cada fatia escreve em pasta própria e
# a rota retoma por sample_id, então relançar continua de onde parou.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
R=/raid/user_danielpedrozo/projects/julia
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif

MIRROR_DIR=${MIRROR_DIR:?defina MIRROR_DIR}
RAW_DIR=${RAW_DIR:-$R/data/RealBokeh_3MP}
MODELS_DIR=${MODELS_DIR:-$R/models}
OUT_BASE=${OUT_BASE:-$PROJECT_ROOT/output/c_full}
N=${N:-3}                 # GPUs
PER_GPU=${PER_GPU:-4}     # processos por GPU
TOTAL=$((N * PER_GPU))

mkdir -p "${PROJECT_ROOT}/logs"
echo "=========================================="
echo " Job:  ${SLURM_JOB_NAME} (${SLURM_JOB_ID})   Host: $(hostname)"
echo " Saída: ${OUT_BASE}_s{0..$((TOTAL-1))}  (${N} GPUs x ${PER_GPU})  Data: $(date)"
echo "=========================================="

for c in "${SIF}" "${PROJECT_ROOT}/third_party/BokehMe/demo.py" \
         "${PROJECT_ROOT}/output/renderer_verification.json"; do
    [[ -e "${c}" ]] || { echo "ERRO: não encontrado: ${c}"; exit 1; }
done

APP='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

# As N GPUs mais livres, por UUID — nunca por índice (a GPU4 da h100n3 tem defeito
# e desalinha os índices do nvidia-smi em relação aos do CUDA).
livres=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
         | grep -E "^[0-9]+," | awk -F, "\$3+0 < 2000" | sort -t, -k3 -n | head -'"${N}"')
n=$(echo "$livres" | grep -c . || true)
if [ "$n" -lt '"${N}"' ]; then echo "[erro] só $n GPU(s) livre(s), pedi '"${N}"'."; exit 1; fi
echo "[gpu] $(echo "$livres" | cut -d, -f1 | tr -d " " | paste -sd" " -)"

export PYTHONUNBUFFERED=1
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

pids=""
i=0
uuids=$(echo "$livres" | cut -d, -f2 | tr -d " ")
for rep in $(seq 1 '"${PER_GPU}"'); do
 for uuid in $uuids; do
  outdir='"${OUT_BASE}"'_s${i}
  mkdir -p "$outdir"
  echo "[fatia $i] gpu ${uuid:0:12} -> $outdir"
  CUDA_VISIBLE_DEVICES=$uuid python3 scripts/run_route_c.py --source realbokeh \
      --shard $i --num-shards '"${TOTAL}"' \
      --output-dir "$outdir" \
      --mirror-dir '"${MIRROR_DIR}"' --raw-dir '"${RAW_DIR}"' \
      --models-dir '"${MODELS_DIR}"' \
      --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
      --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
      --device cuda --seed 0 --max-levels-per-scene 4 \
      > '"${PROJECT_ROOT}"'/logs/shard_${i}_'"${SLURM_JOB_ID}"'.log 2>&1 &
  pids="$pids $!"
  i=$((i+1))
 done
done

echo "[fatias] aguardando:$pids"
rc=0
for p in $pids; do wait $p || rc=1; done
echo "[fatias] todas terminaram (rc=$rc)"
for j in $(seq 0 $(('"${TOTAL}"'-1))); do
  echo "--- fatia $j ---"
  tail -12 '"${PROJECT_ROOT}"'/logs/shard_${j}_'"${SLURM_JOB_ID}"'.log
done
exit $rc
'

srun singularity exec --nv --bind /raid/user_danielpedrozo:/raid/user_danielpedrozo \
  "${SIF}" bash -lc "${APP}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo "=========================================="
