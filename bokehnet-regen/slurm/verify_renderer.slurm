#!/bin/bash
#SBATCH --job-name=bokeh-renderer-verify
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Verificação do renderer (os três testes do agente `renderer-verifier`) contra
# o BokehMe DE VERDADE. É a única parte que não roda sem GPU, e é ela que
# autoriza `is_final_label_renderer`.
#
# NÃO cancela nem toca em nenhum outro job. A QOS `onejob` permite 2 jobs
# rodando; este ocupa a segunda vaga, ao lado do treino da DeblurNet.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/output"

echo "=========================================="
echo " Job:   ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:  $(hostname)"
echo " Root:  ${PROJECT_ROOT}"
echo " Data:  $(date)"
echo "=========================================="

if [[ ! -f "${SIF}" ]]; then echo "ERRO: container não encontrado: ${SIF}"; exit 1; fi
if [[ ! -f "${PROJECT_ROOT}/third_party/BokehMe/demo.py" ]]; then
    echo "ERRO: BokehMe não encontrado em ${PROJECT_ROOT}/third_party/BokehMe"; exit 1
fi

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
# PYTHONPATH só com código do projeto, e PREFIXANDO. Nunca sobrescrever
# PYTHONUSERBASE: tira o ~/.local do sys.path.
# .pydeps traz o cupy (kernel de scatter do BokehMe). PREFIXA, nunca substitui:
# substituir tiraria o ~/.local do sys.path e sumiria com peft/diffusers.
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

python3 -c "import torch; print(\"[env] torch\", torch.__version__, \"|\", torch.cuda.get_device_name(0))"
python3 -c "import numpy, cupy; print(\"[env] numpy\", numpy.__version__, \"| cupy\", cupy.__version__)"

echo
echo ">>> suíte de testes locais (48 testes, sem GPU) <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -5

echo
echo ">>> verificação do renderer contra o BokehMe real <<<"
python3 scripts/verify_renderer.py \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --device cuda \
    --output-json '"${PROJECT_ROOT}"'/output/renderer_verification.json
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo "=========================================="
