#!/bin/bash
#SBATCH --job-name=bokeh-diag-mask
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
P=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
R=/raid/user_danielpedrozo/projects/julia
M=$R/hf-cache/hub/datasets--akcit-pixel--RealBokeh/snapshots/fb183bfd78525ba4ee2c69d19943ea27d3a08ddc/data
APP='
set -euo pipefail
cd '"$P"'
free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
export CUDA_VISIBLE_DEVICES=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
export PYTHONUNBUFFERED=1
export PYTHONPATH='"$P"'/.pydeps-clean:'"$P"'/src:'"$P"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}
echo ">>> CENAS REJEITADAS (esperado: modelo nao acha nada) <<<"
python3 scripts/diagnose_empty_masks.py --mirror-dir '"$M"' --raw-dir '"$R"'/data/RealBokeh_3MP \
  --models-dir '"$R"'/models --output-json '"$P"'/output/diag_mask_rejeitadas.json \
  --scenes train_102,train_1527,train_1800,train_1873 \
  --limit-scenes 12
echo; echo ">>> CENAS ACEITAS (controle) <<<"
python3 scripts/diagnose_empty_masks.py --mirror-dir '"$M"' --raw-dir '"$R"'/data/RealBokeh_3MP \
  --models-dir '"$R"'/models --output-json '"$P"'/output/diag_mask_controle.json --limit-scenes 8
'
srun singularity exec --nv --bind /raid/user_danielpedrozo:/raid/user_danielpedrozo \
  /raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif bash -lc "$APP"
