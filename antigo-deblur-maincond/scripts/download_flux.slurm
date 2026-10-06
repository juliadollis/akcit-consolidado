#!/bin/bash
#SBATCH --job-name=flux-download
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Baixa o FLUX.1-dev COMPLETO pro cache (1 processo, sem contenção). Resolve o
# cache incompleto (faltava text_encoder_2/model-00002-of-00002.safetensors).
# Rode ISSO ANTES dos treinos multi-GPU. Depois, os treinos rodam com
# HF_HUB_OFFLINE=1 e carregam do cache na hora, sem travar.
# =============================================================================

HF_TOKEN="${HF_TOKEN:?defina HF_TOKEN no ambiente}"
export HF_TOKEN

export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}
mkdir -p /raid/user_juliadollis/projects/genrefocus_deblurnet/logs

# Limpa downloads parciais / locks SÓ do FLUX (força recompletar o que faltou).
echo ">>> limpando .incomplete / .lock do FLUX no cache..."
find /raid/user_juliadollis/hf-cache -path "*FLUX.1-dev*" \
    \( -name "*.incomplete" -o -name "*.lock" \) -print -delete 2>/dev/null || true

echo ">>> baixando FLUX.1-dev (1 worker, com resume)..."
singularity exec \
    --bind /raid/user_juliadollis:/workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env HF_HUB_OFFLINE=0 \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    "${IMAGE_PATH}" \
    python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('black-forest-labs/FLUX.1-dev', max_workers=1)
print('DOWNLOAD COMPLETO:', p)
"

echo ">>> verificando que o arquivo que faltava existe agora..."
ls -lh /raid/user_juliadollis/hf-cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/text_encoder_2/model-00002-of-00002.safetensors \
    && echo ">>> OK — FLUX completo. Pode rodar os treinos multi-GPU (offline)."
