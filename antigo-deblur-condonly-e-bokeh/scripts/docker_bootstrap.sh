#!/usr/bin/env bash
# =============================================================================
# Roda DENTRO do container Docker (pytorch/pytorch). Faz o setup e dispara o
# treino multi-GPU. Pensado pro dgx-H100-01 (sem SLURM, /raid não compartilhado).
#
# Espera o código montado em /workspace/genrefocus_deblurnet e estas envs
# (passadas via `docker run -e ...`): HF_TOKEN, WANDB_API_KEY, NUM_GPUS,
# TRAIN_CONFIG, HF_HOME. Veja scripts/run_docker.sh (host-side).
# =============================================================================
set -euo pipefail

cd /workspace/genrefocus_deblurnet
export PYTHONPATH="/workspace/genrefocus_deblurnet:/workspace/genrefocus_deblurnet/third_party/Genfocus:${PYTHONPATH:-}"

NUM_GPUS=${NUM_GPUS:-2}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_base_2gpu.yaml}
RUN_SMOKE=${RUN_SMOKE:-0}
UPLOAD_TO_HF=${UPLOAD_TO_HF:-1}
HF_REPO_ID=${HF_REPO_ID:-genrefocus-deblurnet-paper-2gpu}
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "NUM_GPUS:   ${NUM_GPUS}"
echo "TRAIN_CFG:  ${TRAIN_CONFIG}"
echo "HF_HOME:    ${HF_HOME:-<unset>}"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits || true

# 0) git — a imagem pytorch/pytorch NÃO traz, e o setup_genfocus precisa clonar.
if ! command -v git >/dev/null 2>&1; then
    echo ">>> instalando git..."
    apt-get update && apt-get install -y git
fi

# 1) deps — NÃO instala torch/torchvision (a imagem já traz o par compatível).
echo ">>> instalando requirements (sem torch)..."
python3 -m pip install --no-cache-dir \
    transformers diffusers peft accelerate safetensors sentencepiece protobuf \
    datasets python-dotenv Pillow numpy pyyaml wandb huggingface_hub

# 2) Genfocus (transformer_forward) — clona se faltar
if [ ! -f third_party/Genfocus/Genfocus/pipeline/flux.py ]; then
    echo ">>> clonando Genfocus..."
    bash scripts/setup_genfocus.sh
fi
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print('[ok] Genfocus importável')"

# 3) FLUX no cache (baixa o que faltar; 1 worker = sem travar)
echo ">>> garantindo FLUX.1-dev no cache (${HF_HOME})..."
python3 -c "
import os
from huggingface_hub import snapshot_download
p = snapshot_download('black-forest-labs/FLUX.1-dev', token=os.environ.get('HF_TOKEN'), max_workers=1)
print('FLUX em:', p)
"

# 4) wandb
python3 -c "import wandb, os; wandb.login(key=os.environ['WANDB_API_KEY'])" || echo "WARN: wandb login falhou (segue sem)"

# 5) smoke opcional (1 GPU)
if [ "${RUN_SMOKE}" = "1" ]; then
    echo ">>> SMOKE (1 GPU)..."
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train smoke --config configs/train_smoke.yaml
fi

# 6) treino multi-GPU (DDP-free + all-reduce, igual ao Singularity)
echo ">>> TREINO ${NUM_GPUS} GPUs — ${TRAIN_CONFIG}"
PYTHONUNBUFFERED=1 accelerate launch \
    --multi_gpu --num_machines 1 --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 --dynamo_backend no \
    -m genfocus_train.train deblur --config ${TRAIN_CONFIG}

# 7) upload do modelo final
if [ "${UPLOAD_TO_HF}" = "1" ]; then
    echo ">>> UPLOAD pro HF (${HF_REPO_ID})..."
    python3 scripts/hf_upload.py --config ${TRAIN_CONFIG} --repo-id "${HF_REPO_ID}" \
        || echo "WARN: upload falhou; modelo salvo em outputs/<run>/deblur/deblur.safetensors"
fi
