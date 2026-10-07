#!/bin/bash
#SBATCH --job-name=riemann-setup
#SBATCH --partition=h100n2
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/user_juliadollis/projects/depth-riemannian/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/depth-riemannian/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# SETUP (rodar UMA vez) do fine-tuning Riemanniano do DepthPro no cluster.
#
# Adaptacao Singularity + SLURM do fluxo Docker do Francisco
# (github.com/AKCIT-PIXEL/depth-riemannian). NAO modifica o codigo dele; apenas
# replica o que o Dockerfile fazia no build:
#   [1] instala as deps do requirements.docker.txt num PYTHONUSERBASE proprio
#       (isolado do bokeh: /workspace/python-packages-riemann);
#   [2] instala o pacote depth_pro (Apple ml-depth-pro), sem tocar no torch/
#       torchvision com CUDA que ja vem no .sif;
#   [3] roda o teste de geometria (o MESMO que o Dockerfile roda no build);
#   [4] baixa os pesos depth_pro.pt do HF para /workspace/models.
#
# Pre-requisito (fazer 1x, no host, na h100n2):
#   cd /raid/user_juliadollis/projects
#   git clone https://github.com/AKCIT-PIXEL/depth-riemannian
# =============================================================================

export RAID=/raid/user_juliadollis
export IMAGE_PATH=${IMAGE_PATH:-${RAID}/images/transformers-pytorch-gpu.sif}
export PROJECT_HOST=${PROJECT_HOST:-${RAID}/projects/depth-riemannian}

# --- Segredos (NUNCA hardcode) ----------------------------------------------
# O HF_TOKEN so e necessario para baixar os pesos do depth_pro pelo HF.
# Ordem: ja exportado no ambiente > $ENV_FILE > .env na raiz do projeto.
#   cp .env.example .env && ${EDITOR:-nano} .env && chmod 600 .env
for _env in "${ENV_FILE:-}" "${PROJECT_HOST}/.env" "${SLURM_SUBMIT_DIR:-}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env em ${PROJECT_HOST} (cp .env.example .env)}"
export HF_TOKEN
export RIEMANN_SUBDIR=${RIEMANN_SUBDIR:-riemann_depthpro_h100_container}
export DEPTHPRO_URL=${DEPTHPRO_URL:-https://github.com/apple/ml-depth-pro.git}

# Caminhos "dentro do container" (bind /raid/user_juliadollis -> /workspace)
export RIE_C=/workspace/projects/depth-riemannian/${RIEMANN_SUBDIR}
export DP_C=/workspace/projects/depth-riemannian/ml-depth-pro

mkdir -p "${PROJECT_HOST}/logs" "${RAID}/models" "${RAID}/data"

[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: SIF nao encontrado em ${IMAGE_PATH}"; exit 1; }
if [[ ! -d "${PROJECT_HOST}/${RIEMANN_SUBDIR}" ]]; then
    echo "ERRO: repo do Francisco nao esta em ${PROJECT_HOST}/${RIEMANN_SUBDIR}."
    echo "      Faca antes, no host:"
    echo "        cd ${RAID}/projects && git clone https://github.com/AKCIT-PIXEL/depth-riemannian"
    exit 1
fi

# Clona o ml-depth-pro no HOST (mesmo padrao usado com o Genfocus no eval),
# para nao depender de git dentro do container.
if [[ ! -d "${PROJECT_HOST}/ml-depth-pro/.git" ]]; then
    echo ">>> clonando ml-depth-pro em ${PROJECT_HOST}/ml-depth-pro"
    git clone --depth 1 "${DEPTHPRO_URL}" "${PROJECT_HOST}/ml-depth-pro"
fi

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) | $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail

# Shims de curl/wget: o .sif nao tem esses binarios, e o codigo do Francisco
# chama os dois via os.system/subprocess. Ver slurm/bin/curl e slurm/bin/wget.
export PATH="/workspace/projects/depth-riemannian/slurm/bin:${PATH}"

echo ">>> [1/4] instalando deps (requirements.docker.txt + huggingface_hub) no PYTHONUSERBASE..."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r "${RIE_C}/requirements.docker.txt"
python3 -m pip install --user huggingface_hub

echo ">>> [2/4] instalando depth_pro (Apple ml-depth-pro)..."
# O torch/torchvision ja satisfeitos pelo container NAO sao reinstalados pelo pip.
# Se algum dia o pip tentar trocar o torch por uma build CPU, troque a linha por:
#   pip install --user --no-deps "${DP_C}" && pip install --user timm pillow_heif
python3 -m pip install --user "${DP_C}"
python3 -c "import torch; print(\"[ok] torch\", torch.__version__, \"cuda?\", torch.cuda.is_available())"
python3 -c "import depth_pro; print(\"[ok] depth_pro importavel\")"

echo ">>> [3/4] teste de geometria (curvatura Gaussiana) - o MESMO do Dockerfile..."
cd "${RIE_C}"
python3 scripts/test_geometry.py

echo ">>> [4/4] baixando pesos depth_pro.pt (HF) para /workspace/models..."
# O download_weights.py do Francisco usa wget (ausente no nosso .sif). Se falhar,
# cai no dl_weights.py (API huggingface_hub, sem CLI/wget/PATH).
python3 scripts/download_weights.py --out-dir /workspace/models || \
  DL_OUT=/workspace/models python3 /workspace/projects/depth-riemannian/slurm/dl_weights.py

# Conveniencia: o create_model_and_transforms do depth_pro procura os pesos em
# ./checkpoints/depth_pro.pt relativo ao cwd. Deixa um symlink para o arquivo real.
mkdir -p "${RIE_C}/checkpoints"
ln -sf /workspace/models/checkpoints/depth_pro.pt "${RIE_C}/checkpoints/depth_pro.pt"
ls -lh /workspace/models/checkpoints/depth_pro.pt

echo ">>> SETUP OK. Proximo: run_riemann_data.slurm (baixar Hypersim)."
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind ${RAID}:/workspace \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env PIP_CACHE_DIR=/workspace/pip-cache \
    --env PYTHONUSERBASE=/workspace/python-packages-riemann \
    --env PYTHONPATH=/workspace/python-packages-riemann/lib/python3.10/site-packages:/workspace/python-packages-riemann/lib/python3.11/site-packages \
    --env RIE_C=${RIE_C} \
    --env DP_C=${DP_C} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
