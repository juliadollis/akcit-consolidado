#!/usr/bin/env bash
# =============================================================================
# Sobe um container PERSISTENTE e INTERATIVO (GPUs via DEVICES) e te diz como
# entrar. Você roda os comandos DENTRO e vê os logs ali mesmo — do jeito clássico.
# RODE NA MÁQUINA (host, ex.: dgx-H100-01).
#
#   BASE=/raid/user_juliadollis/julia_docker bash scripts/docker_shell.sh
#   DEVICES=2,3 NAME=deblurnet bash scripts/docker_shell.sh
# =============================================================================
set -euo pipefail

BASE=${BASE:-/raid/user_juliadollis/julia_docker}
DEVICES=${DEVICES:-4,5}                               # IDs das GPUs livres (nvidia-smi)
NAME=${NAME:-deblurnet}
IMAGE=${IMAGE:-pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel}

# --- Segredos (NUNCA hardcode) ----------------------------------------------
# Ordem de precedencia: ja exportado no ambiente > $ENV_FILE > .env na raiz do
# projeto. Crie o .env a partir de .env.example (ele e git-ignored):
#   cp .env.example .env && ${EDITOR:-nano} .env
for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT:-}/.env" "${PROJECT_DIR:-}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env da raiz do projeto (cp .env.example .env)}"
: "${WANDB_API_KEY:?defina WANDB_API_KEY no .env da raiz do projeto}"
export HF_TOKEN WANDB_API_KEY

CODE_DIR="${BASE}/genrefocus_deblurnet"
if [ ! -f "${CODE_DIR}/genfocus_train/train.py" ]; then
    echo "ERRO: código não encontrado em ${CODE_DIR} (copie do Mac via rsync primeiro)."; exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo ">>> container '${NAME}' já existe. Entrar:"
    echo "      docker exec -it ${NAME} bash"
    echo "    (recriar do zero:  docker rm -f ${NAME} && bash scripts/docker_shell.sh)"
    exit 0
fi

mkdir -p "${BASE}/hf-cache"

# -dit = persistente + interativo. As aspas EMBUTIDAS no --gpus são obrigatórias
# p/ múltiplas GPUs (senão dá "cannot set both Count and DeviceIDs").
docker run -dit --name "${NAME}" \
    --gpus "\"device=${DEVICES}\"" \
    --shm-size=32g \
    --network host \
    -v "${CODE_DIR}":/workspace/genrefocus_deblurnet \
    -v "${BASE}/hf-cache":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTHONUNBUFFERED=1 \
    -w /workspace/genrefocus_deblurnet \
    "${IMAGE}" \
    bash

echo ""
echo ">>> container '${NAME}' no ar (GPUs ${DEVICES}). Agora entre nele:"
echo "      docker exec -it ${NAME} bash"
echo ">>> parar/remover depois:  docker rm -f ${NAME}"
