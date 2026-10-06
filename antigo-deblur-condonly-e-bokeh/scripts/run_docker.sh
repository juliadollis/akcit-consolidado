#!/usr/bin/env bash
# =============================================================================
# Lança o treino no DOCKER (ex.: dgx-H100-01, sem SLURM). RODE NA MÁQUINA (host).
#
# Pré-requisito: o código já copiado em ${BASE}/genrefocus_deblurnet (rsync do
# Mac). O FLUX é baixado pro cache na 1ª vez (dentro do container).
#
#   BASE=/caminho/com/espaco bash scripts/run_docker.sh        # GPUs 4,5 (default)
#   DEVICES=2,3,4 NUM_GPUS=3 TRAIN_CONFIG=configs/train_base_3gpu.yaml BASE=... bash scripts/run_docker.sh
#
# Sobe o container destacado (-d). Acompanhe com `docker logs -f`.
# =============================================================================
set -euo pipefail

# >>> ajuste: pasta com ESPAÇO (~70GB p/ código + cache do FLUX) <<<
BASE=${BASE:-/raid/user_juliadollis/julia_docker}

DEVICES=${DEVICES:-4,5}                               # IDs das GPUs livres (nvidia-smi)
NUM_GPUS=${NUM_GPUS:-2}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_base_2gpu.yaml}
IMAGE=${IMAGE:-pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel}
RUN_SMOKE=${RUN_SMOKE:-0}
UPLOAD_TO_HF=${UPLOAD_TO_HF:-1}
HF_REPO_ID=${HF_REPO_ID:-genrefocus-deblurnet-paper-2gpu}
NAME=${NAME:-deblurnet-train}                        # nome do container (docker logs/stop)

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
    echo "ERRO: código não encontrado em ${CODE_DIR}."
    echo "      Copie do Mac primeiro, ex.:"
    echo "      rsync -ah --exclude outputs/ --exclude third_party/ --exclude '__pycache__/' \\"
    echo "        /Users/juliadollis/Projects_Code/repositorio_ref/genrefocus_deblurnet_paper/ \\"
    echo "        dgx-H100-01:${CODE_DIR}/"
    exit 1
fi

mkdir -p "${BASE}/hf-cache"

# Container destacado (background), como os outros do servidor. Sem tmux.
# Se já existe um container com esse nome: bloqueia se estiver RODANDO,
# remove se estiver parado (pra poder relançar/resumir).
if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "ERRO: já existe um container '${NAME}' RODANDO."
    echo "      Acompanhe:  docker logs -f ${NAME}"
    echo "      Ou pare:    docker stop ${NAME}   (e rode de novo p/ resumir)"
    exit 1
fi
docker rm "${NAME}" >/dev/null 2>&1 || true

echo ">>> subindo container '${NAME}' | GPUs=${DEVICES} | cfg=${TRAIN_CONFIG} | base=${BASE}"

# --gpus precisa das aspas EMBUTIDAS p/ múltiplas GPUs ('"device=4,5"'),
# senão o Docker quebra na vírgula e dá "cannot set both Count and DeviceIDs".
docker run -d --name "${NAME}" \
    --gpus "\"device=${DEVICES}\"" \
    --shm-size=32g \
    --network host \
    --restart=no \
    -v "${CODE_DIR}":/workspace/genrefocus_deblurnet \
    -v "${BASE}/hf-cache":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e NUM_GPUS="${NUM_GPUS}" \
    -e TRAIN_CONFIG="${TRAIN_CONFIG}" \
    -e RUN_SMOKE="${RUN_SMOKE}" \
    -e UPLOAD_TO_HF="${UPLOAD_TO_HF}" \
    -e HF_REPO_ID="${HF_REPO_ID}" \
    -e TOKENIZERS_PARALLELISM=false \
    -w /workspace/genrefocus_deblurnet \
    "${IMAGE}" \
    bash scripts/docker_bootstrap.sh

echo ""
echo ">>> container '${NAME}' rodando em BACKGROUND. Comandos:"
echo "    acompanhar (log ao vivo):  docker logs -f ${NAME}"
echo "    ver tudo desde o início:   docker logs ${NAME}"
echo "    parar:                     docker stop ${NAME}"
echo "    status:                    docker ps   |   nvidia-smi"
echo "    (relançar/resumir depois:  bash scripts/run_docker.sh)"
