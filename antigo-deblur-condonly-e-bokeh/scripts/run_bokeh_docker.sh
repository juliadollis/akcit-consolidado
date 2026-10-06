#!/usr/bin/env bash
# =============================================================================
# FASE 2 do BokehNet na dgx-H100-01 (DOCKER, SEM SLURM). RODE NA PROPRIA MAQUINA.
#
# Esta maquina NAO esta no SLURM (sinfo so lista h100n2/h100n3/b200n1), entao
# nao existe fila protegendo ninguem: o que voce pegar de GPU, voce pegou. Por
# isso o default usa 4 das 8 GPUs e deixa as outras 4 livres.
#
# DIFERENCAS DELIBERADAS vs o antigo scripts/run_docker.sh (nao "simplifique"):
#   1. --user $(id -u):$(id -g)
#      Sem isso, TUDO que o container escreve no /raid sai pertencendo ao root.
#      Foi o que aconteceu na rodada anterior nesta maquina: 54 GB de hf-cache
#      ficaram root:root e a usuaria nao consegue apagar. Com a flag, os
#      checkpoints de 2,6 GB saem dela.
#   2. SEM --network host
#      O bridge padrao ja da internet de saida (HF/wandb). --network host expoe
#      a rede do host sem necessidade.
#   3. Imagem com versoes FIXAS (julia-genrefocus:1.0, ver docker/Dockerfile)
#      em vez de `pip install transformers diffusers peft` sem pin, que e a
#      combinacao que custou 4 jobs falhos (secao 4 do historico-ultimo.md).
#   4. hf-cache-julia (dela) em vez de hf-cache (do root), senao o container
#      rodando como ela nao consegue escrever no cache.
#
# NUNCA rode nesta maquina: `docker system prune`, `docker rmi`, `docker rm` de
# container que nao seja nosso. Ha containers parados de outras 4 pessoas e ~15
# imagens orfas de 11,5 GB no host que NAO sao nossas.
#
# USO:
#   bash scripts/run_bokeh_docker.sh                 # fase 2, 4 GPUs (0-3)
#   RUN_SMOKE=1 bash scripts/run_bokeh_docker.sh     # so o smoke de 3 steps
#   DEVICES=0,1 NUM_GPUS=2 TRAIN_CONFIG=configs/train_bokeh_real_2gpu.yaml \
#     bash scripts/run_bokeh_docker.sh               # 2 GPUs
#
# Acompanhar:  docker logs -f julia_bokeh_fase2
# Parar:       docker stop julia_bokeh_fase2     (resume retoma do checkpoint)
# =============================================================================
set -euo pipefail

BASE=${BASE:-/raid/user_juliadollis/julia_docker}
CODE_DIR=${CODE_DIR:-${BASE}/genrefocus_deblurnet_paper}
HF_CACHE=${HF_CACHE:-${BASE}/hf-cache-julia}

# 4 das 8 GPUs. Dentro do container elas aparecem como 0..3, entao o accelerate
# nao precisa saber quais sao no host.
DEVICES=${DEVICES:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}

# grad_accum=8 no yaml de 4 GPUs => 1 x 8 x 4 = batch efetivo 32 (paper 4.1).
# Se mudar NUM_GPUS, TEM que trocar o config junto, senao o batch efetivo muda.
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_bokeh_real_4gpu.yaml}
INIT_LORA=${INIT_LORA:-outputs/bokehnet_synth_2gpu}
HF_REPO_ID=${HF_REPO_ID:-genrefocus-bokehnet-real-4gpu}

IMAGE=${IMAGE:-julia-genrefocus:1.0}
NAME=${NAME:-julia_bokeh_fase2}
RUN_SMOKE=${RUN_SMOKE:-0}

# --- Segredos: mesmo contrato dos .slurm (NUNCA hardcode) -------------------
export PROJECT_ROOT="${CODE_DIR}"
for _env in "${ENV_FILE:-}" "${PROJECT_ROOT}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env em ${CODE_DIR}/.env}"
: "${WANDB_API_KEY:?defina WANDB_API_KEY no .env em ${CODE_DIR}/.env}"

# --- Checagens antes de queimar GPU -----------------------------------------
[ -f "${CODE_DIR}/genfocus_train/train.py" ] || { echo "ERRO: codigo nao esta em ${CODE_DIR}"; exit 1; }
[ -d "${HF_CACHE}" ] || { echo "ERRO: cache do FLUX nao esta em ${HF_CACHE}"; exit 1; }
[ -w "${HF_CACHE}" ] || { echo "ERRO: ${HF_CACHE} nao e gravavel por voce (dono root?). Use uma copia sua."; exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
    echo "ERRO: imagem ${IMAGE} nao existe. Construa antes:";
    echo "  cd ${CODE_DIR} && docker build -f docker/Dockerfile -t ${IMAGE} docker/"; exit 1; }

if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "ERRO: ja existe um container '${NAME}' RODANDO."
    echo "      Acompanhe: docker logs -f ${NAME}   |   Pare: docker stop ${NAME}"
    exit 1
fi
# Remove SO um container parado com exatamente o NOSSO nome. Nunca toca em outros.
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo ">>> removendo container PARADO anterior chamado '${NAME}'"
    docker rm "${NAME}" >/dev/null
fi

# Aviso (nao bloqueia): as GPUs pedidas estao mesmo livres?
echo ">>> estado das GPUs pedidas (${DEVICES}):"
for g in ${DEVICES//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo "?")
    echo "    GPU ${g}: ${used} MiB em uso"
    if [ "${used}" != "?" ] && [ "${used}" -gt 1000 ] 2>/dev/null; then
        echo "    AVISO: GPU ${g} ja tem ${used} MiB em uso — o treino precisa de ~66 GB dos 80."
    fi
done

echo ">>> subindo '${NAME}' | GPUs=${DEVICES} (${NUM_GPUS}) | cfg=${TRAIN_CONFIG}"
echo ">>> uid:gid = $(id -u):$(id -g)  (arquivos sairao SEUS, nao do root)"

# --gpus precisa das aspas EMBUTIDAS para multiplas GPUs ('"device=0,1,2,3"'),
# senao o Docker quebra na virgula ("cannot set both Count and DeviceIDs").
docker run -d --name "${NAME}" \
    --gpus "\"device=${DEVICES}\"" \
    --user "$(id -u):$(id -g)" \
    --shm-size=32g \
    --restart=no \
    -v "${CODE_DIR}":/workspace/genrefocus_deblurnet_paper \
    -v "${HF_CACHE}":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e NUM_GPUS="${NUM_GPUS}" \
    -e TRAIN_CONFIG="${TRAIN_CONFIG}" \
    -e INIT_LORA="${INIT_LORA}" \
    -e HF_REPO_ID="${HF_REPO_ID}" \
    -e RUN_SMOKE="${RUN_SMOKE}" \
    -e TOKENIZERS_PARALLELISM=false \
    -w /workspace/genrefocus_deblurnet_paper \
    "${IMAGE}" \
    bash scripts/docker_bokeh_bootstrap.sh

echo ""
echo ">>> container '${NAME}' rodando em BACKGROUND (sobrevive a queda de ssh/VPN)."
echo "    log ao vivo:       docker logs -f ${NAME}"
echo "    log desde o inicio: docker logs ${NAME}"
echo "    parar:              docker stop ${NAME}    (o resume retoma do checkpoint)"
echo "    GPUs:               nvidia-smi"
