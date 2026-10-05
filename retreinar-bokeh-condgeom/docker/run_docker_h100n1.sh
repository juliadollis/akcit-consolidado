#!/usr/bin/env bash
# =============================================================================
# Sobe o treino da DeblurNet em DOCKER na dgx-H100-01, 4 GPUs, SEM SLURM.
# =============================================================================
# REGRAS SEGUIDAS (DECISOES_FASE2.md e REGISTRO_GEO_COND.md):
#   - `--user $(id -u):$(id -g)` SEMPRE. Sem isso os arquivos nascem root: os
#     54 GB de hf-cache root-owned da rodada anterior nesta maquina vieram disso
#     e a usuaria nao consegue apagar.
#   - Container DETACHED: sobrevive a queda de ssh/VPN (ja aconteceu 2x).
#   - `--gpus` com varias GPUs exige as aspas EMBUTIDAS: '"device=0,1,2,3"'.
#   - Monta o julia_docker INTEIRO em /workspace, nao so a pasta do projeto
#     (montar so o projeto da "No such file or directory").
#   - 4 das 8 GPUs, deixando 4 livres para os outros: nao ha fila protegendo
#     ninguem neste host.
#   - NUNCA `docker rm`/`rmi`/`prune`/`stop` de container alheio. Ha containers
#     de outras 4 pessoas neste host. Este script SO cria; nunca remove nada.
#
# USO (na H100-01):
#   bash docker/run_docker_h100n1.sh              # sobe
#   docker logs -f deblur_paper_4gpu              # acompanha
# =============================================================================
set -euo pipefail

BASE="${BASE:-/raid/user_juliadollis/julia_docker}"
PROJ="${PROJ:-retreinar-deblur}"
IMG="${IMG:-julia-genrefocus:1.0}"
NAME="${NAME:-deblur_paper_4gpu}"
GPUS="${GPUS:-0,1,2,3}"
CFG="${CFG:-configs/train_deblur_docker_4gpu.yaml}"
NPROC="${NPROC:-4}"

# Recusa-se a colidir com um container de mesmo nome ja existente. NAO remove:
# reportar e parar e mais seguro que apagar algo que pode estar treinando.
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "[erro] ja existe um container chamado '${NAME}'."
  echo "       Nao vou remover nada. Confira com: docker ps -a | grep ${NAME}"
  echo "       Se ele estiver morto e voce quiser reusar o nome, remova A MAO."
  exit 1
fi

# Aviso (nao bloqueia): GPUs com memoria de terceiro em uso.
echo "[gpu] estado antes de subir:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | grep -E "^[0-9]+," | awk -F', *' -v l="${GPUS}" \
  'BEGIN{split(l,a,",");for(i in a)q[a[i]]=1} q[$1]{printf "      GPU %s: %s MiB em uso, %s livres\n",$1,$2,$3-$2}'

# O .env nao vive na raiz do julia_docker; procura na lista de candidatos.
ENVF=""
for c in "${ENV_FILE:-}" "${BASE}/.env" "${BASE}/genrefocus_deblurnet_paper/.env" \
         "${BASE}/${PROJ}/.env" "${BASE}/depth-riemannian/.env"; do
  [ -n "$c" ] && [ -f "$c" ] && { ENVF="$c"; break; }
done
[ -n "$ENVF" ] || { echo "[erro] nenhum .env com HF_TOKEN/WANDB_API_KEY encontrado."; exit 1; }
echo "[env] usando ${ENVF}"
set -a; . "${ENVF}"; set +a
: "${HF_TOKEN:?HF_TOKEN ausente no .env}"
: "${WANDB_API_KEY:?WANDB_API_KEY ausente no .env}"

# HF_CACHE — POR QUE hf-cache-v2 E NAO OUTRO. Ha TRES caches nesta maquina e a
# escolha nao e arbitraria:
#
#   hf-cache        54 GB  root:root. So tem o FLUX. Com --user (a regra certa)
#                          o download de dataset novo morre com PermissionError.
#                          NAO apagavel (sem permissao) e NAO escrevivel.
#   hf-cache-julia  284 GB Cache do BokehNet/geo. Escrevivel, MAS so tem o
#                          RealBokeh PARCIAL (2,5 GB dos 46,9 GB). Apontar o
#                          DeblurNet para ca forcaria rebaixar 44 GB.
#   hf-cache-v2     106 GB Cache do DeblurNet. COMPLETO: DDPD + RealBokeh
#                          inteiros + a conversao arrow. O FLUX aqui e um
#                          SYMLINK RELATIVO para o hf-cache (relativo de
#                          proposito: caminho absoluto do host nao resolve
#                          dentro do container, so /workspace e montado).
#
# Duplicacao real entre v2 e julia: ~11,5 GB (o DDPD e o pedaco parcial de
# RealBokeh). NAO crie um quarto cache: use este.
HF_CACHE="${HF_CACHE:-hf-cache-v2}"
mkdir -p "${BASE}/${PROJ}/logs" "${BASE}/${HF_CACHE}/home" "${BASE}/${HF_CACHE}/hub"
if [ ! -w "${BASE}/${HF_CACHE}/hub" ]; then
  echo "[erro] ${BASE}/${HF_CACHE}/hub nao e escrevivel pelo seu usuario."
  echo "       Nao vou mudar dono de nada. Escolha outro com HF_CACHE=<pasta>."
  exit 1
fi
echo "[hf] cache: ${BASE}/${HF_CACHE}"

# `accelerate launch` com N processos; a acumulacao ja esta no YAML de forma
# que batch/GPU x accum x N = 32 (batch efetivo do paper).
APP="
set -euo pipefail
cd /workspace/${PROJ}
export PYTHONPATH=/workspace/${PROJ}:/workspace/${PROJ}/third_party/Genfocus
export PYTHONUNBUFFERED=1
python3 -c 'import torch;print(\"[env] torch\",torch.__version__,\"| GPUs\",torch.cuda.device_count())'
python3 -m genfocus_train.train check --config ${CFG}
exec accelerate launch --multi_gpu --num_machines 1 --num_processes ${NPROC} \
     --mixed_precision bf16 --dynamo_backend no --main_process_port 29601 \
     -m genfocus_train.train deblur --config ${CFG}
"

echo "[docker] subindo '${NAME}' nas GPUs ${GPUS} (imagem ${IMG})"
docker run -d --name "${NAME}" \
  --user "$(id -u):$(id -g)" \
  --gpus "\"device=${GPUS}\"" \
  --ipc=host --shm-size=64g \
  -v "${BASE}":/workspace \
  -e HF_HOME=/workspace/${HF_CACHE} \
  -e HOME=/workspace/${HF_CACHE}/home \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e TOKENIZERS_PARALLELISM=false \
  -w /workspace/"${PROJ}" \
  "${IMG}" bash -lc "${APP}"

echo "[ok] container '${NAME}' de pe."
echo "     acompanhar: docker logs -f ${NAME}"
echo "     se der 'CUDA out of memory', rode:"
echo "       CFG=configs/train_deblur_docker_4gpu_gc.yaml NAME=${NAME}_gc bash docker/run_docker_h100n1.sh"
