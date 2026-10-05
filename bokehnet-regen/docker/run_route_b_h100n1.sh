#!/usr/bin/env bash
# =============================================================================
# Rota B em DOCKER na dgx-H100-01, 2 GPUs, SEM SLURM — com a NOSSA DeblurNet.
# =============================================================================
# REGRAS SEGUIDAS (retreinar-deblur/docker/run_docker_h100n1.sh, DECISOES_FASE2.md):
#   - `--user $(id -u):$(id -g)` SEMPRE. Sem isso os arquivos nascem root: os 54 GB
#     de hf-cache root-owned desta máquina vieram exatamente disso, e a usuária não
#     consegue apagá-los.
#   - Container DETACHED: sobrevive a queda de ssh/VPN (já aconteceu 2x).
#   - `--gpus` com várias GPUs exige as aspas EMBUTIDAS: '"device=1,3"'.
#   - Monta o julia_docker INTEIRO em /workspace, não só a pasta do projeto.
#   - 2 das 8 GPUs. Não há fila protegendo ninguém neste host, e há containers de
#     outras pessoas rodando; deixar folga é regra.
#   - NUNCA `docker rm`/`rmi`/`prune`/`stop` de container alheio. Este script SÓ
#     cria; nunca remove nada. Colisão de nome é erro, não motivo para remover.
#
# O QUE ELE FAZ, e por que é a variante NOSSA e não a oficial:
#   A h100n3 já está gerando a rota B com a DeblurNet OFICIAL (jobs 32487/32488).
#   Aqui roda a variante `ours_cond_only` com o checkpoint local de step 60.000.
#   As duas versões do dataset saem com o MESMO `sample_id`, que é o que permite
#   comparar par a par quanto a AIF influencia o rótulo.
#
#   O peso não distingue as variantes — medido: os dois `.safetensors` têm as
#   mesmas 686 chaves, e a diferença vive no roteamento (`main_adapter`). Por isso
#   a tripla (repo, arquivo, adapter) é indivisível no código, e aqui só se escolhe
#   a variante; o adapter vem dela.
#
# USO (na H100-01):
#   bash docker/run_route_b_h100n1.sh
#   docker logs -f rota_b_nossa_s0
# =============================================================================
set -euo pipefail

BASE="${BASE:-/raid/user_juliadollis/julia_docker}"
PROJ="${PROJ:-bokehnet-regen}"
IMG="${IMG:-julia-genrefocus:1.0}"
GPUS="${GPUS:-1,3}"
NUM_SHARDS="${NUM_SHARDS:-2}"
# ours_cond_only: o retreino de 2026-09 (lora_on_main: false), que e o peso
# que existe em disco. O default era ours_main_cond e ESSE era o defeito.
VARIANTE="${VARIANTE:-ours_cond_only}"
PILOT_N="${PILOT_N:-13800}"

# hf-cache-julia e não v2: é onde vivem os pesos OFICIAIS já baixados e para onde a
# BokehDiffusion estava sendo buscada. É escrevível pelo usuário; o `hf-cache` puro
# é root-owned e o download de dataset novo morreria com PermissionError.
HF_CACHE="${HF_CACHE:-hf-cache-julia}"

# O FLUX pode viver em qualquer um dos caches: o `hf-cache-v2` tinha um SYMLINK para o
# `hf-cache`, e os dois já sumiram e voltaram durante a limpeza. Procura em todos e usa
# o primeiro snapshot real que achar, em vez de cravar um caminho que some.
if [ -z "${FLUX_DIR:-}" ]; then
  for c in hf-cache hf-cache-v2 hf-cache-julia; do
    d=$(ls -d ${BASE}/${c}/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
    [ -n "$d" ] && [ -f "$d/model_index.json" ] && { FLUX_DIR="$d"; break; }
  done
fi
[ -n "${FLUX_DIR:-}" ] || { echo "[erro] FLUX.1-dev não encontrado em nenhum cache."; exit 1; }
echo "[flux] ${FLUX_DIR}"
GENFOCUS_DIR="${GENFOCUS_DIR:-${BASE}/retreinar-deblur/third_party/Genfocus}"
PESOS_DIR="${PESOS_DIR:-${BASE}/retreinar-deblur/outputs/deblur_docker_4gpu/deblur}"

for p in "${FLUX_DIR}" "${GENFOCUS_DIR}" "${PESOS_DIR}" "${BASE}/${PROJ}/src"; do
  [ -e "${p}" ] || { echo "[erro] não encontrado: ${p}"; exit 1; }
done
# O nome do arquivo faz parte da VARIANTE, e não é o mesmo nas duas: a nossa é
# `deblur.safetensors` e a oficial é `deblurNet.safetensors`. A tripla (repo, arquivo,
# adapter) é indivisível — conferir o arquivo errado aqui rejeitaria a variante certa.
# ours_cond_only e ours_main_cond COMPARTILHAM o nome do arquivo. O que os separa e o
# sha256, fixado na spec (deblurnet.py). O peso que existe nesta maquina e o cond-only.
case "${VARIANTE}" in
  ours_cond_only)      ARQ_PESO="deblur.safetensors" ;;
  ours_main_cond)      ARQ_PESO="deblur.safetensors" ;;
  official_cond_only)  ARQ_PESO="deblurNet.safetensors" ;;
  *) echo "[erro] variante desconhecida: ${VARIANTE}"; exit 1 ;;
esac
[ -f "${PESOS_DIR}/${ARQ_PESO}" ] || {
  echo "[erro] ${PESOS_DIR}/${ARQ_PESO} ausente (variante ${VARIANTE})."
  exit 1; }
echo "[peso] ${VARIANTE} -> ${PESOS_DIR}/${ARQ_PESO}"

# ours_main_cond nao fixa sha256 na spec (aquele peso nao esta nesta maquina), e
# `deblur.safetensors` e o nome das DUAS variantes nossas. Sem o hash declarado o
# runtime RECUSA de proposito — mas recusaria LA DENTRO do container, e a mensagem
# ficaria no `docker logs`. Recusa aqui, no stdout do host, antes de criar container.
LORA_SHA="${LORA_SHA:-}"
if [ "${VARIANTE}" = "ours_main_cond" ] && [ -z "${LORA_SHA}" ]; then
  echo "[erro] a variante ours_main_cond exige o sha256 do peso, e nada o declara."
  echo "       'deblur.safetensors' e o nome do treino main+cond ANTIGO"
  echo "       (juliadollis/genrefocus-deblurnet-paper-4gpu, sha deea4988...) E do"
  echo "       cond-only novo (juliadollis/genrefocus-deblurnet, sha a1ed05e4...)."
  echo "       O nome nao distingue os dois: foi assim que o lote b_release inteiro"
  echo "       (13.615 amostras) saiu com o peso cond-only rodado como main+cond,"
  echo "       sem uma unica excecao. Nao vou adivinhar qual arquivo esta ai."
  echo ""
  echo "       Se e mesmo o main+cond antigo:"
  echo "         LORA_SHA=<sha256 completo> VARIANTE=ours_main_cond bash $0"
  echo "       O peso que existe NESTA maquina e o cond-only: a variante certa para"
  echo "       ele e ours_cond_only (o default), que ja e fixada por sha na spec."
  exit 1
fi
[ -n "${LORA_SHA}" ] && echo "[peso] sha256 declarado: ${LORA_SHA}"

# .env com HF_TOKEN — o dataset da ITW ainda precisa ser baixado nesta máquina.
ENVF=""
for c in "${ENV_FILE:-}" "${BASE}/genrefocus_deblurnet_paper/.env" "${BASE}/.env" \
         "${BASE}/${PROJ}/.env"; do
  [ -n "$c" ] && [ -f "$c" ] && { ENVF="$c"; break; }
done
[ -n "$ENVF" ] || { echo "[erro] nenhum .env com HF_TOKEN encontrado."; exit 1; }
set -a; . "${ENVF}"; set +a
: "${HF_TOKEN:?HF_TOKEN ausente no .env}"

[ -w "${BASE}/${HF_CACHE}/hub" ] || {
  echo "[erro] ${BASE}/${HF_CACHE}/hub não é escrevível pelo seu usuário."
  echo "       Não vou mudar dono de nada. Escolha outro com HF_CACHE=<pasta>."; exit 1; }

echo "[gpu] estado antes de subir:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | grep -E "^[0-9]+," | awk -F', *' -v l="${GPUS}" \
  'BEGIN{split(l,a,",");for(i in a)q[a[i]]=1} q[$1]{printf "      GPU %s: %s MiB em uso, %s livres\n",$1,$2,$3-$2}'

# SHARD0: índice GLOBAL da primeira fatia. Antes o índice vinha da ordem das GPUs e
# reiniciava em 0 a cada invocação — duas chamadas produziam "fatia 0" duas vezes,
# escrevendo na MESMA pasta com recortes diferentes. O manifesto saía misturado.
SHARD0="${SHARD0:-0}"
i=${SHARD0}
for gpu in $(echo "${GPUS}" | tr ',' ' '); do
  if [ "${i}" -ge "${NUM_SHARDS}" ]; then
    echo "[erro] fatia ${i} >= NUM_SHARDS=${NUM_SHARDS}. Ajuste SHARD0 ou NUM_SHARDS."
    exit 1
  fi
  NAME="${PREFIXO:-rota_b_nossa}_s${i}"

  # Recusa colidir. NÃO remove: reportar e parar é mais seguro que apagar algo que
  # pode estar rodando — há containers de outras 4 pessoas neste host.
  if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "[erro] já existe um container chamado '${NAME}'. Não vou remover nada."
    echo "       Confira com: docker ps -a | grep ${NAME}"
    exit 1
  fi

  OUT="/workspace/${PROJ}/output/${SAIDA:-b_full_nossa}_s${i}_de${NUM_SHARDS}"
  # `--models-dir /workspace/models`: o depth_pro.pt vive em models/checkpoints e o
  # BiRefNet entrou como symlink RELATIVO para o snapshot em hf-cache-julia — relativo
  # de propósito, porque caminho absoluto do host não resolve dentro do container.
  APP="
set -euo pipefail
cd /workspace/${PROJ}
# pylibs_depth_pro prefixa: a imagem julia-genrefocus:1.0 NAO traz o depth_pro
# (registrado em REGISTRO_GEO_COND.md). As libs vieram do proprio /home desta maquina,
# copiadas com cp -an, no molde do pylibs_birefnet que ja existia aqui.
# SEM CRASES: este comentario vive dentro de uma string de aspas duplas, e crase ali
# vira substituicao de comando — foi o que quebrou a primeira versao.
export PYTHONPATH=/workspace/pylibs_depth_pro:/workspace/${PROJ}/src:/workspace/${PROJ}/scripts:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
python3 -c 'import torch;print(\"[env] torch\",torch.__version__,\"| GPUs\",torch.cuda.device_count())'
mkdir -p ${OUT}
python3 scripts/run_route_b.py --source itw \
    --shard ${i} --num-shards ${NUM_SHARDS} --pilot ${PILOT_N} \
    --output-dir ${OUT} \
    --deblur-variant ${VARIANTE} \
    ${LORA_SHA:+--deblur-lora-sha256 ${LORA_SHA}} \
    --deblur-weights-dir ${PESOS_DIR/${BASE}//workspace} \
    --flux-dir ${FLUX_DIR/${BASE}//workspace} \
    --genfocus-dir /workspace/retreinar-deblur/third_party/Genfocus \
    --models-dir /workspace/models \
    --device cuda --seed 0
"

  echo "[docker] subindo '${NAME}' na GPU ${gpu} -> fatia ${i} de ${NUM_SHARDS} -> ${OUT}"
  docker run -d --name "${NAME}" \
    --user "$(id -u):$(id -g)" \
    --gpus "\"device=${gpu}\"" \
    --ipc=host --shm-size=32g \
    -v "${BASE}":/workspace \
    -e HF_HOME=/workspace/${HF_CACHE} \
    -e HOME=/workspace/${HF_CACHE}/home \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
    -e TOKENIZERS_PARALLELISM=false \
    -w /workspace/"${PROJ}" \
    "${IMG}" bash -lc "${APP}"
  i=$((i+1))
done

echo "[ok] ${i} container(s) de pé, variante ${VARIANTE}."
echo "     acompanhar: docker logs -f rota_b_nossa_s0"
