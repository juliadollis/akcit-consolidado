#!/usr/bin/env bash
# Espera a fase de busca terminar e lanca a fase de K FIXO na MESMA GPU 5.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
while [ "$(docker inspect -f '{{.State.Status}}' julia_ca_full_busca 2>/dev/null)" = "running" ]; do
  sleep 60
done
echo "[encadeia] fase de busca terminou em $(date -Is)"
# Regra 2: nao entra na 5 se houver processo de terceiro.
for t in $(seq 1 60); do
  if [ -z "$(nvidia-smi --id=5 --query-compute-apps=pid --format=csv,noheader)" ]; then
    break
  fi
  echo "[encadeia] GPU5 ocupada, esperando ($t)"
  sleep 60
done
if [ -n "$(nvidia-smi --id=5 --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "[encadeia] ABORTA: GPU5 segue ocupada por terceiro. Nada foi lancado."
  exit 1
fi
mkdir -p $B/caminho_a_rotac/out/full50
docker run -d --name julia_ca_full_kfixo \
  --user "$(id -u):$(id -g)" --gpus '"device=5"' --ipc=host --shm-size=32g \
  -v "$B":/workspace -e HOME=/tmp -e HF_HOME=/workspace/hf-cache-julia \
  -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
  -e RAIZ_OUT=/workspace/caminho_a_rotac/out/full50 -e FASE=kfixo -e K_FIXO=266.782 \
  -w /workspace julia-genrefocus:1.0 \
  bash /workspace/caminho_a_rotac/scripts/roda_condicoes.sh
echo "[encadeia] fase kfixo lancada em $(date -Is) com K_FIXO=266.782"
