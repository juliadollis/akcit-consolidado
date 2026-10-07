#!/usr/bin/env bash
# Valida se a correcao do padding tornou o caminho geometrico reproduzivel.
#   valida_determinismo.sh <gpu>
#
# Roda a MESMA seed do MESMO braco DUAS vezes, com o codigo corrigido, e compara.
# Se bater exato, como o berHu puro ja bate, o nao determinismo acabou e toda
# ablacao futura passa a precisar de metade das seeds.
#
# Usa o braco B1 teto 5, que e curvatura ISOLADA: e o caminho mais curto ate o
# surface_curvatures, entao e o teste mais sensivel.
# Poucas epocas de proposito: se divergir, diverge nas primeiras.
set -uo pipefail
GPU="${1:?uso: valida_determinismo.sh <gpu>}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian-det"
SAIDA="$B/runs_determinismo"
LOG="$B/valida_determinismo.log"
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-70000}
EPOCAS=${EPOCAS:-5}
# Os subdiretorios precisam existir ANTES do docker run: se o alvo do bind mount
# nao existe, o Docker o cria como ROOT e o container, rodando com --user, nao
# consegue escrever. Foi exatamente o que derrubou a primeira tentativa.
mkdir -p "$SAIDA/rodada_A/B1_det" "$SAIDA/rodada_B/B1_det"

espera () {
  local livre e=0
  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
    [ $((e % 12)) -eq 0 ] && echo "[det] gpu$GPU ocupada (${livre} MiB); aguardando $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
    e=$((e+1)); sleep 300
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  done
}

for rodada in A B; do
  [ -f "$SAIDA/rodada_$rodada/B1_det/seed_0/test_metrics.json" ] && continue
  espera
  echo "[det] rodada $rodada $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  docker rm "julia_det_$rodada" >/dev/null 2>&1 || true
  docker run --rm --name "julia_det_$rodada" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -v "$B/models":/models -v "$SAIDA/rodada_$rodada":/workspace/runs \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /workspace riemann-depthpro:latest \
    python scripts/train_single.py \
      --train-root /data/spring_split/train --val-root /data/spring_split/val \
      --test-root /data/spring_split/test \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir /workspace/runs/B1_det --seed-inicio 0 --seeds 1 --epochs "$EPOCAS" \
      --berhu 0.7 --normal 0 --gauss 0.45 --grad 0 --geod 0 --metric 0 \
      --gauss-metrica --fx-orig 2585.859 --gauss-clamp 5 \
    >> "$B/det_rodada_$rodada.log" 2>&1
  echo "[det] rodada $rodada rc=$? $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
done

docker run --rm --user "$(id -u):$(id -g)" -v "$B":/host -e HOME=/tmp -w /host \
  riemann-depthpro:latest python3 /host/compara_determinismo.py >> "$LOG" 2>&1
echo "[det] FIM $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
