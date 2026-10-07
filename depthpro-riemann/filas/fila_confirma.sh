#!/usr/bin/env bash
# Fila de confirmacao dos termos grad, metric e geod, com o protocolo NOSSO.
#   fila_confirma.sh <gpu> <arquivo_de_fila>
# Reivindicacao atomica por mkdir; varias GPUs consomem a mesma fila sem colidir.
# Sobe para o Hub a cada seed que fecha.
set -uo pipefail
GPU="${1:?}"; FILA="${2:?}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
SAIDA="$B/runs_confirma"; RES="$B/reservas_confirma"
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-70000}
LOG="$B/fila_confirma_gpu${GPU}.log"
mkdir -p "$SAIDA" "$RES"

while IFS='|' read -r NOME PESOS SEED; do
  case "$NOME" in ''|\#*) continue;; esac
  ITEM="${NOME}_seed${SEED}"
  [ -f "$SAIDA/$NOME/seed_$SEED/test_metrics.json" ] && continue
  mkdir "$RES/$ITEM" 2>/dev/null || continue

  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  e=0
  while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
    [ $((e % 12)) -eq 0 ] && echo "[conf gpu$GPU] ocupada (${livre} MiB); aguardando $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
    e=$((e+1)); sleep 300
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  done

  echo "[conf gpu$GPU] >>> $ITEM $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  docker rm "julia_conf_gpu${GPU}" >/dev/null 2>&1 || true
  docker run --rm --name "julia_conf_gpu${GPU}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -v "$B/models":/models -v "$SAIDA":/workspace/runs \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /workspace riemann-depthpro:latest \
    python scripts/train_single.py \
      --train-root /data/spring_split/train --val-root /data/spring_split/val \
      --test-root /data/spring_split/test \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir "/workspace/runs/$NOME" --seed-inicio "$SEED" --seeds 1 \
      $PESOS >> "$B/conf_${ITEM}.log" 2>&1
  rc=$?
  echo "[conf gpu$GPU] <<< $ITEM rc=$rc $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    rmdir "$RES/$ITEM" 2>/dev/null || true
    grep -qa "OutOfMemoryError" "$B/conf_${ITEM}.log" && sleep 600
    continue
  fi
  T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
  docker run --rm --user "$(id -u):$(id -g)" -v "$B":/host \
    -v "$B/hf-cache-julia":/workspace/hf-cache -e HF_HOME=/workspace/hf-cache \
    -e HOME=/workspace/hf-cache/home -e HF_TOKEN="$T" \
    -e FR_RAIZ=/host/runs_confirma -e FR_NOME="$NOME" -e FR_SEED="$SEED" \
    -e FR_DEST=pesos_confirma -w /host julia-genrefocus-eval:3.0 \
    python3 /host/sobe_um_seed2.py >> "$LOG" 2>&1
done < "$FILA"
echo "[conf gpu$GPU] FILA COMPLETA $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
