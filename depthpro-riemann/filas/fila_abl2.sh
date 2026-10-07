#!/usr/bin/env bash
# Fecha as seeds que faltam da ablacao (45, 46, 47) e sobe cada uma para o Hub.
#
# Uma instancia por GPU. A reivindicacao e por `mkdir` na pasta da seed: ou cria,
# ou falha. Duas instancias nunca pegam a mesma seed, e nao ha janela entre
# checar e criar como teria um `if not exists`.
#
# Uso: bash fila_abl2.sh <gpu>
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
GPU="${1:?uso: fila_abl2.sh <gpu>}"
LOG="$B/fila_abl2_gpu${GPU}.log"
QUOTA_PARAR=492   # GiB; limite mole 500, duro 600

echo "[abl2 gpu$GPU] inicio $(date -u +%FT%TZ)" >> "$LOG"

for SEED in 45 46 47; do
  SAIDA="$B/runs_ablacao_seeds/seed_${SEED}"
  mkdir -p "$B/runs_ablacao_seeds" 2>/dev/null
  if ! mkdir "$SAIDA" 2>/dev/null; then
    echo "[abl2 gpu$GPU] seed $SEED ja reivindicada, pulo" >> "$LOG"
    continue
  fi

  usado=$(quota -s 2>/dev/null | tail -1 | awk '{print $2}' | tr -d 'G*')
  if [ -n "$usado" ] && [ "${usado%.*}" -ge "$QUOTA_PARAR" ]; then
    echo "[abl2 gpu$GPU] PARO: quota ${usado}G >= ${QUOTA_PARAR}G" >> "$LOG"
    rmdir "$SAIDA" 2>/dev/null   # devolve a seed para a fila
    break
  fi

  # Entra so se a placa estiver vazia de computacao de terceiros. Guarda de
  # memoria nao basta: ela protege o nosso job de tomar OOM, nao impede o nosso
  # de CAUSAR OOM em quem ja estava.
  e=0
  while true; do
    alheios=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${alheios:-1}" -eq 0 ] && break
    [ $((e % 12)) -eq 0 ] && echo "[abl2 gpu$GPU] $alheios processo(s) na placa; espero $(date -u +%FT%TZ)" >> "$LOG"
    e=$((e+1)); sleep 300
  done

  echo "[abl2 gpu$GPU] >>> seed $SEED $(date -u +%FT%TZ)" >> "$LOG"
  docker rm "julia_abl_seed${SEED}" >/dev/null 2>&1 || true
  docker run --rm --name "julia_abl_seed${SEED}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$B/ablacao-wallisson":/workspace \
    -v "$B/data":/data -v "$B/models":/models -v "$B":/host \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w /workspace \
    riemann-depthpro:latest \
    python3 /workspace/scripts/run_ablation.py \
      --train-root /data/spring_split/train \
      --val-root   /data/spring_split/val \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir    "/host/runs_ablacao_seeds/seed_${SEED}" \
      --focal 2585.859 \
      --filter B0 B1 \
      --seed "$SEED" \
      --no-eval-zero-shot >> "$B/abl_seed_${SEED}.log" 2>&1

  n=$(find "$SAIDA" -name best.pt 2>/dev/null | wc -l)
  echo "[abl2 gpu$GPU] <<< seed $SEED fechou com $n best.pt $(date -u +%FT%TZ)" >> "$LOG"

  # Sobe JA. Checkpoint que nao esta no Hub nao existe: o /raid e area de
  # trabalho, e em 10/09 perdemos 37 de 47 best.pt numa liberacao de quota.
  if [ "$n" -gt 0 ]; then
    T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
    docker run --rm --name "julia_sobe_s${SEED}" --user "$(id -u):$(id -g)" \
      -v "$B":/host -v "$B/hf-cache-julia":/workspace/hf-cache \
      -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home \
      -e HF_TOKEN="$T" -w /host julia-genrefocus-eval:3.0 \
      python3 /host/sobe_ablacao_seed.py "$SEED" >> "$B/sobe_seed_${SEED}.log" 2>&1
    echo "[abl2 gpu$GPU] seed $SEED subiu para o Hub $(date -u +%FT%TZ)" >> "$LOG"
  fi
done

echo "[abl2 gpu$GPU] FIM $(date -u +%FT%TZ)" >> "$LOG"
