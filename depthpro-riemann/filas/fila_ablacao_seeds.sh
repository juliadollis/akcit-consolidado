#!/usr/bin/env bash
# Leva a ablacao de n=1 para n=6.
#
# Por que: a ablacao e a unica coisa que testa os cinco termos geometricos
# ISOLADOS, um treino por termo. Mas ela tem n=1, e ja apontou para lado
# diferente da campanha de seeds tres vezes (no Spring, no DIODE interno e no
# externo). Com uma execucao por braco nao da para saber se isso e efeito do
# termo ou variacao de sorteio -- ainda mais nos bracos com curvatura, onde o
# |delta| entre execucoes da MESMA seed e 0,0148 medio e 0,0389 maximo.
#
# O que roda: --filter B0 B1 = o controle berHu mais os cinco termos sozinhos.
# Seeds 43 a 47, somadas a seed 42 que ja existe, fecham n=6 -- o mesmo n da
# campanha de seeds, que e o padrao que a Julia fixou.
#
# Codigo EXATAMENTE como o Wallisson entregou, mesmos parametros do run
# original. So muda a seed. Aplicar aqui a correcao de determinismo do
# depth-riemannian-det misturaria duas variaveis e tornaria a seed 42 existente
# incomparavel com as novas.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
GPU=6
LOG="$B/fila_ablacao_seeds.log"
QUOTA_PARAR=490   # em GiB; o limite mole e 500

echo "[abl-seeds] inicio $(date -u +%FT%TZ)" >> "$LOG"

for SEED in 43 44 45 46 47; do
  SAIDA="$B/runs_ablacao_seeds/seed_${SEED}"

  # reivindicacao atomica: mkdir ou cria, ou falha. Sem janela.
  if ! mkdir -p "$(dirname "$SAIDA")" 2>/dev/null || ! mkdir "$SAIDA" 2>/dev/null; then
    echo "[abl-seeds] seed $SEED ja reivindicada, pulo" >> "$LOG"
    continue
  fi

  # quota: cada best.pt tem 1,37 GB e sao 6 por seed (~8,2 GB)
  usado=$(quota -s 2>/dev/null | tail -1 | awk '{print $2}' | tr -d 'G')
  if [ -n "$usado" ] && [ "${usado%.*}" -ge "$QUOTA_PARAR" ]; then
    echo "[abl-seeds] PARO: quota ${usado}G >= ${QUOTA_PARAR}G $(date -u +%FT%TZ)" >> "$LOG"
    break
  fi

  # so entra se a placa estiver VAZIA. Guarda de memoria nao basta: protege o
  # nosso job de tomar OOM, nao impede o nosso de CAUSAR OOM em quem ja estava.
  e=0
  while true; do
    alheios=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${alheios:-1}" -eq 0 ] && break
    [ $((e % 12)) -eq 0 ] && echo "[abl-seeds] gpu$GPU com $alheios processo(s); espero $(date -u +%FT%TZ)" >> "$LOG"
    e=$((e+1)); sleep 300
  done

  echo "[abl-seeds] >>> seed $SEED $(date -u +%FT%TZ)" >> "$LOG"
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
  echo "[abl-seeds] <<< seed $SEED fechou com $n best.pt $(date -u +%FT%TZ)" >> "$LOG"
done

echo "[abl-seeds] FIM $(date -u +%FT%TZ)" >> "$LOG"
