#!/usr/bin/env bash
# Lanca uma faixa de seeds de um braco do passo 4.
#   roda_passo4.sh <gpu> <b0|b1|b3> <teto> <seed_inicio> <quantas>
#
# GUARDA DE MEMORIA + RETENTATIVA. Um treino destes ocupa ~80 GB, ou seja, a
# placa inteira. Checar utilizacao nao serve: um deploy de serving fica em 0%
# entre requisicoes com dezenas de GB ja reservados. O que decide e a memoria
# LIVRE, e mesmo assim ha corrida entre a checagem e o docker run -- ja tomamos
# OOM exatamente assim. Por isso o laco tenta de novo em vez de morrer.
#
# Saida em runs_riemann/. O runs_riemann_metricas_preservadas/ e o ARQUIVO da
# campanha anterior e NAO e tocado. train_single.py pula seed ja concluida.
set -uo pipefail
GPU="${1:?gpu}"; QUAL="${2:?b0|b1|b3}"; TETO="${3:?teto}"; INI="${4:?seed inicio}"; N="${5:?quantas}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
FX=2585.859
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-78000}
SIZE=${SIZE:-512}
BATCH=${BATCH:-8}
ACUM=${ACUM:-1}
# 768 px nao cabe com batch 8 nem em 80 GB. Use SIZE=768 BATCH=4 ACUM=2 GRADCKPT=1,
# que mantem o batch efetivo em 8 e troca ~30% de tempo por memoria.
GRADCKPT=${GRADCKPT:-0}
EXTRA=()
[ "$GRADCKPT" = "1" ] && EXTRA+=(--grad-checkpointing)
SUFIXO=""
[ "$SIZE" != "512" ] && SUFIXO="_size${SIZE}"
MAX_TENTATIVAS=${MAX_TENTATIVAS:-40}
case "$QUAL" in
  b0) PESOS=(--berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0)
      NOME="B0_berhu${SUFIXO}" ;;
  b1) PESOS=(--berhu 0.7 --normal 0 --gauss 0.45 --grad 0 --geod 0 --metric 0
             --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO")
      NOME="B1_gauss_metrica_teto${TETO}${SUFIXO}" ;;
  b3) PESOS=(--berhu 0.7 --normal 0.9 --gauss 0.45 --grad 0 --geod 0 --metric 0
             --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO")
      NOME="B3_gauss_metrica_teto${TETO}${SUFIXO}" ;;
  *)  echo "braco invalido: $QUAL"; exit 2 ;;
esac
mkdir -p "$B/runs_riemann"
LOG="$B/p4_${NOME}_s${INI}-$((INI+N-1)).log"

for tentativa in $(seq 1 "$MAX_TENTATIVAS"); do
  # todas as seeds pedidas ja concluidas? entao nao ha o que fazer
  falta=0
  for s in $(seq "$INI" $((INI+N-1))); do
    [ -f "$B/runs_riemann/$NOME/seed_$s/test_metrics.json" ] || falta=1
  done
  [ "$falta" -eq 0 ] && { echo "[p4] $NOME seeds ${INI}..$((INI+N-1)) ja concluidas"; exit 0; }

  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  esperas=0
  while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
    [ "$esperas" -eq 0 ] && echo "[p4] gpu$GPU ocupada (${livre} MiB livres, preciso de ${MIN_LIVRE_MIB}); aguardando $(date -u +%H:%M:%SZ)"
    esperas=$((esperas+1)); sleep 300
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  done
  [ "$esperas" -gt 0 ] && echo "[p4] gpu$GPU liberou (${livre} MiB) apos $((esperas*5)) min"

  echo "[p4] tentativa $tentativa: $NOME seeds ${INI}..$((INI+N-1)) na gpu$GPU $(date -u +%FT%H:%M:%SZ) log=$LOG"
  docker rm "julia_p4_${NOME}_s${INI}" >/dev/null 2>&1 || true
  docker run --rm --name "julia_p4_${NOME}_s${INI}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -v "$B/models":/models -v "$B/runs_riemann":/workspace/runs \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /workspace riemann-depthpro:latest \
    python scripts/train_single.py \
      --train-root /data/spring_split/train \
      --val-root   /data/spring_split/val \
      --test-root  /data/spring_split/test \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir "/workspace/runs/$NOME" \
      --seed-inicio "$INI" --seeds "$N" \
      --size "$SIZE" --batch-size "$BATCH" --grad-accum "$ACUM" \
      "${EXTRA[@]}" "${PESOS[@]}" \
    >> "$LOG" 2>&1
  rc=$?
  echo "[p4] tentativa $tentativa rc=$rc $(date -u +%FT%H:%M:%SZ)"
  [ "$rc" -eq 0 ] && break
  if grep -qa "OutOfMemoryError" "$LOG"; then
    echo "[p4] OOM: outra carga pegou a gpu$GPU. Esperando 10 min e tentando de novo."
    sleep 600
  else
    echo "[p4] falha nao-OOM (rc=$rc). Parando para nao queimar GPU as cegas."
    exit "$rc"
  fi
done
echo "[p4] FIM $NOME $(date -u +%FT%H:%M:%SZ)"
