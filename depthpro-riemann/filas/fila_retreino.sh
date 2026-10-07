#!/usr/bin/env bash
# Fila de retreino dos checkpoints perdidos em 2026-09-10.
#   fila_retreino.sh <gpu> <arquivo_de_fila>
#
# Varias GPUs consomem a MESMA fila em paralelo. A reivindicacao de item e
# atomica via mkdir, que ou cria o diretorio ou falha, sem corrida: duas GPUs
# nunca pegam o mesmo item.
#
# A cada seed que FECHA, sobe para o Hub imediatamente. Nao espera a faixa
# inteira. A regra e "checkpoint que nao esta no Hub nao existe", e uma fila de
# 26 h que so sobe no fim deixa uma janela grande demais.
#
# GUARDA DE MEMORIA, com a licao de 2026-09-14: checar memoria livre protege o
# NOSSO job de tomar OOM, mas nao impede o nosso job de CAUSAR OOM em quem esta
# ciclando na mesma placa. Por isso a guarda exige folga alem do que usamos
# (~53 GB) e so entra em GPU que esteja de fato ociosa.
set -uo pipefail
GPU="${1:?uso: fila_retreino.sh <gpu> <arquivo_de_fila>}"
FILA="${2:?uso: fila_retreino.sh <gpu> <arquivo_de_fila>}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
SAIDA="$B/runs_retreino"
RESERVAS="$B/reservas_retreino"
FX=2585.859
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-70000}
LOG="$B/fila_retreino_gpu${GPU}.log"
mkdir -p "$SAIDA" "$RESERVAS"

espera_gpu () {
  local livre esperas=0
  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
    [ $((esperas % 12)) -eq 0 ] && echo "[fila gpu$GPU] ocupada (${livre} MiB livres, preciso de ${MIN_LIVRE_MIB}); aguardando $(date -u +%FT%H:%M:%SZ)"
    esperas=$((esperas + 1)); sleep 300
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  done
  [ "$esperas" -gt 0 ] && echo "[fila gpu$GPU] liberou (${livre} MiB) apos $((esperas * 5)) min"
  return 0
}

echo "[fila gpu$GPU] inicio $(date -u +%FT%H:%M:%SZ) — $(grep -vc '^#' "$FILA") itens na fila" >> "$LOG"

while IFS='|' read -r NOME QUAL TETO SEED; do
  case "$NOME" in ''|\#*) continue;; esac
  ITEM="${NOME}_seed${SEED}"

  # ja concluido? pula sem reivindicar
  [ -f "$SAIDA/$NOME/seed_$SEED/test_metrics.json" ] && continue
  # reivindicacao atomica: quem criar o diretorio primeiro fica com o item
  mkdir "$RESERVAS/$ITEM" 2>/dev/null || continue

  case "$QUAL" in
    b0) PESOS=(--berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0) ;;
    b1) PESOS=(--berhu 0.7 --normal 0 --gauss 0.45 --grad 0 --geod 0 --metric 0
               --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO") ;;
    b3) PESOS=(--berhu 0.7 --normal 0.9 --gauss 0.45 --grad 0 --geod 0 --metric 0
               --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO") ;;
    *)  echo "[fila gpu$GPU] item invalido: $NOME|$QUAL|$TETO|$SEED" >> "$LOG"; continue ;;
  esac

  espera_gpu
  echo "[fila gpu$GPU] >>> $ITEM  $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  docker rm "julia_fr_gpu${GPU}" >/dev/null 2>&1 || true
  docker run --rm --name "julia_fr_gpu${GPU}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -v "$B/models":/models -v "$SAIDA":/workspace/runs \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /workspace riemann-depthpro:latest \
    python scripts/train_single.py \
      --train-root /data/spring_split/train \
      --val-root   /data/spring_split/val \
      --test-root  /data/spring_split/test \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir "/workspace/runs/$NOME" \
      --seed-inicio "$SEED" --seeds 1 \
      "${PESOS[@]}" \
    >> "$B/fr_${ITEM}.log" 2>&1
  rc=$?
  echo "[fila gpu$GPU] <<< $ITEM rc=$rc $(date -u +%FT%H:%M:%SZ)" >> "$LOG"

  if [ "$rc" -ne 0 ]; then
    # devolve o item para a fila, para outra passada tentar de novo
    rmdir "$RESERVAS/$ITEM" 2>/dev/null || true
    if grep -qa "OutOfMemoryError" "$B/fr_${ITEM}.log"; then
      echo "[fila gpu$GPU] OOM em $ITEM, esperando 10 min" >> "$LOG"; sleep 600
    else
      echo "[fila gpu$GPU] falha nao-OOM em $ITEM (rc=$rc)" >> "$LOG"
    fi
    continue
  fi

  # subida imediata: o peso nao fica so no raid nem por uma hora
  T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
  docker run --rm --name "julia_fr_sobe_${GPU}" --user "$(id -u):$(id -g)" \
    -v "$B":/host -v "$B/hf-cache-julia":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home -e HF_TOKEN="$T" \
    -e FR_NOME="$NOME" -e FR_SEED="$SEED" \
    -w /host julia-genrefocus-eval:3.0 python3 /host/sobe_um_seed.py >> "$LOG" 2>&1
  echo "[fila gpu$GPU] hub rc=$? $ITEM $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
done < "$FILA"

echo "[fila gpu$GPU] FILA COMPLETA $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
