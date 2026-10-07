#!/usr/bin/env bash
# Retreina o B0 berHu 512, o melhor braco da campanha, cujos pesos foram apagados.
#   retreina_b0.sh <gpu> <seed_inicio> <quantas>
#
# O objetivo e ficar O MAIS IGUAL POSSIVEL ao original, entao NAO mudamos nada:
# mesmo codigo (byte a byte igual ao origin/main), mesmo split (593/184/485,
# seed 42, mesmas sequencias), mesmos pesos de perda, mesmo batch 8, mesma
# resolucao 512, mesmo monitor e mesmo early stop.
#
# NAO ligamos torch.use_deterministic_algorithms de proposito: liga-lo mudaria a
# computacao em relacao as rodadas originais, que e justamente o que queremos
# evitar. A consequencia e que os numeros NAO vao bater bit a bit.
#
# A saida vai para runs_b0_retreino/, pasta NOVA. Nada existente e tocado, e os
# numeros originais continuam sendo os do relatorio.
#
# Ao fim de cada faixa, sobe para o Hub. Regra: checkpoint que nao esta no Hub
# nao existe.
set -uo pipefail
GPU="${1:?gpu}"; INI="${2:?seed inicio}"; N="${3:?quantas}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-78000}
NOME=B0_berhu
SAIDA="$B/runs_b0_retreino"
LOG="$B/retreino_b0_gpu${GPU}_s${INI}-$((INI+N-1)).log"
mkdir -p "$SAIDA"

for tentativa in $(seq 1 60); do
  falta=0
  for s in $(seq "$INI" $((INI+N-1))); do
    [ -f "$SAIDA/$NOME/seed_$s/test_metrics.json" ] || falta=1
  done
  if [ "$falta" -eq 0 ]; then echo "[retreino] seeds ${INI}..$((INI+N-1)) prontas"; break; fi

  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  esperas=0
  while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
    [ $((esperas % 12)) -eq 0 ] && echo "[retreino] gpu$GPU ocupada (${livre} MiB livres, preciso de ${MIN_LIVRE_MIB}); aguardando $(date -u +%FT%H:%M:%SZ)"
    esperas=$((esperas+1)); sleep 300
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  done
  [ "$esperas" -gt 0 ] && echo "[retreino] gpu$GPU liberou (${livre} MiB) apos $((esperas*5)) min"

  echo "[retreino] tentativa $tentativa: seeds ${INI}..$((INI+N-1)) na gpu$GPU $(date -u +%FT%H:%M:%SZ)"
  docker rm "julia_retreino_gpu${GPU}" >/dev/null 2>&1 || true
  docker run --rm --name "julia_retreino_gpu${GPU}" --gpus "\"device=${GPU}\"" \
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
      --seed-inicio "$INI" --seeds "$N" \
      --berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0 \
    >> "$LOG" 2>&1
  rc=$?
  echo "[retreino] tentativa $tentativa rc=$rc $(date -u +%FT%H:%M:%SZ)"
  [ "$rc" -eq 0 ] && break
  if grep -qa "OutOfMemoryError" "$LOG"; then
    echo "[retreino] OOM, esperando 10 min"; sleep 600
  else
    echo "[retreino] falha nao-OOM (rc=$rc), parando"; exit "$rc"
  fi
done

echo "[retreino] subindo para o Hub $(date -u +%FT%H:%M:%SZ)"
T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
docker run --rm --name "julia_sobe_retreino_${GPU}" --user "$(id -u):$(id -g)" \
  -v "$B":/host -v "$B/hf-cache-julia":/workspace/hf-cache \
  -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home -e HF_TOKEN="$T" \
  -w /host julia-genrefocus-eval:3.0 python3 /host/sobe_retreino.py >> "$LOG" 2>&1
echo "[retreino] FIM gpu$GPU $(date -u +%FT%H:%M:%SZ)"
