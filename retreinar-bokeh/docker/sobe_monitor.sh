#!/usr/bin/env bash
# =============================================================================
# Monitor EXTERNO de controlabilidade (LVCorr) — GPU e container separados.
# =============================================================================
# O princípio, que é da usuária: o treino começa e termina do mesmo jeito.
# Instrumento é OBSERVADOR, não participante. Este processo só lê os `step_*.pt`
# que o treino já grava e escreve num JSONL próprio; não toca em RNG, optimizer,
# pesos nem coletivo. Se ele morrer, o treino nem percebe — já foi testado (ele
# tomou OOM quando terceiros encheram a GPU e o treino seguiu intacto).
#
# O PROBE SET É O MESMO DA FASE 1, DE PROPÓSITO.
# `probe_set/` tem as 8 cenas EBB da partição de VALIDAÇÃO da rota A, geradas em
# 2026-09-22. Trocar de conjunto entre as fases tornaria a série incomparável —
# e a série é o instrumento inteiro: o que precisamos ver é se o LVCorr cai da
# fase 1 para a fase 2, como caiu de +0,9059 para +0,4365 na rodada anterior,
# degradando MONOTONICAMENTE enquanto a loss caía. Um número medido noutro
# conjunto não responde essa pergunta.
#
# Referências para comparar, no MESMO conjunto:
#   fase 1 desta rodada ..... 0,983 (mediana de 100 medições, faixa 0,967-0,992)
#   fase 1 anterior ......... +0,9059
#   pesos oficiais do paper . +0,8868
#   fase 2 anterior (colapso) +0,4365
set -euo pipefail

GPU="${GPU:-6}"
NOME="${NOME:-julia_monitor_fase2}"
SAIDA="${SAIDA:-/workspace/retreinar-bokeh/outputs/bokeh_fase2_bc}"
CFG="${CFG:-configs/train_bokeh_fase2_bc.yaml}"
# Release só para MONTAR o conjunto — e ele já existe, então nem é lido.
RELEASE="${RELEASE:-/workspace/retreinar-bokeh/fase1_dados/parcial}"
FLUX="/workspace/hf-cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"

docker run -d --name "$NOME" \
  --user "$(id -u):$(id -g)" \
  --gpus "\"device=${GPU}\"" \
  --shm-size=8g \
  -v /raid/user_juliadollis/julia_docker:/workspace \
  -w /workspace/retreinar-bokeh \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false \
  -e HF_HOME=/workspace/hf-cache-julia \
  -e FLUX_PATH="$FLUX" \
  -e MONITOR_MIN_GB=30 \
  -e MIRROR_ROOTS='{"EBB!":"/workspace/fontes_rota_a/EBB","GenerativePhotography":"/workspace/fontes_rota_a/GenerativePhotography"}' \
  -e PYTHONPATH=/workspace/retreinar-bokeh:/workspace/retreinar-bokeh/third_party/Genfocus \
  julia-genrefocus:1.0 \
  python3 scripts/monitor_externo.py \
    --output-dir "$SAIDA" --release "$RELEASE" --config "$CFG" \
    --probe-set /workspace/retreinar-bokeh/probe_set --n-probe 8 --intervalo 600

echo "docker logs -f $NOME"
echo "cat ${SAIDA}/monitor_externo.jsonl"
