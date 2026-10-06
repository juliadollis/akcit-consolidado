#!/usr/bin/env bash
# =============================================================================
# CADEIA AUTOMATICA do treino "kfix" na dgx-H100-01.
#
# Encadeia, sem intervencao:
#   1. espera o job `julia_kfix` (recomposicao do K da rota b) terminar;
#   2. DIAGNOSTICO dos mapas — reprova e ABORTA se sairem saturados/mortos;
#   3. SMOKE de 20 steps no caminho real (kfix + 2 condicoes + upload HF);
#   4. libera 4 GPUs (para os keepers delas) e sobe o TREINO de 60K steps.
#
# Por que o passo 2 existe: a Etapa A.2 mostrou que a escolha do normalizador
# decide entre mapa util e inutil. Um mapa saturado ensina "borre tudo" e o
# prejuizo so apareceria dias depois. Custa 1 minuto conferir.
#
# Por que o passo 3 existe: na fase 2 original o smoke pegou um bug do filtro de
# SSIM que teria custado 60K steps.
#
# GPUs: o treino toma 0,1,6,7 (accum 8 x 4 GPUs = batch efetivo 32, igual ao
# paper). As GPUs 2,3,5 seguem com as filas de avaliacao, entao a maquina
# continua 100% ocupada.
#
# NADA e sobrescrito: output_dir e repo HF proprios (ver o yaml).
# =============================================================================
set -uo pipefail

B=/raid/user_juliadollis/julia_docker
PROJ="$B/genrefocus_deblurnet_paper"
IMG_EVAL=julia-genrefocus-eval:2.0
IMG_TRAIN=julia-genrefocus:1.0
GPUS_TREINO="0,1,6,7"
T="$(grep -h '^HF_TOKEN' "$PROJ/.env" | cut -d= -f2)"
W="$(grep -h '^WANDB_API_KEY' "$PROJ/.env" | cut -d= -f2)"

log() { echo "[cadeia $(date -u +%H:%M:%SZ)] $*"; }

# --- 1) esperar o kfix -------------------------------------------------------
log "aguardando julia_kfix terminar..."
while docker ps --format '{{.Names}}' | grep -qx julia_kfix; do sleep 120; done
if ! grep -q "FIM KFIX" "$B/kfix.log" 2>/dev/null; then
  log "ERRO: kfix nao chegou ao fim. Abortando (o treino NAO sobe)."
  tail -20 "$B/kfix.log" 2>/dev/null | grep -viE "^\s*[0-9]+%|MB/s|it/s"
  exit 1
fi
log "kfix concluido. $(grep -E 'CALIBRACAO|variacao' "$B/kfix.log" | tail -2 | tr '\n' ' ')"

docker_comum=(--user "$(id -u):$(id -g)" --shm-size=16g
  -v "$PROJ":/workspace/genrefocus_deblurnet_paper
  -v "$B/hf-cache-julia":/workspace/hf-cache
  -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch
  -e HOME=/workspace/hf-cache/home
  -e HF_TOKEN="$T" -e HUGGINGFACE_HUB_TOKEN="$T" -e WANDB_API_KEY="$W"
  -w /workspace/genrefocus_deblurnet_paper)

# --- 2) diagnostico dos mapas ------------------------------------------------
log "diagnostico dos mapas do modo kfix..."
docker run --rm --name julia_diag_kfix --gpus '"device=3"' "${docker_comum[@]}" \
  "$IMG_EVAL" python3 scripts/diag_mapa_kfix.py configs/train_bokeh_fase2_kfix.yaml 24 \
  > "$B/diag_kfix.log" 2>&1
if [ $? -ne 0 ]; then
  log "DIAGNOSTICO REPROVOU — treino NAO sera disparado."
  grep -E "\[diag\]|REPROVADO" "$B/diag_kfix.log" | tail -10
  exit 2
fi
log "diagnostico aprovado."
grep -E "\[diag\]|APROVADO" "$B/diag_kfix.log" | tail -6

# --- 3) smoke de 20 steps ----------------------------------------------------
log "smoke de 20 steps (caminho real + upload HF)..."
docker run --rm --name julia_smoke_kfix --gpus '"device=3"' "${docker_comum[@]}" \
  -e NUM_GPUS=1 -e TRAIN_CONFIG=configs/train_bokeh_fase2_kfix_smoke.yaml \
  -e INIT_LORA=outputs/bokehnet_synth_2gpu \
  -e HF_REPO_ID=genrefocus-bokehnet-fase2-kfix-smoke \
  "$IMG_TRAIN" bash scripts/docker_bokeh_bootstrap.sh > "$B/smoke_kfix.log" 2>&1
if ! grep -q "\[train\] step=20/20" "$B/smoke_kfix.log"; then
  log "SMOKE FALHOU — treino NAO sera disparado."
  grep -viE "^loading|it/s\]" "$B/smoke_kfix.log" | tail -25
  exit 3
fi
log "smoke passou. $(grep -E '\[data\].*kfix|\[upload\]' "$B/smoke_kfix.log" | tail -3 | tr '\n' ' ')"

# --- 4) liberar 4 GPUs e subir o treino --------------------------------------
log "liberando as GPUs $GPUS_TREINO dos keepers..."
for g in ${GPUS_TREINO//,/ }; do touch "$B/PARAR_KEEPER_$g"; done
for g in ${GPUS_TREINO//,/ }; do
  while docker ps --format '{{.Names}}' | grep -q "^julia_fila${g}_"; do sleep 60; done
  log "  GPU $g livre"
done

log "subindo o TREINO kfix (60K steps, 4 GPUs)..."
docker rm julia_bokeh_kfix >/dev/null 2>&1 || true
docker run -d --name julia_bokeh_kfix --gpus "\"device=${GPUS_TREINO}\"" \
  --restart=no "${docker_comum[@]}" \
  -e NUM_GPUS=4 -e TRAIN_CONFIG=configs/train_bokeh_fase2_kfix.yaml \
  -e INIT_LORA=outputs/bokehnet_synth_2gpu \
  -e HF_REPO_ID=genrefocus-bokehnet-fase2-kfix \
  "$IMG_TRAIN" bash scripts/docker_bokeh_bootstrap.sh
log "TREINO NO AR: docker logs -f julia_bokeh_kfix"
log "FIM DA CADEIA"
