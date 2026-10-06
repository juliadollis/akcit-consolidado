#!/usr/bin/env bash
# Espera o container que ainda roda nesta GPU cair e emenda a proxima fila.
# Existe porque os lacos originais foram encerrados (para nao dispararem os
# itens do LF-repro), mas os containers em andamento foram preservados: matar o
# laco nao mata o container, e nao havia motivo para jogar fora horas de GPU ja
# gastas.
set -uo pipefail
GPU="${1:?uso: espera_e_roda_eq4.sh <gpu> <fila>}"
FILA="${2:?uso: espera_e_roda_eq4.sh <gpu> <fila>}"
B=/raid/user_juliadollis/julia_docker
P=$B/genrefocus_deblurnet_paper
echo "[espera gpu$GPU] aguardando a GPU vagar $(date -u +%H:%M:%SZ)"
while docker ps --format '{{.Names}}' | grep -q "^julia_eq4_${GPU}_"; do sleep 120; done
echo "[espera gpu$GPU] livre, emendando a fila $(date -u +%H:%M:%SZ)"
cd "$P" && exec bash scripts/fila_gpu_eq4.sh "$GPU" "$FILA"
