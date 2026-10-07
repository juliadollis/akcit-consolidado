#!/usr/bin/env bash
# =============================================================================
# PASSO 4 — AMPLIACAO DE SEEDS (n=3 -> n=10)
#
# Por que existe: com n=3 as faixas dos bracos se sobrepoem (B0 tem minimo
# 0.5895 e o B3 teto1000 tem maximo 0.5908 no F-score de borda). "B0 vence" e
# hoje uma direcao, nao um fato. O relatorio que vai sair daqui e um NULO, e
# nulo com n=3 e a forma mais fraca de nulo. Subir para n=10 aperta o intervalo
# e torna a afirmacao defensavel.
#
# Uso:  passo4_seeds.sh <gpu> <tarefa> [tarefa...]
#       tarefa = <b0|b3>:<teto>:<seed_inicio>:<quantas>
#       (teto e ignorado no b0; use "-" para deixar claro)
#
# Ex.:  passo4_seeds.sh 0 b0:-:3:5
#       passo4_seeds.sh 7 b0:-:8:2 b3:1000:8:2 b3:5:8:2
#
# NAO SOBRESCREVE: escreve na MESMA pasta do experimento (e esse o ponto, as
# seeds novas tem de conviver com as antigas), mas o train_single.py pula
# qualquer seed que ja tenha test_metrics.json. As faixas passadas aqui sao
# disjuntas por construcao, entao duas GPUs nunca disputam a mesma seed.
# =============================================================================
set -uo pipefail
GPU="${1:?uso: passo4_seeds.sh <gpu> <tarefa> [tarefa...]}"; shift
[ $# -ge 1 ] || { echo "faltou tarefa"; exit 2; }
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
FX=2585.859

while docker ps --format "{{.Names}}" | grep -q "julia_fila${GPU}_"; do
  echo "[seeds gpu$GPU] gpu ocupada pela fila de avaliacao; aguardando $(date -u +%H:%M:%SZ)"
  sleep 60
done

for TAREFA in "$@"; do
  IFS=: read -r QUAL TETO INI N <<< "$TAREFA"
  case "$QUAL" in
    b3) PESOS=(--berhu 0.7 --normal 0.9 --gauss 0.45 --grad 0 --geod 0 --metric 0
               --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO")
        NOME="B3_gauss_metrica_teto${TETO}" ;;
    b0) PESOS=(--berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0)
        NOME="B0_berhu" ;;
    # b1: a curvatura SOZINHA sobre o berHu. O b3 mistura normal+gauss, entao o
    # nulo dele nao diz de qual dos dois veio. b1 menos b0 e exatamente o termo
    # de curvatura, que e a hipotese. Sem esse braco o teste e indireto.
    b1) PESOS=(--berhu 0.7 --normal 0 --gauss 0.45 --grad 0 --geod 0 --metric 0
               --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO")
        NOME="B1_gauss_metrica_teto${TETO}" ;;
    *)  echo "[seeds gpu$GPU] tarefa invalida: $TAREFA"; continue ;;
  esac
  LOG="$B/passo4_seeds_${NOME}_s${INI}-$((INI+N-1)).log"
  echo "[seeds gpu$GPU] >>> $NOME seeds ${INI}..$((INI+N-1))  $(date -u +%H:%M:%SZ)  log=$LOG"
  docker run --rm --name "julia_seeds_gpu${GPU}_${NOME}_s${INI}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -v "$B/models":/models -v "$B/runs_riemann":/workspace/runs \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -w /workspace riemann-depthpro:latest \
    python scripts/train_single.py \
      --train-root /data/spring_split/train \
      --val-root   /data/spring_split/val \
      --test-root  /data/spring_split/test \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir "/workspace/runs/$NOME" \
      --seed-inicio "$INI" --seeds "$N" \
      "${PESOS[@]}" \
    > "$LOG" 2>&1
  echo "[seeds gpu$GPU] <<< $NOME seeds ${INI}..$((INI+N-1)) rc=$? $(date -u +%H:%M:%SZ)"
done
echo "[seeds gpu$GPU] TODAS AS TAREFAS TERMINARAM $(date -u +%H:%M:%SZ)"
