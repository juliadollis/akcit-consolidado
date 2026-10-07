#!/usr/bin/env bash
# PASSO 4 do RETESTE_CURVATURA — exatamente o par pedido: B3 + controle B0.
#
# Uso: passo4.sh <gpu> <b0|b3>
#
# Split: o mesmo das rodadas anteriores (prepare_spring --particao, fracoes
# 50/15/35, seed 42, sequencias DISJUNTAS). train 593 / val 184 / test 485.
#
# B3  = pesos default do train_single.py (berhu 0.7 + normal 0.9 + gauss 0.45),
#       que e a config "champion", trocando gauss_loss por gauss_loss_metrica
#       via --gauss-metrica, com a focal do passo 2 e o teto do passo 3.
# B0  = B0_berhu do riemann/ablation_configs.py: berHu PURO, todos os termos
#       geometricos zerados. E o controle.
#
# Teto = 1000, escolhido com o dado do passo 3 (9,1% de saturacao com a perda
# ainda bem distribuida; o 5,0 que o documento previa saturaria 49,7%).
set -uo pipefail
GPU="${1:?uso: passo4.sh <gpu> <b0|b3>}"
QUAL="${2:?uso: passo4.sh <gpu> <b0|b3>}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
FX=2585.859
TETO=1000

# Espera a GPU vagar (o keeper foi sinalizado para parar entre itens).
while docker ps --format '{{.Names}}' | grep -q "julia_fila${GPU}_"; do
  echo "[p4-$QUAL] gpu$GPU ainda ocupada pela fila; aguardando $(date -u +%H:%M:%SZ)"
  sleep 60
done

case "$QUAL" in
  b3) PESOS=(--berhu 0.7 --normal 0.9 --gauss 0.45 --grad 0 --geod 0 --metric 0
             --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO")
      NOME=B3_gauss_metrica_teto${TETO} ;;
  b0) PESOS=(--berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0)
      NOME=B0_berhu ;;
  *)  echo "segundo argumento deve ser b0 ou b3"; exit 2 ;;
esac

echo "[p4-$QUAL] iniciando $NOME na gpu$GPU $(date -u +%H:%M:%SZ)"
docker run --rm --name "julia_p4_${QUAL}" --gpus "\"device=${GPU}\"" \
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
    "${PESOS[@]}" \
  > "$B/passo4_${QUAL}.log" 2>&1
echo "[p4-$QUAL] $NOME rc=$? $(date -u +%H:%M:%SZ)"
