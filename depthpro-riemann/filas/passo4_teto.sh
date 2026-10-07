#!/usr/bin/env bash
# PASSO 4 — variante com TETO parametrizado.
#
# Uso: passo4_teto.sh <gpu> <b0|b3> <teto>
#
# Existe para rodar o B3 com o teto 5,0 que o RETESTE_CURVATURA.md previa, ao
# lado do teto 1000 que o dado do passo 3 indicou. Sem esse par nao da para
# separar "a hipotese falhou" de "o teto estava errado".
#
# NAO SOBRESCREVE NADA: a pasta de saida e o log carregam o teto no nome, entao
# a rodada de teto 1000 (B3_gauss_metrica_teto1000 e passo4_b3.log) fica intacta.
set -uo pipefail
GPU="${1:?uso: passo4_teto.sh <gpu> <b0|b3> <teto>}"
QUAL="${2:?uso: passo4_teto.sh <gpu> <b0|b3> <teto>}"
TETO="${3:?uso: passo4_teto.sh <gpu> <b0|b3> <teto>}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
FX=2585.859

case "$QUAL" in
  b3) PESOS=(--berhu 0.7 --normal 0.9 --gauss 0.45 --grad 0 --geod 0 --metric 0
             --gauss-metrica --fx-orig "$FX" --gauss-clamp "$TETO")
      NOME="B3_gauss_metrica_teto${TETO}" ;;
  b0) PESOS=(--berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0)
      NOME="B0_berhu" ;;
  *)  echo "segundo argumento deve ser b0 ou b3"; exit 2 ;;
esac
LOG="$B/passo4_${QUAL}_teto${TETO}.log"
SAIDA="$B/runs_riemann/$NOME"

# Guarda contra sobrescrita: se a pasta ja existe com resultado, para.
if [ -d "$SAIDA" ] && [ -n "$(ls -A "$SAIDA" 2>/dev/null)" ]; then
  echo "[p4-$QUAL-$TETO] ABORTADO: $SAIDA ja existe e nao esta vazia."
  exit 3
fi

while docker ps --format '{{.Names}}' | grep -q "julia_fila${GPU}_"; do
  echo "[p4-$QUAL-$TETO] gpu$GPU ocupada pela fila; aguardando $(date -u +%H:%M:%SZ)"
  sleep 60
done

echo "[p4-$QUAL-$TETO] iniciando $NOME na gpu$GPU $(date -u +%H:%M:%SZ)"
docker run --rm --name "julia_p4_${QUAL}_t${TETO}" --gpus "\"device=${GPU}\"" \
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
  > "$LOG" 2>&1
echo "[p4-$QUAL-$TETO] $NOME rc=$? $(date -u +%H:%M:%SZ)"
