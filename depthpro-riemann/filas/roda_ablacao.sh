#!/usr/bin/env bash
# Ablacao do Wallisson, EXATAMENTE como ele entregou, so trocando o dataset para
# o Spring.
#   roda_ablacao.sh <gpu>
#
# O que NAO foi alterado, de proposito: todos os defaults dele.
#   --filter B0 B1 B7      7 configuracoes (o default dele, nao as 39)
#   --epochs 30
#   --batch-size 2 --grad-accum 4 --grad-checkpointing
#   --align-mode detach    (a nossa campanha usou "full"; e default dele)
#   --gauss-clamp-metrico 5.0
#   --eval-zero-shot
#   codigo dele intocado em ablacao-wallisson/, inclusive o metrics.py da branch
#   fix-geometry, que nao tem a mascara de validade.
#
# O UNICO desvio: --focal 2585.859, que e a mediana que o nosso prepare_spring
# reportou para as 37 sequencias. Sem focal a formulacao metrica nem ativa, e o
# proprio documento dele manda informar a focal original.
set -uo pipefail
GPU="${1:?uso: roda_ablacao.sh <gpu>}"
B=/raid/user_juliadollis/julia_docker
R="$B/ablacao-wallisson"
SAIDA="$B/runs_ablacao_spring"
LOG="$B/ablacao_spring_gpu${GPU}.log"
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-45000}
mkdir -p "$SAIDA"

livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
esperas=0
while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
  [ $((esperas % 12)) -eq 0 ] && echo "[ablacao] gpu$GPU ocupada (${livre} MiB livres, preciso de ${MIN_LIVRE_MIB}); aguardando $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  esperas=$((esperas + 1)); sleep 300
  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
done
[ "$esperas" -gt 0 ] && echo "[ablacao] gpu$GPU liberou (${livre} MiB) apos $((esperas * 5)) min" >> "$LOG"

echo "[ablacao] inicio na gpu$GPU $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
docker rm "julia_ablacao_gpu${GPU}" >/dev/null 2>&1 || true
docker run --rm --name "julia_ablacao_gpu${GPU}" --gpus "\"device=${GPU}\"" \
  --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
  -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
  -v "$B/data":/data -v "$B/models":/models -v "$SAIDA":/workspace/runs \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -w /workspace riemann-depthpro:latest \
  python scripts/run_ablation.py \
    --train-root /data/spring_split/train \
    --val-root   /data/spring_split/val \
    --checkpoint /models/checkpoints/depth_pro.pt \
    --out-dir    /workspace/runs/ablacao_spring \
    --focal 2585.859 \
  >> "$LOG" 2>&1
echo "[ablacao] rc=$? $(date -u +%FT%H:%M:%SZ)" >> "$LOG"

# Regra: checkpoint e resultado que nao estao no Hub nao existem.
T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
docker run --rm --name "julia_ablacao_sobe" --user "$(id -u):$(id -g)" \
  -v "$B":/host -v "$B/hf-cache-julia":/workspace/hf-cache \
  -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home -e HF_TOKEN="$T" \
  -w /host julia-genrefocus-eval:3.0 python3 /host/sobe_ablacao.py >> "$LOG" 2>&1
echo "[ablacao] FIM $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
