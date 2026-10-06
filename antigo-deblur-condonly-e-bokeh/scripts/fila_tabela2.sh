#!/usr/bin/env bash
# Reproducao da TABELA 2 do paper (deblurring: RealDOF + DPDD), sequencial numa
# GPU. Usa run_tabela2.py, que difere do run_3models.py: modelo DeblurNet,
# datasets da Tab. 2 e metricas nas variantes do paper (lpips+, clipiqa+,
# maniqa-kadid, musiq).
set -uo pipefail
GPU="${1:?uso: fila_tabela2.sh <gpu>}"
B=/raid/user_juliadollis/julia_docker
PROJ="$B/genrefocus_deblurnet_paper"
T="$(grep -h '^HF_TOKEN' "$PROJ/.env" | cut -d= -f2)"
for m in nosso oficial; do
  cont="julia_tab2_$m"
  docker rm "$cont" >/dev/null 2>&1 || true
  echo "[tab2 gpu$GPU] >>> $m ($(date -u +%H:%M:%SZ))"
  docker run --rm --name "$cont" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=16g \
    -v "$PROJ/vision-pipeline":/workspace/vision-pipeline \
    -v "$B/hf-cache-julia":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
    -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
    -e HF_TOKEN="$T" -e HUGGINGFACE_HUB_TOKEN="$T" \
    -w /workspace/vision-pipeline julia-genrefocus-eval:2.0 \
    python3 run_tabela2.py --modelo "$m" > "$B/tab2_$m.log" 2>&1
  echo "[tab2 gpu$GPU] <<< $m rc=$? ($(date -u +%H:%M:%SZ))"
done
echo "[tab2 gpu$GPU] FIM"
