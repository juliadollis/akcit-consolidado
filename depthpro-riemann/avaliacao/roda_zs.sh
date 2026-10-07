#!/usr/bin/env bash
# Zero-shot do DepthPro no teste do Spring. GPU 2.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
GPU=2
echo "[zs] iniciando $(date -u +%H:%M:%SZ) na gpu$GPU"
docker run --rm --name julia_zeroshot_spring --gpus "\"device=${GPU}\"" \
  --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
  -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
  -v "$B/data":/data -v "$B/models":/models -v "$B/runs_riemann":/workspace/runs \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -w /workspace riemann-depthpro:latest \
  python scripts/zero_shot_spring.py \
    --test-root /data/spring_split/test \
    --checkpoint /models/checkpoints/depth_pro.pt \
    --out-dir /workspace/runs/ZERO_SHOT_spring
echo "[zs] rc=$? $(date -u +%H:%M:%SZ)"
