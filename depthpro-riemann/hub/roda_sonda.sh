#!/usr/bin/env bash
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$B":/host -v "$B/hf-cache-julia":/workspace/hf-cache \
  -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home \
  -e HF_TOKEN="$T" -w /host julia-genrefocus-eval:3.0 \
  python3 /host/sonda_orgs.py 2>&1 \
  | grep -viE 'CUDA|Container image|governed by|By pulling|^https|A copy of|^====='
