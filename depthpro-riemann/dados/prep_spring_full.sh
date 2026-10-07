#!/usr/bin/env bash
# Passo 3 do setup: prepara o Spring COMPLETO (37 seqs x 248 quadros, passo 10).
# Depois refaz a medida de |K| sobre esse conjunto maior, para os percentis do
# passo 3 deixarem de depender da amostra de 111 quadros.
# Saida NOVA (spring_prep/full); nao mexe em spring_prep/test.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
roda () {
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -e HOME=/tmp -w /workspace riemann-depthpro:latest "$@"
}
echo "[prep] prepare_spring completo $(date -u +%H:%M:%SZ)"
roda python scripts/prepare_spring.py \
  --spring-root /data/spring/train/spring/train \
  --out-root /data/spring_prep/full \
  --camera left --passo 10 --max-depth 100
echo "[prep] prepare rc=$? $(date -u +%H:%M:%SZ)"

echo "[prep] medindo |K| no conjunto completo"
roda python scripts/medir_k_spring_porseq.py \
  --raiz /data/spring_prep/full \
  --spring-root /data/spring/train/spring/train \
  --fx-mediana 2585.859
echo "[prep] FIM $(date -u +%H:%M:%SZ)"
