#!/usr/bin/env bash
# Pre-requisitos do passo 4: pesos do DepthPro + os TRES splits do Spring.
# O split e o mesmo das rodadas anteriores: --particao com fracoes 50/15/35 e
# seed 42, que o prepare_spring garante DISJUNTO entre as tres chamadas.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
mkdir -p "$B/models"
roda () {
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
    -v "$B/data":/data -v "$B/models":/models -e HOME=/tmp -w /workspace \
    riemann-depthpro:latest "$@"
}
echo "[p4] pesos do DepthPro $(date -u +%H:%M:%SZ)"
roda python scripts/download_weights.py --out-dir /models
echo "[p4] pesos rc=$?"

for part in train val test; do
  echo "[p4] preparando particao $part $(date -u +%H:%M:%SZ)"
  roda python scripts/prepare_spring.py \
    --spring-root /data/spring/train/spring/train \
    --out-root "/data/spring_split/$part" \
    --camera left --passo 4 --max-depth 100 \
    --particao "$part" --particao-seed 42 --particao-fracoes 0.50 0.15 0.35 \
    2>&1 | tail -12
  echo "[p4] $part rc=$?"
done
echo "[p4] contagem final:"
for part in train val test; do
  echo "  $part: $(ls $B/data/spring_split/$part/depth 2>/dev/null | wc -l) quadros"
done
echo "[p4] FIM $(date -u +%H:%M:%SZ)"
