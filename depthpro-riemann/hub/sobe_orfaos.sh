#!/usr/bin/env bash
# Sobe as seeds 45 e 46 da ablacao, que existem SO em disco.
#
# A fila anterior registrou "subiu para o Hub" sem checar o codigo de saida do
# upload, entao 45 e 46 falharam calado. Aqui o resultado e verificado.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
T=$(grep -h '^HF_TOKEN' "$B/genrefocus_deblurnet_paper/.env" | cut -d= -f2)
if [ -z "$T" ]; then echo "SEM TOKEN"; exit 1; fi

echo "== por que 45 e 46 falharam =="
for S in 45 46; do
  echo "-- sobe_seed_${S}.log (fim)"
  tail -6 "$B/sobe_seed_${S}.log" 2>/dev/null | grep -viE 'CUDA|Container image|governed by|By pulling|^https|A copy of|^=====' || echo "   (log vazio)"
done

echo
echo "== subindo =="
for S in 45 46; do
  echo "-- seed $S"
  docker run --rm --name "julia_orf_s${S}" --user "$(id -u):$(id -g)" \
    -v "$B":/host -v "$B/hf-cache-julia":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home \
    -e HF_TOKEN="$T" -w /host julia-genrefocus-eval:3.0 \
    python3 /host/sobe_ablacao_seed.py "$S" 2>&1 |
    grep -viE 'CUDA|Container image|governed by|By pulling|^https|A copy of|^====='
  echo "   codigo de saida: ${PIPESTATUS[0]}"
done
