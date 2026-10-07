#!/usr/bin/env bash
# Reavalia TODOS os checkpoints em uma ou mais mesas, com agregado e por imagem.
#   roda_avaliacao.sh <gpu> [nome=raiz ...]
# Sem mesas, usa o teste do Spring.
set -uo pipefail
GPU="${1:?uso: roda_avaliacao.sh <gpu> [nome=raiz ...]}"; shift
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
MESAS=("$@"); [ ${#MESAS[@]} -eq 0 ] && MESAS=("spring_test=/data/spring_split/test")
ARGS=(); for m in "${MESAS[@]}"; do ARGS+=(--dataset "$m"); done
LOG="$B/avaliacao_gpu${GPU}.log"
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-40000}

livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
e=0
while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
  [ $((e % 12)) -eq 0 ] && echo "[avalia] gpu$GPU ocupada (${livre} MiB); aguardando $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  e=$((e+1)); sleep 300
  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
done

echo "[avalia] inicio gpu$GPU $(date -u +%FT%H:%M:%SZ) mesas: ${MESAS[*]}" >> "$LOG"
docker run --rm --name "julia_avalia_gpu${GPU}" --gpus "\"device=${GPU}\"" \
  --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
  -v "$R/riemann":/workspace/riemann:ro -v "$B/data":/data -v "$B/models":/models \
  -v "$B":/host -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w /workspace \
  riemann-depthpro:latest \
  python3 /host/avalia_modelos.py \
    --checkpoints "/host/runs_*/*/seed_*/best.pt" \
    "${ARGS[@]}" \
    --checkpoint-base /models/checkpoints/depth_pro.pt \
    --saida /host/avaliacoes --incluir-zero-shot \
  >> "$LOG" 2>&1
echo "[avalia] rc=$? $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
