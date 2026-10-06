#!/usr/bin/env bash
# Fila de avaliacao COM a Eq. 4 de verdade.
#
# Diferenca para o fila_gpu.sh: monta /host e injeta PYTHONPATH=/host/pylibs_birefnet.
# Sem isso o BiRefNet nao importa (falta einops e kornia na imagem) e o plano de
# foco cai no PIXEL CENTRAL -- que foi o que aconteceu em TODA a campanha
# anterior, inclusive na curada, apesar de a tabela dizer "Eq. 4 (BiRefNet)".
# O proprio codigo avisa disso no log da FASE 1b.
set -uo pipefail
GPU="${1:?uso: fila_gpu_eq4.sh <gpu> <arquivo_de_fila>}"
FILA="${2:?uso: fila_gpu_eq4.sh <gpu> <arquivo_de_fila>}"
B=/raid/user_juliadollis/julia_docker
PROJ="$B/genrefocus_deblurnet_paper"
IMG=julia-genrefocus-eval:3.0
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-40000}   # o item usa ~34 GB; 40 GB da folga
HF_TOKEN="$(grep -h '^HF_TOKEN' "$PROJ/.env" | cut -d= -f2)"
[ -n "$HF_TOKEN" ] || { echo "ERRO: HF_TOKEN nao encontrado"; exit 1; }

echo "[eq4 gpu$GPU] iniciando — $(wc -l < "$FILA") itens $(date -u +%H:%M:%SZ)"
while IFS='|' read -r nome argumentos; do
  case "$nome" in ''|\#*) continue;; esac
  nome="$(echo "$nome" | xargs)"
  cont="julia_eq4_${GPU}_${nome}"
  log="$B/eq4_${nome}.log"
  while docker ps --format '{{.Names}}' | grep -qx "$cont"; do sleep 60; done
  docker rm "$cont" >/dev/null 2>&1 || true

  # GUARDA DE MEMORIA. Checar utilizacao nao serve: um deploy de serving (VLLM)
  # fica em 0% entre requisicoes com 75 GB ja reservados, e a GPU PARECE livre.
  # Foi assim que 12 itens desta campanha morreram de OOM em 3 minutos, nas
  # quatro placas que o `iago_qwe_38_next` tinha tomado. O que decide e a
  # memoria LIVRE.
  livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  esperas=0
  while [ "${livre:-0}" -lt "$MIN_LIVRE_MIB" ]; do
    if [ "$esperas" -eq 0 ]; then
      echo "[eq4 gpu$GPU] gpu ocupada por outro processo (${livre} MiB livres, preciso de ${MIN_LIVRE_MIB}); aguardando"
    fi
    esperas=$((esperas + 1))
    sleep 300
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  done
  [ "$esperas" -gt 0 ] && echo "[eq4 gpu$GPU] gpu liberou (${livre} MiB) apos $((esperas * 5)) min"
  echo "[eq4 gpu$GPU] >>> $nome  ($(date -u +%H:%M:%SZ))"
  docker run --rm --name "$cont" --gpus "\"device=${GPU}\"" \
      --user "$(id -u):$(id -g)" --shm-size=16g \
      -v "$PROJ/vision-pipeline":/workspace/vision-pipeline \
      -v "$B/hf-cache-julia":/workspace/hf-cache \
      -v "$B":/host \
      -e HF_HOME=/workspace/hf-cache \
      -e TORCH_HOME=/workspace/hf-cache/torch \
      -e XDG_CACHE_HOME=/workspace/hf-cache/xdg \
      -e HOME=/workspace/hf-cache/home \
      -e PYTHONPATH=/host/pylibs_birefnet \
      -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
      -e BOKEHNET_METRICS_REPO=juliadollis/bokeh-eval-metricas-eq4 \
      -w /workspace/vision-pipeline "$IMG" \
      python3 run_3models.py $argumentos > "$log" 2>&1
  rc=$?
  bn=$(grep -c "origem=birefnet" "$log" 2>/dev/null || echo 0)
  ct=$(grep -c "origem=centro" "$log" 2>/dev/null || echo 0)
  echo "[eq4 gpu$GPU] <<< $nome rc=$rc  ($(date -u +%H:%M:%SZ))  foco: birefnet=$bn centro=$ct"
done < "$FILA"
echo "[eq4 gpu$GPU] FILA COMPLETA — $(date -u +%H:%M:%SZ)"
