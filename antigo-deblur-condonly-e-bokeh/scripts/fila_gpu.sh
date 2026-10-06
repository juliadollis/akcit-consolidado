#!/usr/bin/env bash
# =============================================================================
# FILA DE EXPERIMENTOS por GPU na dgx-H100-01 (docker, SEM SLURM).
#
# Roda uma lista de experimentos SEQUENCIALMENTE numa GPU, um container por vez.
# Motivo de existir: esta maquina nao tem escalonador, entao GPU ociosa e GPU
# que o grupo pode perder. A fila mantem a GPU ocupada sem intervencao.
#
# Cada linha do arquivo de fila e:  <nome_curto>|<argumentos do run_3models.py>
# Linhas em branco e as que comecam com # sao ignoradas.
#
# Uso (na maquina):
#   bash scripts/fila_gpu.sh 2 /raid/.../fila_gpu2.txt
#
# Idempotencia: o pipeline do bokeh_net PULA lotes que ja estao no repo HF de
# saida, entao re-rodar um item ja concluido custa quase nada.
# =============================================================================
set -uo pipefail

GPU="${1:?uso: fila_gpu.sh <gpu> <arquivo_de_fila>}"
FILA="${2:?uso: fila_gpu.sh <gpu> <arquivo_de_fila>}"
B=/raid/user_juliadollis/julia_docker
PROJ="$B/genrefocus_deblurnet_paper"
IMG="${IMG:-julia-genrefocus-eval:2.0}"

# HOME apontando para o volume montado (NAO remova): a imagem define
# HOME=/tmp/juliahome, e durante o `docker build` o pip (como root) criou
# /tmp/juliahome/.cache pertencendo ao root. Em runtime, como uid do usuario,
# nao da para escrever la. O TORCH_HOME/XDG_CACHE_HOME resolvem pyiqa e afins,
# mas o CLIP da OpenAI ignora ambos e usa `~/.cache/clip` fixo. Trocar o HOME
# cobre todos de uma vez.

HF_TOKEN="$(grep -h '^HF_TOKEN' "$PROJ/.env" | cut -d= -f2)"
[ -n "$HF_TOKEN" ] || { echo "ERRO: HF_TOKEN nao encontrado em $PROJ/.env"; exit 1; }

echo "[fila gpu$GPU] iniciando — $(grep -cvE '^\s*(#|$)' "$FILA") itens"

PARAR="$B/PARAR_KEEPER_${GPU}"
while IFS='|' read -r nome argumentos; do
  case "$nome" in ''|\#*) continue;; esac
  # Checa a flag de parada ENTRE ITENS, nao so entre voltas da fila. Sem isto,
  # quando a cadeia do kfix pediu as GPUs em 2026-08-20 o keeper ja tinha
  # engatado o proximo item e 3 GPUs ficaram ociosas por mais de uma hora
  # esperando a fila drenar.
  if [ -f "$PARAR" ]; then
    echo "[fila gpu$GPU] flag de parada detectada — encerrando antes de '$nome'"
    exit 0
  fi
  nome="$(echo "$nome" | xargs)"
  cont="julia_fila${GPU}_${nome}"
  log="$B/fila_gpu${GPU}_${nome}.log"

  # Nao duplica: se ja existe container com esse nome rodando, espera.
  while docker ps --format '{{.Names}}' | grep -qx "$cont"; do sleep 60; done
  docker rm "$cont" >/dev/null 2>&1 || true

  echo "[fila gpu$GPU] >>> $nome  ($(date -u +%H:%M:%SZ))"
  docker run --rm --name "$cont" --gpus "\"device=${GPU}\"" \
      --user "$(id -u):$(id -g)" --shm-size=16g \
      -v "$PROJ/vision-pipeline":/workspace/vision-pipeline \
      -v "$B/hf-cache-julia":/workspace/hf-cache \
      -e HF_HOME=/workspace/hf-cache \
      -e TORCH_HOME=/workspace/hf-cache/torch \
      -e XDG_CACHE_HOME=/workspace/hf-cache/xdg \
      -e HOME=/workspace/hf-cache/home \
      -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
      -e BOKEHNET_METRICS_REPO=juliadollis/bokeh-eval-metricas \
      -w /workspace/vision-pipeline "$IMG" \
      python3 run_3models.py $argumentos > "$log" 2>&1
  rc=$?
  echo "[fila gpu$GPU] <<< $nome terminou rc=$rc  ($(date -u +%H:%M:%SZ))  log=$log"
done < "$FILA"

echo "[fila gpu$GPU] FILA COMPLETA — $(date -u +%H:%M:%SZ)"
