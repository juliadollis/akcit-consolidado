#!/usr/bin/env bash
# Pipeline completo de pos-processamento da avaliacao de bokeh.
#
#   1. espera a fila de inferencia terminar
#   2. remonta as listas de repos (ja incluindo os novos)
#   3. recalcula metrica POR IMAGEM nas 7 GPUs, com deduplicacao e conferencia
#      automatica contra a tabela agregada antiga
#   4. monta a tabela final com IC95 agrupado por cena e teste pareado
#   5. empurra tudo para o Hub
#
# Existe para a cadeia ser reprodutivel com um comando, em vez de sete passos
# manuais que ninguem lembra na ordem certa.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
P=$B/genrefocus_deblurnet_paper
HF_TOKEN="$(grep -h '^HF_TOKEN' "$P/.env" | cut -d= -f2)"
IMG=julia-genrefocus-eval:3.0
GPUS="0 1 2 3 5 6 7"

roda() {  # roda <nome-container> <gpu> <comando...>
  local nome=$1 gpu=$2; shift 2
  docker run --rm --name "$nome" --gpus "\"device=${gpu}\"" \
    --user "$(id -u):$(id -g)" --shm-size=16g \
    -v "$P/vision-pipeline":/workspace/vision-pipeline \
    -v "$B/hf-cache-julia":/workspace/hf-cache -v "$B":/host \
    -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
    -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
    -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    -w /workspace/vision-pipeline "$IMG" "$@"
}

echo "[pipeline] 1/5 aguardando a fila de inferencia $(date -u +%H:%M:%SZ)"
while [ "$(docker ps --format '{{.Names}}' | grep -c '^julia_fila')" -gt 0 ]; do sleep 120; done
echo "[pipeline] fila terminou $(date -u +%H:%M:%SZ)"

echo "[pipeline] 2/5 remontando as listas de repos"
roda julia_listas 0 python3 /host/monta_listas2.py 2>&1 | grep -E "repos no total|Error"

echo "[pipeline] 3/5 metrica por imagem nas GPUs"
for g in $GPUS; do
  ( roda "julia_porimg_gpu${g}" "$g" python3 metricas_por_imagem.py \
      --lista "/host/lista_porimg_gpu${g}.txt" --saida-local "/host/por_imagem_gpu${g}" \
      > "$B/porimg3_gpu${g}.log" 2>&1 ) &
done
wait
echo "[pipeline] metrica por imagem terminou $(date -u +%H:%M:%SZ)"
ok=$(grep -h "OK conferencia" "$B"/porimg3_gpu*.log 2>/dev/null | wc -l)
dv=$(grep -h "!! conferencia" "$B"/porimg3_gpu*.log 2>/dev/null | wc -l)
echo "[pipeline] conferencias: $ok conferem, $dv divergem"

echo "[pipeline] 4/5 tabela final"
roda julia_tabela 0 python3 tabela_estilo_paper.py \
  --entrada "/host/por_imagem_gpu*/por_imagem.parquet" \
  --saida-md /host/TABELA_FINAL.md \
  --saida-hf juliadollis/genrefocus-tabela-final > "$B/tabela_final.log" 2>&1
echo "[pipeline] tabela rc=$?"

echo "[pipeline] 5/5 subindo metricas por imagem"
roda julia_sobe 0 python3 /host/sobe_por_imagem.py 2>&1 | grep -E "linhas:|enviado|Error"
echo "[pipeline] CONCLUIDO $(date -u +%H:%M:%SZ)"
