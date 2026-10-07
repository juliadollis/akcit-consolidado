#!/usr/bin/env bash
# CADEIA DA NOITE: da inferencia ate a tabela final, sem intervencao.
#
#  1. espera toda a inferencia terminar
#  2. monta a lista de repos da campanha Eq. 4 (+ identidades herdadas)
#  3. metrica por imagem nas 7 GPUs, com deduplicacao e conferencia automatica
#  4. metricas riemannianas (campo de desfoque) nos mesmos repos
#  5. tabela final: IC95 agrupado por cena + teste pareado
#  6. sobe tudo para o Hub
#
# Cada etapa registra em log proprio; uma falha nao impede a seguinte de tentar.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
P=$B/genrefocus_deblurnet_paper
HF_TOKEN="$(grep -h '^HF_TOKEN' "$P/.env" | cut -d= -f2)"
IMG=julia-genrefocus-eval:3.0
GPUS="0 1 2 3 5 6 7"

roda() {
  local nome=$1 gpu=$2; shift 2
  docker run --rm --name "$nome" --gpus "\"device=${gpu}\"" \
    --user "$(id -u):$(id -g)" --shm-size=16g \
    -v "$P/vision-pipeline":/workspace/vision-pipeline \
    -v "$B/depth-riemannian":/workspace/depth-riemannian \
    -v "$B/hf-cache-julia":/workspace/hf-cache -v "$B":/host \
    -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
    -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
    -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    -w /workspace/vision-pipeline "$IMG" "$@"
}

echo "[noite] 1/6 aguardando a inferencia $(date -u +%H:%M:%SZ)"
while [ "$(docker ps --format '{{.Names}}' | grep -cE '^julia_(eq4|ident)')" -gt 0 ]; do sleep 180; done
echo "[noite] inferencia terminou $(date -u +%H:%M:%SZ)"

echo "[noite] 2/6 montando listas"
roda julia_listas 0 python3 /host/monta_listas_eq4.py 2>&1 | grep -E "repos|identidade|listas"

echo "[noite] 3/6 metrica por imagem $(date -u +%H:%M:%SZ)"
for g in $GPUS; do
  ( roda "julia_pi_gpu${g}" "$g" python3 metricas_por_imagem.py \
      --lista "/host/lista_eq4_gpu${g}.txt" --saida-local "/host/pi_eq4_gpu${g}" \
      > "$B/noite_pi_gpu${g}.log" 2>&1 ) &
done
wait
echo "[noite] por imagem: $(grep -h 'OK conferencia' "$B"/noite_pi_gpu*.log 2>/dev/null | wc -l) conferem, $(grep -h '!! conferencia' "$B"/noite_pi_gpu*.log 2>/dev/null | wc -l) divergem"

echo "[noite] 4/6 metricas riemannianas $(date -u +%H:%M:%SZ)"
for g in $GPUS; do
  ( roda "julia_ri_gpu${g}" "$g" python3 metricas_riemannianas.py \
      --lista "/host/lista_eq4_gpu${g}.txt" --saida-local "/host/ri_eq4_gpu${g}" \
      > "$B/noite_ri_gpu${g}.log" 2>&1 ) &
done
wait
echo "[noite] riemannianas terminadas $(date -u +%H:%M:%SZ)"

echo "[noite] 5/6 tabela final"
roda julia_tabela 0 python3 tabela_estilo_paper.py \
  --entrada "/host/pi_eq4_gpu*/por_imagem.parquet" \
  --incluir-repos "bokeh-eq4-,identidade" \
  --saida-md /host/TABELA_EQ4.md \
  --saida-hf juliadollis/genrefocus-tabela-eq4 > "$B/noite_tabela.log" 2>&1
echo "[noite] tabela rc=$?"

echo "[noite] 6/6 subindo dados por imagem"
roda julia_sobe 0 python3 /host/sobe_eq4.py 2>&1 | grep -E "linhas|enviado|Error"
echo "[noite] CONCLUIDO $(date -u +%H:%M:%SZ)"
