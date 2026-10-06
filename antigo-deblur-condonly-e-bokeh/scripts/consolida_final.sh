#!/usr/bin/env bash
# SEGUNDA CONSOLIDACAO: refaz a tabela depois que o EBB400 fechar.
#
# A cadeia da noite montou a lista de repos no momento em que comecou, entao ela
# cobre RealDOF e RealBokeh completos mas so a parte do EBB que ja tinha
# fechado. Este passo espera o EBB inteiro terminar e refaz a consolidacao com
# tudo. Custa pouco: e metrica sobre imagem JA GERADA, sem inferencia nenhuma.
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

echo "[final] aguardando a cadeia da noite $(date -u +%H:%M:%SZ)"
until grep -q "CONCLUIDO" "$B/pipeline_noite.log" 2>/dev/null; do sleep 120; done
echo "[final] aguardando o EBB400 fechar $(date -u +%H:%M:%SZ)"
while [ "$(docker ps --format '{{.Names}}' | grep -c '^julia_extra_ebb')" -gt 0 ]; do sleep 300; done
echo "[final] tudo fechado, reconsolidando $(date -u +%H:%M:%SZ)"

roda julia_listas_f 0 python3 /host/monta_listas_eq4.py 2>&1 | grep -E "repos|identidade"

for g in $GPUS; do
  ( roda "julia_pif_gpu${g}" "$g" python3 metricas_por_imagem.py \
      --lista "/host/lista_eq4_gpu${g}.txt" --saida-local "/host/pi_eq4_gpu${g}" \
      > "$B/final_pi_gpu${g}.log" 2>&1 ) &
done
wait
echo "[final] por imagem: $(grep -h 'OK conferencia' "$B"/final_pi_gpu*.log 2>/dev/null | wc -l) conferem, $(grep -h '!! conferencia' "$B"/final_pi_gpu*.log 2>/dev/null | wc -l) divergem"

for g in $GPUS; do
  ( roda "julia_rif_gpu${g}" "$g" python3 metricas_riemannianas.py \
      --lista "/host/lista_eq4_gpu${g}.txt" --saida-local "/host/ri_eq4_gpu${g}" \
      > "$B/final_ri_gpu${g}.log" 2>&1 ) &
done
wait

roda julia_tabela_f 0 python3 tabela_estilo_paper.py \
  --entrada "/host/pi_eq4_gpu*/por_imagem.parquet" \
  --incluir-repos "bokeh-eq4-,identidade" \
  --saida-md /host/TABELA_EQ4.md \
  --saida-hf juliadollis/genrefocus-tabela-eq4 > "$B/final_tabela.log" 2>&1
echo "[final] tabela rc=$?"

roda julia_sobe_f 0 python3 /host/sobe_eq4.py 2>&1 | grep -E "linhas|enviado|Error"
echo "[final] CONSOLIDACAO FINAL CONCLUIDA $(date -u +%H:%M:%SZ)"
