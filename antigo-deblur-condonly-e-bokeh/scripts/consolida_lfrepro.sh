#!/usr/bin/env bash
# CONSOLIDACAO DA MESA LF-REPRO, EM DUAS PASSADAS.
#
# Passada 1 dispara quando os SEIS itens que estao no ar terminam (por volta das
# 09:00Z). Passada 2 dispara quando o `semtreino`, que ficou de resto na GPU 1,
# terminar (por volta das 21:00Z).
#
# Por que nao esperar tudo de uma vez: a GPU 1 pegou dois itens e as outras
# cinco pegaram um. Esperar a fila inteira deixaria a tabela presa por 12 h por
# causa de UMA linha, a do FLUX cru, que e so o piso da mesa e ja existe medida
# na campanha anterior. A passada 1 entrega a tabela util de manha; a passada 2
# so acrescenta essa linha.
#
# Por que a espera olha os WRAPPERS e nao o `docker ps`: entre um item e outro da
# GPU 1 existe um intervalo de segundos sem container nenhum. Quem olhasse so o
# `docker ps` acharia que a fila acabou e consolidaria com o item pela metade.
#
# Nao apaga nada: escreve logs novos (lf1_*, lf2_*) e reescreve apenas a
# TABELA_EQ4.md, que e arquivo GERADO.
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

consolida() {
  local e=$1

  # folga para o ultimo upload de imagens ao Hub assentar antes de listar os repos
  sleep 120

  echo "[lf/$e] montando listas"
  roda "julia_listas_lf${e}" 0 python3 /host/monta_listas_eq4.py 2>&1 | grep -E "repos|identidade|listas"

  echo "[lf/$e] metrica por imagem $(date -u +%H:%M:%SZ)"
  for g in $GPUS; do
    ( roda "julia_pilf${e}_gpu${g}" "$g" python3 metricas_por_imagem.py \
        --lista "/host/lista_eq4_gpu${g}.txt" --saida-local "/host/pi_eq4_gpu${g}" \
        > "$B/lf${e}_pi_gpu${g}.log" 2>&1 ) &
  done
  wait
  echo "[lf/$e] por imagem: $(grep -h 'OK conferencia' "$B"/lf${e}_pi_gpu*.log 2>/dev/null | wc -l) conferem, $(grep -h '!! conferencia' "$B"/lf${e}_pi_gpu*.log 2>/dev/null | wc -l) divergem"

  echo "[lf/$e] metricas riemannianas $(date -u +%H:%M:%SZ)"
  for g in $GPUS; do
    ( roda "julia_rilf${e}_gpu${g}" "$g" python3 metricas_riemannianas.py \
        --lista "/host/lista_eq4_gpu${g}.txt" --saida-local "/host/ri_eq4_gpu${g}" \
        > "$B/lf${e}_ri_gpu${g}.log" 2>&1 ) &
  done
  wait

  echo "[lf/$e] tabela"
  roda "julia_tabela_lf${e}" 0 python3 tabela_estilo_paper.py \
    --entrada "/host/pi_eq4_gpu*/por_imagem.parquet" \
    --incluir-repos "bokeh-eq4-,identidade" \
    --saida-md /host/TABELA_EQ4.md \
    --saida-hf juliadollis/genrefocus-tabela-eq4 > "$B/lf${e}_tabela.log" 2>&1
  echo "[lf/$e] tabela rc=$?"

  roda "julia_sobe_lf${e}" 0 python3 /host/sobe_eq4.py 2>&1 | grep -E "linhas|enviado|Error"
  echo "[lf/$e] PASSADA $e CONCLUIDA $(date -u +%H:%M:%SZ)"
}

echo "[lf] passada 1: aguardando os seis itens no ar $(date -u +%H:%M:%SZ)"
while true; do
  prontas=0
  for g in 2 3 5 6 7; do
    grep -q "FILA COMPLETA" "$B/lfrepro_gpu${g}_wrap.log" 2>/dev/null && prontas=$((prontas+1))
  done
  # a GPU 1 nao fecha a fila aqui: ela ainda tem o semtreino pela frente, entao
  # o sinal dela e o fim do PRIMEIRO item
  grep -q "<<< eq4-rotac60k-lfrepro rc=" "$B/lfrepro_gpu1_wrap.log" 2>/dev/null && prontas=$((prontas+1))
  [ "$prontas" -eq 6 ] && break
  sleep 120
done
echo "[lf] os seis fecharam $(date -u +%H:%M:%SZ)"
consolida 1

echo "[lf] passada 2: aguardando o semtreino na GPU 1 $(date -u +%H:%M:%SZ)"
until grep -q "FILA COMPLETA" "$B/lfrepro_gpu1_wrap.log" 2>/dev/null; do sleep 300; done
echo "[lf] semtreino fechou $(date -u +%H:%M:%SZ)"
consolida 2
echo "[lf] CONSOLIDACAO LF-REPRO CONCLUIDA (duas passadas) $(date -u +%H:%M:%SZ)"
