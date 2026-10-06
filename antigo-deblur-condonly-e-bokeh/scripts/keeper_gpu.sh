#!/usr/bin/env bash
# =============================================================================
# KEEPER: garante que a GPU NUNCA fique ociosa (dgx-H100-01, sem SLURM).
#
# Motivo: esta maquina nao tem escalonador. GPU parada e GPU que o grupo pode
# perder para outra equipe. O keeper roda a fila e, quando ela acaba, volta para
# o inicio em vez de deixar a GPU vaga.
#
# Como nao desperdica: o pipeline do bokeh_net PULA lotes que ja estao no repo
# HF de saida. Uma segunda passada por um item concluido custa segundos (so
# lista o repo e sai), entao o loop e barato ate aparecer trabalho novo.
#
# Para adicionar trabalho: basta EDITAR o arquivo de fila. O keeper le o arquivo
# a cada volta, entao o item novo entra sem reiniciar nada.
#
# Uso:  setsid nohup bash scripts/keeper_gpu.sh 2 /caminho/fila_gpu2.txt &
# Parar: touch /raid/user_juliadollis/julia_docker/PARAR_KEEPER_<gpu>
# =============================================================================
set -uo pipefail

GPU="${1:?uso: keeper_gpu.sh <gpu> <arquivo_de_fila>}"
FILA="${2:?uso: keeper_gpu.sh <gpu> <arquivo_de_fila>}"
B=/raid/user_juliadollis/julia_docker
PARAR="$B/PARAR_KEEPER_${GPU}"
volta=0

echo "[keeper gpu$GPU] iniciado. Para parar: touch $PARAR"
while [ ! -f "$PARAR" ]; do
  volta=$((volta + 1))
  echo "[keeper gpu$GPU] ===== volta $volta — $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  bash "$(dirname "$0")/fila_gpu.sh" "$GPU" "$FILA"
  [ -f "$PARAR" ] && break
  # Se a fila inteira passou rapido, tudo ja estava feito: espera antes de
  # repetir, para nao ficar martelando o Hub.
  # Espera CURTA (60s): o requisito e a GPU nunca ficar ociosa. Uma volta na
  # fila ja concluida custa segundos (a inferencia pula lotes que estao no Hub e
  # as metricas tem guarda anti-duplicata), entao repetir e barato.
  echo "[keeper gpu$GPU] fila esgotada; nova volta em 60s (edite o arquivo de fila para injetar trabalho)"
  sleep 60
done
echo "[keeper gpu$GPU] PARADO por $PARAR — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
