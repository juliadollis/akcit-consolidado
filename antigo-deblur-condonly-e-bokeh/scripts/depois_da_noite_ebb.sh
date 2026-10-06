#!/usr/bin/env bash
# Depois que a cadeia da noite fechar, roda os 6 modelos restantes no EBB400.
#
# Fica DEPOIS de proposito: durante a cadeia as 7 GPUs sao usadas pelas passadas
# de metrica, e subir inferencia junto so faria as duas coisas ficarem lentas.
# A prioridade da noite e a tabela; o EBB completo e o dia seguinte.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
P=$B/genrefocus_deblurnet_paper
echo "[pos] aguardando a cadeia da noite $(date -u +%H:%M:%SZ)"
until grep -q "CONCLUIDO" "$B/pipeline_noite.log" 2>/dev/null; do sleep 300; done
echo "[pos] cadeia fechou, subindo o EBB dos outros 6 modelos $(date -u +%H:%M:%SZ)"
cd "$P"
gpus=(0 1 3 5 6 7)
i=0
while IFS= read -r linha; do
  [ -z "$linha" ] && continue
  g=${gpus[$((i % 6))]}; i=$((i+1))
  echo "$linha" > "$B/filas_eq4/fila_pos_gpu${g}.txt"
done < "$B/filas_eq4/fila_ebb_resto.txt"
for g in "${gpus[@]}"; do
  [ -f "$B/filas_eq4/fila_pos_gpu${g}.txt" ] || continue
  setsid nohup bash scripts/fila_gpu_eq4.sh "$g" "$B/filas_eq4/fila_pos_gpu${g}.txt" \
    > "$B/pos_gpu${g}_wrap.log" 2>&1 < /dev/null &
done
echo "[pos] lancado $(date -u +%H:%M:%SZ)"
