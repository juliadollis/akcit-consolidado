#!/usr/bin/env bash
# =============================================================================
# VIGIA: religa keeper morto e mata keeper duplicado, para nenhuma GPU nossa
# ficar ociosa nem duas filas disputarem a mesma GPU.
#
# Motivo (2026-08-20): os keepers das GPUs 3 e 5 morreram silenciosamente junto
# com uma queda de sessao e as duas ficaram paradas ate alguem notar. Depois,
# ao religa-los na mao, sobrou um keeper DUPLICADO na GPU 2. O keeper sozinho
# nao se auto-recupera nem detecta gemeos; este vigia faz as duas coisas.
#
# NAO religa GPU com a flag PARAR_KEEPER_<n> (usada pela cadeia do treino para
# tomar a GPU de proposito).
#
# Uso:  setsid nohup bash scripts/vigia_keepers.sh "2 3 5" &
# Parar: touch /raid/user_juliadollis/julia_docker/PARAR_VIGIA
# =============================================================================
set -uo pipefail
GPUS="${1:?uso: vigia_keepers.sh \"2 3 5\"}"
B=/raid/user_juliadollis/julia_docker
PROJ="$B/genrefocus_deblurnet_paper"
PARAR="$B/PARAR_VIGIA"

echo "[vigia] iniciado para as GPUs: $GPUS. Parar: touch $PARAR"
while [ ! -f "$PARAR" ]; do
  for g in $GPUS; do
    [ -f "$B/PARAR_KEEPER_$g" ] && continue
    pids=$(pgrep -u "$(id -u)" -f "keeper_gpu.sh $g" | sort -n)
    n=$(echo "$pids" | grep -c . || true)
    if [ "${n:-0}" -eq 0 ]; then
      echo "[vigia] $(date -u +%H:%M:%SZ) keeper da GPU $g MORTO — religando"
      ( cd "$PROJ" && setsid nohup bash scripts/keeper_gpu.sh "$g" \
          "$PROJ/filas/fila_gpu$g.txt" >> "$B/keeper_gpu$g.log" 2>&1 < /dev/null & )
    elif [ "$n" -gt 1 ]; then
      extras=$(echo "$pids" | tail -n +2)
      echo "[vigia] $(date -u +%H:%M:%SZ) GPU $g com $n keepers — encerrando extras: $(echo $extras)"
      for p in $extras; do kill "$p" 2>/dev/null || true; done
    fi
  done
  sleep 120
done
echo "[vigia] PARADO por $PARAR — $(date -u +%H:%M:%SZ)"
