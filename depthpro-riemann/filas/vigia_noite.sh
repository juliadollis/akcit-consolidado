#!/usr/bin/env bash
# Registro continuo do estado, a cada 5 min, num arquivo unico e legivel.
# Existe para haver uma LINHA DO TEMPO de manha, mesmo que ninguem estivesse
# olhando: qual etapa estava rodando, quando cada item fechou, e se em algum
# momento as GPUs ficaram paradas com a cadeia inacabada.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
LOG=$B/VIGIA_NOITE.log
echo "[vigia] inicio $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
while true; do
  ts=$(date -u +%H:%M:%SZ)
  conts=$(docker ps --format '{{.Names}}' | grep -c '^julia_' || true)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
  etapa=$(grep -o '\[noite\] [0-9]/6[^$]*' "$B/pipeline_noite.log" 2>/dev/null | tail -1)
  concl=$(grep -c CONCLUIDO "$B/pipeline_noite.log" 2>/dev/null || echo 0)
  echo "$ts | containers=$conts | util=[$util] | ${etapa:-sem etapa}" >> "$LOG"
  # anomalia: nada rodando e a cadeia nao terminou
  if [ "$conts" -eq 0 ] && [ "$concl" -eq 0 ]; then
    echo "$ts | *** ALERTA: nenhum container nosso e a cadeia nao concluiu ***" >> "$LOG"
  fi
  [ "$concl" -gt 0 ] && { echo "$ts | cadeia CONCLUIDA, vigia encerrando" >> "$LOG"; break; }
  sleep 300
done
