#!/usr/bin/env bash
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
DEST="$B/data/diode_val.tar.gz"
LOG="$B/baixa_diode.log"
URL="https://diode-dataset.s3.amazonaws.com/val.tar.gz"

echo "[baixa] inicio $(date -u +%FT%TZ)" >> "$LOG"
echo "[baixa] url=$URL" >> "$LOG"
echo "[baixa] SEM token HF (repo publico S3 oficial)" >> "$LOG"

curl -L -C - --retry 5 --retry-delay 10 --retry-connrefused \
     -o "$DEST" "$URL" 2>> "$LOG"
rc=$?
echo "[baixa] curl rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
ls -l "$DEST" >> "$LOG" 2>&1

if [ "$rc" -ne 0 ]; then
  echo "[baixa] FALHOU no download" >> "$LOG"; exit 1
fi

echo "[baixa] testando integridade do gzip" >> "$LOG"
if ! gzip -t "$DEST" 2>>"$LOG"; then
  echo "[baixa] gzip corrompido" >> "$LOG"; exit 1
fi

echo "[baixa] extraindo para $B/data/diode $(date -u +%FT%TZ)" >> "$LOG"
tar xzf "$DEST" -C "$B/data/diode" 2>> "$LOG"
echo "[baixa] tar rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "[baixa] topo:" >> "$LOG"
ls "$B/data/diode" >> "$LOG" 2>&1
ls "$B/data/diode/val" >> "$LOG" 2>&1
echo "[baixa] PRONTO $(date -u +%FT%TZ)" >> "$LOG"
