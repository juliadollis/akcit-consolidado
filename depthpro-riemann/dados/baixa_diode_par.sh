#!/usr/bin/env bash
# Baixa o DIODE val.tar.gz oficial em blocos paralelos (S3 limita ~25 KB/s por conexao).
# Resumivel: bloco ja completo no disco e pulado.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
URL="https://diode-dataset.s3.amazonaws.com/val.tar.gz"
PARTS="$B/data/diode_parts"
DEST="$B/data/diode_val.tar.gz"
LOG="$B/baixa_diode.log"
TOTAL=2774625282
CHUNK=$((32*1024*1024))
JOBS=32

mkdir -p "$PARTS"
N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
echo "[par] inicio $(date -u +%FT%TZ) total=$TOTAL blocos=$N jobs=$JOBS" >> "$LOG"

baixa_bloco () {
  i="$1"
  B=/raid/user_juliadollis/julia_docker
  URL="https://diode-dataset.s3.amazonaws.com/val.tar.gz"
  PARTS="$B/data/diode_parts"
  TOTAL=2774625282
  CHUNK=$((32*1024*1024))
  s=$(( i * CHUNK ))
  e=$(( s + CHUNK - 1 ))
  [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
  esperado=$(( e - s + 1 ))
  f=$(printf "%s/part.%04d" "$PARTS" "$i")
  atual=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$atual" -eq "$esperado" ] && return 0
  for tent in 1 2 3 4 5 6 7 8; do
    curl -sS -L --retry 3 --retry-delay 5 --speed-time 60 --speed-limit 1000 \
         -r "${s}-${e}" -o "$f" "$URL" 2>/dev/null
    atual=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$atual" -eq "$esperado" ] && return 0
    sleep 5
  done
  echo "[par] BLOCO $i FALHOU (tem $atual de $esperado)" >> "$B/baixa_diode.log"
  return 1
}
export -f baixa_bloco

for volta in 1 2 3 4 5; do
  seq 0 $((N-1)) | xargs -P "$JOBS" -I{} bash -c 'baixa_bloco {}'
  faltam=0
  for i in $(seq 0 $((N-1))); do
    s=$(( i * CHUNK )); e=$(( s + CHUNK - 1 ))
    [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    esperado=$(( e - s + 1 ))
    f=$(printf "%s/part.%04d" "$PARTS" "$i")
    atual=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$atual" -ne "$esperado" ] && faltam=$((faltam+1))
  done
  soma=$(du -sb "$PARTS" | cut -f1)
  echo "[par] volta $volta: faltam $faltam blocos, baixado ~$soma bytes $(date -u +%FT%TZ)" >> "$LOG"
  [ "$faltam" -eq 0 ] && break
done

if [ "$faltam" -ne 0 ]; then
  echo "[par] DESISTI: $faltam blocos faltando" >> "$LOG"; exit 1
fi

echo "[par] juntando blocos $(date -u +%FT%TZ)" >> "$LOG"
cat $(for i in $(seq 0 $((N-1))); do printf "%s/part.%04d " "$PARTS" "$i"; done) > "$DEST"
tam=$(stat -c %s "$DEST")
echo "[par] tamanho final=$tam esperado=$TOTAL" >> "$LOG"
[ "$tam" -ne "$TOTAL" ] && { echo "[par] TAMANHO ERRADO" >> "$LOG"; exit 1; }

echo "[par] gzip -t $(date -u +%FT%TZ)" >> "$LOG"
gzip -t "$DEST" 2>>"$LOG" || { echo "[par] GZIP CORROMPIDO" >> "$LOG"; exit 1; }

echo "[par] extraindo $(date -u +%FT%TZ)" >> "$LOG"
tar xzf "$DEST" -C "$B/data/diode" 2>>"$LOG"
echo "[par] tar rc=$? $(date -u +%FT%TZ)" >> "$LOG"
ls "$B/data/diode/val" >> "$LOG" 2>&1
echo "[par] PRONTO $(date -u +%FT%TZ)" >> "$LOG"
