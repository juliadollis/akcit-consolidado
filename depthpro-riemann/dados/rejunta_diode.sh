#!/usr/bin/env bash
# Rejunta os blocos do DIODE DEPOIS que todo baixador saiu, e verifica o conteudo.
# A juncao anterior pegou partes ainda sendo escritas: o tamanho total batia, o
# conteudo nao. Aqui a espera e explicita.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
PARTS="$B/data/diode_parts"
DEST="$B/data/diode_val.tar.gz"
URL="https://diode-dataset.s3.amazonaws.com/val.tar.gz"
LOG="$B/rejunta_diode.log"
TOTAL=2774625282
CHUNK=$((32*1024*1024))
N=$(( (TOTAL + CHUNK - 1) / CHUNK ))

echo "[re] inicio $(date -u +%FT%TZ)" >> "$LOG"

# 1. esperar TODO baixador sair
for i in $(seq 1 180); do
  viv=$(pgrep -u "$(id -u)" -f baixa_bloco | wc -l)
  [ "$viv" -eq 0 ] && break
  echo "[re] esperando $viv baixadores $(date -u +%TZ)" >> "$LOG"
  sleep 20
done
echo "[re] baixadores fora, juntando" >> "$LOG"

# 2. conferir tamanho de cada bloco antes de juntar
faltam=0
for i in $(seq 0 $((N-1))); do
  s=$(( i * CHUNK )); e=$(( s + CHUNK - 1 ))
  [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
  esp=$(( e - s + 1 ))
  f=$(printf "%s/part.%04d" "$PARTS" "$i")
  at=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$at" -ne "$esp" ]; then echo "[re] bloco $i tem $at de $esp" >> "$LOG"; faltam=$((faltam+1)); fi
done
echo "[re] blocos com tamanho errado: $faltam" >> "$LOG"

# 3. juntar em ordem numerica explicita
: > "$DEST"
for i in $(seq 0 $((N-1))); do
  cat "$(printf "%s/part.%04d" "$PARTS" "$i")" >> "$DEST"
done
echo "[re] tamanho=$(stat -c %s "$DEST") esperado=$TOTAL" >> "$LOG"

# 4. testar de verdade
if gzip -t "$DEST" 2>>"$LOG"; then
  echo "[re] GZIP OK" >> "$LOG"
else
  echo "[re] GZIP AINDA CORROMPIDO -- achando o bloco ruim" >> "$LOG"
  # onde o fluxo quebra, em bytes de SAIDA; nao mapeia direto para bloco de
  # entrada, mas o md5 de cada bloco contra o S3 mapeia.
  saida=$(gzip -dc "$DEST" 2>/dev/null | wc -c)
  echo "[re] descomprimiu $saida bytes antes de quebrar" >> "$LOG"
  ruins=0
  for i in $(seq 0 $((N-1))); do
    s=$(( i * CHUNK )); e=$(( s + CHUNK - 1 ))
    [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    f=$(printf "%s/part.%04d" "$PARTS" "$i")
    local_md5=$(md5sum "$f" | cut -d" " -f1)
    remoto_md5=$(curl -sS -L -r "${s}-${e}" "$URL" 2>/dev/null | md5sum | cut -d" " -f1)
    if [ "$local_md5" != "$remoto_md5" ]; then
      echo "[re] bloco $i DIVERGE, rebaixando" >> "$LOG"
      curl -sS -L --retry 5 --retry-delay 5 -r "${s}-${e}" -o "$f" "$URL" 2>/dev/null
      ruins=$((ruins+1))
    fi
  done
  echo "[re] blocos divergentes: $ruins" >> "$LOG"
  : > "$DEST"
  for i in $(seq 0 $((N-1))); do cat "$(printf "%s/part.%04d" "$PARTS" "$i")" >> "$DEST"; done
  if gzip -t "$DEST" 2>>"$LOG"; then echo "[re] GZIP OK apos conserto" >> "$LOG"; else echo "[re] AINDA RUIM" >> "$LOG"; fi
fi
echo "[re] fim $(date -u +%FT%TZ)" >> "$LOG"
