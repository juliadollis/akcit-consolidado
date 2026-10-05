#!/usr/bin/env bash
# Espera o download dos releases e ENTÃO baixa os parquets da RealBokeh.
#
# Rodar os dois juntos já falhou hoje: o dos releases faz ~10,6 requisições/s e
# o dos parquets morreu de `LocalEntryNotFoundError` por inanição de conexão.
# São 90 arquivos de ~500 MB (train + validation, ~44 GB) — limitado por banda,
# não por cota, então sozinho ele voa.
#
# Por que é preciso: o snapshot local tinha SÓ o split `test` (1.257 linhas), e
# 94,7% da rota C sai de `train`/`validation`. Ver PLANO §11.7.
set -u
L=/workspace/releases/realbokeh_download.log
echo "esperando o download dos releases terminar..." > "$L"
while ! grep -q TUDO_PRONTO /workspace/releases/download_bc.log 2>/dev/null; do
  sleep 60
done
echo "releases prontos; baixando os parquets da RealBokeh" >> "$L"
python3 /workspace/retreinar-bokeh/scripts/baixa_espelho_realbokeh.py >> "$L" 2>&1
echo "TUDO_PRONTO_REALBOKEH exit=$?" >> "$L"
