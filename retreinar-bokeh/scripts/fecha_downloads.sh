#!/usr/bin/env bash
# Fecha o que faltou: os 3 arquivos da rota C e os 90 shards do espelho.
#
# A rota C caiu com 30.849 de 30.852 num `RemoteProtocolError: Server
# disconnected` — uma hora perdida por um soluço de rede no fim. O downloader
# agora retenta erro de rede por arquivo; arquivo ausente e erro de código
# continuam falhando alto.
set -u
L=/workspace/releases/final_download.log
echo "### fecha a rota C" > "$L"
python3 /workspace/retreinar-bokeh/scripts/baixa_por_manifest.py \
  juliadollis/bokehnet-regen-rota-c /workspace/releases/rota_c depth meta >> "$L" 2>&1
echo "### rota C exit=$?" >> "$L"
echo "### espelho RealBokeh, train+validation, ~44 GB" >> "$L"
python3 /workspace/retreinar-bokeh/scripts/baixa_espelho_realbokeh.py >> "$L" 2>&1
echo "### espelho exit=$?" >> "$L"
echo TUDO_PRONTO_FINAL >> "$L"
