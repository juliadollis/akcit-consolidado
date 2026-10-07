#!/usr/bin/env bash
# Roda as condicoes do Caminho A na rota C sobre o RealDOF. DENTRO do container.
#   FASE=busca  -> base / Bgrad / Bgeod, com calibrate_k (e os dois cruz sem alinhar)
#   FASE=kfixo  -> os mesmos tres bracos com K FIXO (exige K_FIXO=<valor>)
set -uo pipefail

PROJ=/workspace/bokehnet-regen
cd "$PROJ"
export PYTHONPATH=/workspace/pylibs_depth_pro:$PROJ/src:$PROJ/scripts:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

MIRROR=/workspace/caminho_a_rotac/realdof_mirror
RAIZ_OUT="${RAIZ_OUT:?RAIZ_OUT nao definido}"
PILOT_ARG="${PILOT_ARG:-}"
FASE="${FASE:?FASE nao definido}"

CK_BASE=/workspace/models/checkpoints/depth_pro.pt
CK_GRAD=/workspace/caminho_b_analise/ckpt_merged/depth_pro_Bgrad_s0.pt
CK_GEOD=/workspace/caminho_b_analise/ckpt_merged/depth_pro_Bgeod_s0.pt

# nome|checkpoint|alinhar
if [ "$FASE" = "busca" ]; then
  COND=(
    "base_busca|$CK_BASE|1"
    "Bgrad_al_busca|$CK_GRAD|1"
    "Bgeod_al_busca|$CK_GEOD|1"
    "Bgrad_cru_busca|$CK_GRAD|0"
    "Bgeod_cru_busca|$CK_GEOD|0"
  )
  K_ARG=""
else
  : "${K_FIXO:?FASE=kfixo exige K_FIXO}"
  COND=(
    "base_kfixo|$CK_BASE|1"
    "Bgrad_al_kfixo|$CK_GRAD|1"
    "Bgeod_al_kfixo|$CK_GEOD|1"
  )
  K_ARG="--k-fixo $K_FIXO"
fi

for linha in "${COND[@]}"; do
  IFS='|' read -r NOME CK AL <<< "$linha"
  OUT="$RAIZ_OUT/$NOME"
  # O diretorio nasce ANTES do run: alvo inexistente montado vira root e nada escreve.
  mkdir -p "$OUT"
  AL_ARG=""
  [ "$AL" = "1" ] && AL_ARG="--alinhar-afim-com-base --dominio-alinhamento disparidade"
  echo "############ $NOME  ($(date -Is))"
  python3 scripts/run_route_c.py --source realdof \
      --mirror-dir "$MIRROR" \
      --output-dir "$OUT" \
      --models-dir /workspace/models \
      --bokehme-dir third_party/BokehMe \
      --renderer-report output/renderer_verification.json \
      --depth-checkpoint "$CK" \
      --depth-checkpoint-base "$CK_BASE" \
      $AL_ARG $K_ARG --registrar-psnr \
      --device cuda --seed 0 $PILOT_ARG 2>&1 | sed "s/^/[$NOME] /"
  echo "############ $NOME fim rc=${PIPESTATUS[0]} ($(date -Is))"
done
echo "TODAS AS CONDICOES DA FASE $FASE TERMINARAM"
