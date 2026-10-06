#!/usr/bin/env bash
# =============================================================================
# Submete uma CADEIA de jobs curtos encadeados (SLURM --dependency=afterany).
# Cada pedaço:
#   - entra rápido por BACKFILL (tempo curto cabe na fila cheia);
#   - retoma do último checkpoint (resume:true);
#   - começa rápido (o filtro do RealBokeh é cacheado após o 1º pedaço).
# Você submete UMA vez e ele roda sozinho em sequência — sem ficar de babá.
#
# Uso:
#   bash scripts/train_chain.sh [SLURM_SCRIPT] [CHUNK_TIME] [N_CHUNKS]
# Ex.:
#   bash scripts/train_chain.sh scripts/train_2gpu.slurm 04:00:00 8
#   bash scripts/train_chain.sh scripts/train_4gpu.slurm 04:00:00 6
#
# Quando o treino convergir / completar os steps, cancele os pedaços restantes:
#   scancel -n deblurnet-2gpu     (ou o job-name do script)
# =============================================================================
set -euo pipefail

SCRIPT="${1:-scripts/train_2gpu.slurm}"
CHUNK="${2:-04:00:00}"
N="${3:-8}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERRO: script não encontrado: $SCRIPT"; exit 1
fi

echo ">>> Cadeia: ${N} pedaços de ${CHUNK} usando ${SCRIPT}"
prev=""
for i in $(seq 1 "$N"); do
    if [[ -z "$prev" ]]; then
        jid=$(sbatch --parsable --time="$CHUNK" "$SCRIPT")
    else
        # afterany: o próximo começa quando o anterior TERMINA (por tempo, erro
        # ou conclusão) — e retoma do checkpoint.
        jid=$(sbatch --parsable --dependency=afterany:"$prev" --time="$CHUNK" "$SCRIPT")
    fi
    echo "  pedaço $i/$N -> job $jid (time=$CHUNK, dep=${prev:-nenhuma})"
    prev="$jid"
done

echo ""
echo ">>> Cadeia submetida. O 1º pedaço entra por backfill; os outros esperam (Dependency)."
echo ">>> Acompanhe:  squeue -u \$USER"
echo ">>> Quando convergir, pare o resto:  scancel -n \$(grep -m1 job-name $SCRIPT | sed 's/.*=//')"
