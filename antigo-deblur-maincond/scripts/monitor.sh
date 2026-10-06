#!/usr/bin/env bash
# =============================================================================
# Painel de acompanhamento dos treinos GenRefocus (RODE NO CLUSTER).
#
#   bash scripts/monitor.sh                  # snapshot único
#   watch -n 15 "bash scripts/monitor.sh"    # atualiza a cada 15s (recomendado)
#
# Mostra: fila (squeue) + histórico de hoje (sacct, p/ ver FAILED) + ERROS
# recentes nos .err (Traceback/CUDA/OOM) + as últimas linhas dos .out ativos.
# Assim, se algo falhar, você vê NA HORA o quê e por quê.
# =============================================================================
set -uo pipefail   # sem -e de propósito: continua mesmo se um comando falhar

LOGDIR="${LOGDIR:-/raid/user_juliadollis/projects/genrefocus_deblurnet/logs}"
ME="${USER:-user_juliadollis}"

echo "######## GenRefocus monitor — $(date '+%Y-%m-%d %H:%M:%S') ########"
echo
echo "===================== FILA (squeue) ====================="
squeue -u "$ME" -o "%.10i %.20j %.9T %.11M %.11L %.5D %R" 2>/dev/null \
    || echo "(squeue indisponível)"

echo
echo "=============== HISTÓRICO HOJE (sacct) =================="
# só os jobs "pai" (tira .batch/.extern); destaca quem FALHOU
sacct -u "$ME" --starttime=today \
    --format="JobID%14,JobName%22,State%13,Elapsed,ExitCode" 2>/dev/null \
    | grep -vE "\.batch|\.extern|\.0 " || echo "(sem histórico hoje)"

echo
echo "================= ERROS RECENTES (.err) ================="
found_err=0
for f in $(ls -t "$LOGDIR"/*.err 2>/dev/null | head -8); do
    hits=$(grep -nEi "traceback|error|cuda|out of memory|oom|failed|exception|assert" "$f" 2>/dev/null | tail -6)
    if [[ -n "$hits" ]]; then
        echo "--- $(basename "$f") ---"
        echo "$hits"
        echo
        found_err=1
    fi
done
[[ $found_err -eq 0 ]] && echo "(nenhum erro nos .err recentes ✅)"

echo
echo "=========== ÚLTIMAS LINHAS DOS LOGS ATIVOS (.out) ==========="
for f in $(ls -t "$LOGDIR"/*.out 2>/dev/null | head -3); do
    echo "--- $(basename "$f") ---"
    tail -n 8 "$f" 2>/dev/null
    echo
done
