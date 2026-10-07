#!/usr/bin/env bash
# FISCAL: nao deixa a GPU ociosa enquanto houver avaliacao pendente.
#   fiscal_avaliacao.sh <gpu>
#
# Laco: espera a placa vagar, varre TODO checkpoint em disco contra TODA mesa
# disponivel, e avalia o que ainda nao tem test_metrics.json. O avalia_modelos.py
# pula par ja feito, entao rodar de novo e barato e idempotente.
#
# Conforme novos checkpoints chegam (download do Hub) ou novas mesas ficam
# prontas (DIODE), o fiscal os pega na volta seguinte, sem relancamento.
#
# Dois fiscais podem rodar juntos: a colisao e possivel mas inofensiva, porque o
# pior caso e avaliar o mesmo par duas vezes, e o segundo sobrescreve com o mesmo
# numero. O treino e que exigia reivindicacao atomica; avaliacao nao.
set -uo pipefail
GPU="${1:?uso: fiscal_avaliacao.sh <gpu>}"
B=/raid/user_juliadollis/julia_docker
R="$B/depth-riemannian"
LOG="$B/fiscal_gpu${GPU}.log"
MIN_LIVRE_MIB=${MIN_LIVRE_MIB:-40000}
VOLTAS=${VOLTAS:-200}
MODO=${MODO:-nossa}   # nossa | emprestada

mesas () {
  local m=()
  [ -d "$B/data/spring_split/test" ] && m+=(--dataset "spring_test=/data/spring_split/test")
  [ -d "$B/data/diode_prep/val" ]    && m+=(--dataset "diode_val=/data/diode_prep/val")
  [ -d "$B/data/spring_split/val" ]  && m+=(--dataset "spring_val=/data/spring_split/val")
  printf '%s\n' "${m[@]}"
}

echo "[fiscal gpu$GPU] inicio $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
for volta in $(seq 1 "$VOLTAS"); do
  mapfile -t ARGS < <(mesas)
  if [ "${#ARGS[@]}" -eq 0 ]; then
    echo "[fiscal gpu$GPU] nenhuma mesa pronta; esperando" >> "$LOG"; sleep 600; continue
  fi
  n_ckpt=$(ls -d "$B"/runs_*/*/seed_*/best.pt 2>/dev/null | wc -l)
  if [ "$n_ckpt" -eq 0 ]; then
    echo "[fiscal gpu$GPU] nenhum checkpoint em disco; esperando" >> "$LOG"; sleep 600; continue
  fi

  # DUAS POLITICAS, e a diferenca importa.
  #
  # MODO=nossa (GPU 5): basta ter memoria livre. A placa e do projeto.
  #
  # MODO=emprestada (GPU 6): so entra se a placa estiver VAZIA, sem processo de
  # computacao de ninguem. Checar memoria livre NAO basta: a 6 tinha 57 GB livres
  # porque o job do vizinho usa 24 dos 80, e ao entrar ali deixamos a placa dele
  # com 9 GB. Guarda de memoria protege o NOSSO job de tomar OOM, nao impede o
  # nosso job de CAUSAR OOM em quem ja estava. Ja erramos isso duas vezes.
  e=0
  while true; do
    livre=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    alheios=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ "$MODO" = "emprestada" ]; then
      [ "${alheios:-1}" -eq 0 ] && break
      [ $((e % 12)) -eq 0 ] && echo "[fiscal gpu$GPU] emprestada: $alheios processo(s) de terceiros; nao entro $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
    else
      [ "${livre:-0}" -ge "$MIN_LIVRE_MIB" ] && break
      [ $((e % 12)) -eq 0 ] && echo "[fiscal gpu$GPU] ocupada (${livre} MiB); aguardando $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
    fi
    e=$((e+1)); sleep 300
  done

  antes=$(find "$B/avaliacoes" -name test_metrics.json 2>/dev/null | wc -l)
  echo "[fiscal gpu$GPU] volta $volta: $n_ckpt ckpts, ${#ARGS[@]} args de mesa, $antes pares feitos $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  docker rm "julia_fiscal_gpu${GPU}" >/dev/null 2>&1 || true
  docker run --rm --name "julia_fiscal_gpu${GPU}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
    -v "$R/riemann":/workspace/riemann:ro -v "$B/data":/data -v "$B/models":/models \
    -v "$B":/host -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w /workspace \
    riemann-depthpro:latest \
    python3 /host/avalia_modelos.py --checkpoints "/host/runs_*/*/seed_*/best.pt" \
      "${ARGS[@]}" --checkpoint-base /models/checkpoints/depth_pro.pt \
      --saida /host/avaliacoes --incluir-zero-shot >> "$B/fiscal_run_gpu${GPU}.log" 2>&1
  depois=$(find "$B/avaliacoes" -name test_metrics.json 2>/dev/null | wc -l)
  echo "[fiscal gpu$GPU] volta $volta fim: $antes -> $depois pares $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
  # nada novo nesta volta: espera antes de varrer de novo
  [ "$depois" -eq "$antes" ] && sleep "${ESPERA_OCIOSA:-900}"
done
echo "[fiscal gpu$GPU] FIM $(date -u +%FT%H:%M:%SZ)" >> "$LOG"
