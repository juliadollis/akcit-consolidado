#!/bin/bash
#SBATCH --job-name=condgeom-stage-release
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# STAGING DO RELEASE PARA A BANCADA DE CONDICIONAMENTO GEOMETRICO. SEM GPU.
# =============================================================================
#
# O QUE FAZ
#   Roda `scripts/stage_release_n3.py`, que traz do Hub SO o pedaco do release
#   que a bancada precisa: `depth/` e `meta/` INTEIROS (o insumo geometrico e a
#   escala metrica, ~1 GB juntos) + os arquivos de topo + `generated/`/`mask/`
#   de um numero de CENAS parametrizado. O release completo e ~30 GB e nao cabe.
#
# SEM --gres, DE PROPOSITO
#   E trabalho de rede e CPU. Ocupar uma GPU para baixar JPEG desperdica a GPU e
#   — pior — queima uma das DUAS vagas da QOS `onejob`, que o treino precisa.
#
# DOIS PASSOS, E POR QUE VALE A PENA
#   Por default este job so PLANEJA (BAIXAR=0): lista o repo pela API, estima o
#   volume e imprime a tabela. Nada e escrito alem do `split.json`. So depois de
#   ler o numero voce resubmete com BAIXAR=1.
#   Motivo: ha registro de 552 GB de egress inexplicado neste projeto vindo de
#   download automatico, e a COTA DO RAID ESTA ESTOURADA (520G de 500G soft,
#   duro 600G, grace correndo). Disco cheio ja matou 3 treinos aqui, ~18 h de
#   3 GPUs. Planejar custa ~1 minuto de fila.
#
# `--cenas` CONTA CENA, NAO ARQUIVO
#   Cada cena traz ~41 amostras em `generated/` e ~41 em `mask/`. CENAS=24 sao
#   ~1.000 JPEGs, nao 24. A confusao "o limite conta por arquivo ou no total?"
#   ja custou 20 h de GPU neste cluster.
#
# NAO APAGA NADA. Sem `rm`, sem `--delete`, sem `scancel`. O script e IDEMPOTENTE:
# arquivo ja em disco com o tamanho do remoto nao e rebaixado, e o
# `staging_manifest.json` ACUMULA as execucoes em vez de sobrescrever.
#
# COMO RODAR
#   ssh dgx-H100-03
#   cd /raid/user_danielpedrozo/projects/julia/retreinar-bokeh-condgeom
#   mkdir -p logs
#   # 1) planejar (le a tabela de volume no log)
#   CENAS=24 sbatch slurm/condgeom_stage_release_n3.slurm
#   # 2) baixar de verdade
#   CENAS=24 BAIXAR=1 CONFIRMAR=1 sbatch slurm/condgeom_stage_release_n3.slurm
#   # so o insumo geometrico, sem pixel nenhum (~1 GB):
#   CENAS=0 BAIXAR=1 sbatch slurm/condgeom_stage_release_n3.slurm
#
# VARIAVEIS
#   CENAS       quantas CENAS em generated/ e mask/ (default 24; 0 = nenhuma)
#   CENAS_VAL   quantas dessas vem do lado val do split (default: proporcional)
#   SEMENTE     semente da amostragem deterministica (default 20260922)
#   TETO_GB     acima disto exige CONFIRMAR=1 (default 20)
#   BAIXAR=1    opt-in do download (default 0 = so plano)
#   CONFIRMAR=1 autoriza passar do teto
#   DESTINO     pasta de staging (FORA do projeto: o rsync do codigo nao deve
#               carregar GB de JPEG)
#   EXTRA       argumentos crus para o script (ex.: "--sem-mask")
#
# CRITERIO DE SUCESSO
#   rc=0 e, no modo BAIXAR=1, a linha "[stage] OK. Aponte o YAML/--release para:".
#   rc=3 espaco/cota; rc=4 sha256 divergente; rc=5 acima do teto sem CONFIRMAR;
#   rc=6 release incompleto para o que foi pedido.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_danielpedrozo}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/julia/retreinar-bokeh-condgeom}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
DESTINO_PADRAO="${BIND_ROOT}/projects/julia/releases/rota_a_bancada"

# Precedencia declarada no .env.example: ambiente > .env > default do script.
# O `. .env` sobrescreve variavel ja exportada, entao guardamos o que veio na
# linha do sbatch ANTES de carregar o .env e restauramos depois. Sem isto um
# CENAS esquecido dentro do .env venceria o CENAS=24 do comando — e o numero de
# cenas e exatamente o que decide quantos GB descem.
for _v in REPO DESTINO CENAS CENAS_VAL SEMENTE TETO_GB BAIXAR CONFIRMAR EXTRA; do
  eval "_cli_${_v}=\${${_v}:-}"
done

cd "${PROJECT_ROOT}"
mkdir -p logs

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done

for _v in REPO DESTINO CENAS CENAS_VAL SEMENTE TETO_GB BAIXAR CONFIRMAR EXTRA; do
  eval "_val=\${_cli_${_v}}"
  if [ -n "${_val}" ]; then eval "${_v}=\"\${_val}\""; fi
done

# O release e PRIVADO: sem token a listagem volta 401 e o job morre na hora
# certa, antes da fila. O valor nunca e impresso.
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. cp .env.example .env && edite (chmod 600 .env)}"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${BIND_ROOT}/projects/julia/hf-cache}"

REPO="${REPO:-juliadollis/bokehnet-regen-rota-a}"
DESTINO="${DESTINO:-${DESTINO_PADRAO}}"
CENAS="${CENAS:-24}"
SEMENTE="${SEMENTE:-20260922}"
TETO_GB="${TETO_GB:-20}"
BAIXAR="${BAIXAR:-0}"
CONFIRMAR="${CONFIRMAR:-0}"
EXTRA="${EXTRA:-}"

ARGS="--repo ${REPO} --destino ${DESTINO} --cenas ${CENAS} --semente ${SEMENTE}"
ARGS="${ARGS} --teto-gb ${TETO_GB}"
[ -n "${CENAS_VAL:-}" ] && ARGS="${ARGS} --cenas-val ${CENAS_VAL}"
[ "${BAIXAR}" = "1" ]    && ARGS="${ARGS} --baixar"
[ "${CONFIRMAR}" = "1" ] && ARGS="${ARGS} --confirmar"
[ -n "${EXTRA}" ]        && ARGS="${ARGS} ${EXTRA}"

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }
[ -f "${PROJECT_ROOT}/scripts/stage_release_n3.py" ] || {
  echo "ERRO: scripts/stage_release_n3.py ausente. Confira o rsync."; exit 1; }

echo "=============================================================="
echo "Job:      ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-local}) @ $(hostname)"
echo "Projeto:  ${PROJECT_ROOT}"
echo "Repo:     ${REPO}"
echo "Destino:  ${DESTINO}"
echo "Args:     ${ARGS}"
echo "Modo:     $([ "${BAIXAR}" = "1" ] && echo 'BAIXAR' || echo 'SO PLANO (BAIXAR=1 para baixar)')"
echo "GPU:      nenhuma (job de rede/CPU, de proposito)"
echo "--- cota ANTES (a do raid esta estourada: 520G de 500G soft) ---"
quota -s 2>/dev/null | tail -3 || echo "(comando quota indisponivel)"
df -h "$(dirname "${DESTINO}")" 2>/dev/null | tail -2 || true
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
# PYTHONPATH so com codigo do projeto, e montado AQUI DENTRO. Passar
# site-packages por --env PYTHONPATH tira o ~/.local do sys.path e some com
# diffusers/peft (INSTRUCOES_H100.md).
export PYTHONPATH='"${PROJECT_ROOT}"'${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

python3 -c "import huggingface_hub as h; print(\"[env] huggingface_hub\", h.__version__)" \
  || echo "[env] huggingface_hub ausente — o script cai na API HTTP sozinho."

exec python3 scripts/stage_release_n3.py '"${ARGS}"'
'

set +e
srun singularity exec \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  "${SIF}" bash -lc "${APP_CMD}"
RC=$?   # capturado ANTES do `quota`, senao seria o rc do `|| true`
set -e

echo "--- cota DEPOIS ---"
quota -s 2>/dev/null | tail -3 || true
du -sh "${DESTINO}" 2>/dev/null || true
echo "=== FIM (rc=${RC}) ==="
exit "${RC}"
