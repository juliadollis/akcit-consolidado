#!/bin/bash
#SBATCH --job-name=condgeom-calibrar
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
# SEM --gres: este job e numpy+PIL, nao toca GPU. Pedir uma gastaria
# uma das 2 vagas da QOS `onejob` que os TREINOS precisam.
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# CALIBRA AS CONSTANTES DE NORMALIZACAO DOS CANAIS GEOMETRICOS. 1 GPU.
# =============================================================================
#
# O QUE FAZ
#   Mede, sobre o CONJUNTO DE TREINO, as constantes fixas que normalizam os
#   canais do `geocond`: o tau do mapa de oclusao, o teto de `u`, a faixa de
#   `s`, o `k0` da compressao logaritmica da curvatura e o teto simetrico de
#   `K~`. Elas viram um JSON que entra no config do run.
#
# POR QUE CONSTANTE, E NAO PERCENTIL POR IMAGEM
#   `geocond/contrato.py`, convencao 3: sob crop aleatorio o percentil do
#   recorte nao bate com o da imagem inteira, e treino e inferencia veriam
#   normalizacoes diferentes. Numa cena sem descontinuidade real o percentil
#   vira ruido e o mapa de oclusao satura de quadro cheio.
#
# SOBRE O LADO DO SPLIT — NAO NEGOCIAVEL
#   A calibracao le o lado **train** do `split.json`. Constante medida com cena
#   de val e vazamento: a validacao passa a medir uma normalizacao ajustada a
#   ela mesma. Se/quando o entrypoint ganhar uma flag de split, passe por
#   EXTRA="--split train".
#
# O QUE ELE *NAO* PRECISA
#   `generated/`. A calibracao le `depth/` + `meta/`, que sao ~1 GB. Entao este
#   job roda em cima de um staging feito com CENAS=0 — util enquanto a cota do
#   raid estiver estourada.
#
#   ATENCAO — CONTRATO A CUMPRIR, NAO CODIGO EXISTENTE:
#   `geocond.jobs.calibrar` AINDA NAO EXISTE (o pacote `geocond/` esta sendo
#   escrito em paralelo). A assinatura assumida aqui e:
#
#       python3 -m geocond.jobs.calibrar --release <dir> --amostras N --out <json>
#
#   Isto e um CONTRATO: quem escrever o modulo tem que expor exatamente estas
#   tres flags, ou este .slurm deixa de valer e tem que ser corrigido junto.
#   O job abaixo falha alto e cedo se o modulo nao estiver importavel, em vez de
#   segurar a GPU ate o timeout.
#
# NAO APAGA NADA. Sem `rm`, sem `--delete`, sem `scancel`.
#
# QOS `onejob`: 2 jobs simultaneos. Confira antes de submeter:
#   ssh dgx-H100-03 'squeue -u $USER -o "%.8i %.10P %.16j %.2t %.12M %R"'
#
# COMO RODAR
#   ssh dgx-H100-03
#   cd /raid/user_danielpedrozo/projects/julia/retreinar-bokeh-condgeom
#   mkdir -p logs outputs
#   sbatch slurm/condgeom_calibrar_constantes_n3.slurm
#
# ACOMPANHAR
#   tail -n +1 -f logs/condgeom-calibrar-<JOBID>.out logs/condgeom-calibrar-<JOBID>.err
#
# VARIAVEIS
#   RELEASE_DIR  staging do release (default: o do stage_release_n3)
#   AMOSTRAS     quantas amostras entram na estatistica (default 2000)
#   OUT          JSON de saida (default outputs/condgeom/constantes_<job>.json)
#   EXTRA        flags cruas para o entrypoint
#
# CRITERIO DE SUCESSO
#   rc=0 e o JSON em OUT existindo, com uma constante por chave e NENHUMA nula.
#   Constante nula e pior que erro: vira default plausivel silencioso, que e a
#   classe de defeito que esta campanha inteira persegue.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_danielpedrozo}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/julia/retreinar-bokeh-condgeom}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"

# Precedencia: linha do sbatch > .env > default. Ver o comentario gemeo em
# condgeom_stage_release_n3.slurm.
for _v in RELEASE_DIR AMOSTRAS OUT EXTRA; do eval "_cli_${_v}=\${${_v}:-}"; done

cd "${PROJECT_ROOT}"
mkdir -p logs outputs/condgeom

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done

for _v in RELEASE_DIR AMOSTRAS OUT EXTRA; do
  eval "_val=\${_cli_${_v}}"
  if [ -n "${_val}" ]; then eval "${_v}=\"\${_val}\""; fi
done

export HF_HOME="${HF_HOME:-${BIND_ROOT}/projects/julia/hf-cache}"
# Leitura e de DISCO. O release ja foi baixado pelo job de staging; se algum
# caminho tentar rede, que falhe alto em vez de repetir os 552 GB de egress
# inexplicado que a auditoria deste projeto registrou.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# O release da rota A foi GERADO nesta maquina e esta em disco: 57 GB,
# 69.700 amostras, dono user_juliadollis (ja contam na cota). Ler no
# lugar custa ZERO disco novo — e a cota esta em 520G de 500G.
RELEASE_DIR="${RELEASE_DIR:-${BIND_ROOT}/projects/julia/bokehnet-regen/output/a_release}"
# 300 e o suficiente: o pool e de PIXEIS (300 x 20k = 6M por grandeza),
# e o p99.5 sobre 6M valores e estavel. 2000 amostras custariam 7x o
# tempo para mudar a terceira casa.
AMOSTRAS="${AMOSTRAS:-300}"
OUT="${OUT:-${PROJECT_ROOT}/outputs/condgeom/constantes_${SLURM_JOB_ID:-local}.json}"
EXTRA="${EXTRA:-}"

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }
[ -d "${RELEASE_DIR}" ] || {
  echo "ERRO: release nao encontrado em ${RELEASE_DIR}."
  echo "      Rode antes: CENAS=0 BAIXAR=1 sbatch slurm/condgeom_stage_release_n3.slurm"
  exit 1; }
# O que a calibracao de fato exige. Testar so `-d` no diretorio passaria com um
# staging pela metade e a mensagem de erro daria conselho errado.
for _req in depth meta manifest.jsonl split.json; do
  [ -e "${RELEASE_DIR}/${_req}" ] || {
    echo "ERRO: staging INCOMPLETO em ${RELEASE_DIR}: falta '${_req}'."
    echo "      depth/ e o insumo geometrico; meta/ traz disparity_min/max, que e"
    echo "      a ESCALA METRICA — sem ela s, n e K~ nao significam nada;"
    echo "      split.json e a fronteira por cena."
    exit 1; }
done
mkdir -p "$(dirname "${OUT}")"

ARGS="--release ${RELEASE_DIR} --amostras ${AMOSTRAS} --out ${OUT}"
[ -n "${EXTRA}" ] && ARGS="${ARGS} ${EXTRA}"

echo "=============================================================="
echo "Job:      ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-local}) @ $(hostname)"
echo "Projeto:  ${PROJECT_ROOT}"
echo "Release:  ${RELEASE_DIR}"
echo "Args:     ${ARGS}"
echo "--- staging usado (semente e cenas) ---"
head -c 600 "${RELEASE_DIR}/staging_manifest.json" 2>/dev/null || echo "(sem staging_manifest.json)"
echo
echo "--- cota ---"
quota -s 2>/dev/null | tail -3 || true
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

# SEM GPU, DE PROPOSITO. Este job e numpy + PIL: le o PNG de disparidade,
# suaviza, deriva, e acumula percentis. Nao ha tensor em device nenhum.
#
# Pedir uma GPU aqui teria dois custos e zero beneficio: gastaria uma das 2
# vagas da QOS `onejob` — que os TREINOS de condicionamento precisam — e
# seguraria uma das 3 GPUs que o SLURM gerencia na n3 (das 8 do no, 5 rodam
# VLLM de terceiros fora dele).
export CUDA_VISIBLE_DEVICES=""

# Aspas DUPLAS aqui dentro: o APP_CMD inteiro esta entre aspas simples, e uma
# aspa simples neste ponto fecharia a string e faria o resto do script virar
# nome de arquivo (exit 126, "File name too long"). Ja aconteceu.
python3 -c "import numpy, PIL; print(\"[env] numpy\", numpy.__version__, \"| pillow\", PIL.__version__)"

# CONTRATO: se `geocond.jobs.calibrar` ainda nao existe, falhe AGORA — em 2
# segundos, com mensagem que diz o que falta — e nao depois de segurar uma das
# 2 vagas da QOS enquanto carrega dado.
# Heredoc com delimitador ENTRE ASPAS: o shell nao expande nada la dentro, entao
# o Python chega intacto. (Aspas duplas, nao simples: APP_CMD ja esta dentro de
# aspas simples e uma aspa simples aqui fecharia a string.)
python3 - <<"PYC"
import importlib.util, sys
if importlib.util.find_spec("geocond.jobs.calibrar") is None:
    sys.exit(
        "[erro] modulo geocond.jobs.calibrar ausente.\n"
        "       Este .slurm assume o contrato:\n"
        "         python3 -m geocond.jobs.calibrar --release <dir> --amostras N --out <json>\n"
        "       O pacote geocond/ esta sendo escrito em paralelo; ou ele cumpre a\n"
        "       assinatura, ou este arquivo tem que ser corrigido junto."
    )
print("[ok] geocond.jobs.calibrar importavel")
PYC

exec python3 -m geocond.jobs.calibrar '"${ARGS}"'
'

set +e
srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  --env TOKENIZERS_PARALLELISM=false \
  "${SIF}" bash -lc "${APP_CMD}"
RC=$?
set -e

echo "--- saida ---"
ls -l "${OUT}" 2>/dev/null || echo "(OUT nao foi escrito: ${OUT})"
echo "=== FIM (rc=${RC}) ==="
exit "${RC}"
