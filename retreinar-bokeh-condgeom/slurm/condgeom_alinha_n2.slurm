#!/bin/bash
#SBATCH --job-name=condgeom-alinha
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# CONDICAO A' — PERDA PONDERADA. FASE 1, h100n2, 1 GPU.
# =============================================================================
#
#   sbatch slurm/condgeom_alinha_n2.slurm                 # o braco A'
#   RUN_SMOKE=1 sbatch slurm/condgeom_alinha_n2.slurm     # 3 steps antes
#
# Copia fiel do `condgeom_fase1_n3.slurm` com os caminhos da n2. As guardas sao
# as mesmas e existem por precedente medido — nao sao cerimonia:
#   - cota: ja matou os dois bracos no step 1730 com checkpoint truncado
#   - GPU ocupada: o SLURM pode dar uma GPU que ja tem job de terceiro em cima
#   - release incompleto: um prefetch parcial passa num teste de `-d`
#
# POR QUE A n2 E NAO A n3
#   A n3 tem 3 GPUs e as duas vagas de QOS costumam estar tomadas. A n2 tem 8.
#   `/raid` e LOCAL DE CADA NO, entao FLUX, release e arvore foram copiados para
#   ca em 01/10 (destinos NOVOS, nada sobrescrito):
#     hf-cache-bokeh/              <- FLUX da n1, 54 GB
#     releases-bokeh/rota_a/       <- release da rota A da n1, 28 GB
#     releases-bokeh/_pareamento_n3/ <- split.json e manifest.jsonl da n3
#
# O PAREAMENTO, QUE E O PONTO DELICADO
#   O A rodou na n1 com o split 80/20 POR CENA; os bracos B0/B0linha replicaram
#   esse split num link farm na n3. O release traz o 95/5 oficial. Se este braco
#   usasse o 95/5 os conjuntos de treino diferem e a comparacao com A fica
#   confundida com composicao de dado. Por isso `RELEASE_DIR` aponta para o link
#   farm `rota_a_pareado`, cujo `split.json` e `manifest.jsonl` vieram da n3 com
#   sha256 conferido nos dois lados.
#   O manifesto e copia REAL e nao symlink de proposito: o guard do release pegou
#   que o manifesto do release tambem carrega `split` e divergia em 391 cenas.
# =============================================================================


set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_juliadollis}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/retreinar-bokeh-condgeom}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
# NAO e o caminho do prefetch. O release da rota A foi GERADO nesta maquina e
# esta em disco (57 GB, 69.700 amostras). `dados/rota_a_pareado` e um LINK FARM
# sobre ele: os pixels ficam onde estao (zero disco novo, a cota esta em 520G
# de 500G) e o unico arquivo real e o `split.json` — o da h100n1, 80/20, para
# que os conjuntos de treino dos dois lados sejam os MESMOS. Com o 95/5 do
# release a comparacao ficaria confundida com composicao de dado.
RELEASE_DIR="${RELEASE_DIR:-${BIND_ROOT}/releases-bokeh/rota_a_pareado}"
# BRACO=B0 (canais) ou B0linha (controle de ruido). Ver os cabecalhos
# dos YAMLs: o par B0/B0linha e o que sustenta qualquer afirmacao.
BRACO="${BRACO:-Alinha}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_bokeh_condgeom_${BRACO}.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/train_bokeh_condgeom_smoke.yaml}"
RUN_SMOKE="${RUN_SMOKE:-0}"
BATCH_EFETIVO_ALVO="${BATCH_EFETIVO_ALVO:-32}"

# 1 GPU com accum 32 = 32 micro-batches por step, o MESMO que a n1 faz com
# 4 GPUs e accum 8. O batch efetivo 32 do §4.1 fica preservado e os
# checkpoints seguem comparaveis; muda so o relogio de parede.
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

# O `batch_size` vem do YAML e ENTRA na conta. Antes a formula era
# `32/NUM_GPUS`, que so acerta com batch_size=1 — numa H100 de 80 GB alguem
# poderia subir para 2 e o efetivo viraria 64 sem nada avisar.
BATCH_SIZE=$(grep -E "^ +batch_size:" "${PROJECT_ROOT}/${TRAIN_CONFIG}" \
             | head -1 | tr -dc "0-9")
BATCH_SIZE="${BATCH_SIZE:-1}"
DIVISOR=$(( BATCH_SIZE * NUM_GPUS ))
ACCUM=$(( BATCH_EFETIVO_ALVO / DIVISOR ))
if [ $(( ACCUM * DIVISOR )) -ne "${BATCH_EFETIVO_ALVO}" ]; then
  echo "ERRO: batch_size=${BATCH_SIZE} x ${NUM_GPUS} GPUs nao divide ${BATCH_EFETIVO_ALVO}."
  echo "      O §4.1 pede batch efetivo 32; mudar isso muda o treino."
  exit 1
fi
EFETIVO=$(( BATCH_SIZE * ACCUM * NUM_GPUS ))

cd "${PROJECT_ROOT}"
mkdir -p logs outputs

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. cp .env.example .env && edite}"
: "${WANDB_API_KEY:?ERRO: WANDB_API_KEY nao definido}"
export HF_TOKEN WANDB_API_KEY HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${BIND_ROOT}/hf-cache-bokeh}"
export TOKENIZERS_PARALLELISM=false
export MAIN_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# ── GUARDA DE COTA ──────────────────────────────────────────────────────────
# Precedente caro: a cota estourou com o grace expirado e os DOIS bracos morreram
# no step 1730/1610, com o checkpoint saindo truncado. Antes disso, o registro do
# projeto ja tinha "disco cheio matou 3 treinos, ~18h de 3 GPUs".
# Cada checkpoint e 2,78 GB; com keep_last_n=1 mais best.pt sao ~6 GB por braco.
# Exijo 40 GB livres na cota para comecar — folga para os dois bracos e para o
# arquivo temporario que o torch.save cria antes de renomear.
COTA_MIN_GB="${COTA_MIN_GB:-40}"
_q=$(quota -s 2>/dev/null | tail -1)
_uso=$(echo "$_q" | awk '{print $2}' | tr -d 'G*')
_lim=$(echo "$_q" | awk '{print $3}' | tr -d 'G*')
if [ -n "${_uso}" ] && [ -n "${_lim}" ]; then
  _livre=$(( _lim - _uso ))
  echo "[cota] uso=${_uso}G limite=${_lim}G livre=${_livre}G (minimo exigido: ${COTA_MIN_GB}G)"
  if [ "${_livre}" -lt "${COTA_MIN_GB}" ]; then
    echo "[erro] so ${_livre}G livres na cota. ABORTANDO ANTES de comecar, em vez"
    echo "       de morrer no meio com checkpoint truncado. Libere espaco."
    exit 1
  fi
else
  echo "[cota] AVISO: nao consegui ler a cota; seguindo sem a guarda."
fi

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado: ${SIF}"; exit 1; }
[ -f "${PROJECT_ROOT}/${TRAIN_CONFIG}" ] || { echo "ERRO: config: ${TRAIN_CONFIG}"; exit 1; }
[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ] || {
  echo "ERRO: third_party/Genfocus ausente. Confira o rsync."; exit 1; }
# B2 — um prefetch PARCIAL cria o diretorio e passaria num teste de `-d`, e a
# mensagem de erro daria conselho errado. Testamos o que o treino de fato exige.
[ -d "${RELEASE_DIR}" ] || {
  echo "ERRO: release nao encontrado em ${RELEASE_DIR}."
  echo "      Rode antes: monte o link farm rota_a_pareado (ver cabecalho)"; exit 1; }
for _req in manifest.jsonl split.json meta generated; do
  [ -e "${RELEASE_DIR}/${_req}" ] || {
    echo "ERRO: release INCOMPLETO em ${RELEASE_DIR}: falta '${_req}'."
    echo "      Medido em 2026-09-17: o repo tinha so 'depth/'. Sem 'meta/' nao ha"
    echo "      k_value/focus_disparity (o mapa de defocus nao se monta) e sem"
    echo "      'generated/' nao ha ALVO. Espere o upload terminar."; exit 1; }
done

echo "=============================================================="
echo "Job:        ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-local}) @ $(hostname)"
echo "Projeto:    ${PROJECT_ROOT}"
echo "Config:     ${TRAIN_CONFIG}"
echo "Release:    ${RELEASE_DIR}"
echo "GPUs:       ${NUM_GPUS}  batch_size=${BATCH_SIZE}  accum=${ACCUM}"
echo "Batch EFETIVO: ${EFETIVO}   (§4.1 pede ${BATCH_EFETIVO_ALVO})"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
  | grep -E "^[0-9]+," || true

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
# PYTHONPATH so com CODIGO DO PROJETO, montado AQUI DENTRO. Passar site-packages
# por --env PYTHONPATH tira o ~/.local do sys.path e some com diffusers/peft.
# `.pydeps` PREFIXA, e isso e obrigatorio por causa do cv2: o ~/.local (que e
# NFS e compartilhado entre nos) tem o opencv-python completo, que puxa libGL
# exigindo GLIBC 2.38 enquanto o container tem 2.35 — ImportError na hora de
# importar cv2. A correcao prescrita e opencv-python-headless num .pydeps do
# PROJETO, prefixado (nunca instalado no ~/.local, nunca via --env do
# singularity, que tiraria o ~/.local do sys.path e sumiria com diffusers/peft).
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps:'"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "--- versoes ---"
python3 - <<PYV
import torch, transformers, diffusers, peft, accelerate, datasets
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
for m in (transformers, diffusers, peft, accelerate, datasets):
    print(f"{m.__name__:13s}", m.__version__)
PYV
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print(\"[ok] Genfocus importavel\")"

# A GPU que o SLURM deu pode ja estar ocupada. Abortamos em vez de tomar a de
# outro job — com 5 GPUs fora do SLURM nesta maquina, escolher a mao e arriscado.
python3 - <<PYG
import torch
for i in range(torch.cuda.device_count()):
    livre, total = torch.cuda.mem_get_info(i)
    usado = (total - livre) / 1e9
    print(f"[gpu] cuda:{i} usado={usado:.1f} GB de {total/1e9:.1f} GB")
    if usado > 5.0:
        raise SystemExit(
            f"[erro] cuda:{i} ja tem {usado:.1f} GB em uso. Abortando em vez de "
            "disputar memoria com job de terceiro."
        )
PYG

echo ">>> CHECK (sem GPU, sem rede) <<<"
python3 -m genfocus_train.train check --config '"${TRAIN_CONFIG}"' --stage bokeh

if [ "'"${RUN_SMOKE}"'" = "1" ]; then
  echo ">>> SMOKE — 3 steps no caminho real <<<"
  python3 -m genfocus_train.train smoke --stage bokeh --config '"${SMOKE_CONFIG}"'
fi

echo ">>> TREINO FASE 1 — '"${NUM_GPUS}"' GPUs, accum '"${ACCUM}"' <<<"
# `--multi_gpu` so com 2+ processos: o accelerate recusa a flag com 1
# ("You need to use at least 2 processes to use `--multi_gpu`"). Estes bracos
# rodam em 1 GPU com accum 32 de proposito — mesmo batch efetivo 32 da n1, que
# usa 4 GPUs com accum 8 — entao a flag tem de sair.
# `if` e nao `a && b`: sob `set -e` um `[ ... ] && VAR=x` com teste FALSO
# devolve 1 e aborta o script inteiro. Com 1 GPU o teste e sempre falso, ou
# seja o atalho derrubaria exatamente o caso que este bloco existe para tratar.
MULTI=""
if [ '"${NUM_GPUS}"' -gt 1 ]; then MULTI="--multi_gpu"; fi

accelerate launch \
  ${MULTI} \
  --num_machines 1 \
  --num_processes '"${NUM_GPUS}"' \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port '"${MAIN_PORT}"' \
  -m genfocus_train.train bokeh --config '"${TRAIN_CONFIG}"' \
  --grad-accum '"${ACCUM}"'
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env TOKENIZERS_PARALLELISM=false \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=== FIM (rc=$?) ==="
