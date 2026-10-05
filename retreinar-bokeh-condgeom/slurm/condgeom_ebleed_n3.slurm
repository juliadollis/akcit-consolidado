#!/bin/bash
#SBATCH --job-name=condgeom-ebleed
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# E_bleed — O PORTAO DA CAMPANHA. 1 GPU.
# =============================================================================
#
# O QUE FAZ
#   Mede o erro CONCENTRADO na regiao de borda de profundidade, que e onde vive
#   o vazamento de cor (o "bleed"). As metricas globais diluem o artefato: a
#   borda e ~1,2% dos pixels, entao um LPIPS de quadro inteiro mal se mexe
#   enquanto o defeito que a campanha existe para consertar continua la.
#
#   O mapa de oclusao e o `tau` usados aqui TEM que ser os mesmos do
#   condicionamento — senao a metrica mede uma regiao diferente da que a perda
#   supervisiona. Por isso o JSON de constantes e entrada OBRIGATORIA, e vem do
#   job de calibracao.
#
# SOBRE O LADO DO SPLIT
#   Mede no lado **val** do `split.json`. E o lado que o staging trouxe com
#   `generated/` (o bokeh ALVO) justamente para isto.
#
#   ATENCAO — CONTRATO A CUMPRIR, NAO CODIGO EXISTENTE:
#   `geocond.jobs.ebleed_bench` AINDA NAO EXISTE. A assinatura assumida e:
#
#       python3 -m geocond.jobs.ebleed_bench \
#           --release <dir> --constantes <json> --split val \
#           --amostras N --lora <path.safetensors|none> \
#           --rotulo <nome> --saida <json>
#
#   Modelada na versao antiga (`referencia_antiga/geo_cond_v1/jobs/
#   f1_ebleed_bench.py`, que recebia --bench/--constantes/--n/--saida/--rotulo),
#   trocando o `--bench` pelo `--release`: a campanha nova mede sobre o release
#   da rota A, que e o dado com a escala metrica certa. Se o modulo expuser
#   outra coisa, este arquivo tem que ser corrigido junto.
#
# NADA DA CAMPANHA ANTIGA VOLTA COMO NUMERO. `referencia_antiga/LEIA.md`: as 8
# constantes e a tabela de E_bleed dos 7 modelos foram medidas contra modelos
# treinados com o mapa de defocus errado. Sao referencia de METODO, nao linha de
# base.
#
# NAO APAGA NADA. Sem `rm`, sem `--delete`, sem `scancel`.
#
# QOS `onejob`: 2 jobs simultaneos.
#
# COMO RODAR
#   ssh dgx-H100-03
#   cd /raid/user_danielpedrozo/projects/julia/retreinar-bokeh-condgeom
#   mkdir -p logs outputs
#   CONSTANTES=outputs/condgeom/constantes_<JOBID>.json \
#     LORA=outputs/bokeh_fase1/bokeh/checkpoints/step_XXXX.safetensors \
#     ROTULO=fase1-geo sbatch slurm/condgeom_ebleed_n3.slurm
#
#   Para a LINHA DE BASE (sem condicionamento geometrico): LORA=none ROTULO=base.
#   Rode a base ANTES: E_bleed so significa alguma coisa em comparacao pareada.
#
# VARIAVEIS
#   RELEASE_DIR  staging do release (com generated/ das cenas de val)
#   CONSTANTES   JSON do job de calibracao (OBRIGATORIO)
#   LORA         peso a medir, ou `none` para a linha de base
#   AMOSTRAS     quantas amostras (default 200)
#   ROTULO       nome da condicao no JSON de saida (default: derivado do LORA)
#   SAIDA        JSON de saida (default outputs/condgeom/ebleed_<rotulo>_<job>.json)
#   EXTRA        flags cruas
#
# CRITERIO DE SUCESSO
#   rc=0, o JSON em SAIDA, e o `n` da amostra IMPRESSO — amostragem cujo `n` nao
#   aparece nao e verificavel.
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_danielpedrozo}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/julia/retreinar-bokeh-condgeom}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"

for _v in RELEASE_DIR CONSTANTES LORA AMOSTRAS ROTULO SAIDA EXTRA; do
  eval "_cli_${_v}=\${${_v}:-}"
done

cd "${PROJECT_ROOT}"
mkdir -p logs outputs/condgeom

for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done

for _v in RELEASE_DIR CONSTANTES LORA AMOSTRAS ROTULO SAIDA EXTRA; do
  eval "_val=\${_cli_${_v}}"
  if [ -n "${_val}" ]; then eval "${_v}=\"\${_val}\""; fi
done

export HF_HOME="${HF_HOME:-${BIND_ROOT}/projects/julia/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # le do disco; download e opt-in
export TOKENIZERS_PARALLELISM=false

RELEASE_DIR="${RELEASE_DIR:-${BIND_ROOT}/projects/julia/releases/rota_a_bancada}"
LORA="${LORA:-none}"
AMOSTRAS="${AMOSTRAS:-200}"
ROTULO="${ROTULO:-$( [ "${LORA}" = "none" ] && echo base || basename "${LORA}" .safetensors )}"
SAIDA="${SAIDA:-${PROJECT_ROOT}/outputs/condgeom/ebleed_${ROTULO}_${SLURM_JOB_ID:-local}.json}"
EXTRA="${EXTRA:-}"

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }
: "${CONSTANTES:?ERRO: CONSTANTES nao definido. E o JSON do job de calibracao; sem ele o tau da metrica difere do tau do condicionamento e a metrica mede outra regiao.}"
[ -f "${CONSTANTES}" ] || { echo "ERRO: constantes nao encontradas: ${CONSTANTES}"; exit 1; }
[ -d "${RELEASE_DIR}" ] || {
  echo "ERRO: release nao encontrado em ${RELEASE_DIR}."
  echo "      Rode antes: CENAS=24 BAIXAR=1 sbatch slurm/condgeom_stage_release_n3.slurm"
  exit 1; }
for _req in depth meta generated manifest.jsonl split.json; do
  [ -e "${RELEASE_DIR}/${_req}" ] || {
    echo "ERRO: staging INCOMPLETO em ${RELEASE_DIR}: falta '${_req}'."
    echo "      E_bleed compara com o bokeh ALVO, entao 'generated/' das cenas de"
    echo "      val e obrigatorio aqui (ao contrario da calibracao, que roda com"
    echo "      CENAS=0)."
    exit 1; }
done
if [ "${LORA}" != "none" ] && [ ! -f "${LORA}" ]; then
  echo "ERRO: peso nao encontrado: ${LORA} (use LORA=none para a linha de base)"
  exit 1
fi
mkdir -p "$(dirname "${SAIDA}")"

ARGS="--release ${RELEASE_DIR} --constantes ${CONSTANTES} --split val"
ARGS="${ARGS} --amostras ${AMOSTRAS} --lora ${LORA} --rotulo ${ROTULO} --saida ${SAIDA}"
[ -n "${EXTRA}" ] && ARGS="${ARGS} ${EXTRA}"

echo "=============================================================="
echo "Job:        ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-local}) @ $(hostname)"
echo "Release:    ${RELEASE_DIR}"
echo "Constantes: ${CONSTANTES}"
echo "LoRA:       ${LORA}"
echo "Args:       ${ARGS}"
echo "--- cota ---"
quota -s 2>/dev/null | tail -3 || true
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

# GPU por UUID (a GPU4 da n3 tem defeito e contamina os indices); grep filtra o
# "Unable to determine the device handle for GPU4" que sai no meio do stdout.
free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -z "$free_gpu" ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi
gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
gpu_mem=$(echo  "$free_gpu" | cut -d, -f3 | tr -d " ")
echo "[gpu] escolhida ${gpu_uuid} | memory.used=${gpu_mem} MiB"
if [ "$gpu_mem" -gt 500 ]; then echo "[erro] GPU ocupada (${gpu_mem} MiB). Abortando."; exit 1; fi
export CUDA_VISIBLE_DEVICES=$gpu_uuid

python3 -c "import torch;print(\"[env] torch\",torch.__version__,\"|\",torch.cuda.get_device_name(0))"

# O staging trouxe generated/ das cenas de VAL? Medir E_bleed em cena de treino
# nao e E_bleed, e memorizacao. Isto custa 2 segundos e evita descobrir no fim.
python3 - <<PYV
import json, pathlib, sys
raiz = pathlib.Path("'"${RELEASE_DIR}"'")
man = raiz / "staging_manifest.json"
if not man.is_file():
    print("[aviso] sem staging_manifest.json: nao da para conferir quais cenas de "
          "val vieram. Seguindo, mas o resultado fica sem procedencia.")
    sys.exit(0)
atual = json.loads(man.read_text(encoding="utf-8")).get("atual", {})
cenas_val = (atual.get("cenas") or {}).get("val", [])
com_pixel = [c for c in cenas_val if (raiz / "generated" / c).is_dir()]
print(f"[bancada] semente={atual.get(\"semente\")} cenas val={len(cenas_val)} "
      f"com generated/={len(com_pixel)}")
if not com_pixel:
    sys.exit("[erro] nenhuma cena de VAL tem generated/ em disco. Refaca o staging "
             "com CENAS>=2 (o script garante ao menos 1 cena de val).")
PYV

python3 - <<PYC
import importlib.util, sys
if importlib.util.find_spec("geocond.jobs.ebleed_bench") is None:
    sys.exit(
        "[erro] modulo geocond.jobs.ebleed_bench ausente.\n"
        "       Contrato assumido por este .slurm:\n"
        "         python3 -m geocond.jobs.ebleed_bench --release <dir> --constantes <json>\n"
        "           --split val --amostras N --lora <path|none> --rotulo <nome> --saida <json>"
    )
print("[ok] geocond.jobs.ebleed_bench importavel")
PYC

exec python3 -m geocond.jobs.ebleed_bench '"${ARGS}"'
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
ls -l "${SAIDA}" 2>/dev/null || echo "(SAIDA nao foi escrita: ${SAIDA})"
echo "=== FIM (rc=${RC}) ==="
exit "${RC}"
