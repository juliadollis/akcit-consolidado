#!/bin/bash
#SBATCH --job-name=bokehnet-infer-test
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TESTE DE INFERÊNCIA DO BOKEHNET — roda scripts/infer_bokeh_test.py no cluster,
# seguindo EXATAMENTE o contrato oficial (Inference_bokehNet.py):
#   conditions = [Condition(aif,"bokeh"), Condition(defocus,"bokeh",[0,0],1.0,No_preprocess=True)]
#   prompt="an excellent photo with a large aperture", guidance_scale=1.0, main_adapter=None
# O mapa de defocus é RECOMPOSTO de depth+k+s1 com clip(k*|D-s1|/max_coc,0,1),
# a MESMA fórmula do dataloader de treino (defocus_source=recompute). NÃO usa a
# coluna `defocus_map` do df, que está normalizada por imagem (o k não age).
# Passe --defocus-source column só para ablação contra o comportamento legado.
# Salva uma tira "AIF | defocus | gerado | GT" por amostra e (opcional) sobe
# um df de exemplos pro HF.
#
# USO:
#   # a partir de um snapshot no HF (repo por-step do upload periódico):
#   HF_MODEL_REPO=juliadollis/genrefocus-bokehnet-synth-condlora-step5000 \
#   EXAMPLES_REPO=juliadollis/bokeh-exemplos \
#     sbatch scripts/infer_bokeh_test.slurm
#
#   # a partir do .safetensors final no HF:
#   HF_MODEL_REPO=juliadollis/genrefocus-bokehnet-real-4gpu HF_MODEL_FILE=bokeh.safetensors \
#     sbatch scripts/infer_bokeh_test.slurm
#
#   # a partir de um arquivo LOCAL (ex.: outputs/bokehnet_real_4gpu/bokeh/bokeh.safetensors):
#   WEIGHT_DIR=outputs/bokehnet_real_4gpu/bokeh WEIGHT_NAME=bokeh.safetensors \
#     sbatch scripts/infer_bokeh_test.slurm
#
# GPU: escolhe a REALMENTE mais livre dentro do container (SLURM deste cluster
# às vezes aloca GPU ocupada).
# =============================================================================

export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet_paper
# --- Segredos (NUNCA hardcode) ----------------------------------------------
# Ordem de precedencia: ja exportado no ambiente > $ENV_FILE > .env na raiz do
# projeto. Crie o .env a partir de .env.example (ele e git-ignored):
#   cp .env.example .env && ${EDITOR:-nano} .env
for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT:-}/.env" "${PROJECT_DIR:-}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env da raiz do projeto (cp .env.example .env)}"
export HF_TOKEN

export IMAGE_PATH=${IMAGE_PATH:-/raid/user_juliadollis/images/transformers-pytorch-gpu.sif}
export INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS:-0}

# --- de ONDE vem o LoRA (escolha UMA das duas formas) ---
export HF_MODEL_REPO=${HF_MODEL_REPO:-}          # ex.: juliadollis/genrefocus-bokehnet-real-4gpu
export HF_MODEL_FILE=${HF_MODEL_FILE:-bokeh.safetensors}
export WEIGHT_DIR=${WEIGHT_DIR:-}                # alternativa local (dir do .safetensors)
export WEIGHT_NAME=${WEIGHT_NAME:-bokeh.safetensors}

# --- dados de teste e saída ---
export SOURCE_DATASET=${SOURCE_DATASET:-AKCITPixel3/CMiQdveBBzNii}   # rota c (pares reais)
export SPLIT=${SPLIT:-train}
export N=${N:-8}
export STEPS=${STEPS:-28}
export LONG_SIDE=${LONG_SIDE:-512}
export MAIN_ADAPTER=${MAIN_ADAPTER:-none}        # 'none' = oficial/cond-only (NOSSO bokeh)
export OUT_DIR=${OUT_DIR:-outputs/bokeh_test}
export EXAMPLES_REPO=${EXAMPLES_REPO:-}                  # df HF p/ subir exemplos (opcional)

export HF_HOME=/workspace/hf-cache
# NAO exportar PYTHONUSERBASE: o trio ML limpo (transformers 5.0.0.dev0 /
# diffusers 0.37.1 / peft 0.18.1 + wandb, python-dotenv) vive no ~/.local, que o
# Python ja usa por default. Apontar o userbase pra outro dir tira o ~/.local do
# sys.path e o peft/diffusers "somem" (ModuleNotFoundError). Ver secao 4.3 do
# historico-ultimo.md — isso custou 4 jobs falhos no treino.

mkdir -p "${PROJECT_ROOT}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: sif não encontrado em ${IMAGE_PATH}"; exit 1; }
[[ -f "${PROJECT_ROOT}/third_party/Genfocus/Genfocus/pipeline/flux.py" ]] || { echo "ERRO: rode setup_genfocus.sh"; exit 1; }
if [[ -z "${HF_MODEL_REPO}" && -z "${WEIGHT_DIR}" ]]; then
    echo "ERRO: informe HF_MODEL_REPO=... OU WEIGHT_DIR=... (de onde carregar o LoRA do bokeh)."; exit 1
fi

echo "=========================================="
echo "Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host:     $(hostname)"
echo "LoRA:     repo='${HF_MODEL_REPO:-<local>}' file='${HF_MODEL_FILE}' | local_dir='${WEIGHT_DIR:-<none>}'"
echo "Dataset:  ${SOURCE_DATASET}:${SPLIT}  n=${N} steps=${STEPS} main_adapter=${MAIN_ADAPTER}"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t"," -k2 -nr | head -1 | cut -d"," -f1 | tr -d " ")
echo ">>> GPU escolhida (mais livre): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd /workspace/projects/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/projects/genrefocus_deblurnet_paper:/workspace/projects/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
    python3 -m pip install --user -r requirements.txt >/dev/null
    python3 -m pip install --user peft accelerate huggingface_hub >/dev/null
fi
python3 -c "from Genfocus.pipeline.flux import Condition, generate; print(\"[ok] Genfocus importável\")"

ARGS="--source-dataset ${SOURCE_DATASET} --split ${SPLIT} --n ${N} --steps ${STEPS} --long-side ${LONG_SIDE} --main-adapter ${MAIN_ADAPTER} --out-dir /workspace/projects/genrefocus_deblurnet_paper/${OUT_DIR}"
if [[ -n "${HF_MODEL_REPO}" ]]; then
    ARGS="${ARGS} --hf-model-repo ${HF_MODEL_REPO} --hf-model-file ${HF_MODEL_FILE}"
else
    ARGS="${ARGS} --weight-dir /workspace/projects/genrefocus_deblurnet_paper/${WEIGHT_DIR} --weight-name ${WEIGHT_NAME}"
fi
[[ -n "${EXAMPLES_REPO}" ]] && ARGS="${ARGS} --examples-hf-repo ${EXAMPLES_REPO}"

echo ">>> INFER BOKEH TEST: python3 scripts/infer_bokeh_test.py ${ARGS}"
PYTHONUNBUFFERED=1 python3 scripts/infer_bokeh_test.py ${ARGS}

echo ">>> PRONTO. Tiras salvas em ${OUT_DIR}:"
ls -la /workspace/projects/genrefocus_deblurnet_paper/${OUT_DIR} || true
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

# NV_FLAG: mecanismo de GPU do singularity. Default --nvccli. Se o no tiver GPU
# com defeito e o nvidia-container-cli falhar (nvml error), use NV_FLAG=--nv.
export NV_FLAG=${NV_FLAG:---nvccli}
singularity exec \
    ${NV_FLAG} \
    --bind /raid/user_juliadollis:/workspace \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS} \
    --env HF_MODEL_REPO=${HF_MODEL_REPO} \
    --env HF_MODEL_FILE=${HF_MODEL_FILE} \
    --env WEIGHT_DIR=${WEIGHT_DIR} \
    --env WEIGHT_NAME=${WEIGHT_NAME} \
    --env SOURCE_DATASET=${SOURCE_DATASET} \
    --env SPLIT=${SPLIT} \
    --env N=${N} \
    --env STEPS=${STEPS} \
    --env LONG_SIDE=${LONG_SIDE} \
    --env MAIN_ADAPTER=${MAIN_ADAPTER} \
    --env OUT_DIR=${OUT_DIR} \
    --env EXAMPLES_REPO=${EXAMPLES_REPO} \
    --env HF_TOKEN=${HF_TOKEN} \
    --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
