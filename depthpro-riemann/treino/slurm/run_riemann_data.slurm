#!/bin/bash
#SBATCH --job-name=riemann-data
#SBATCH --partition=h100n2
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --output=/raid/user_juliadollis/projects/depth-riemannian/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/depth-riemannian/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# DOWNLOAD EM STREAMING do Hypersim (download_hypersim_stream.py do Francisco).
# Baixa cena a cena, converte para o layout rgb/ + depth/, e apaga o zip antes
# da proxima cena. NAO modifica o codigo dele.
#
# ATENCAO (rede): o downloader acessa a CDN da Apple
# (docs-assets.developer.apple.com). O Francisco marcou que NAO testou contra a
# Apple. PRIMEIRO rode o modo de teste (1 cena) e confira os pares antes de
# baixar em massa. Se a Apple estiver bloqueada na h100n2, rode num host com
# internet aberta ou peca liberacao.
#
# O job pede 1 GPU so por conformidade da particao; o download e CPU + rede
# (a GPU fica ociosa). TMPDIR aponta para o raid (zips de ~1-2 GB por cena nao
# cabem no /tmp do container).
#
# Uso (exemplos):
#   # 0) TESTE (1 cena) - SEMPRE rodar isto primeiro:
#   MODE=test sbatch slurm/run_riemann_data.slurm
#
#   # 1) FASE INTEIRA num job so (treino + validacao em sequencia) - RECOMENDADO:
#   PHASE=smoke    sbatch slurm/run_riemann_data.slurm
#   PHASE=ablation sbatch slurm/run_riemann_data.slurm
#   PHASE=full     sbatch slurm/run_riemann_data.slurm
#
#   # 1b) ou manualmente, um subset por job:
#   # subset SMOKE (treino ~60, val ~20):
#   DATA_OUT=hypersim/smoke/train N_IMAGES=60  LIMIT_SCENES=8            sbatch slurm/run_riemann_data.slurm
#   DATA_OUT=hypersim/smoke/val   N_IMAGES=20  LIMIT_SCENES=6 START_INDEX=200 sbatch slurm/run_riemann_data.slurm
#
#   # 2) subset ABLATION (treino ~3000, val ~500) - cenas DISTINTAS entre train/val:
#   DATA_OUT=hypersim/ablation/train N_IMAGES=3000 LIMIT_SCENES=80             sbatch slurm/run_riemann_data.slurm
#   DATA_OUT=hypersim/ablation/val   N_IMAGES=500  LIMIT_SCENES=25 START_INDEX=200 sbatch slurm/run_riemann_data.slurm
#
#   # 3) subset FULL (treino ~20000, val ~2000) - so quando for treinar a campea:
#   DATA_OUT=hypersim/full/train N_IMAGES=20000 LIMIT_SCENES=600             sbatch slurm/run_riemann_data.slurm
#   DATA_OUT=hypersim/full/val   N_IMAGES=2000  LIMIT_SCENES=120 START_INDEX=700 sbatch slurm/run_riemann_data.slurm
# =============================================================================

export RAID=/raid/user_juliadollis
export IMAGE_PATH=${IMAGE_PATH:-${RAID}/images/transformers-pytorch-gpu.sif}
export PROJECT_HOST=${PROJECT_HOST:-${RAID}/projects/depth-riemannian}
export RIEMANN_SUBDIR=${RIEMANN_SUBDIR:-riemann_depthpro_h100_container}
export RIE_C=/workspace/projects/depth-riemannian/${RIEMANN_SUBDIR}

# --- Parametros de download (env; ver exemplos acima) ---
export MODE=${MODE:-batch}
# PHASE=smoke|ablation|full  -> baixa TREINO e VALIDACAO no MESMO job, em
# sequencia. Faixas de cena disjuntas entre treino e validacao (START_INDEX),
# para a metrica de validacao nao ser contaminada por cenas vistas no treino.
export PHASE=${PHASE:-}                 # "test" = 1 cena de sanidade | "batch" = usa os params abaixo
export DATA_OUT=${DATA_OUT:-hypersim/ablation/train}   # subdir sob /workspace/data
export N_IMAGES=${N_IMAGES:-3000}
export LIMIT_SCENES=${LIMIT_SCENES:-80}
export START_INDEX=${START_INDEX:-0}
export SCENES=${SCENES:-}                  # lista explicita (sobrepoe N_IMAGES): "ai_001_001 ai_001_002"
export FOCAL_PX=${FOCAL_PX:-886.81}

export PLAN=""
if [[ -n "${PHASE}" ]]; then
    case "${PHASE}" in
        # formato: <subdir>:<n_images>:<limit_scenes>:<start_index>
        smoke)    PLAN="hypersim/smoke/train:60:8:0;hypersim/smoke/val:20:6:200" ;;
        ablation) PLAN="hypersim/ablation/train:3000:80:0;hypersim/ablation/val:500:25:200" ;;
        full)     PLAN="hypersim/full/train:20000:600:0;hypersim/full/val:2000:120:700" ;;
        *) echo "ERRO: PHASE invalida: ${PHASE} (use smoke|ablation|full)"; exit 1 ;;
    esac
    MODE=phase
    echo ">>> PHASE=${PHASE} -> plano: ${PLAN}"
fi

mkdir -p "${PROJECT_HOST}/logs" "${RAID}/data" "${RAID}/tmp"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: SIF nao encontrado em ${IMAGE_PATH}"; exit 1; }
[[ -d "${PROJECT_HOST}/${RIEMANN_SUBDIR}" ]] || { echo "ERRO: repo nao esta em ${PROJECT_HOST}/${RIEMANN_SUBDIR}. Rode git clone (ver run_riemann_setup.slurm)."; exit 1; }

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) | $(hostname) | MODE=${MODE} DATA_OUT=${DATA_OUT}"

APP_CMD='
set -euo pipefail

# Shims de curl/wget: o .sif nao tem esses binarios, e o codigo do Francisco
# chama os dois via os.system/subprocess. Ver slurm/bin/curl e slurm/bin/wget.
export PATH="/workspace/projects/depth-riemannian/slurm/bin:${PATH}"

# Pre-check de rede. SEM isto, uma falha de DNS faz o download_hypersim_stream.py
# classificar TODAS as cenas como "(inexistente, pulando)" e terminar com
# "sucesso" e ZERO imagens (foi exatamente o que aconteceu nos jobs 27460-27471).
if ! getent hosts docs-assets.developer.apple.com > /dev/null 2>&1; then
    echo "ERRO: DNS nao resolve docs-assets.developer.apple.com de dentro do container." >&2
    echo "      O download abortou AQUI de proposito: sem isso o job terminaria" >&2
    echo "      com 0 imagens e sem erro. Cheque o bind de /etc/resolv.conf e se" >&2
    echo "      a particao tem saida para a internet." >&2
    cat /etc/resolv.conf >&2 || true
    exit 1
fi
echo ">>> DNS ok para docs-assets.developer.apple.com"
cd "${RIE_C}"

if [[ "${MODE}" == "phase" ]]; then
    echo ">>> MODO FASE: treino e validacao no MESMO job, em sequencia."
    IFS=";"
    for spec in ${PLAN}; do
        unset IFS
        SUB=$(echo ${spec} | cut -d: -f1)
        NI=$(echo ${spec}  | cut -d: -f2)
        LS=$(echo ${spec}  | cut -d: -f3)
        SI=$(echo ${spec}  | cut -d: -f4)
        OUT=/workspace/data/${SUB}
        echo ""
        echo "============================================================"
        echo ">>> ${SUB}: alvo=${NI} imgs | limit-scenes=${LS} | start-index=${SI}"
        echo "============================================================"
        python3 scripts/download_hypersim_stream.py \
            --out-root "${OUT}" --n-images "${NI}" \
            --limit-scenes "${LS}" --start-index "${SI}" --focal-px "${FOCAL_PX}"
        echo ">>> ${SUB}: rgb=$(ls ${OUT}/rgb 2>/dev/null | wc -l) depth=$(ls ${OUT}/depth 2>/dev/null | wc -l)"
        IFS=";"
    done
    unset IFS
    echo ""
    echo ">>> FASE CONCLUIDA. Resumo:"
    for spec in $(echo ${PLAN} | tr ";" " "); do
        SUB=$(echo ${spec} | cut -d: -f1)
        OUT=/workspace/data/${SUB}
        echo "  ${SUB}: rgb=$(ls ${OUT}/rgb 2>/dev/null | wc -l) depth=$(ls ${OUT}/depth 2>/dev/null | wc -l)"
    done
    exit 0
fi

if [[ "${MODE}" == "test" ]]; then
    echo ">>> TESTE: baixando 1 cena (ai_001_001) para /workspace/data/hypersim/_test1 ..."
    python3 scripts/download_hypersim_stream.py \
        --out-root /workspace/data/hypersim/_test1 --scenes ai_001_001 --focal-px "${FOCAL_PX}"
    echo ">>> pares gerados:"
    echo "  rgb:   $(ls /workspace/data/hypersim/_test1/rgb   2>/dev/null | wc -l)"
    echo "  depth: $(ls /workspace/data/hypersim/_test1/depth 2>/dev/null | wc -l)"
    echo ">>> se os dois contadores baterem e forem > 0, o streaming da Apple funciona. Siga para os subsets."
    exit 0
fi

OUT=/workspace/data/${DATA_OUT}
echo ">>> baixando para ${OUT}"
if [[ -n "${SCENES}" ]]; then
    python3 scripts/download_hypersim_stream.py \
        --out-root "${OUT}" --scenes ${SCENES} --focal-px "${FOCAL_PX}"
else
    python3 scripts/download_hypersim_stream.py \
        --out-root "${OUT}" --n-images "${N_IMAGES}" \
        --limit-scenes "${LIMIT_SCENES}" --start-index "${START_INDEX}" --focal-px "${FOCAL_PX}"
fi

echo ">>> conferencia final:"
echo "  rgb:   $(ls ${OUT}/rgb   2>/dev/null | wc -l)"
echo "  depth: $(ls ${OUT}/depth 2>/dev/null | wc -l)"
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --bind ${RAID}:/workspace \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env TMPDIR=/workspace/tmp \
    --env PYTHONUSERBASE=/workspace/python-packages-riemann \
    --env PYTHONPATH=/workspace/python-packages-riemann/lib/python3.10/site-packages:/workspace/python-packages-riemann/lib/python3.11/site-packages \
    --env RIE_C=${RIE_C} \
    --env MODE=${MODE} \
    --env PLAN="${PLAN}" \
    --env DATA_OUT=${DATA_OUT} \
    --env N_IMAGES=${N_IMAGES} \
    --env LIMIT_SCENES=${LIMIT_SCENES} \
    --env START_INDEX=${START_INDEX} \
    --env SCENES="${SCENES}" \
    --env FOCAL_PX=${FOCAL_PX} \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
