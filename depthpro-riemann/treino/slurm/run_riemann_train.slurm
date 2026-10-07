#!/bin/bash
#SBATCH --job-name=riemann-train
#SBATCH --partition=h100n2
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=UNLIMITED
#SBATCH --output=/raid/user_juliadollis/projects/depth-riemannian/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/depth-riemannian/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TREINO/EXPERIMENTOS do fine-tuning Riemanniano do DepthPro (codigo do Francisco,
# intacto), via Singularity na h100n2. Job de 1 GPU: usa o seletor de GPU VAZIA
# (aborta se a mais livre tiver >500 MiB em uso), igual ao eval do bokeh.
#
# STEP escolhe o experimento:
#   smoke     -> run_ablation.py com 2 configs (B0_berhu + B5_full), 40 imagens,
#                2 epocas. So valida que dados+modelo+GPU conversam.
#   ablation  -> run_ablation.py com as 39 configs (VARIANT=heads ou heads_lora).
#   optuna    -> run_optuna.py (busca bayesiana multi-objetivo).
#   champion  -> train_single.py (config vencedora, multi-seed, +curvatura).
#
# Uso (rode o smoke ANTES de tudo e CONFIRA a linha "params treinaveis"):
#   STEP=smoke sbatch slurm/run_riemann_train.slurm
#
#   STEP=ablation VARIANT=heads      sbatch slurm/run_riemann_train.slurm
#   STEP=ablation VARIANT=heads_lora sbatch slurm/run_riemann_train.slurm
#
#   STEP=optuna VARIANT=heads TRIALS=200 EPOCHS_PER_TRIAL=20 sbatch slurm/run_riemann_train.slurm
#
#   # campea: passe os pesos vencedores da ablacao/optuna
#   STEP=champion VARIANT=heads BERHU=0.7 NORMAL=0.9 GAUSS=0.45 SEEDS=3 EPOCHS=100 \
#     sbatch slurm/run_riemann_train.slurm
#
# Datasets: por padrao usa o subset da fase (smoke/ablation/full). Sobreponha com
# TRAIN_ROOT=... VAL_ROOT=... se quiser outro caminho.
# =============================================================================

export RAID=/raid/user_juliadollis
export IMAGE_PATH=${IMAGE_PATH:-${RAID}/images/transformers-pytorch-gpu.sif}
export PROJECT_HOST=${PROJECT_HOST:-${RAID}/projects/depth-riemannian}
export RIEMANN_SUBDIR=${RIEMANN_SUBDIR:-riemann_depthpro_h100_container}
export RIE_C=/workspace/projects/depth-riemannian/${RIEMANN_SUBDIR}
export RUNS_C=/workspace/projects/depth-riemannian/runs

# --- knobs gerais ---
export STEP=${STEP:-smoke}
export VARIANT=${VARIANT:-heads}          # heads | heads_lora
export BATCH_SIZE=${BATCH_SIZE:-8}
export SIZE=${SIZE:-512}
export NUM_WORKERS=${NUM_WORKERS:-16}
export CKPT=${CKPT:-/workspace/models/checkpoints/depth_pro.pt}
export TRAIN_ROOT=${TRAIN_ROOT:-}
export VAL_ROOT=${VAL_ROOT:-}
export OUT_DIR=${OUT_DIR:-}
export EPOCHS=${EPOCHS:-}
export MAX_TRAIN=${MAX_TRAIN:-}
export FILTER=${FILTER:-}
# optuna
export TRIALS=${TRIALS:-200}
export EPOCHS_PER_TRIAL=${EPOCHS_PER_TRIAL:-20}
# champion (defaults = campea sugerida no roteiro; sobreponha com os vencedores)
export SEEDS=${SEEDS:-3}
export BERHU=${BERHU:-0.7}
export GRAD=${GRAD:-0.0}
export NORMAL=${NORMAL:-0.9}
export GAUSS=${GAUSS:-0.45}
export GEOD=${GEOD:-0.0}
export METRIC=${METRIC:-0.0}
export GAUSS_SMOOTH_SIGMA=${GAUSS_SMOOTH_SIGMA:-0.5}
export EXPORT_CURVATURE=${EXPORT_CURVATURE:-1}

# --- defaults por STEP (env explicito sempre vence) ---
case "${STEP}" in
    smoke)
        : "${TRAIN_ROOT:=/workspace/data/hypersim/smoke/train}"
        : "${VAL_ROOT:=/workspace/data/hypersim/smoke/val}"
        # B0_berhu (baseline sem geometria) + B5_full (TODOS os termos geometricos).
        # Sao 2 configs e juntas exercitam todo o caminho da loss. Cuidado: o
        # --filter casa por PREFIXO (run_ablation.py:80), entao "B1" sozinho
        # pegaria as 5 configs B1_*, nao uma.
        : "${EPOCHS:=2}"; : "${MAX_TRAIN:=40}"; : "${FILTER:=B0_berhu B5_full}"
        : "${OUT_DIR:=${RUNS_C}/smoke}"
        ;;
    ablation)
        : "${TRAIN_ROOT:=/workspace/data/hypersim/ablation/train}"
        : "${VAL_ROOT:=/workspace/data/hypersim/ablation/val}"
        : "${EPOCHS:=40}"
        : "${OUT_DIR:=${RUNS_C}/ablation_${VARIANT}}"
        ;;
    optuna)
        : "${TRAIN_ROOT:=/workspace/data/hypersim/ablation/train}"
        : "${VAL_ROOT:=/workspace/data/hypersim/ablation/val}"
        : "${OUT_DIR:=${RUNS_C}/optuna_${VARIANT}}"
        ;;
    champion)
        : "${TRAIN_ROOT:=/workspace/data/hypersim/full/train}"
        : "${VAL_ROOT:=/workspace/data/hypersim/full/val}"
        : "${EPOCHS:=100}"
        : "${OUT_DIR:=${RUNS_C}/champion}"
        ;;
    *)
        echo "ERRO: STEP invalido: ${STEP} (use smoke|ablation|optuna|champion)"; exit 1
        ;;
esac

# argumentos opcionais montados no host (expansao sem aspas dentro do container)
FILTER_ARG=""; [[ -n "${FILTER}" ]] && FILTER_ARG="--filter ${FILTER}"
MAXTRAIN_ARG=""; [[ -n "${MAX_TRAIN}" ]] && MAXTRAIN_ARG="--max-train ${MAX_TRAIN}"
CURV_ARG=""; [[ "${EXPORT_CURVATURE}" == "1" ]] && CURV_ARG="--export-curvature"
export FILTER_ARG MAXTRAIN_ARG CURV_ARG

mkdir -p "${PROJECT_HOST}/logs"
[[ -f "${IMAGE_PATH}" ]] || { echo "ERRO: SIF nao encontrado em ${IMAGE_PATH}"; exit 1; }
[[ -d "${PROJECT_HOST}/${RIEMANN_SUBDIR}" ]] || { echo "ERRO: repo nao esta em ${PROJECT_HOST}/${RIEMANN_SUBDIR}. Rode git clone (ver run_riemann_setup.slurm)."; exit 1; }

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) | $(hostname) | STEP=${STEP} VARIANT=${VARIANT}"
echo "  train=${TRAIN_ROOT}  val=${VAL_ROOT}  out=${OUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true

APP_CMD='
set -euo pipefail

# Shims de curl/wget: o .sif nao tem esses binarios, e o codigo do Francisco
# chama os dois via os.system/subprocess. Ver slurm/bin/curl e slurm/bin/wget.
export PATH="/workspace/projects/depth-riemannian/slurm/bin:${PATH}"

# --- GPU VAZIA (SLURM aloca GPU ocupada; jobs de 1 GPU exigem GPU sem ninguem) ---
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t"," -k2 -n | head -1)
GPU_IDX=$(echo "${GPU_INFO}" | cut -d"," -f1 | tr -d " ")
GPU_USED=$(echo "${GPU_INFO}" | cut -d"," -f2 | tr -d " ")
if [[ "${GPU_USED}" -gt 500 ]]; then
    echo "ERRO: nenhuma GPU VAZIA (a mais livre, GPU ${GPU_IDX}, tem ${GPU_USED} MiB). Abortando pra nao compartilhar."
    exit 1
fi
export CUDA_VISIBLE_DEVICES=${GPU_IDX}
echo ">>> GPU VAZIA: CUDA_VISIBLE_DEVICES=${GPU_IDX} (${GPU_USED} MiB em uso)"

cd "${RIE_C}"
export PYTHONUNBUFFERED=1

[[ -f "${CKPT}" ]] || { echo "ERRO: pesos ${CKPT} nao existem. Rode antes o run_riemann_setup.slurm."; exit 1; }
[[ -d "${TRAIN_ROOT}/rgb" ]] || { echo "ERRO: ${TRAIN_ROOT}/rgb nao existe. Rode antes o run_riemann_data.slurm."; exit 1; }

echo ">>> ATENCAO: confira no log a linha [Modelo] ... params treinaveis=... (se for 0 ou minusculo, ver secao 15 do roteiro)."

case "${STEP}" in
    smoke|ablation)
        python3 scripts/run_ablation.py \
            --train-root "${TRAIN_ROOT}" --val-root "${VAL_ROOT}" --checkpoint "${CKPT}" \
            --out-dir "${OUT_DIR}" --variant "${VARIANT}" \
            --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --size "${SIZE}" \
            --num-workers "${NUM_WORKERS}" ${FILTER_ARG} ${MAXTRAIN_ARG}
        ;;
    optuna)
        python3 scripts/run_optuna.py \
            --train-root "${TRAIN_ROOT}" --val-root "${VAL_ROOT}" --checkpoint "${CKPT}" \
            --out-dir "${OUT_DIR}" --variant "${VARIANT}" \
            --trials "${TRIALS}" --epochs-per-trial "${EPOCHS_PER_TRIAL}" \
            --batch-size "${BATCH_SIZE}" --size "${SIZE}" --num-workers "${NUM_WORKERS}" ${MAXTRAIN_ARG}
        ;;
    champion)
        python3 scripts/train_single.py \
            --train-root "${TRAIN_ROOT}" --val-root "${VAL_ROOT}" --checkpoint "${CKPT}" \
            --out-dir "${OUT_DIR}" --variant "${VARIANT}" \
            --berhu "${BERHU}" --grad "${GRAD}" --normal "${NORMAL}" --gauss "${GAUSS}" \
            --geod "${GEOD}" --metric "${METRIC}" --gauss-smooth-sigma "${GAUSS_SMOOTH_SIGMA}" \
            --epochs "${EPOCHS}" --seeds "${SEEDS}" --batch-size "${BATCH_SIZE}" --size "${SIZE}" \
            --num-workers "${NUM_WORKERS}" ${CURV_ARG}
        ;;
esac

echo ">>> STEP=${STEP} concluido. Saidas em ${OUT_DIR}"
'

export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all

singularity exec \
    --nvccli \
    --bind ${RAID}:/workspace \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --pwd /workspace \
    --env HF_HOME=/workspace/hf-cache \
    --env TMPDIR=/workspace/tmp \
    --env PYTHONUSERBASE=/workspace/python-packages-riemann \
    --env PYTHONPATH=/workspace/python-packages-riemann/lib/python3.10/site-packages:/workspace/python-packages-riemann/lib/python3.11/site-packages \
    --env RIE_C=${RIE_C} \
    --env STEP=${STEP} \
    --env VARIANT=${VARIANT} \
    --env CKPT=${CKPT} \
    --env TRAIN_ROOT=${TRAIN_ROOT} \
    --env VAL_ROOT=${VAL_ROOT} \
    --env OUT_DIR=${OUT_DIR} \
    --env EPOCHS=${EPOCHS} \
    --env BATCH_SIZE=${BATCH_SIZE} \
    --env SIZE=${SIZE} \
    --env NUM_WORKERS=${NUM_WORKERS} \
    --env TRIALS=${TRIALS} \
    --env EPOCHS_PER_TRIAL=${EPOCHS_PER_TRIAL} \
    --env SEEDS=${SEEDS} \
    --env BERHU=${BERHU} \
    --env GRAD=${GRAD} \
    --env NORMAL=${NORMAL} \
    --env GAUSS=${GAUSS} \
    --env GEOD=${GEOD} \
    --env METRIC=${METRIC} \
    --env GAUSS_SMOOTH_SIGMA=${GAUSS_SMOOTH_SIGMA} \
    --env FILTER_ARG="${FILTER_ARG}" \
    --env MAXTRAIN_ARG="${MAXTRAIN_ARG}" \
    --env CURV_ARG="${CURV_ARG}" \
    --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} \
    --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    "${IMAGE_PATH}" \
    bash -lc "${APP_CMD}"
