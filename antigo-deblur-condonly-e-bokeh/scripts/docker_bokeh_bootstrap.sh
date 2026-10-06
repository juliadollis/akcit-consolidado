#!/usr/bin/env bash
# =============================================================================
# Roda DENTRO do container (imagem julia-genrefocus:1.0) na dgx-H100-01.
# Chamado por scripts/run_bokeh_docker.sh — nao rode na mao.
#
# DIFERENCA vs o antigo scripts/docker_bokeh_bootstrap.sh do DEBLUR: aqui NAO ha
# `pip install` nenhum. As versoes ja vem FIXAS na imagem (docker/Dockerfile).
# Instalar pacote em tempo de execucao e o que produziu a combinacao quebrada
# que custou 4 jobs na H100-02 (secao 4 do historico-ultimo.md).
# =============================================================================
set -euo pipefail

cd /workspace/genrefocus_deblurnet_paper
export PYTHONPATH="/workspace/genrefocus_deblurnet_paper:/workspace/genrefocus_deblurnet_paper/third_party/Genfocus:${PYTHONPATH:-}"

NUM_GPUS=${NUM_GPUS:-4}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_bokeh_real_4gpu.yaml}
# Config do SMOKE e SEPARADA da do treino: o smoke default da H100-02
# (train_bokeh_smoke.yaml) le a pasta local /workspace/data-bokeh/rota_a, que so
# existe naquele no. Aqui o smoke le a rota c do HUB, que e fonte da propria
# fase 2. Use SMOKE_CONFIG=... para trocar.
SMOKE_CONFIG=${SMOKE_CONFIG:-configs/train_bokeh_smoke_hub.yaml}
INIT_LORA=${INIT_LORA:-}
HF_REPO_ID=${HF_REPO_ID:-genrefocus-bokehnet-real-4gpu}
RUN_SMOKE=${RUN_SMOKE:-0}

echo "=========================================="
echo "BOKEHNET FASE 2 — docker (dgx-H100-01, sem SLURM)"
echo "NUM_GPUS:   ${NUM_GPUS}"
echo "TRAIN_CFG:  ${TRAIN_CONFIG}"
echo "INIT_LORA:  ${INIT_LORA:-<nenhum>}"
echo "HF_HOME:    ${HF_HOME:-<unset>}"
echo "uid:gid:    $(id -u):$(id -g)"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

# 1) Ambiente: confere ANTES de gastar dias de GPU. A base da imagem e uma tag
#    movel, entao uma atualizacao dela pode trocar o transformers sem aviso.
echo ">>> conferindo o ambiente contra o manifesto..."
python3 docker/verify_env.py || { echo "ERRO: ambiente divergente. Abortando ANTES do treino."; exit 1; }

# 2) Genfocus (transformer_forward dos autores) — clona se faltar.
if [ ! -f third_party/Genfocus/Genfocus/pipeline/flux.py ]; then
    echo ">>> clonando Genfocus..."
    bash scripts/setup_genfocus.sh
fi
python3 -c "from Genfocus.pipeline.flux import transformer_forward; print('[ok] Genfocus importavel')"

# 3) FLUX: ja deve estar no cache montado (copiado no host). NAO baixa 54 GB de
#    novo sem necessidade — so avisa se faltar.
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("black-forest-labs/FLUX.1-dev", token=os.environ.get("HF_TOKEN"),
                          local_files_only=True)
    print(f"[ok] FLUX ja no cache: {p}")
except Exception:
    print("[aviso] FLUX nao esta completo no cache; sera baixado (54 GB, demora).")
PY

# 4) wandb (nao fatal: o trainer degrada para JSONL com aviso)
python3 -c "import wandb, os; wandb.login(key=os.environ['WANDB_API_KEY'])" \
    || echo "[aviso] wandb login falhou; o treino segue e loga em JSONL"

# 5) smoke opcional — 3 steps no caminho real de 2 condicoes, 1 GPU.
if [ "${RUN_SMOKE}" = "1" ]; then
    echo ">>> SMOKE (3 steps, 1 GPU) — ${SMOKE_CONFIG} — valida o caminho antes dos 60K"
    PYTHONUNBUFFERED=1 python3 -m genfocus_train.train smoke \
        --config "${SMOKE_CONFIG}" --stage bokeh
    echo ">>> smoke terminou. Saindo (RUN_SMOKE=1 nao dispara o treino)."
    exit 0
fi

# 6) FASE 2: 60K steps em dados REAIS (rotas b+c), iniciando do LoRA da fase 1
#    com optimizer/scheduler RESETADOS (o --init-lora so age quando o run comeca
#    do zero; num resume ele e ignorado).
INIT_ARG=""
[ -n "${INIT_LORA}" ] && INIT_ARG="--init-lora ${INIT_LORA}"

echo ">>> TREINO FASE 2 — ${NUM_GPUS} GPUs — ${TRAIN_CONFIG} ${INIT_ARG}"
# --multi_gpu EXIGE >= 2 processos: com 1 GPU o accelerate aborta com
# "You need to use at least 2 processes to use `--multi_gpu`". Isso derrubou a
# cadeia do kfix em 2026-08-20, no smoke de 1 GPU. Em single-GPU o caminho
# correto e sem a flag (o trainer ja trata os dois casos).
MULTI=""
[ "${NUM_GPUS}" -gt 1 ] && MULTI="--multi_gpu"
PYTHONUNBUFFERED=1 accelerate launch \
    ${MULTI} --num_machines 1 --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 --dynamo_backend no \
    -m genfocus_train.train bokeh --config "${TRAIN_CONFIG}" ${INIT_ARG}
