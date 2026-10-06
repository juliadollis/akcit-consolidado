#!/usr/bin/env bash
#SBATCH --job-name=deblurnet
#SBATCH --partition=gpu              # AJUSTE para a partição do seu cluster
#SBATCH --gres=gpu:h100:1            # 1× H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# --- Caminho do projeto (AJUSTE) -------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

# --- Ambiente ---------------------------------------------------------------
# Ative seu venv/conda aqui. Ex.:
#   source .venv/bin/activate
# ou
#   module load cuda/12.4 && conda activate genrefocus

# Genfocus (transformer_forward) precisa estar importável. O pacote `Genfocus`
# fica DENTRO do clone (third_party/Genfocus/Genfocus/), por isso apontamos
# para third_party/Genfocus (a raiz do clone), e não só para third_party.
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/third_party/Genfocus:${PYTHONPATH:-}"

# Cache do HF em disco rápido/local do nó (AJUSTE se necessário).
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"

# --- Segredos (NUNCA hardcode) ----------------------------------------------
# Ordem de precedencia: ja exportado no ambiente > $ENV_FILE > .env na raiz do
# projeto. Crie o .env a partir de .env.example (ele e git-ignored):
#   cp .env.example .env && ${EDITOR:-nano} .env
for _env in "${ENV_FILE:-}" "${SLURM_SUBMIT_DIR:-}/.env" "${PROJECT_ROOT:-}/.env" "${PROJECT_DIR:-}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?defina HF_TOKEN no .env da raiz do projeto (cp .env.example .env)}"
: "${WANDB_API_KEY:?defina WANDB_API_KEY no .env da raiz do projeto}"
export HF_TOKEN WANDB_API_KEY

echo "[slurm] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "[slurm] PYTHONPATH=${PYTHONPATH}"

# --- Sanity check: Genfocus importável -------------------------------------
python -c "from Genfocus.pipeline.flux import transformer_forward; print('[slurm] Genfocus ok')"

# --- Smoke test (rápido) antes do treino real ------------------------------
echo "[slurm] Rodando smoke test..."
python -m genfocus_train.train smoke --config configs/train_smoke.yaml

# --- Treino real ------------------------------------------------------------
echo "[slurm] Iniciando treino da DeblurNet (60K steps)..."
python -m genfocus_train.train deblur --config configs/train_base.yaml
