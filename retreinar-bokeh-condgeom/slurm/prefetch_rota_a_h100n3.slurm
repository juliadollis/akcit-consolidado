#!/bin/bash
#SBATCH --job-name=bokeh-prefetch-rota-a
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# BAIXA O RELEASE DA ROTA A PARA O DISCO. UMA VEZ. SEM GPU.
# =============================================================================
#
# POR QUE ISTO EXISTE, E NAO UM download dentro do dataloader:
#   Ha registro de 552 GB de egress inexplicado neste projeto, vindo de download
#   automatico. Por isso `permitir_download_do_release: false` e o DEFAULT no
#   YAML: o treino LE DO DISCO e falha alto se o release nao estiver la.
#   Baixar e um ato deliberado, com job proprio, log proprio e volume medido.
#
# SEM --gres: e trabalho de CPU e rede. Nao ocupa GPU, e nao conta contra as
# 2 vagas da QOS `onejob` que o treino precisa.
#
# IDEMPOTENTE: o `snapshot_download` reusa o que ja esta em disco. Rodar de novo
# depois que o release for completado baixa SO o que faltava.
#
# NAO APAGA NADA. Sem `rm`, sem `--delete`. Se o destino ja existir, completa.
#
# COMO RODAR
#   ssh dgx-H100-03
#   cd /raid/user_danielpedrozo/projects/julia/retreinar-bokeh
#   mkdir -p logs
#   sbatch slurm/prefetch_rota_a_h100n3.slurm
#
# VARIAVEIS
#   REPO=juliadollis/bokehnet-regen-rota-a   # o release
#   DEST=<caminho>                           # destino em disco
# =============================================================================

set -euo pipefail

BIND_ROOT="${BIND_ROOT:-/raid/user_danielpedrozo}"
PROJECT_ROOT="${PROJECT_ROOT:-${BIND_ROOT}/projects/julia/retreinar-bokeh}"
SIF="${SIF:-${BIND_ROOT}/images/transformers-pytorch-gpu.sif}"
REPO="${REPO:-juliadollis/bokehnet-regen-rota-a}"
DEST="${DEST:-${BIND_ROOT}/projects/julia/releases/rota_a}"

cd "${PROJECT_ROOT}"
mkdir -p logs

for _env in "${ENV_FILE:-}" "${PROJECT_ROOT}/.env" "${PWD}/.env"; do
  if [ -n "${_env}" ] && [ -f "${_env}" ]; then set -a; . "${_env}"; set +a; break; fi
done
: "${HF_TOKEN:?ERRO: HF_TOKEN nao definido. cp .env.example .env && edite (chmod 600 .env)}"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${BIND_ROOT}/projects/julia/hf-cache}"

[ -f "${SIF}" ] || { echo "ERRO: container nao encontrado em ${SIF}"; exit 1; }

echo "=============================================================="
echo "Job:     ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-local}) @ $(hostname)"
echo "Repo:    ${REPO}"
echo "Destino: ${DEST}"
echo "Cota antes:"; quota -s 2>/dev/null | tail -2 || true
echo "=============================================================="

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'
export PYTHONPATH='"${PROJECT_ROOT}"':'"${PROJECT_ROOT}"'/third_party/Genfocus${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

python3 - <<PYEOF
import os
from huggingface_hub import HfApi, snapshot_download

repo = "'"${REPO}"'"
dest = "'"${DEST}"'"
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

arquivos = api.list_repo_files(repo, repo_type="dataset")
from collections import Counter
def pasta(f): return f.split("/")[0] if "/" in f else "(raiz)"
print("[prefetch] arquivos no release:", len(arquivos))
for p, n in Counter(pasta(f) for f in arquivos).most_common():
    print(f"[prefetch]   {p:22s} {n}")

# CONFERENCIA DE COMPLETUDE, antes de gastar rede. O release tem que trazer os
# quatro grupos; sem eles o treino nao roda e e melhor descobrir aqui.
faltando = [g for g in ("meta/", "generated/", "manifest.jsonl", "split.json")
            if not any(f == g or f.startswith(g) for f in arquivos)]
if faltando:
    print()
    print("[prefetch] AVISO GRAVE: o release esta INCOMPLETO. Ausentes:", faltando)
    print("[prefetch]   meta/        -> k_value, focus_disparity, disparity_min/max")
    print("[prefetch]                   sem eles o mapa de defocus NAO pode ser montado")
    print("[prefetch]   generated/   -> o bokeh ALVO, o que a rede aprende a produzir")
    print("[prefetch]   manifest/split -> a fronteira por CENA entre treino e validacao")
    print("[prefetch] Vou baixar o que existe, mas o TREINO NAO RODA assim.")
    print()

os.makedirs(dest, exist_ok=True)
caminho = snapshot_download(
    repo_id=repo, repo_type="dataset", token=token,
    local_dir=dest, max_workers=8,
)
print("[prefetch] em disco:", caminho)

total = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, fs in os.walk(caminho) for f in fs
)
print(f"[prefetch] tamanho em disco: {total/1e9:.2f} GB")
PYEOF

echo "[prefetch] aponte o YAML para: '"${DEST}"'"
'

srun singularity exec \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}" \
  "${SIF}" bash -lc "${APP_CMD}"

RC=$?   # capturado ANTES do `quota`, senao `rc` seria o do `|| true`
echo "Cota depois:"; quota -s 2>/dev/null | tail -2 || true
echo "=== FIM (rc=${RC}) ==="
