#!/bin/bash
#SBATCH --job-name=hf-upload-test
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.out
#SBATCH --error=/raid/user_juliadollis/projects/genrefocus_deblurnet/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# TESTE de upload pro HuggingFace — NÃO TREINA, NÃO APAGA NADA.
# Só cria os 2 repos no seu HF e sobe um arquivo de sanidade (UPLOAD_TEST.txt)
# em cada. Serve pra confirmar que o token e a permissão de escrita funcionam
# ANTES de depender do upload automático no fim do treino.
#
# Rodar:   sbatch scripts/test_upload.slurm
# Conferir: tail -n 40 logs/hf-upload-test-<jobid>.out
#           e abrir os links impressos no navegador.
# =============================================================================

HF_TOKEN="${HF_TOKEN:?defina HF_TOKEN no ambiente}"
export HF_TOKEN

export PROJECT_ROOT=/raid/user_juliadollis/projects/genrefocus_deblurnet
export IMAGE_PATH=/raid/user_juliadollis/images/transformers-pytorch-gpu.sif
export PYTHONUSERBASE=/workspace/python-packages

# repos de destino (sem '/' => usa o namespace do seu usuário HF)
REPOS=("genrefocus-deblurnet-ddpd" "genrefocus-deblurnet-paper")

mkdir -p "${PROJECT_ROOT}/logs"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "ERRO: imagem Singularity não encontrada em ${IMAGE_PATH}"; exit 1
fi

echo "=========================================="
echo "Job:   ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Host:  $(hostname)"
echo "Repos: ${REPOS[*]}"
echo "=========================================="

for repo in "${REPOS[@]}"; do
    echo ">>> testando upload no repo: ${repo}"
    singularity exec \
        --bind /raid/user_juliadollis:/workspace \
        --env HF_TOKEN=${HF_TOKEN} \
        --env HUGGINGFACE_HUB_TOKEN=${HF_TOKEN} \
        --env PYTHONUSERBASE=/workspace/python-packages \
        "${IMAGE_PATH}" \
        python3 /workspace/projects/genrefocus_deblurnet/scripts/hf_upload.py \
            --test --repo-id "${repo}"
done

echo ">>> TESTE DE UPLOAD CONCLUÍDO. Confira os links acima no navegador."
