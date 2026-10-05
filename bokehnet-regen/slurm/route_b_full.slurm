#!/bin/bash
#SBATCH --job-name=bokeh-b-full
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Rota B completa — 13.800 fotos do ITW, K pela Eq. 3, alvo = a própria foto.
#
# PRÉ-REQUISITO 1: o piloto rodou e os limiares foram ESCOLHIDOS por alguém que
# olhou a distribuição. Este job recusa começar com THRESHOLDS vazio — rodar 13
# mil amostras sem filtro é gerar o dataset de novo com o mesmo problema.
#
# PRÉ-REQUISITO 2: a variante da DeblurNet é uma DECISÃO, e ela vai num
# --output-dir próprio. Duas variantes na mesma pasta compartilhariam
# manifest.jsonl e split.json.
#
# NÃO existe renderizador nesta rota. O alvo é a fotografia real
# (paper.txt:271,283,292,321), então não há laudo de BokehMe a exigir, não há
# SSIM de calibração e não há k_effective_factor a aplicar.
#
# NÃO cancela job nenhum. Nada é apagado: a rota retoma por sample_id, então
# relançar continua de onde parou em vez de reescrever.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
FLUX_DIR=${FLUX_DIR:-${MODELS_DIR}/FLUX.1-dev}
GENFOCUS_DIR=${GENFOCUS_DIR:-${PROJECT_ROOT}/third_party/Genfocus}
HF_CACHE=${HF_CACHE:-${PROJECT_ROOT}/.cache/hf}

DEBLUR_VARIANT=${DEBLUR_VARIANT:-official_cond_only}
DEBLUR_WEIGHTS_DIR=${DEBLUR_WEIGHTS_DIR:-}
# Trava opcional de peso: um arquivo renomeado passa pela checagem de nome, o
# sha256 não passa. Preencha depois do piloto com o valor que o run_config.json
# gravou — é a defesa contra o único cruzamento de variante que o nome não pega.
DEBLUR_LORA_SHA256=${DEBLUR_LORA_SHA256:-}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/b_itw_${DEBLUR_VARIANT}}

# --- os limiares congelados, vindos do piloto --------------------------------
# Preencha a partir da distribuição que o piloto imprimiu. Cada linha que ficar
# vazia é um gate que MEDE e não bloqueia — o que é uma escolha legítima, desde
# que seja escolha. Registre os valores usados em reference/ACHADOS.md.
THRESHOLDS=${THRESHOLDS:-""}
# exemplo (NÃO são números medidos — são a FORMA da linha):
# THRESHOLDS="--min-deblur-structural-ssim 0.35 --max-deblur-structural-ssim 0.97 \
#             --max-pair-registration-shift-px 6.0 --min-mask-area-ratio 0.01 \
#             --max-mask-area-ratio 0.85 --min-focus-mask-sharpness-ratio 1.0 \
#             --min-k-value 1.0 --max-k-value 200.0"

if [[ -z "${THRESHOLDS}" ]]; then
    echo "ERRO: THRESHOLDS vazio."
    echo "Rode o piloto (route_b_pilot.slurm), leia a distribuição, escolha os números"
    echo "e relance com:"
    echo "  sbatch --export=ALL,THRESHOLDS='--min-deblur-structural-ssim 0.35 ...' $0"
    echo
    echo "Atenção especial aos dois limiares de K: o --k-max 300 da rota C virou 47,0%"
    echo "de amostras censuradas no teto EXATO (ACHADOS.md:19) sem que nada denunciasse."
    echo "Aqui K fora da faixa REJEITA (não censura), então um teto errado aparece como"
    echo "k_out_of_configured_range no histograma — mas ainda é um teto errado."
    echo
    echo "Se a intenção for MESMO gerar tudo sem filtro para calibrar depois,"
    echo "passe THRESHOLDS='--seed 0' explicitamente."
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "${HF_CACHE}"

echo "=========================================="
echo " Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:     $(hostname)"
echo " Variante: ${DEBLUR_VARIANT}"
echo " Saída:    ${OUTPUT_DIR}"
echo " Limiares: ${THRESHOLDS}"
echo " Data:     $(date)"
echo "=========================================="

[[ -e "${SIF}" ]] || { echo "ERRO: SIF não encontrado: ${SIF}"; exit 1; }
for dir in "${MODELS_DIR}" "${FLUX_DIR}" "${GENFOCUS_DIR}"; do
    [[ -d "${dir}" ]] || { echo "ERRO: diretório não encontrado: ${dir}"; exit 1; }
done
for arquivo in "${FLUX_DIR}/model_index.json" \
               "${GENFOCUS_DIR}/Inference_deblurNet.py" \
               "${GENFOCUS_DIR}/Genfocus/pipeline/flux.py"; do
    [[ -e "${arquivo}" ]] || { echo "ERRO: não encontrado: ${arquivo}"; exit 1; }
done

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -z "$free_gpu" ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi
gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
gpu_mem=$(echo "$free_gpu"  | cut -d, -f3 | tr -d " ")
echo "[gpu] mais livre: memory.used=${gpu_mem} MiB"
if [ "$gpu_mem" -gt 2000 ]; then
  echo "[erro] GPU mais livre com ${gpu_mem} MiB em uso. Abortando em vez de tomar OOM."
  exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu_uuid

export PYTHONUNBUFFERED=1
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME='"${HF_CACHE}"'
export HF_DATASETS_CACHE='"${HF_CACHE}"'/datasets

python3 -c "import torch, numpy, diffusers, peft; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| diffusers\", diffusers.__version__, \"| peft\", peft.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> smoke de 2 amostras, com os MESMOS limiares do lote <<<"
python3 scripts/run_route_b.py --source itw --pilot 2 \
    --output-dir '"${OUTPUT_DIR}"'_smoke \
    --deblur-variant '"${DEBLUR_VARIANT}"' \
    '"${DEBLUR_WEIGHTS_DIR:+--deblur-weights-dir ${DEBLUR_WEIGHTS_DIR}}"' \
    '"${DEBLUR_LORA_SHA256:+--deblur-lora-sha256 ${DEBLUR_LORA_SHA256}}"' \
    --models-dir '"${MODELS_DIR}"' --flux-dir '"${FLUX_DIR}"' \
    --genfocus-dir '"${GENFOCUS_DIR}"' \
    --cache-dir '"${HF_CACHE}"'/datasets \
    --device cuda --seed 0 '"${THRESHOLDS}"' || {
  echo "[erro] o smoke com os limiares congelados nao aceitou NENHUMA amostra."
  echo "       Isso normalmente significa limiar escolhido fora da distribuicao."
  echo "       Veja o histograma acima antes de rodar 13.800."
  exit 1
}

echo
echo ">>> rota B completa <<<"
python3 scripts/run_route_b.py --source itw \
    --output-dir '"${OUTPUT_DIR}"' \
    --deblur-variant '"${DEBLUR_VARIANT}"' \
    '"${DEBLUR_WEIGHTS_DIR:+--deblur-weights-dir ${DEBLUR_WEIGHTS_DIR}}"' \
    '"${DEBLUR_LORA_SHA256:+--deblur-lora-sha256 ${DEBLUR_LORA_SHA256}}"' \
    --models-dir '"${MODELS_DIR}"' --flux-dir '"${FLUX_DIR}"' \
    --genfocus-dir '"${GENFOCUS_DIR}"' \
    --cache-dir '"${HF_CACHE}"'/datasets \
    --device cuda --seed 0 '"${THRESHOLDS}"'

echo
echo ">>> validação do release (NÃO sobe nada; --yes é manual) <<<"
python3 scripts/publish_release.py --release-dir '"${OUTPUT_DIR}"'
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo
echo " O release desta pasta é de UMA variante da DeblurNet: ${DEBLUR_VARIANT}."
echo " A outra versão vai em pasta e repo HF PRÓPRIOS, com o mesmo sample_id —"
echo " é o que torna as duas comparáveis par a par."
echo "=========================================="
