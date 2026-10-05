#!/bin/bash
#SBATCH --job-name=bokeh-b-pilot
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
# PILOTO da rota B — ~200 amostras do ITW, TODOS os gates em modo medir.
#
# O produto deste job NAO é dataset. É o histograma de rejeição e a distribuição
# de cada métrica — e, nesta rota, três números que ainda não existem:
#
#   1. a distribuição de `k_value` pela Eq. 3 sobre o ITW, contra as âncoras
#      16,6 (kfix/1000) · 20,1 (EXIF) · 15,0 (default oficial);
#   2. o deslocamento AIF<->bokeh que a DeblurNet introduz — o [A] A13, que
#      NUNCA foi medido, e que o pipeline antigo trancava em 6,0 px sem dado;
#   3. a taxa de `focus_mask_empty`, que decide se a Decisão 2 (nao aplicar o
#      refinamento do §3.2(c) na rota B) se sustenta no domínio do ITW.
#
# NÃO existe renderizador nesta rota: o alvo é a própria fotografia com bokeh
# (paper.txt:271,283,292,321). Por isso este job NÃO exige o laudo do BokehMe,
# ao contrário do `route_c_pilot.slurm`.
#
# NÃO cancela nem toca em nenhum outro job. A QOS `onejob` permite 2 rodando.
# Nada é apagado: `--output-dir` é destino novo, e a rota retoma por sample_id.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
FLUX_DIR=${FLUX_DIR:-${MODELS_DIR}/FLUX.1-dev}
GENFOCUS_DIR=${GENFOCUS_DIR:-${PROJECT_ROOT}/third_party/Genfocus}
HF_CACHE=${HF_CACHE:-${PROJECT_ROOT}/.cache/hf}
PILOT_N=${PILOT_N:-200}
# Flags extras repassadas ao entrypoint — é por aqui que entram `--shard`/`--num-shards`.
# Sem isto, exportar EXTRA_ARGS não tem efeito nenhum e a "fatia" roda o dataset inteiro.
EXTRA_ARGS=${EXTRA_ARGS:-}

# --- a variante da DeblurNet -------------------------------------------------
# DEFAULT: a OFICIAL (`nycu-cplab/Genfocus-Model`, deblurNet.safetensors,
# cond-only, main_adapter=None). É a decisão declarada: peso publicado pelos
# autores, rodado na convenção que ele exige.
#
# Quando o nosso treino main+cond terminar, a rota B é REGERADA com
#   sbatch --export=ALL,DEBLUR_VARIANT=ours_main_cond,OUTPUT_DIR=.../b_pilot_ours
# em diretório NOVO. Misturar as duas na mesma pasta faria as duas metades
# compartilharem manifest.jsonl e split.json — a mesma família do `max_coc` por
# rota, que foi o mecanismo do kfix. O script recusa, mas a disciplina é aqui.
DEBLUR_VARIANT=${DEBLUR_VARIANT:-official_cond_only}
DEBLUR_WEIGHTS_DIR=${DEBLUR_WEIGHTS_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/b_pilot_${DEBLUR_VARIANT}}

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "${HF_CACHE}"

echo "=========================================="
echo " Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:     $(hostname)"
echo " Variante: ${DEBLUR_VARIANT}"
echo " Saída:    ${OUTPUT_DIR}"
echo " Data:     $(date)"
echo "=========================================="

[[ -e "${SIF}" ]] || { echo "ERRO: SIF não encontrado: ${SIF}"; exit 1; }
for dir in "${MODELS_DIR}" "${FLUX_DIR}" "${GENFOCUS_DIR}"; do
    [[ -d "${dir}" ]] || { echo "ERRO: diretório não encontrado: ${dir}"; exit 1; }
done
# O checkout do Genfocus tem que ser o de verdade: o `DeblurNetRuntime` confere por
# AST que `generate` ainda declara `main_adapter` ANTES de subir 50 GB de backbone.
# Sem isso, o nosso `main_adapter=` seria engolido pelo `**params` em silêncio e a
# inferência voltaria a ser cond-only — o defeito B1, AIF LAVADA.
for arquivo in "${FLUX_DIR}/model_index.json" \
               "${GENFOCUS_DIR}/Inference_deblurNet.py" \
               "${GENFOCUS_DIR}/Genfocus/pipeline/flux.py"; do
    [[ -e "${arquivo}" ]] || { echo "ERRO: não encontrado: ${arquivo}"; exit 1; }
done

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

# --- seleção de GPU por UUID, nunca por índice -------------------------------
# --- a GPU vem do SLURM, e NÃO de "a mais livre agora" ---------------------------
#
# A versão anterior escolhia a GPU com menos memória em uso, via nvidia-smi, no
# arranque. Com **um** job isso funciona. Com dois subindo ao mesmo tempo, os dois
# olham, os dois veem a mesma GPU livre, e os dois vão para ela: dois FLUX de ~45 GB
# no mesmo device = `CUDA error: out of memory` dois minutos depois. Foi o que
# derrubou os jobs 32470 e 32471.
#
# O `--gres=gpu:1` já faz o SLURM reservar um device e apontar `CUDA_VISIBLE_DEVICES`
# para ele, sem corrida. Respeitar isso é o certo; a seleção manual por UUID existia
# por causa da GPU4 defeituosa, que nem sequer está no pool do SLURM nesta partição.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[erro] SLURM não exportou CUDA_VISIBLE_DEVICES. Falta --gres=gpu:N?"; exit 1
fi
echo "[gpu] alocada pelo SLURM: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          -i "${CUDA_VISIBLE_DEVICES}" 2>/dev/null | head -1)
if [ -n "${gpu_mem}" ] && [ "${gpu_mem}" -gt 2000 ]; then
  echo "[erro] a GPU alocada já tem ${gpu_mem} MiB em uso. Abortando em vez de tomar OOM."
  exit 1
fi

export PYTHONUNBUFFERED=1
# PREFIXA, nunca substitui: substituir tiraria o ~/.local do sys.path e sumiria com
# peft/diffusers, que a DeblurNet precisa. Nunca sobrescrever PYTHONUSERBASE.
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src${PYTHONPATH:+:$PYTHONPATH}
# Cache do HF em pasta do PROJETO. O ~/.cache é compartilhado entre nós.
export HF_HOME='"${HF_CACHE}"'
export HF_DATASETS_CACHE='"${HF_CACHE}"'/datasets

python3 -c "import torch, numpy, diffusers, peft; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| diffusers\", diffusers.__version__, \"| peft\", peft.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local (compileall e AST não pegam NameError; a suíte pega) <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> smoke de 2 amostras ANTES de qualquer coisa <<<"
# Foi um `import json` ausente numa linha dentro de um `finally` que matou o
# pipeline antigo na amostra 0, depois de passar por revisão. Duas amostras custam
# ~1 minuto e provam que o caminho INTEIRO executa.
python3 scripts/run_route_b.py --source itw '"${EXTRA_ARGS}"' --pilot 2 \
    --output-dir '"${OUTPUT_DIR}"'_smoke \
    --deblur-variant '"${DEBLUR_VARIANT}"' \
    '"${DEBLUR_WEIGHTS_DIR:+--deblur-weights-dir ${DEBLUR_WEIGHTS_DIR}}"' \
    --models-dir '"${MODELS_DIR}"' \
    --flux-dir '"${FLUX_DIR}"' \
    --genfocus-dir '"${GENFOCUS_DIR}"' \
    --cache-dir '"${HF_CACHE}"'/datasets \
    --device cuda --seed 0

echo
echo ">>> piloto de '"${PILOT_N}"' amostras, todos os gates em modo medir <<<"
python3 scripts/run_route_b.py --source itw '"${EXTRA_ARGS}"' --pilot '"${PILOT_N}"' \
    --output-dir '"${OUTPUT_DIR}"' \
    --deblur-variant '"${DEBLUR_VARIANT}"' \
    '"${DEBLUR_WEIGHTS_DIR:+--deblur-weights-dir ${DEBLUR_WEIGHTS_DIR}}"' \
    --models-dir '"${MODELS_DIR}"' \
    --flux-dir '"${FLUX_DIR}"' \
    --genfocus-dir '"${GENFOCUS_DIR}"' \
    --cache-dir '"${HF_CACHE}"'/datasets \
    --device cuda --seed 0

echo
echo ">>> distribuições e cortes propostos <<<"
python3 scripts/calibrate_thresholds.py \
    --pilot-dir '"${OUTPUT_DIR}"' \
    --output-json '"${PROJECT_ROOT}"'/output/thresholds_proposta_b_'"${DEBLUR_VARIANT}"'.json \
    || echo "[aviso] calibrate_thresholds.py foi escrito para a rota C; se ele nao \
aceitar os gates novos da rota B, os numeros estao no resumo acima e em \
'"${OUTPUT_DIR}"'/meta/*.json (campo quality)."
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo
echo " Antes de rodar o lote completo, leia no resumo:"
echo "   * a mediana de k_value contra 16,6 / 20,1 / 15,0. Longe das três = a Eq. 3"
echo "     está recebendo alguma coisa errada, e o diagnóstico por amostra"
echo "     (k_eq3_diagnostics) diz qual termo."
echo "   * SSIM(AIF, bokeh) e a dispersão da retenção. SSIM->1 com dispersão->0"
echo "     significa que a DeblurNet NÃO fez nada — o defeito B1."
echo "   * o deslocamento AIF<->bokeh. É o [A] A13, medido pela primeira vez."
echo "   * a taxa de focus_mask_empty. Acima de 5% revisita a Decisão 2"
echo "     (ver reference/ROTA_B_DECISOES.md)."
echo " Nenhum limiar foi aplicado aqui."
echo "=========================================="
