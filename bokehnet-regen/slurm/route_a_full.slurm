#!/bin/bash
#SBATCH --job-name=bokeh-a-full
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Rota A completa — ~1,7K imagens x ~41 variantes = ~70K pares sintéticos
# (paper.txt:527-528, 998). O alvo de cada par é RENDERIZADO pelo BokehMe [43].
#
# TRÊS PRÉ-REQUISITOS DUROS, e nenhum é formalidade:
#
#  1. **B e C regeradas**, e `manifests/k_distribution_bc.json` construído a
#     partir dos manifestos delas. O entrypoint recusa distribuição degenerada.
#  2. **O piloto rodou** e os limiares foram ESCOLHIDOS por alguém que olhou
#     output/thresholds_rota_a_proposta.json.
#  3. **O orçamento de disco foi conferido.** O entrypoint imprime a conta antes
#     de gerar: a folga medida é ~84 GB (115 GB menos os 31 GB de B+C,
#     REGISTRO.md:585). A rota A cabe a <=1 MP por imagem e NÃO cabe a >=2 MP.
#     Se a conta não fechar, reduza --samples-per-image ANTES de rodar 70 mil
#     renders — quatro coisas desta rota não têm conserto retroativo: a fonte, a
#     multiplicidade, o scene_id e a escala de K.
#
# --time é 20 dias de propósito: o custo por render do BokehMe não foi medido
# (é o que o piloto mede), e `--time` curto num job de 70 mil renders é a forma
# mais fácil de perder trabalho feito. A retomada por sample_id existe, mas
# relançar custa a fila.
#
# NÃO cancela job nenhum. Nada é apagado: a rota retoma por sample_id, então
# relançar continua de onde parou em vez de reescrever.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

# Os caminhos MEDIDOS no no, nao supostos. O default anterior apontava para
# `data/GenerativePhotography` e `data/EBB_sharp`, que nao existem — o pre-voo barrou o
# job 32690 em 0 s. O que existe:
#   fontes_rota_a/GenerativePhotography/camera_bokehK/gallery : 1000 imagens-base
#     (uma por cena; `bokehK_list` do annotations e anotacao, nao arquivo — as imagens
#      da gallery sao as BASE nitidas, nao os renders com bokeh)
#   fontes_rota_a/EBB/train/original                          : 4694 imagens nitidas
#     (o lado com bokeh do EBB! nao foi baixado, entao nao ha como contaminar a AIF)
# O loader usa rglob com filtro de extensao, entao apontar para o topo de cada fonte
# pega as imagens e ignora os .json de annotations.
DATA_ROOT=${DATA_ROOT:-/raid/user_danielpedrozo/projects/julia/fontes_rota_a}
GENPHOTO_DIR=${GENPHOTO_DIR:-${DATA_ROOT}/GenerativePhotography}
EBB_DIR=${EBB_DIR:-${DATA_ROOT}/EBB}
MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/a_synthetic}
K_DIST=${K_DIST:-${PROJECT_ROOT}/manifests/k_distribution_bc.json}
SAMPLES_PER_IMAGE=${SAMPLES_PER_IMAGE:-41}
N_FATIAS=${N_FATIAS:-2}
TOP_N_GENPHOTO=${TOP_N_GENPHOTO:-850}
TOP_N_EBB=${TOP_N_EBB:-850}

# --- os limiares congelados, vindos do piloto --------------------------------
# Preencha a partir de output/thresholds_rota_a_proposta.json. Cada linha vazia
# é um gate que MEDE e não bloqueia — escolha legítima, desde que seja escolha.
# Registre os valores usados em reference/ACHADOS.md.
THRESHOLDS=${THRESHOLDS:-""}
# exemplo:
# THRESHOLDS="--min-rendered-coc-p99-px 1.0 --max-defocus-saturation-ratio 0.30 \
#             --max-bokeh-over-aif-sharpness 0.90 --min-mask-area-ratio 0.005"

if [[ -z "${THRESHOLDS}" ]]; then
    echo "ERRO: THRESHOLDS vazio."
    echo "Rode o piloto, leia output/thresholds_rota_a_proposta.json, escolha os"
    echo "números e relance com:"
    echo "  sbatch --export=ALL,THRESHOLDS='--min-rendered-coc-p99-px 1.0 ...' $0"
    echo "Se a intenção for MESMO gerar 70 mil amostras sem filtro para calibrar"
    echo "depois, passe THRESHOLDS='--seed 0' explicitamente."
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

echo "=========================================="
echo " Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:     $(hostname)"
echo " Saída:    ${OUTPUT_DIR}"
echo " K dist:   ${K_DIST}"
echo " N/img:    ${SAMPLES_PER_IMAGE}"
echo " Limiares: ${THRESHOLDS}"
echo " Data:     $(date)"
echo "=========================================="

for caminho in "${SIF}" "${PROJECT_ROOT}/third_party/BokehMe/demo.py" \
               "${PROJECT_ROOT}/output/renderer_verification.json" "${K_DIST}"; do
    [[ -e "${caminho}" ]] || { echo "ERRO: não encontrado: ${caminho}"; exit 1; }
done
for dir in "${GENPHOTO_DIR}" "${EBB_DIR}" "${MODELS_DIR}"; do
    [[ -d "${dir}" ]] || { echo "ERRO: diretório não encontrado: ${dir}"; exit 1; }
done

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

# A GPU vem do SLURM, NAO de "a mais livre agora". Escolher por nvidia-smi no arranque
# e uma corrida: dois jobs subindo juntos veem a mesma GPU livre e vao os dois para ela.
# Ja custou um OOM nesta sessao. O --gres ja reserva e aponta CUDA_VISIBLE_DEVICES.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[erro] SLURM nao exportou CUDA_VISIBLE_DEVICES. Falta --gres=gpu:N?"; exit 1
fi
echo "[gpu] alocadas pelo SLURM: ${CUDA_VISIBLE_DEVICES}"

export PYTHONUNBUFFERED=1

# NADA em /home. O /home do cluster esta 100% cheio (425 MB livres de 1,5 TB) e e NFS
# compartilhado: o BokehMe compila kernels CuPy em tempo de execucao e o cache ia para
# $HOME/.cupy/kernel_cache -> "OSError: [Errno 28] No space left on device" matou o job
# 32691 depois de 5.371 amostras. Todo cache vai para /raid, que e local e tem 7,8 TB.
# NAO mexer no HOME: o depth_pro so existe em /home/user_juliadollis/.local/lib/
# python3.10/site-packages, que entra no sys.path via site.USER_SITE, que deriva do HOME.
# Redirecionar o HOME tirou o depth_pro do caminho e matou o job 32692 com
# "ModuleNotFoundError: No module named 'depth_pro'". Redireciona-se so os CACHES, que
# sao o que de fato enchia o disco.
export CUPY_CACHE_DIR='"${PROJECT_ROOT}"'/.cache_raid/.cupy/kernel_cache
export XDG_CACHE_HOME='"${PROJECT_ROOT}"'/.cache_raid/.cache
export TMPDIR='"${PROJECT_ROOT}"'/.cache_raid/tmp
export TRITON_CACHE_DIR='"${PROJECT_ROOT}"'/.cache_raid/.triton
export TORCH_HOME='"${PROJECT_ROOT}"'/.cache_raid/.torch
export MPLCONFIGDIR='"${PROJECT_ROOT}"'/.cache_raid/.mpl
export NUMBA_CACHE_DIR='"${PROJECT_ROOT}"'/.cache_raid/.numba
mkdir -p "${CUPY_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}" \
         "${TORCH_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
echo "[disco] HOME=${HOME} (intocado; e dele que vem o depth_pro)"
python3 -c "import depth_pro, os; print(\"[env] depth_pro ok:\", os.path.dirname(depth_pro.__file__))"
df -h '"${PROJECT_ROOT}"' | tail -1 | awk '"'"'{print "[disco] /raid: "$4" livres ("$5" usado)"}'"'"'
case "${CUPY_CACHE_DIR}" in /home/*) echo "[erro] cache ainda em /home"; exit 1 ;; esac
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

python3 -c "import torch, numpy, cupy; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| cupy\", cupy.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> rota A completa <<<"
# Uma fatia por GPU. TODOS os argumentos ficam DENTRO do laco: na versao anterior o
# laco foi inserido ANTES da lista de argumentos e o `exit $rc` caiu no meio dela --
# cada fatia era chamada so com --output-dir/--shard/--num-shards, e o job morreu em
# 26 s com "the following arguments are required: --models-dir, --bokehme-dir,
# --renderer-report, --k-distribution". Os argumentos orfaos eram codigo morto.
i=0
pids=""
for dev in $(echo "${CUDA_VISIBLE_DEVICES}" | tr "," " "); do
  out='"${OUTPUT_DIR}"'_s${i}_de${N_FATIAS}
  mkdir -p "$out"
  echo "[fatia $i] gpu ${dev} -> $out"
  CUDA_VISIBLE_DEVICES=${dev} python3 scripts/run_route_a.py \
    --output-dir "$out" \
    --shard ${i} --num-shards ${N_FATIAS} \
    --samples-per-image '"${SAMPLES_PER_IMAGE}"' \
    --k-distribution '"${K_DIST}"' \
    --source GenerativePhotography='"${GENPHOTO_DIR}"' \
    --source "EBB!"='"${EBB_DIR}"' \
    --top-n-per-source GenerativePhotography='"${TOP_N_GENPHOTO}"' \
    --top-n-per-source "EBB!"='"${TOP_N_EBB}"' \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda '"${THRESHOLDS}"' \
      > '"${PROJECT_ROOT}"'/logs/rota_a_s${i}_'"${SLURM_JOB_ID}"'.log 2>&1 &
  pids="$pids $!"
  i=$((i+1))
done
echo "[fatias] aguardando:$pids"
rc=0; for p in $pids; do wait $p || rc=1; done
for j in $(seq 0 $((N_FATIAS-1))); do
  echo "--- fatia $j ---"; tail -14 '"${PROJECT_ROOT}"'/logs/rota_a_s${j}_'"${SLURM_JOB_ID}"'.log
done

echo
echo ">>> validacao do release (NAO sobe nada; --yes e manual) <<<"
for j in $(seq 0 $((N_FATIAS-1))); do
  python3 scripts/publish_release.py --release-dir '"${OUTPUT_DIR}"'_s${j}_de${N_FATIAS} \
    || echo "[aviso] validacao da fatia $j falhou; rode a mao antes de publicar"
done
exit $rc
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo "=========================================="
