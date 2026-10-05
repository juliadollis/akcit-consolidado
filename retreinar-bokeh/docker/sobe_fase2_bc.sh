#!/usr/bin/env bash
# =============================================================================
# Fase 2 da BokehNet (§4.1 fase ii): rotas B e C JUNTAS, 60K steps, 4 GPUs.
# =============================================================================
# Cada flag aqui existe por um defeito MEDIDO. Não tire nenhuma sem ler o motivo.
#
#   --user $(id -u):$(id -g)   sem isto os arquivos nascem root. Os 54 GB de
#                              hf-cache root-owned desta máquina vieram disso, e
#                              é por causa deles que o FLUX só carrega por
#                              caminho de SNAPSHOT (ver FLUX_PATH abaixo).
#   --shm-size=64g --ipc=host  SEM ISTO O TREINO MORRE. Os workers do DataLoader
#                              trocam tensores por memória compartilhada; com os
#                              64 MB default do Docker eles caem com
#                              "Bus error" DEPOIS de o FLUX carregar, e em
#                              multi-GPU o processo nem morre: PENDURA, com um
#                              rank a 0% e outro a 100% (medido em 2026-09-25).
#   --gpus '"device=..."'      as aspas embutidas são obrigatórias com várias.
#   -d                         detached: sobrevive à queda de ssh/VPN (já
#                              aconteceu 2x).
#   HF_HOME=hf-cache-julia     GRAVÁVEL. O `hf-cache/hub` é root-owned e o
#                              download fica em 0 arquivos só avisando
#                              "Ignored error ... Permission denied".
#   FLUX_PATH                  caminho do SNAPSHOT. O repo-id manda o diffusers
#                              pelo `hf_hub_download`, que quer escrever em
#                              `blobs/` e falha; e baixar é contra a regra do
#                              projeto (552 GB de egress inexplicado). O treino
#                              agora RECUSA um caminho que não seja local.
#   TORCH_NCCL_...=3600        limite alto, como manda o CLAUDE.md.
#   .env lido DENTRO             HF_TOKEN e WANDB_API_KEY não entram como `-e`
#                              nem `--env-file`: os dois vazam em
#                              `docker inspect`, que qualquer um na máquina
#                              roda. O container lê o arquivo montado.
#   --init-lora                a fase 2 parte do LoRA da fase 1, com optimizer e
#                              scheduler FRESCOS (§10.1 — decisão nossa, o paper
#                              não menciona LR).
#
# Este script SÓ CRIA. Nunca remove container de ninguém, inclusive nosso.
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
NOME="${NOME:-julia_fase2_bc}"
CFG="${CFG:-configs/train_bokeh_fase2_bc.yaml}"
INIT="${INIT:-/workspace/retreinar-bokeh/outputs/bokeh_fase1_1700cenas/bokeh/checkpoints/step_40000.pt}"
NPROC="$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')"
# ACUMULAÇÃO DERIVADA DO NÚMERO DE GPUS, não fixa.
# O §4.1 especifica BATCH EFETIVO 32 = 1 x accum x n_gpus. Com `--grad-accum 8`
# hardcoded, 4 GPUs dão 32 (certo) e 2 GPUs dão 16 (o experimento errado, com o
# banner anunciando o número certo de accum). O que se conserva é o 32.
ACCUM="${ACCUM:-$((32 / NPROC))}"
if [ $((ACCUM * NPROC)) -ne 32 ]; then
  echo "AVISO: accum $ACCUM x $NPROC GPUs = $((ACCUM * NPROC)), e o §4.1 pede 32." >&2
fi
FLUX="/workspace/hf-cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"

echo "GPUs=$GPUS (${NPROC} processos) · batch efetivo = 1 x ${ACCUM} x ${NPROC} = $((ACCUM * NPROC))"
echo "config=$CFG"
echo "init-lora=$INIT"

docker run -d --name "$NOME" \
  --user "$(id -u):$(id -g)" \
  --gpus "\"device=${GPUS}\"" \
  --shm-size=64g --ipc=host \
  -v /raid/user_juliadollis/julia_docker:/workspace \
  -w /workspace/retreinar-bokeh \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
  -e NCCL_DEBUG=WARN \
  -e HF_HOME=/workspace/hf-cache-julia \
  -e FLUX_PATH="$FLUX" \
  -e PYTHONPATH=/workspace/retreinar-bokeh:/workspace/retreinar-bokeh/third_party/Genfocus \
  julia-genrefocus:1.0 bash -lc "
    # O .env e lido AQUI DENTRO, em vez de virar -e / --env-file. Com qualquer
    # uma das duas o token aparece em \`docker inspect\`, que nao pede
    # privilegio nenhum, numa maquina com containers de outras cinco pessoas.
    # Assim o Config.Env do container nao contem segredo: contem o literal
    # \". .env\", e o valor so existe dentro do processo.
    set -a; . /workspace/retreinar-bokeh/.env; set +a
    accelerate launch --multi_gpu --num_machines 1 --num_processes ${NPROC} \
      --mixed_precision bf16 --dynamo_backend no --main_process_port 29540 \
      -m genfocus_train.train bokeh --config ${CFG} --grad-accum ${ACCUM} \
      --init-lora ${INIT}
  "

echo
echo "docker logs -f $NOME"
echo "monitor externo (GPU separada):  bash docker/sobe_monitor.sh"
