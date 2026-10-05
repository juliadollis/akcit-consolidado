#!/usr/bin/env bash
# Smoke da fase 2: prova o caminho de arranque ANTES de o espelho voltar.
#
# O que nunca foi exercitado e importa:
#   1. `--init-lora` carregando os 686 tensores do checkpoint da fase 1 num
#      backbone novo, com `exigir_completo=True` — se `missing` ou `unexpected`
#      não vierem vazios, o treino recusa, e é melhor saber agora.
#   2. Na RETOMADA o `init-lora` tem de ser PULADO (a condição é
#      `global_step == 0`), senão a fase 2 reiniciaria dos pesos da fase 1 toda
#      vez que o container caísse.
#   3. O arranque multi-GPU com todas as guardas novas (shm, FLUX, probe).
set -u
CK=/workspace/retreinar-bokeh/outputs/bokeh_fase1_1700cenas/bokeh/checkpoints/step_40000.pt
L=/workspace/releases/smoke_fase2.log

roda() {
  accelerate launch --multi_gpu --num_machines 1 --num_processes 2 \
    --mixed_precision bf16 --dynamo_backend no --main_process_port "$1" \
    -m genfocus_train.train bokeh --config configs/_smoke_fase2.yaml \
    --grad-accum 2 --init-lora "$CK" >> "$L" 2>&1
}

echo "--- 1a rodada: 6 steps do zero, com --init-lora da fase 1 ---" > "$L"
roda 29620
echo "exit1=$?" >> "$L"
echo "--- 2a rodada: RETOMADA, o init-lora tem de ser PULADO ---" >> "$L"
roda 29621
echo "exit2=$?" >> "$L"
echo SMOKE_FIM >> "$L"
