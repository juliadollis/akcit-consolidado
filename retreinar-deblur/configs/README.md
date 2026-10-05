# Configs de treino — DeblurNet (Stage 1)

Objetivo desta arvore: **replicar o paper e a inferencia oficial, e treinar**.
Nao ha configs de experimento aqui.

| Config | Para que serve | Roda com |
|---|---|---|
| `train_smoke.yaml` | 3 steps no caminho real, 4 amostras, sem wandb e sem upload. Valida import do Genfocus, carga do FLUX, a trava dos 343 modulos LoRA, forward/backward e export. | `sbatch slurm/smoke.slurm` |
| `train_deblur_paper.yaml` | **O config principal.** Replica o §4.1 e o §B.1: rank 128, 60.000 steps, batch efetivo 32, DPDD train inteiro + top-3000 do RealBokeh. Gera um peso que roda no `Inference_deblurNet.py` oficial sem parametro extra. | `sbatch slurm/train_deblur_4gpu.slurm` |
| `train_deblur_maincond.yaml` | Variante `lora_on_main: true`, a familia do DeblurNet de 60k ja avaliado. **Exige `main_adapter="deblurring"` na inferencia** — com `main_adapter=None` a saida sai lavada. | `TRAIN_CONFIG=configs/train_deblur_maincond.yaml sbatch slurm/train_deblur_4gpu.slurm` |
| `_override_1gpu.yaml` | Gerado automaticamente pelo `train_deblur_1gpu.slurm` quando se passa `GRAD_ACCUM=32`. Nao edite a mao; nao versione. | (automatico) |

## Jobs

| Script | O que faz |
|---|---|
| `slurm/smoke.slurm` | Smoke de 3 steps, 1 GPU. Rode isto antes do treino longo. |
| `slurm/train_deblur_4gpu.slurm` | Treino principal, 4 GPUs, batch efetivo 32. `--time` de 14 dias. |
| `slurm/train_deblur_1gpu.slurm` | Mesmo treino em 1 GPU. Use `GRAD_ACCUM=32` para manter o batch efetivo do paper. |
| `slurm/medicoes_c0.slurm` | Diagnosticos C0 (resolucao dos dfs, GTs distintas, contagem de modulos LoRA). Nao treina nada. |

## Antes do primeiro `sbatch`

```bash
cd /raid/user_juliadollis/projects/retreinar-deblur
cp .env.example .env && chmod 600 .env   # preencha HF_TOKEN e WANDB_API_KEY
mkdir -p logs                             # o SLURM nao cria o diretorio de log
sbatch slurm/smoke.slurm                  # so depois o treino longo
```

## O que os tres campos de eixo significam

`scale_mode`, `sigma_mu_source` e `top_k_mode` existem no schema mas estao no
valor que **preserva o comportamento anterior** (`short_side`, `crop`, `row`).
E com esses valores que o projeto chegou a LPIPS 0.1446 no DPDD, contra 0.1440
do paper. O paper nao especifica resolucao de treino, tamanho de crop, nem como
desempatar o ranking de nitidez — entao nao existe "valor do paper" para copiar.
Trocar qualquer um deles sem medir arrisca perder o resultado que ja temos.

## Retomada e seguranca

- `resume: true` + checkpoint a cada 250 steps: re-`sbatch` continua de onde parou.
- `wandb_resume: true`: continua o **mesmo** grafico, em vez de abrir um run novo.
- `save_best: true`: guarda o melhor por `val_loss` alem dos 5 ultimos.
- `upload_every_steps: 1000`: snapshot no HF durante o treino, um arquivo por step.
- Nenhum script deste diretorio apaga arquivo, cancela job ou sobrescreve peso remoto.
