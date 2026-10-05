# retreinar-deblur — DeblurNet (GenRefocus Stage 1)

Replicação do **Stage 1** do paper *Generative Refocusing: Flexible Defocus
Control from a Single Image* (arXiv **2512.16923v3**): o LoRA da **DeblurNet**
sobre FLUX.1-dev, condicionado por concatenação de tokens no estilo
OminiControl/OminiControl2.

O objetivo desta árvore é **um só**: reproduzir o paper e a inferência oficial
dos autores. Sem experimentos, sem varredura de hiperparâmetro.

## Por que esta pasta existe separada

Havia duas árvores de treino, e as duas continuam **intactas e rodáveis**:

| Árvore | Variante de LoRA | Situação |
|---|---|---|
| `genrefocus_deblurnet_paper/` | cond-only | preservada, base desta cópia |
| `genrefocus_deblurnet/` | main+cond | preservada, é o modelo de 60k já avaliado |
| **`retreinar-deblur/`** (esta) | cond-only corrigida | onde o retreino acontece |

Nada aqui altera as outras duas. A separação existe porque esta revisão mudou
coisas que **invalidam checkpoints antigos** — em especial o C1, que remove uma
camada LoRA a mais que estava sendo injetada.

## O que mudou em relação ao treino antigo

Resumo dos achados que motivaram o retreino. O detalhe completo, com o código
de antes e de depois, está em **[`MUDANCAS_CODIGO.md`](MUDANCAS_CODIGO.md)**.

| | O que era | O que é |
|---|---|---|
| **C1** | 344 módulos LoRA — a string `"proj_out"` capturava também a projeção final do transformer, que roda **fora** do controle por branch | **343**, igual ao checkpoint oficial. Com trava dura que aborta se divergir |
| **C2** | `guidance` 1.0 no treino, 3.5 na inferência oficial de deblur | 3.5 nos dois, configurável por estágio |
| **C3** | o 1º update rodava com LR 1e-4 quando a rampa prevê 2e-7 | `scheduler.step()` antes do laço. Correção de **uma** linha |
| **C7** | as 4 GPUs sorteavam o mesmo σ e o mesmo ruído | seed por rank |
| **C8** | o checkpoint não registrava qual variante o gerou | metadata completa + `.json` ao lado do `.safetensors` |
| **C10** | `best_loss` era a loss de um micro-batch sorteado | validação com σ e ruído **fixos**, e `best.pt` por ela |

## Qual arquivo responde qual pergunta

| Pergunta | Arquivo |
|---|---|
| O que mudou no código, e por quê? | [`MUDANCAS_CODIGO.md`](MUDANCAS_CODIGO.md) |
| Qual era o plano de correções (C0..C11), item por item? | [`docs/PLANO_CORRECOES_DEBLURNET.md`](docs/PLANO_CORRECOES_DEBLURNET.md) |
| O que é fato medido e o que é hipótese? De onde veio cada afirmação? | [`docs/REGISTRO_REVISAO.md`](docs/REGISTRO_REVISAO.md) |
| O plano em página navegável | [`docs/plano.html`](docs/plano.html) |
| Qual config responde qual pergunta? | [`configs/README.md`](configs/README.md) |
| O que cada script mede, e precisa de GPU/rede? | [`scripts/README.md`](scripts/README.md) |
| Como reproduzir os testes do paper (Tabela 2)? | [`inferencia/`](inferencia/) |

> Os comentários dentro de `genfocus_train/*.py` e de `slurm/*.slurm` citam
> `PLANO_CORRECOES_DEBLURNET.md` e `REGISTRO_REVISAO.md` pelo nome, sem caminho.
> Esses dois agora vivem em **`docs/`**. Os arquivos não foram editados de
> propósito: há treino rodando que lê essa árvore.

## Estrutura

```
retreinar-deblur/
├── MUDANCAS_CODIGO.md     documento de entrada
├── genfocus_train/        o pacote de treino (backbone, data, trainer, models, config)
├── configs/               YAMLs — um por cenário, com as citações do paper
├── slurm/                 jobs para o cluster DGX (h100n2 / h100n3)
├── docker/                imagem e launcher para a H100-01 (sem SLURM)
├── scripts/               medições (C0), inferência e avaliação
├── inferencia/            reprodução da Tabela 2 do paper
├── tests/                 rodam sem GPU e sem rede
├── docs/                  plano, registro da revisão, página
├── third_party/Genfocus/  clone de REFERÊNCIA dos autores — nunca modificar
└── geo_cond/              condicionamento geométrico (herdado, desligado por default)
```

`third_party/Genfocus/` é o código dos autores e o treino **importa dele** o
`transformer_forward`. Ele é a fonte da verdade de como a inferência funciona;
qualquer desvio nosso vai em outro lugar, nunca ali.

## Configuração do treino

Batch efetivo **32**, que é o do §4.1 — e se mantém em 32 em qualquer número de
GPUs, porque a acumulação é derivada da contagem real:

| GPUs | `batch_size` | `grad_accum` | efetivo |
|---|---|---|---|
| 4 | 1 | 8 | 32 *(a decomposição literal do paper)* |
| 2 | 1 | 16 | 32 |
| 1 | 1 | 32 | 32 |

```
rank 128 · alpha 128 · 343 módulos LoRA · 463,6 M params treináveis
cond-only (adapters = [texto=None, main=None, cond=LoRA])
guidance 3.5 (main/texto) e 1.0 (condição)
60000 steps · 512² · bf16 · AdamW lr=1e-4 wd=1e-4 · cosine warmup 500
dados: akcit-pixel/DDPD:train (344) + akcit-pixel/RealBokeh:train top-3000 = 3.344 pares
```

Otimizador, lr, scheduler, warmup e resolução **não são do paper** — ele não os
especifica. São escolha nossa, documentada.

## Comandos

### `check` — valida config e credenciais, não treina

```bash
python3 -m genfocus_train.train check --config configs/train_deblur_paper.yaml
```

Roda no login node, sem GPU. Sai com código ≠ 0 se faltar `HF_TOKEN`/
`WANDB_API_KEY` ou se o config for incoerente. É o que se roda antes de gastar fila.

### Smoke — 3 steps no caminho real

```bash
sbatch slurm/smoke.slurm
```

### Treino no SLURM — h100n3, 2 GPUs

```bash
ssh dgx-H100-03 'cd /raid/user_danielpedrozo/projects/julia/retreinar-deblur && sbatch slurm/train_deblur_n3.slurm'
ssh dgx-H100-03 'cd /raid/user_danielpedrozo/projects/julia/retreinar-deblur && tail -n +1 -f logs/deblur-paper-n3-<JOBID>.out'
```

Re-`sbatch` é seguro: `resume: true` retoma do último checkpoint e o id do wandb
é estável, então a curva continua no **mesmo** gráfico. Para resiliência noturna,
encadeie continuações: `sbatch --dependency=afterany:<JOBID> slurm/train_deblur_n3.slurm`.

### Treino no Docker — H100-01, 4 GPUs, sem SLURM

```bash
ssh dgx-H100-01 'cd /raid/user_juliadollis/julia_docker/retreinar-deblur && bash docker/run_docker_h100n1.sh'
docker logs -f deblur_paper_4gpu
```

Aceita `CFG=`, `NAME=`, `GPUS=`, `HF_CACHE=`, `NPROC=` por env var.

### Prefetch — baixa os ~56 GB de dataset sem segurar GPU

```bash
sbatch slurm/prefetch_n3.slurm
```

### Testes — sem GPU, sem rede

```bash
python3 -m pytest tests/ -q
```

## Regras do cluster que este código respeita

De `INSTRUCOES_H100.md` e `DECISOES_FASE2.md` — cada uma custou um incidente:

- **Nunca cancelar job**, nem próprio nem de terceiro. Nenhum script aqui tem `scancel`.
- **Nunca apagar nada** no cluster sem perguntar. Nenhum script tem `rm` ou `--delete`.
- **Nunca baixar fora do SLURM.** O prefetch existe por isso.
- `--time` bem alto (20 dias nos jobs de treino).
- Segredos só do `.env`, nunca hardcoded.
- Docker: `--user $(id -u):$(id -g)` sempre, detached, e **nunca** `rm`/`rmi`/
  `prune`/`stop` de container alheio — há containers de outras 4 pessoas na H100-01.
- `/raid` é **local de cada nó**. h100n3 usa `/raid/user_danielpedrozo/projects/julia/`.
