# akcit-consolidado

Código consolidado dos projetos AKCIT-PIXEL.

| Pasta | O que é |
|---|---|
| [`retreinar-deblur/`](retreinar-deblur/) | Retreino da **DeblurNet** (Stage 1 do GenRefocus, arXiv 2512.16923v3): LoRA rank 128 cond-only sobre FLUX.1-dev, 60.000 steps, batch efetivo 32. Inclui treino, configs, jobs SLURM/Docker, testes e a reprodução da Tabela 2. |

Os pesos ficam no Hugging Face, nunca aqui. Datasets: `akcit-pixel/DDPD:train`
(344) + `akcit-pixel/RealBokeh:train` top-3000 por variância do Laplaciano
= 3.344 pares.
