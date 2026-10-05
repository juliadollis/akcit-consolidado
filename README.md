# akcit-consolidado

Código consolidado dos projetos AKCIT-PIXEL — reprodução do **GenRefocus**
(*Generative Refocusing: Flexible Defocus Control from a Single Image*,
arXiv 2512.16923v3) sobre FLUX.1-dev + LoRA.

| Pasta | Estágio | O que é |
|---|---|---|
| [`retreinar-deblur/`](retreinar-deblur/) | **Stage 1** | Retreino da **DeblurNet**: LoRA rank 128 cond-only, 1 condição (a imagem borrada), 60.000 steps, batch efetivo 32. Dados: `akcit-pixel/DDPD:train` (344) + `akcit-pixel/RealBokeh:train` top-3000 por variância do Laplaciano = 3.344 pares. Inclui a reprodução da Tabela 2 em `inferencia/`. |
| [`retreinar-bokeh/`](retreinar-bokeh/) | **Stage 2** | Treino da **BokehNet**: LoRA rank 64, 2 condições (AIF + mapa de defocus), currículo de 40.000 steps sintéticos (rota a) → 60.000 reais (rotas b+c). Contrato do sinal de controle em `genfocus_train/control.py`, probe de LVCorr em `probe.py`. |
| [`retreinar-bokeh-condgeom/`](retreinar-bokeh-condgeom/) | **Stage 2** | Variante da BokehNet com **condicionamento geométrico**: os 6 canais de `geocond/` (`u, O, s, n_x, n_y, K~`) entram como dois branches extras de condição, `G1=[s,n_x,n_y]` e `G2=[u,O,K~]`. |

Os pesos ficam no Hugging Face, nunca aqui. Cada árvore é autocontida e tem o
seu próprio `README.md`, `configs/README.md` e registro de mudanças.
