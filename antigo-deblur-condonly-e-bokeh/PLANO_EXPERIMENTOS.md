# Fila de experimentos — GPUs 2, 3 e 5 (dgx-H100-01)

> Objetivo duplo: (a) responder perguntas cientificas que ainda estao abertas,
> (b) manter as 3 GPUs do grupo ocupadas para nao serem realocadas.
> Criado em 2026-08-19, depois de a fase 2 (60K steps) terminar.
>
> Regra desta maquina: NAO ha SLURM. Ocupacao e por container docker nosso,
> sempre com `--user $(id -u):$(id -g)` e nome com prefixo `julia_`.
> NUNCA `docker system prune` / `rmi` / `rm` de container alheio.

## Estado atual

| Item | Situacao |
|---|---|
| DeblurNet (estagio 1) | reproduzido, LPIPS 0.1446 na DDPD test |
| BokehNet fase 1 (40K sintetico) | concluida |
| BokehNet fase 2 (60K real) | **concluida** 2026-08-19, 60000/60000 |
| Avaliacao 3 modelos (20 img, 512px) | inferencia OK, metricas rodando |

## Perguntas em aberto (o que justifica cada experimento)

1. **Quanto a fase 2 realmente somou?** So temos evidencia visual. Falta o
   numero comparando fase-1-only vs fase-2 — e a ablacao da Tab. 6 do paper.
2. **Nosso modelo bate o oficial?** Depende da avaliacao rodando.
3. **A limitacao do K da rota b aparece em numero?** A LVCorr e a metrica que
   expoe isso (ver DECISOES_FASE2.md secao 6).
4. **Os nossos numeros sao comparaveis aos publicados?** Hoje nao: rodamos a
   512px e o paper usa resolucao original com tiling.
5. **O filtro de SSIM (secao 4 do DECISOES) ajudou ou atrapalhou?**

---

## FILA — ordenada por (valor cientifico / custo)

### A. AVALIACOES (baratas, horas)

**A1. Fase-1-only vs fase-2 — a ablacao da Tab. 6**  ~2h/GPU
Roda a mesma avaliacao com o peso da fase 1 (`genrefocus-bokehnet-synth-2gpu:
bokeh.safetensors`). Junto com os 3 ja medidos, da a curva completa:
sem-treino -> fase 1 -> fase 2 -> oficial. **Maior valor da fila**: e a evidencia
quantitativa do que hoje so temos em imagem.

**A2. Checkpoints intermediarios da fase 2**  ~2h cada
Steps 20000 e 40000 (ja estao no HF). Da a curva de aprendizado em metrica, e
mostra se 60K era necessario ou se saturou antes — informacao direta para
decidir o orcamento de treinos futuros.

**A3. RealDOF**  ~2h/modelo
O paper avalia em DPDD **e** RealDOF (Tab. 2). So fizemos DDPD.
`akcit-pixel/RealDOF`, split validation (50 imagens).

**A4. Amostra completa da DDPD**  ~7h/modelo
Hoje sao 20 das 73 imagens do split de validacao. Com 73 o intervalo de
confianca encolhe. Rodar pelo menos no nosso e no oficial.

**A5. Resolucao original (long_side=0) com tiling**  ~10h/modelo
UNICA forma de comparar com os numeros PUBLICADOS do paper. Caro; so vale
depois que A1-A3 fecharem e se a gente quiser reportar contra a tabela deles.

**A6. Estudo de controlabilidade dedicado**  ~3h
Sweep denso de K (10 valores) no nosso modelo e no oficial, medindo a
monotonicidade da variancia do Laplaciano (a Fig. 12 do paper). Quantifica a
consequencia do K travado da rota b melhor que a LVCorr de 4 pontos.

### B. TREINOS (dias — sao os que seguram as GPUs)

**B1. Aperture shape, secao 3.3 do paper**  ~3-5 dias
LoRA EXTRA com o LoRA base CONGELADO, condicao adicional de forma de abertura.
E a unica parte do paper que ainda nao reproduzimos. Precisa do PointLight-1K,
que nao temos: o paper descreve como construir (secao C — keywords do Flickr,
prompts via LLM, FLUX fine-tunado anti-blur, depois passa pela DeblurNet).
**Bloqueio**: montar o dataset primeiro. Vale abrir como tarefa separada.

**B2. Ablacao do filtro de SSIM**  ~5 dias
Refazer a fase 2 com `min_calibration_ssim: null`. Responde se o filtro que
adicionamos (fiel ao paper) ajudou de fato. Caro, mas e a defesa de uma decisao
nossa de metodo.

**B3. Fase 2 com lr menor (5e-5)**  ~5 dias
O knob documentado em DECISOES_FASE2.md secao 9, para o caso de a fase 2 estar
degradando o que a fase 1 aprendeu. So faz sentido se A1 mostrar degradacao em
alguma metrica.

**B4. Fase 2 com o K da rota b corrigido**  ~5 dias
DEPENDE do time de dados devolver a profundidade metrica. Quando vier, este e o
treino de maior impacto: hoje 80% do dado da fase 2 nao ensina controle de
intensidade.

### C. PENDENCIAS ANTIGAS (baratas)

**C1. DeblurNet na RealDOF** — pendencia do handoff de julho, nunca rodada.
**C2. Confirmar a linha `paper-oficial`** no df `juliadollis/deblur-metrics`.

---

## Ordem de execucao proposta

```
GPU 2:  A1 (fase-1-only)  ->  A3 (RealDOF, nosso)   ->  A4 (73 img, nosso)
GPU 3:  A2 (step 20000)   ->  A2 (step 40000)       ->  A3 (RealDOF, oficial)
GPU 5:  A6 (controlabilidade) -> C1 (DeblurNet RealDOF) -> A4 (73 img, oficial)
```

Isso mantem as 3 GPUs ocupadas por ~12-15 h. Depois disso, o que segura as GPUs
por dias e a faixa B, e a escolha entre B1/B2/B3 depende do que A1 e A6
mostrarem.
