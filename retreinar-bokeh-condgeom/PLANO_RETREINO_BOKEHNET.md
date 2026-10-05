# Plano de retreino da BokehNet — ordem, pré-condições e critérios de decisão

Data: 2026-09-10
Escopo: **o treino**. A regeração dos dados é o `../bokehnet-regen/`, e a ordem
de execução dela está em `../bokehnet-regen/PLANO_EXECUCAO.md`.

Este documento existe para que os critérios de decisão sejam fixados **antes** de
olhar os resultados. Todo braço de experimento abaixo tem o seu critério escrito,
com o número, e a lista do que é bloqueante.

---

## 0. A aposta, em uma frase

O gargalo medido não foi arquitetura, nem número de steps, nem aparência: foi o
**rótulo de controle**. Consertar só o `K` da rota B, sem tocar em mais nada,
levou o LVCorr no LF-Bokeh de **+0,4365 para +0,8288**, colando nos pesos
oficiais (+0,8868) [M].

Logo, o retreino aposta em: **contrato de controle correto + instrumento que o
vê durante o treino**. Se essa aposta valer, a controlabilidade volta sem
nenhuma mudança de arquitetura. Se não valer, o probe do T11 diz isso em 10K
steps em vez de em 60K.

---

## 1. Pré-condições — o que bloqueia o quê

| # | pré-condição | quem entrega | bloqueia |
|---|---|---|---|
| **P0** | **DECISÃO: o release é autocontido, ou o dataloader resolve referência?** O `bokehnet-regen` publica uma ÁRVORE DE ARQUIVOS e, por cota, a AIF e a bokeh podem ficar no espelho de origem. O `BokehMetricDataset` lê uma TABELA com os pixels dentro — não lê o release como ele é hoje. Ver L1 da auditoria. | time de dados + treino | **TUDO**. É a pré-condição de todas as outras |
| P1 | Release da **rota C** publicado no contrato `metric_disparity_official_v1` | `bokehnet-regen` (etapas 4 e 5) | fase 2, e o conjunto do probe |
| P2 | **Manifesto de split por cena** publicado junto do release | `bokehnet-regen/src/dataio/split.py` | T6 — sem ele não há validação honesta |
| P3 | Distribuição de `calibration_ssim` do release novo | P1 | T8 — o limiar |
| P4 | Release da **rota A** com K sorteado da distribuição de B/C | `bokehnet-regen` (etapa 7) | fase 1 |
| P5 | Release da **rota B** com a **nossa** DeblurNet | DeblurNet nova + `bokehnet-regen` (etapa 6) | fase 2 completa |
| P6 | **PointLight-1K** + BokehMe com kernel de forma (Eq. 6) | `bokehnet-regen` | fase 3 (§3.3) |
| P7 | Conjunto do probe (`scripts/build_probe_set.py`) | P1 + P2 | o probe do T11 |

**A rota C é a mais crítica e é a que menos depende de nós.** Ela não precisa da
DeblurNet, tem pares reais dos dois lados, e é contra ela que as outras duas vão
ser comparadas — prioridade já declarada no `PLANO_EXECUCAO.md`.

Uma pré-condição que **não** é sobre dado e vale registrar: a razão
**~40 pares por imagem** do sintético (1,7K imagens → ~70K pares, supplement B.2
+ §4.1) [I]. 70K pares vindos de 20K imagens com 3 combinações cada é um dataset
diferente, com muito menos sinal sobre "a mesma cena com K diferente" — que é
justamente o que ensina controle. É T26 na auditoria e é pré-condição da fase 1.

---

## 2. Ordem de execução

```
  P1 (rota C)  ──┬──►  [0] SMOKE                    ~1 h, 1 GPU
                 │
  P7 (probe)  ───┤
                 │
  P4 (rota A)  ──┴──►  [1] FASE 1 — 40K sintético   ~4-6 dias, 4 GPUs
                              │
                              ▼
  P5 (rota B)  ───────►  [2] FASE 2 — 60K real      ~6-9 dias, 4 GPUs
                              │   braço primário (literal do §4.1)
                              │
                              ├──►  [2b] braço B: lr 5e-5      10K, comparativo
                              ├──►  [2c] braço C: replay 15%   10K, comparativo
                              │
                              ▼
                         [3] AVALIAÇÃO  (LF-Bokeh, RealBokeh, RealDOF)
                              │
  P6 (PointLight) ──────►  [4] FASE 3 — §3.3 forma  ~1-2 dias
```

Os braços 2b e 2c rodam **a partir do mesmo LoRA da fase 1**, por 10K steps cada,
só para a decisão. O braço vencedor completa os 60K.

---

## 3. Etapa 0 — SMOKE. Bloqueia tudo.

```bash
grep -rn PREENCHER configs/                    # tem que voltar VAZIO
python -m genfocus_train.train check --config configs/train_bokeh_fase1_synth.yaml --stage bokeh
sbatch slurm/smoke_bokeh.slurm
```

O que o smoke tem que mostrar, e cada linha corresponde a um achado:

```
[lora] 343 módulos | rank=64 alpha=64 | ...M params treináveis | variante=cond-only
[dados/<release>] entrada=N
[dados/<release>]   -n (x%) k_censored
[dados/<release>]   -n (x%) scene_level_cap
[dados/<release>] saida=M
[train] stage=bokeh ... guidance=1.0 scale_mode=short_side sigma_mu=full_image
[train] step=1 ... bokeh/defocus_mean=... bokeh/k_efetivo=... bokeh/k_efetivo_std=...
```

**Critérios de aprovação do smoke, declarados agora:**

| checagem | reprova se |
|---|---|
| o dataloader **lê o release** (L1) | levanta o diagnóstico de referência/árvore-de-arquivos |
| o probe devolve **número**, não `{}` | vier vazio — ele engole exceção de propósito, então "não quebrou" NÃO é "funcionou" (L2) |
| contagem de módulos LoRA | ≠ 343 |
| `k_efetivo_std` ao longo de ~20 batches | for 0 (K constante = defeito D2 de volta) |
| `defocus_frac_saturado` mediano | > 0,05 (mapa saturando = `max_coc` errado para a fonte) |
| `defocus_max` mediano | ≈ 1,0 em **todas** as amostras (assinatura do D1) |
| histograma de descarte | um único motivo comendo > 50% do release |
| probe de LVCorr num checkpoint aleatório | não roda / estoura VRAM |

O último é importante: rodar o probe **uma vez** no smoke, mesmo que o número
saia sem sentido (pesos aleatórios), prova que o caminho de geração funciona e
mede o custo real dele — que hoje é [A] (~5 min estimados, nunca medidos).

---

## 4. Etapa 1 — Fase 1, 40K sintético

```bash
sbatch slurm/train_bokeh_fase1_4gpu.slurm
```

Paper §4.1, literal: *"(i) 40K steps on synthetic data"*. Só a rota A.

**O que acompanhar, em ordem de importância:**

1. `bokeh/lvcorr` — a referência é **+0,9059**, que foi o que a fase 1 anterior
   atingiu. Se a fase 1 nova não chegar perto disso, o problema está no dado
   sintético novo e não adianta ir para a fase 2.
2. `bokeh/k_efetivo_std` — tem que ser > 0 e estável.
3. `bokeh/val_loss` — a validação determinística, que escolhe o `best.pt`.
4. `bokeh/loss_ema`.

**Critério de saída:** LVCorr ≥ +0,85 no fim da fase 1. Abaixo disso, **não
prosseguir** — a fase 2 só piora controle, nunca melhora, e começar de uma fase 1
ruim é gastar 9 dias para confirmar isso.

---

## 5. Etapa 2 — Fase 2, 60K real. O trecho que quebrou.

```bash
sbatch slurm/train_bokeh_fase2_4gpu.slurm      # braço primário
```

**A referência do que deu errado**, para não haver dúvida sobre o que se está
procurando:

| LVCorr | LF-Bokeh | RealBokeh | RealDOF |
|---|---|---|---|
| fim da fase 1 | +0,9059 | +0,8609 | +0,9247 |
| fim da fase 2 (a+b+c) | **+0,4365** | +0,8257 | **+0,1324** |
| pesos oficiais | +0,8868 | +0,8498 | +0,9644 |

E a degradação foi **monótona**: na variante só-rota-c, no RealDOF, +0,2509 aos
10K, +0,0281 aos 20K, −0,1776 aos 30K, −0,2061 aos 60K [M]. **Com a loss caindo
o tempo todo.**

**Critério de parada, declarado ANTES de rodar:** o código emite ALARME quando o
LVCorr cai mais de 0,10 abaixo do melhor do run. Em **dois probes consecutivos
com alarme**, o run para e a causa é investigada. Não se deixa rodar até o fim
"para ver no que dá" — já se viu.

### Braços comparativos (10K steps cada, a partir do LoRA da fase 1)

| braço | mudança | hipótese | critério de decisão |
|---|---|---|---|
| **A** (primário) | nenhuma — literal do §4.1 | — | é a referência |
| **B** `..._lrbaixo.yaml` | `lr 5e-5`, warmup 2000 | o salto de 10× no LR (fase 1 termina em 1e-5, fase 2 sobe a 1e-4) reescreve o LoRA de controle antes de a fase 2 consolidar algo [I] | fica o de maior LVCorr aos 10K, **desde que** não perca > 5% de LPIPS na validação |
| **C** `..._replay.yaml` | 15% de batches da rota A | o sintético é a única fonte com variação densa de K por cena; 60K steps sem nenhum exemplo de "mesma cena, K diferente" é esquecimento catastrófico clássico [I] | mesmo critério. **É desvio declarado do §4.1** e vai escrito no texto |

Custo dos três braços: 3 × 10K steps ≈ 3-4 dias de 4 GPUs. É caro, e é menos
caro que descobrir aos 60K.

**Ordem recomendada se o orçamento não der para os três:** rodar A e C. O braço B
é uma hipótese sobre otimização; o C testa diretamente a explicação mais provável
da degradação medida, e o resultado dele vale para o texto do artigo de
qualquer forma — confirmando o paper ou contrariando-o com medida.

---

## 6. Etapa 3 — Avaliação

Fora do escopo desta pasta (é o `../deblurnet-eval-pipeline/` e o
`../genrefocus_deblurnet_paper/vision-pipeline/`), mas duas ressalvas que
**vieram desta auditoria** e afetam como os números vão ser lidos:

1. **A convenção do harness de avaliação.** Está registrado que, na primeira
   rodada, os três modelos — **os pesos oficiais incluídos** — escolheram
   `best_k` no **piso** da faixa. Que os pesos *deles* saturem sugere que a
   convenção do harness (`MAX_COC=100`, disparidade em 1/m, `long_side=512` em
   vez de 0) não bate com a de treino deles, e o candidato óbvio é a **resolução**
   — porque CoC em pixel escala com ela. É o mesmo defeito de escala do T2
   aparecendo pela terceira vez, agora na avaliação. **Conferir antes de
   publicar qualquer tabela de controlabilidade.**

2. **A ablação da Tab. 6 não se reproduziu.** Hoje a+c ganha de a+b+c nas quatro
   mesas, o contrário do paper [M]. A hipótese é que a rota B atrapalhava por
   causa do K constante, e a correção testa isso diretamente. Se depois de
   corrigida a rota B continuar atrapalhando, **é achado nosso** e vai para o
   texto — não é falha de reprodução.

---

## 7. Etapa 4 — Fase 3, §3.3 (forma de abertura)

```bash
sbatch slurm/train_bokeh_fase3_shape.slurm    # exige --init-lora da fase 2
```

O lado de treino está pronto (segundo adapter, base congelado, terceira
condição). O que falta é dado, e é P6.

**O que o paper exige e não pode ser contornado no treino:** *"when an
all-in-focus image lacks point-light stimuli, the simulated bokeh carries weak
shape cues, making the model reluctant to learn shape-conditioned responses."*
Ou seja, treinar forma em imagens comuns não funciona — o PointLight-1K não é
conveniência, é requisito.

O paper não publica o número de steps desta fase, o rank do LoRA de forma, o
vocabulário de formas nem a resolução do kernel. Os nossos (10K, rank 64) são
escolha declarada.

---

## 8. Riscos conhecidos e o que fazer com cada um

| risco | probabilidade | mitigação |
|---|---|---|
| A hipótese da rota A do `PLANO_EXECUCAO.md` falhar (o K muda entre a DeblurNet oficial e a nossa) e a fase 1 precisar ser refeita | média | o teste pareado de ~200 amostras, já planejado, roda **antes** de regerar a rota B inteira |
| `sigma_mu_source: full_image` mudar o comportamento mais do que se espera | média | é a única mudança de eixo C4 no default; o smoke mostra a distribuição de σ. Reverter é uma linha |
| O probe custar muito mais que os ~5 min estimados | média | medido no smoke (etapa 0). Se custar > 15 min, subir `probe_every_steps` para 10000 |
| `max_levels_per_scene: 4` cortar demais o release | baixa | o histograma de descarte diz exatamente quanto, no início do run |
| A fase 2 degradar controle mesmo com o contrato correto | **é a pergunta do experimento** | os braços B e C existem para isso; o alarme do probe evita gastar 60K steps para descobrir |

---

## 9. O que continua sem resposta

- **O limiar de SSIM da Eq. 5** (T8). Bloqueado por P3.
- **A proporção entre rotas B e C** (T10). O paper não especifica; a nossa é
  proporcional por ora, e isso é acidente declarado como tal.
- **`short_side` × `native`** (T14). Trade-off real, a medir.
- **A ressalva geral sobre "100%":** ninguém fora do grupo dos autores reproduz o
  paper bit a bit hoje — o repositório oficial não publicou código de treino,
  dados, o benchmark LF-Bokeh, limites de K, limiar de SSIM, configuração do
  renderer, resolução de treino, otimizador nem scheduler. O objetivo possível é
  uma reprodução pública, matematicamente coerente e auditável, com **cada desvio
  declarado** — não identidade com o treino privado.
