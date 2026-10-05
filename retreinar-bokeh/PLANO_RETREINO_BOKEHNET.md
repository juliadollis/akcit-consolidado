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

---

## 10. O currículo da fase 2 — o que o paper diz, o que ele cala, e o que decidimos

Registrado em 2026-09-25, depois de ir ao texto em vez de responder de memória.
Evidência por `grep` no `bokehnet-regen/reference/paper.txt` (inclui o supplement).

### 10.1 — Learning rate: o paper NÃO MENCIONA. Zero vezes.

```
grep -icE "learning rate|optimizer|adamw|scheduler|warmup|cosine"  ->  0
```

O §4.1 inteiro sobre treino é isto, literal:

> *"Our backbone is FLUX-1-dev [58], fine-tuned via LoRA [25] with a conditioning
> scheme following [62, 63]. DeblurNet employs LoRA rank r=128, while BokehNet
> uses r=64. Both models are trained with a per-GPU batch size of 1 and gradient
> accumulation over 8 steps on 4× RTX A6000 GPUs. DeblurNet is trained for 60K
> steps. BokehNet is trained in two stages: (i) 40K steps on synthetic data, and
> (ii) 60K steps on real data."*

Backbone, rank, batch, acumulação, GPUs e steps. **Nada** de otimizador, LR ou
scheduler.

**DECISÃO NOSSA — optimizer e scheduler FRESCOS na fase 2**, com novo warmup e
novo cosseno sobre os 60K steps. Base textual: o §3.2(a) chama a fase sintética
de *"We **pretrain** with synthetic data"*, e pré-treino seguido de treino não é
continuação.

**A RESSALVA, que é o risco desta decisão:** a fase 1 termina em `1e-5`
(`min_lr_ratio 0,1` × `1e-4`) e a fase 2 sobe de volta a `1e-4` — **um salto de
10×** exatamente quando entram os dados reais. É candidato direto ao
esquecimento catastrófico do controle de K medido na rodada anterior
(LVCorr +0,9059 → +0,4365, **monótono**). Por isso `train_bokeh_fase2_lrbaixo.yaml`
existe: `lr 5e-5`, warmup 2.000. **Medir, não supor.**

### 10.2 — B e C são SIMULTÂNEAS, não sequenciais. Isto tem evidência.

Não é silêncio do paper: é afirmação dele. §4.1, literal:

> *"we utilize a **hybrid dataset** comprising ∼70K synthetic pairs derived from
> [27, 80], **combined with** approximately 26K real examples sourced from ITW
> dataset [19], RealBokeh [57], and LFDOF [52]."*

"hybrid dataset", "combined with", três fontes numa lista. Sem ordem.

E a **Tab. 6** confirma pela forma: ela varia **quais datasets entram**, não em
que sequência.

| (a) sintético | (b) ITW | (c) LFDOF+RealBokeh | LPIPS ↓ |
|---|---|---|---|
| ✓ | – | – | 0,1289 |
| ✓ | ✓ | – | 0,1156 |
| ✓ | – | ✓ | 0,0972 |
| ✓ | ✓ | ✓ | **0,0833** |

São marcações de **inclusão**. Currículo sequencial teria ordens, não checkboxes.

**DECISÃO — a fase 2 carrega B e C JUNTAS**, como um `datasets` de duas fontes.
É o que `train_bokeh_fase2_real.yaml` já faz. **Não existe "fase B depois fase C".**

### 10.3 — A proporção entre B e C: aberto, e não pode ficar por acaso

O paper **não** publica a proporção. Hoje o nosso `route_weights` é `null`, que
significa **proporcional ao tamanho dos shards** — isto é, acidente.

Por que importa, medido: B e C ocupam faixas de defocus diferentes **por
física** — a rota B em `[0, 0,05]` e a C em `[0, 0,4]`. E a Tab. 6 mostra
**(a)+(c) melhor que (a)+(b)**: 0,0972 contra 0,1156. A rota C carrega mais
sinal de controle; a B carrega aparência óptica real com AIF vinda da DeblurNet,
que é a distribuição da inferência.

Deixar a proporção ser o tamanho relativo dos arquivos é deixar o experimento
ser decidido por quantos JPEGs cada release tem.

### 10.4 — Os erros antigos que cada guarda existe para impedir

Esta seção é o motivo de as guardas serem chatas. Cada uma custou uma rodada.

| erro | o que produziu | a guarda hoje |
|---|---|---|
| `max_coc` por rota (o `kfix` usou 10,5107) | duas convenções no MESMO lote: LF-Bokeh subiu e RealBokeh/RealDOF caíram | `MAX_COC` é constante de módulo, **sem parâmetro**; o YAML só pode ASSERTAR 100,0 |
| `k = 50,0` em 11.635 de 11.635 amostras | rota B inteira sem variação de K | **sem fallback numérico**: falta de dado levanta erro com motivo nomeado |
| mapa normalizado por imagem (`dm/dm.max()`) | o K apagado algebricamente; `max(defocus)==1` em TODAS as amostras | log de `defocus_max`/`k_efetivo_std` por batch; `defocus_source` aposentado exige porta explícita |
| `except: pass` trocando Depth Pro por Depth Anything | profundidade virava disparidade, amostra espelhada **sem rastro** | `depth_backend` obrigatório e conferido |
| K na escala da imagem original, crop em 512 | erro de 0,34× a 0,89×, **variando por amostra** | `k_at_resolution`/`k_by_scale` obrigatórias, travadas por teste pós-crop |
| checar só a EXISTÊNCIA da coluna `calibration_ssim` | descartaria as 11.635 amostras da rota B, que tem a coluna NULA | filtro **por amostra**, e rota sem sweep passa inteira |
| ledger indexado só por `sample_id` | a rota A (por `scene_id`) ficava **sem conferência de sha256** | indexa pelas DUAS chaves |
| split por linha | a mesma cena nos dois lados; validação media memorização | split por **cena**, materializado no release |
| `best.pt` pela loss de treino de um micro-batch | rótulo "melhor" que é sorteio | validação determinística — **e nesta run ela quebrou, então o artefato publicado é o step 40.000, não um "best"** |
| probe dentro do treino | pausa de 5 min no rank 0 → watchdog do NCCL mata o run | **monitor externo**, em GPU e container separados |
| paralelizar download em 7 processos | estourou a quota de 2.500 req/5min; 6 morreram com `429` | um processo, baixando pelos caminhos do `manifest.jsonl` |

### 10.5 — A regra que resume tudo

Toda escolha vai numa de três caixas, e a caixa fica escrita:

1. **o paper DIZ** — literal, com citação;
2. **o paper CALA e a inferência oficial decide** — `Inference_bokehNet.py` é a autoridade;
3. **os dois calam** — decisão nossa, **declarada**, com o motivo e o custo.

O erro que este projeto mais repetiu não foi de conta: foi **apresentar (3) como
(1)**. O teto por cena rotulado "literal do supplement B.2" era decisão nossa; o
`sigma_mu_source` é decisão nossa; a proporção entre rotas é decisão nossa. Todas
defensáveis — nenhuma do paper.


---

## 11. A fase 2 — os dados chegaram, e o que eles obrigam a decidir (2026-09-25)

| | rota B | rota C |
|---|---|---|
| repo | `juliadollis/bokehnet-regen-rota-b-nossa` | `juliadollis/bokehnet-regen-rota-c` |
| amostras · cenas | 13.765 · **13.765** | 15.423 · **4.399** |
| origem | `atfortes/BokehDiffusion` | `akcit-pixel/RealBokeh` |
| AIF | **gerada** pela DeblurNet, em `generated/` | referência (`image_focus`) |
| bokeh (alvo) | referência (`repo#train[i].image`) | referência (`image_blur`) |
| K | **Eq. 3** da EXIF | **Eq. 5**, sweep de SSIM |
| `calibration_ssim` | presente e **NULA** | preenchida em 15.423/15.423 |

Total real: **29.188**. O §4.1 fala em ~26K. As duas entram **juntas** (§10.2).

### 11.1 — VAZAMENTO: a rota C treina no split `test` da RealBokeh

Todo meta da rota C traz `source_split: "test"`. Os benches
`juliadollis/bokeh-bench-realbokeh-test` e `-v2` saem do **mesmo** split.
Medido:

```
realbokeh-test-v2   cenas no bench=217  em comum=217  amostras da rota C=804 (5,2%)
realbokeh-test      cenas no bench=220  em comum=220  amostras da rota C=814 (5,3%)
UNIÃO: 220 cenas, 814/15.423 amostras = 5,3%
```

**220 de 220.** Não é sobreposição parcial: é o bench inteiro dentro do treino,
com 162 `file_name_base` batendo exatamente. Treinar assim e avaliar depois não
dá um número otimista — dá um número que mede memorização, e que **sobe quando o
modelo piora**, porque decorar é mais fácil que generalizar.

**Decisão nossa:** excluir as 220 cenas. Custa 5,3% da rota C (sobram 14.609
amostras de 4.179 cenas) e preserva os dois benches. A lista é gerada por
`scripts/cenas_de_avaliacao.py`, fica versionada em
`releases/rota_c_cenas_do_bench.json`, e o dataloader a consome por
`excluir_cenas_de_avaliacao` — nada é inferido em tempo de treino. Amostra
descartada por ela entra no histograma como `cena_de_avaliacao`.

Não é o paper que manda: é higiene de avaliação, e o paper não tinha este
problema porque o LF-Bokeh dele nunca foi publicado.

### 11.2 — `min_calibration_ssim`: medido, não chutado

Distribuição real da rota C (15.423 amostras, todas preenchidas):

| p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|
| 0,771 | 0,917 | 0,963 | 0,986 | 0,997 |

| limiar | mantém |
|---|---|
| 0,6 | 99,5% |
| **0,7** | **97,7%** |
| 0,8 | 93,2% |

**Decisão nossa: 0,7.** Corta a cauda que o sweep da Eq. 5 não conseguiu ajustar
sem jogar fora dado bom. O 0,6 anterior tinha sido calibrado na distribuição
ANTIGA — é o tipo de constante que sobrevive a uma regeração sem ninguém notar.
O paper não publica valor: o §3.2(c) diz que o limiar existe, não qual é.

A rota B tem a coluna **presente e nula** e não pode ser cortada por ele — ela
deriva o K da Eq. 3, não de sweep. `tests/test_rotas_b_c.py` trava isso.

### 11.3 — `route_weights`: 50/50 explícito

Fecha o §10.3. Sem peso, a proporção seria 13.765 contra 14.609 — 48,5/51,5, que
por acaso é quase equilibrado, mas por **acaso**. A Tab. 6 mostra que a
composição muda o resultado ((a)+(c) = 0,0972 contra (a)+(b) = 0,1156), então o
número tem de ser escolhido. 50/50, declarado.

### 11.4 — Divergências declaradas em relação ao §4.1

| o que o paper diz | o que temos |
|---|---|
| ~26K reais de **ITW + RealBokeh + LFDOF** | 29.188 de **BokehDiffusion + RealBokeh**. Sem LFDOF. |
| §3.2(b) usa o ITW | usamos `atfortes/BokehDiffusion` |

### 11.5 — Falso alarme registrado: `channel_order: "bgr"`

Todo meta das três rotas traz `channel_order: "bgr"`. Medido antes de mexer em
nada, comparando a AIF do espelho com o bokeh do release da rota A: diferença de
médias **1 a 46 direto** contra **66 a 79 trocando R↔B**. Os bytes em disco são
**RGB**; o campo descreve o espaço interno do pipeline de geração. E a fase 1
treinou com a mesma convenção, então não há divergência entre as fases.

Fica registrado porque "tem um campo dizendo BGR" é exatamente o tipo de coisa
que, daqui a três meses, vira uma correção apressada que quebra tudo.

### 11.6 — O loader, exercitado no dado real da rota B

Contra as 2.056 amostras que já tinham baixado (`scripts/valida_rota_parcial.py`):

| medida | valor | leitura |
|---|---|---|
| amostras carregadas | 1.953 (−103 pelo split por cena) | o caminho inteiro roda |
| pixels resolvidos | `generated=8 · espelho=8` | AIF do release **e** bokeh do espelho |
| `k_efetivo` p5/p50/p95 | 1,59 · **11,08** · 29,48 | **48 distintos em 48** |
| `defocus_max` p50 | **0,0277** | longe de 1,0 — sem a assinatura do D1 |
| fração saturada | **0,0** | `max_coc` certo para a fonte |
| shapes | todas `(3, 512, 512)` | |
| **alertas** | **nenhum** | |

É a primeira vez que a rota B passa pelo dataloader com dado de verdade. Os
testes sintéticos de `test_rotas_b_c.py` previram a forma certa.

### 11.7 — O espelho da RealBokeh estava 94,7% incompleto

Descoberto ao pré-construir o índice do espelho (`IndiceEspelho`) antes de subir
o treino, justamente para os 4 ranks não o construírem em paralelo:

```
índice local: 1.257 linhas, shards = data/test-0000{0..5}-of-00006.parquet
rota C:      15.423 amostras, 15.423 source_sample_id distintos
NÃO encontrados no espelho: 14.609 de 15.423  →  cobertura 5,3%
```

O snapshot local tinha **só o split `test`**. A rota C sai de três splits:

| split de origem | amostras | espelho |
|---|---|---|
| `train` | 13.796 | **faltando** — 85 shards, ~42 GB |
| `validation` | 813 | **faltando** — 5 shards, ~2,4 GB |
| `test` | 814 | já em disco |

Sem isso a rota C levanta `KeyError` do índice em 94,7% das amostras.

**CORREÇÃO (escrevi errado antes):** eu disse que isso apareceria "no meio do
treino, na primeira amostra que pedir um nome ausente". Não é verdade —
`BokehReleaseTreeDataset.preflight` existe justamente para isso e pegou o
`KeyError` no arranque. O que é verdade, e é mais fino: o `preflight_de_pixels`
resolve **8 amostras espalhadas**, então pega um espelho ausente ou quase todo
ausente, e **não** pega um buraco esparso — 5% de nomes faltando passaria pelas
oito e explodiria lá no step 8.000. Fechado no §11.11.

**Coincidência que vale anotar:** as 814 amostras que já estavam em disco são
exatamente as que o §11.1 manda EXCLUIR por vazamento. Se ninguém tivesse
conferido a cobertura, a rota C teria entrado no treino com ~0 amostras úteis —
tudo o que resolvia era o que tem de sair.

**A lição de processo:** construir o índice do espelho é caro e por isso parecia
coisa de arranque. Foi ANTECIPÁ-LO que revelou o buraco. Um artefato derivado
que ninguém materializa antes da hora é um lugar onde defeito de dado se
esconde.

### 11.8 — Escrita atômica do índice do espelho

Ainda no mesmo caminho: `IndiceEspelho._construir` gravava o cache com
`write_text`, que trunca e depois escreve. Em 4 ranks os quatro constroem o
índice ao mesmo tempo e gravam o MESMO arquivo — um lendo no meio pega JSON
cortado, dois escrevendo deixam o cache corrompido de forma permanente. Defeito
intermitente, que aparece no arranque de um run de 60K steps e some quando se
tenta reproduzir.

Agora é temporário + `os.replace` (atômico no mesmo sistema de arquivos), e a
leitura de um cache ilegível **reconstrói** em vez de seguir com índice pela
metade — um índice incompleto aponta amostras para a linha errada do espelho, e
a imagem carrega normalmente, que é o pior tipo de defeito possível aqui.
Travado por `tests/test_release_tree.py`.

### 11.9 — BLOQUEIO: a org `akcit-pixel` estourou a cota de armazenamento do HF

Com o release da rota C completo em disco (15.423 amostras, 30.852 arquivos, 0
ausentes), o **espelho** não pode ser baixado:

```
403 Forbidden: Private repository storage limit reached for akcit-pixel,
please upgrade your plan to restore full access to your private repositories
or free up space by deleting some repositories.
Cannot access content at:
  .../datasets/akcit-pixel/RealBokeh/resolve/main/data/train-00000-of-00085.parquet
```

O bloqueio é nos arquivos **LFS**; metadados pequenos ainda respondem, o que
fazia o erro chegar disfarçado de `LocalEntryNotFoundError` ("check your
connection"). Testado arquivo a arquivo:

| shards | acesso |
|---|---|
| `data/train-*` (85) | **403** |
| `data/validation-*` (5) | **403** |
| `data/test-*` (6) | OK — mas **só porque já estão em cache local** |

E os 814 do `test` são exatamente os que o §11.1 manda excluir por vazamento.
Ou seja: **a rota C tem 0 amostras utilizáveis hoje**.

Procurado antes de concluir:

| onde | resultado |
|---|---|
| outros caches do `/raid` (20 usuários) | nenhum `train-000*-of-00085.parquet` |
| upstream `timseizinger/RealBokeh_3MP` | **público**, 20,9 GB, 31.853 JPEGs — mas a akcit-pixel **realinhou** as imagens (`..._aligned`), então os `aif_sha256`/`bokeh_sha256` do ledger não bateriam. Reconstruir de lá seria trocar o dado calado. |

Nota colateral: `akcit-pixel/DDPD` (a fonte da DeblurNet) está no mesmo bloqueio.
`akcit-pixel/RealDOF` é público e segue acessível.

**Isto não se resolve do lado do código.** É liberar espaço ou subir o plano da
org no HF — decisão de quem administra a conta, e nada aqui deve tentar
contornar apagando repositório de ninguém.

### 11.10 — O caminho enquanto isso: `train_bokeh_fase2_somente_b.yaml`

Fase 2 com a **rota B apenas** — 13.106 amostras de treino, validadas ponta a
ponta contra o dado real (§11.6). É **desvio declarado** do §4.1, que pede o
dataset híbrido; não é a reprodução, é o que dá para rodar hoje.

Quando o espelho voltar, a fase 2 completa é `train_bokeh_fase2_bc.yaml` **em
run novo**. Não se acrescenta rota no meio de um run: o princípio desta árvore é
que o treino começa e termina do mesmo jeito, e comparar duas fases 2 exige que
cada uma tenha rodado inteira com a sua composição.

### 11.11 — `conferir_cobertura_do_espelho`: cobertura antes de pixel

O `preflight` agora faz duas checagens, de custos deliberadamente diferentes:

| checagem | custo | pergunta que responde |
|---|---|---|
| **cobertura**, sobre TODOS os nomes | operação de conjunto contra um índice já em memória; nenhum pixel decodificado | "o espelho cobre este release?" |
| **resolução**, em 8 amostras espalhadas | I/O + decode | "os bytes saem de verdade?" |

A primeira é nova, e é a que faltava: o buraco de 94,7% da RealBokeh só apareceu
depois de uma tentativa de baixar 44 GB, quando uma comparação de conjuntos
respondia em um segundo. Ela vale para a forma `coluna` (rota C), a única que
passa pelo índice — na `linha_hf` (rota B) e na `caminho` (rota A) a ausência já
falha nomeada. Travada por `tests/test_rotas_b_c.py`.

### 11.12 — A rota C, exercitada no dado real

Não dá para treinar nela hoje (§11.9), mas dá para PROVAR o código: as 220 cenas
do bench são as únicas com espelho em cache, e elas caem na partição `val` do
release. Rodado contra elas:

| medida | valor | leitura |
|---|---|---|
| amostras | 758 (de 814; −42 `invalid_for_control`, −14 pelo SSIM 0,7) | filtros do contrato agindo |
| cobertura do espelho | **758/758** | a checagem nova passa |
| pixels | `espelho=16` | **os dois lados** vêm do espelho, via índice shard/linha |
| `k_efetivo` p5/p50/p95 | 0,62 · **3,74** · 12,37 | 46 distintos em 48 |
| `defocus_max` p50 | **0,038** | longe de 1,0 |
| fração saturada | **0,0** | |
| **alertas** | **nenhum** | |

Com isso **as duas rotas têm o caminho de código verificado contra dado real**, e
não só contra fixturas sintéticas. O que falta na rota C é dado, não código.

### 11.13 — Um confundidor a declarar: μ difere entre as rotas

Medido acima, com `sigma_mu_source: full_image`:

| rota | resolução típica | `full_seq_len` |
|---|---|---|
| B (BokehDiffusion) | 1024×683 | 1.700 – 3.712 |
| C (RealBokeh 3MP) | 2000×1500 | **11.750**, constante |

O μ do rectified flow sai de `image_seq_len`, então as duas rotas recebem
distribuições de σ sistematicamente diferentes — e a diferença está
**correlacionada com a rota**, não distribuída dentro dela. Não é defeito: é o
que a inferência faz (μ vem do tamanho real da imagem). Mas é confundidor, e
tem de estar escrito antes de alguém atribuir um ganho da fase 2 à composição do
dado quando pode ser o regime de ruído.

O braço `sigma_mu_source: crop` existe justamente para separar isso, e não roda
nesta rodada — ver §8 do README.

---

## 12. O smoke da fase 2 (2026-09-25) — o que ele provou e o bug que achou

Rodado nas GPUs 0 e 1 enquanto o espelho da rota C segue bloqueado, para o
caminho de arranque não ser exercitado pela primeira vez **depois** dos 44 GB de
download. Config `_smoke_fase2.yaml` (rota B), 2 processos.

### 12.1 — O que passou

| item | evidência |
|---|---|
| `--init-lora` da fase 1 | `[ckpt] 686 tensores LoRA carregados de .../step_40000.pt` + `[train] fase 2: LoRA inicializado da fase 1` |
| LoRA reinjetado igual | `343 módulos · rank=64 · 231.8M treináveis · cond-only` |
| FLUX por snapshot | `[flux] snapshot local: ...` (a guarda R3 funcionando) |
| `--grad-accum` sobrepõe o YAML | `gradient_accumulation_steps: 8 (YAML) -> 2` |
| **validação de verdade** | `val=0,3094 → 0,2805` ao longo de 4.150 steps, caindo |
| **retomada** | 2ª rodada abriu com `start=1764` e **sem** recarregar o init-lora — a condição `global_step == 0` está certa |
| salvamento em exceção | o `finally` gravou `step_1764.pt` **e** o safetensors antes de morrer |
| `/dev/shm` | 4.150 steps sem `Bus error`, com `--shm-size=64g --ipc=host` |

### 12.2 — O bug: `0 or -1` é `-1`

Depois de 1.764 steps o run morreu com

```
ReleaseError: amostra 'b_20002539892': `bokeh_ref` pede a linha 0 de
'atfortes/BokehDiffusion#train', que tem 15305 linhas. Snapshot de outra
revisão — a linha não é o mesmo pixel.
```

A mensagem se contradiz — linha 0 de 15.305 é válida. A guarda era

```python
0 <= int(ref.linha or -1) < len(ds)
```

e em Python **`0 or -1` avalia para `-1`**, porque zero é falsy. Ou seja: a única
amostra do release que endereça a **linha 0** do espelho era rejeitada, com uma
mensagem que acusava o dado em vez do código. Corrigido para
`ref.linha is not None and 0 <= int(ref.linha) < len(ds)`; verificado no dado
real (`b_20002539892`: `aif` de `generated`, `bokeh` do espelho, 1024×574).

**Por que nenhum teste pegou:** as fixturas sintéticas usam `[1]`, `[776]` e
afins — a linha 0 nunca aparecia. E em produção é **1 amostra em 13.765**, ou
seja 0,007%: sorteio bastou para levar 1.764 steps até topar nela. É o tipo
exato de defeito que só o dado real encontra, e a razão de o smoke existir.

Travado por `tests/test_rotas_b_c.py::test_rota_b_aceita_a_linha_zero`, que
exercita a guarda antiga e a nova lado a lado.

### 12.3 — A cota da org no HF, em números

O bloqueio do §11.9 não é de arquivo faltando: é **cota de armazenamento privado
da org `akcit-pixel`**. Medido pela API:

| repo privado | tamanho |
|---|---|
| `akcit-pixel/RealBokeh` | 46,9 GB |
| `akcit-pixel/LFDOF` | 36,9 GB |
| `akcit-pixel/DDPD` | 9,6 GB |
| **total privado** | **93,4 GB** |

Enquanto está acima do limite, o HF recusa a leitura dos arquivos **LFS** dos
três — metadado pequeno ainda responde, o que faz o erro chegar disfarçado de
`LocalEntryNotFoundError` ("check your connection").

Saídas, todas na conta do HF e todas decisão de quem administra a org:

| opção | efeito | observação |
|---|---|---|
| **tornar `RealBokeh` público** | libera 46,9 GB de uma vez | é repack realinhado de `timseizinger/RealBokeh_3MP`, que **já é público** — mas a decisão de publicar é do time |
| tornar `LFDOF` público | libera 36,9 GB | idem |
| subir o plano da org | libera tudo sem mexer em repo | custa dinheiro |
| apagar repo | libera | **não fazer** — nada é apagado sem pedido explícito |

Nota: `DDPD` é a fonte da **DeblurNet**. Se o retreino dela ainda precisar
baixar dado, bate na mesma parede.

### 12.4 — A conta fechada: 100,3 GB contra um limite de 100 GB

Inventário completo do armazenamento **privado** da org `akcit-pixel` (datasets
**e** models — os dois contam para a mesma cota):

| tipo | tamanho | repo |
|---|---|---|
| dataset | 46,9 GB | `akcit-pixel/RealBokeh` ← o espelho que a rota C precisa |
| dataset | 36,9 GB | `akcit-pixel/LFDOF` |
| dataset | 9,6 GB | `akcit-pixel/DDPD` ← fonte da DeblurNet |
| model | 6,8 GB | `akcit-pixel/depthpro-spring-ft` |
| **total** | **100,3 GB** | contra o limite de **100 GB** do plano gratuito |

**Estão 0,3 GB acima do limite**, e o HF responde 403 na leitura de LFS de
*todos* os quatro. Não é um problema de escala: é um estouro de 0,3%.

O `depthpro-spring-ft` (6,8 GB) é o menor item e sozinho resolveria — tornar ele
público, ou qualquer um dos quatro, devolve a leitura de todos. A decisão é de
quem administra a org; nada aqui apaga nem publica repo sem pedido explícito.

---

## 13. A fase 2 oficial subiu (2026-09-28) — e os dois tetos no caminho

### 13.1 — Teto 1: cota do HF. Estourada por 0,3 GB.

O inventário privado da org `akcit-pixel` (datasets **e** models contam juntos)
somava **100,3 GB** contra o limite de **100 GB** do plano gratuito. Resolvido
tornando `akcit-pixel/depthpro-spring-ft` (6,8 GB) público — o menor dos quatro.

Um detalhe que atrasou o diagnóstico: o 403 vale só para os arquivos **LFS**.
Metadado responde normalmente, então `repo_info` listava os 98 arquivos e o
download falhava — e o `huggingface_hub` traduzia o 403 em
`LocalEntryNotFoundError: check your connection`, que aponta para a rede. A
resposta crua do servidor é que resolveu:

```
HTTP 403
x-error-message: Private repository storage limit reached for akcit-pixel
```

**E por que "publicar em juliadollis" não era saída:** republicar exige baixar
primeiro, e baixar era o que estava bloqueado. Circular. (A saída sem download
existia — `move_repo`, transferência server-side — mas mudaria a proveniência
gravada no ledger da rota C, que diz `source_dataset: akcit-pixel/RealBokeh`.)

### 13.2 — Teto 2: cota de DISCO do cluster, no mesmo dia

Com o HF liberado, o download morreu em `OSError: [Errno 122] Disk quota
exceeded`. A cota do `/raid` é **por usuário**: 500 GB de aviso, **600 GB de
teto**, e estávamos em 600 GB. Liberados 23 GB, todos artefatos de teste desta
semana (`outputs/_smoke_fase2` 17 GB, `_verificacao_codigo`/`_verif3`/`_diag_b`
5,4 GB, 52.527 blobs `.incomplete`, logs da raiz) — nada de run bom, nada de
terceiros, tudo com autorização explícita.

**Guarda nova:** `snapshot_download` voltou `"PRONTO em 0.0 min"` com **53 de 96
shards** em disco, depois de duas falhas de cota. Ele devolve o caminho do
snapshot, não uma confirmação de que baixou. Agora o script **conta** os
parquets e só declara sucesso com os 96; e detecta `quota`/`No space` para
morrer com a mensagem certa em vez de retentar oito vezes.

### 13.3 — O bug que a primeira subida do oficial encontrou

```
ValueError: `route_weights` não cobre as rotas
['/workspace/releases/rota_b', '/workspace/releases/rota_c']
```

O rótulo de rota vinha de `source.name` — o **caminho** do release — e não do
campo `route` do manifesto. Então `route_weights: {b: 0.5, c: 0.5}`, que é a
forma óbvia de escrever, não casava com nada. A guarda do §10.4 fez o certo:
falhou alto em vez de tratar peso ausente como zero.

Duas consequências, e a segunda é pior que a primeira:

1. o YAML tinha de repetir caminhos absolutos como chave;
2. **dois releases da MESMA rota receberiam rótulos diferentes**, e o peso
   pedido para a rota seria aplicado a cada um por separado — silenciosamente
   dobrando a massa daquela rota.

Corrigido nos dois leitores (árvore e tabela) com `_etiqueta_de_rota`, que lê
`route` e cai no nome da fonte só para release antigo sem a coluna. A mensagem
de erro agora imprime **os rótulos presentes com a contagem** e **as chaves que o
YAML declarou**, em vez de deixar adivinhar.

### 13.4 — O que está rodando

| | |
|---|---|
| `julia_fase2_oficial` | GPUs 0, 1, 2, 5 · `Up` |
| batch efetivo | **32** = 1 × 8 × 4 — e o `sobe_fase2_bc.sh` agora **deriva** o accum de `32 / n_gpus`, porque com `--grad-accum 8` fixo um run em 2 GPUs daria 16 com o banner anunciando o número certo |
| dados | B 13.106 + C 12.925 = **26.031** reais (o §4.1 fala em ~26K) |
| espelho | `cobre 12925/12925 nomes` — a checagem do §11.11 passando em produção |
| prazo | ~4,0 dias |

**Pendente:** o monitor externo de LVCorr, que precisa de uma GPU livre. Sem ele
o treino roda, mas cego para o único modo de falha conhecido desta fase.
