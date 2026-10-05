# retreinar-bokeh — o treino da BokehNet

Árvore de treino da **BokehNet** (GenRefocus, Stage 2). Independente: não altera
`../genrefocus_deblurnet_paper/` (o treino anterior) nem `../retreinar-deblur/`.

> **Estado em 2026-09-25 — FASE 1 CONCLUÍDA**
> `40.000/40.000` · `Exited (0)` · loss ema 0,0984 · `val_loss` **0,0602** (validação real)
> 1.700 cenas · 55.760 amostras de treino (+13.940 retidas para validação) · batch efetivo 32 · 4 GPUs
> **LVCorr 0,983** (100 medições, faixa 0,967–0,992) — curva **PLANA**
> contra +0,9059 da fase 1 anterior e +0,8868 dos pesos oficiais.
>
> **Modelo:** `https://huggingface.co/juliadollis/regen-bokeh-rotaA`
> (`bokeh_fase1_step40000.safetensors`, 927,3 MB, privado)
> Snapshots por step: `juliadollis/genrefocus-bokehnet-v2-fase1` (43 arquivos)
>
> **FASE 2 (B+C) EM EXECUÇÃO desde 2026-09-28 00:30**
>
> | | |
> |---|---|
> | container | `julia_fase2_oficial` · GPUs **0, 1, 2, 5** da h100n1 |
> | config | `configs/train_bokeh_fase2_bc.yaml` · `outputs/bokeh_fase2_bc` |
> | steps | 60.000 (§4.1 fase ii) · batch efetivo **32** (1 × 8 × 4) |
> | dados | rota B **13.106** + rota C **12.925** = 26.031 reais |
> | espelhos | `espelho cobre 12925/12925 nomes` |
> | init | LoRA da fase 1, `step_40000.pt`, optimizer/scheduler frescos (§10.1) |
> | prazo | **~4,0 dias** (310 steps/h medido em 2 GPUs → ~620 em 4) |
> | pendente | **monitor externo de LVCorr** — precisa de 1 GPU livre |
>
> Decisões declaradas e medidas: `route_weights` 50/50, `min_calibration_ssim`
> 0,7 (p50 real 0,963), 220 cenas do bench excluídas por vazamento (§11.1).

---

## 1. A fase 1 — como rodou

| | |
|---|---|
| container | `julia_fase1_4gpu` na **h100n1** (Docker, sem SLURM) — `Exited (0)` |
| GPUs | **0, 1, 2, 5** — PIDs consecutivos, ~66,4 GB cada |
| config | `configs/train_bokeh_fase1_h100n1.yaml` |
| `output_dir` | `outputs/bokeh_fase1_1700cenas` |
| steps | 40.000 (§4.1 fase (i)) |
| batch efetivo | **32** = 1 × accum 8 × 4 GPUs |
| dados | 1.700 cenas · 69.700 amostras |
| wandb | `genrefocus-bokehnet-v2` / `bokeh-fase1-rota-a-1700cenas` |
| checkpoints | a cada 250 steps, 5 últimos + `best.pt` |

```bash
ssh dgx-H100-01
docker logs -f julia_fase1_4gpu        # treino
docker logs -f julia_monitor           # controlabilidade
cat /raid/user_juliadollis/julia_docker/retreinar-bokeh/outputs/bokeh_fase1_1700cenas/monitor_externo.jsonl
```

### Banner de confirmação

```
[train] stage=bokeh steps=40000 start=0 num_gpus=4 grad_accum=8 batch_efetivo=32
        seed=42 lora=cond-only guidance=1.0 scale_mode=short_side sigma_mu=full_image
[lora]  343 módulos | rank=64 alpha=64 | 231.8M params treináveis | variante=cond-only
```

`start=0` = do zero. O run anterior (1.043 cenas, até step 750) está preservado
em `outputs/bokeh_fase1_rota_a` — **nada foi apagado**.

---

## 2. Os dados — `juliadollis/bokehnet-regen-rota-a`

Release completo: **210.809 arquivos**, baixado para
`/raid/user_juliadollis/julia_docker/releases/rota_a`.

| pasta | arquivos | o que é |
|---|---|---|
| `generated/<cena>/<id>_bokeh.jpg` | 69.700 | **o alvo** — bokeh renderizado pelo BokehMe |
| `meta/<cena>/<id>.json` | 69.700 | escalares: `k_value`, `focus_disparity`, `disparity_min/max`, `image_h/w` |
| `mask/<cena>/<id>.png` | 69.700 | **não usado pelo treino** |
| `depth/<cena>.png` | **1.700** | disparidade uint16, **uma por cena** |
| raiz | 9 | `manifest.jsonl`, `split.json`, os dois ledgers, `run_config.json`, 2 de rejeições |

**A AIF não está no release** — é referência, resolvida por espelho local:

| fonte | amostras | espelho |
|---|---|---|
| `EBB!` (ref. [27]) | 34.850 | `/workspace/fontes_rota_a/EBB` |
| `GenerativePhotography` (ref. [80]) | 34.850 | `/workspace/fontes_rota_a/GenerativePhotography` |

São exatamente as duas fontes que o §4.1 cita para o pool sintético.

### Medido no dado real, antes de treinar

| medida | valor | leitura |
|---|---|---|
| profundidade | `I;16` uint16, lado longo **768** | = `DEPTH_LONG_SIDE` do contrato |
| ocupação do uint16 | **91,96%** (60.266 níveis) | no dado ANTIGO, 24,7% das amostras usavam < 256 níveis |
| saturação em 0 e 65535 | **0,00%** | sem clipping |
| `k_efetivo` p5/p50/p95 | 0,22 · **9,0** · 34,4 | **59 distintos em 64** — K varia de verdade |
| `defocus_max` p50 | 0,019 | longe de 1,0 → sem a assinatura do defeito D1 |
| fração saturada | **0,0** | `max_coc` certo para a fonte |
| **alertas** | **nenhum** | |

**O gargalo que derrubou o LVCorr de +0,91 para +0,44 não está neste release.**

---

## 3. Por que esta árvore existe

O pré-processamento anterior construía o mapa de defocus em **profundidade
linear normalizada por imagem**; a inferência oficial constrói em **disparidade
métrica absoluta**. A diferença **não é de escala** — nenhuma constante em `K`
converte uma na outra.

Medido (cena com 90% do conteúdo em 1–20 m, foco em 3 m, K=15), com o melhor
`α` por mínimos quadrados:

| | |
|---|---|
| erro relativo residual | **0,926** |
| Pearson entre os dois mapas | 0,341 |
| razão do CoC a 1,5 m · 20 m · 10.000 m | **4408×** · 331× · **0,66×** |

Por isso a busca binária por K da avaliação absorve a diferença de **escala** e
não a de **forma**. E é por isso que a controlabilidade colapsou:

| LVCorr | LF-Bokeh | RealBokeh | RealDOF |
|---|---|---|---|
| fim da fase 1 (só sintético) | **+0,9059** | +0,8609 | +0,9247 |
| fim da fase 2 (a+b+c) | **+0,4365** | +0,8257 | +0,1324 |
| pesos oficiais do paper | +0,8868 | +0,8498 | +0,9644 |

A degradação foi **monótona no tempo** — e **a loss caiu o tempo todo**.

---

## 4. O paper: o que diz e o que cala

### Especifica

| item | valor | onde |
|---|---|---|
| backbone | FLUX.1-dev + LoRA | §4.1 |
| LoRA rank | DeblurNet 128, **BokehNet 64** | §4.1 |
| batch | 1/GPU × accum 8 × 4 GPUs = **32** | §4.1 |
| currículo | **40K sintético + 60K real** | §4.1 |
| dados | ~70K sintéticos de [27]/[80]; ~26K reais | §4.1 |
| pool sintético | ~1,7K imagens → ~70K pares = **~41/imagem** | supl. B.2 |
| Eq. 2 | `D_def = K·\|D − D_focus\|`, **crua** | §3.2 |
| inferência | 28 passos | §4.1 |

As **41 variantes por cena** do release batem exatamente com o supl. B.2.

### NÃO especifica

Resolução de treino, otimizador, LR, scheduler, warmup, `K_min`/`K_max` da Eq. 5,
limiar de SSIM, `gamma`/`defocus_scale` do renderer, proporção entre rotas, **o
normalizador do mapa** e **σ**. Onde o paper cala, a autoridade é
`third_party/Genfocus/Inference_bokehNet.py`; onde os dois calam, é decisão
nossa **declarada**.

### O que a inferência oficial fixa

```python
MAX_COC   = 100.0                                    # :20
disp      = 1.0 / depth_metrico                      # :94  ← DISPARIDADE
disp_focus= median(disp[mask])                       # :118 ← mediana NA disparidade
cond_map  = clip(|K*(disp-disp_focus)|/MAX_COC,0,1).repeat(3,1,1)   # :138-140
prompt    = "an excellent photo with a large aperture"
guidance_scale = 1.0 ; num_inference_steps = 28
```

---

## 5. O monitor externo — e por que é externo

`scripts/monitor_externo.py`, na **GPU 6**, container separado.

**O princípio, que é da usuária:** o treino começa e termina do mesmo jeito.
Instrumento é observador, não participante.

O probe interno rodaria no rank 0 por ~5 min gerando 32 imagens de 28 passos, e
nesse tempo os outros três ranks ficariam parados no próximo coletivo — que é
**exatamente** o cenário do watchdog do NCCL que matou o multi-GPU duas vezes.
E a parada automática mudaria o comportamento de um run no meio do caminho.

O monitor então: lê os `step_*.pt` que o treino já grava, mede o **LVCorr** num
conjunto fixo de 8 cenas da partição de **validação** (disjuntas do treino), e
escreve em `monitor_externo.jsonl`. Não toca em RNG, optimizer, pesos nem
coletivo. **Se morrer, o treino nem percebe** — e isso já foi testado: ele tomou
OOM quando terceiros encheram a GPU 6, e o treino seguiu intacto no step 13.400.

Referência de comparação: **+0,9059** (fase 1 anterior).

---

## 6. O que mudou em relação ao treino anterior

Detalhe em `MUDANCAS_CODIGO.md`; o raciocínio em `AUDITORIA_TREINO_BOKEHNET.md`.

| eixo | antes | agora |
|---|---|---|
| mapa de defocus | profundidade normalizada por imagem | **disparidade métrica 1/z** |
| `K` no crop | ficava na escala original | **reescalado** (fator 0,34–0,89, varia por amostra) |
| `max_coc` | flag por rota (o `kfix` usou 10,5107) | **constante de módulo**, sem setter |
| profundidade no resize | BILINEAR | **NEAREST**, como a geração |
| LoRA | 344 módulos (pegava o `proj_out` de topo) | **343**, travado por teste |
| warmup | primeiro update com LR cheio | LR aplicado **antes** do primeiro update |
| seed | depois de criar o LoRA | **antes** |
| loss do log | 1 microbatch do rank 0 | média da **acumulação e das GPUs** |
| retomada | pesos | pesos + **RNG + época**, com guard de fase/convenção |
| §3.3 forma | não existia | 2º adapter, base **ativo e congelado** |
| acumulação | sync no fim da época (batch menor 1×/época) | `sync_with_dataloader=False` |

---

## 7. Verificação em GPU — 2026-09-18, 14/14

`scripts/teste_gpu_h100n1.py`. Fechou a lacuna "nada rodou em GPU".

| # | verificou | resultado |
|---|---|---|
| T1 | ambiente + `transformer_forward` | torch 2.9.0+cu126, diffusers 0.37.1, peft 0.18.1, accelerate 1.11.0 ✅ |
| T2 | `GradientAccumulationPlugin(sync_with_dataloader=False)` | **a API aceita o parâmetro** ✅ |
| T3 | LoRA contra o FLUX real | `343 módulos, rank 64, 231,8M treináveis` ✅ |
| T3 | `transformer.proj_out` de topo FORA | só `x_embedder` no topo ✅ |
| T4 | `mu` × `calculate_shift` do diffusers | batem ✅ |
| T5 | §3.3: `set_adapter([base, forma])` | **os DOIS adapters ATIVOS** ✅ |
| T5 | §3.3: só a forma treina | 686 tensores, todos do adapter de forma ✅ |
| T6 | release em árvore → dataloader → 3 steps reais | loss 0,878 → 0,675 → 0,555 ✅ |

T2 e T5 eram afirmações minhas **sem execução**. T5 confirmou que o `add_adapter`
do diffusers desativava o LoRA base — o defeito A1.

**252 testes** sem GPU: `python3 -m pytest tests/ geo_cond/tests/ -q`.

---

## 7-bis. Auditoria do código de treino — 2026-09-25

Sete correções, quatro delas em coisas que já tinham custado GPU. Detalhe e
evidência em `MUDANCAS_CODIGO.md` (seção "Rodada de 2026-09-25").

| # | achado | estado |
|---|---|---|
| R0 | **o README dizia que a validação estava desligada — estava errado** | corrigido; ver a nota no §8 |
| R1 | probe interno ligado nos configs da fase 2/3: em multi-GPU é o deadlock do NCCL | o laço **recusa**; 7 configs a `probe_every_steps: 0` |
| R2 | campo de config desconhecido virava default em silêncio | agora **levanta** |
| R3 | FLUX pelo repo-id → `FileNotFoundError` no cache root-owned (matou 2 lançamentos hoje) | `train.py` honra `FLUX_PATH` e recusa caminho não-local |
| R4 | o config da fase 1 **não reproduzia a fase 1** (faltava o 2º espelho, entre outros) | alinhado ao `effective_config.yaml` |
| R5 | `bokeh-shape` sem `--init-lora` dava `TypeError` opaco | mensagem própria |
| R6 | 2 testes estavam errados (o código, certo) | viraram asserções estruturais |
| R7 | `/dev/shm` de 64 MB mata os workers com `Bus error` depois do FLUX carregar | guarda nova, com a flag que resolve |

**253 testes** passam. Verificado em GPU: a guarda do FLUX, a do probe em 2
processos (dispara em **todos** os ranks, sem deadlock), e o caminho de crash —
sob OOM o `finally` gravou checkpoint e safetensors antes de morrer.

O `--shm-size=64g --ipc=host` do `docker run` **não é opcional**: foi o que
separou a fase 1 (que rodou 40K steps) das minhas duas tentativas de verificação
(que morreram em `Bus error` dez minutos depois de carregar o FLUX).

---

## 8. O que está ERRADO neste run — e é sabido

> **CORREÇÃO (2026-09-25).** Este quadro dizia que a validação estava
> desligada e que o `best.pt` era lixo. **Era errado, e o erro era meu.** O que
> me enganou: os ranks 1-3 imprimem `[eval] AVISO: ... validação DESLIGADA`
> porque `_build_val_loader` devolve `None` fora do rank 0 **por projeto** — foi
> assim que o travamento do NCCL foi resolvido. Grepar o log encontrou o aviso e
> eu li como defeito. O rank 0 **não** imprimiu.
>
> O que o artefato mostra: `best.pt` carrega
> `metrics_snapshot = {"val_loss": 0.06018715, ...}`, e esse campo só é gravado
> no ramo `if melhorou and save_best and val_loss is not None`. E
> `effective_config.yaml` registra `val_datasets` apontando para o release com
> `scene_split_partition` forçado a `"val"` — 20% das cenas, disjuntas por CENA
> das 80% do treino (o log confirma: `-13940 (20.0%) scene_split_excluded`).
>
> **A validação rodou. O `best.pt` é legítimo**, do step 40.000, escolhido por
> `val_loss` em dado não visto. O defeito real era a mensagem mentir nos ranks
> ≠ 0 — já corrigido (`accelerator.is_main_process` na guarda).

| # | o quê | efeito | por que está assim |
|---|---|---|---|
| 1 | **probe interno desligado** (`probe_every_steps: 0`) | o treino corre cego para o modo de falha da fase 2 | mitigado pelo **monitor externo** (§5), que é a decisão declarada |
| 2 | **`sigma_mu_source: full_image`** | μ vai de 0,630 para 2,446 numa foto 3MP | casa com a inferência, mas **o paper não menciona σ** — contamina atribuição |
| ~~3~~ | ~~validação desligada~~ | — | **não era verdade**, ver a correção acima |

**O `best.pt` desta run pode ser usado**: step 40.000, `val_loss` 0,0602 na
partição de validação. Ele é idêntico ao `step_40000.pt` em pesos — a validação
melhorou até o fim, que é o mesmo que o monitor externo diz sobre o LVCorr.

**Sobre o #3:** se este treino melhorar, não dá para atribuir todo o ganho ao
dado — mudaram também LoRA, warmup e a distribuição de σ. Está declarado.

---

## 9. Histórico da execução — o que custou tempo

| o quê | causa | lição |
|---|---|---|
| download de 20h→34h | paralelizei em 7 processos e estourei a quota de **2.500 req/5min**; 6 morreram com `429` | paralelizar piorou; o gargalo era requisição, não banda |
| `snapshot_download` travado | listava 210.809 arquivos antes de baixar | baixar pelos caminhos do `manifest.jsonl` eliminou a listagem: 1,7 → 4,0 arq/s |
| `mask/` baixando primeiro | ordem alfabética; 69.700 arquivos que o treino não lê | filtrar por padrão tirou 40% do trabalho |
| 1º treino morreu | config apontava o **repo-id** do FLUX, não o snapshot local | o `FLUX_PATH` só valia nos scripts de teste |
| 2º e 3º morreram | **watchdog do NCCL**: 4 ranks construíam o loader de validação em paralelo, um ficava para trás | subir o timeout de 480→3600s **não** resolveu: era dessincronia, não lentidão |
| correção | val loader **só no rank 0** | destravou o multi-GPU |
| release incompleto | `aif_ref` → `GenerativePhotography` sem espelho mapeado | o loader **falhou alto** em vez de adivinhar — a regra "sem fallback" funcionando |

---

## 10. Layout

```
genfocus_train/control.py    a fórmula do controle, implementação ÚNICA
genfocus_train/release.py    lê as duas formas de release, resolve espelhos
genfocus_train/probe.py      LVCorr
genfocus_train/data.py       dataloader; `metric_disparity` é o caminho novo
genfocus_train/backbone.py   FLUX + LoRA; 2º adapter da §3.3
genfocus_train/trainer.py    laço, checkpoints, validação, retomada
scripts/monitor_externo.py   o observador na GPU 6
scripts/teste_gpu_h100n1.py  os 14 testes em placa
scripts/valida_rota_a_parcial.py  mede o dado antes de treinar
configs/                     fase 1 (n1/n3), fase 2 (3 braços), fase 3, smoke
tests/                       152 testes, sem GPU e sem rede
```

---

## 11. Aberto

### O currículo da fase 2 — conferido no texto (2026-09-25)

| pergunta | resposta | evidência |
|---|---|---|
| **LR continuada entre as fases?** | **o paper não menciona LR. Zero vezes.** `grep -icE "learning rate\|optimizer\|scheduler\|warmup" → 0` | decisão nossa: optimizer e scheduler **frescos**, porque o §3.2(a) diz *"we **pretrain** with synthetic data"* |
| **Rota B e depois a C?** | **NÃO — as duas juntas.** §4.1: *"a **hybrid dataset** ... **combined with** ~26K real examples"*; a Tab. 6 varia **inclusão**, não ordem | já é o que `train_bokeh_fase2_real.yaml` faz |
| **Proporção entre B e C?** | o paper cala. Hoje é `null` = tamanho relativo dos shards, ou seja **acidente** | a Tab. 6 mostra (a)+(c) **melhor** que (a)+(b) — 0,0972 vs 0,1156 |

**A ressalva do LR:** a fase 1 termina em `1e-5` e a fase 2 sobe a `1e-4` — salto
de 10× justo quando entram os dados reais, candidato ao esquecimento que derrubou
a rodada anterior. Por isso existe o braço `..._lrbaixo.yaml`. Medir, não supor.

Detalhe completo, com as citações literais e a tabela dos erros antigos que cada
guarda impede: **`PLANO_RETREINO_BOKEHNET.md` §10**.

---

| # | o quê | destrava com |
|---|---|---|
| 1 | σ: `full_image` × `crop` | decisão declarada; A/B custaria reiniciar |
| 2 | **fase 2** (60K real) | download das rotas B e C (~2 h); código já validado no dado real |
| 3 | **§3.3** (forma de abertura) | PointLight-1K + BokehMe com kernel de forma (Eq. 6) |
| ~~4~~ | ~~`min_calibration_ssim`, proporção entre rotas~~ | **decidido e medido** — §11.2 e §11.3 |
| 5 | testes de acumulação e export→recarga | exigem `accelerate`/`peft` no ambiente local |

### Regras do cluster (`../INSTRUCOES_H100.md`, `../CLAUDE.md`)

**Nunca** cancelar job, **nunca** apagar nada sem perguntar, `--time` sempre
alto. Na h100n1 é Docker sem fila: `--user $(id -u):$(id -g)`, container
detached, `--gpus '"device=N,M"'` com aspas embutidas, e **nunca**
`docker rm`/`stop`/`prune` de terceiro. Há containers de outras 4 pessoas no
host — usamos 4 das 8 GPUs, mais a 6 para o monitor.

### A ressalva geral

Ninguém fora do grupo dos autores reproduz este paper bit a bit: o repositório
oficial não publicou código de treino, dados, o benchmark LF-Bokeh, limites de K,
limiar de SSIM, configuração do renderer, resolução de treino, otimizador nem
scheduler. O alvo possível é uma reprodução **pública, coerente e auditável, com
cada desvio declarado** — não identidade com o treino privado.
