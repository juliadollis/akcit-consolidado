# Auditoria do treino da nossa DeblurNet contra o paper

Pergunta: *por que a nossa AIF tem menos detalhe que a oficial, se treinamos seguindo o
paper?*

**Resposta curta: o treino está fiel ao paper nos eixos que o paper especifica. O defeito
não está no treino — está na INVOCAÇÃO. O peso que gerou as 11.563 AIF foi treinado
`cond-only` e foi rodado com `main_adapter="deblurring"`, que liga a LoRA em dois ramos
que o treino nunca tocou (o ramo principal e o do texto). O peso oficial, no mesmo lote,
rodou com `main_adapter=None` — que é o roteamento CORRETO para ele. O descasamento é
assimétrico: só a nossa AIF o sofre.**

Isto não contradiz o fato medido de que `main_adapter="deblurring"` está provado em
11.563/11.563 — confirma. O ajuste é idêntico nos dois modelos, e é por isso que ele não
aparecia como suspeito; mas ele só é *correto* para um dos dois.

---

## 1. A prova de proveniência

### 1.1 Qual arquivo gerou as AIF

`bokehnet-regen/docker/run_route_b_h100n1.sh:59` fixa o diretório dos pesos:

```
PESOS_DIR="${PESOS_DIR:-${BASE}/retreinar-deblur/outputs/deblur_docker_4gpu/deblur}"
```

e `:68` escolhe o arquivo pela variante: `ours_main_cond) ARQ_PESO="deblur.safetensors"`.
A variante default é `ours_main_cond` (`run_route_b_h100n1.sh:39`).

Medido por mim no cluster (somente leitura):

```
sha256  a1ed05e48f7a3469cf825eda6a8ceca4cca1e6f6dbdb665d78a6f9e207d8cecc
        /raid/user_juliadollis/julia_docker/retreinar-deblur/outputs/
        deblur_docker_4gpu/deblur/deblur.safetensors                     (1 854 533 784 B)
```

Esse é exatamente o `a1ed05e4…8cecc` que `reference/ROTA_B_AUDITORIA_2.md:541` registra
como *"nossa, **a que os runs usaram**"*, e o mesmo que `REGISTRO.md:1372-1373` registra
para a rota B (*"Variante `ours_main_cond`, `main_adapter="deblurring"`, checkpoint de step
60.000 com sha256 `a1ed05e4…`"*).

### 1.2 Esse arquivo é cond-only, não main+cond

O sidecar irmão, escrito pelo próprio treino no mesmo export, em
`outputs/deblur_docker_4gpu/deblur/deblur.json`:

```json
"adapter_variant": "cond-only",
"text_adapter": null,
"rank": 128, "alpha": 128, "n_lora_modules": 343, "n_tensores_lora": 686,
"steps_total": 60000, "train_guidance": 3.5, "cond_guidance": 1.0
```

Que esse JSON é o irmão daquele `.safetensors` e não de outro: mesma pasta, mesmo mtime
(`Sep 13 12:24` nos dois), e `"config_hash": "855f21ac569a0f70"` idêntico ao
`outputs/deblur_docker_4gpu/run_metadata.json` do run — o `trainer.py` escreve
*"um `.json` irmão acompanha todo `.safetensors` exportado"* (`trainer.py:26-27`).

E o config que produziu esse run, `retreinar-deblur/configs/train_deblur_docker_4gpu.yaml:100`:
`lora_on_main: false`, com `:102` `lora_on_text: false`. (O único config alternativo que
poderia ter subido esse `output_dir`, `train_deblur_docker_4gpu_gc.yaml:96`, também é
`lora_on_main: false` — cond-only em qualquer dos dois.)

### 1.3 Mas ele foi rodado como main+cond

`bokehnet-regen/src/model_runtime/deblurnet.py:214` define, para a variante
`OURS_MAIN_COND`, `main_adapter=ADAPTER_NAME` (= `"deblurring"`), e `:942` passa
`main_adapter=self._spec.main_adapter` **sempre, explicitamente** — por desenho, para
impedir o defeito inverso.

A spec está apoiada em evidência obsoleta. Seu campo `evidence`
(`deblurnet.py:216`) cita `HANDOFF_PROJECT_HISTORY.md:92,109,161 (treino main+cond,
step_60000)`, e seu `repo_id` é `juliadollis/genrefocus-deblurnet-paper-4gpu` — que de
fato é main+cond (`genrefocus_deblurnet_paper/PLANO_CORRECOES_DEBLURNET.md:691`: *"é
main+cond apesar do nome"*). Mas o `PESOS_DIR` do script aponta para **outro** arquivo:
o export cond-only do run novo. O nome do arquivo é `deblur.safetensors` nos dois casos,
e a única trava efetiva é o nome do arquivo — a trava de sha256 existe e **não foi usada**
nos runs em voo (`ROTA_B_AUDITORIA_2.md:552-556`: *"`docker/run_route_b_h100n1.sh` não a
passa, e `expected_lora_sha256` fica `None`"*).

Há confirmação independente pela contagem de chaves: o arquivo publicado no HF sob o
`repo_id` da spec tem **688** chaves (344 módulos, arquitetura pré-C1) e o arquivo que os
runs usaram tem **686** (343 módulos, pós-C1) — `ROTA_B_AUDITORIA_2.md:541-543,566-571`,
achado A17 em `:1079`. São arquivos diferentes, de famílias diferentes.

### 1.4 O que `main_adapter="deblurring"` faz, mecanicamente

`Genfocus/pipeline/flux.py:758,791,812` monta `adapters = [main_adapter] * 2 + c_adapters`.
O índice 0 é o ramo de **texto**, o 1 é o ramo **principal** (tokens latentes ruidosos),
os demais são as condições. `specify_lora` (`flux.py:146-162`) põe `scaling = 1` para o
adapter nomeado naquele ramo e `0` nos outros:

```python
module.scaling[adapter] = 1 if adapter == specified_lora else 0
```

Ou seja:

| | ramo texto | ramo principal | condições |
|---|---|---|---|
| nosso TREINO (`backbone.py:643-645`, `lora_on_main=false`) | LoRA **off** | LoRA **off** | LoRA on |
| nossa GERAÇÃO de AIF (`main_adapter="deblurring"`) | LoRA **ON** | LoRA **ON** | LoRA on |
| geração OFICIAL (`main_adapter=None`) | off | off | on |

A LoRA entra em escala plena em Q/K/V, `norm1.linear`, `ff.net.2`, `proj_mlp`, `proj_out`
dos 19 blocos duplos e 38 single, **e no `x_embedder`** — que é a projeção de entrada do
patch. `flux.py:385` aplica `specify_lora((self.x_embedder,), adapters[i + txt_n])` por
ramo, então o delta treinado para embutir a *condição limpa* passa a ser aplicado também
ao *latente ruidoso*. É perturbação na entrada do denoiser, em todos os 28 passos.

### 1.5 Por que o roteamento é correto para o oficial e errado para o nosso

`third_party/Genfocus/Inference_deblurNet.py:103-111` chama `generate(...)` **sem**
`main_adapter`; o default é `None` (`flux.py:480`). Logo o peso oficial é cond-only por
construção, e a variante `OFFICIAL_COND_ONLY` (`deblurnet.py:222`, `main_adapter=None`)
o roda exatamente como os autores rodam. Nosso peso é *também* cond-only — e é o único
dos dois que foi rodado fora da sua convenção.

### 1.6 Por que a magnitude fecha

O projeto já documentou um descasamento de roteamento com custo catastrófico: LPIPS ~0,85,
*"saída LAVADA"* (`HANDOFF_PROJECT_HISTORY.md:102,171`; `configs/README.md:10`;
`deblurnet.py:13-17`). Aquele caso é o **inverso** do nosso: um peso main+cond rodado
como cond-only, em que a LoRA **desaparece** do ramo principal e o modelo volta a ser
quase o FLUX cru — daí a catástrofe. O nosso caso é aditivo: a LoRA está presente onde foi
treinada (condições) **mais** onde não foi. Degradação moderada e sistemática — mais macio,
menos textura — é a magnitude esperada, e é a medida (razão de variância do Laplaciano
0,467; energia de alta frequência 0,751 no texturizado contra 0,882 no liso).

E há um controle interno que fecha o argumento: **o mesmo arquivo**, avaliado na Tabela 2
com o roteamento certo (`main_adapter=None`, `inferencia/RESULTADOS.md:8`;
`rodar_tabela2.py:183` tem default `"none"`), dá DPDD LPIPS 0,1744 contra 0,1596 do
oficial e **ganha** do oficial no RealDOF (0,2291 contra 0,2397) —
`inferencia/RESULTADOS.md:13-35,60-67`. O peso não está lavado. A invocação está.

### 1.7 A medição já existia no repositório, e a conclusão dela precisa ser revista

`ROTA_B_AUDITORIA_2.md:859-873` já mediu isto, pareado em **1.644** fotos do ITW:

| grandeza | oficial p50 | nossa p50 | razão p50 |
|---|---|---|---|
| `aif_laplacian_variance` | 1005,0 | **358,9** | **0,438** |
| `bokeh/aif` laplaciano | 0,638 | **1,344** | `>1` em **77,1%** da nossa contra 22,8% da oficial |
| SSIM(AIF, bokeh) | 0,830 | **0,564** | 0,697 |
| resposta da correlação de fase | 562,5 | **83,7** | 0,154 |
| `z_max_m` | 103,61 | **7,24** | 0,064 |
| `k_value` | 17,077 | 17,117 | 0,999 |

A razão de 0,438 em n=1.644 e a de **0,467** em n=11.563 são a mesma medida, replicada:
o achado está confirmado em duas amostras independentes. Note `bokeh/aif > 1` em 77,1%:
**em três de cada quatro fotos a nossa "AIF" tem menos alta frequência que a foto borrada
que ela deveria estar deblurrando.**

Aquela auditoria registrou isso como `[A]` **A16, aberto, risco médio**
(`ROTA_B_AUDITORIA_2.md:1077`) e adotou a interpretação (a): *"o modelo trocando energia de
alta frequência (ruído de sensor e artefato de JPEG do rendition do Flickr …) por estrutura
limpa"* — logo não seria degradação. **Os dois pilares dessa interpretação caem:**

1. **O fato medido que motivou esta auditoria a refuta diretamente.** A perda de alta
   frequência é **maior no texturizado (0,751) que no liso (0,882)**. Ruído de sensor e
   artefato de JPEG vivem preferencialmente nas regiões lisas; se o modelo estivesse
   trocando ruído por estrutura limpa, o déficit se concentraria no liso. Ele se concentra
   na textura. É textura real que está sendo perdida.
2. **O contra-argumento "uma AIF lavada não melhora LPIPS nem DISTS contra gabarito real"
   (`ROTA_B_AUDITORIA_2.md:931-935`) não se aplica ao lote das AIF.** Aquele ganho no
   RealDOF foi medido com **`main_adapter=None`** (`inferencia/RESULTADOS.md:6-8`, literal:
   *"Todas as linhas `medido` saíram do mesmo pipeline … `main_adapter=None`"*) — ou seja,
   com o roteamento **certo** para este peso cond-only. A rota B gerou as AIF com o
   roteamento **errado**. As duas medições são de configurações diferentes do mesmo
   arquivo, e a boa não atesta a má. Pelo contrário: ela é o controle que mostra que o peso
   está são.

E a hipótese (c) daquela auditoria — *"alguma coisa na inferência da nossa variante difere
da oficial além do adapter"* (`:913-915`) — é o que §1.1–§1.5 demonstra, com uma torção:
**é** o adapter, e a checagem de proveniência não podia vê-lo. O que §3.1 de lá provou é
que `main_adapter="deblurring"` foi *usado* em 1.644/1.644; o que ela não podia provar é
que esse valor era o *correto* para o arquivo carregado. Nada no `.safetensors` diz de qual
convenção ele precisa — é o que o próprio módulo declara (`deblurnet.py:21-24`).

### 1.8 Dois campos de proveniência estão falsos neste lote

A spec é hardcoded na enum e não é lida do arquivo local, então a proveniência gravada em
cada amostra da nossa variante afirma (`ROTA_B_AUDITORIA_2.md:498-501`):

- `deblur_lora_mode: main_cond` — **falso**: o peso é `cond-only` (`deblur.json`);
- `deblur_repo_id: juliadollis/genrefocus-deblurnet-paper-4gpu` — **falso**: o peso veio de
  `--deblur-weights-dir` apontando para `outputs/deblur_docker_4gpu/deblur`
  (`run_route_b_h100n1.sh:59,137`), e o `repo_id` sai de `deblurnet.py:212`, não do arquivo.
  O sha256 gravado (`a1ed05e4…`) é do retreino cond-only, publicado como
  `juliadollis/genrefocus-deblurnet` (`inferencia/RESULTADOS.md:153`) — repositório
  diferente.

`deblur_lora_sha256` e `main_adapter`, esses sim, são fiéis — e é por eles que este
diagnóstico foi possível. A recomendação de `ROTA_B_AUDITORIA_2.md:561` (passar
`--deblur-lora-sha256` nos próximos lotes) vira, à luz disto, insuficiente: a trava
precisaria amarrar sha256 **e** `lora_mode`, porque é o `lora_mode` que estava errado.

### 1.9 Sobre o tamanho do lote

O número **11.563** não aparece em nenhum arquivo do repositório. Os lotes registrados são
**13.615** amostras em `output/b_release` (a nossa variante nova — `AVALIACAO_DATASETS_BC.md:14`,
de 13.800 linhas do ITW) e **11.635** na rota B **antiga** do `bokehnet-preprocessing`
(`bokehnet-preprocessing/docs/ERROS_GERACAO_ORIGINAL_BOKEHNET.md:224-225`). Se o pareamento
de 11.563 saiu de `b_release` ∩ `b_full_official`, o diagnóstico acima vale integralmente.
**Se ele saiu do lote antigo de 11.635, vale outro diagnóstico ainda pior**: aquele lote
rodou com `main_adapter` **não passado** (`bokehnet-preprocessing/src/model_runtime/deblurnet.py:201-211`),
o que é o defeito B1 clássico, e o peso era o main+cond de
`juliadollis/genrefocus-deblurnet-paper-4gpu`. Vale confirmar de qual lote vieram as
11.563 antes de agir.

---

## 2. Hipótese 1 do roteiro: a função de perda — **descartada como divergência**

O que o paper manda: `paper.txt:305` é a única frase sobre a supervisão da DeblurNet —
*"We train DeblurNet in a supervised manner using real paired data [1, 57]."* Busca no
paper por `loss`, `objective`, `L1`, `L2`, `MSE`, `LPIPS`, `perceptual`, `velocity`,
`flow match`: **zero** ocorrências definindo o objetivo do Stage 1. O paper não especifica
a perda.

O que usamos, `genfocus_train/models.py:275-295`:

```python
if weight is None:
    return F.mse_loss(prediction.float(), target.float())
```

com alvo de rectified flow `x_t = (1-σ)x₀ + σε`, `v* = ε - x₀`
(`genfocus_train/backbone.py:581-584`) — o objetivo canônico de flow matching do FLUX.

- **Sem reponderação.** O construtor de batch da DeblurNet retorna
  `TrainBatchOutputs(prediction=pred, target=target)` sem `weight`
  (`models.py:141`), então `weight is None` e a perda é MSE puro. A reponderação
  geométrica por oclusão (`geo_cond/loss_weight.py`) só existe no caminho do **BokehNet**
  (`models.py:252-268`) e estava desligada por default
  (`REGISTRO_GEO_COND.md:340-342`, `occlusion_lambda=0.0` reproduz a perda anterior bit a
  bit, com teste).
- **Sem termo perceptual/GAN.** Verdade, e vale dizer: um objetivo de máxima
  verossimilhança pontual não premia textura. Mas isso **não é divergência do paper** — é
  o objetivo padrão para LoRA em FLUX, o paper não diz outra coisa, e a inferência é
  iterativa (28 passos), que é o que restitui a alta frequência. A assinatura "regride à
  média" da imagem final não sai de uma MSE no espaço de velocidade.

**Conclusão: a perda é a canônica e não divergente. Não é a causa.**

---

## 3. Hipótese 2 do roteiro: `--sem-identidade` — **descartada, é um falso amigo**

`retreinar-deblur/inferencia/rodar_tabela2.py:175-176`:

```python
p.add_argument("--sem-identidade", action="store_true",
               help="não calcula a linha Input (útil se já foi calculada)")
```

e `:213-214`:

```python
if not args.sem_identidade:
    alvos.append(("Input (nosso)", raiz / chave / "input", dict(identidade=True)))
```

"Identidade" aqui é a **linha de base pass-through** da tabela — a imagem de entrada
copiada para a saída, sem modelo, usada como âncora de protocolo. A flag apenas **não
recalcula essa linha**. Ela:

- **não** é um termo de perda de preservação de identidade;
- **não** entra no treino: as duas únicas ocorrências da string em todo o repositório
  estão em `rodar_tabela2.py` (`grep -rn "sem.identidade"` na raiz → 2 hits, ambos nesse
  arquivo);
- **não** altera em nada a linha do nosso modelo: os alvos são construídos
  independentemente (`:215-218`), com os mesmos parâmetros.

Que o step 60.000 tenha sido medido com a flag e o 15.500 sem ela é irrelevante para a
comparação entre os dois — muda só se a linha `Input` foi reinferida naquela execução.

---

## 4. Hipótese 3 do roteiro: os dados de treino

### 4.1 O que é fiel

| paper | nosso | veredito |
|---|---|---|
| `paper.txt:988-991`: *"all images from the official training split of the DPDD dataset"* + *"top 3,000 sharpest images"* do RealBokeh_3MP, por variância do Laplaciano | `train_deblur_docker_4gpu.yaml:160-168`: `akcit-pixel/DDPD:train` inteiro + `akcit-pixel/RealBokeh:train` com `top_k_sharpest: 3000`, `sharpness_column: image_focus` | receita fiel |
| `paper.txt:525-526`: *"3.5K pairs"* | 344 (DDPD train) + 3000 = **3.344 pares** | fiel na contagem |

### 4.2 Duas divergências reais, ambas na *implementação* do filtro

**(a) O ranking é por LINHA, e o RealBokeh é cena × abertura.** A nitidez é medida na
coluna `image_focus`, que é a MESMA imagem em todas as linhas de uma cena (uma linha por
abertura). Os empates são exatos e o `argsort` puxa cenas inteiras. O próprio código
registra a magnitude, em `genfocus_train/data.py:570-574`:

> *"no RealBokeh cada CENA aparece em ~5 linhas — uma por abertura — e todas compartilham
> o MESMO `image_focus` … um ranking por linha puxa cenas inteiras: **~580 alvos distintos
> para 3000 linhas selecionadas**."*

Ou seja: **~580 AIF distintas repetidas ~5,2x**, não 3.000 imagens distintas
(`PLANO_CORRECOES_DEBLURNET.md:494,503-504`). Com o DDPD, ~924 alvos distintos contra os
3.344 que a contagem sugere — **~3,6x menos diversidade de conteúdo**. `top_k_mode: row`
é uma escolha declarada e documentada como *"leitura literal do §B.1"*
(`train_deblur_docker_4gpu.yaml:186-187`), e é defensável: 3.000 **cenas** × 5 aberturas
daria 15K pares, não os 3,5K que `paper.txt:525` publica. A frase do paper é ambígua; o
que **não** é ambíguo é que treinamos 60K steps sobre ~900 alvos distintos com rank 128.

**(b) O Laplaciano é medido num proxy 256², não na imagem.** `data.py:509`:

```python
img = _to_pil_rgb(image_like).convert("L").resize((probe_size, probe_size), Image.BILINEAR)
```

`paper.txt:989-990` diz *"computing the Laplacian variance of each image"*. Variância do
Laplaciano depende de escala; medi-la num 256² reamostrado de um RealBokeh_3MP
(~2048×1536, fator ~0,125 em área) produz um ranking que **não é** o de resolução nativa.
As 3.000 linhas que selecionamos não são, em geral, as 3.000 que o paper selecionaria.
Divergência real, magnitude não medida.

### 4.3 O que eu **não** consegui medir

A nitidez dos alvos de treino, e o número real de alvos distintos no run. O cache de
índices do filtro vive em `$HF_HOME/genfocus_filter_cache/` (`data.py:543-553`) e o
`HF_HOME` do treino era `/workspace/hf-cache-v2` (`run_docker_h100n1.sh:105`). Esse cache
**não existe mais** no host: só há `hf-cache` e `hf-cache-julia`
(`ls -d .../hf-cache*`), e o `run_route_b_h100n1.sh:48-50` confirma que *"os dois já
sumiram e voltaram durante a limpeza"*. O `logs/` do treino está vazio, então a linha
`[filter] mantendo top-3000/...` e o `[cena] … linhas/cena` que o código imprime não estão
recuperáveis. Recalcular exigiria rebaixar o RealBokeh (46,9 GB) — proibido pela regra de
não baixar nada. **Os ~580 seguem sendo a previsão documentada do código, não uma medição
minha.**

---

## 5. Hipótese 4 do roteiro: os hiperparâmetros contra o paper

Conferido contra o `deblur.json` do run — não contra o YAML, para medir o que de fato
aconteceu:

| `paper.txt` | manda | nosso (`deblur.json`) | divergência |
|---|---|---|---|
| :513 | LoRA rank r=128 | `"rank": 128`, `"alpha": 128` | **nenhuma** |
| :514 | batch por GPU = 1 | `"batch_size": 1` | **nenhuma** |
| :514 | acumulação de 8 passos | `"gradient_accumulation_steps": 8` | **nenhuma** |
| :514-515 | 4 GPUs (→ batch efetivo 32) | `"num_gpus": 4`, `"batch_efetivo": 32` | **nenhuma** |
| :515 | 60K steps | `"steps_total": 60000`, e `checkpoints/latest.json` = `step_60000` | **nenhuma** |
| :515 | 4× RTX **A6000** | 4× H100 80 GB | hardware, não hiperparâmetro. Sem BatchNorm e com σ sorteado por amostra, o batch efetivo 32 é o que importa |

Não especificados pelo paper (busca exaustiva: zero ocorrências de optimizer, learning
rate, scheduler, warmup, weight decay, augmentation, precisão, resolução de treino,
tamanho de crop, guidance de treino) e portanto **escolha nossa, não divergência**: AdamW
lr 1e-4 / wd 1e-4, cosine com warmup 500, bf16, 512×512, `scale_mode: short_side`,
`sigma_mu_source: crop`.

**Conclusão: nos quatro eixos que o paper publica, zero divergência. Capacidade e
orçamento de treino estão descartados — e o rank do nosso (128) é o que bate com
`paper.txt:513`, enquanto o peso oficial liberado tem rank 64
(`ROTA_B_AUDITORIA_2.md:586-587`), ou seja o paper e o peso oficial divergem entre si.**

### 5.1 Um eixo não especificado que ainda merece medição: a escala espacial

O treino viu 512×512 com o lado **menor** reescalado para 512 e `μ` do `calculate_shift`
computado sobre os 1024 tokens do crop (`data.py:225-230`,
`train_deblur_docker_4gpu.yaml:181-183`). A geração de AIF roda em resolução **nativa**
com tiling, e nesse regime o `μ` sai da imagem inteira. A aritmética está em
`PLANO_CORRECOES_DEBLURNET.md:376-388`: `exp(μ)` vai de 1,878 no treino para 2,516 na
inferência nativa, e a entrada chega com blur **1,34x maior** do que qualquer coisa que o
treino viu. Sub-deblur nesse regime produziria exatamente "menos alta frequência, e mais
falta no texturizado". Está registrado como **divergência assumida e não medida**
(`MUDANCAS_CODIGO.md:457-458`), e os três experimentos que decidiriam foram cortados
(`MUDANCAS_CODIGO.md:15-22`).

**Mas a medição de §1.7 a enfraquece.** Sub-deblur por escala deixaria a saída *perto da
entrada*: SSIM(AIF, bokeh) alto e correlação de fase alta, com a nitidez parecida com a da
bokeh. Medido é o oposto — SSIM 0,564 contra 0,830 da oficial e correlação de fase 83,7
contra 562,5. A nossa AIF **se afasta muito mais** da entrada **e** é mais macia: isso é
perturbação ativa no denoiser, não blur residual não removido. Fica como hipótese nº 2,
bem atrás do roteamento, e é a única candidata *de treino* que sobrevive.

---

## 6. Hipótese 5 do roteiro: o checkpoint intermediário e a curva

Medido por mim em `outputs/deblur_docker_4gpu/deblur/metrics.jsonl` (6.022 linhas, 120
pontos de validação; σ e ruído fixos, então é comparável entre steps —
`trainer.py:894-932`):

| step | `val_loss` |
|---|---|
| 500 | 0,300660 |
| 5.000 | 0,278089 |
| 10.000 | 0,272205 |
| **15.500** | **0,270269 ← mínimo** |
| 20.000 | 0,270999 |
| 30.000 | 0,273952 |
| 40.000 | 0,277294 |
| 50.000 | 0,279836 |
| **60.000** | **0,281548 (+4,17% sobre o mínimo)** |

A curva é **monotonamente crescente do step ~21.000 ao 60.000**. `checkpoints/best.pt`
ficou congelado no step 15.500 e nunca foi superado em 44.500 steps. Isto é sobreajuste
medido, e é coerente com ~900 alvos distintos e rank 128 (§4.2).

**Mas não é a causa principal, e há medição contra:** `inferencia/RESULTADOS.md:79-86`
mediu 15.5k contra 60k nas duas mesas e o **60k ganha nas duas**, em LPIPS e em DISTS
(DPDD 0,1942 → 0,1744; RealDOF 0,2410 → 0,2291). A `val_loss` de denoise não é proxy de
qualidade perceptual aqui. O que **não** foi medido em lugar nenhum é nitidez em espaço de
imagem entre 15.5k, 60k e o oficial — e é isso que o fato que motivou esta auditoria mede
pela primeira vez. Não consegui responder "a nitidez degrada ao longo do treino?" com os
logs: `metrics.jsonl` só tem `loss`, `loss_ema`, `lr`, `best_loss`, `val_loss` e
`peak_vram_gb`. Não há métrica de imagem por step, e os checkpoints intermediários
guardados são só os cinco últimos (59.000–60.000) mais o `best.pt`.

---

## 7. Placar da auditoria

### Causa mais provável
**O peso cond-only (`a1ed05e4…`, `adapter_variant: "cond-only"`) foi rodado com
`main_adapter="deblurring"` nas AIF da rota B**, ligando a LoRA no ramo principal e no ramo
de texto — inclusive no `x_embedder` do latente ruidoso — em configuração que o treino
nunca viu, enquanto o peso oficial rodou no seu roteamento correto. Evidência:
§1.1–§1.6. O achado **reabre `[A]` A16** (`ROTA_B_AUDITORIA_2.md:1077`) com causa
identificada, e substitui a interpretação (a) que aquela auditoria adotou (§1.7).

### Divergências do treino contra o paper, e o tamanho
1. Filtro top-3000 ranqueia linhas ⇒ **~580 alvos distintos em vez de 3.000** (~3,6x menos
   diversidade contando o DDPD). Previsto pelo código (`data.py:570-574`), não remedido por
   mim.
2. Variância do Laplaciano medida em **proxy 256²** e não na imagem (`data.py:509` contra
   `paper.txt:989-990`) ⇒ o top-3000 selecionado não é o de resolução nativa. Magnitude
   desconhecida.
3. DDPD é **10,3%** das amostras (344 contra 3.000) com shuffle uniforme, sendo o DDPD a
   mesa da Tabela 2 (`PLANO_CORRECOES_DEBLURNET.md:824-836`). O paper não fala de
   balanceamento; é divergência de protocolo em aberto.
4. Hardware: 4× H100 em vez de 4× RTX A6000. Batch efetivo idêntico (32); sem efeito
   esperado.

### Fiel, e descartado como causa
- Rank 128, batch 1, acumulação 8, 4 GPUs, batch efetivo 32, 60K steps — todos batem
  com `paper.txt:513-515`, conferidos no `deblur.json` do run (§5).
- Função de perda: MSE de flow matching, alvo `v* = ε - x₀`, **sem** reponderação e **sem**
  termo perceptual. O paper não especifica perda (`paper.txt:305`); a nossa é a canônica
  (§2).
- Reponderação geométrica por oclusão: era do BokehNet e estava desligada
  (`models.py:141`, `REGISTRO_GEO_COND.md:340-342`).
- `--sem-identidade`: flag de avaliação que só suprime a linha `Input` da tabela; zero
  efeito no treino e no modelo (§3).
- Capacidade: rank 128 é o do paper; o peso oficial tem 64 (§5).
- Guidance e prompt de inferência: `train_guidance: 3.5` no treino
  (`deblur.json`) e `guidance_scale` default 3,5 mais
  `PROMPT = "a sharp photo with everything in focus"` na geração
  (`deblurnet.py:120`, `flux.py:467`) — casam.

### O experimento que decide
**Regerar a AIF de uma amostra do ITW com o MESMO arquivo `a1ed05e4…`, mudando só o
roteamento para `main_adapter=None`, e remedir a variância do Laplaciano contra a AIF
oficial nas mesmas fotos.**

- Custo: ~300–500 imagens, 1 GPU, poucas horas. Nenhum retreino.
- Caminho sem tocar em código: `retreinar-deblur/inferencia/infer_deblur.py`, cujo
  `--main-adapter` já é `"none"` por default (`infer_deblur.py:228-229`). Por desenho,
  `bokehnet-regen` **não** aceita `main_adapter` como parâmetro — ele vem da spec da
  variante e de nenhum outro lugar (`deblurnet.py:931-944`),
  então a rota B não serve para o teste sem alterar a spec — e alterar a spec é
  justamente a decisão que este experimento deve informar.
- Leitura: se a razão nossa/oficial de variância do Laplaciano subir de **0,467** para a
  vizinhança de 1,0, o roteamento é a causa e a correção é trocar a spec da variante (ou
  regerar as 11.563 AIF). Se ficar em ~0,5, o roteamento está exonerado e o suspeito passa
  a ser a escala espacial treino-vs-inferência (§5.1), cujo teste seguinte é gerar a mesma
  amostra com `long_side=512` em vez de nativo.
- Controle ainda mais barato, e que a auditoria anterior já recomendou sem executar
  (`ROTA_B_AUDITORIA_2.md:940-944`): abrir 20–30 pares `generated/<id>_aif.jpg` × foto de
  origem, lado a lado, nas duas variantes. Os arquivos estão em disco.
- Controle barato que vale rodar junto, porque o dado já existe: comparar CLIP-IQA /
  MANIQA / MUSIQ **por imagem** entre a nossa linha e a oficial nos JSONs da Tabela 2
  (`RESULTADOS.md:124-131,136-137`). Aquela avaliação usou o roteamento certo; se lá a
  nitidez sem referência não estiver deficitária, é mais uma confirmação de que o defeito
  é de invocação e não de treino.

### O que não consegui verificar, e por quê
1. **A nitidez dos alvos de treino, e quantos alvos distintos o run realmente usou.** O
   cache de índices do filtro (`$HF_HOME/genfocus_filter_cache/`) ia para `hf-cache-v2`,
   que não existe mais no host; `retreinar-deblur/logs/` está vazio e os logs do container
   não sobreviveram. Recalcular exigiria rebaixar o RealBokeh — proibido.
2. **Se a nitidez em espaço de imagem degrada ao longo do treino.** `metrics.jsonl` só tem
   loss/lr/VRAM; não há métrica de imagem por step, e dos checkpoints intermediários só
   restam `best.pt` (15.500) e 59.000–60.000. Dá para medir 15.500 contra 60.000, mas não
   a trajetória.
3. **Como o peso oficial foi treinado.** Só sabemos, medido, que ele tem rank 64 contra os
   128 que `paper.txt:513` publica (`ROTA_B_AUDITORIA_2.md:586-587`) e que a inferência dos
   autores é cond-only (`Inference_deblurNet.py:103-111`). Qualquer afirmação sobre os
   dados ou o objetivo *deles* seria especulação.
4. **A magnitude do dano do roteamento.** É o experimento de §7; não o rodei porque
   exigiria GPU e porque a regra desta auditoria é somente leitura no cluster.
5. **De qual lote saíram as 11.563 fotos pareadas.** O número não existe em nenhum arquivo
   do repositório (§1.9). O diagnóstico vale para `b_release` (13.615, nossa variante nova);
   se o pareamento veio do lote antigo de 11.635, a causa é o defeito B1 clássico e não
   este. Só quem fez o pareamento pode desambiguar.
