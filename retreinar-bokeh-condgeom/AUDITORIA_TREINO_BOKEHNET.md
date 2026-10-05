# Auditoria do treino da BokehNet — código atual × paper × inferência oficial

Data: 2026-09-10
Escopo: **o treino** (Stage 2), não a geração de dados. A geração está sendo
refeita em `../bokehnet-regen/`; esta auditoria é sobre o que recebe esses dados.

Árvores auditadas:

| árvore | papel |
|---|---|
| `../genrefocus_deblurnet_paper/genfocus_train/` | o treino que rodou a fase 1 e a fase 2 da BokehNet |
| `../retreinar-deblur/genfocus_train/` | o mesmo código com as correções C1–C10 do DeblurNet aplicadas — **base desta pasta** |
| `../genrefocus_deblurnet_paper/third_party/Genfocus/` | código oficial dos autores (inferência); autoridade onde o paper cala |
| `../bokehnet-regen/src/control/contract.py` | o contrato canônico do sinal de controle, já escrito e testado |

Convenção de evidência, herdada do `bokehnet-regen`:
**[M]** medido · **[I]** inferido de código/paper · **[A]** assumido, não medido.

---

## 0. Resumo em uma página

O treino da BokehNet está **arquiteturalmente correto** e **contratualmente
desalinhado**.

O que casa com o paper e com a inferência oficial, conferido linha a linha e
listado na §2: rank 64, batch efetivo 32, currículo 40K + 60K, prompt, guidance
1.0, timestep 0 nas condições, `group_mask` diagonal entre condições, mapa de
defocus entrando no VAE em `[0,1]` cru, 343 módulos LoRA. Nada disso precisa
mudar, e a §2 existe para que ninguém "conserte" nenhum deles.

O que **não** casa é o sinal de controle. Três defeitos independentes:

1. **O dataloader fala um dialeto que a geração nova não vai falar.** Ele lê
   `depth` como profundidade métrica normalizada por imagem em `[0,1]` mais um
   `s1` na mesma escala. A geração nova grava **disparidade métrica** quantizada,
   com `disparity_min`/`disparity_max`, `k_value` na escala de pixel da imagem
   ORIGINAL e `focus_disparity` em 1/m. São fórmulas que se parecem e não são a
   mesma — é exatamente a raiz descrita em `PLANO_REGERACAO_BOKEHNET.txt` §0.
   Sem um leitor novo, o retreino ou não roda, ou roda medindo outra coisa.

2. **O K não é reescalado para a resolução de treino.** `data.py` calcula o fator
   de resize, guarda em `escala` e **não o aplica ao K**. CoC em pixel escala com
   a resolução; disparidade não. O fator varia POR AMOSTRA (medidos 0,892 e
   0,821 [M], `bokehnet-regen/src/control/contract.py:k_at_resolution`), então o
   mapa que o modelo recebe não é o mapa que foi gravado, e o erro é diferente em
   cada amostra. É a regra 3 do `CONTRATO.md` violada no consumidor.

3. **`max_coc` é parâmetro.** No treino ele é `model.max_coc` no YAML, viaja pelo
   `DatasetRuntimeConfig` e chega em `defocus_from_depth`. O contrato novo o
   removeu de propósito da assinatura da função, porque foi como o experimento
   `kfix` rodou duas convenções de normalizador no mesmo lote.

E um quarto, de instrumentação, que é o que fez o problema custar caro:

4. **Nada no treino mede controlabilidade.** A degradação medida na fase 2
   (LVCorr LF-Bokeh +0,9059 → +0,4365 [M]) é **monótona no tempo de treino** e
   **invisível na loss** — a loss de flow matching caiu o tempo todo. Um probe de
   LVCorr a cada N steps custa ~1% do orçamento e teria pego o problema no
   primeiro milhar de steps em vez de no fim de 60K.

O resto desta auditoria enumera 26 achados (T1–T26), cada um com o que o paper
diz, o que o código faz, o impacto e a correção.

---

## 1. O que o paper especifica sobre a BokehNet

Extraído de `paper.pdf` (arXiv:2512.16923v3). Citações literais entre aspas.

### §3.2 — Shallow Depth-of-Field Synthesis

- Eq. 2: `D_def = K · |D − D_focus|`. **Crua, sem normalizador.**
- `D` é "the monocular depth map estimated from `I_aif` using an off-the-shelf
  depth estimator [7]" — [7] = **Depth Pro** (métrico, em metros).
- Condicionamento: "BokehNet concatenates the condition tokens of `I_aif` and
  `D_def` with the noisy latent tokens at each denoising step."
- Três fontes de supervisão: (a) sintético via renderer [43] = BokehMe;
  (b) ITW [19] com K pela Eq. 3 da EXIF; (c) LFDOF [52] + RealBokeh [57] com K
  pelo sweep de SSIM da Eq. 5.
- Eq. 4: `D_focus = median(D[M])`, `M` = máscara do BiRefNet [86].
- Eq. 5: `K* = argmax_K SSIM(R(I_aif, D; D_focus, K), I_real)`, "provided that
  its corresponding SSIM exceeds a predefined threshold".

### §3.3 — Bokeh-Shape Aware Synthesis

- "we append its tokens directly to the unified sequence" — a forma de abertura
  `A` é uma **terceira condição**.
- "we **freeze all original LoRA weights** and introduce a **new, trainable LoRA
  module**. Only this dedicated LoRA is fine-tuned to handle the shape
  conditioning."
- Dados: **PointLight-1K** (supplement C) + BokehMe estendido com kernel de
  forma (Eq. 6).

### §4.1 — Implementation details

| item | valor |
|---|---|
| backbone | FLUX.1-dev, LoRA [25] |
| esquema de condicionamento | "following [62, 63]" = OminiControl / OminiControl2 |
| LoRA rank | DeblurNet **128**, BokehNet **64** |
| batch | 1 por GPU × acumulação 8 × 4× RTX A6000 = **32 efetivo** |
| BokehNet | **(i) 40K steps sintético, (ii) 60K steps real** |
| dados | **~70K pares sintéticos** de [27] e [80]; **~26K reais** de ITW [19], RealBokeh [57], LFDOF [52] |
| inferência | 28 passos de denoise; tiling da §3.5 é **de inferência** |

### Supplement B.2 — dados da BokehNet

- Pool sintético: candidatos de [80] e EBB [27], filtrados por variância do
  Laplaciano → "**approximately 1.7K sharp images**". (1,7K imagens → ~70K pares
  ⇒ ~40 combinações `(D_focus, K)` por imagem [I].)
- Reais: 26K = 13K do ITW já verificados + 13K novos, "focus-consistent series
  captured with varying apertures, containing **2 to 4 images per set**".
- "we then optimized the parameter K using simulator [43]" seguido de "we applied
  a Structural Similarity (SSIM) threshold to filter out sub-optimal results".

### O que o paper NÃO publica

Resolução de treino, otimizador, learning rate, scheduler, warmup, `K_min`/`K_max`
da Eq. 5, o limiar de SSIM, `gamma`/`defocus_scale`/`highlight` do renderer, a
proporção entre rotas na fase 2, e o normalizador do mapa (a Eq. 2 é crua).
Onde o paper cala, a autoridade é `Inference_bokehNet.py`; onde os dois calam, é
decisão nossa e vai **declarada**.

### O que a inferência oficial fixa

`third_party/Genfocus/Inference_bokehNet.py` e `demo.py`:

```python
MAX_COC   = 100.0                                   # :20
disp      = 1.0 / depth_metrico                     # :94   ← DISPARIDADE
disp_focus= median(disp[mask])                      # :118  ← mediana NA disparidade
cond_map  = clip(|K*(disp - disp_focus)| / MAX_COC, 0, 1).repeat(3,1,1)   # :138-140
k_value   default = 15.0                            # :53
prompt    = "an excellent photo with a large aperture"
guidance_scale = 1.0 ; num_inference_steps = 28
cond_img  = Condition(clean_input, "bokeh")                              # [-1,1] via preprocess
cond_dmf  = Condition(cond_map, "bokeh", [0,0], 1.0, No_preprocess=True) # [0,1] CRU
```

E em `Genfocus/pipeline/flux.py`: `c_timesteps = 0`, `c_guidances = 1.0`,
`group_mask[2:,2:] = diag(1)`, `TILE_SIZE = 32` tokens = **512 px**,
`mu = calculate_shift(image_seq_len da imagem INTEIRA)` antes do tiling (`:624`).

---

## 2. O que já está CERTO — não mexer

Cada linha abaixo foi conferida contra o código oficial. Estão aqui para que uma
revisão futura não as "corrija".

| # | Item | Onde, no nosso código | Confere com |
|---|---|---|---|
| V1 | LoRA rank 64 no estágio bokeh | `config.py:ModelConfig.bokeh_lora_rank`, `lora_rank_for()` | §4.1 |
| V2 | 343 módulos LoRA, `proj_out` de topo FORA | `backbone.py:LORA_TARGET_MODULES` (regex) + trava `expected_lora_modules` | `bokehNet.safetensors`: 686 tensores ÷ 2 = 343 [M] |
| V3 | Batch efetivo 32 | `1 × grad_accum 8 × 4 GPUs` nos YAMLs | §4.1 |
| V4 | Currículo 40K sintético → 60K real, com LoRA da fase 1 e optimizer/scheduler FRESCOS | `trainer.py:_train_loop` (`init_lora_path`), `train.py --init-lora` | §4.1 + §3.2(a) chamar a fase sintética de "pretrain" |
| V5 | Prompt `"an excellent photo with a large aperture"` | `train.py:BOKEH_PROMPT` | `Inference_bokehNet.py`, `demo.py:244` |
| V6 | `bokeh_train_guidance = 1.0` (branch principal e texto) | `config.py`, `backbone.py` | `Inference_bokehNet.py` passa `guidance_scale=1.0` |
| V7 | `cond_train_guidance = 1.0` | idem | `flux.py:616` `c_guidances = torch.ones` |
| V8 | Timestep **0** em cada branch de condição | `backbone.py:forward_train_step` | `flux.py:615` `c_timesteps.append(zeros)` |
| V9 | LoRA só nas condições (`main_adapter=None`) | `lora_on_main: false` | `Inference_bokehNet.py` chama `generate()` sem `main_adapter` |
| V10 | 2 condições: AIF em `[-1,1]`, defocus em `[0,1]` **cru** | `models.py:BokehNet.make_train_batch` | `No_preprocess=True` só na segunda condição |
| V11 | Mapa replicado em 3 canais | `data.py:_defocus_pil_to_3ch` | `.repeat(3,1,1)` na inferência |
| V12 | `img_ids` das condições **sem** `position_delta` | `backbone.py` usa `_prepare_latent_image_ids` igual para as 3 | `Condition(img)` com delta `None` e `Condition(map, ..., [0,0])` |
| V13 | `group_mask` diagonal entre condições (AIF e mapa mutuamente cegos) | `backbone.py:forward_train_step` | `flux.py:656-658`, incondicional |
| V14 | `alpha = rank` (escala LoRA efetiva 1) | `backbone.py:_inject_lora` | `specify_lora` força `scaling = 1` na inferência **e** no treino |
| V15 | σ logit-normal com shift de `calculate_shift`; alvo `v* = ε − x₀` | `backbone.py:sample_sigma`, `forward_train_step` | rectified flow do FLUX; `flux.py:625` |
| V16 | Crop de treino 512² = `TILE_SIZE` da inferência | `image_size: 512` | `flux.py:486` `TILE_SIZE=32` tokens × 16 px |
| V17 | Condição encodada com `latent_dist.sample()` | `backbone.py:encode_image_to_tokens` | `flux.py:44` `encode_images` também usa `.sample()` |
| V18 | Fase 1 = **só rota A**; rota B (Flickr/EXIF) é REAL e vai na fase 2 | `train_bokeh_synth_*.yaml` | §3.2: (b) ITW é "real bokeh images" |

---

## 3. Os achados — T1 a T26

Ordenados por severidade. `arquivo:símbolo` refere-se a esta pasta
(`retreinar-bokeh/`), cujo conteúdo é cópia de `retreinar-deblur/`.

---

### T1 — CRÍTICO — O dataloader não fala o contrato novo

**Paper/contrato.** `bokehnet-regen/reference/CONTRATO.md` fixa
`metric_disparity_official_v1`:

```
z          = depth_pro(aif)               # METROS
disp       = 1/z                          # 1/m
focus_disp = median(disp[mask])
defocus    = clip(|K*(disp - focus_disp)| / 100.0, 0, 1)
```

e a geração grava: `depth/<id>.png` uint16 **linear em disparidade** com
`disparity_min`/`disparity_max`/`depth_h`/`depth_w`, `k_value` **na escala de
pixel da imagem original**, `focus_disparity` em 1/m, `image_h`/`image_w`,
`scene_id`, `route`, `k_source`, `is_k_censored`, `is_valid_for_control`,
`calibration_ssim`, `focus_*`. **O mapa de defocus não é gravado** — é derivado no
dataloader, de propósito (`bokehnet-regen/src/dataio/sample.py`, docstring).

**Código.** `genfocus_train/data.py`:

- `BOKEH_REQUIRED_COLUMNS = {"aif", "bokeh", "depth", "k", "s1"}`
- `defocus_from_depth(depth01, k, s1, max_coc)` = `clip(|k·(depth01 − s1)|/max_coc, 0, 1)`
  com `depth01` = uint16/65535 de **profundidade métrica normalizada por imagem** e
  `s1` na mesma escala.

**Impacto.** As duas expressões têm a mesma forma e vivem em espaços
diferentes, e a troca de espaço é uma **reparametrização não linear** — portanto
**nenhuma constante multiplicativa em K corrige**. Isso costuma ser dito e não
medido, então foi medido [M], numa cena realista (90% do conteúdo em 1–20 m, 10%
de fundo a 200–10.000 m, foco em 3 m, K=15):

| grandeza | valor |
|---|---|
| melhor `α` que aproxima `K·\|D01−s1\|` de `K·\|Δ(1/z)\|` (mínimos quadrados) | 0,504 |
| **erro relativo residual com esse α ótimo** | **0,926** |
| Pearson entre os dois mapas | 0,341 |
| Spearman (a ORDEM se preserva?) | 0,872 |

E o erro não é sequer monotônico na profundidade — a razão entre o CoC que a
disparidade pede e o que a profundidade normalizada entrega (já com o `α` ótimo):

| z | 1,5 m | 5 m | 10 m | 20 m | 200 m | 10.000 m |
|---|---|---|---|---|---|---|
| razão | **4408×** | 1322× | 661× | 331× | 33× | 0,66× |

É por isso que a busca binária por K da avaliação absorve a diferença de
**escala** e não a de **forma** — a frase do `PLANO_REGERACAO_BOKEHNET.txt` §0,
aqui virando número. Em profundidade linear normalizada o fundo distante ocupa
quase toda a faixa e a disparidade quase não varia lá; foi o que deixou 24,7% das
amostras da rota B com a cena útil em **menos de 256 níveis** de 65535 [M].

Além disso as colunas `depth`/`k`/`s1` **não vão existir** no dataset novo: ou o
treino levanta `_validate_columns`, ou alguém "adapta" renomeando coluna e
reintroduz o defeito.

Travado por `test_nenhuma_constante_K_converte_profundidade_normalizada_em_disparidade`.

**Sub-achado T1-b — a mina do `/65535` em float.** Levantado por revisão externa
e confirmado: `_defocus_to_float_pil` dividia por 65535 **incondicionalmente**.
Se uma profundidade do formato novo chegasse como float — PIL modo `'F'`,
`datasets` devolvendo o PNG já convertido, ou disparidade métrica crua — ela
seria esmagada para ~1e-5, o mapa sairia praticamente zero, o modelo aprenderia
"nunca borrar", e **nada levantaria erro**. Corrigido: a divisão passou a ser
condicionada ao **dtype** (inteiro divide, float é assumido já em `[0,1]` e
**validado**), tanto em `control.decode_disparity_u16` quanto no caminho
aposentado. Distinguir pelo dtype e não pelo valor é deliberado — um mapa uint16
legítimo pode ter máximo 1.

Duas fixtures de teste existentes **estavam exercitando essa mina**: `_fake_depth_u16`
e a coluna legada do `test_data_pipeline.py` produziam `float32` em 0..65535 com
nome de uint16. O guard novo as pegou; foram corrigidas para uint16 de verdade.

**Correção.** Fonte de defocus nova, `metric_disparity`, com uma única
implementação importada de `genfocus_train/control.py` — que é a cópia read-side
do `bokehnet-regen/src/control/contract.py`. Feito nesta pasta: ver
`MUDANCAS_CODIGO.md` §1 e §2.

---

### T2 — CRÍTICO — O K não é reescalado para a resolução do crop

**Contrato.** Regra 3 do `CONTRATO.md`: *"Toda quantidade em pixel carrega a
resolução em que foi medida. Se o dataloader reescala e recorta, o fator
`512/min(H,W)` — que varia por amostra — tem que ser aplicado, ou o mapa que o
modelo recebe não é o mapa que foi gravado."*

**Código.** `data.py:prepare_aligned_bokeh`:

```python
new_w, new_h, box, flip, full_seq_len = _plano_geometrico(...)
scale = new_w / float(w)          # <- calculado
...
defocus_arr = defocus_from_depth(np.asarray(def_c), k, s1, max_coc)   # <- k CRU
```

`scale` só é devolvido no dict de geometria, para o `geo_cond`. **Nunca multiplica
o K.**

**Impacto.** `CoC_px = K·|Δdisp|` é uma quantidade em pixel. Reduzir o lado menor
de 1500 para 512 divide o raio de borrão real por 2,93, mas o mapa continua
dizendo o raio da imagem cheia.

Quanto isso erra, por resolução de origem (fator = `512/min(H,W)`) [M]:

| origem | fator | o K gravado é… |
|---|---|---|
| 1500×2000 (RealBokeh 3MP) | 0,341 | **2,93×** grande demais |
| 683×1024 (LFDOF) | 0,750 | 1,33× |
| 624×832 | 0,821 ← medido no contrato | 1,22× |
| 574×766 | 0,892 ← medido no contrato | 1,12× |

O ponto **não** é a magnitude — é que o fator **varia por amostra**, de 1,12× a
2,93×. Um viés constante o modelo absorveria numa escala aprendida; um viés que
muda a cada amostra não, e a saída racional passa a ser **ignorar o K e prever a
média**. É consistente com o LVCorr medido despencando exatamente na fase em que
entram fontes de resolução heterogênea.

**Correção.** `control.k_at_resolution(k, image_hw, dst_short_side)` antes de
montar o mapa, com `image_hw` vindo do metadado (a resolução em que o K foi
medido), não do array de profundidade. Feito: `MUDANCAS_CODIGO.md` §2.
Teste que trava: `tests/test_control_contract.py::test_k_pos_crop`.

---

### T3 — ALTO — Profundidade reamostrada com BILINEAR no treino, NEAREST na geração

**Código.** `data.py:prepare_aligned_bokeh` → `def_pil.resize(..., Image.BILINEAR)`.
`bokehnet-regen/src/dataio/encoding.py:resize_depth_nearest` → vizinho mais
próximo, com a justificativa explícita: *"interpolar profundidade atravessa
descontinuidade e inventa um plano intermediário que não existe na cena. Numa
borda de objeto, a média entre 1 m e 20 m é 10,5 m — uma superfície fantasma que
o renderer depois borra como se fosse real."*

**Impacto.** Duas políticas na mesma cadeia. O alvo (bokeh) foi renderizado a
partir da profundidade com bordas duras; a condição que o modelo recebe tem as
bordas suavizadas. O modelo aprende que a borda do mapa é uma rampa e a borda do
alvo é um degrau — ruído sistemático justo nos pixels de oclusão, que é onde a
qualidade do bokeh se julga.

**Correção.** NEAREST no dataloader também, e o mesmo argumento no comentário.
[A] o ganho não foi medido; o que se ganha com certeza é **uma política só**.

---

### T4 — ALTO — `max_coc` é parâmetro configurável

**Código.** `ModelConfig.max_coc: float = 100.0` → `DatasetRuntimeConfig.max_coc`
→ `defocus_from_depth(..., max_coc=...)`. Editável em YAML.

**Contrato.** `bokehnet-regen/src/control/contract.py:defocus_map` **removeu**
`max_coc` da assinatura, com a justificativa: *"Como parâmetro com default, um
override parcial gravaria `max_coc: 100.0` nos metadados ao lado de um mapa
normalizado por outro valor — e a proveniência mentiria sem que nada denunciasse."*

**Impacto.** É o mecanismo exato do `kfix`, que usou `max_coc = 10,5107` (P82,66
do `coc_p99_px` da rota B) para "fazer o mapa da rota B ocupar `[0,1]` como o da
rota C". Resultado medido: LF-Bokeh +0,8288 **e** RealBokeh +0,4832 e RealDOF
−0,4599 [M] — duas convenções no mesmo lote. E com `max_coc = 10,5107`,
`coc_p99 ≥ max_coc` em 2.018/11.635 = 17,3% da rota B [M]: o mapa já saturava.

Aceitar que a rota B ocupe `[0, 0,05]` e a rota C `[0, 0,4]` é o ponto: **a
diferença é física e é o que o modelo tem de aprender.**

**Correção.** `MAX_COC = 100.0` vira constante de módulo em
`genfocus_train/control.py`. O campo do YAML sobrevive apenas como **asserção**:
se presente e diferente de 100.0, o load falha.

---

### T5 — ALTO — Convenções aposentadas continuam a um typo de distância

**Código.** `DEFOCUS_SOURCES = ("recompute", "column", "kfix")`, todos alcançáveis
por YAML.

- `"column"` = o defeito **D1** (mapa normalizado por imagem, `dm/dm.max()`, que
  apaga o K algebricamente).
- `"kfix"` = normalização **por rota** (`max_coc` calibrado), o D1 com
  granularidade mais grossa.

**Impacto.** Uma linha de YAML separa um run de 60K steps da convenção que já
produziu o pior LPIPS entre as variantes reais. O projeto já foi mordido por isso
uma vez.

**Correção.** Manter os caminhos (são histórico reprodutível) mas exigir
`allow_retired_defocus_sources: true` explícito no YAML, com aviso gritado no log
e gravado no `run_metadata.json`. Default: recusa.

---

### T6 — CRÍTICO — Sem split de validação, `best.pt` segue a loss de treino

**Código.** Nenhum YAML de bokeh define `val_datasets`. Em `trainer.py`:

```python
referencia = val_loss if val_loss is not None else (loss_val if val_loader is None else None)
```

Sem `val_datasets`, `best.pt` é escolhido pela loss de **um micro-batch de treino**,
com σ e ruído sorteados. Isso não é sinal de nada — foi exatamente o diagnóstico
C10 do DeblurNet, corrigido lá e **não aplicado** ao bokeh porque os YAMLs de
bokeh não ganharam `val_datasets`.

**Agravante.** O gate de aceitação nº 20 do `PLANO_REGERACAO_BOKEHNET.txt`
("nenhum `scene_id` aparece em treino e validação") **não é verificável** sem um
split por cena materializado, e o dataset novo grava `scene_id` justamente para
isso. Split por linha, num dataset em que a mesma cena aparece com 2 a 21
aberturas, vaza a cena inteira.

**Correção.** `val_datasets` obrigatório no estágio bokeh + split por `scene_id`
lido do manifesto do release, nunca sorteado no dataloader.

---

### T7 — ALTO — Sem filtro por `is_valid_for_control` / `is_k_censored`

**Código.** `_filter_by_calibration_ssim` filtra só `calibration_ssim`. Não existe
filtro por `is_valid_for_control` nem por `is_k_censored`.

**Contrato.** `sample.py:metadata()` grava
`is_valid_for_control = is_valid_for_control and not is_k_censored`.
`is_k_censored` marca a amostra cujo sweep da Eq. 5 **bateu na borda do bracket** —
o `K*` gravado é o limite da busca, não o argmax. Gate nº 8 do plano de
regeração: *"nenhuma amostra com k no teto exato do sweep"*.

**Impacto.** K censurado é rótulo com viés conhecido e direção conhecida
(subestima ou superestima sistematicamente). Treinar com ele é ensinar o viés.

**Correção.** `require_valid_for_control: true` por default; contagem de descartes
por motivo impressa no início do run (o histograma que denuncia fallback novo).

---

### T8 — MÉDIO — `min_calibration_ssim: 0.6` foi calibrado no dado velho

**Paper §3.2(c).** *"provided that its corresponding SSIM exceeds a predefined
threshold"*. O valor não é publicado; 0,6 é escolha nossa, registrada em
`DECISOES_FASE2.md` §4 — **medida sobre a distribuição antiga**, gerada com um
renderer que nunca era o BokehMe (o pipeline antigo caía num gaussiano de kernel
51 px [M]) e com o K na convenção errada.

**Correção.** Recalibrar sobre a distribuição nova de `calibration_ssim` antes de
travar, e registrar o percentil escolhido, não só o valor absoluto.
Enquanto não houver a distribuição nova, o YAML deve deixar `null` e falhar
explicitamente em vez de herdar 0,6 por inércia.

---

### T9 — MÉDIO — Nenhum teto de níveis por cena

**Medido.** `PLANO_EXECUCAO.md`: *"25% das amostras vindo de 6,2% das cenas"*, e o
paper diz *"2 to 4 images per set"* (supplement B.2). Sem teto, o dataset efetivo
é dominado por um punhado de cenas com 21 aberturas.

**Impacto.** Duplo. (i) O modelo vê a mesma cena dezenas de vezes por época —
memorização, não generalização óptica. (ii) A distribuição de K fica enviesada
para as cenas ricas, e é a distribuição de K que o controle aprende.

**Correção.** `max_levels_per_scene` (default 4, literal do supplement), com
sorteio do subconjunto de níveis **por época** e não fixo — assim nenhum nível se
perde ao longo de 60K steps, mas nenhuma cena domina um batch. Muda o tamanho do
dataset de ~20K para ~13K, o que é o número do paper.

---

### T10 — MÉDIO — A mistura entre rotas é um `concatenate_datasets` sem peso

**Código.** `_load_hf_sources_bokeh` concatena e o `shuffle` do DataLoader amostra
uniformemente por linha. A proporção efetiva é o tamanho relativo dos shards.

**Paper.** Não especifica a proporção. Mas B e C ocupam faixas de defocus
diferentes por física (`[0, 0,05]` × `[0, 0,4]` [M]), e B ensina **aparência
óptica real com AIF vinda da DeblurNet** — que é a distribuição da inferência —
enquanto C ensina **variação de K dentro da cena**, que é o controle.

**Impacto.** Uma proporção acidental de 1:10 e outra de 10:1 treinam modelos
diferentes, e a ablação da Tab. 6 (a+b+c melhor que a+c) não se reproduziu no
nosso lado (hoje a+c ganha nas quatro mesas [M]). Não dá para atribuir isso ao
método enquanto a proporção for acidente.

**Correção.** `route_weights` explícito no YAML (default: proporcional, para não
mudar duas coisas de uma vez), com as contagens por rota impressas no início do
run e gravadas no `run_metadata.json`.

---

### T11 — ALTO — Nada mede controlabilidade durante o treino

**Medido, e é a razão desta pasta existir:**

| LVCorr | LF-Bokeh | RealBokeh | RealDOF |
|---|---|---|---|
| fase 1, só sintético | **+0,9059** | +0,8609 | +0,9247 |
| pesos oficiais do paper | +0,8868 | +0,8498 | +0,9644 |
| nosso, fase 2, a+b+c | **+0,4365** | +0,8257 | +0,1324 |

E a degradação é **monótona no tempo de treino**: na variante só-rota-c, no
RealDOF, +0,2509 em 10K steps → +0,0281 em 20K → −0,1776 em 30K → −0,2061 em 60K
[M]. **A loss de flow matching caiu o tempo todo.**

**Impacto.** O único instrumento do treino é cego para o modo de falha que
efetivamente ocorreu. 60K steps × 4 GPUs foram gastos para descobrir depois.

**Correção — a melhoria de maior retorno desta auditoria.** Probe de LVCorr
periódico:

- conjunto fixo de 8 AIF (fora do treino, `scene_id` disjunto), plano de foco
  fixo por imagem, **`K ∈ {1, 5, 10, 15}`** — a grade do harness de avaliação
  deste projeto (`vision-pipeline/evaluation/eval_bokeh_synthesis.py:35`), e
  **não** a da Fig. 12. A Fig. 12 mostra `{0,5,10,15}` mas rotula o primeiro
  como "K=0(Input)": é a entrada exibida, não uma geração. Com `K=0` o mapa é
  todo zero, a saída tende à AIF, e esse ponto de nitidez máxima infla a
  correlação — o probe deixaria de ser comparável com a tabela que ele vigia;
- 28 passos, `NO_TILED_DENOISE=True`, seed fixa, `guidance_scale=1.0` — a
  configuração da inferência oficial;
- métrica: Pearson entre `K` e a variância do Laplaciano da saída, como em
  §4.1(ii);
- custo [A]: 32 gerações a 512², ~10 s cada ≈ 5 min por probe. A cada 5.000 steps
  num run de 60K = 12 probes ≈ 1 h ≈ **~1% do orçamento**.

Critério de parada declarado ANTES de rodar: se LVCorr cair abaixo do valor da
fase 1 menos 0,10 em dois probes consecutivos, o run para e a causa é
investigada, em vez de rodar até o fim.

---

### T12 — MÉDIO — Nenhuma estatística do sinal de controle é logada

**Código.** O `logger.log` do treino emite `loss`, `loss_ema`, `lr`, `best_loss`,
`peak_vram_gb`. Nada sobre o mapa de defocus.

**Impacto.** O defeito D1 (`max(defocus) == 65535` em **todas** as amostras, com
k de 33 a 195 [M]) teria aparecido no **primeiro step** num histograma de
`defocus.max()`. Levou meses.

**Correção.** Logar por batch, custo desprezível: `defocus_mean`, `defocus_max`,
`defocus_p99`, `frac_saturado` (fração de pixels em 1,0), `k_value` (média e
desvio na janela), e contagem por rota. Mais um dump de 8 tríades
(AIF, mapa, alvo) como imagem no wandb a cada N steps — o teste mais barato
contra "o mapa está alinhado com a imagem certa".

---

### T13 — MÉDIO — `sigma_mu_source: crop` × inferência que usa a imagem inteira

**Inferência.** `flux.py:624` calcula `mu = calculate_shift(image_seq_len)` com
`image_seq_len = latents.shape[1]` da **imagem inteira**, antes do tiling. Todo
tile herda esse cronograma.

**Treino.** `sigma_mu_source: "crop"` deriva `mu` dos 1.024 tokens do crop 512².

**Ordem de grandeza.** Aritmética [M] (recalculada); as quatro constantes são
[I] — vêm do `scheduler_config.json` do FLUX.1-dev (`base_image_seq_len 256`,
`max_image_seq_len 4096`, `base_shift 0,5`, `max_shift 1,15`) e são conferidas em
tempo de execução por `backbone._verificar_mu_bate_com_diffusers()`, que compara
a nossa fórmula vetorizada com o `calculate_shift` do diffusers e falha alto se
divergirem:

| entrada | tokens | `mu` | `exp(mu)` |
|---|---|---|---|
| crop 512² (o que o treino usa) | 1.024 | 0,630 | 1,88 |
| foto 1024×683 (DPDD) | 2.688 | 0,912 | 2,49 |
| foto 2000×1500 (RealBokeh 3MP) | 11.750 | 2,446 | 11,30 |
| foto 2048×1536 | 12.288 | 2,537 | 12,64 |

A distribuição de σ do treino fica muito mais concentrada em ruído baixo do que a
que a inferência percorre.

**Por que importa mais no bokeh que no deblur.** As entradas da BokehNet são fotos
reais grandes (RealBokeh 3MP, LFDOF), não crops de 1024×688 — a diferença de
`image_seq_len` é maior. E é um desbalanceamento de **densidade**, não
fora-de-domínio: o treino cobre `(0,1)` inteiro.

**Correção.** `full_seq_len` **já é calculado e já viaja no batch** (C4 do
DeblurNet). Trocar o default do estágio bokeh para `full_image` custa uma linha.
Recomendação: `full_image`, declarado, com smoke antes — é a escolha que
reproduz a inferência, que é a regra do projeto onde o paper cala.

---

### T14 — MÉDIO — `scale_mode: short_side` × tiles nativos da inferência

**Inferência.** Com `--long_side 0` (o default), `resize_and_pad_image` não
reescala, e o tiling processa tiles de **512 px nativos**.

**Treino.** `short_side` leva o lado menor a 512 e recorta — numa foto 2000×1500
isso reduz o conteúdo em 0,341×.

**A tensão, honestamente:**

| | `short_side` | `native` |
|---|---|---|
| escala de pixel = a da inferência | ✗ | ✓ |
| o crop enxerga a cena inteira (variação de profundidade dentro do crop) | ✓ | ✗ — 512² de uma foto 3MP é **1/11,4** do quadro [M]; muitos crops ficam com mapa quase constante |
| bit a bit igual ao comportamento anterior | ✓ | ✗ |

**Recomendação.** Manter `short_side` como default (não mudar duas coisas ao mesmo
tempo — T2 já muda a escala do K) e rodar `native` como braço de A/B de 10K steps
na fase 1, medindo com o probe do T11. Registrar a decisão com o número, não com
a intuição. Marcado [A]: nenhum dos dois foi medido neste projeto.

---

### T15 — MÉDIO — Fase 2 reinicia o LR em 10× o valor em que a fase 1 terminou

**Código.** `scheduler: cosine, warmup 500, min_lr_ratio 0.1, lr 1e-4`. A fase 1
termina em `1e-5`. A fase 2 começa com optimizer/scheduler frescos e sobe de novo
até `1e-4`.

**Hipótese [I], não medida.** Um salto de 10× no LR no começo da fase que
introduz dados reais é um candidato direto para o esquecimento catastrófico do
controle de K medido no T11 — o LoRA que aprendeu a modular o CoC pelo mapa é
reescrito antes de a fase 2 consolidar qualquer coisa.

**Correção proposta (declarada como desvio, o paper não especifica LR).**
Braço A: como está. Braço B: `lr: 5e-5` na fase 2 com warmup de 2.000 steps.
Decidir pelo probe do T11 aos 10K steps. Custo: 2 × 10K steps.

---

### T16 — MÉDIO — Fase 2 sem nenhuma repetição do sintético

**Paper §4.1, literal.** *"(i) 40K steps on synthetic data, and (ii) 60K steps on
real data."* Nossa config faz exatamente isso, e está certo como leitura literal.

**Mas.** A Tab. 6 mede (a)+(b)+(c) como a melhor combinação, e o dado sintético é
a **única fonte com variação densa e limpa de K por cena** — é dele que veio
LVCorr +0,9059. As rotas reais têm 2 a 4 níveis por cena (B tem 1, porque o K vem
da EXIF de uma foto só).

**Proposta, declarada como desvio.** Braço C: fase 2 com `synthetic_replay_fraction:
0.15` — 15% dos batches sorteados da rota A. Testa diretamente a hipótese de
esquecimento. Se LVCorr da fase 2 voltar para perto de +0,88 sem perder LPIPS, é
um achado nosso e vai para o texto; se não, a leitura literal do paper fica
confirmada por medição, o que também vale.

O braço primário continua sendo o literal (b+c). Não trocar o primário por uma
hipótese nossa.

---

### T17 — MÉDIO — §3.3 (controle de forma de abertura) não existe no treino

**Paper §3.3.** Terceira condição `A` (imagem da forma), **congelar todo o LoRA
original** e treinar um **LoRA novo e dedicado**.

**Código.** `models.py` diz, no docstring: *"NÃO IMPLEMENTADO: aperture-shape
control (paper §3.3)"*. `backbone.py` injeta **um** adapter (`ADAPTER_NAME =
"default"`) e `forward_train_step` usa o mesmo nome para todas as condições.

**Impacto.** É uma das três contribuições anunciadas no abstract e a Fig. 1(c) e a
Fig. 7 do paper. Sem ela a reprodução fica incompleta.

**Correção (feita nesta pasta, lado treino).** `backbone.py` ganha um segundo
adapter opcional; `forward_train_step` aceita **um nome de adapter por condição**;
`BokehNet.make_train_batch` aceita `shape_image` como terceira condição.
O que **falta** e é do lado de dados: PointLight-1K (supplement C) e o BokehMe
estendido com kernel de forma (Eq. 6). Ver `PLANO_RETREINO_BOKEHNET.md` fase 3.

---

### T18 — BAIXO — `set_epoch` nunca é chamado no sampler distribuído

**Código.** `_cycle(dataloader)` itera para sempre; com `accelerator.prepare` o
loader ganha um `DistributedSampler` cujo `set_epoch` nunca é chamado.

**Impacto.** A permutação é idêntica em todas as épocas. Com ~26K amostras reais e
60K steps × 32 = 1,92M amostras vistas, são ~74 épocas com a **mesma ordem** e o
mesmo particionamento entre GPUs. Reduz a diversidade efetiva de combinação
dentro do batch acumulado.

**Correção.** Contador de épocas em `_cycle`, chamando `set_epoch` quando o
sampler expõe o método. Cinco linhas.

---

### T19 — BAIXO — `sigma_mu_source` é descartado com aviso a cada run

**Código.** `trainer.py:_dataset_runtime` monta `"sigma_mu_source": ...` mas
`DatasetRuntimeConfig` não tem o campo; o guard filtra e avisa.

**Impacto.** Nenhum, hoje — o `full_seq_len` vem por outro caminho. Mas um aviso
que aparece em todo run treina o operador a ignorar avisos, e o guard existe
justamente para que um aviso signifique alguma coisa.

**Correção.** Adicionar o campo ao `DatasetRuntimeConfig`. Uma linha.

---

### T20 — BAIXO — `run_validation` não repassa os pesos de oclusão

**Código.** `run_validation` chama `_make_train_batch(model, stage, batch)` sem
`occlusion_lambda`. Inofensivo no deblur (`lambda = 0`); no bokeh com
`occlusion_lambda > 0` a `val_loss` mede uma perda diferente da treinada, e é ela
que escolhe o `best.pt`.

**Correção.** Repassar os quatro parâmetros. (E, se `geo_condition` continuar
desligado no retreino, isso vira inofensivo por outro caminho — mas o
descasamento fica.)

---

### T21 — BAIXO — `model.vae_subfolder` / `transformer_subfolder` não são lidos

Estão em todo YAML de bokeh e ninguém os consome (`FluxPipeline.from_pretrained`
usa os defaults). Mudar no YAML não tem efeito e não avisa. Remover dos YAMLs ou
passar de fato.

---

### T22 — OBSERVAÇÃO — A AIF do treino sintético não passa pela DeblurNet

Na inferência, a entrada da BokehNet é **sempre** saída da DeblurNet. Na fase 1,
as AIF são imagens reais nítidas (EBB/[80] filtradas por Laplaciano). Há um gap de
distribuição.

O paper resolve isso pela rota (b), que por construção usa AIF produzida pela
DeblurNet — *"training on them inherently aligns with our single-image refocusing
pipeline"* (§3.2(b)). E o supplement C passa o **PointLight-1K** pela DeblurNet
(passo 4). Ou seja: os autores fizeram isso onde acharam que importava e não
fizeram na rota (a).

**Sem ação.** Registrado para que a pergunta não volte. Reforça a prioridade da
rota B com a **nossa** DeblurNet (passo 5 do `PLANO_EXECUCAO.md`).

---

### T23 — OBSERVAÇÃO — 3 canais idênticos no mapa gastam 2/3 da banda do VAE

O mapa é escalar e entra replicado em RGB. É desperdício, e é **exatamente o que a
inferência oficial faz**. Não mexer. Registrado porque parece uma otimização óbvia
e não é: qualquer mudança aqui quebra a compatibilidade com
`Inference_bokehNet.py`.

---

### T24 — OBSERVAÇÃO — `latent_dist.sample()` adiciona ruído do VAE à condição

`encode_image_to_tokens` amostra do posterior do VAE em vez de usar a moda. Isso
injeta ruído no **sinal de controle** a cada step. A inferência oficial
(`flux.py:44`) também usa `.sample()`, então treino e inferência casam.
Não mexer. Se algum dia for medido que `.mode()` melhora, é desvio declarado dos
dois lados.

---

### T25 — OBSERVAÇÃO — Crop aleatório e assinatura óptica fora de eixo

Bokeh de lente real tem assinatura dependente da posição no quadro (cat-eye,
vinhetagem). O crop aleatório apresenta uma região fora de eixo como se fosse o
centro. A inferência com tiling faz o mesmo, então o descasamento treino↔inferência
é nulo; o que se perde é a chance de o modelo **aprender** a dependência posicional.
Sem ação — corrigir exigiria condicionar em coordenada, o que o paper não faz.

---

### T26 — OBSERVAÇÃO — Volume de dados: 1,7K imagens → ~70K pares sintéticos

Supplement B.2 diz "approximately 1.7K sharp images"; §4.1 diz "~70K synthetic
pairs". A razão é ~40 pares por imagem, isto é, ~40 combinações `(D_focus, K)`
sorteadas por imagem [I]. Nossa rota A precisa mirar essa razão, não só o total —
70K pares vindos de 20K imagens com 3 combinações cada é um dataset diferente,
com muito menos sinal sobre "a mesma cena com K diferente", que é o que ensina
controle.

Isso é do lado de dados (`bokehnet-regen`), mas o treino é quem sofre, então fica
registrado aqui e vai para o `PLANO_RETREINO_BOKEHNET.md` como pré-condição.

---

## 4. Tabela de fechamento

| # | Severidade | Onde vive a correção | Status nesta pasta |
|---|---|---|---|
| T1 | CRÍTICO | `control.py` + `data.py` | **implementado**; o default dos DOIS dataclasses é `metric_disparity` e os três caminhos antigos exigem porta explícita |
| T1-b | ALTO | `/65535` condicionado ao dtype | **implementado** (achado de revisão externa; 2 fixtures mentirosas corrigidas junto) |
| T2 | CRÍTICO | `control.k_at_resolution` no dataloader | **implementado** |
| T3 | ALTO | `data.py` (NEAREST) | **implementado** |
| T4 | ALTO | `control.MAX_COC` constante + asserção no config | **implementado** |
| T5 | ALTO | `config.py` (`allow_retired_defocus_sources`) | **implementado** |
| T6 | CRÍTICO | YAMLs + split por `scene_id` | **implementado**; o `train check` REPROVA `val_datasets` sem split por cena. Com release em árvore o split vem do `split.json` materializado, e `carregar_split_por_cena` passou a ler esse formato (L1) |
| T7 | ALTO | `data.py` (filtros) | **implementado** |
| T8 | MÉDIO | recalibrar limiar | **bloqueado** até existir a distribuição nova; default virou `null` e o `train check` avisa |
| T9 | MÉDIO | `max_levels_per_scene` | **implementado** |
| T10 | MÉDIO | `route_weights` + log | **implementado** |
| T11 | ALTO | probe de LVCorr | **implementado**: `genfocus_train/probe.py`, hook no trainer com alarme declarado, `scripts/build_probe_set.py`, e o `train check` REPROVA probe sem conjunto. Não rodado em GPU |
| T12 | MÉDIO | log do sinal de controle | **implementado** |
| T13 | MÉDIO | `sigma_mu_source: full_image` | **trocado nos YAMLs de bokeh**, declarado no comentário do próprio arquivo. O default da dataclass segue `crop` (é o do DeblurNet, que está treinando) |
| T14 | MÉDIO | A/B `native` × `short_side` | **config pronto**, a medir |
| T15 | MÉDIO | LR da fase 2 | **braço B em YAML**, a medir |
| T16 | MÉDIO | `synthetic_replay_fraction` | **braço C em YAML**, a medir |
| T17 | ALTO | 2º adapter LoRA + 3ª condição | **implementado** (lado treino) |
| T18 | BAIXO | `set_epoch` | **implementado** |
| T19 | BAIXO | campo no `DatasetRuntimeConfig` | **implementado** |
| T20 | BAIXO | `run_validation` | **implementado** |
| T21 | BAIXO | limpar YAML | **implementado** |
| T22–T26 | — | observações | registradas, sem ação |

### Correções feitas nesta auditoria depois de reverificar as contas

Três erros meus, encontrados ao refazer os cálculos, e o que cada um teria
custado:

| # | erro | consequência se tivesse passado | correção |
|---|---|---|---|
| E1 | O probe usava `K ∈ {0,5,10,15}`, lido da Fig. 12. O harness de avaliação deste projeto usa `[1,5,10,15]` (`vision-pipeline/evaluation/eval_bokeh_synthesis.py:35`), e é com ele que os números +0,9059 / +0,4365 foram medidos. Na Fig. 12 o primeiro ponto é rotulado **"K=0(Input)"** — é a entrada exibida, não uma geração. | Com `K=0` o mapa é todo zero e a saída tende à AIF; esse ponto de nitidez máxima ancora a correlação e **inflaria** o LVCorr. O probe leria sistematicamente mais alto que a tabela que ele existe para vigiar — ou seja, exatamente o instrumento errado. | grade trocada para `{1,5,10,15}`, com o raciocínio no `probe.py` |
| E2 | No T13 eu atribuí `12.288 tokens / mu 2,54 / exp 12,7` a uma foto **2000×1500**. Esses são os números de **2048×1536**. Uma 2000×1500 dá 11.750 tokens, `mu 2,446`, `exp 11,30`. | Nenhuma consequência de código — o argumento (μ do crop ≠ μ da imagem inteira) sobrevive intacto. Mas é número errado num documento que pede para as pessoas confiarem nos números. | tabela recalculada, quatro resoluções |
| E3 | O teste `test_k_cru_no_crop_erra_por_mais_de_um_terco` **exagerava**: as duas resoluções realmente medidas no `CONTRATO.md` (fatores 0,892 e 0,821) dão 12% e 22% de erro, não "mais de um terço". | Um teste cujo nome afirma mais do que ele prova é pior que teste nenhum: quem lê o nome para de conferir. | renomeado para `..._e_o_erro_varia_por_amostra`, e a asserção passou a travar **a dispersão** (max/min > 2,5×), que é o argumento de verdade |

E uma divergência que **não** é erro mas estava afirmada de forma vaga demais: o
`resize_nearest` do dataloader e o `resize_depth_nearest` da geração usam grades
de amostragem diferentes quando o arredondamento morde (medido: 18,0% dos pixels
num 1503×2001; 0% nas resoluções reais conferidas). Está medido, travado por
teste, e com pendência aberta de alinhar com o `bokehnet-regen`.

---

Detalhe de cada mudança de código: `MUDANCAS_CODIGO.md`.
Ordem de execução, pré-condições e critérios de decisão:
`PLANO_RETREINO_BOKEHNET.md`.

**Nada foi rodado em GPU.** O `python -m genfocus_train.train check --stage bokeh`
roda sem GPU e sem rede, e hoje reprova os quatro YAMLs de treino com 6 problemas
— todos pré-condições legítimas (releases não publicados, manifesto de split
ausente, conjunto do probe não gerado, credenciais). É o comportamento desejado:
o `check` existe para que uma fila de cluster não seja gasta descobrindo isso.

---

## 5. O que NÃO está pronto — as lacunas, nomeadas

Esta seção existe porque a pergunta "está 100% alinhado com o paper?" tem uma
resposta específica, e ela é **não**. Abaixo, o que falta, em ordem de quanto
bloqueia.

### L1 — FECHADA — O dataloader lê as DUAS formas de release

**Era o maior buraco desta pasta. Foi fechada em 2026-09-16; o histórico do
achado fica abaixo, e o que mudou está no fim da seção.**

`BokehMetricDataset` faz `load_dataset(repo)` e lê `registro["aif"]`,
`registro["bokeh"]`, `registro["depth"]` mais escalares — ou seja, assume uma
**tabela** com os pixels dentro.

O `bokehnet-regen` publica outra coisa
(`src/dataio/writer.py:FileSampleWriter`, `scripts/publish_release.py`):

```
<release>/depth/<id>.png          disparidade uint16
<release>/mask/<id>.png
<release>/meta/<id>.json          TUDO que não é pixel
<release>/generated/<id>_*.jpg    SÓ o que a rota GERA
<release>/manifest.jsonl
<release>/split.json
<release>/source_images.jsonl     sha256 dos bytes de origem
```

subido com `api.upload_folder`. Não é uma tabela com colunas de imagem.

E pior, por decisão de cota (365 GB contra 115 GB livres), **os pixels podem não
estar no release**. `sample.py` é explícito — *"guardamos o que geramos,
referenciamos o que já existe"*:

| rota | bokeh | AIF |
|---|---|---|
| C | **referência** | **referência** |
| B | referência | gravada |
| A | gravado | referência |

`publish_release.py` chama isso de release "autocontido" ou não, e o modo é
escolhido em tempo de geração com `--store-source-images`.

**Consequência.** Como está, o dataloader não lê o release. Não é ajuste de
nome de coluna: falta um **resolvedor de referências** (dado `source_dataset` +
`source_sample_id` + `aif_ref`/`bokeh_ref`, buscar a imagem no espelho), e falta
um leitor da árvore de arquivos.

**O que estava feito antes:** um diagnóstico que falhava alto e nomeava esta
situação, em vez de "faltou coluna".

---

#### O que fechou a lacuna (2026-09-16)

O leitor novo vive em **`genfocus_train/release.py`** e aceita as duas formas, de
modo que **o modo do release deixou de ser decisão bloqueante** — que era o ponto:
a escolha entre publicar autocontido ou não voltou a ser uma decisão de disco, e
não uma que muda a forma do loader.

| peça | o que faz |
|---|---|
| `ArvoreDeRelease` | `manifest.jsonl` + `meta/<id>.json` + `depth/<id>.png` + `split.json`. Aceita o layout **plano** e o **agrupado por cena** (`depth/<scene_id>/<id>.png`), que é como `publish_release.py` sobe quando a pasta passa de 10.000 arquivos |
| `ResolvedorDePixels` | `generated/` → `source/` → espelho de origem, nesta ordem, com o sha256 conferido contra `source_images.jsonl` quando há linha |
| `IndiceEspelho` | read-side de `mirror_images.MirrorIndex`: mesma chave (`file_name_base`), mesmo cache (`_mirror_index.json`, mesmo formato), mesmo `locate()`. Um índice construído pela geração é lido aqui sem reconstruir |
| `parse_referencia` | as **três** gramáticas reais: nome de coluna (rota C), `repo#split[linha].coluna` (rota B/ITW) e caminho relativo (rota A/EBB) |
| `data.BokehReleaseTreeDataset` | o dataset; reusa **o mesmo** `_selecionar_indices_metric`, o mesmo `RegistroDescartes` e o mesmo `_pesos_por_rota` do caminho de tabela |
| `data.construir_dataset_metric` | escolhe o leitor pela FORMA do release; misturar as duas formas no mesmo estágio é erro nomeado |

Quatro decisões nossas, **declaradas** no cabeçalho de `release.py` (D-R1 a D-R4):
os escalares vêm do manifesto para varrer e do `meta/` para a amostra (o manifesto
**não** traz `disparity_min`/`disparity_max`); os dois layouts são aceitos; o
`snapshot_download` roda com `local_files_only=True` por default; pixel que não
resolve é erro com motivo nomeado, nunca amostra substituta.

E dois efeitos colaterais que fecham buracos vizinhos:

- `carregar_split_por_cena` passou a ler o `split.json` do release
  (`{"assignment": {cena: lado}}`), que era **o único dos três formatos que ela
  não lia** — e é justamente o que o release grava. Sem `scene_split_manifest`
  no YAML, o dataloader usa o split materializado no próprio release (T6), e
  `verificar_split()` recusa release cujo `split` de manifesto divirja do
  `split.json`.
- `scripts/build_probe_set.py` tinha o **mesmo** defeito (`registro["aif"]`) e
  passou a aceitar as duas formas. Sem isso o probe do T11 não sairia do release
  publicado — e o `train check` reprova run sem conjunto de probe.

**O que continua aberto:** nada disso rodou contra um release real (L2). O que os
testes provam é que o leitor lê uma árvore montada com as chaves de `writer.py`;
o que não provam é que o release publicado tem exatamente essas chaves.

### L2 — ALTO — Nada rodou em GPU

Nenhum step. Os 120 testes provam aritmética, contratos, round-trip de config e,
desde o fechamento do L1, a leitura de uma árvore de release **sintética** —
**não provam integração**. Especificamente nunca executaram:

- `BokehMetricDataset` nem `BokehReleaseTreeDataset` contra dado real. O leitor
  da árvore nunca viu um `manifest.jsonl` publicado, um `_mirror_index.json` do
  espelho de verdade, nem um sha256 do `source_images.jsonl` real: as fixtures
  são montadas em `tmp_path` com as chaves copiadas de `writer.py`;
- o segundo adapter LoRA da §3.3 (o congelamento depende do comportamento do
  `add_adapter` do PEFT, que já mudou entre versões — há trava de contagem, mas
  ela nunca disparou de verdade);
- o probe de LVCorr. **Risco concreto e conhecido:** ele chama o `generate`
  oficial, que chama `encode_prompt`, depois de `release_text_encoders()` ter
  posto `text_encoder = None`. Versões do diffusers que leem `self.text_encoder.dtype`
  sem guarda levantariam `AttributeError`. O probe engole a exceção e segue (a
  política é nunca derrubar o treino), então o sintoma seria **um probe
  silenciosamente vazio** — que é o pior modo de falha possível para o
  instrumento cuja ausência custou a rodada anterior. **O smoke tem que
  confirmar que o probe devolve número, não só que não quebra.**

### L3 — MÉDIO — Metade da §3.3 é do lado de dados

O treino de forma está pronto; PointLight-1K (supplement C) e o BokehMe estendido
com o kernel de forma (Eq. 6) não existem. Sem eles a §3.3 não roda.

### L4 — Decisões que o paper não publica e que continuam abertas

`min_calibration_ssim` (T8), proporção entre rotas (T10), `short_side` × `native`
(T14), LR da fase 2 (T15), replay sintético (T16), `K_min`/`K_max` da Eq. 5,
`gamma`/`defocus_scale`/`highlight` do renderer, otimizador/scheduler/warmup.
Nenhuma delas é "alinhamento com o paper" — o paper cala. São decisões nossas, e
o que se pode exigir é que estejam **declaradas**, o que estão.

### L5 — Divergências conhecidas, medidas e declaradas

| o quê | tamanho | onde |
|---|---|---|
| `sigma_mu_source: full_image` — casa com a inferência, mas o efeito nunca foi medido | a distribuição de σ muda bastante (μ 0,63 → 2,42) | T13 |
| `resize_nearest` × `resize_depth_nearest` da geração | 0% nas resoluções reais; 18,0% quando o `round()` morde | teste dedicado |
| Laplaciano do probe × `cv2.Laplacian` do harness | constante multiplicativa por imagem (borda) | `probe.py` |
| §3.4 (pre-deblur) | não implementado — é da DeblurNet, fora do escopo | — |

### L6 — A ressalva que vale para o projeto inteiro

Ninguém fora do grupo dos autores reproduz este paper bit a bit: o repositório
oficial não publicou código de treino, dados, o benchmark LF-Bokeh, limites de K,
limiar de SSIM, configuração do renderer, resolução de treino, otimizador nem
scheduler. O alvo possível é uma reprodução **pública, coerente e auditável, com
cada desvio declarado** — não identidade com o treino privado. "100%" não é uma
meta atingível aqui, e tratá-la como meta é como se produz número bonito e
errado.

---

---

## 6. Auditoria adversarial independente (2026-09-16)

Um revisor independente auditou esta árvore contra o paper, a inferência oficial
e o contrato, **com instrução explícita de tratar este documento como afirmação a
conferir, não como verdade**. Achou 12 itens, 2 críticos. Conferi os dois
críticos por execução antes de aceitar.

### Aplicados

**A1 — CRÍTICO — o `add_adapter` DESATIVA o LoRA base; "congelado" virou "ausente".**
`PeftAdapterMixin.add_adapter` do diffusers termina em `set_adapter(nome)`
(`diffusers/loaders/peft.py:529`), cujo docstring diz *"forcing the model to only
use that adapter and **disables the other adapters**"* — verificado por mim na
biblioteca instalada [M]. Como `specify_lora` só percorre `active_adapters`, na
fase 3 o LoRA base treinado na fase 2 **não participaria do forward**, e os
branches de AIF e do mapa rodariam sem LoRA nenhum. O §3.3 pede **congelar**, e
isto implementava **desativar**.

A trava que eu tinha escrito não pegava porque conferia `requires_grad` — o mesmo
predicado que o `set_adapter` do PEFT já havia ajustado. **Uma trava baseada no
predicado errado concorda com o bug.** `requires_grad` responde "quem treina";
`active_adapters` responde "quem age". O §3.3 exige respostas diferentes: base
ativo e não treinável, forma ativa e treinável.

Corrigido: `set_adapter([ADAPTER_NAME, SHAPE_ADAPTER_NAME])` depois do
`add_adapter`, e a trava passou a conferir os **dois** predicados em todas as
343 camadas. Travado por dois testes, um deles lendo o fonte do diffusers
instalado — se uma versão futura parar de desativar, o teste avisa em vez de a
premissa envelhecer em silêncio.

**A5 — ALTO — a validação do braço replay continha dado de treino.**
`_build_val_loader` não zerava `synthetic_replay_fraction`, e
`synthetic_replay_datasets` aponta para a rota A com `split: train` — a fonte de
treino do próprio braço. Metade do `val_loss` que escolhe o `best.pt` seria
medida em dado visto, **no único braço que existe para testar uma hipótese
nossa**. Não erra um número: invalida o experimento. Corrigido.

**A2 — CRÍTICO — o `check` dava luz verde para um estágio que morre.**
Eu já sabia que a §3.3 não tem dado (L3), mas o CLI expõe `bokeh-shape`, existe
YAML para ele, e `train check --stage bokeh_shape` passava. Um operador que
seguisse o README gastaria fila de cluster para receber
`NotImplementedError("Stage 'bokeh_shape' não suportado")`. Agora o `check`
**reprova** e a exceção **nomeia a lacuna** (PointLight-1K + Eq. 6, P6 do plano).

### Aplicados na segunda passada (2026-09-16)

**A3 — o probe passou a rodar na grade do TREINO.** `build_probe_set.py` agora
usa o **mesmo** `_plano_geometrico` do dataloader (lado menor → 512, crop central
determinístico), com `--image-size` para o caso de precisar mudar. Fecha três
coisas de uma vez: o desalinhamento entre latente/AIF/mapa quando H,W não são
múltiplos de 16, o custo (~40 GB e ~50 s/imagem sem tiling numa foto 3MP), e a
distribuição — o modelo é treinado em crop 512² e agora é probado nela.
`focus_disparity` **não** é reescalado, e há teste travando isso: disparidade é
1/m e não depende da resolução.

**A4 — `full_seq_len` passou a sair da imagem de ORIGEM.** Medido depois da
correção: o vão fechado foi de 3,0% para **100,0%** nas quatro resoluções
testadas — 1024×688, 1536×1024, 2000×1500 e 2048×1536 dão agora exatamente o
`mu` que a inferência usaria. O contrato no topo do `data.py` sempre disse
"imagem de ORIGEM inteira"; era o código que discordava dele.

**A6** — `geo_condition: true` no caminho novo passou a **falhar alto** em vez de
virar no-op. **A7** — `route_weights` e `synthetic_replay_fraction` agora
compõem: a parte real fica com a massa `(1-f)` distribuída na proporção pedida,
em vez de uniforme. **A8** — duas correções: o rótulo (o teto por cena é decisão
NOSSA; o supplement descreve a composição do dado dos autores) e o critério —
o corte agora ordena por `k_value` e mantém os extremos, porque pegar as 4
primeiras de uma série ordenada por abertura truncava justamente a faixa de K,
que é o sinal que ensina controle. **A9** — o probe repassa `lora_on_main`.
**A10** — RNG semeado por worker nos datasets de bokeh, então crop e flip
obedecem a `runtime.seed`/`seed_per_rank` como o `run_metadata` promete.
**A12** e a citação `:139`→`:138` — comentários corrigidos.

Dos 12 achados, **11 aplicados**. Resta A11 (caminho aposentado, e o revisor não
conseguiu inspecionar a tabela do kfix para confirmar).

### Pendentes — mudam decisão, não só código

~~**A3**~~ e ~~**A4**~~ — APLICADOS acima. Ficam registrados aqui como eram:

**A3 — o probe não rodava em 512², apesar de o docstring afirmar que sim.**
`build_probe_set.py` grava resolução nativa e `probe.py` usa `aif_pil.size`.
Consequência que dói: a grade `K ∈ {1,5,10,15}` fica numa escala de pixel
diferente da que o treino vê (`k_efetivo = k · 512/min(H,W)`) — é o T2
reaparecendo dentro do próprio instrumento. Some-se o risco de grades
divergentes entre latente, AIF e mapa quando H,W não são múltiplos de 16, que a
inferência oficial evita por construção (`resize_and_pad_image` + `cv2.resize`
do mapa) e o nosso script não reproduz.

**A4 — `sigma_mu_source: full_image` fechava 3% do vão, não o vão.**
Medido pelo revisor [M]: numa foto 2000×1500, μ vai de 0,630 para **0,684**
quando o alvo da inferência é **2,446**. Porque `full_seq_len` é calculado sobre
a imagem **já redimensionada**, não sobre a de origem. O eixo existe, mas não é
o eixo que o comentário do T13 anuncia.

### Demais achados, registrados e não aplicados

A6 `geo_condition` virou no-op silencioso no caminho novo (`BokehMetricDataset`
não herda `_GeoMixin`) — flag ligada vira flag desligada, que é a classe de
defeito do fallback. A7 `route_weights` descartado quando o replay liga.
A8 `max_levels_per_scene` rotulado como "literal do supplement" **não é**: o
paper descreve a composição do dado dos autores, não um filtro para as outras
rotas — é decisão nossa apresentada como do paper, exatamente o erro de
categoria que este projeto persegue; e o corte pega as 4 **primeiras** linhas da
cena, o que pode truncar a faixa de K. A9 probe ignora `lora_on_main`.
A10 datasets de bokeh sem RNG semeado (reprodutibilidade prometida no
`run_metadata` não se cumpre). A11 `defocus_kfix` (caminho aposentado).
A12 o comentário do `lora_alpha` afirma que mudá-lo descasaria treino de
inferência — não descasaria, porque `specify_lora` força `scaling = 1` nos dois.

### O que o revisor DERRUBOU

A minha preocupação do L2 com o probe: ele verificou que `encode_prompt` tem
guarda `if self.text_encoder is not None`, então o probe sobrevive ao
`release_text_encoders()`. E refez as contas que eu havia corrigido — tabela do
T13, os 18,0% do resize, o resíduo 0,9256, os fatores 0,892/0,821: todas batem.
Reconferiu também o casamento treino↔inferência linha a linha e os 343 módulos.

---
