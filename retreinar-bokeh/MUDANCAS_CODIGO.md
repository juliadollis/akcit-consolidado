# O que mudou no código — `retreinar-bokeh/` vs. a base

Esta pasta é uma árvore de treino **nova e independente**. Nada aqui altera
`../genrefocus_deblurnet_paper/` (o treino que produziu a fase 1 e a fase 2) nem
`../retreinar-deblur/` (o retreino da DeblurNet, em execução).

**Base da cópia:** `../retreinar-deblur/`, que já traz as correções C1–C10 do
DeblurNet (regex do LoRA, guidance por estágio, warmup aplicado antes do primeiro
update, seed por rank, validação determinística, metadata do checkpoint).
Copiar de lá e não do `genrefocus_deblurnet_paper/` foi decisão: as dez correções
valem igual para os dois estágios, e refazê-las seria retrabalho com chance de
divergir.

**Objetivo:** treinar a BokehNet sobre o release novo do `../bokehnet-regen/`,
no contrato `metric_disparity_official_v1`, com instrumento capaz de ver o modo
de falha que custou a rodada anterior.

Cada seção referencia o achado correspondente em `AUDITORIA_TREINO_BOKEHNET.md`.

---

## 0. O que NÃO mudou, e por quê

A §2 da auditoria lista 18 itens conferidos linha a linha contra
`Inference_bokehNet.py` e `Genfocus/pipeline/flux.py` — rank 64, batch 32,
currículo 40K+40K... (40K+60K), prompt, guidance 1.0, timestep 0 nas condições,
`group_mask` diagonal, mapa em `[0,1]` cru no VAE, 3 canais replicados, `img_ids`
sem `position_delta`, `alpha = rank`, `.sample()` no VAE, crop 512 = `TILE_SIZE`.

**Nenhum foi tocado.** Vários parecem melhoráveis e não são: os 3 canais
idênticos do mapa desperdiçam 2/3 da banda do VAE, e mudar isso quebra a
compatibilidade com a inferência oficial. Estão registrados na auditoria
justamente para que uma revisão futura não os "conserte".

---

## 1. `genfocus_train/control.py` — arquivo NOVO

Espelho *read-side* de `../bokehnet-regen/src/control/contract.py`. A geração
grava, este lê. Existe como módulo, e não como conta espalhada pelo `data.py`,
pelo motivo registrado no `CONTRATO.md`: **cópias divergem — foi assim que o
projeto chegou a quatro interpretações de K.**

```python
CONTROL_VERSION = "metric_disparity_official_v1"
MAX_COC = 100.0        # Inference_bokehNet.py:20, CONSTANTE DE MÓDULO

decode_disparity_u16(u16, disp_min, disp_max) -> disparidade em 1/m
k_at_resolution(k, image_hw, dst_short_side)  -> K na escala do destino
k_by_scale(k, escala)                         -> idem, com o fator já calculado
signed_coc_px(disp, focus_disp, k)            -> CoC com sinal, em px
defocus_map(disp, focus_disp, k)              -> condição em [0,1]
validar_registro_controle(registro)           -> recusa release de outra convenção
```

Três coisas que ele impede **por construção**, e não por comentário:

| impedimento | como | qual defeito |
|---|---|---|
| `MAX_COC` por fonte | não é parâmetro de função nenhuma | T4 — o `kfix` com `max_coc=10,5107` |
| fallback numérico | `ControlContractError`, nunca default | o `k = 50,0` em 11.635 de 11.635 |
| escala de pixel implícita | `k_at_resolution` é obrigatória entre o gravado e o mapa | T2 |

`k_by_scale` existe além de `k_at_resolution` porque `_plano_geometrico` já
decidiu o resize e o fator depende do `scale_mode`: em `native` não há resize
(fator 1,0) e em `long_side` o fator sai do lado maior. Derivar de novo a partir
de `dst_short_side` daria o número errado em dois dos três modos. As duas rotas
são travadas como equivalentes em `short_side` por
`tests/test_control_contract.py::test_k_by_scale_bate_com_k_at_resolution_em_short_side`.

---

## 1-bis. `genfocus_train/release.py` — arquivo NOVO (fecha o **L1**)

O `bokehnet-regen` publica uma **árvore de arquivos**, não uma tabela com os
pixels dentro; e por decisão de cota (365 GB contra 115 GB livres) os pixels
podem **não estar** no release. O dataloader lia só a forma de tabela, então não
lia o release — era a lacuna L1 da auditoria, e ela bloqueava o retreino inteiro.

Este módulo aceita **as duas formas**, o que faz o modo do release (autocontido
ou não) deixar de ser decisão bloqueante: vira uma escolha de disco do time de
dados, e não uma que muda a forma do loader.

```python
ArvoreDeRelease       manifest.jsonl + meta/<id>.json + depth/<id>.png + split.json
ResolvedorDePixels    generated/ -> source/ -> espelho, com sha256 conferido
IndiceEspelho         read-side de mirror_images.MirrorIndex (MESMO cache)
parse_referencia      as três gramáticas reais de `aif_ref`/`bokeh_ref`
resolver_raiz_do_release / preflight_de_pixels
```

**Reuso, e não reimplementação.** `IndiceEspelho` usa a mesma chave
(`file_name_base` = `source_sample_id`), o mesmo arquivo de cache
(`_mirror_index.json`) e o mesmo formato (`{"shards": [...], "locations":
{nome: [shard, linha]}}`) do `bokehnet-regen/src/sources/mirror_images.py`. Um
índice construído pela geração é lido aqui sem reconstruir — e reconstruir custa
ler 85 shards. O que impede as duas cópias de divergirem é esse formato
compartilhado, e ele está travado por teste.

Também herdada de lá: a conferência de que a linha apontada pelo índice **contém
o nome esperado**. Sem ela, um índice cacheado de outro snapshot troca TODOS os
pares e o histograma de rejeição fica limpo, porque nada falha.

**As três gramáticas de referência** convivem porque as três rotas têm origens de
natureza diferente, e cada uma foi gravada na forma que a origem entende:

| forma | quem usa | exemplo |
|---|---|---|
| nome de coluna | rota C (RealBokeh, LFDOF) | `image_focus` — a LINHA vem do `source_sample_id`, pelo índice |
| `repo#split[linha].coluna` | rota B (ITW) | `atfortes/BokehDiffusion#train[41].image` |
| caminho relativo | rota A (EBB/[80]) | `train/0001.jpg` |

Forma desconhecida é **erro**, não palpite: adivinhar aqui é como uma amostra
acaba apontando para o pixel errado sem que nada denuncie.

**Decisões nossas, declaradas** (o paper e a inferência oficial calam; estão como
D-R1 a D-R4 no cabeçalho do módulo):

| # | decisão | por quê |
|---|---|---|
| D-R1 | escalares do `manifest.jsonl` para varrer, do `meta/<id>.json` para a amostra | o manifesto **não traz** `disparity_min`/`disparity_max`, sem os quais a profundidade não decodifica, nem `aif_ref`/`bokeh_ref`. Abrir 20 mil JSONs só para filtrar seria desperdício; abrir os que sobreviveram não é |
| D-R2 | aceitar o layout plano **e** o agrupado por cena | os dois existem: o writer grava plano em disco e `publish_release.py` reagrupa no push, porque o Hub recusa mais de 10.000 arquivos por pasta |
| D-R3 | `snapshot_download(local_files_only=True)` por default | 552 GB de egress inexplicado medidos neste projeto. Um download que acontece por default é um download que ninguém decidiu |
| D-R4 | pixel que não resolve é erro com motivo nomeado | regra 4 do `CONTRATO.md` do lado de leitura. Nunca amostra substituta, nunca descarte silencioso |

O sha256 é conferido contra `source_images.jsonl` sempre que a linha existe. É o
que transforma "o release aponta para um espelho privado" em **evidência**: o K
desta amostra foi calibrado contra bytes específicos, e sem conferir isso é
afirmação. A proveniência distingue `True` ("conferi e bateu"), `False` (impossível
— esse caso levanta) e `None` ("não havia sha para conferir"), e o resumo do run
conta quantos entraram sem prova.

---

## 2. `genfocus_train/data.py`

### T1 — `defocus_source: "metric_disparity"`, o novo default

`DEFOCUS_SOURCES` ganhou um quarto valor, que passou a ser o default. As colunas
exigidas mudam junto:

```python
BOKEH_REQUIRED_COLUMNS_METRIC = {
    "aif", "bokeh", "depth",
    "control_version", "k_value", "focus_disparity",
    "disparity_min", "disparity_max", "image_h", "image_w",
}
```

Note o que **não** está aí: `defocus_map`. O mapa não é gravado no release — é
derivado no dataloader com a mesma função da geração, de propósito
(`bokehnet-regen/src/dataio/sample.py`). Gravá-lo criaria uma segunda fonte de
verdade, que é o defeito D1.

### T2 + T3 — `prepare_aligned_bokeh_metric()`, função nova

Difere de `prepare_aligned_bokeh` em exatamente três pontos, e os três são
correções:

1. **K reescalado** pelo fator efetivo do `_plano_geometrico`. A função antiga
   calculava `escala`, devolvia no dict de geometria para o `geo_cond`, e nunca
   multiplicava o K.
2. **NEAREST na profundidade**, via `resize_nearest()` — gather de índice, sem
   PIL, testável. A geração usa nearest pelo motivo explícito de que interpolar
   profundidade atravessa descontinuidade e inventa um plano intermediário.

   **Divergência de convenção, medida e declarada.** A geração amostra em
   `i / escala`; aqui é `i * H / novo_H`. Coincidem sempre que
   `novo_H == H·escala` exatamente — o caso comum, conferido em três resoluções
   reais. Quando o `round()` morde (aspecto não exato), a grade da geração
   desliza até 1 px de origem; medido: **18,0% dos pixels** num 1503×2001.
   A nossa fica porque mapeia a grade de saída uniformemente sobre a origem, em
   aritmética inteira exata — a definição padrão. E identidade bit a bit nunca
   foi possível de todo modo: no treino o resize vai da grade da profundidade
   GRAVADA (lado longo 768) para a do crop, transformação que a geração não faz.
   Travado por `test_convencao_do_resize_diverge_da_geracao_so_com_arredondamento`.
   **Pendência:** propor ao `bokehnet-regen` adotar `i*H/novo_H`.
3. **`MAX_COC` não é parâmetro.**

A **ordem das operações** é load-bearing e está no docstring: decodificar →
reamostrar → recortar → *só então* aplicar a Eq. 2. É equivalente a aplicar antes
e recortar depois (ambas são por-pixel, e o NEAREST é um gather puro), mas nesta
ordem o `abs` mantém o vinco **exato** no plano de foco; reamostrar o mapa já
absolutizado suavizaria o vinco.

Recusa (não ajusta) resolução divergente do metadado: se a AIF chega 800×600 e o
`k_value` foi medido em 1600×1200, isso é `source_image_unreadable` do lado da
geração e é erro aqui. Resolução heterogênea muda o K em pixel sem mudar nada
visível no JSON.

Devolve também `k_efetivo`, `defocus_mean`, `defocus_max` e
`defocus_frac_saturado` — os escalares do T12, calculados de graça no caminho.

### T7, T8, T9, T6 — `_selecionar_indices_metric()` e `RegistroDescartes`

Filtros: `control_version`, `is_valid_for_control`, `is_k_censored`,
`calibration_ssim`, `focus_was_refined`, split por `scene_id`, teto de níveis por
cena.

**Regra que governa todos:** um filtro cuja coluna não existe no release é
**desligado com aviso**, nunca aplicado a partir de um default. Um filtro que se
aplica por default é a mesma família de defeito do `k = 50,0`. A única exceção é
`scene_split_manifest`: pedir split por cena sem a coluna `scene_id` é **erro**,
porque a alternativa (split por linha) vaza a cena inteira — a mesma cena aparece
com 2 a 21 aberturas.

`RegistroDescartes` é o histograma de motivos, com **vocabulário fechado**
(motivo não registrado levanta `KeyError`). Sem ele, "o dataset tem 18.402
amostras" é um número sem significado: não se sabe se 2.000 saíram por censura de
K ou se um filtro novo comeu metade do release em silêncio. É o instrumento que
denuncia fallback novo, e é impresso no início de todo run.

### T10 + T16 — amostragem ponderada

- `_pesos_por_rota(rotas, route_weights)` — cada amostra da rota `r` pesa
  `w_r / n_r`, então a massa da rota é `w_r` independentemente do tamanho do
  shard. É o que transforma a proporção entre rotas de acidente em decisão.
- `montar_pesos_de_replay(n_real, n_sint, f)` — realiza exatamente a fração `f`
  de replay sintético, também independentemente dos tamanhos.
- `BokehConcatDataset` — concatena **preservando** `sampler_weights`. O
  `torch.utils.data.ConcatDataset` os perde, e com eles a proporção volta a ser o
  tamanho relativo dos shards.
- `build_dataloader` passa a montar um `WeightedRandomSampler` quando o dataset
  expõe pesos. `shuffle` e `sampler` são mutuamente exclusivos no `DataLoader`,
  então o sampler substitui o shuffle; `num_samples = len(dataset)` mantém a
  definição de "época".

### T5 — as convenções aposentadas ficam atrás de uma porta

`recompute`, `column` e `kfix` continuam alcançáveis — são histórico
reprodutível — mas exigem `allow_retired_defocus_sources: true` **explícito**, e
o run imprime `AVISO GRAVE` e grava a escolha no metadata. Uma linha de YAML não
pode separar um treino de 60K steps da convenção que produziu o pior LPIPS entre
as variantes reais.

### L1 — `BokehReleaseTreeDataset` e `construir_dataset_metric`

O gêmeo do `BokehMetricDataset` para a forma que o `bokehnet-regen` de fato
publica. O contrato de saída é **idêntico** — as mesmas chaves, e há teste que lê
as chaves do outro leitor **direto do AST do fonte**, em vez de copiá-las, porque
copiá-las faria o teste passar mesmo que os dois divergissem. `collate_strict`
recusa batch heterogêneo, então uma chave a mais aqui quebraria qualquer mistura.

O que **não** foi duplicado: os filtros. `_selecionar_indices_metric`,
`RegistroDescartes` e `_pesos_por_rota` são literalmente os mesmos nos dois
caminhos; o adaptador `_EscalaresDoManifesto` (vinte linhas) apresenta o
`manifest.jsonl` com a interface que aquela função já usa. Uma segunda cópia da
lógica de filtro é a cópia que diverge.

`_EscalaresDoManifesto` **recusa manifesto heterogêneo** — chave presente em umas
linhas e ausente em outras. Isso é lote misto, e tratar a ausente como `None`
faria um filtro decidir por um valor que ninguém mediu: a mesma família do
`k = 50,0`.

`construir_dataset_metric` escolhe o leitor pela FORMA do release. Misturar as
duas formas no mesmo estágio é **erro nomeado**, não concatenação: os
pré-requisitos são diferentes (uma precisa do espelho, a outra precisa dos pixels
na tabela) e uma mensagem que diga "faltou coluna" para metade das fontes não
ajuda ninguém.

`build_val_dataset` passou a usar o mesmo seletor. Antes ela ia direto para o
`HuggingFaceBokehDataset`, que é o caminho ANTIGO: com o default
`metric_disparity`, a validação leria o sinal de controle por outra convenção que
não a do treino — e é a val_loss que escolhe o `best.pt`.

### T6 + L1 — `carregar_split_por_cena` lê o `split.json` do release

Faltava exatamente o formato que o release grava
(`{"assignment": {cena: "train"|"val"}}`, de `dataio/split.py:SceneSplit.to_json`).
Agora, sem `scene_split_manifest` no YAML, o dataloader usa o split
**materializado no próprio release** — que é para isso que ele existe. E
`ArvoreDeRelease.verificar_split()` recusa release cujo `split` gravado em cada
linha do manifesto divirja do `split.json`: é a versão de leitura do
`check_no_leak`, e pega o caso de alguém ter trocado o arquivo depois da geração.
Nada disso apareceria na loss — a validação simplesmente passaria a medir
memorização.

### T19 — `sigma_mu_source` no `DatasetRuntimeConfig`

O trainer já passava a chave e o dataclass não a tinha; o guard a descartava com
aviso **em todo run**. Um aviso que sempre aparece treina o operador a ignorar
avisos, e o guard existe justamente para que um aviso signifique alguma coisa.
Uma linha.

---

## 3. `genfocus_train/config.py`

Campos novos em `StageConfig` (todos com a linha correspondente em
`_as_stage_config` — a **armadilha** documentada no topo do módulo, coberta por
`tests/test_config_roundtrip.py`):

| campo | default | achado |
|---|---|---|
| `defocus_source` | `"metric_disparity"` | T1 |
| `require_valid_for_control` | `True` | T7 |
| `exclude_censored_k` | `True` | T7 |
| `max_levels_per_scene` | `4` | T9 — supplement B.2, "2 to 4 images per set" |
| `route_weights` | `None` | T10 |
| `exclude_refined_focus` | `False` | requisito do `bokehnet-regen` |
| `scene_split_manifest` / `scene_split_partition` | `None` / `"train"` | T6 |
| `allow_retired_defocus_sources` | `False` | T5 |
| `synthetic_replay_fraction` / `synthetic_replay_datasets` | `0.0` / `[]` | T16 |
| `shape_column` | `None` | T17 |
| `min_calibration_ssim` | `None` (era 0.6 nos YAMLs antigos) | T8 |
| `release_format` | `"auto"` | L1 — `"auto"` decide só pelo que dá para ver SEM REDE |
| `mirror_roots` | `None` | L1 — `source_dataset` -> snapshot local do espelho |
| `permitir_download_do_release` | `False` | L1/D-R3 — 552 GB de egress |
| `verificar_sha256_da_origem` | `True` | L1 — o que faz do join uma evidência |

Em `ModelConfig`: `bokeh_shape_lora_rank` (64) e `bokeh_shape_train_guidance`
(1.0).

**`max_coc` virou asserção.** `_coerce_config_dict` compara com `control.MAX_COC`
e **falha o load** se divergir. Mudar o normalizador exigiria mudar a constante,
subir o `CONTROL_VERSION` e regerar os dados — não passar um argumento.

`stage_config`, `train_guidance_for` e `lora_rank_for` passaram a aceitar
`"bokeh_shape"`. `stage_config("bokeh_shape")` **recusa** o estágio se
`shape_column` não estiver definido.

Validações novas no `__post_init__`: convenção aposentada sem porta,
`synthetic_replay_fraction` fora de `[0,1)`, fração > 0 sem fonte de replay,
`max_levels_per_scene < 1`, pesos de rota negativos ou somando zero.

---

## 4. `genfocus_train/backbone.py` — §3.3

`SHAPE_ADAPTER_NAME = "shape"`. O construtor ganhou `shape_lora_rank`; quando
não-`None`, `_inject_shape_lora()` roda depois do `_inject_lora()`:

1. o adapter base já existe;
2. `add_adapter` cria o segundo;
3. **congelamos explicitamente** todo parâmetro com `.default.` no nome e
   conferimos, por lista, que sobrou treinável **só** o adapter de forma.

O passo 3 não é redundância. O comportamento do `add_adapter` quanto a
`requires_grad` já mudou entre versões do PEFT, e um congelamento que dependa
disso é um congelamento que um `pip install -U` desfaz em silêncio. Aqui, se
desfizer, a trava aborta com a lista dos vazados.

`forward_train_step` ganhou `condition_adapters: Sequence[str | None] | None` —
**um nome de adapter por condição**. Com `None` (o caso de todos os estágios que
não são o de forma) o comportamento é idêntico ao anterior. Na inferência,
`specify_lora` liga um adapter por branch e zera a escala dos outros, então isto
é exatamente reproduzível lá.

`lora_info()` ganhou `shape_rank` e `n_lora_modules_shape`, que vão para o
metadata do checkpoint e para o `.json` irmão do `.safetensors`.

---

## 5. `genfocus_train/models.py` — §3.3

`BokehNet.make_train_batch` aceita `shape_image` e monta `cond_adapters`:

```
sem forma:  [AIF, D_def]        adapters [default, default]
com forma:  [AIF, D_def, A]     adapters [default, default, shape]
```

A imagem da forma entra em `[-1,1]` (é uma imagem, não um mapa escalar): a
inferência oficial só usa `No_preprocess=True` no mapa de defocus.

Um detalhe que o `group_mask` impõe e que vale registrar: as condições são
**mutuamente cegas** — o branch da forma não vê a AIF nem o mapa, e a combinação
acontece no branch principal, que vê tudo. É o mesmo isolamento que a AIF e o
mapa já têm entre si, e é o que a inferência oficial faz incondicionalmente.

---

## 6. `genfocus_train/probe.py` — arquivo NOVO (T11)

O instrumento que faltava.

- `laplacian_variance(img)` — luminância BT.601, kernel de 4 vizinhos, **sem** o
  reescalonamento para 256² que `data._laplacian_variance` faz (reamostrar
  mudaria a escala do gradiente, que é a grandeza medida).
- `lvcorr(ks, nitidez)` — Pearson com o **sinal invertido**. Nitidez cai quando
  K cresce, e o número do paper é positivo. `nan` quando alguma série é
  constante — o que já é diagnóstico: nitidez constante significa que o K não
  mudou nada.
- `lvcorr_agregado(por_imagem)` — média **por imagem**, não sobre a nuvem de
  pontos toda: a variância do Laplaciano tem escala muito diferente entre cenas,
  e juntar tudo deixaria a cena mais texturizada dominar.
- `run_lvcorr_probe(...)` — gera com a configuração da inferência oficial
  (`guidance_scale=1.0`, 28 passos, `NO_TILED_DENOISE=True`, `kv_cache=False`,
  prompt e adapter da BokehNet), com prompt pré-computado (os text encoders já
  foram liberados). **Nunca derruba o treino**: exceção vira aviso e `{}`.
- `K_GRID_PAPER = (1, 5, 10, 15)` — **os K do harness de avaliação deste
  projeto** (`vision-pipeline/evaluation/eval_bokeh_synthesis.py:35`), não os da
  Fig. 12. A Fig. 12 mostra `{0,5,10,15}`, mas o K=0 dela é rotulado
  "K=0(Input)": é a imagem de entrada exibida, não uma geração com mapa nulo.
  Usar `{0,...}` daria um número sistematicamente maior — com mapa todo zero a
  saída tende à AIF, e esse ponto de nitidez máxima ancora a correlação — e
  portanto **não comparável** com a tabela (+0,9059 / +0,4365) que este probe
  existe para vigiar.

`scripts/build_probe_set.py` monta o conjunto fixo a partir de um release, com
`scene_id` disjunto do treino, recusando amostra com `is_k_censored` ou
`is_valid_for_control=False`.

**Não rodado em GPU.** A estimativa de custo (~5 min por probe, ~1% do
orçamento) é [A] e precisa de um smoke antes do primeiro treino longo.

---

## 7. `genfocus_train/trainer.py`

| # | mudança |
|---|---|
| T11 | probe de LVCorr a cada `probe_every_steps`, com **alarme declarado**: queda > `probe_alerta_queda` (0,10) abaixo do melhor do run imprime ALARME e o critério é PARAR e investigar. Sem `probe_set_dir`, aviso GRITADO de 6 linhas no início do run — não um aviso de uma linha entre outros. |
| T12 | `_metricas_de_controle()` — `defocus_mean/max/frac_saturado` e `k_efetivo` (média e **desvio dentro do batch**: desvio zero = K constante, que é o defeito D2) no log de todo step de log. |
| T18 | `_cycle` chama `set_epoch` no sampler a cada volta. Sem isso, ~74 épocas com a mesma permutação e o mesmo particionamento entre GPUs. |
| T20 | `run_validation` repassa os quatro parâmetros de oclusão — é esta loss que escolhe o `best.pt`. |
| T6 | `_build_val_loader` força `max_levels_per_scene=None` (na validação queremos **todos** os níveis da cena, é onde a variação de K se mede) e `scene_split_partition="val"`. |
| T17 | `run_bokeh_shape_stage()`, que **exige** `init_lora_path` — congelar um LoRA de inicialização gaussiana treinaria forma sobre um modelo que não faz bokeh. |
| T17 | `load_lora_checkpoint_into_backbone(..., exigir_completo=False)` para o estágio de forma, com a checagem trocada por uma **mais forte**: `unexpected` tem que ser vazio e o adapter **base** tem que ficar coberto. |
| T17 | `export_lora_safetensors` exporta o adapter de FORMA no estágio de forma. Exportar o base ali republicaria um arquivo idêntico com nome novo. |

---

## 8. `genfocus_train/train.py`

Subcomando novo `bokeh-shape`, com `--init-lora` **obrigatório**.
`STAGE_PROMPTS["bokeh_shape"] = BOKEH_PROMPT` — o mesmo prompt da fase 2, porque
a fase 3 congela justamente o LoRA que aprendeu a responder a esse prompt.
`smoke`, `check` e `export` aceitam `--stage bokeh_shape`.

---

## 9. `configs/`

| arquivo | o que é |
|---|---|
| `train_bokeh_fase1_synth.yaml` | fase 1, 40K, rota A |
| `train_bokeh_fase2_real.yaml` | fase 2, 60K, rotas B+C. **Braço primário**, leitura literal do §4.1 |
| `train_bokeh_fase2_lrbaixo.yaml` | braço B (T15): `lr 5e-5`, warmup 2000. Difere do primário em 3 linhas |
| `train_bokeh_fase2_replay.yaml` | braço C (T16): 15% de replay sintético. **Desvio declarado do §4.1** |
| `train_bokeh_fase3_shape.yaml` | §3.3, 10K, LoRA base congelado |
| `train_bokeh_smoke.yaml` | 3 steps, wandb e upload desligados |

Todos validados pelo `load_config` real (não só `yaml.safe_load`), o que também
prova que os campos novos não caem na armadilha do `_as_stage_config`.

Os repos de release estão como `PREENCHER/...` **de propósito**: um repo antigo
ali treinaria na convenção velha em silêncio, que é exatamente o que esta árvore
existe para impedir. `grep -n PREENCHER configs/` antes de qualquer `sbatch`.

Mudanças de default em relação aos YAMLs antigos de bokeh, todas declaradas nos
comentários do próprio arquivo: `sigma_mu_source: full_image` (T13),
`min_calibration_ssim: null` em vez de 0.6 (T8), `max_levels_per_scene: 4` (T9),
`probe_every_steps: 5000` (T11), e `vae_subfolder`/`transformer_subfolder`
removidos (T21 — ninguém os lia).

---

## 10. `slurm/`

`train_bokeh_fase1_4gpu.slurm`, `train_bokeh_fase2_4gpu.slurm`,
`train_bokeh_fase3_shape.slurm`, `smoke_bokeh.slurm`.

Todos com `--time` alto (14 dias nos treinos, 1 dia no smoke — regra do
`CLAUDE.md`), `logs/` criado antes, `.env` carregado com falha cedo se faltar
token, versões impressas antes de começar, e re-`sbatch`-áveis.

As fases 2 e 3 **abortam antes da fila** se o `--init-lora` não existir, com a
mensagem dizendo o que rodar primeiro.

**Nenhum script contém `scancel`, `rm` ou `--delete`** — verificado por grep.

---

## 11. `tests/`

`tests/test_control_contract.py` — **22 testes novos**, sem GPU e sem rede. Cada
um trava um achado nomeado:

| teste | trava |
|---|---|
| `test_max_coc_nao_e_parametro_de_defocus_map` | T4 — inspeciona a **assinatura**, não o comentário |
| `test_max_coc_divergente_no_release_e_recusado` | T4 — o valor exato do `kfix` (10,5107) |
| `test_control_version_divergente_e_recusado` | T1 |
| `test_defocus_map_replica_a_inferencia_oficial` | T1 — transcreve `Inference_bokehNet.py:138-140` e compara com `rtol=0` |
| `test_focus_disparity_nao_e_um_sobre_a_mediana_da_profundidade` | regra 1 do CONTRATO, com contagem PAR |
| `test_k_by_scale_bate_com_k_at_resolution_em_short_side` | T2 — as duas rotas do fator |
| `test_gate_mapa_pos_crop` | T2 — o **gate NOVO** do `PLANO_REGERACAO`: passa uma amostra sintética pelo caminho real e confere o mapa contra a fórmula recalculada na escala do crop |
| `test_k_cru_no_crop_erra_por_mais_de_um_terco` | T2 — prende o número que justifica a correção |
| `test_resize_nearest_nao_inventa_valor_intermediario` | T3 |
| `test_resolucao_divergente_do_metadado_e_recusada` | T2 |
| `test_pesos_de_replay_realizam_a_fracao_exata` | T16 |
| `test_peso_ausente_e_erro_e_nao_zero_silencioso` | T10 |
| `test_lvcorr_*` | T11 — inclusive o sinal invertido |

Testes existentes atualizados por consequência da mudança de default:
`test_config_roundtrip.py` (11 campos novos em `VALORES` — o teste falhou
sozinho, que é o que ele existe para fazer), `test_config_validacao.py` (dois
testes novos sobre a porta do T5), `test_data_pipeline.py` (os testes da pasta
local exercitam `recompute`, que agora precisa ser pedido).

`scripts/prefetch_bokeh_local.py` foi trazido do `genrefocus_deblurnet_paper` —
ele estava referenciado por um teste e ausente da árvore desde o `retreinar-deblur`,
onde o teste falhava por `FileNotFoundError`.

`tests/test_release_tree.py` — **37 testes novos** (L1), sem GPU e sem rede. As
fixturas montam a árvore de release em `tmp_path` com as chaves copiadas de
`writer.py`, não inventadas: o valor de um teste destes é exatamente o quanto a
fixture se parece com o release real.

| teste | trava |
|---|---|
| `test_le_a_arvore_de_arquivos_que_o_regen_publica` | L1 — o release é `manifest.jsonl` + `depth/` + `meta/`, não uma tabela |
| `test_chaves_do_batch_sao_as_mesmas_do_leitor_de_tabela` | lê as chaves do outro leitor do **AST do fonte**, não de uma cópia |
| `test_o_K_e_reescalado_tambem_no_leitor_de_arvore` | T2 no caminho novo, com oráculo independente (`np.repeat` = nearest exato por 2) e contraprova com o K cru |
| `test_layout_agrupado_por_cena_e_lido_igual_ao_plano` | D-R2 — os dois layouts publicados |
| `test_disparity_min_max_so_existem_no_meta` | por que ler o `meta/` por amostra não é opcional |
| `test_manifesto_heterogeneo_e_recusado` | lote misto |
| `test_rota_b_*` / `test_rota_a_*` / `test_modo_autocontido_*` | a tabela "guardamos o que geramos, referenciamos o que já existe" |
| `test_sha256_divergente_e_recusado` | o join verificável |
| `test_sem_pixels_e_sem_espelho_o_erro_diz_as_duas_saidas` | D-R4 — motivo nomeado, com as duas saídas |
| `test_indice_do_espelho_le_o_cache_gravado_pela_geracao` | o cache compartilhado com o `bokehnet-regen` |
| `test_indice_de_outro_snapshot_e_pego_pela_conferencia_do_nome` | o modo de falha que deixa o histograma limpo |
| `test_split_do_release_e_usado_sem_scene_split_manifest` | T6 — split materializado |
| `test_split_gravado_divergente_do_split_json_e_recusado` | a versão de leitura do `check_no_leak` |
| `test_download_desligado_por_default` | D-R3 |
| `test_build_probe_set_le_a_arvore_*` | T11 — o probe também saía de `registro["aif"]` |

**Total: 120 testes passando, 53 subtestes.**

```bash
python -m pytest tests/ -q
```

---

## 12. O que NÃO foi feito, e por quê

| item | por quê |
|---|---|
| Nada rodado em GPU | pedido explícito. O smoke (`slurm/smoke_bokeh.slurm`) é o próximo passo e não roda sem o release. |
| `min_calibration_ssim` continua `null` | T8 — o 0,6 foi calibrado na distribuição antiga. Precisa da distribuição nova para escolher um percentil. |
| `route_weights` continua `null` | T10 — deixar proporcional é o comportamento anterior; mudar isso e o contrato do K ao mesmo tempo tornaria a comparação inútil. |
| `scale_mode` continua `short_side` | T14 — é um trade-off real com dois lados, não uma correção óbvia. Vai a A/B com o probe. |
| `scene_split_manifest` continua `null` | e agora é o valor **certo** para um release em árvore: o split vem do `split.json` materializado dentro dele (L1). Só o caminho de tabela achatada ainda precisa de um manifesto à parte. |
| `mirror_roots` continua `null` nos YAMLs | L1 — só é preciso se o release NÃO for publicado autocontido, e essa decisão é do time de dados. O `train check` reprova a combinação "árvore sem `source/` + `mirror_roots` vazio" antes da fila. |
| O leitor da árvore nunca viu release real | L2 — as fixturas são sintéticas. As chaves foram copiadas de `writer.py`; o que não está provado é que o release publicado traz exatamente essas chaves, nem que o `_mirror_index.json` do espelho real casa com as amostras. |
| PointLight-1K e o BokehMe com kernel de forma | T17 — é lado de dados. O lado de treino da §3.3 está pronto e sem dado para exercitar. |
| A §3.4 (pre-deblur) | é da DeblurNet, fora do escopo desta pasta. |

---

# Rodada de 2026-09-25 — auditoria do código de treino depois da fase 1

Pedido: *"foque no codigo de treino. ta tudo 100% correto?"*

A resposta honesta está no fim desta seção. Antes, o que foi achado — incluindo
**um erro meu de documentação**, que é o mais importante da lista porque estava
prestes a fazer a gente jogar fora um artefato bom.

## R0 — O README dizia que a validação estava desligada. Estava ERRADO.

O quadro "o que está ERRADO neste run" tinha como item #1: *"validação desligada
— bug: o YAML tem `val_datasets` e o código não enxerga"*, e concluía que o
`best.pt` seguia a loss de treino de um micro-batch e não devia ser usado.

**Nada disso aconteceu.** O que me enganou foi um `grep` no log de 110 mil
linhas, que achou três ocorrências de

```
[eval] AVISO: eval_every_steps > 0 mas não há `val_datasets` no YAML —
       validação DESLIGADA e `best_loss` seguirá a loss de treino.
```

Três, com quatro ranks. Eram os ranks 1, 2 e 3 — onde `_build_val_loader`
devolve `None` **por projeto**, porque construir o dataset de validação nos
quatro processos é a dessincronia que travou o multi-GPU no watchdog do NCCL.
O rank 0 não imprimiu nada. Eu li o aviso como defeito e não conferi o artefato.

O que o artefato diz:

| evidência | valor |
|---|---|
| `best.pt` → `metadata.metrics_snapshot` | `{"val_loss": 0.06018715, "best_loss": 0.06018715}` |
| o campo `val_loss` só é gravado em | `if melhorou and save_best and val_loss is not None` |
| `effective_config.yaml` → `val_datasets` | o release, com `scene_split_partition` forçado a `"val"` |
| log do treino | `-13940 (20.0%) scene_split_excluded` |

Ou seja: **20% das cenas foram retidas, disjuntas POR CENA**, a validação rodou
a cada 500 steps e o `best.pt` é do step 40.000, escolhido por `val_loss` em dado
não visto. O `best.pt` é legítimo e pode ser usado.

O defeito real era menor e é este: **a mensagem mentia nos ranks ≠ 0**. Corrigido
com `and accelerator.is_main_process`. Uma frase errada num log é um defeito, não
um detalhe — esta custou dois dias de README errado.

## R1 — Probe interno + multi-GPU: era um campo minado armado para a fase 2

Os configs `fase2_real`, `fase2_lrbaixo`, `fase2_replay`, `fase3_shape` e
`fase1_synth` traziam `probe_every_steps: 5000` (ou 2000), contradizendo a
decisão declarada de que **o instrumento roda por fora**.

Se alguém preenchesse `probe_set_dir` — e o comentário do YAML mandava preencher
— o probe interno ligaria, e em multi-GPU ele:

1. roda ~5 min só no rank 0 gerando 32 imagens de 28 passos, enquanto os outros
   ranks avançam sozinhos até o `all_reduce` dos gradientes e ficam presos lá:
   **exatamente o cenário do watchdog do NCCL que matou este treino duas vezes**;
2. levanta `ParadaPorControlabilidade` só no rank 0, que vai para o `finally`,
   salva e para num `wait_for_everyone()` — enquanto os demais esperam num
   `all_reduce`. Coletivos diferentes: deadlock até o `SIGABRT`.

Correção em duas camadas, porque comentário de YAML não impede nada:

- `_train_loop` **recusa** subir com `probe_every_steps > 0` e mais de um
  processo, e a mensagem aponta o `scripts/monitor_externo.py` (recusar sem dizer
  o caminho que funciona faz o operador só desligar o probe e ficar sem
  instrumento);
- os sete configs passaram a `probe_every_steps: 0`, com o motivo escrito.

Verificado em 2 GPUs: a guarda dispara **em todos os ranks**, então não há
deadlock na própria recusa.

## R2 — Campo de config descartado em silêncio virou erro

`_dataset_runtime` montava o `DatasetRuntimeConfig` e, para todo campo que o
dataloader não conhecesse, **imprimia um aviso e seguia** — caindo no default.

É a forma exata do defeito `--env GRAD_ACCUM`: uma chave que o operador escreve
e que nada lê. O custo de seguir é um run de 40K steps com configuração
diferente da que o YAML declara, com o aviso perdido num log de quatro ranks.
Agora levanta. Um `TypeError` no segundo 3 custa um restart; o silêncio custa a
rodada.

(No run da fase 1 esse aviso **não** apareceu — a configuração chegou inteira.)

## R3 — O FLUX pelo repo-id: a causa de dois lançamentos mortos hoje

Às 14h47 de 2026-09-25 dois containers (`julia_condgeom_A`, `julia_condgeom_A2`)
morreram com:

```
FileNotFoundError: .../hub/models--black-forest-labs--FLUX.1-dev/blobs
```

Causa, medida: o diretório do FLUX nesta máquina é **root-owned**
(`drwxr-xr-x root root`, de uma rodada antiga que correu sem `--user`). Passar o
**repo-id** manda o diffusers pelo `hf_hub_download`, que quer ESCREVER em
`blobs/`. Passar o **caminho do snapshot** faz ele ler do disco e pronto — foi
assim que a fase 1 de 40K steps rodou.

Dois problemas somados:

- `FLUX_PATH` era honrado por `monitor_externo.py`, `teste_gpu_h100n1.py` e
  `valida_rota_a_parcial.py`, e **não** pelo entrypoint do treino. Variável que
  só metade do código lê é a mesma armadilha de novo.
- todos os YAMLs traziam o repo-id como valor default.

Correção: `_resolver_caminho_do_flux()` no `train.py` honra `FLUX_PATH` e
**recusa** um caminho que não seja diretório local, nomeando o snapshot certo;
`PERMITIR_DOWNLOAD_DO_FLUX=1` é a saída explícita. Os sete YAMLs passaram a
apontar o snapshot. Isto também serve à regra do projeto de não baixar do Hub
(552 GB de egress inexplicado medidos aqui): um repo-id que *funciona* é pior
que um que falha — ele baixa.

Verificado: com `FLUX_PATH=black-forest-labs/FLUX.1-dev` o treino para com a
mensagem, antes de tocar na rede.

## R4 — O config da fase 1 não reproduzia a fase 1

Diff do `configs/train_bokeh_fase1_h100n1.yaml` contra o
`outputs/bokeh_fase1_1700cenas/effective_config.yaml` (que o próprio código
grava, e é a fonte da verdade). Descontando os defaults materializados, sobraram
**quatro divergências reais**:

| chave | no repo | o que rodou |
|---|---|---|
| `runtime.output_dir` | `outputs/bokeh_fase1_rota_a` | `outputs/bokeh_fase1_1700cenas` |
| `logging.run_name` | `bokeh-fase1-rota-a-h100n1` | `bokeh-fase1-rota-a-1700cenas` |
| `data.bokeh.datasets` | `/workspace/releases/rota_a` | `/workspace/retreinar-bokeh/fase1_dados/parcial` |
| `mirror_roots` | **só `EBB!`** | `EBB!` **e** `GenerativePhotography` |

A última é a que importa: com um espelho só, o loader levanta `ReleaseError` na
primeira amostra que referencia a outra fonte — e foi assim que um lançamento
morreu. O loader estava certo (falha alto em vez de adivinhar); o config é que
estava incompleto. Os quatro foram alinhados ao que rodou.

## R5 — `bokeh-shape` sem `--init-lora`

`_resolve_init_lora(None)` estoura um `TypeError` opaco do `Path()`, e a
mensagem de `run_bokeh_shape_stage` — que explica por que a fase de forma precisa
do LoRA da fase 2 — nunca é impressa. Guarda posta no `train.py`. (O `argparse`
já marca `--init-lora` como obrigatório, então a guarda protege o chamador
programático, não a CLI.)

## R6 — Dois testes estavam errados, não o código

`test_trainer_repassa_geo_nos_dois_sites_de_runtime` exigia que o padrão
aparecesse em **dois** sites. Mas o `_dataset_runtime` virou site **único**, que
é mais forte — a duplicação era justamente a origem do defeito que o teste
descrevia. E `outputs.weight == 2` quebrou quando nasceu o terceiro call site da
perda (a validação determinística), estando os três corretos.

Os dois viraram asserções estruturais: *todo* campo `geo_*` do `StageConfig` é
repassado, e *toda* chamada de `flow_matching_loss` passa o peso. O número deixa
de ser o critério.

## R7 — Decisões em aberto agora estão marcadas como tal

`route_weights: null` e `min_calibration_ssim: null` nos configs da fase 2 liam
como default neutro. Não são: `route_weights: null` faz a proporção entre as
rotas b e c virar o **tamanho relativo dos shards** — um acidente do pipeline de
dados — e a Tab. 6 do paper mostra que a composição importa. Ambos ganharam
`PREENCHER` e o porquê, apontando o §10.3 do plano.

---

## Então: está 100% correto?

**Não, e a parte honesta é dizer onde.**

O que está **verificado**, com evidência:

- 252 testes passam sem GPU;
- a fase 1 rodou 40.000 steps, validou em dado retido, e o LVCorr medido por
  fora ficou **plano em 0,983** ao longo de 100 medições — contra +0,9059 da
  fase 1 anterior e +0,8868 dos pesos oficiais;
- acumulação de gradiente conferida **no `accelerate` 1.11.0 instalado**, não de
  memória: `AcceleratedOptimizer.step` e `.zero_grad` são no-op quando
  `sync_gradients` é falso, então `optimizer.step()` fora do bloco de sync está
  certo;
- `run_validation` não tem coletivo nenhum, então rodar só no rank 0 é seguro;
- as guardas novas foram exercitadas em GPU, em 2 processos.

O que **não** está verificado, e não dá para chamar de correto:

| # | o quê | por quê |
|---|---|---|
| 1 | **A fase 2 nunca rodou.** | As rotas B e C não existem como release. O que existe é `tests/test_rotas_b_c.py` contra releases SINTÉTICOS no schema real — isso não substitui o dado de verdade, substitui não ter nada. |
| 2 | **A §3.3 (forma) nunca rodou.** | Falta PointLight-1K e o BokehMe com o kernel da Eq. 6. O lado de treino está pronto e sem dado. |
| 3 | `sigma_mu_source: full_image` | casa com a inferência, mas o paper não menciona σ. Contamina a atribuição de qualquer ganho. |
| 4 | a proporção entre B e C, e o limiar de SSIM | decisões em aberto (§10.3), agora marcadas nos configs. |

### Anexo ao R7 — em multi-GPU o `Bus error` não mata: PENDURA

Confirmado no `julia_verif2` (2 GPUs), observado por 59 minutos:

```
GPU 0:  25.531 MiB,   0% util     <- o rank cujo worker morreu
GPU 1:  25.531 MiB, 100% util     <- o outro, girando no coletivo
```

Nenhuma linha nova de log depois do banner. Em processo único o `Bus error`
derruba o treino em segundos; em multi-GPU um rank perde o worker do DataLoader
e para, o outro gira para sempre num `all_reduce` que nunca fecha, e o container
fica **Up** segurando as duas GPUs. Sem o `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` bem
alto o watchdog acabaria abortando; com ele alto — que é o que a fase 1 usa, e
está certo — o travamento é indefinido.

Ou seja, a guarda de `/dev/shm` no arranque não é conveniência: em multi-GPU ela
é a diferença entre uma mensagem no segundo 3 e GPUs presas até alguém reparar.

E o item que não é do código: **segredo exposto**. `HF_TOKEN` e `WANDB_API_KEY`
aparecem em `docker inspect` do container de treino, numa máquina com containers
de pelo menos cinco outras pessoas. `docker inspect` não pede privilégio. O
`CLAUDE.md` do `bokehnet-regen` já manda os segredos virem de `.env`
git-ignored; o que falta é não passá-los como `--env` no `docker run`. Vale
rotacionar os dois e montar um arquivo em vez de exportar a variável.
