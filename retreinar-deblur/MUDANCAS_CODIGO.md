# O que mudou no código — `retreinar-deblur/` vs. o treino antigo

Esta pasta é uma árvore de treino **nova e independente**. O treino antigo
continua intacto e rodável em `genrefocus_deblurnet_paper/` (cond-only) e
`genrefocus_deblurnet/` (main+cond) — nada aqui os altera.

**Objetivo desta árvore:** replicar o paper (arXiv 2512.16923v3) e a inferência
oficial dos autores. Sem experimentos, sem varredura de hiperparâmetro.

Base da cópia: `genrefocus_deblurnet_paper/` (variante cond-only, que é a que
corresponde à inferência oficial, onde `main_adapter=None`).

---

## 0. Decisões tomadas no lugar dos experimentos

A revisão anterior propunha três experimentos (fatorial de escala, A/B de
seleção por cena, matriz 2×2 de guidance). **Foram cortados.** Onde o paper não
especifica, a regra passou a ser: escolher o valor que reproduz a inferência
oficial; e onde nem isso decide, **manter o comportamento anterior**, porque foi
com ele que o projeto chegou a LPIPS 0,1446 no DPDD contra 0,1440 do paper —
mudar sem medir arriscaria perder isso.

| Eixo | Decisão | Base |
|---|---|---|
| `deblur_train_guidance` | **3,5** | a inferência oficial de deblur roda a 3,5 (`Inference_deblurNet.py` e `demo.py` chamam `generate()` sem `guidance_scale`; default em `flux.py:467`) |
| `cond_train_guidance` | **1,0** | `c_guidances = torch.ones` (`flux.py:616`) |
| `lora_on_main` | **false** | a inferência oficial usa `main_adapter=None` |
| `scale_mode` | **short_side** | mantido. O paper não especifica resolução nem crop de treino |
| `sigma_mu_source` | **crop** | mantido, mesmo motivo |
| `top_k_mode` | **row** | leitura literal do §B.1: *"By ranking the dataset based on this metric"* |
| rank / steps / batch | **128 / 60000 / 32** | §4.1, literal |
| dados | DPDD train inteiro + top-3000 RealBokeh por Laplaciano | §B.1, literal |

Os campos `scale_mode`, `sigma_mu_source` e `top_k_mode` **existem** na config e
os três caminhos estão implementados e testados — mas o default não muda agora.
Ficam disponíveis para quando houver orçamento de medição.

---

## 1. `genfocus_train/config.py` — reescrito

Motivo: nada que muda o modelo podia continuar hardcodado, e um checkpoint sem
registro do que o gerou não é reproduzível.

**Campos novos:**

| Campo | Default | Para quê |
|---|---|---|
| `ModelConfig.expected_lora_modules` | 343 | trava do C1 — aborta se a contagem de módulos LoRA divergir do checkpoint oficial |
| `ModelConfig.deblur_train_guidance` | 3.5 | C2 |
| `ModelConfig.bokeh_train_guidance` | 1.0 | C2 |
| `ModelConfig.cond_train_guidance` | 1.0 | C2 |
| `ModelConfig.lora_on_main` | False | C8 — cond-only vs main+cond por YAML, em vez de duas árvores de código |
| `ModelConfig.lora_on_text` | False | existe só para tornar explícito que o treino nunca põe LoRA no texto (C6) |
| `RuntimeConfig.seed_per_rank` | True | C7 |
| `RuntimeConfig.eval_every_steps` | 500 | C10 |
| `RuntimeConfig.eval_max_samples` | 32 | C10 |
| `RuntimeConfig.eval_num_sigmas` | 9 | C10 — grade fixa de σ |
| `RuntimeConfig.eval_seed` | 1234 | C10 — ruído determinístico |
| `RuntimeConfig.save_best` | True | C10 |
| `StageConfig.val_datasets` | `[]` | C10 |
| `StageConfig.scale_mode` | `"short_side"` | C4 (implementado, default mantido) |
| `StageConfig.sigma_mu_source` | `"crop"` | C4 (idem) |
| `StageConfig.top_k_mode` | `"row"` | C5 (idem) |
| `StageConfig.scene_key` | `"auto"` | C5 |
| `LoggingConfig.wandb_resume` | True | re-`sbatch` continua o mesmo run |
| `LoggingConfig.wandb_id` | None | id estável, persistido em disco |
| `LoggingConfig.upload_every_steps` | 1000 | era 0 (desligado) |
| `LoggingConfig.upload_final` | True | novo |
| `LoggingConfig.upload_best` | True | novo |
| `LoggingConfig.upload_private` | True | novo |
| `LoggingConfig.upload_retries` | 3 | novo |

**Validação nova:** vocabulários fechados (`SCALE_MODES`, `SIGMA_MU_SOURCES`,
`TOP_K_MODES`, `DEFOCUS_SOURCES`) checados no `__post_init__` — um typo no YAML
agora explode em vez de virar comportamento silencioso. Também: `image_size`
múltiplo de 16, e `scale_mode="long_side"` com `batch_size>1` é erro (produz
shape variável).

**Helpers novos:** `stage_config()`, `train_guidance_for()`, `lora_rank_for()`.

**Mudanças de default herdadas:** `gradient_accumulation_steps` 32 → **8** (o
padrão passa a ser 4 GPUs, como no paper), `save_every_steps` 1000 → **250**,
`keep_last_n_checkpoints` 3 → **5**, `wandb_project` → `genrefocus-deblurnet`.

---

## 2. `genfocus_train/backbone.py`

### C1 — LoRA estava sendo injetado numa camada a mais

**Antes:** `LORA_TARGET_MODULES` era uma lista de strings. PEFT casa por
`key == target or key.endswith("." + target)`, e a projeção final do
`FluxTransformer2DModel` chama-se literalmente `proj_out` no topo do módulo —
então `transformer.proj_out` também recebia LoRA. E `transformer_forward`
executa essa projeção **fora** de qualquer `specify_lora` (`flux.py:453`), logo
ela ficava ativa incondicionalmente, inclusive com `main_adapter=None`. A
variante "cond-only" não era cond-only.

**Depois:** regex ancorado, que PEFT resolve com `re.fullmatch`:

```python
LORA_TARGET_MODULES = (
    r"^(transformer_blocks|single_transformer_blocks)\.\d+\."
    r"(attn\.(to_q|to_k|to_v|to_out\.0)|norm1\.linear|ff\.net\.2|"
    r"norm\.linear|proj_mlp|proj_out)$"
    r"|^x_embedder$"
)
```

Mais uma trava dura depois do `add_adapter` que conta `LoraLayer` e aborta se
divergir de `expected_lora_modules`, listando os módulos de topo suspeitos.

**Verificado por execução** (simulando a árvore de módulos do FLUX.1-dev e a
regra de casamento do PEFT): lista antiga → **344** módulos; regex nova →
**343**; diferença = exatamente `proj_out` de topo; zero falsos positivos
(`norm_out.linear`, `norm1_context.linear`, `ff_context.net.2`, `add_*_proj`,
`to_add_out`, `context_embedder` continuam de fora). 343 bate com o header do
`bokehNet.safetensors` oficial (686 tensores ÷ 2).

**Consequência para checkpoints antigos:** eles contêm a chave extra e são
incompatíveis. Ver o guard no `trainer.py` (seção 4).

### C2 — guidance deixou de ser hardcodado

**Antes:** `guidance_main = torch.ones(B)` e `guidance_cond = torch.ones(B)`,
com um comentário afirmando que "usar 3.5 atrapalha convergência".

**Depois:** `torch.full((B,), self.train_guidance)` e `self.cond_guidance`,
vindos da config por estágio. O comentário foi trocado: a afirmação sobre
convergência nunca foi medida neste projeto, e o que é FATO é que a inferência
oficial de deblur roda a 3,5 e a de bokeh a 1,0.

### C4 — `sample_sigma` por amostra + docstring falso corrigido

**Antes:** `sample_sigma(batch_size, seq_len: int, ...)` com um `mu` escalar
derivado dos tokens do crop, e um docstring afirmando *"garantindo distribuição
idêntica entre treino e inferência"*.

**Depois:** `sample_sigma(seq_len: Tensor(B,), ...)` com `mu` vetorizado pela
mesma reta do `calculate_shift` do diffusers, e `_seq_len_para_sigma()`
escolhendo entre `crop` e `full_image`. Com `full_image` e sem `full_seq_len`,
**erro explícito** — nada de fallback silencioso.

O docstring foi corrigido porque era falso: a inferência tira o `mu` do
`image_seq_len` da imagem **inteira**, antes do tiling (`flux.py:624`), enquanto
o treino tirava do crop. `_verificar_mu_bate_com_diffusers()` roda no `load()` e
compara a versão vetorizada com o `calculate_shift` escalar.

### C8 — `lora_on_main` e `lora_info()`

`adapters = [texto, main, *conds]`, com texto **sempre** `None` (o treino nunca
põe LoRA no texto — é o descasamento do C6, documentado no ponto exato) e main
condicionado a `lora_on_main`. `lora_info()` devolve os 9 campos que o trainer
grava no metadata. `create_backbone(config, stage)` monta tudo a partir da
config.

---

## 3. `genfocus_train/data.py`

### C4 — `scale_mode` (implementado; default mantido em `short_side`)

**Antes:** `scale = image_size / min(w, h)` fixo — o lado menor sempre ia para
512, e o crop saía de uma imagem reduzida.

**Depois:** função pura `_plano_geometrico(w, h, image_size, scale_mode, train, rng)`
compartilhada por `prepare_aligned_pair` e `prepare_aligned_bokeh` — duplicar a
conta era o caminho conhecido para as duas divergirem em silêncio. Três modos:

| modo | o que faz | arredondamento a 16 |
|---|---|---|
| `short_side` | lado menor → 512, depois crop 512² | `floor` (= `resize_and_pad_image` com `long_side>0`) |
| `native` | sem resize, crop 512² no pixel nativo | `ceil` (= `resize_and_pad_image(img, 0)`) |
| `long_side` | lado maior → 512, imagem inteira | `floor` |

**Garantia importante:** `short_side` foi verificado **bit a bit idêntico** ao
comportamento antigo em 800 casos — resize, caixa de crop, flip e ordem dos
sorteios do RNG. Como é o default, o treino novo produz exatamente os mesmos
lotes que o antigo.

Todos os modos passam a devolver `full_seq_len` — o nº de tokens da imagem de
origem inteira, que é o que a inferência usa para o `mu` (`flux.py:624`,
`image_seq_len = latents.shape[1]`, antes do tiling).

### C5 — seleção por cena (implementada; default mantido em `row`)

**Antes:** `np.argsort(scores)[::-1][:k]` sobre **linhas**. Como o score sai de
`image_focus`, idêntico em todas as linhas de uma cena, os empates eram exatos e
o top-k puxava cenas inteiras.

**Depois:** `top_k_mode` escolhe. Com `"scene"`: `_scene_keys` resolve a
identidade da cena (coluna `scene`/`stem` → prefixo de `file_name_base` → md5 de
`image_focus`), `_agrupar_por_cena` agrupa, a nitidez é medida **uma vez por
cena**, o ranking é de cenas, `__len__` vira o nº de cenas e `__getitem__`
sorteia a linha. Quando nenhuma estratégia agrupa (caso do DPDD), **não é erro**:
avisa e segue com 1 linha = 1 cena.

Bug de índice evitado e comentado no código: o sorteio devolve a posição no
grupo, não o índice global — `row_i = grupo[pos]`.

### C5-a — chave do cache (bug fechado, aplicado sempre)

**Antes:** `f"{safe}_{column}_top{k}_n{n}"`. Nenhum campo mudava com a lógica de
seleção, então um seletor novo leria os índices antigos em silêncio.

**Depois:** `_FILTER_SELECTOR_VERSION = "v2"` mais `top_k_mode` e `scene_key` na
chave; o payload virou dict com versão e modo validados na leitura.

### C9 — `select_columns` antes do concat

**Antes:** `concatenate_datasets` direto. Schema divergente entre `DDPD` e
`RealBokeh` explodiria **depois** dos ~20 min do filtro de nitidez, e
`image_pre_deblur` era carregado sem uso.

**Depois:** `_DEBLUR_KEEP` + `select_columns` dentro de `_load_hf_split`, antes
do filtro, com validação de coluna por fonte e interseção antes do concat.

### C10 — dataset de validação

Novo: `build_val_dataset(stage, stage_config, runtime, max_samples=None) -> Dataset | None`.
Devolve `None` sem `val_datasets`. Força `train=False` (crop central, sem flip) e
`top_k_mode="row"`; mantém o `scale_mode` do treino de propósito.

---

## 4. `genfocus_train/trainer.py`

### C3 — warmup: correção de UMA linha, não duas

**Antes:** o `param_group["lr"]` nascia com `optimizer.lr` (1e-4) e o scheduler
só era aplicado **depois** do primeiro `optimizer.step()`. Resultado: o primeiro
update rodava a 1,0e-04 quando a rampa prevê 2,0e-07 — 500× acima.

**Depois:** `scheduler.step(state.global_step)` uma vez antes do laço (e no
resume, reposicionando a curva).

> **Correção de um erro meu na análise.** Eu havia proposto *também* trocar
> `float(step + 1)` por `float(max(step, 1))` em `_lr_at`. Está **errado** e não
> foi aplicado. Verificado numericamente: com o `scheduler.step()` inicial, o
> update *k* usa `_lr_at(k-1)`, e a sequência-alvo 2e-7 → 4e-7 → 6e-7 → 8e-7 só
> sai com a fórmula **original**. `max(step,1)` repetiria o primeiro valor
> (2e-7, 2e-7, 4e-7…) e faria a rampa terminar em 9,98e-5 em vez de emendar no
> `base_lr`. A fórmula nunca esteve errada — o bug era ela não ser aplicada
> antes do primeiro update.

### C7 — seed por rank

**Antes:** `_set_seed(config.runtime.seed)` idêntico em todos os processos: as 4
GPUs sorteavam o mesmo σ e o mesmo ruído, e o batch efetivo 32 tinha só 8 σ
distintos.

**Depois:** `_set_seed` movido para depois do `_build_accelerator`, somando
`accelerator.process_index` sob `seed_per_rank`. Seguro porque os pesos LoRA já
são sincronizados por broadcast explícito do rank 0. O smoke mantém a seed pura
(é single-process e precisa ser reprodutível).

### C8 — metadata e recusa de checkpoint pré-C1

`build_run_metadata()` reúne `lora_info()` + eixos + prompt + batch efetivo, e
vai para três lugares: o checkpoint, um `.json` irmão de todo `.safetensors`
exportado, e o `config` do run do wandb.

`_recusar_checkpoint_pre_c1()` levanta erro em `maybe_resume_checkpoint` **e** em
`load_lora_checkpoint_into_backbone`. Antes, uma chave `unexpected` só gerava um
WARN — e depois do C1 um checkpoint antigo traria `proj_out.lora_A.default.weight`,
que seria descartado em silêncio, mudando o modelo sem avisar.

### C10 — validação determinística

**Antes:** não existia. `best_loss` era a loss de treino de um micro-batch com σ
e ruído sorteados, ou seja, não era sinal de nada.

**Depois:** `run_validation()` a cada `eval_every_steps`, com grade
`linspace(0.1, 0.9, n)` em **rodízio** por amostra (não produto cartesiano —
9×32 forwards custariam minutos a cada 500 steps; assim são 32 cobrindo a
grade), ruído de `Generator` com `manual_seed(eval_seed)`, e o RNG global salvo
e restaurado para a validação não deslocar a sequência do treino. Alimenta
`deblur/val_loss` e o `best.pt`.

### wandb — o re-`sbatch` continua o mesmo gráfico

**Antes:** `wandb.init` abria um run NOVO a cada re-`sbatch`, picando a curva da
loss em N runs.

**Depois:** id estável (sha1 de `output_dir::stage`) persistido em
`<output_dir>/<stage>/wandb_id.txt`, com `resume="allow"`. O `config` do run
carrega o `lora_info()` e os eixos, para os braços ficarem comparáveis.

### Hugging Face — upload que não perde nada e não derruba o treino

**Antes:** `upload_every_steps` default 0 (desligado), sem retry, repo público,
sem sidecar.

**Depois:** default 1000, retries com backoff exponencial, `private=True`, sobe
também o `.json` de metadata, rótulos `best` e `final_stepN`, e aviso único
quando `upload_hf_repo_base` é nulo. Falha de rede ou token **nunca** derruba o
treino — só avisa.

### Checkpoints — nada se perde

Escrita atômica (`.tmp` + `os.replace`), `best.pt` fora da série `step_*.pt` e
portanto **nunca podado**, `latest.json` apontando para o último, salvamento
final em `finally` protegido para não mascarar a exceção que abortou o treino.
`save_every_steps: 250`, `keep_last_n_checkpoints: 5`.

---

## 5. `genfocus_train/models.py` e `train.py`

- `make_train_batch` dos dois modelos aceita e repassa `full_seq_len`.
- `create_backbone(config, stage)` (era `create_backbone(config.model, stage=stage)`).
- Docstring do `models.py` corrigido: afirmava "guidance 1.0 na inferência" sem
  qualificar o estágio. O correto é que **bokeh** passa 1.0 explicitamente e
  **deblur** cai no default 3.5.
- Registrado no docstring que o branch de TEXTO nunca recebe LoRA no treino, e
  que `lora_on_main=true` exige `main_adapter="deblurring"` na inferência.
- Subcomando novo **`check`**: não treina. Carrega o config, imprime
  `lora_info()`, os eixos, o batch efetivo, os steps, e valida `HF_TOKEN` /
  `WANDB_API_KEY` quando o config pede upload/wandb. É o que se roda antes de
  gastar fila.
- `export` passa a escrever o `.json` sidecar também.

---

## 6. `configs/`

| Arquivo | O que é |
|---|---|
| `train_deblur_paper.yaml` | **o treino.** Replica §4.1 + §B.1, cada valor com a citação literal no comentário |
| `train_deblur_maincond.yaml` | idêntico exceto `lora_on_main: true` (+ output_dir/run_name/repo). Difere em 4 linhas |
| `train_smoke.yaml` | 3 steps, wandb off, upload off |

Os três foram validados pelo `load_config` real, não só por `yaml.safe_load` — o
que também prova que os campos novos não caem na armadilha do `_as_stage_config`.

---

## 7. `slurm/`

`train_deblur_4gpu.slurm`, `train_deblur_1gpu.slurm`, `smoke.slurm`,
`medicoes_c0.slurm`. Todos com `--time` alto (14 dias no treino, conforme a
regra do `CLAUDE.md`), `logs/` criado, `.env` carregado com falha cedo se faltar
token, versões de torch/peft/diffusers/transformers impressas antes de começar,
e re-`sbatch`-áveis (o `resume: true` do config retoma de onde parou).

**Nenhum script contém `scancel`, `rm` ou `--delete`** — verificado por grep.

Três divergências deliberadas em relação ao `train_4gpu.slurm` antigo, seguindo
o `INSTRUCOES_H100.md`: sem `PEFT_PIN`, sem sobrescrever `PYTHONUSERBASE`, e sem
passar `--env PYTHONPATH` ao singularity.

---

## 8. `scripts/` e `tests/`

Scripts de diagnóstico (`c0_1_resolucao_dfs.py`, `c0_2_gts_distintas.py`,
`c0_3_contagem_lora.py`, `verificar_mu.py`) e `infer_deblur.py`, que expõe por
CLI todos os eixos que a inferência oficial fixa: `--guidance-scale`,
`--main-adapter`, `--text-adapter`, `--long-side`, `--no-tiling`.

O `--text-adapter` é o C6: o `generate` oficial monta `adapters = [main_adapter]*2 + c_adapters`
e o índice 0 é o **texto**, então `main_adapter="deblurring"` liga LoRA num
branch que o treino nunca treinou. Como `third_party/Genfocus/` é referência e
não pode ser tocado, o desvio é local ao script.

Testes que rodam sem GPU e sem rede: **35 passando, 39 subtestes**. Cobrem o
round-trip de todo campo de `StageConfig` (falha se alguém adicionar campo na
dataclass e esquecer do `_as_stage_config`), a rejeição de valores fora dos
vocabulários, a rampa do warmup, a reta do `mu`, e o casamento da regex do LoRA.

---

## 9. Execução — o treino no cluster (2026-09-10)

### Configuração real

| | |
|---|---|
| Nó | `dgx-H100-03`, partição `h100n3` |
| GPUs | **2** (índices 1 e 2, 54,4 GB livres na mais apertada) |
| `grad_accum` | **16** → batch efetivo `1 × 16 × 2 = 32`, exatamente o §4.1 |
| VRAM de pico | **33,8 GB** por GPU (gradient checkpointing ligado) |
| Tempo por step | ~15 s → 60K steps ≈ 10,5 dias (`--time=20-00:00:00`) |
| wandb | `genrefocus-deblurnet-v2`, run `3787375c` |
| HF | `juliadollis/genrefocus-deblurnet-v2-paper`, privado, **um arquivo por step** |

### Por que 2 GPUs e não 4

Medido, não suposto: `scontrol show node dgx-H100-03` devolve `Gres=gpu:h100:3`.
O SLURM da h100n3 gerencia **3** das 8 GPUs físicas — as outras 5 rodam VLLM de
terceiros, fora do SLURM. Com 1 já alocada, sobravam 2. E `h100n2` (8 GPUs) e
`b200n1` (8) estavam ambos com `AllocTRES=gres/gpu=8`, ou seja, **zero** livres.
Um `--gres=gpu:4` ficaria PENDING indefinidamente.

O batch efetivo foi preservado em 32 pela acumulação, então o treino é o do
paper — só mais lento em relógio de parede. O script deriva `grad_accum` do
número de GPUs recebidas, então um `--gres=gpu:4` futuro se ajusta sozinho.

**Não tomamos GPU fora da alocação do SLURM.** O `INSTRUCOES_H100.md` permite
escolher GPU livre à mão no caso de 1 GPU, mas com 5 GPUs fora do controle do
SLURM isso arriscaria colidir com job de terceiro. O script confere a memória
livre do que recebeu e **aborta** se não couber, em vez de pegar a de outro.

### O que foi confirmado em produção

```
[lora] 343 módulos | rank=128 alpha=128 | 463.6M params treináveis | variante=cond-only
[train] start=500 num_gpus=2 grad_accum=16 batch_efetivo=32 seed=42 / seed=43
        lora=cond-only guidance=3.5 scale_mode=short_side sigma_mu=crop
[logger] wandb: project=genrefocus-deblurnet-v2 id=3787375c resume=True
```

- **C1**: 343 módulos, não 344. O `proj_out` de topo saiu.
- **C2**: `guidance=3.5`, igual à inferência oficial de deblur.
- **C3**: no step 190 o LR era `3.82e-05` = `1e-4 × 190/500` — a rampa do warmup
  está correta e contínua, e chega a `1.00e-04` exatamente no step 500.
- **C7**: `seed=42` no rank 0 e `seed=43` no rank 1.
- **C8/C10**: `val=0.3023` no step 500, `best.pt` escrito e subido ao HF.
- Export: **686 tensores** = 343 módulos × 2 (`lora_A` + `lora_B`).

### Queda e recuperação automática

O job **32184** morreu em ~40 min com
`OSError: [Errno 107] Transport endpoint is not connected` ao ler
`/usr/local/lib/python3.10/dist-packages/rich-.../METADATA` — o mount do
`.sif` caiu. **Não é bug do nosso código**; é infraestrutura.

A corrente de continuações `--dependency=afterany` fez o previsto: **32185**
assumiu sozinho, carregou `step_500.pt`, retomou em `start=500` e continuou no
**mesmo** run do wandb. Perda: ~40 steps.

Corrente enfileirada para a noite: `32185 → 32186 → … → 32193` (9 elos). Cada
elo retoma do último checkpoint. Se o mount cair de novo, o treino se levanta
sozinho sem intervenção.

### Persistência — nada se perde

- `step_*.pt` a cada 250 steps, 5 mais recentes mantidos (~5,5 GB cada).
- `best.pt` por `val_loss`, **fora** da série podada.
- Upload ao HF a cada 1000 steps, **arquivo por step** (`deblur_step<N>.safetensors`),
  mais `deblur_best.safetensors` — nada é sobrescrito.
- Cota: 364 G de 500 G após os datasets. Folga para a série de checkpoints.

---

## 10. Achados da auditoria que NÃO foram corrigidos (deliberado)

Auditoria independente do código novo contra o paper e a inferência oficial.
Nada aqui bloqueia o treino; está registrado para não se perder.

### Divergências assumidas

| # | O quê | Situação |
|---|---|---|
| 1 | `sigma_mu_source: crop` — a inferência tira o μ do `seq_len` da imagem inteira antes do tiling; o treino tira do crop | Escolha registrada. O caminho `full_image` existe e funciona |
| 2 | `scale_mode: short_side` — a inferência com `long_side=0` usa tiles de 512² **nativos** | Escolha registrada. `native` existe e funciona |
| 3 | O Laplaciano é medido num proxy 256², não na imagem inteira (`data.py:509`). O §B.1 diz *"of each image"*, e variância do Laplaciano depende de escala | **Não estava no plano.** Achado novo — o ranking top-3000 não é o de resolução nativa |
| 4 | `deblur_train_guidance: 3.5` casa com a inferência, mas o paper **não menciona guidance em lugar nenhum** | Não tratar 3,5 como "do paper" |

### Contratos frouxos entre arquivos (nenhum quebra o treino)

| # | O quê | Efeito |
|---|---|---|
| 1 | `trainer.py` passa `sigma_mu_source` ao `DatasetRuntimeConfig`, que não tem esse campo | Aviso em todo run. O guard filtra, mas treinar o operador a ignorar aviso anula o guard |
| 2 | `build_val_dataset` (`data.py`) não tem chamador — o trainer usa outro caminho, que **não** força `top_k_mode="row"` na validação | Código morto + duas implementações do mesmo contrato |
| 3 | `model.vae_subfolder` / `transformer_subfolder` estão nos YAMLs e ninguém lê | Mudar no YAML não tem efeito e não avisa |
| 4 | `run_validation` não repassa os pesos de oclusão | Irrelevante para deblur (`occlusion_lambda=0`); quebraria a seleção de `best.pt` no BokehNet |

**Não sincronizei correções para o cluster de propósito.** Os jobs enfileirados
leem o Python no momento em que começam, então subir código novo agora faria uma
continuação pegar código não testado no meio da noite. As correções ficam para
depois, com smoke antes.

### Erros encontrados nos registros do projeto

- `HANDOFF_PROJECT_HISTORY.md` §7 lista como "Tabela 2 publicada"
  `GenRefocus: LPIPS 0.1598 | FID 33.08`. No v3 o GenRefocus no DPDD é
  **LPIPS 0.1440 / DISTS 0.0772**, e **0.1598 é o Bokehlicious**; o v3 não tem
  coluna FID. Com o número errado, o nosso 0.1446 parecia vitória folgada —
  com o real, é empate técnico.
- A referência **[58]** do paper, citada como *"our backbone is FLUX-1-dev [58]"*,
  aponta para `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro`. Erro de citação
  dos próprios autores; não afeta nosso código.

---

## 11. Rota Docker — dgx-H100-01, 4 GPUs, sem SLURM

A H100-01 não usa SLURM: os treinos sobem em Docker direto, e **não há fila
protegendo ninguém**. As regras vieram do `DECISOES_FASE2.md` e do
`REGISTRO_GEO_COND.md`, e cada uma custou um incidente:

| Regra | Por quê |
|---|---|
| `--user $(id -u):$(id -g)` em todo `docker run` | sem isso os arquivos nascem root — é a origem dos 54 GB de `hf-cache` root-owned que ninguém consegue apagar |
| container **detached** | sobrevive a queda de ssh/VPN (já aconteceu 2×) |
| `--gpus '"device=0,1,2,3"'` com aspas embutidas | com várias GPUs a forma sem aspas não funciona |
| montar o `julia_docker` **inteiro** em `/workspace` | montar só a pasta do projeto dá `No such file or directory` |
| 4 das 8 GPUs | deixa 4 para os outros; há containers de 4 outras pessoas no host |
| nunca `docker rm`/`rmi`/`prune`/`stop` alheio | idem |

`docker/run_docker_h100n1.sh` implementa tudo isso e **só cria — nunca remove**.
Se já existir container com o mesmo nome, ele **para e avisa** em vez de apagar.

### Batch com 4 GPUs

`batch_size: 4 × gradient_accumulation_steps: 2 × 4 GPUs = 32`. O batch efetivo
é **idêntico ao §4.1**; muda só o fatiamento. É equivalente aqui: não há
BatchNorm, o transformer está congelado, e o σ é sorteado **por amostra** — 4
amostras num forward são 4 σ, igual a acumular 4 passos.

`gradient_checkpointing: false` (era `true` no SLURM): ~30% mais rápido em troca
de guardar as ativações. Só faz sentido com a GPU inteira nossa.
`configs/train_deblur_docker_4gpu_gc.yaml` é a reserva com GC ligado, para o
caso de OOM.

### Nomes distintos — nada é sobrescrito

| | SLURM (h100n3) | Docker (h100n1) |
|---|---|---|
| `output_dir` | `outputs/deblur_paper` | `outputs/deblur_docker_4gpu` |
| wandb projeto | `genrefocus-deblurnet-v2` | `genrefocus-deblurnet-docker` |
| wandb run | `deblur-paper-60k` | `deblur-docker-4gpu-h100n1` |
| repo HF | `...-v2-paper` | `...-docker-4gpu` |

O id do wandb é derivado de `sha1(output_dir::stage)`, então os dois runs são
independentes por construção.

### Defeito encontrado e corrigido: cache HF root-owned

Primeira tentativa morreu com
`PermissionError: [Errno 13] '/workspace/hf-cache/hub/datasets--akcit-pixel--DDPD'`.
Diagnóstico: `hf-cache/` é do usuário, mas `hf-cache/hub/` é **root:root** — o
tal cache da rodada anterior. Com `--user` (que é a regra certa) não dá para
escrever nele.

Correção sem tocar em nada root-owned: um hub paralelo `hf-cache-v2/hub`, do
usuário, com **symlink** para o `models--black-forest-labs--FLUX.1-dev` já
baixado. Lê do cache antigo, escreve no novo. O script ganhou `HF_CACHE`
parametrizável e um guard que aborta se o hub escolhido não for escrevível —
em vez de tentar mudar dono de arquivo.

Sinal de que só a escrita falhava: `[lora] 343 módulos` apareceu nos **4 ranks**
antes do erro, ou seja, FLUX e LoRA carregaram do cache root-owned sem problema.

### Correcao de uma afirmacao minha sobre o cache

Registrei antes que o `hf-cache-v2` era "106 GB de duplicacao, erro meu". **Os
dois numeros estavam errados e o remedio tambem.**

O que e de fato duplicado entre `hf-cache-v2` e `hf-cache-julia` sao ~11,5 GB (o
DDPD e um RealBokeh parcial de 2,5 GB). Os outros ~94 GB do `hf-cache-v2` sao
dados que **so existem ali**: o RealBokeh completo (44 GB) e a conversao arrow.

E o conserto que propus — "apontar as proximas rodadas para o `hf-cache-julia`" —
faria o oposto do pretendido: como aquele cache tem so 2,5 GB dos 46,9 GB do
RealBokeh, forcaria rebaixar 44 GB. O certo e o `hf-cache-v2` continuar sendo o
cache do DeblurNet, e nao criar um quarto. Isso agora esta escrito no proprio
`docker/run_docker_h100n1.sh`.

O que se sustenta da autocritica original: criar um cache novo em vez de olhar
antes o que ja existia custou um re-download do DDPD (9 GB).

### Limpeza de disco (2026-09-10)

Apagados, com autorizacao explicita, 92 GB:

| Caminho | Tamanho |
|---|---|
| `julia_docker/data/spring_prep` | 12 GB |
| `julia_docker/data/spring_split` | 23 GB |
| `julia_docker/runs_riemann` | 57 GB |

Antes de apagar o `runs_riemann`, os **119 arquivos de metrica**
(`test_summary.json`, `test_metrics.json`, `por_imagem.csv`, `meta.json`) foram
copiados para `julia_docker/runs_riemann_metricas_preservadas/` (2,7 MB). O peso
estava nos `seed_*`, 1,3 GB cada; os resultados dos experimentos ficaram.

**Nao tocado, e por que:**
- `Build Cache` do Docker (2,79 TB) e `Images` (640 GB recuperaveis): sao de
  TODOS os usuarios do host. Um `prune` apagaria o trabalho de iago, emanuel e
  hyago junto.
- `hf-cache` (54 GB, root): sem permissao, e o symlink do FLUX do treino aponta
  para la.
- `hf-cache-julia` (284 GB): tres containers vivos montam.
- `genrefocus_deblurnet_paper/outputs` (29 GB): e o treino antigo, a preservar.
- `data/spring` (40 GB): fonte dos derivados apagados.
