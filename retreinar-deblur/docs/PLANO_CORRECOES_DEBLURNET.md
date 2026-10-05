# Plano de correções — DeblurNet (Stage 1)

Auditoria de `genfocus_train/` contra o paper (arXiv 2512.16923v3) e a inferência
oficial (`third_party/Genfocus/`). Cada item traz: **como está**, **por que é
problema**, **como tem que ficar**, **como verificar**.

Duas árvores de código são afetadas. Onde disser "nas duas", aplique em:
- `genrefocus_deblurnet_paper/genfocus_train/` (variante cond-only)
- `genrefocus_deblurnet/genfocus_train/`       (variante main+cond, modelo já treinado a 60k)

## Ordem de execução (não reordene)

1. **C0** — medições que não mudam código (decidem C4 e confirmam C1/C3/C5)
2. **C1** — remover `proj_out` de topo do LoRA  ← muda a arquitetura, tem que vir antes de qualquer treino novo
3. **C3** — warmup
4. **C7**, **C8**, **C9** — baratos, sem efeito sobre resultado
5. **C5** — top-3000 por cena
6. **C4** — regime de escala + sigma (exige os runs curtos do C0-4)
7. **C2** — guidance (medir só DEPOIS do C1, porque o C1 muda quem é afetado)
8. **C6** — inferência do modelo main+cond (não bloqueia o treino)
9. **C10** — validação
10. **C11** — dfs

---

## C0. Medições prévias (nenhuma mudança de código)

Rodar antes de tudo. Três resultados decidem C4, C5 e C11.

### C0-1. Resolução realmente armazenada nos dfs

```python
import os
from datasets import load_dataset
tok = os.environ["HF_TOKEN"]
for repo, split in [("akcit-pixel/DDPD","train"), ("akcit-pixel/RealBokeh","train")]:
    st = load_dataset(repo, split=split, streaming=True, token=tok)
    for i, r in zip(range(20), st):
        print(repo, r["file_name_base"], r["image_blur"].size, r["image_focus"].size)
```

Anotar `W×H` de cada fonte. **Se o DDPD não estiver em 1680×1120**, o df já
reduziu o blur em relação ao paper, e o LPIPS da Tabela 2 não é diretamente
comparável ao publicado. Registrar isso explicitamente (ver C11-1).

### C0-2. Quantas GTs distintas o top-3000 realmente seleciona

```python
import os, json, hashlib
from datasets import load_dataset
tok = os.environ["HF_TOKEN"]
ds = load_dataset("akcit-pixel/RealBokeh", split="train", token=tok)
cache = os.path.expanduser("~/.cache/huggingface/genfocus_filter_cache/"
                           f"akcit-pixel_RealBokeh_train_image_focus_top3000_n{len(ds)}.json")
idx = json.load(open(cache))          # se não existir, rode o treino uma vez p/ gerar
sub = ds.select(idx).select_columns(["image_focus"])
h = {hashlib.md5(r["image_focus"].tobytes()).hexdigest() for r in sub}
print("linhas selecionadas:", len(idx), "| GTs distintas:", len(h))
```

Hipótese: ~580 GTs distintas para 3000 linhas. Confirma T3.

### C0-3. Contagem de módulos LoRA injetados

```python
# depois de backbone.load(...), antes de treinar
from peft.tuners.lora import LoraLayer
mods = [n for n, m in backbone.transformer.named_modules() if isinstance(m, LoraLayer)]
print(len(mods))                                  # hoje: 344 | oficial: 343
print([n for n in mods if n.count(".") == 0])     # hoje: ['x_embedder', 'proj_out']
```

Confirma C1.

### C0-4. Runs curtos para decidir o C4

Depois de C1+C3, rodar **3 treinos de ~3000 steps** (mesma seed, mesmos dados) e
avaliar cada um no seu próprio regime + nos outros dois. Detalhe em C4.

---

## C1. `proj_out` de topo está recebendo LoRA (nas duas árvores)

**Arquivo:** `genfocus_train/backbone.py:83` (idêntico nas duas árvores)

**Como está**

```python
LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "norm1.linear",
    "ff.net.2",
    "norm.linear",
    "proj_mlp",
    "proj_out",        # ← linha 83
    "x_embedder",
]
```

**Por que é problema**

PEFT casa alvo por `key == target or key.endswith("." + target)`. A projeção
final do `FluxTransformer2DModel` chama-se literalmente `proj_out` no topo do
módulo, então além dos 38 `single_transformer_blocks.N.proj_out` o PEFT também
injeta LoRA em `transformer.proj_out`.

Essa camada é executada em `Genfocus/pipeline/flux.py`:

```python
image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
output = self.proj_out(image_hidden_states)      # ← FORA de qualquer specify_lora
```

Sem `specify_lora`, o `scaling` do adapter fica no valor natural (`alpha/r = 1`)
e a LoRA age **incondicionalmente**, inclusive quando `main_adapter=None`. Duas
consequências:

1. A variante `_paper` **não é** "LoRA só na condição": existe um módulo
   treinável agindo direto na saída do branch principal.
2. A entrada desse `proj_out` depende do `temb` do main, que é construído com o
   `guidance`. É exatamente por ali que o descasamento de guidance (C2) entra
   mesmo na variante cond-only.

Prova por contagem, contra o header do `bokehNet.safetensors` oficial já dumpado
(686 tensores / **343** módulos LoRA):

```
19 duplos  × (to_q, to_k, to_v, to_out.0, norm1.linear, ff.net.2) = 114
38 single  × (to_q, to_k, to_v, norm.linear, proj_mlp, proj_out)  = 228
x_embedder                                                        =   1
                                                            total = 343   oficial
+ transformer.proj_out (topo)                                     =   1
                                                            total = 344   nosso
```

Conferido que **não há** outros falsos positivos: `norm_out.linear`,
`norm1_context.linear`, `ff_context.net.2`, `add_q_proj/add_k_proj/add_v_proj`
e `to_add_out` não casam com nenhum alvo da lista.

**Como tem que ficar**

Trocar as strings genéricas por regex ancorado nos blocos, para que a projeção
de topo nunca case. Substituir a lista inteira por:

```python
# Alvos ancorados nos BLOCOS. Strings soltas não servem: PEFT casa por
# `key == target or key.endswith("." + target)`, e a projeção final do
# FluxTransformer2DModel chama-se `proj_out` no topo do módulo — ela seria
# capturada e ficaria ativa FORA do controle por branch do specify_lora
# (`transformer_forward` chama self.proj_out sem specify_lora). O checkpoint
# oficial tem 343 módulos LoRA; com a string solta injetávamos 344.
LORA_TARGET_MODULES = r"^(transformer_blocks|single_transformer_blocks)\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|norm1\.linear|ff\.net\.2|norm\.linear|proj_mlp|proj_out)$|^x_embedder$"
```

`LoraConfig` aceita `target_modules` como `str` e, nesse caso, PEFT usa
`re.fullmatch(target_modules, key)` — por isso as âncoras `^...$` são
obrigatórias.

Adicionar em `_inject_lora()`, logo depois do `add_adapter`, uma trava dura:

```python
from peft.tuners.lora import LoraLayer
lora_modules = [n for n, m in self.transformer.named_modules() if isinstance(m, LoraLayer)]
if len(lora_modules) != 343:
    raise RuntimeError(
        f"LoRA injetado em {len(lora_modules)} módulos; o checkpoint oficial tem 343. "
        f"Módulos de topo (suspeitos): {[n for n in lora_modules if '.' not in n]}"
    )
```

**Migração dos checkpoints existentes (obrigatório ler)**

Depois desta mudança, `maybe_resume_checkpoint` (`trainer.py`) vai encontrar a
chave `proj_out.lora_A.default.weight` no checkpoint antigo, classificá-la como
`unexpected` e **apenas imprimir um WARN, descartando o peso em silêncio**. Isso
muda o modelo sem avisar. Portanto:

- Treinos novos: **começam do zero**, não retomam checkpoint pré-C1.
- Em `maybe_resume_checkpoint`, trocar o WARN por erro quando o `unexpected`
  contiver `proj_out` sem ponto:

```python
topo = [n for n in unexpected if "." not in n.split(".lora_")[0]]
if topo:
    raise RuntimeError(
        f"Checkpoint anterior ao C1 (contém LoRA em módulo de topo: {topo}). "
        "Não é compatível com a arquitetura corrigida — treine do zero."
    )
```

- O peso já publicado em `juliadollis/genrefocus-deblurnet-paper-4gpu` contém a
  chave extra e é **autoconsistente** na inferência (o diffusers recria a LoRA
  em `transformer.proj_out` ao carregar). Ele continua avaliável — mas **não é
  comparável cabeça a cabeça** com um modelo pós-C1. Registrar isso na tabela.

**Verificar:** C0-3 tem que passar de 344 para 343, e a lista de módulos de topo
tem que conter só `x_embedder`.

---

## C2. `guidance` do treino não bate com a inferência oficial do Deblur

**Arquivo:** `genfocus_train/backbone.py:383-384` (nas duas árvores)

**Como está**

```python
guidance_main = torch.ones(B, device=device, dtype=dtype)   # 1.0
guidance_cond = torch.ones(B, device=device, dtype=dtype)   # 1.0
```

**Por que é problema**

A inferência oficial usa valores **diferentes por estágio**:

| origem | chamada | guidance efetivo `[texto, main, cond]` |
|---|---|---|
| `Inference_bokehNet.py:175` | `generate(..., guidance_scale=1.0)` | `[1.0, 1.0, 1.0]` |
| `demo.py:240` (bokeh) | `generate(..., guidance_scale=1.0)` | `[1.0, 1.0, 1.0]` |
| `Inference_deblurNet.py:~104` | `generate(...)` **sem** `guidance_scale` | `[3.5, 3.5, 1.0]` |
| `demo.py:168` (deblur) | `generate(...)` **sem** `guidance_scale` | `[3.5, 3.5, 1.0]` |

O default é `guidance_scale: float = 3.5` (`Genfocus/pipeline/flux.py:467`), e
`flux.py:707` monta `guidances = [guidance]*2 + c_guidances` com
`c_guidances = [torch.ones(1)]`. Os autores **baixaram de propósito** a guidance
no BokehNet e **deixaram 3,5** no DeblurNet — duas fontes independentes
(`Inference_*.py` e `demo.py`) confirmam.

A tabela de auditoria em `historico-ultimo.md:283` registra "Guidance: todas 1.0
— confere com a inferência de bokeh". Está certa **para o BokehNet**; a
conclusão foi carregada para o DeblurNet sem reconferir, e lá ela não vale.

O `guidance` entra só via `get_temb` → `time_text_embed(timestep, guidance,
pooled)`, que produz o vetor de modulação `temb` daquele branch. A severidade
depende de quais módulos LoRA consomem esse `temb`:

- **cond-only pós-C1**: nenhum LoRA no main → 3,5 vs 1,0 muda só features do
  FLUX base. Efeito indireto, leve.
- **cond-only pré-C1**: o `proj_out` de topo age sobre um hidden state modulado
  com 3,5 na inferência e 1,0 no treino. Moderado.
- **main+cond**: `norm1.linear` (19 blocos) e `norm.linear` (38 single) recebem
  o `temb` **como entrada direta**. Treinado com `temb(1.0)`, rodado com
  `temb(3.5)`. Severo — e é a variante do modelo de 60k já avaliado.

**Como tem que ficar**

Parar de hardcodar. Adicionar em `ModelConfig` (`config.py`):

```python
# Guidance embutido no temb de cada branch. NÃO é CFG — o FLUX.1-dev é
# guidance-distilled e recebe esse escalar como entrada do modelo.
# Tem que ser o MESMO da inferência do estágio:
#   deblur: Inference_deblurNet.py não passa guidance_scale -> default 3.5
#   bokeh:  Inference_bokehNet.py passa guidance_scale=1.0
# A condição recebe 1.0 nos dois casos (flux.py:622, c_guidances).
deblur_train_guidance: float = 3.5
bokeh_train_guidance: float = 1.0
cond_train_guidance: float = 1.0
```

`create_backbone` passa o valor do estágio para o `FluxBackbone`; em
`forward_train_step`:

```python
guidance_main = torch.full((B,), self.train_guidance, device=device, dtype=dtype)
guidance_cond = torch.full((B,), self.cond_guidance, device=device, dtype=dtype)
```

Gravar o valor usado no `metadata` do checkpoint (ver C8).

**Como medir antes de retreinar (barato, sem GPU-semana)**

O modelo de 60k já existe. Antes de mudar o treino, rodar a avaliação em
`akcit-pixel/DDPD:test` com `guidance_scale` ∈ {1.0, 3.5} passado explicitamente
para `generate` (o resto idêntico). Interpretação:

- se **1,0 ganhar**: o treino a 1,0 está certo e é a *inferência* que tem que
  passar `guidance_scale=1.0` — nesse caso não mexa no treino, corrija os
  scripts de inferência e documente o desvio em relação ao oficial;
- se **3,5 ganhar**: confirma que o alvo é 3,5 e o treino é que estava errado.

Fazer esse A/B **depois do C1**, porque o C1 muda quais módulos sofrem o efeito.

---

## C3. Warmup: o primeiro update roda com LR 500× maior que o previsto

**Arquivo:** `genfocus_train/trainer.py:686-688` (nas duas árvores)

**Como está**

```python
if accelerator.sync_gradients:
    ...
    optimizer.step()            # ← usa o LR que estiver no param_group
    optimizer.zero_grad(set_to_none=True)

if accelerator.sync_gradients:
    state.global_step += 1
    scheduler.step(state.global_step)   # ← só AGORA o LR é ajustado
```

**Por que é problema**

O `param_group["lr"]` nasce com `config.optimizer.lr` (1e-4) e o scheduler só é
aplicado **depois** do primeiro `optimizer.step()`. Além disso o índice fica
deslocado: `state.global_step` já foi incrementado, e `_lr_at` internamente usa
`(step + 1) / warmup`. Executando o scheduler isolado, com `lr=1e-4`,
`warmup=500`:

```
LR que a própria fórmula prevê para o início (step=0): 2.000e-07

REAL:     update #1 -> 1.000e-04   #2 -> 4.000e-07   #3 -> 6.000e-07   #4 -> 8.000e-07
CORRETO:  update #1 -> 2.000e-07   #2 -> 4.000e-07   #3 -> 6.000e-07   #4 -> 8.000e-07
```

O primeiro update aplica LR 500× acima do início do warmup, sobre pesos LoRA
recém-inicializados. É exatamente o update em que o warmup existe para proteger.
Depois disso o warmup inteiro roda um step adiantado.

**Como tem que ficar**

UMA mudança. (Ver a correção logo abaixo — a versão anterior deste
plano pedia duas, e a segunda estava errada.)

1. Aplicar o LR **antes** do primeiro update. Logo depois do
   `maybe_resume_checkpoint` e antes de `iterator = _cycle(dataloader)`
   (`trainer.py:~618`):

```python
# O param_group nasce com o lr base; sem isto o PRIMEIRO update roda com
# 1e-4 em vez do primeiro passo do warmup. No resume, reposiciona o LR no
# ponto certo da curva.
scheduler.step(state.global_step)
```

> **Correção — este plano estava errado aqui.** A versão anterior mandava
> trocar `float(step + 1)` por `float(max(step, 1))` em `_lr_at`. **Não faça
> isso.** Verificado numericamente: com o `scheduler.step()` aplicado antes do
> laço, o update *k* usa `_lr_at(k-1)`, e a sequência-alvo 2e-7 → 4e-7 → 6e-7 →
> 8e-7 sai da fórmula **original**. Com `max(step,1)` o primeiro valor se
> repetiria (2e-7, 2e-7, 4e-7…) e a rampa terminaria em 9,98e-5 em vez de
> emendar no `base_lr`. A fórmula nunca esteve errada — o defeito era só a
> ordem. Há teste que fixa isso: `tests/test_warmup_scheduler.py`.
>
> Atenuante honesto: como `lora_B` nasce em zero, no primeiro update o
> gradiente em relação a `lora_A` é nulo e só `B` se move. O dano real é menor
> do que "500×" sugere — mas continua sendo um passo grande e não intencional.

**Verificar:** `logs`/wandb — o `deblur/lr` do step 10 tem que ser `2.0e-06`
(= 1e-4 · 10/500), e nunca deve aparecer `1.0e-04` antes do step 500.

---

## C4. Regime de escala espacial × cronograma de sigma (decisão medida)

**Arquivos:** `genfocus_train/data.py:144-200` (`prepare_aligned_pair`) e
`genfocus_train/backbone.py:301-325 + 364` (`sample_sigma`)

**Como está**

```python
scale = image_size / min(w, h)          # data.py:166 — lado MENOR vai para 512
new_w = max(image_size, round(w * scale))
new_h = max(image_size, round(h * scale))
...
sigma_u = self.sample_sigma(B, N, device, dtype)   # backbone.py:364 — N = 1024 (o crop)
```

**Por que é problema**

Dois eixos precisam bater entre treino e inferência, e hoje **nenhum regime
acerta os dois**:

| regime | dims por forward | seq | mu | exp(mu) | escala espacial vs treino |
|---|---|---|---|---|---|
| treino | 512×512 | 1024 | 0,6300 | 1,878 | 1,000× (referência) |
| eval `long_side=512`, 3:2 | 512×336 | 672 | 0,5704 | 1,769 | **0,672×** (blur 1,49× menor) |
| eval `long_side=0` (1024×688) | tile 512×512 | 1024 | **0,9225** | **2,516** | **1,344×** (blur 1,34× maior) |

- `long_side=512` erra a **escala espacial**: o treino reduz o lado *menor* e a
  avaliação reduz o lado *maior*.
- `long_side=0` acerta melhor a escala e o `seq` do forward (o tile é 32×32
  tokens = 1024, igual ao treino), mas erra o **cronograma de sigma**: em
  `flux.py:624`, `image_seq_len = latents.shape[1]` é da **imagem inteira**,
  calculado antes do tiling, então o `mu` vem de 2752 tokens e não de 1024.

Números conferidos com a config do FLUX-dev (`base_image_seq_len=256`,
`max_image_seq_len=4096`, `base_shift=0.5`, `max_shift=1.15`). Refazer a tabela
com a resolução real medida em C0-1.

**Como tem que ficar**

Existe uma opção que acerta os **dois** eixos, e é a que corresponde ao regime
que os autores assumem (`demo.py:298`: *"Disable tiling tricks: Not recommended
when the longer side is around 1000px or more"* — ou seja, imagem grande +
tiling de 512×512 px nativos).

### Opção A (recomendada) — crop nativo + mu da imagem inteira

1. Em `prepare_aligned_pair`, **não redimensionar**. Alinhar a 16, cropar 512×512
   direto no pixel nativo; só fazer upscale se algum lado for menor que 512:

```python
w, h = aif_pil.size
if min(w, h) < image_size:                      # única situação que exige resize
    scale = image_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    aif_r  = aif_pil.resize((new_w, new_h), Image.BICUBIC)
    blur_r = blur_pil.resize((new_w, new_h), Image.BICUBIC)
else:
    new_w, new_h = w, h                          # ESCALA NATIVA: o crop 512x512 é
    aif_r, blur_r = aif_pil, blur_pil            # o mesmo tile que a inferência vê
```

2. Passar para o `sample_sigma` o `seq_len` da **imagem de origem inteira**
   (alinhada a 16), não o do crop. O dataloader passa a devolver a chave
   `full_seq_len`:

```python
# data.py, dentro de prepare_aligned_pair / __getitem__
fw, fh = (new_w // 16) * 16, (new_h // 16) * 16
full_seq_len = (fw // 16) * (fh // 16)          # ex.: 1024x688 -> 64*43 = 2752
```

3. `sample_sigma` passa a aceitar `seq_len` **por amostra** (tensor `(B,)`) e a
   calcular um `mu` por amostra:

```python
def sample_sigma(self, seq_len: torch.Tensor, device, dtype) -> torch.Tensor:
    """seq_len: (B,) — tokens da IMAGEM DE ORIGEM inteira, não do crop.
    A inferência oficial calcula mu de latents.shape[1] da imagem completa
    (flux.py:624), ANTES do tiling; cada tile herda esse cronograma."""
    B = seq_len.shape[0]
    u = torch.sigmoid(torch.randn(B, device=device, dtype=torch.float32))
    cfg = self._scheduler_config
    m = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len)
    mu = seq_len.to(torch.float32) * m + (cfg.base_shift - cfg.base_image_seq_len * m)
    exp_mu = torch.exp(mu)
    sigma = (exp_mu * u) / (1.0 + (exp_mu - 1.0) * u)
    return sigma.to(dtype=dtype)
```

`calculate_shift` do diffusers é exatamente essa reta — reimplementada aqui só
para aceitar tensor. Manter um teste que compare os dois em alguns valores.

4. Avaliar sempre com `long_side=0` + tiling ligado (já é o
   `DEFAULT_LONG_SIDE = 0` do `vision-pipeline`), e **remover** `long_side=512`
   dos scripts de exemplo (`scripts/make_examples.py:80`).

### Opção B (alternativa) — tudo a 512², sem tiling

Treinar reduzindo o lado **maior** para 512 (imagem inteira, não crop) e avaliar
com `long_side=512` + `NO_TILED_DENOISE=True`. Acerta os dois eixos também, mas
abandona o regime nativo dos autores e perde detalhe fino.

### O experimento que decide (C0-4)

Não escolha por argumento. Três treinos de ~3000 steps, mesma seed, mesmos dados,
já com C1+C3 aplicados:

| run | dataloader | mu do sigma |
|---|---|---|
| R1 (atual) | lado menor → 512, crop 512² | do crop (1024) |
| R2 (Opção A) | crop nativo 512² | da imagem inteira |
| R3 (Opção B) | lado maior → 512, imagem inteira | do próprio 512² |

Avaliar os três em `akcit-pixel/DDPD:test` **nos três regimes de inferência**
(`long_side=0` tiled, `long_side=512` tiled, `long_side=512` no-tile) e montar a
matriz 3×3 de LPIPS/DISTS. Reportar a matriz inteira, não só a diagonal — o
objetivo é saber se algum run é robusto fora do próprio regime.

---

## C5. `top_k_sharpest` seleciona linhas, não cenas

**Arquivo:** `genfocus_train/data.py:401-463` (`_select_top_k_sharpest`)

**Como está**

```python
for i, row in enumerate(dataset.select_columns([column])):
    scores[i] = _laplacian_variance(row[column])     # column = "image_focus"
top_idx = np.argsort(scores)[::-1][:k]               # data.py:442 — ranking por LINHA
```

**Por que é problema**

`akcit-pixel/RealBokeh:train` tem 20.495 **linhas** para ~3.960 **cenas**: cada
linha é (uma entrada borrada de abertura X, o **mesmo** `image_focus` f/22 da
cena). Como o score é medido em `image_focus`, ele é **idêntico** dentro da cena
— os empates são exatos. Logo `argsort` puxa cenas inteiras, e o top-3000 vira
~580 GTs distintas repetidas ~5,2×, não 3.000 imagens distintas.

O paper é explícito em §B.1: *"we retain the top 3,000 sharpest **images** to
serve as additional supervision"*, e §4.1 fecha em *"3.5K pairs"* (344 DPDD +
3000 RealBokeh). A intenção é 3.000 alvos distintos.

Não é um desastre — mesmo alvo com blurs diferentes é boa aumentação — mas é
**5× menos diversidade de conteúdo** do que o número sugere, e não é o que o
paper descreve.

**Como tem que ficar**

Ranquear por **cena** e amostrar a linha dentro da cena. Isso preserva o número
de pares do paper (3000/época) *e* recupera a diversidade *e* mantém a
aumentação por abertura ao longo do treino.

1. Identidade da cena — **medir primeiro** qual chave existe (C0-2). Preferir,
   nesta ordem: (a) coluna explícita de cena, se houver; (b) prefixo de
   `file_name_base` antes do sufixo de abertura; (c) hash dos bytes de
   `image_focus`. Implementar como função isolada e testável:

```python
def _scene_key(record: dict) -> str:
    """Identidade da CENA. O ranking de nitidez tem que ser por cena, não por
    linha: no RealBokeh o mesmo image_focus f/22 aparece em ~5 linhas (uma por
    abertura), com score de Laplaciano idêntico, e o top-k puxaria cenas
    inteiras (~580 GTs para 3000 linhas) em vez de 3000 imagens distintas."""
```

2. Novo seletor:

```python
def _select_top_k_sharpest_by_scene(dataset, k, column, cache_id=None):
    # 1) agrupa índices por cena
    # 2) mede _laplacian_variance UMA vez por cena (na 1a linha da cena)
    # 3) ranqueia CENAS, pega as top-k
    # 4) devolve a lista de índices de TODAS as linhas dessas k cenas
    #    + a lista de grupos, para o __getitem__ sortear a linha
```

3. `HuggingFaceDeblurDataset` passa a ter `__len__ == k` (uma entrada por cena) e
   `__getitem__` sorteia uma linha do grupo:

```python
def __getitem__(self, index):
    grupo = self.scene_groups[index]           # linhas daquela cena
    row_i = grupo[0] if not self.runtime.train else int(self._rng.integers(len(grupo)))
    record = self.dataset[row_i]
    ...
```

Assim a época tem 3000 pares (= paper), com 3000 GTs distintas, e ao longo dos
60K steps o modelo vê todas as aberturas de cada cena.

**Versão mínima**, se a acima for muito invasiva: deduplicar para **uma linha por
cena** (a de maior blur) *antes* de medir, e então ranquear e pegar top-3000.
Perde a aumentação por abertura, mas corrige a diversidade.

4. **Invalidar o cache — crítico.** `_filter_cache_path` (`data.py:388`) monta a
chave com `f"{safe}_{column}_top{k}_n{n}"`. Nenhum desses campos muda com a nova
lógica, então o próximo run leria os **índices antigos** e a correção não teria
efeito nenhum, em silêncio. Incluir a versão do seletor na chave:

```python
_FILTER_SELECTOR_VERSION = "v2-by-scene"   # BUMP a cada mudança na lógica de seleção
...
return os.path.join(cache_dir, f"{safe}_{column}_top{k}_n{n}_{_FILTER_SELECTOR_VERSION}.json")
```

**Verificar:** rodar C0-2 de novo depois da mudança — `GTs distintas` tem que
ser 3000 (ou igual ao `k` escolhido).

---

## C6. Inferência do modelo main+cond liga LoRA no branch de TEXTO

**Arquivos:** `scripts/infer_mainlora.py`,
`vision-pipeline/inference/src/pipelines/deblur_net.py`

**Como está**

Passa-se `main_adapter="deblurring"` para compensar o treino main+cond.

**Por que é problema**

`generate` monta `adapters = [main_adapter] * 2 + c_adapters`, e em
`transformer_forward` o **índice 0 é o branch de TEXTO** (`txt_n = 1`). No
`single_block_forward` os hidden states são `[*text, *image]` e o texto passa por:

```python
with specify_lora((self.norm.linear, self.proj_mlp), adapters[i]):   # i = 0 -> texto
with specify_lora((self.proj_out,), adapters[i]):
```

Ou seja, `main_adapter="deblurring"` liga LoRA no texto em 38 blocos single
(`norm.linear`, `proj_mlp`, `to_q/k/v`, `proj_out`). Mas o treino original
(`genrefocus_deblurnet/genfocus_train/backbone.py:407`) usa
`adapters = [None, ADAPTER_NAME] + [ADAPTER_NAME] * n_cond` — **texto sem
adapter**. A inferência não reproduz o treino.

Nos blocos duplos não há efeito (os módulos do texto — `norm1_context.linear`,
`add_q_proj`, `to_add_out`, `ff_context.net.2` — não são alvos de LoRA e nem
passam por `specify_lora`). O efeito é só nos 38 single blocks.

**Como tem que ficar**

Não tocar em `third_party/Genfocus/` (é o clone de referência, tem que ficar
intacto para comparação). Adicionar o parâmetro na **nossa cópia**,
`vision-pipeline/inference/Genfocus/pipeline/flux.py`, marcando claramente o
desvio:

```python
def generate(
    ...,
    main_adapter: Optional[List[str]] = None,
    text_adapter: Optional[str] = "__same_as_main__",   # DESVIO NOSSO, ver nota
    ...
):
    # NOTA (desvio deliberado do upstream): o generate oficial monta
    # `adapters = [main_adapter]*2 + c_adapters`, e o índice 0 é o branch de
    # TEXTO. Nosso DeblurNet main+cond foi treinado com [texto=None,
    # main=LoRA, cond=LoRA]; passar main_adapter="deblurring" ligaria LoRA no
    # texto em 38 single blocks, coisa que o treino nunca fez. text_adapter
    # separa os dois. Default preserva o comportamento upstream.
    _text = main_adapter if text_adapter == "__same_as_main__" else text_adapter
```

e substituir os `[main_adapter] * 2` pelas duas ocorrências por
`[_text, main_adapter]` (há **três** chamadas a `transformer_forward` no
`generate`: ramo tiled, ramo normal e ramo `image_guidance_scale != 1.0` —
alterar as três).

Nos scripts de inferência do modelo main+cond, passar
`main_adapter="deblurring", text_adapter=None`.

**Verificar:** avaliar o mesmo peso de 60k nas duas configurações
(`text_adapter=None` vs `text_adapter="deblurring"`) em `DDPD:test` e reportar a
diferença. Se for desprezível, documentar e seguir; se não, a versão com
`text_adapter=None` é a correta e as métricas anteriores precisam ser refeitas.

---

## C7. Todos os ranks sorteiam o mesmo ruído e o mesmo sigma

**Arquivo:** `genfocus_train/trainer.py:549`

**Como está**

```python
_set_seed(config.runtime.seed)      # idêntico em todos os processos
```

**Por que é problema**

Com 4 GPUs × `gradient_accumulation_steps=8`, o batch efetivo é 32, mas os 4
ranks sorteiam **os mesmos 8 sigmas e os mesmos 8 tensores de ruído**. O batch
efetivo tem só 8 σ distintos em vez de 32 — aumenta a variância do gradiente sem
nenhum ganho.

Os pesos LoRA continuam idênticos entre ranks porque há `broadcast` explícito do
rank 0 (`trainer.py:~608`), então mudar a seed por rank é seguro.

**Como tem que ficar**

Mover o `_set_seed` para depois da criação do `accelerator` e diferenciar:

```python
accelerator = _build_accelerator(config)
# Seed POR RANK: os pesos LoRA já são sincronizados por broadcast explícito
# (abaixo), então ranks com RNG diferente é o que queremos — senão as 4 GPUs
# sorteiam o mesmo sigma/ruído e o batch efetivo 32 tem só 8 σ distintos.
_set_seed(config.runtime.seed + accelerator.process_index)
```

Manter o smoke test (`trainer.py:893`) com a seed pura (é single-process e
precisa ser reprodutível).

---

## C8. Checkpoint não registra a variante nem os hiperparâmetros que mudam o modelo

**Arquivo:** `genfocus_train/trainer.py:260`

**Como está**

```python
"metadata": {
    "stage": stage, "global_step": ..., "config_hash": ...,
    "metrics_snapshot": ..., "git_commit": ...,
}
```

**Por que é problema**

Existem duas variantes de DeblurNet no projeto (cond-only e main+cond) e elas
exigem chamadas de inferência **diferentes**. O repo HF
`juliadollis/genrefocus-deblurnet-paper-4gpu` é main+cond apesar do nome, e foi
exatamente esse tipo de confusão que produziu o bug da saída lavada (LPIPS
~0,85). Nada no artefato diz qual é qual.

**Como tem que ficar**

```python
"metadata": {
    ...,
    # Tudo que MUDA o modelo e precisa ser reproduzido na inferência:
    "adapter_variant": "cond-only" if not config.model.lora_on_main else "main+cond",
    "text_adapter": None,                       # o treino nunca põe LoRA no texto
    "train_guidance_main": float(train_guidance),
    "train_guidance_cond": float(cond_guidance),
    "lora_rank": int(backbone.lora_rank),
    "n_lora_modules": n_lora_modules,           # 343 pós-C1
    "image_size": int(stage_cfg.image_size),
    "dataloader_scale_mode": stage_cfg.scale_mode,   # "short_side" | "native" | "long_side"
    "sigma_mu_source": stage_cfg.sigma_mu_source,    # "crop" | "full_image"
    "prompt": STAGE_PROMPTS[stage],
}
```

Aproveitar e escrever um `<stage>.json` ao lado do `.safetensors` exportado com
o mesmo dicionário, para que o artefato publicado no HF carregue essa informação.

Adicionar também `lora_on_main: bool = False` em `ModelConfig` e usar em
`forward_train_step` (`adapters = [None, ADAPTER_NAME if lora_on_main else None]
+ [ADAPTER_NAME] * n_cond`), para que as duas variantes fiquem a um campo de
YAML de distância em vez de duas árvores de código divergentes.

---

## C9. `concatenate_datasets` sem alinhar colunas

**Arquivo:** `genfocus_train/data.py:479-484`

**Como está**

```python
loaded = [_load_hf_split(source) for source in sources]
return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)
```

**Por que é problema**

`concatenate_datasets` exige `features` idênticas. Qualquer coluna a mais/a menos
entre `akcit-pixel/DDPD` e `akcit-pixel/RealBokeh` levanta exceção — e isso
acontece **depois** do filtro de nitidez, que leva ~20 min. Além disso, carregar
`image_pre_deblur` (não usado) desperdiça I/O e memória.

**Como tem que ficar**

```python
_DEBLUR_KEEP = ["image_blur", "image_focus", "file_name_base"]

def _load_hf_sources(sources):
    if not sources:
        raise ValueError("Pelo menos um source HF precisa estar configurado.")
    loaded = []
    for source in sources:
        ds = _load_hf_split(source)
        faltando = [c for c in _DEBLUR_KEEP if c not in ds.column_names]
        if faltando:
            raise ValueError(f"{source.name}:{source.split} não tem {faltando}")
        # Alinha o schema ANTES do concat: o concatenate_datasets exige features
        # idênticas e falharia só depois dos ~20 min do filtro de nitidez.
        # image_pre_deblur é descartado de propósito (variante §3.4, não usada).
        loaded.append(ds.select_columns(_DEBLUR_KEEP))
    return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)
```

Fazer o `select_columns` **antes** do `_select_top_k_sharpest` dentro de
`_load_hf_split`, para o filtro também ficar mais leve.

---

## C10. Nenhuma validação durante o treino

**Arquivo:** `genfocus_train/trainer.py` (loop principal)

**Como está**

`state.best_loss` é a loss de treino de **um único micro-batch**, com σ e ruído
sorteados — não é sinal de nada. Não há avaliação em split de validação em 60K
steps, e `keep_last_n_checkpoints=5` descarta checkpoints antigos por ordem, não
por qualidade.

**Como tem que ficar**

A cada `eval_every_steps` (novo campo em `RuntimeConfig`, sugestão 1000):

1. Rodar o **mesmo forward de treino** sobre N amostras fixas de
   `akcit-pixel/DDPD:validation` (73 imagens), com **σ e ruído fixos por seed**
   (não sorteados) — assim a métrica é comparável entre steps:

```python
# σ determinístico: uma grade fixa cobrindo o range, mesma para todo step.
sigmas_val = torch.linspace(0.1, 0.9, 9)
gen = torch.Generator(device=device).manual_seed(1234)
```

2. Logar `deblur/val_loss` e usar **essa** métrica para `best_loss` e para
   decidir qual checkpoint preservar (guardar sempre o melhor, além dos N
   últimos).

Não tentar rodar as 28 etapas de denoise + LPIPS dentro do treino — caro demais
e não é necessário para seleção de checkpoint.

---

## C11. Dfs de treino

### C11-1. Resolução armazenada (bloqueante para a Tabela 2)

Depende de C0-1. Se `akcit-pixel/DDPD` estiver gravado em ~1024×688 em vez do
original 1680×1120 do DPDD:

- o df **já** reduziu o blur em 1,64× em relação aos dados do paper;
- as nossas métricas de Tabela 2 são calculadas sobre imagens reduzidas e
  **não são diretamente comparáveis** aos números publicados;
- a decisão do C4 muda de escala absoluta.

Ação: registrar a resolução medida em `PANORAMA.md` com essa ressalva explícita
e, se possível, regerar os dfs na resolução original.

### C11-2. Escala inconsistente entre fontes

Com "lado menor → 512", um DDPD 1024×688 encolhe 0,744× e um RealBokeh_3MP
(~2048×1536) encolhe 0,333×. As duas fontes entregam blur em regimes espaciais
completamente diferentes **dentro do mesmo treino**. A Opção A do C4 (crop
nativo) elimina isso automaticamente. Se optarem pela Opção B, é obrigatório
normalizar as fontes para a mesma escala física antes.

### C11-3. DPDD com 344 linhas, não 350

O split oficial do DPDD é 350/74/76 (500 cenas). O repo tem 344/73/75 = 492.
Faltam 8 cenas. O paper diz *"all images from the official training split"*.
Verificar se a perda foi intencional (imagens corrompidas?) e documentar. Baixo
impacto, mas é um desvio silencioso do "all images".

### C11-4. Balanceamento DPDD vs RealBokeh

344 + 3000 com shuffle uniforme → DPDD é 10,3% das amostras, e DPDD é a mesa da
Tabela 2. O paper não especifica peso de amostragem. Não mudar às cegas —
adicionar `sampling_weight` opcional em `DatasetSourceConfig` e medir com um run
curto (peso 1:1 vs proporcional) junto do experimento do C4.

### C11-5. Dfs corretamente fora do treino da DeblurNet

Confirmado que `akcit-pixel/LFDOF` e `akcit-pixel/RealDOF` **não** entram no
treino do Stage 1: LFDOF é fonte do BokehNet (§3.2c) e RealDOF é benchmark
(Tabela 2). Não incluir.

---

## Itens conferidos e que estão CORRETOS (não mexer)

Reauditado contra `Inference_deblurNet.py`, `demo.py` e `Genfocus/pipeline/flux.py`:

- Ordem dos branches `[texto, main, *cond]` e `assert len(adapters) == len(timesteps)`.
- Timesteps por branch `[σ, σ, 0]` — condição é "limpa" (`flux.py:621`).
- `guidance` da **condição** = 1.0 (`flux.py:622`) — bate com o treino.
- IDs de posição: `position_delta=[0,0]`, `position_scale=1.0`; treino usa
  `_prepare_latent_image_ids(B, H//2, W//2)`, idêntico ao fallback do
  `encode_images`.
- `group_mask`: com 1 condição, `ones` e `ones + diag(1)` são o mesmo tensor.
- Prompt `"a sharp photo with everything in focus"` (`Inference_deblurNet.py`,
  `demo.py:172`).
- Alvo do flow: `x_t = (1-σ)x₀ + σε`, `v* = ε - x₀` — bate com o
  `FlowMatchEulerDiscreteScheduler.step`.
- Normalização `[-1,1]` via `/127.5 - 1` == `image_processor.preprocess`.
- `alpha = r` → `scaling = 1`, que é o valor que `specify_lora` força na
  inferência.
- Export `.safetensors` em fp32: o `load_state_dict` do PEFT converte para bf16
  no load; não é problema.
- `AcceleratedOptimizer.step()/zero_grad()` são guardados por `sync_gradients`,
  então chamá-los fora do `if` não duplica updates.
- 28 etapas de denoise, `TILE_SIZE=32` (= 512 px), overlap 4 (= 64 px), blend
  gaussiano — nada disso é decidido no treino.
- `image_pre_deblur` não usado: correto, é a variante §3.4, que na inferência usa
  `position_delta=[0, 32]` na condição extra
  (`Inference_deblurNet_with_pre_deblur.py:114`) — anotar para quando for
  implementada.

## Desvios conhecidos e aceitos (documentar, não "corrigir")

- **rank 128 vs 64**: o texto do paper (§4.1) diz `r=128`; o
  `deblurNet.safetensors` oficial tem rank 64 em todos os tensores. Divergência
  dos próprios autores. Mantemos 128 (texto do paper).
- **steps**: paper 60K; `train_base.yaml` usa 8000 e `train_ddpd*.yaml` 1800-2000,
  por limite de cluster. É subtreino, não erro.
- **AdamW lr=1e-4, wd=1e-4, cosine warmup 500, bf16**: não especificados pelo
  paper. Escolha documentada.
