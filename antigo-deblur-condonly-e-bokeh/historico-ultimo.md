# GenRefocus / BokehNet — contexto completo e estado atual

> **Para quem lê isto primeiro (ex.: uma sessão nova do Claude Code):** este arquivo é
> auto-suficiente. Ele descreve o projeto, o que já foi feito, o que está rodando agora,
> e principalmente as **regras de GPU e de ambiente** que não podem ser violadas.
> Histórico mais antigo e detalhado: `../HANDOFF_PROJECT_HISTORY.md` e
> `../HANDOFF_CODEBASE_GUIDE.md` (fora deste repo git, na pasta `repositorio_ref/`).
>
> Última atualização: 2026-08-06.

---

## 1. O que é o projeto

Reprodução do paper **GenRefocus** (arXiv 2512.16923) para depois avançar e publicar algo
em cima. O pipeline do paper tem dois estágios, ambos LoRA sobre o **FLUX.1-dev**:

| Estágio | Modelo | Entrada → Saída | Condições | LoRA rank |
|---|---|---|---|---|
| 1 | **DeblurNet** | foto borrada → all-in-focus (AIF) | 1 (a foto borrada) | 128 |
| 2 | **BokehNet** | AIF + mapa de defocus → foto com bokeh | 2 (AIF, defocus map) | 64 |

- **DeblurNet: JÁ REPRODUZIDO.** Treinado e avaliado (LPIPS 0.1446, melhor que o publicado).
- **BokehNet: EM TREINAMENTO AGORA** (é o assunto deste documento).

O condicionamento não é ControlNet: é **concatenação de tokens** no DiT. O código dos autores
(`transformer_forward` em `Genfocus/pipeline/flux.py`) é usado tal e qual, sem reimplementação.

### Currículo do BokehNet (paper §4.1, citação literal)
> "BokehNet is trained in two stages: (i) 40K steps on synthetic data, and (ii) 60K steps on real data."

- **Fase 1**: 40.000 steps em dados sintéticos (rota a).
- **Fase 2**: 60.000 steps em dados reais (rotas b + c), **iniciando do LoRA da fase 1**,
  com optimizer/scheduler **resetados** (novo warmup + cosseno).

---

## 2. Onde fica cada coisa

| O quê | Onde |
|---|---|
| Código (local, Mac) | `~/Projects_Code/repositorio_ref/genrefocus_deblurnet_paper` |
| Código (cluster) | `/raid/user_juliadollis/projects/genrefocus_deblurnet_paper` |
| Repo git (privado) | https://github.com/juliadollis/genrefocus-training |
| Dados da rota a (local no cluster) | `/raid/user_juliadollis/data-bokeh/rota_a` (= `/workspace/data-bokeh/rota_a` dentro do container) |
| Imagem Singularity | `/raid/user_juliadollis/images/transformers-pytorch-gpu.sif` |
| Repo oficial dos autores | github.com/rayray9999/Genfocus (clonado em `third_party/Genfocus`) |
| Pesos oficiais | HF `nycu-cplab/Genfocus-Model` |
| Pipeline de dados | github.com/AKCIT-PIXEL/bokehnet-data-pipeline |
| wandb | projeto `genrefocus-bokehnet` |

O cluster é acessado por `ssh dgx-H100-02`. **Não há git no cluster**: o código vai por
`rsync` (`scripts/sync_to_cluster.sh --go`, rodado **no Mac**).

---

## 3. ⚠️ REGRAS DE GPU (as mais importantes)

### 3.1 Nunca rode nada pesado fora do SLURM
O "nó de login" **é o próprio `dgx-H100-02`**, a mesma máquina onde os jobs rodam. Qualquer
processo iniciado no shell interativo consome CPU/RAM que o SLURM já prometeu a outras
pessoas. **Outros membros do time usam as mesmas GPUs.**

- Treino e inferência: **sempre** via `sbatch`.
- Preparo de dados (download, resize, conversão): via `sbatch` **sem pedir GPU**
  (ver `scripts/prefetch_bokeh.slurm`, que não tem `--gres`). Jobs sem GPU entram rápido
  na fila porque não disputam o recurso escasso.
- Já aconteceu de rodar prefetch com `nohup` no login node. Funciona, mas foi corrigido
  para SLURM justamente por educação de cluster.

### 3.2 O SLURM às vezes aloca GPU ocupada
Neste cluster o escalonador ocasionalmente entrega uma GPU que já está em uso. Por isso os
scripts de **smoke e inferência** escolhem a GPU realmente mais livre **dentro do container**:

```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t"," -k2 -nr | head -1 | cut -d"," -f1 | tr -d " ")
```

Não remova esse trecho. Os scripts de treino multi-GPU não fazem isso (usam o que o SLURM deu).

### 3.3 Multi-GPU: NÃO existe DDP wrap
Isto é sutil e quebra de formas confusas se alguém "consertar":

- `transformer_forward` (código dos autores) acessa submódulos direto (`self.x_embedder`,
  `self.transformer_blocks`, ...). O wrapper `DistributedDataParallel` **não expõe** esses
  atributos → `AttributeError`.
- Além disso, esse forward **não passa pelo `.forward()`** do módulo, então o DDP nem
  sincronizaria os gradientes.

**Solução implementada em `genfocus_train/trainer.py`:**
1. Multi-GPU mantém o transformer **cru** (só o optimizer passa por `accelerator.prepare`).
2. No início, `torch.distributed.broadcast` dos pesos treináveis do rank 0 → todos
   (o LoRA é init gaussiano aleatório; sem isso cada GPU começaria diferente).
3. No step de sync, `all_reduce(grad, op=AVG)` manual em cada parâmetro treinável.

Em single-GPU o caminho é o normal (`accelerator.prepare` no transformer).

### 3.4 Batch efetivo é SEMPRE 32 (paper §4.1)
`batch_size` por GPU = 1. O ajuste é no `gradient_accumulation_steps`:

| GPUs | grad_accum | batch efetivo |
|---|---|---|
| 2 | 16 | 32 |
| 4 | 8 | 32 |

Se mudar o número de GPUs, **tem que** ajustar o accum. Checkpoints são portáveis entre
2 e 4 GPUs justamente porque o batch efetivo é idêntico.

### 3.5 cuDNN SDPA precisa ficar DESLIGADO
Em `backbone.py`, dentro de `load()`:

```python
torch.backends.cuda.enable_cudnn_sdp(False)
```

Motivo: o `attn_forward` do Genfocus chama `scaled_dot_product_attention` com query e
key/value de **seq-lens diferentes** (concatena texto + main + condições), e o FLUX usa
`head_dim=128`. Esse formato faz o backend cuDNN escolher um kernel que **quebra no backward
em H100/bf16** com `RuntimeError: Expected mha_graph->execute(...).is_good()`. Flash e
mem-efficient lidam bem. Não reative.

### 3.6 VRAM
Com `gradient_checkpointing: false` (configs de treino), o pico é **~66 GB de 80 GB** por GPU
a 512². Não aumente `batch_size` nem resolução sem religar o checkpointing. O smoke usa
`gradient_checkpointing: true` e 256² (mais leve).

### 3.7 Antes de submeter, olhe o estado das GPUs
```bash
ssh dgx-H100-02 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'
```

---

## 4. ⚠️ REGRAS DE AMBIENTE (custaram 4 jobs falhos)

### 4.1 O trio ML limpo vive no `~/.local` do cluster
```
transformers 5.0.0.dev0   diffusers 0.37.1   peft 0.18.1   (+ datasets 4.3.0 no container)
```
`wandb` e `python-dotenv` também foram instalados no `~/.local`.

### 4.2 NUNCA use `PEFT_PIN='peft==0.13.2'`
Esse pin aparece recomendado em handoffs antigos e em comentários velhos. **É veneno:**
o diffusers 0.37.1 exige `peft>=0.17`, então o job morre com
`ImportError: peft>=0.17.0 is required`. O pin ainda rebaixava o transformers para 4.37.2.
**Rode sem `PEFT_PIN`.** A variável continua existindo nos scripts, mas deve ficar vazia.

### 4.3 NUNCA sobrescreva `PYTHONUSERBASE` nem passe `--env PYTHONPATH=<site-packages>`
Qualquer um dos dois tira o `~/.local` (ou o site-packages interno do container) do
`sys.path`, e o resultado é `ModuleNotFoundError: No module named 'peft'` (ou `diffusers`).

O contrato correto, já implementado nos scripts:
- pacotes de ML: vêm do `~/.local` e do container, sozinhos;
- `PYTHONPATH`: montado **dentro do `APP_CMD`**, contendo **apenas** o código do projeto
  (`.../genrefocus_deblurnet_paper` e `.../third_party/Genfocus`);
- nada de `--env PYTHONPATH` na linha do `singularity exec`.

### 4.4 Segredos
`HF_TOKEN` e `WANDB_API_KEY` ficam num `.env` na raiz do projeto (git-ignored, `chmod 600`).
Todos os scripts carregam automaticamente, nesta ordem: variável já exportada →
`$ENV_FILE` → `$SLURM_SUBMIT_DIR/.env` → `$PROJECT_ROOT/.env` → `$PWD/.env`.
O `.env` **vai junto no rsync** (não está nos excludes) — é o canal intencional.
**Nunca hardcode token em script.** Já houve um episódio em que ~21 scripts tinham os
tokens em texto puro; foi sanitizado antes do primeiro push.

### 4.5 O SLURM congela o script no `sbatch`
Editar/sincronizar depois **não** altera um job já enfileirado. Sempre:
`sync --go` no Mac → conferir no cluster → `sbatch`.
Logs de jobs antigos (28779, 28871, 28877, 29128, 29186) são **pré-correção**: ignore.

### 4.6 Warnings benignos (podem ser ignorados)
```
Skipping import of cpp extensions due to incompatible torch version...
Unable to import `torchao` Tensor objects...
```

### 4.7 Cota de disco
500 GB (soft) / 600 GB (hard). Quando a graça expira, a soft vira bloqueio imediato.
Estado atual: **371 GB**. O que mais cresce: `hf-cache/` e checkpoints.
Um download normal da rota a (341 GB) **estoura a cota** — ver seção 6.

---

## 5. 🐛 O BUG DOS DADOS (o achado mais importante desta reprodução)

### O que estava errado
A coluna `defocus_map` dos datasets `AKCITPixel3/*` está **normalizada por imagem**:

```
valor_gravado = |D − s1| / max|D − s1|        ← o k NÃO entra
```

quando deveria ser (fórmula da inferência oficial, `Inference_bokehNet.py:138-140`):

```
valor_correto = clip(k · |D − D_foco| / MAX_COC, 0, 1)     MAX_COC = 100
```

**Evidência medida** (rotas a e b, várias amostras): `max(defocus_map)/65535 == 1.00000` em
todas, com `k` variando de 33 a 195. O erro contra `|D−s1|/max|D−s1|` é 2e-5 (a quantização
do uint16); contra a fórmula correta, 0.09 a 0.78.

**Origem**: `bokehnet_common.py:87` do `bokehnet-data-pipeline` grava
`dm/dm.max()*65535` dentro do bloco `save_visualizations`. Era a imagem de *visualização*,
e virou o dado de treino.

**Por que quebraria tudo**: o brilho do mapa **é** a intensidade do bokeh. Com todo mapa
normalizado ao mesmo máximo, o modelo recebe entradas idênticas com alvos de blur diferente,
aprende o blur médio e **ignora o K** — justamente o ponto central do paper (bokeh controlável).
E a inferência oficial geraria mapas fora da distribuição de treino.

### ⚠️ Armadilha metodológica
Uma auditoria anterior validou o mapa por **correlação** com `|k(D−s1)|` e deu ~1.0, passando
o bug. **Correlação é invariante a escala.** O teste que discrimina é comparar o **`max`
absoluto** entre amostras de `k` diferente.

### A correção (aplicada)
O dataloader **ignora** a coluna quebrada e recompõe o mapa das colunas `depth`, `k` e `s1`
(que existem e estão intactas nos datasets):

```python
defocus = clip(k * |depth − s1| / max_coc, 0, 1)      # max_coc = 100
```

Controlado por `defocus_source` no YAML: `recompute` (padrão, correto) ou `column` (legado,
só para ablação). **Nenhum dado precisou ser regerado.** O `scripts/infer_bokeh_test.py`
também foi migrado, senão o teste rodaria numa representação diferente da treinada.

Nota de calibração: o `k` das rotas vai de ~10 a 300 e o `depth` é profundidade normalizada
em [0,1], então `k/100` cobre bem a faixa [0,1]. Isso bate com o que uma pessoa do time de
dados havia achado empiricamente ("15 era baixo, coloquei 100").

### ⏳ Pendência que continua com o time de dados (não bloqueia)
Os **alvos** sintéticos da rota a foram renderizados pelo `_render_bokeh_simple`, porque o
`_render_bokehme` levanta `ImportError` incondicional (BokehMe nunca foi instalado). Esse
fallback limita o kernel a 51 px, então amostras com `k` alto **saturam** o blur real: medi
que a nitidez do alvo não acompanha o sigma esperado. O próprio arquivo diz que esse
renderizador "NÃO substitui o BokehMe para treinamento real".

Consequência prática: a **fase 1** ensina bem só a faixa de K abaixo do teto. A **fase 2**
não é afetada (alvos são fotos reais). Corrigir exige regerar a rota a com o BokehMe de verdade.

---

## 6. Dados: por que existe um "prefetch"

A rota a tem **341,6 GB** no Hugging Face (68.000 amostras). O caminho normal do `datasets`
ainda duplica isso ao converter → **não cabe** nos 500 GB de cota. Um job já morreu com
`Disk quota exceeded` depois de 16 minutos baixando.

**Solução**: o treino roda a 512², e o dataloader reduz o lado menor para 512 **antes** do
crop. Então `scripts/prefetch_bokeh_local.py` faz um passe em **streaming** (sem cache),
aplica **o mesmo resize** e salva PNGs pequenos + `metadata.jsonl` (com `k`, `s1`, `stem`).

- 341,6 GB → **~88 GB** (1,29 MB por amostra, medido).
- O tensor que chega no treino é **idêntico** ao que viria do dataset original
  (mesma fórmula de resize; há teste garantindo que as duas não divirjam).
- **Resumível**: conta as linhas do `metadata.jsonl` e dá `skip`.

O dataloader roteia por caminho: **path absoluto = pasta local**, nome com `/` = repo HF.
Fase 1 usa a pasta local; fase 2 usa o hub direto (rotas b 42 GB + c, cabem).

**Dedup**: o `metadata.jsonl` é append-only. Rodar o prefetch duas vezes na mesma pasta
repete linhas (aconteceu: o processo do login node e o job do SLURM rodaram em paralelo →
136.575 linhas para 68.000 índices únicos). As imagens ficam íntegras (mesmo índice = mesmo
nome de arquivo, reescrito com conteúdo idêntico). O loader agora **deduplica por `idx`** e
avisa no log.

---

## 7. Auditoria: treino × paper × inferência oficial

Conferido linha a linha contra `Genfocus/pipeline/flux.py` e `Inference_bokehNet.py`.
**Tudo bate.** Os invariantes que importam:

| Invariante | Valor no treino | Confere com |
|---|---|---|
| Adapters por branch | `[None, None, "bokeh", "bokeh"]` | `[main_adapter]*2 + c_adapters`, `main_adapter=None` |
| LoRA rank / alpha | 64 / alpha=rank (scaling 1) | `specify_lora` força scaling 1; peso oficial é rank 64 |
| Módulos LoRA | `LORA_TARGET_MODULES` (10 categorias) | os 8 pontos de `specify_lora` = 343 módulos do `bokehNet.safetensors` |
| Timesteps | `[t, t, 0, 0]` | condições são "limpas" (t=0) |
| Guidance | todas 1.0 | inferência de bokeh usa `guidance_scale=1.0` |
| `group_mask` | ones + diag no bloco cond↔cond | idem (provado empiricamente com o `attn_forward` oficial) |
| Ordem das condições | `[AIF, defocus]` | idem |
| Ranges | AIF `[-1,1]`, defocus `[0,1]` | `No_preprocess=True` só na 2ª condição |
| Prompt | "an excellent photo with a large aperture" | idem |
| Sigma | logit-normal + `calculate_shift(seq_len do main)` | mesma fórmula da inferência |
| Alvo do flow | `x_t=(1−σ)x₀+σε`, `v*=ε−x₀` | FlowMatchEuler do FLUX |

**Bugs históricos já corrigidos** (não regridam):
1. `group_mask` sem a diagonal → as duas condições se cross-atendiam no treino, coisa que a
   inferência oficial nunca faz. (Não afetava o DeblurNet, que só tem 1 condição.)
2. Fase real com 40K steps em vez de **60K**.
3. Rota b classificada como sintética; ela é **real com EXIF** (análoga ao ITW do paper).
4. `main_adapter`: o LoRA age **só nas condições** (cond-only), igual ao artefato oficial.

**Hiperparâmetros não especificados pelo paper** (escolha documentada, não desvio):
AdamW `lr=1e-4`, `weight_decay=1e-4`, cosseno com 500 steps de warmup, bf16, 512².

**Divergência conhecida entre paper e peso oficial** (não é erro nosso): o paper diz
DeblurNet rank 128, mas o `deblurNet.safetensors` oficial é rank 64. Nosso DeblurNet seguiu
o paper (128) e ficou melhor que o publicado.

---

## 7.5 💀 A MORTE DO JOB 29267 (e por que o loader agora pula arquivo ruim)

**Resumo:** um único PNG truncado matou 13h54m de treino em 2 H100.

O job `29267` rodou 3.380 steps sem problema nenhum (loss saudável, VRAM estável,
o upload ao HF do step 2500 subiu inteiro) e morreu às 2026-08-07 01:26:40 com:

```
OSError: image file is truncated
  data.py _to_pil_rgb -> image_like.convert("RGB")   # a imagem AIF
```

O erro estoura num worker do DataLoader, sobe para o rank 1, e o `accelerate`
mata o job inteiro (`ChildFailedError`, exit 1). Não teve nada a ver com upload,
NCCL, cota (371G/500G) nem VRAM.

**Origem: a corrida do prefetch.** O processo do login node e o job SLURM `29258`
escreveram na MESMA pasta em paralelo (é o mesmo evento que deixou 136.575 linhas
no `metadata.jsonl` para 68.000 idx únicos). O histórico registrou isso como
"resolvida, sem impacto", com a justificativa de que "mesmo idx = mesmo nome de
arquivo, reescrito com conteúdo idêntico". **Essa justificativa estava errada:**
dois escritores no mesmo arquivo ao mesmo tempo deixam o conteúdo cortado. A
dedup por `idx` (commit `e9198b1`) consertou a CONTAGEM de amostras, não os bytes
no disco.

**Extensão do estrago (medida, não estimada).** `scripts/verify_prefetch_integrity.py`
decodificou os 204.000 PNGs da rota a um a um: **exatamente 1 arquivo ruim**,
`aif/0000858_110390d4-559e-444a-b395-0f63e1d46f89_s018.png` (196.673 bytes, sem o
chunk `IEND`). Índice 858 dos 68.000. Decidido NÃO re-baixar: recuperar 1 imagem
exigiria puxar um shard parquet de vários GB, e 0,0015% do dataset não muda nada.
O loader pula essa amostra e avisa no log.

**As duas correções:**
1. `data.py::LocalBokehFolderDataset.__getitem__` captura erro de imagem
   corrompida (`OSError`/`SyntaxError`/`struct.error`), avisa com o nome do
   arquivo e sonda a amostra seguinte, com limite de 8 tentativas. O limite
   importa: falha sistemática (pasta sumiu, disco fora) continua quebrando o
   treino em vez de virar loop silencioso. `ValueError` ficou de fora de
   propósito — é o que `_validate_image_tensor` usa para bug de shape/dtype, que
   deve continuar sendo fatal.
   **NÃO** usar `ImageFile.LOAD_TRUNCATED_IMAGES = True`: isso preenche o resto
   da imagem com cinza e envenena o treino em silêncio.
2. `scripts/verify_prefetch_integrity.{py,slurm}`: varredura de integridade sem
   GPU, só leitura. Modo `quick` (assinatura + `IEND`, 45 s para 204.000
   arquivos) e `FULL=1` (decodifica tudo, 4min28s). Reutilizável nas rotas b e c.

**Lição transferível:** qualquer treino longo tem que sobreviver a um arquivo
ruim. A fase 2 são 60K steps (10 dias em 2 GPUs) lendo dados do hub — a mesma
blindagem vale lá.

---

## 8. 🟢 O QUE ESTÁ RODANDO AGORA

> **Job atual: `29358`** (fase 1 retomada do step 3250 após a morte do 29267).
> Submetido em 2026-08-07. O bloco abaixo descreve a configuração, que não mudou.
> Histórico: `29267` (06/08 11:32 → 07/08 01:26, FAILED no step 3380, ver §7.5).

**Fase 1 do BokehNet — job SLURM `29267`**, iniciado em 2026-08-06 11:32.

| | |
|---|---|
| Config | `configs/train_bokeh_synth_2gpu.yaml` |
| Dados | `/workspace/data-bokeh/rota_a` (68.000 amostras, prefetch completo antes do início) |
| GPUs | 2, `grad_accum=16`, batch efetivo 32 |
| Steps | 40.000 |
| Saída | `outputs/bokehnet_synth_2gpu/` |
| Checkpoint | a cada **250 steps** (2,6 GB cada, mantém 5) |
| Upload HF | a cada **2.500 steps** → `genrefocus-bokehnet-synth-condlora` (1 repo, arquivo por step) |
| wandb | projeto `genrefocus-bokehnet`, run `bokehnet-synth-2gpu` |

**Saúde verificada no step 1550**: `loss_ema` 0,74 → 0,18; lr fez warmup até 1e-4 e entrou no
cosseno; VRAM 66,3/80 GB estável; `effective_config.yaml` confirma `defocus_source: recompute`.

**Ritmo real: 14,8 s/step.** Projeção: fase 1 ≈ **6,9 dias** (o script pede `--time=7-00:00:00`
— passa raspando). Fase 2 (60K steps) ≈ **10,3 dias** em 2 GPUs → vai precisar de 2
relançamentos, ou usar 4 GPUs (corta pela metade, mesmo resultado).

**Se o job morrer por tempo/preempção: basta re-submeter o mesmo comando.** O `resume: true`
recarrega LoRA + optimizer + scheduler + contador de steps do último checkpoint.

---

## 9. Comandos

Todos no formato `ssh dgx-H100-02 '...'` para rodar direto do Mac.

### Acompanhar a fase 1
```bash
ssh dgx-H100-02 'cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper && squeue -u $USER && grep "\[train\] step=" logs/*29267*.out | tail -5 && grep "\[upload\]" logs/*29267*.out | tail -3'
```

### Verificar integridade da pasta do prefetch (sem GPU, só leitura)
```bash
ssh dgx-H100-02 'cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper && FULL=1 sbatch scripts/verify_prefetch_integrity.slurm'
```

### Checkpoints e cota
```bash
ssh dgx-H100-02 'ls -lht /raid/user_juliadollis/projects/genrefocus_deblurnet_paper/outputs/bokehnet_synth_2gpu/bokeh/checkpoints/ | head -7; quota -s | tail -1'
```

### Sincronizar código (rodar NO MAC, antes de qualquer sbatch)
```bash
cd ~/Projects_Code/repositorio_ref/genrefocus_deblurnet_paper && bash scripts/sync_to_cluster.sh --go
```

### Smoke test (3 steps, valida o caminho de 2 condições)
```bash
ssh dgx-H100-02 'cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper && sbatch scripts/smoke_bokeh.slurm'
```

### Fase 1 (relançar após queda — retoma sozinho)
```bash
ssh dgx-H100-02 'cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper && sbatch scripts/train_bokeh_2gpu.slurm'
```

### Fase 2 (só depois da fase 1 terminar)
```bash
ssh dgx-H100-02 'cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper && TRAIN_CONFIG=configs/train_bokeh_real_2gpu.yaml INIT_LORA=outputs/bokehnet_synth_2gpu HF_REPO_ID=genrefocus-bokehnet-real-2gpu sbatch scripts/train_bokeh_2gpu.slurm'
```

### Prefetch de outra rota (sem GPU)
```bash
ssh dgx-H100-02 'cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper && DATASET=AKCITPixel3/BKXcuVXCmeRvN OUT=/workspace/data-bokeh/rota_b sbatch scripts/prefetch_bokeh.slurm'
```

### Testes locais (no Mac, sem GPU)
```bash
cd ~/Projects_Code/repositorio_ref/genrefocus_deblurnet_paper && python -m pytest tests/ -q
```

---

## 10. Mapa do código

```
genfocus_train/
  backbone.py    FLUX + LoRA. forward_train_step monta os branches, group_mask, sigma.
                 É aqui que mora o contrato com transformer_forward dos autores.
  data.py        Datasets. prepare_aligned_bokeh (resize+crop alinhado), defocus_from_depth
                 (a fórmula), LocalBokehFolderDataset (pasta do prefetch), build_dataset (roteia).
  models.py      DeblurNet / BokehNet: convertem batch → tokens → chamam o backbone.
  trainer.py     Loop, checkpoint/resume, multi-GPU manual, upload HF, smoke test.
  config.py      Schema dos YAMLs (dataclasses).
  train.py       CLI: smoke | deblur | bokeh | export.
scripts/
  *.slurm                    jobs (treino, smoke, inferência, prefetch, upload)
  prefetch_bokeh_local.py    streaming → pasta local (o coração da solução de cota)
  infer_bokeh_test.py        inferência de teste com o contrato oficial
  sync_to_cluster.sh         rsync Mac → cluster
configs/
  train_bokeh_synth_{2,4}gpu.yaml   fase 1 (40K, rota a local)
  train_bokeh_real_{2,4}gpu.yaml    fase 2 (60K, rotas b+c do hub)
  train_bokeh_smoke.yaml            3 steps, 256², rank 16
```

---

## 11. Pendências

- [ ] **Fase 1 terminar** (~6,9 dias de treino; retomada em 07/08 do step 3250) e
      então lançar a fase 2.
- [x] ~~Conferir o primeiro upload ao HF no step 2500~~ **OK**: subiu inteiro
      (928 MB, `bokeh_step2500.safetensors`) no job 29267, e o treino seguiu normal
      depois. O formato novo (1 repo, arquivo por step) está validado.
- [ ] **Rodar a varredura de integridade nas rotas b e c** antes da fase 2:
      `ROOT=/workspace/data-bokeh/rota_b FULL=1 sbatch scripts/verify_prefetch_integrity.slurm`
      (só se elas forem para pasta local; se vierem do hub direto, não se aplica).
- [ ] **Renderizador da rota a** (time de dados): alvos sintéticos saturam o blur porque o
      BokehMe nunca foi instalado. Limita a fase 1; não afeta a fase 2.
- [ ] **Inferência em imagens novas**: gerar o depth **normalizado [0,1]** (não disparidade!)
      e usar a mesma fórmula `recompute`. O `infer_bokeh_test.py` já está certo para os dados
      dos datasets; falta o caminho para uma foto qualquer.
- [ ] Decidir 2 vs 4 GPUs para a fase 2 (10,3 dias vs ~5).
- [ ] (Opcional, paper §3.3) Aperture-shape control: condição extra, treinada depois com o
      LoRA base congelado.

---

## 12. Como pedir ajuda a uma sessão nova do Claude Code

Contexto mínimo para colar:

> Estou reproduzindo o paper GenRefocus. O DeblurNet já está pronto; agora estou treinando o
> BokehNet (LoRA rank 64 sobre FLUX.1-dev, 2 condições: AIF + mapa de defocus). O treino da
> fase 1 está rodando no cluster (job SLURM, 2× H100). Leia `historico-ultimo.md` na raiz do
> projeto: ele tem o estado atual, as regras de GPU e de ambiente, e o histórico dos bugs.
> Antes de sugerir qualquer mudança em scripts `.slurm`, leia a seção 3 (regras de GPU) e a
> seção 4 (regras de ambiente) — várias "correções óbvias" já quebraram o treino antes.
