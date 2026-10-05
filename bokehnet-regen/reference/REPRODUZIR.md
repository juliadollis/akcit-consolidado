# Reproduzir — do zero até o dataset

**Escrito em 2026-09-10.** Arquivo novo; não substitui nada. As decisões estão em
`REGISTRO.md`, o contrato em `reference/CONTRATO.md`, as medições em
`reference/ACHADOS.md`, a ordem de execução em `PLANO_EXECUCAO.md`, e a procedência de cada
número em `reference/TABELA_DE_EVIDENCIAS.md`.

Este arquivo é o caminho operacional: o que baixar, quais comandos, em que ordem, quanto
tempo, quanto de disco. E — o que costuma faltar — **o que não funciona, e por quê**.

Duas advertências antes de qualquer coisa:

- **O dataset final ainda não foi gerado.** O que existe e roda de verdade é: a suíte de
  testes (650 testes, sem GPU), o laudo do renderer (job 32212), o piloto da rota C
  (job 32224, 204 pares) e o diagnóstico de máscara (job 32231). A §9 diz exatamente o que
  já rodou e o que nunca rodou.
- **O repositório estava sendo editado enquanto isto foi escrito.** Estado congelado em
  **2026-09-10 19h35**. Em 24 minutos a suíte foi de 519 para 650 testes e `src/` de 7.993
  para 9.297 linhas — a rota B e o teto de níveis por cena passaram a existir no meio da
  redação. Confira o estado antes de citar qualquer número deste arquivo.
- **Nunca cancelar job, nunca apagar nada no cluster sem perguntar, `--time` sempre alto.**
  Regra 1 de `CLAUDE.md`, e já houve um `rsync --delete` que apagou checkpoints.

---

## 0. O caminho mais curto que existe: rodar a suíte, sem GPU

Isto funciona numa máquina local, em 6 segundos, e é a primeira coisa que um avaliador
deve fazer.

```bash
cd "<raiz>/bokehnet-regen"
PYTHONPATH=src:scripts:tests \
  /Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python \
  -m unittest discover -s tests -p "test_*.py"
```

**Resultado observado em 2026-09-10 19:31:45 −03:** `Ran 650 tests in 6,584s` ·
`OK (skipped=7)`, zero falhas e zero erros.

O número se mexe: 519 às 19:07, 521 às 19:11, 650 às 19:31 — três agentes trabalhavam no
repositório em paralelo. Rode você mesmo em vez de citar este número.

### O que **não** funciona, e por quê

| tentativa | erro | por quê |
|---|---|---|
| `python3 -m unittest ...` (python do sistema) | `ModuleNotFoundError: No module named 'numpy'` | o python do sistema desta máquina não tem numpy. Use o interpretador do venv acima |
| `PYTHONPATH=src` (só `src`) | `ModuleNotFoundError` nos testes de `scripts/` | `test_publish_release.py`, `test_validate_focus_refinement.py` e `test_deblurnet.py` importam de `scripts/` e de fixtures em `tests/`. Precisa de `src:scripts:tests` |
| o comando do `README.md` (`PYTHONPATH=src python3 -m unittest ...`) | as duas coisas acima | o `README.md` está desatualizado — ele também afirma "48 testes", quando são 650 |

### Os 7 skips, e como ligar cada um

| quantos | mensagem | como ligar |
|---|---|---|
| 4 | `pyarrow não instalado nesta máquina` / `build sem cache importa pyarrow` | `pip install pyarrow` no venv. São os testes que leem parquet de verdade (3 em `test_mirror_images`, 1 em `test_lfdof`) |
| 3 | `toca a rede; ligue com BOKEHNET_HF_TESTS=1` | `BOKEHNET_HF_TESTS=1` **e** token do HF em `~/.cache/huggingface/token`. Batem no espelho privado `akcit-pixel/RealBokeh` |

**Nunca coloque o token em arquivo versionado.** `.env` é git-ignored e o exemplo é
`.env.example`, com `HF_TOKEN=` vazio.

---

## 1. O que baixar, e onde

| # | artefato | tamanho | onde vai | público? |
|---|---|---|---|---|
| 1 | `JuewenPeng/BokehMe` (git clone) | ~15 MB, com `arnet.pth` (11 MB) e `iunet.pth` (2,9 MB) **dentro do repositório** — não precisa Google Drive | `third_party/BokehMe` | sim |
| 2 | `nycu-cplab/Genfocus` (git clone) | pequeno | `third_party/Genfocus` | sim |
| 3 | `depth_pro.pt` (Depth Pro, `[7]`) | ~1,9 GB | `<models>/checkpoints/depth_pro.pt` | sim |
| 4 | BiRefNet (`[86]`) | ~1 GB | `<models>/BiRefNet/` | sim |
| 5 | espelho `akcit-pixel/RealBokeh` | **44 GB**, **96 shards** parquet | `<data>/RealBokeh_mirror` (ou o snapshot do `hf-cache`) | **privado** |
| 6 | `timseizinger/RealBokeh_3MP` — **só o `metadata/`** | 4.400 JSONs, alguns MB | `<data>/RealBokeh_3MP/<split>/metadata/` | sim |
| 7 | espelho `akcit-pixel/LFDOF` | **37 GB**, **83 shards** parquet | `<data>/LFDOF_mirror` | **privado** |
| 8 | FLUX.1-dev (só para a rota B) | dezenas de GB | `<models>/FLUX.1-dev` | sim, com aceite de licença |
| 9 | LoRA da DeblurNet (só para a rota B) | ver §6 | `<models>/deblurnet/<variante>/` | um privado, um público |

**O item 6 é o que muitos esqueceriam.** O espelho traz os pixels e o **nível ordinal**
(`level_3`), mas **não o f-number**. Recuperar o f-number é join com
`metadata/<cena>.json`, indexando `target_avs` por `level - 1`. E tem que ser os **três**
splits: a numeração de cena **reinicia** em cada um, então ler um `metadata/` só entregaria
à cena `1` de `test` a distância de foco da cena `1` de `train`, com JSON válido e
completo — o modo de falha silencioso.

### Estado medido do cluster (conferência da etapa 7, nada foi tocado)

| item | estado |
|---|---|
| `/raid` livre | 7,0 TB |
| espelho `akcit-pixel/RealBokeh` | **presente**, 96 shards, 44 GB |
| `depth_pro.pt` | presente em `vision-pipeline/checkpoints/` |
| laudo do renderer | presente |
| **BiRefNet** | **ausente** — precisa baixar |
| **`RealBokeh_3MP` bruto (metadata)** | **ausente** — precisa baixar |
| código novo no cluster | **ausente** — precisa de rsync |

---

## 2. O ambiente — e é aqui que mora a maior parte da dor

Tudo abaixo foi aprendido a duras penas, rodando. Nada é precaução hipotética.

### 2.1 A regra de ouro do `PYTHONPATH`

```bash
export PYTHONPATH=<projeto>/.pydeps-clean:<projeto>/src:<projeto>/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}
```

**PREFIXA, nunca substitui.** Substituir tira o `~/.local` do `sys.path` e some com `peft`
e `diffusers`. E **nunca** sobrescreva `PYTHONUSERBASE`, pelo mesmo motivo.

### 2.2 `.pydeps-clean`, **não** `.pydeps`

Isto não é preferência de nome. O `.pydeps` recebeu um `pip install kornia` **sem
`--no-deps`**, que arrastou `torch 2.14+cu130` junto. Como o `PYTHONPATH` **prefixa**,
aquele torch **sombrearia** o `2.9.0+cu126` do container — que é exatamente a versão contra
a qual o renderer foi verificado no job 32212. O laudo deixaria de descrever o que está
rodando.

`.pydeps-clean` tem `cupy 12.3` + `einops` + `kornia`, todos com `--no-deps`, e nada mais.

```bash
pip install --no-deps --target <projeto>/.pydeps-clean "cupy-cuda12x<13" fastrlock einops kornia
```

**`pip install --target` em pasta do projeto. Nunca no ambiente do usuário nem no
`~/.local`, que é compartilhado entre nós.**

### 2.3 A versão do cupy importa por **dois** motivos independentes

| versão | o que quebra |
|---|---|
| `cupy 14.x` | compilado contra numpy 2.x; o container tem **1.26.4** → `numpy.core.multiarray failed to import` |
| `cupy 13.x` | removeu `cupy.cuda.compile_with_cache`, que o `scatter.py` do BokehMe usa |
| **`cupy 12.3`** | **funciona** — e precisa do patch da §2.4 |

`--no-deps` é **obrigatório**: sem ele o cupy arrasta `numpy 2.2.6` para o `.pydeps` e
sombreia o 1.26.4 do container. Isso já aconteceu, na primeira tentativa.

`fastrlock 0.8.3` é dependência do cupy e **`--no-deps` não a traz** — instale
explicitamente.

### 2.4 O patch do `cupy.int`

`third_party/bokehme_cupy12_int_alias.patch` — **3 ocorrências** de `cupy.int(x)` → `int(x)`.
`cupy.int` era alias do `int` builtin (igual a `numpy.int`), removido no cupy 12; o BokehMe
é de 2022, escrito para cupy ~9/10.

```bash
cd third_party/BokehMe && git apply ../bokehme_cupy12_int_alias.patch
```

Versionado e revisável, e o sha256 do `scatter.py` **depois** do patch entra na
proveniência de cada amostra. Não é edição solta.

### 2.5 `cv2` **não pode ser importado** no container

`ImportError: GLIBC_2.38 not found`. O `~/.local` tem o opencv completo, que puxa libGL
exigindo GLIBC 2.38; o container tem **2.35**. Está documentado no `INSTRUCOES_H100.md`, e
o projeto pisou nisso mesmo assim.

Consequências no código, todas deliberadas:

- o adaptador do BokehMe injeta uma sentinela `_Unavailable` no lugar de `cv2`, que explode
  com mensagem clara se alguém passar a usar esse caminho — o `pipeline` só toca `cv2`
  dentro de ramos `if args.save_intermediate:`, que nunca executam;
- `src/dataio/writer.py` grava PNG uint16 com **PIL**, não cv2;
- `src/qc/metrics.py`, `src/qc/focus_region.py` e `src/renderer/calibration.py`
  implementam SSIM, laplaciano, erosão, média por janela e os dois redimensionamentos em
  **numpy puro**;
- a checagem do `Genfocus` é por **AST**, não por `import`, porque `pipeline/flux.py`
  importa cv2 no nível de módulo.

### 2.6 `from demo import pipeline` **não** funciona

O `demo.py` do BokehMe roda `args = parser.parse_args()` **no nível de módulo** (linha 130)
e em seguida instancia os modelos, carrega os checkpoints e executa o demo inteiro.
Importá-lo parsearia o nosso `argv`, duplicaria os modelos na GPU e escreveria em
`outputs/`.

O adaptador extrai `pipeline` e `gaussian_blur` do fonte **por AST** e executa só essas
duas definições, num namespace controlado. O sha256 do trecho extraído entra na
proveniência — prova melhor que o commit, porque identifica o corpo exato que rodou.

### 2.7 `torch.load(weights_only=False)`

Default virou `True` desde torch 2.6, e os checkpoints do BokehMe carregam objetos numpy.
`weights_only=False` é seguro **aqui e só aqui**, porque a origem é o repositório oficial
clonado por git e o sha256 de cada arquivo entra na proveniência.

### 2.8 Uma armadilha de mensagem de erro, que vale como método

A primeira versão do adaptador levantava `ImportError: "não consegui importar o BokehMe"`,
engolindo o `ModuleNotFoundError: No module named 'cupy'` por baixo. **Um erro sem fallback
que esconde o motivo não é melhor que um fallback silencioso — só falha mais alto.** A
mensagem agora carrega `type(exc).__name__: exc` e o comando de instalação.

### 2.9 GPU por **UUID**, nunca por índice

A GPU4 da h100n3 tem defeito e imprime `Unable to determine the device handle for GPU4` **no
meio do stdout** do `nvidia-smi`, desalinhando os índices em relação aos do CUDA. Todos os
`.slurm` deste repositório fazem:

```bash
free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
export CUDA_VISIBLE_DEVICES=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
```

O `grep -E "^[0-9]+,"` é o que filtra a linha de defeito. E os jobs **abortam** se a GPU
mais livre tiver mais de 2000 MiB em uso, em vez de tomar OOM.

### 2.10 Verificação de ambiente antes do job longo

```bash
python3 -c "import torch, numpy, cupy; print(torch.__version__, numpy.__version__, cupy.__version__, torch.cuda.get_device_name(0))"
```

Esperado: torch `2.9.0+cu126`, numpy `1.26.4`, cupy `12.3.0`.

---

## 3. Passo 1 — verificar o renderer (obrigatório, e é o que autoriza tudo)

Sem este laudo, `is_final_label_renderer` é declaração, não medição. O entrypoint da rota C
**recusa rodar** sem ele, e recusa também se o laudo for de outro checkout — compara três
hashes (`arnet_sha256`, `iunet_sha256`, `demo_pipeline_sha256`).

```bash
sbatch slurm/verify_renderer.slurm
```

Ou, direto:

```bash
python3 scripts/verify_renderer.py \
  --bokehme-dir third_party/BokehMe \
  --device cuda \
  --output-json output/renderer_verification.json
# flags opcionais: --scene-size 257 (default), --slope-tolerance 0.05 (default)
```

**Já rodou: job 32212, h100n3, `COMPLETED`, exit 0, 2026-09-10.** O laudo está versionado
em `output/renderer_verification.json` — é o único artefato de execução em GPU que
sobreviveu.

### O que o laudo mediu

| teste | resultado | veredito |
|---|---|---|
| disco, não gaussiana | `edge_width_ratio = 0,1434` (disco < 0,5; gaussiana ≈ 1,43) | PASSOU |
| linearidade em K | `bokeh_classical`: slope 0,96185, resíduo **8,9e-15 px** (reta exata) | PASSOU |
| escala | `bokeh_pred`: slope **0,98729**, intercept +0,7917 px, resíduo 0,4168 px (1,09%) | PASSOU |
| `highlight` | `pipeline` **não lê** `args.highlight` — a flag é inerte no nosso caminho | registrado |

K testado: **{8, 16, 32, 64, 96}**, raios esperados 3,2 a 38,4 px. E
`k_effective_factor = 0,9873` fica **gravado, não corrigido**.

### Três coisas que este passo ensina, e valem além dele

1. **`bokeh_neural` isolado não é renderer.** Num ponto de luz sobre fundo preto ele
   devolve ~128 px independente de K (slope 0,099, intercept 128,2), porque está fora da
   distribuição de treino. Confirma que `bokeh_pred` é a saída certa.
2. **Quantizar é gravação, não renderização.** Um ponto de 255 espalhado num disco de raio
   12 dá **0,56 por pixel**, que `astype(np.uint8)` trunca para zero: a imagem inteira vira
   preto e o harness mede raio 0. Medido: raio 5 → 3,2/px; raio 12 → 0,56; raio 25 → 0,13.
   A `render_fn` **tem** que devolver float.
3. **Linearidade vive na inclinação e no resíduo, não no erro ponto a ponto.** O primeiro
   veredito humano foi "SATURA (reprovou)", olhando erro relativo: 28,2% em K=8 caindo para
   0,2% em K=64. Estava **errado**, e errado de um jeito que teria feito descartar um
   renderer correto: saturação faria o resíduo **crescer** com K e a razão **cair**. O que
   existe é um **piso aditivo de ~1 px** do medidor (fonte pontual discretizada), que
   estoura o erro relativo em raio pequeno.

### Duas ressalvas honestas sobre este laudo

- **Não carrega o número do job nem a data.** O vínculo "job 32212, 2026-09-10" existe só
  em prosa. `logs/` está no `.gitignore`.
- **`bokeh_classical` e `bokeh_pred` têm raios medidos idênticos em 4 dos 5 K** — só K=96
  difere (37,961 vs 38,987). O "resíduo 0,0000 px do clássico" e o "0,4168 px do híbrido"
  são, portanto, a mesma informação vista de dois lados. Detalhe em
  `TABELA_DE_EVIDENCIAS.md`, §9(b).

---

## 4. Passo 2 — o piloto da rota C, com todos os limiares em modo medir

**Nunca rode o completo antes do piloto.** Foi assim que o `--k-max 300` do pipeline antigo
virou **47,0% de amostras com K censurado no valor exato do teto**, sem que nada
denunciasse.

```bash
sbatch slurm/route_c_pilot.slurm
# variáveis: PILOT_N (200), MIRROR_DIR, RAW_DIR, MODELS_DIR, OUTPUT_DIR
```

O job faz, em ordem: seleção de GPU por UUID → checagem de ambiente → **a suíte de testes**
→ **smoke de 2 amostras** → piloto de `PILOT_N` → proposta de limiares.

O smoke de 2 é a disciplina do `CLAUDE.md`, e existe porque `compileall` e parsing AST
**não pegam `NameError`** — foi assim que um `import json` removido matou um pipeline na
primeira amostra depois de passar por revisão.

### A CLI completa da rota C

```bash
python3 scripts/run_route_c.py \
  --source realbokeh \            # obrigatório: {realbokeh, lfdof} — mas ver §6.1
  --output-dir output/c_pilot \   # obrigatório
  --models-dir <models> \         # obrigatório: contém checkpoints/depth_pro.pt e BiRefNet/
  --bokehme-dir third_party/BokehMe \   # obrigatório
  --renderer-report output/renderer_verification.json \  # obrigatório
  --mirror-dir <data>/RealBokeh_mirror \
  --raw-dir <data>/RealBokeh_3MP \
  --pilot 200 \                   # sorteia CENAS inteiras com seed
  --device cuda --seed 0 --val-fraction 0.05
```

Outras flags: `--rebuild-index`, `--store-source-images`, `--sensor-width-mm`, `--limit`,
os 11 limiares (`--min-calibration-ssim`, `--min-focus-mask-sharpness-ratio`,
`--min-mask-area-ratio`, `--max-mask-area-ratio`, `--max-mask-border-coverage`,
`--min-aif-laplacian-variance`, `--max-bokeh-over-aif-sharpness`, `--min-aif-f-number`,
`--min-mask-iou`, `--min-focus-region-retention`, `--min-depth-useful-levels`), os quatro
botões do refinamento (`--focus-retention-long-side`, `--focus-retention-window-px`,
`--focus-top-fraction`, `--focus-agreement-floor`) e os três da busca (`--k-min`,
`--k-max`, `--k-absolute-max`).

### `--pilot` e `--limit` são coisas diferentes, e é de propósito

`--pilot N` **sorteia cenas inteiras** com seed. `--limit N` é um teto cru de amostras
aceitas. Os shards do espelho estão agrupados por cena, então os 200 primeiros pares da
ordem de disco saem de **uma dúzia de cenas** — e os limiares sairiam calibrados para
aquela dúzia. Use `--pilot` para calibrar.

### O que o piloto imprime, sempre, inclusive em caso de erro

Três resumos, num bloco `finally`:

1. **`writer.stats`** — amostras gravadas e MB.
2. **`RejectionLog.summary()`** — o histograma de motivos de rejeição. **Sem ele não dá para
   calibrar limiar nenhum, e é ele que denuncia fallback novo:** uma categoria que
   desaparece significa que alguém "consertou" o caminho de falha.
3. **`RouteCStats.summary()`** — percentis de `k_value` e de `k_analytic` **separados**, com
   a explicação de que um é borrão incremental e o outro absoluto; percentis de SSIM; e a
   composição do lote por `focus_source`.

Arquivos produzidos em `--output-dir`: `manifest.jsonl` (27 campos por linha),
`split.json`, `meta/<id>.json`, `depth/<id>.png` (disparidade uint16, lado longo 768),
`mask/<id>.png` (a **região final** de foco), `rejections.jsonl`, `source_rejections.jsonl`,
`source_images.jsonl` (sha256 da AIF e da bokeh por amostra) e `run_config.json`.

**Já rodou: job 32224**, 204 pares, 162 aceitas, 29 cenas — mas com a versão **anterior** ao
refinamento automático, que descartava 20,6% das amostras. Os números dele são a **linha de
base** contra a qual o refinamento tem que ser medido, não o resultado atual.

---

## 5. Passo 3 — julgar o piloto **antes** de gerar o lote

Dois scripts, nesta ordem. O primeiro é o que decide se vale gerar.

### 5.1 A validação do refinamento da região em foco

```bash
python3 scripts/validate_focus_refinement.py \
  --pilot-dir output/c_pilot \                 # obrigatório
  --output-json output/focus_validation.json \ # obrigatório
  --raw-dir <data>/RealBokeh_3MP \             # ou --gabarito-json; um dos dois é exigido
  --top-divergences 10
```

Compara o `focus_disparity` gravado contra `1/focus_plane_distance_m` — a distância de foco
**medida na captura**, publicada pela origem com incerteza mediana de ±0,010 m. Reporta em
três blocos: por `focus_source` (declaradamente **confundido**, porque o grupo `birefnet` é
por construção o subgrupo em que o segmentador já concordava), **pareado na mesma amostra**
(usando `focus_disparity_from_initial_mask`) — e é este que decide —, e separação de causas
entre máscara e profundidade.

**A régua é 35,2% dentro de ±25%**, do piloto 32224 (57 de 162 amostras, 29 cenas). Se o
refinamento não bater isso no bloco pareado, o relatório imprime **PIOROU** e diz para não
congelar nada e não gerar o lote. O parâmetro a revisar primeiro é `focus_top_fraction`,
depois `focus_agreement_floor`.

**O que o relatório declara que NÃO consegue fazer**, e vale ler: separar viés de escala do
Depth Pro de erro de máscara **por amostra** exigiria uma profundidade métrica de
referência na cena, e a RealBokeh publica uma distância só. O que dá são três limites:
alcançabilidade (se o gabarito cai fora de `[disparity_min, disparity_max]` da amostra,
nenhuma máscara o alcançaria), escala global (dividir cada razão pela mediana global sobra a
dispersão, que nenhum fator único explica) e isolamento das amostras com incerteza relativa
acima de 10%.

O veredito **não** vira exit code de propósito — sair 1 num "piorou" faria um job em lote
tratar informação como falha.

### 5.2 A proposta de limiares

```bash
python3 scripts/calibrate_thresholds.py \
  --pilot-dir output/c_pilot \                       # obrigatório
  --output-json output/thresholds_proposta.json \    # obrigatório
  --reviewed-csv <painel>.csv --min-precision 0.95   # opcionais
```

Lê as distribuições de `quality` dos `meta/*.json` e o histograma de `rejections.jsonl`, e
**propõe** cortes por percentil para 10 gates. **Não aplica nenhum.** Para o limiar de SSIM
ele implementa o caminho recomendado — painel humano de 300 a 500 casos revisados, corte por
precisão com monotonicidade a partir do topo — em vez de percentil cego, e diz por escrito
que percentil sozinho é ponto de partida ruim.

**Três lacunas conhecidas desta ferramenta**, e importa saber antes de confiar nela:

- `min_aif_f_number` **não** aparece nas 10 entradas (são 11 limiares em `RouteCConfig`);
- `focus_depth_m_min`/`max` também não, e esses dois **bloqueiam por default** — então o
  piloto não os calibra (detalhe em `DESVIOS_DO_PAPER.md`, §3.4);
- ela **ignora o campo `detail`** do `rejections.jsonl`, que é onde o valor rejeitado está
  gravado.

---

## 6. Passo 4 — o lote completo, com os limiares escolhidos por um humano

```bash
sbatch --export=ALL,THRESHOLDS='--min-calibration-ssim 0.72 --min-mask-area-ratio 0.01 ...' \
  slurm/route_c_full.slurm
```

O job **recusa começar** com `THRESHOLDS` vazio, e a mensagem de erro explica o que fazer.
Se a intenção for mesmo gerar tudo sem filtro para calibrar depois, é preciso passar
`THRESHOLDS='--seed 0'` **explicitamente** — a escolha é permitida, o silêncio não.

Ele também: confere que o SIF, o `demo.py` e o laudo existem; roda a suíte; roda a rota C;
e no fim roda `publish_release.py` **sem** `--yes`, ou seja, só valida.

**Retomada**: a rota pula amostras já gravadas por `sample_id`, e **rejeitadas são
reprocessadas** — um gate recalibrado pode aceitá-las depois, e é por isso que o motivo fica
gravado em vez de a linha sumir. Relançar continua de onde parou; **nada é apagado**.

### 6.1 O LFDOF: o adaptador existe e **não está ligado**

`--source lfdof` está no parser e falha com `SystemExit: fonte 'lfdof' ainda não tem
adaptador`. **A mensagem é falsa hoje**: `src/sources/lfdof.py` (879 linhas, 112 testes) e
`src/sources/lfdof_images.py` (403 linhas) existem, com 28 medições próprias — 11.972 pares,
840 cenas, 83 shards, `level` 1-based confirmado, alinhamento medido (11.528 `aligned` /
204 `misaligned` / 240 `shift_*px`). Falta o `_carrega_fonte` do entrypoint tratar o caso.

O paper exige o LFDOF em quatro lugares. Sem ele, o que se gera é **meia rota C**.

E **antes de usar `--store-source-images` com o LFDOF**: a página upstream não tem licença
explícita. Isso bloqueia publicar os pixels, não o rótulo.

### 6.2 Orçamento de disco

| configuração | custo |
|---|---|
| rotas B+C, rótulo + referência | ~31 GB |
| rota C autocontida (`--store-source-images`) | ~49 GB |
| rota A a 70K amostras, 0,5 / 1,0 / 2,0 MP | 60,4 / 82,5 / 126,6 GB |

**Aviso de procedência**: nenhum desses números tem medição registrada, e a "folga de
115 GB" contra a qual eles foram decididos não tem `quota`/`df` gravado em lugar nenhum —
enquanto a conferência do cluster mediu 7,0 TB livres em `/raid`. Ver
`TABELA_DE_EVIDENCIAS.md`, §17. Rode `dataio.writer.estimate_disk_budget` e `quota -s` antes
de dimensionar, e registre a saída.

### 6.3 Quanto tempo leva — o que se sabe e o que não

| etapa | custo |
|---|---|
| suíte de testes | **5,6 s** (medido) |
| `detail_retention` a 1500×2000 | **291 ms** por amostra (CPU, medido) |
| `refine_focus_region` ponta a ponta | **391 ms** por amostra (CPU, medido) |
| rota C inteira por amostra, com dublê de renderer barato em CPU | **~4,2 s** (medido), dos quais ~0,39 s de refinamento |
| **rota C por amostra, com GPU e BokehMe real** | **NÃO MEDIDO** |
| **rota A, 70K renders do BokehMe** | **NÃO MEDIDO** |

O sweep da Eq. 5 gasta **14 a 18 avaliações** típicas (teto 40) por amostra, cada uma uma
renderização de BokehMe a 512 px. É o item dominante do orçamento de GPU e **não tem
estimativa em lugar nenhum do repositório**. Cronometre o piloto antes de dimensionar
`--time` — e ponha `--time` alto de qualquer forma (`20-00:00:00` é o que os `.slurm` usam).

### 6.4 Uma economia grande que ainda não foi feita

`process_pair` chama `calibrate_k` **antes** de `enforce_gates`. Toda amostra reprovada por
máscara, nitidez, f-stop ou profundidade paga 14 a 40 renderizações de BokehMe antes de ser
descartada. Só `calibration_ssim_is_reliable` depende do sweep; os outros dez não. É a maior
economia disponível e **não muda rótulo nenhum**.

---

## 7. Passo 5 — validar e publicar o release

```bash
python3 scripts/publish_release.py --release-dir output/c_realbokeh          # só valida
python3 scripts/publish_release.py --release-dir output/c_realbokeh \
  --repo-id <org>/<nome> --yes [--public] [--allow-existing]                 # publica
```

Privado por default. Recusa publicar por cima de repositório que já tem arquivos sem
`--allow-existing`. Sem `--yes`, não sobe nada.

### As 11 checagens que **reprovam** — e todas as 11 têm teste que as faz reprovar

| reprova quando | por quê |
|---|---|
| `manifest.jsonl` vazio ou ausente | não há release |
| falta `split.json` | split em config diverge entre runs e some no rsync |
| `sample_id` repetido | a contagem publicada infla sem que nada denuncie |
| amostra sem arquivo em `depth/`, `mask/` ou `meta/` | listada e ausente |
| `validate_metadata` falha | pega `max_coc` fora de 100 — o normalizador escondido — e proveniência incompleta |
| `control_version` divergente | duas convenções no mesmo release |
| `depth_backend` divergente | métrica e disparidade normalizada não são a mesma coisa |
| amostra da rota C sem sha256 de origem em `source_images.jsonl` | o release não prova contra quais bytes o K foi calibrado |
| cena no manifesto e fora do `split.json` | fronteira do split incompleta |
| vazamento de split (cena nos dois lados) | validação mediria memorização |
| K censurado **acima de 20%** | eram 47,0% no release anterior |

Isso é a lição do `check_no_leak` antigo aplicada: aquele gate era **tautológico** —
`scene_splits[scene].add(split.of(scene))` sobre uma função pura, então `len(v) > 1` era
impossível por construção, o ramo de detecção era código morto, e o teste chamado
`test_detecta_vazamento_por_cena` afirmava `clean is True`. **Um gate que não pode reprovar
é pior que gate nenhum: dá garantia falsa.**

### Quatro avisos que **não** reprovam

Release não autocontido; `val` vazio; censura abaixo de 20%; e mais de 50% do lote com a
região em foco refinada — este último pedindo o laudo de
`scripts/validate_focus_refinement.py`.

### Três lacunas conhecidas do publicador

- **Nenhuma checagem de uniformidade de `deblur_variant`.** `PLANO_EXECUCAO.md:104` afirma
  que existe; não existe. Um release meio de uma variante e meio de outra é o modo de falha
  exato do `kfix`, e passaria.
- **O card traz 35,2% e 20,6% *hardcoded*.** Esses são os números do piloto 32224, e vão
  para qualquer release — inclusive um cujo refinamento tenha medido outra coisa.
- **O card afirma duas coisas que a validação não confere**: que `rejections.jsonl` faz
  parte do release, e que o renderer foi verificado em GPU. As duas são verdadeiras na
  prática (o `RejectionLog` grava; o entrypoint recusa rodar sem laudo), mas o release não
  as **prova**.

---

## 8. As rotas A e B — o que existe e o que não

### Rota B

Estado em 2026-09-10 **19h35** — e esta é a parte do repositório que mais se mexeu durante
a redação:

| peça | estado |
|---|---|
| `src/model_runtime/deblurnet.py` | **existe**, 967 linhas, 65 testes. `DeblurVariant` fechada, `ResizePolicy` com o recorte medido, 24 campos de proveniência |
| `src/routes/route_b.py` | **passou a existir entre 19:11 e 19:32**, 1.168 linhas, 104 testes. Não auditado por este documento |
| `scripts/run_route_b.py` | **passou a existir**, 623 linhas. Não auditado |
| `src/sources/bokehdiffusion.py` | **não existe** — o adaptador da fonte da rota B |

**A tripla é indivisível**, e isto é o defeito B1 tornado inexpressável:

| variante | repo HF | arquivo (spec) | `main_adapter` | `pinned_sha256` |
|---|---|---|---|---|
| `OURS_MAIN_COND` | `juliadollis/genrefocus-deblurnet-paper-4gpu` | `deblur.safetensors` | `"deblurring"` | — (exige `expected_lora_sha256`) |
| `OURS_COND_ONLY` | `juliadollis/genrefocus-deblurnet` | `deblur.safetensors` | `None` | `a1ed05e4…` |
| `OFFICIAL_COND_ONLY` | `nycu-cplab/Genfocus-Model` | `deblurNet.safetensors` | `None` | — |

**A NOSSA é `ours_cond_only`** (retreino de 2026-09, `lora_on_main: false`).
`ours_main_cond` é o treino antigo, de outro repositório; ela e a nossa compartilham o
**nome do arquivo**, e era essa a única trava quando o lote `b_release` inteiro (13.615
amostras) saiu com o peso cond-only invocado como main+cond. Por isso `ours_main_cond`
só constrói com o sha256 declarado, e `ours_cond_only` é fixada por `pinned_sha256`.

**O nome do arquivo não é identidade, nos dois sentidos.** Medido com
`HfApi.repo_info(files_metadata=True)`, sem baixar: o peso da `ours_cond_only` está
publicado como **`deblurNet.safetensors`** (`a1ed05e4…`, 1.854.533.784 B) — o nome que a
variante oficial declara —, e em disco no cluster se chama `deblur.safetensors`. O módulo
resolve isso pela regra "hash mais forte que nome": com `pinned_sha256` casando, o nome não
decide nada; sem pin, o nome volta a ser a evidência. A proveniência grava o nome **real**
do arquivo carregado (`deblur_weight_filename`) e o nome canônico da spec
(`deblur_weight_filename_declared`).

Baixe **sempre** por `model_runtime.deblurnet.resolve_weights(variante)`, que deriva repo e
nome de arquivo da variante — com uma ressalva medida: para `ours_cond_only` o nome no Hub
(`deblurNet.safetensors`) não é o da spec (`deblur.safetensors`), então o download direto
não acha o arquivo; use `--deblur-weights-dir` com a cópia do cluster, ou baixe à mão (o
pin identifica o peso sob qualquer nome). `hf_hub_download` na mão permite pedir o arquivo de uma
variante no repositório da outra — e **os dois `.safetensors` têm as mesmas chaves de
LoRA**, então nenhuma inspeção do peso detecta o cruzamento. A diferença entre main+cond e
cond-only vive no **roteamento**, não no arquivo. Rodar um peso main+cond com
`main_adapter=None` produz AIF **lavada**, e já custou LPIPS ~0,85.

Por isso há também uma checagem **por AST** que exige que `generate` ainda declare
`main_adapter`: a assinatura termina em `**params: dict`, então um kwarg renomeado pelo
upstream seria **absorvido em silêncio** — sem `TypeError`, sem log, só uma AIF lavada.

**Decisão de ordem registrada** (`PLANO_EXECUCAO.md`): a rota B roda **com a DeblurNet
oficial** por ora, porque a nossa ainda está treinando; um release e um repositório HF por
variante, com `sample_id` **igual nas duas** para permitir comparação pareada.

Sobre o FLUX.1-dev: a proveniência grava `flux_backbone_fingerprint` — caminho relativo +
tamanho de todo arquivo, mais o conteúdo integral dos `.json`/`.txt` pequenos — e **não** um
hash de conteúdo. São dezenas de GB; hashear por run é inviável, e **um hash que ninguém
roda é pior que nenhum**. A chave `flux_backbone_fingerprint_kind` diz isso no próprio dado.

### Rota A

Nada implementado. E há uma **dependência de ordem, não uma preferência**: o amostrador de K
lê as distribuições de B e C, que hoje são `k = 50` constante e 47% no teto. Amostrar dessa
"distribuição" hoje é amostrar de duas constantes.

---

## 9. Ordem completa, num quadro

```
 0. suíte de testes, local, sem GPU ............................ 6,6 s    ✔ RODOU (650 testes)
 1. clonar BokehMe + aplicar o patch do cupy.int ............... minutos
 2. .pydeps-clean com cupy 12.3 --no-deps ...................... minutos
 3. baixar depth_pro.pt, BiRefNet, metadata/ da RealBokeh_3MP .. ~3 GB
 4. rsync do código para o cluster
 5. verify_renderer.slurm ...................................... minutos  ✔ JOB 32212
 6. route_c_pilot.slurm (suíte -> smoke 2 -> piloto 200) ....... ?        ✔ JOB 32224 (versão anterior)
 7. validate_focus_refinement.py  <-- a régua é 35,2% .......... segundos ✘ NUNCA RODOU
 8. calibrate_thresholds.py -> escolha HUMANA dos limiares ..... segundos ✘ NUNCA RODOU
 9. route_c_full.slurm com THRESHOLDS preenchido ............... ?        ✘ NUNCA RODOU
10. publish_release.py (valida; --yes é manual) ................ segundos ✘ NUNCA RODOU
11. ligar o LFDOF no entrypoint e repetir 6-10 ................. ?        ✘ NUNCA RODOU
12. rota B (código apareceu em 19:32; falta a fonte) ........... —        ✘ NUNCA RODOU
13. rota A (bloqueada por B e C) ............................... —        ✘ NÃO EXISTE
```

Jobs que rodaram, todos em 2026-09-10, h100n3: **32212** (laudo do renderer), **32224**
(piloto da rota C), **32231** (diagnóstico de máscara vazia). Rodaram ao lado do treino
**32186** (`deblur-n2-4gpu`), que **não é nosso e não foi tocado** — a QOS `onejob` permite
2 jobs rodando.

**Dos três, só o 32212 deixou artefato versionado.** Os números do 32224 e do 32231 existem
só como prosa em `reference/MEDICAO_PLANO_FOCO.md`, porque `logs/` e `output/` estão no
`.gitignore`. Conserto barato e de alto valor para a auditabilidade: versionar, por job, o
`run_config.json`, o `rejections.jsonl` agregado e o resumo de stdout — sem os pixels.

---

## 10. Checklist do avaliador cético

Em ordem de custo crescente. Os quatro primeiros não precisam de GPU nem de credencial.

| # | pergunta | como responder |
|---|---|---|
| 1 | os testes passam? | o comando da §0. Observado: 650 testes, `OK (skipped=7)`, zero falhas |
| 2 | os gates podem reprovar, ou só passam? | `tests/test_publish_release.py` tem um teste por checagem que a faz **reprovar**. `tests/test_gates.py` exercita os dois sentidos. E `scripts/verify_renderer.py:125` é o contraexemplo: `highlight_decidido` é `True` incondicional |
| 3 | o mapa de defocus é reconstruível só dos escalares? | `tests/test_route_c.py::test_metadado_em_disco_reconstroi_o_mapa`, que também prova que nenhum arquivo `*defocus*` é escrito |
| 4 | `max_coc` pode variar por amostra? | não: não é campo nem parâmetro, e `validate_metadata` rejeita. Prove passando `10.510746` |
| 5 | a análise dimensional fecha? | `CONTRATO.md:59-68`, e `tests/test_contract.py` trava a conversão mm↔m nas duas direções |
| 6 | o renderer é disco ou gaussiana? | `output/renderer_verification.json`: `edge_width_ratio = 0,1434`. E o discriminador teórico da gaussiana (1,43) é refazível: `(2,146 − 0,459)/1,177` |
| 7 | de onde vem cada número que o projeto afirma? | `reference/TABELA_DE_EVIDENCIAS.md` — inclusive os 18 que **não** têm procedência |
| 8 | onde divergimos do paper, e por quê? | `reference/DESVIOS_DO_PAPER.md` — 12 declarados, 6 acidentais, 24 silêncios |
| 9 | os números de controlabilidade (+0,44 → +0,83) são refazíveis? | **não.** O harness de LVCorr não está neste repositório. É a lacuna mais séria |
| 10 | o dataset existe? | **não.** Nenhum lote completo foi gerado |
