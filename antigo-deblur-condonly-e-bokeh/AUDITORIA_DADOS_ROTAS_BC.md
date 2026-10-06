# Auditoria dos dfs das rotas b e c

> Feita em 2026-09-03 para decidir se os 6 canais geométricos propostos são
> executáveis. Método: dois agentes independentes com ferramentas diferentes,
> mais uma terceira medição minha para desempatar onde eles divergiram.
>
> **Nada aqui é amostra, salvo onde escrito.** As contagens vêm do `null_count`
> e do `num_rows` dos footers Parquet, ou da decodificação da coluna inteira.
> Ferramentas em `scripts/audit_hf/`.
>
> Contexto e consequências: `PLANO_CONDICIONAMENTO_GEOMETRICO.md`.

---

## 0. Por que a ferramenta é estranha

O `datasets-server` está quebrado para estes repos, então `/statistics` e
`/filter` não são utilizáveis:

```
is-valid  rota b : viewer=true  filter=true  statistics=false
is-valid  rota c : viewer=false filter=false statistics=false
/statistics rota b -> HTTP 500 "I/O error: Permission denied (os error 13)"
/filter     rota b -> {"error":"Unexpected error."}
/info, /parquet rota c -> vazio (config-info e config-parquet failed)
kfix (privado)  -> "Private datasets are only supported for PRO users"
```

A medição foi feita lendo **footers Parquet por HTTP Range** e decodificando só
as colunas escalares. Nenhuma coluna de imagem foi baixada em massa; apenas 15
PNGs de `depth` individuais, via `/rows`, para o teste da seção 3.

---

## 1. Contagens exatas

Schemas de b e c são **idênticos**, 16 colunas, mesmos tipos. A divergência é
toda de preenchimento, e é quase um espelho.

| coluna | ROTA B (n=11.635) | ROTA C (n=2.932) |
|---|---|---|
| `stem` | 0 nulos, 11.635 distintos (`b_*`) | 0 nulos, 2.932 distintos (`c_*`) |
| `s1` | 0 nulos; 2,9e-6 a 0,974; med 0,0234 | 0 nulos; 1,4e-5 a 0,957; med 0,2027 |
| `k` | 0 nulos, **constante 50,0** (fallback) | 0 nulos; 0 a 300; med 251,65 |
| `exif` | **11.635 preenchidas (100%)** | **2.932 NULAS (100%)** |
| `calibration_ssim` | **11.635 NULAS (100%)** | 0 nulos; 0,338 a 0,991; med 0,871 |
| `qc` | 100% preenchida | 100% preenchida |
| `shape_kernel` | 100% NULA | 100% NULA |
| `source_path` | 0 nulos (URLs Flickr) | 100% NULA |
| `source_aif` / `source_bokeh` | 100% NULAS | 0 nulos (`RealBokeh_3MP/train/gt/...`) |

**Nenhuma coluna escalar carrega escala métrica em nenhuma das duas rotas.**
A escala vive exclusivamente na tabela kfix.

### Armadilha de string vazia, descartada com prova
`null_count` não distingue `NULL` de `""`. Mas o `min` estatístico do footer é o
menor valor lexicográfico, e `""` seria o mínimo absoluto. O `min` de `exif` na
rota b é `{"f_number": 1.39...`. Logo, zero strings vazias. Onde o `/first-rows`
mostra `exif: ""` na rota c, o valor real é `NULL`.

### Achado colateral
A rota c é **2.932/2.932 RealBokeh_3MP split `train`, zero LFDOF**. A descrição
"LFDOF e RealBokeh" que circula nos documentos está errada.

---

## 2. Tabela kfix: cobertura exata

`juliadollis/rota-b-kfix-eq3`, 11.635 linhas, 14 colunas (`stem` + 13 float64),
zero nulos e zero NaN.

```
|kfix|            = 11.635        |kfix ∩ rota b| = 11.635  (100,00%)
|rota b|          = 11.635        |kfix ∩ rota c| =      0  (  0,00%)
|rota c|          =  2.932        |rota b ∩ rota c| =    0
kfix \ rota b = 0   rota b \ kfix = 0     -> bijeção perfeita
cobertura da união (14.567) = 79,87%
```

Faixas medidas:

| coluna | min | mediana | max | distintos |
|---|---|---|---|---|
| `k_eq3` | 833,77 | 16.553,9 | 48.514 | 11.631 |
| `z_focus_m` | 0,2191 | 3,1101 | 10.000 | 11.564 |
| `z_min_m` | 0,1052 | 1,9767 | 26,594 | 11.632 |
| `z_max_m` | 0,3621 | 70,800 | 10.000 | 8.647 |
| `pixel_ratio` | 22,22 | 42,625 | 277,33 | 542 |
| `coc_p99_px` | 0,1002 | 4,665 | 112,0 | 11.635 |
| `k_antigo` | **50,0 constante** | | | 1 |
| `max_coc_calibrado` | **10,5107 constante** | | | 1 |

### Como a kfix foi derivada (engenharia reversa numérica)
`scripts/rotab_kfix.py` **não existe neste repositório**; rodou só no cluster.
O procedimento foi reconstruído a partir dos escalares, com erro nulo:

```
k_eq3 == f_mm² · D_foco_mm / (2·F·(D_foco_mm − f_mm)) · pixel_ratio
   erro relativo máximo = 0,000e+00 em 11.635/11.635      (Eq. 3 do paper, exata)

coc_max_px == k_eq3 · max(|1/z_min_mm − 1/z_foco_mm| , |1/z_max_mm − 1/z_foco_mm|)
   < 1e-6 em 11.635/11.635      (z_min e z_max são os extremos do MESMO mapa)

max_coc_calibrado = 10,510746 = percentil 82,66% de coc_p99_px  (calibração GLOBAL)
```

Entradas exigidas: EXIF (`f_mm`, `f_number`, `focal35_mm`), `pixel_ratio`, e um
mapa de profundidade **métrica** por imagem com `z_focus` = mediana dentro da
`foreground_mask` (Eq. 4). O `max_coc` só é computável depois de processar tudo.

---

## 3. `depth` é profundidade métrica, não disparidade

O repositório se contradizia: `data.py:483` assume métrica,
`HANDOFF_PROJECT_HISTORY.md:428` afirma disparidade. Os dois agentes chegaram a
leituras opostas. Resolvido lendo pixels.

Teste, que **não usa `s1`**: recalcular `coc_p99_px` a partir dos pixels de
`depth` sob cada convenção e comparar com o valor gravado na kfix.

```
H_M (métrica):     z(x)   = z_min + d01(x)·(z_max − z_min)
H_D (disparidade): 1/z(x) = 1/z_max + d01(x)·(1/z_min − 1/z_max)
CoC(x) = k_eq3 · |1/z(x) − 1/z_focus|      (z em mm, como em data.py:485)
```

| hipótese | erro mediano | erro máximo | abaixo de 1% |
|---|---|---|---|
| **métrica min-max** | **0,011%** | 0,16% | **15/15** |
| disparidade min-max | 3,008% | 75,33% | 4/15 |

15 amostras, offsets 0 e 4000. As 4 em que a disparidade também passa são
degeneradas (faixa de z estreita, as duas convenções coincidem).

**Conclusão: `data.py:483` está correto.** `depth` é profundidade métrica
min-max normalizada por imagem, 0 = mais perto.

O decodificador PNG usado foi validado por round-trip nos 5 filtros e 3 formatos
(`scripts/audit_hf/test_png16.py`) e contra um fato conhecido: `defocus_map`
volta com `max = 65535` exato, a assinatura da normalização por imagem que o time
já tinha medido de forma independente.

### 3.1 A inconsistência que sobra: `s1` contra `z_focus_m`
Se os dois planos de foco fossem o mesmo número em unidades diferentes, valeria
`s1 = (z_focus − z_min)/(z_max − z_min)`. Medido nas 11.635:

```
|s1_implicado − s1_gravado|:  mediana 0,00308   p90 0,07927   max 0,48821
   < 1e-4 : 2.139/11.635 (18,4%)
   < 1e-2 : 7.598/11.635 (65,3%)
```

O caminho `recompute` usa `s1`; o caminho `kfix` usa `z_focus_m`. **São planos de
foco diferentes.** Não invalida a convenção de profundidade, mas é uma
inconsistência não documentada entre os dois caminhos do dataloader, e o modelo
`kfix` já treinado foi supervisionado com o segundo.

---

## 4. Focal em pixels

| | rota b | rota c |
|---|---|---|
| `exif` | 100% preenchida | **100% NULA** |
| subchaves | exatamente 5: `f_number`, `focal_length`, `focal_length_35`, `pseudo_aif`, `flickr_url` | nenhuma |
| `pixel_ratio` | na kfix, 0 nulos | ausente |
| **`fx` derivável** | **11.635/11.635 (100%)** | **0/2.932 (0%)** |

Fórmula, com identidade verificada (diferença mediana zero):

```
fx_px = f_mm · pixel_ratio = W_px · focal_length_35 / 36
```

Não existe coluna de largura/altura, mas `W` é recuperável invertendo o
`pixel_ratio`, e o valor implícito bate com o lado maior da imagem **gravada** em
**900/900** linhas conferidas em 9 offsets (modas: 1024 em 9.210, 1023 em 2.223).

**O medo do fator de resize irrecuperável foi refutado:** o `pixel_ratio` foi
computado sobre a imagem já redimensionada, então `fx` é coerente com os pixels
gravados. Usar `max(W,H)` em vez da largura está correto, porque o pixel é
quadrado e 36 mm é a dimensão longa do full-frame; 32 das 100 amostras
conferidas são retrato e todas batem.

**Ressalva que ninguém tinha:** o dataloader reescala o lado menor para 512 e
recorta. Isso muda `fx` por `512/min(W,H)`, fator que varia por amostra
(1024x574 dá 0,892; 624x1024 dá 0,821). É recuperável em runtime a partir de
`img.size`, mas **nenhum código hoje aplica essa correção**.

---

## 5. Sentinelas e degenerações que nenhum gate atual pega

| achado | contagem | por que passa despercebido |
|---|---|---|
| `z_max_m == 10.000,0` exato (teto do Depth Pro) | **2.988/11.635 (25,7%)** | invisível no mapa de defocus, que usa `1/z`; fatal para `s`, `n_x`, `n_y`, `K~`, que usam `z` linear |
| `z_max_m >= 1.000 m` | 3.136 (27,0%) | idem |
| Cena útil em menos de 256 níveis uint16 | **2.870 (24,7%)** | derivada segunda de campo com poucos degraus é ruído de quantização. No subgrupo `z_max >= 1000`, a mediana é **23 níveis** |
| `k == 300` na rota c (teto do sweep da Eq. 5) | **1.379/2.932 (47,0%)** | o filtro atual só descarta `k <= 0`, pelo argumento do intervalo aberto. O teto é sentinela pelo mesmo argumento e ninguém o remove |
| `k == 0` na rota c | 51 (1,7%), SSIM mediano 0,759 | o limiar de SSIM não os pega; já documentado no DECISOES_FASE2 |
| `coc_p99_px >= max_coc` (mapa satura) | **2.018/11.635 (17,3%)** | `max_coc_calibrado` é global (10,5107), não por amostra; o `diag_mapa_kfix.py` olha 24 amostras e aprova pela mediana |
| `coc_p99_px < 5%` do teto (mapa quase morto) | 177 (1,5%) | idem |

Rota c com K estritamente interior **e** `cal_ssim >= 0,6`: **1.483/2.932 (50,6%)**.

---

## 6. Benchmarks e o E_bleed

| benchmark | linhas | tem `depth`? | metadados |
|---|---|---|---|
| `juliadollis/bokeh-bench-realbokeh-test-v2` | 217 | não | `focal_length`, `focus_plane_distance`, `iso`, `ev`, `nivel_bokeh`, `cena_id` |
| `juliadollis/lf-bokeh-repro-blb` | 500 | não, mas tem **`disparity`** (500/500) | `k_ref`, `focus_distance`, `f_stop`, `focal_length` |
| `akcit-pixel/RealDOF` | 50 | não | nenhum |

**Isso NÃO bloqueia o E_bleed.** O pipeline de avaliação já roda o Depth Pro em
toda imagem de benchmark e salva o mapa em `.npy`
(`vision-pipeline/inference/src/pipelines/bokeh_net.py:93-141`). A profundidade
já é calculada, só não é gravada no df. Como é a mesma profundidade em todas as
condições comparadas, o estimador cancela na comparação pareada.

Detalhe não usado por ninguém: `bokeh_net.py:133` chama
`depth_model.infer(img_t, f_px=None)`. Com `f_px=None` o Depth Pro **estima a
focal** e devolve `focallength_px` junto da profundidade métrica, no mesmo
forward. `grep -rn "focallength_px"` no repositório inteiro retorna **zero usos**:
o valor é calculado e descartado nas três cópias do pipeline.

### Falso vazamento, descartado
159 dos 217 `cena_id` do benchmark RealBokeh coincidem numericamente com ids da
rota c. **Não é vazamento:** a rota c é `RealBokeh_3MP/train/...` (2.932/2.932) e
o benchmark é `..._test_...`; o id reinicia por split. Continua valendo que os
dois vêm da mesma fonte e protocolo, o que a própria tabela curada já rotula como
"mesma distribuicao do treino, mede qualidade e nao generalizacao".

---

## 7. Dispersão observada, base para o limiar de relevância prática

Da tabela `juliadollis/genrefocus-resultados-curados`, 24 linhas, lida inteira.

| benchmark | modelos | piso de identidade (LPIPS) | amplitude LPIPS entre modelos |
|---|---|---|---|
| LF-Bokeh repro (BLB) | 5 | 0,2371 | 0,0300 |
| RealBokeh test v2 | 6 | 0,3587 | 0,2320 |
| RealDOF | 7 | 0,3279 | 0,1519 |

Piso de ruído (mesmo modelo, dois checkpoints):
```
RealBokeh v2: dLPIPS = 0,0006   RealDOF: dLPIPS = 0,0008
```

Menor diferença que o projeto já tratou como real: **ΔLPIPS ≈ 0,0119**.

**Duas ressalvas que importam para a publicação.** O piso acima é entre
checkpoints do mesmo treino, não entre sementes: não existe nenhuma repetição com
semente diferente nas 24 linhas, então o ruído entre execuções independentes é
**não determinado** e declarar o limiar sobre esse piso o subestima. E o
**LVCorr troca de sinal entre modelos no mesmo benchmark** (kfix +0,4599 contra
oficial −0,9644 no RealDOF, amplitude 1,42): é a métrica de controlabilidade, é
justamente a que os canais geométricos deveriam melhorar, e nesse estado não
sustenta limiar nenhum.

---

## 8. O que continua não determinado

| item | por quê |
|---|---|
| Ruído entre execuções com sementes diferentes | nenhuma repetição existe nos resultados curados |
| Se o EXIF do RealBokeh cobre o split `train` (rota c) | o benchmark de teste tem `focal_length`, mas não há script no repo que o gere, e o upstream não foi verificado |
| Resolução e faixa do `disparity` do `lf-bokeh-repro-blb` | só se sabe que são 500/500 não nulos, ~3,7 KB por amostra |
| Detalhes do `rotab_kfix.py` | o arquivo não está no repositório; a reconstrução é numérica, não por leitura de código |
| Se as splits train/test do RealBokeh têm os mesmos locais físicos | só há ids locais a cada split |
