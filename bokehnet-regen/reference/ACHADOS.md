# Registro de fatos medidos

Tudo aqui foi **medido**, não inferido. Cada linha traz a origem. Os agentes citam
daqui em vez de re-medir, e acrescentam linhas quando medem coisa nova.

Convenção: `[M]` medido, `[I]` inferido de algo medido, `[A]` assumido e não verificado.
Nunca promova `[A]` para `[M]` sem a medição.

## Dados publicados hoje (v0)

| fato | valor | origem |
|---|---|---|
| rota B, amostras | 11.635 | footer parquet [M] |
| rota B, coluna `k` | **1 valor distinto: 50,0** | footer parquet [M] |
| rota B, `s1` | 2,9e-6 a 0,974, mediana 0,0234 | footer parquet [M] |
| rota B, `calibration_ssim` | 100% NULA | footer parquet [M] |
| rota C, amostras | 2.932, **todas RealBokeh, zero LFDOF** | footer parquet [M] |
| rota C, `k` | 0 a 300, mediana **251,65** | footer parquet [M] |
| rota C, `k == 300` exato | **1.379 / 2.932 = 47,0%** | footer parquet [M] |
| rota C, `k == 0` | 51 = 1,7% | footer parquet [M] |
| rota C, K interior **e** ssim ≥ 0,6 | 1.483 / 2.932 = 50,6% | footer parquet [M] |
| rota C, `exif` | 100% NULA | footer parquet [M] |
| rota A, amostras | ~68.000, stems só UUID | [M] |
| união real B+C | 14.567 contra ~26.000 do paper | [M] |
| coluna `depth` | profundidade **métrica** min-max, não disparidade | teste de pixels, erro mediano 0,011% contra 3,008% [M] |
| `max(defocus_map)` | **65535 exato em todas**, k de 33 a 195 | [M] |

## Tabela `rota-b-kfix-eq3`

| coluna | min | mediana | max | distintos |
|---|---|---|---|---|
| `k_eq3` | 833,77 | **16.553,9** | 48.514 | 11.631 |
| `z_focus_m` | 0,2191 | 3,1101 | 10.000 | 11.564 |
| `z_min_m` | 0,1052 | 1,9767 | 26,594 | 11.632 |
| `z_max_m` | 0,3621 | 70,800 | 10.000 | 8.647 |
| `pixel_ratio` | 22,22 | 42,625 | 277,33 | 542 |
| `coc_p99_px` | 0,1002 | **4,665** | 112,0 | 11.635 |
| `k_antigo` | 50,0 constante | | | 1 |
| `max_coc_calibrado` | **10,510746** constante | | | 1 |

Reconstruído com erro relativo **0,000e+00** em 11.635/11.635 [M]:
`k_eq3 == f_mm² · z_foco_mm / (2·F·(z_foco_mm − f_mm)) · pixel_ratio`
`coc_px == k_eq3 · |1/z_mm − 1/z_foco_mm|`, erro < 1e-6 [M]
`max_coc_calibrado = 10,510746 = percentil 82,66 de coc_p99_px` [M]

Cobertura: 100% da rota B, **0% da rota C** [M]. `scripts/rotab_kfix.py` **não existe
no repositório** — rodou só no cluster, então mediana-vs-média e a regra do percentil
são reconstrução numérica [I], não código verificado.

## Degenerações que nenhum gate pega

| achado | contagem | nota |
|---|---|---|
| `z_max_m == 10.000,0` exato (teto do DepthPro) | 2.988 / 11.635 = **25,7%** [M] | em disparidade `1/10000 ≈ 0` e é **inofensivo**; o problema é dos canais geométricos, que usam `z` linear |
| `z_max_m >= 1.000 m` | 3.136 = 27,0% [M] | idem |
| cena útil em < 256 níveis de 65535 | 2.870 = **24,7%** [M] | mediana **23 níveis** no subgrupo `z_max >= 1000` — este *é* defeito |
| `coc_p99_px >= max_coc` (mapa satura) | 2.018 / 11.635 = **17,3%** [M] | com `max_coc = 10,5107` |
| `s1` implícito vs gravado | mediana 0,00308, p90 **0,07927**, máx 0,48821 [M] | são planos de foco **diferentes** |
| `fx` derivável, rota B | 11.635 / 11.635 = 100% [M] | via `focal_length_35` |
| `fx` derivável, rota C | 0 / 2.932 = 0% [M] | `exif` é NULA |

`fx_px = f_mm · pixel_ratio = max(W,H) · focal_length_35 / 36`, identidade verificada
em 900/900 linhas, 32 delas em retrato [M]. Logo **`max(H,W)` é o correto**, não a largura.

`512/min(W,H)`, o fator do crop de treino: 1024×574 → 0,892; 624×1024 → 0,821 [M].
**Nenhum código aplica** [M].

## RealBokeh — estrutura bruta (`timseizinger/RealBokeh_3MP`)

Listagem da API do HF, 31.853 arquivos [M]:

```
train/in/        3.960 imagens, TODAS _f22.JPG, uma por cena
train/gt/       20.554 imagens em 3.960 pastas de cena
train/metadata/  3.960 JSONs
test/in    220  ·  test/gt    1.257  ·  test/metadata    220
validation/in 220 · validation/gt 1.240
```

Nomes: `train/in/1000_f22.JPG`, `train/gt/1/1_f2.0.JPG`. O `parse_aperture` da rota C
(`name.split("_f")[-1]`) casa com **20.554 de 20.554 = 100%** [M]. Logo **as 1.028
cenas perdidas NÃO são falha de parser** [M].

Distribuição de imagens por cena em `train/gt/` [M]:
`2 → 704 · 3 → 620 · 5 → 2.346 · 7 → 9 · 9 → 34 · 21 → 247`  (soma 3.960 cenas, 20.554 imgs)

f-stop máximo dentro de `gt/`, por cena [M]:
`≤ f/2.8 → 119 (3,0%) · ≤ f/5.6 → 504 (12,7%) · ≤ f/8 → 874 (22,1%) · ≤ f/11 → 1.528 (38,6%)`, mediana **f/14**

`metadata/<id>.json` traz `focal_length`, `target_avs` (o f-number de cada nível) e
`focus_plane_distance` em metros com `focus_plane_uncertainty` [M]. Os três termos da
Eq. 3, anotados de fábrica. O `focus_plane_distance` também é **profundidade métrica
medida**, o que permite calibrar a escala do DepthPro por cena.

**Hipótese em aberto** [I]: `3.960 − 2.932 = 1.028 = 26%` são rejeições do QC
(`qc.severity == "red"`), provavelmente via `discriminate_depth_source`, porque o D6
entregava como "AIF" uma foto de f/5.6 a f/14 que genuinamente parece borrada.
Checagem: `grep -c rejected_qc route_c_log.jsonl`.

## Espelho `akcit-pixel/RealBokeh` — é o caso bom

20.495 linhas de treino, todas distintas, **3.959 cenas** [M].
Schema: `image_pre_deblur`, `image_blur`, `image_focus`, `file_name_base` [M].
`file_name_base` = `timseizinger_realbokeh_3mp_train_f_<cena>_level_<N>_aligned` [M].

Histograma de linhas por cena, contra o `gt/` bruto [M]:

| imgs/cena | espelho | gt/ bruto |
|---|---|---|
| 2 | 705 | 704 |
| 3 | 621 | 620 |
| 5 | 2.341 | 2.346 |
| **7** | **9** | **9** |
| **9** | **34** | **34** |
| 21 | 244 | 247 |
| 1, 4, 6, 12 | 2, 1, 1, 1 | não existem |

**Conclusão** [I, forte]: os buckets 7 e 9 batem exato. Se o `image_focus` viesse de
dentro de `gt/`, essas cenas apareceriam em 6 e 8 com contagem 9 e 34 — o bucket 6 tem
1 cena e o 8 não existe. Logo há **uma linha por imagem de `gt/`**.

### CONFIRMADO por comparação de bytes [M] — 2026-09-10

A inferência acima virou medição direta. Cena 1038, duas linhas (`level_3` e `level_4`):

```
image_focus do espelho          (2000,1500) JPEG  sha256 = 59d8e910ca69...  1927 KB
                                (byte-idêntico entre as duas linhas da cena)

train/in/1038_f22.JPG           (2000,1500)       sha256 = 59d8e910ca69...  ← MATCH
train/gt/1038/1038_f18.JPG                        sha256 = d872f982ba38...    não bate
train/gt/1038/1038_f2.0.JPG                       sha256 = c569d5d274df...    não bate
train/gt/1038/1038_f2.2.JPG                       sha256 = 03df408c8c6f...    não bate
train/gt/1038/1038_f5.0.JPG                       sha256 = 8cf5f893f02d...    não bate
train/gt/1038/1038_f7.1.JPG                       sha256 = 3e03e0013c58...    não bate
```

**`image_focus` É o `train/in/<id>_f22.JPG`, byte a byte.** Resolução 2000×1500
preservada, sem reencode. O espelho resolve D5 e D6 de uma vez, e a rota C pode
consumi-lo direto, fazendo join com `metadata/<id>.json` por cena para obter o
f-number de cada `level`.

Faltam 59 imagens e 1 cena inteira contra o `gt/` bruto (0,3%) — **não identificadas** [A].

Atenção: o nome traz o **nível ordinal**, não o f-number. Recuperar o f-number é join
com `metadata/<id>.json`, indexando `target_avs` por `level`.
`_aligned` sugere registro geométrico na construção do espelho — **não confirmado** [A].

## `atfortes/BokehDiffusion` — EXIF (rota B)

Leitura da coluna inteira, 1,11 MB baixados de `train.parquet` [M]:

```
linhas totais                                    : 15.305
passam o filtro da rota B (não pseudo_aif, f>0, F>0): 13.800
  com focal_length_35 presente                   : 13.800  (100,0%)
  sem focal_length_35                            :      0  (0,0%)
```

Sanidade do crop factor implícito [M]:

```
crop factor : min 1,00 · mediana 1,50 · p95 2,75 · max 9,75
sensor_mm   : min 3,69 · mediana 24,00 (APS-C) · max 36,00
crop factor EXATAMENTE 1,0 : 4.185  = 30,33%
crop factor > 8 (sensor < 4,5 mm) :    35
```

**Presença não é correção.** 30,33% com crop factor exatamente 1,0 é ou full-frame de
verdade, ou a assinatura de uma câmera que ecoa a focal no campo de 35 mm quando não
sabe. Trinta por cento de full-frame num dataset do Flickr é alto. A tabela
`make/model → sensor_width_mm` serve para **auditar esses 30%**, não para preencher
lacuna — não há lacuna. O `flickr_exif` tem make e model.

Errar aqui custa caro: crop 1,0 assumido onde o real é 5,6 subestima `pixel_ratio`,
logo o K, por **5,6×**.

## `akcit-pixel/LFDOF`

train **11.247**, test 725 [M]. Schema da DeblurNet, com AIF **real** renderizada da
light field — melhor que a AIF aproximada do RealBokeh. Fonte original pública:
`sweb.cityu.edu.hk/miullam/AIFNET/` → `LFDOF.zip`, ~11 GB [M]. Sem licença explícita
na página [M]. N desfocadas por AIF, então **split obrigatoriamente por cena**.

## Controlabilidade medida (LVCorr, maior é melhor)

| | LF-Bokeh | RealBokeh | RealDOF |
|---|---|---|---|
| fase 1, só sintético | **+0,9059** | +0,8609 | **+0,9247** |
| pesos oficiais do paper | +0,8868 | +0,8498 | +0,9644 |
| nosso, fase 2, a+b+c | **+0,4365** | +0,8257 | **+0,1324** |
| kfix (K da rota B pela Eq. 3) | **+0,8288** | +0,4832 | −0,4599 |
| só rota c (a+c) | +0,7296 | +0,8245 | −0,2061 |

Degradação **monótona no tempo de treino**, variante só-rota-c no RealDOF [M]:
`+0,2509 @10K · +0,0281 @20K · −0,1776 @30K · −0,2061 @60K`

Consertar só o K da rota B leva o LF-Bokeh de +0,4365 para +0,8288, colando no oficial
(+0,8868). **É a prova direta de que o rótulo de K era o gargalo** — não a arquitetura,
não os steps, não a aparência.

Fidelidade, em contraste: a variante só-rota-c bate os pesos oficiais em LPIPS nas três
mesas (0,1804 / 0,1134 / 0,1251 contra 0,2047 / 0,3454 / 0,2677) [M]. Os números de
fidelidade valem; os de controlabilidade não medem o método do paper.

## Como medir sem baixar imagem

Padrão usado em tudo acima: token em `~/.cache/huggingface/token`, um objeto
file-like seekable sobre HTTP `Range`, e `pyarrow.parquet` com projeção de coluna.
Lê **68 KB de um shard de 470 MB**. As 85 shards da RealBokeh inteiras seriam ~40 GB;
o histograma completo custou ~6 MB.

`datasets-server` está quebrado para estes repos: `501 Not Implemented` em
`/size` e `/splits`, `500` em `/rows` [M]. Não insistir nele.
`WebFetch` é anônimo e **não carrega o token** — dá 401 em repo privado. Usar a API
com header `Authorization: Bearer`.

---

## Etapa 1 — contrato implementado (2026-09-10)

| fato | valor | origem |
|---|---|---|
| testes do contrato passando | **36 / 36** | `unittest discover -s tests` [M] |
| nomes indefinidos em `src/` e `tests/` | **0** em 4 módulos | varredura AST [M] |
| `except:` / `except Exception: pass` em código de `src/` | **0** | grep [M] |
| constantes físicas em `src/` | 3, todas documentadas | grep [M] |
| `k_official(16553.9)` | **16,5539** | execução [M], bate com a âncora da kfix |

Limiares que entraram **assumidos**, a calibrar pelo histograma do piloto:

| constante | valor | risco |
|---|---|---|
| `FOCUS_DEPTH_MIN_M` | 0.05 | baixo [A] |
| `FOCUS_DEPTH_MAX_M` | 1000.0 | **médio** [A] — `z_focus_m` medido vai a 10.000, o topo legítimo é desconhecido |
| `MIN_DEPTH_RANGE_RATIO` | 1.02 | baixo [A] — permissivo de propósito |

Correção a um erro meu no plano: o gate "rejeitar `z_max == 10000`" estava **errado**.
Os 25,7% medidos são problema dos canais geométricos, que usam `z` linear; em
disparidade `1/10000 ≈ 0` é inofensivo. O gate certo é sobre `z_focus` no teto e sobre
faixa útil degenerada. Corrigido no código e no plano.

---

## Renderer verificado no cluster — 2026-09-10, job 32212 (h100n3)

Os três testes do `renderer-verifier` contra o **BokehMe real**, com GPU. É a medição
que autoriza `is_final_label_renderer` — antes disso era declaração.

**Proveniência do renderer** [M]:

```
commit BokehMe        8b3ed556dc146207c9bb39fccd078a113f8d1bd8
demo.pipeline sha256  b1274e2bed088d82653b...   (função extraída por AST)
scatter.py sha256     cc56904e9cede5b9bcc8...   (já com o patch do cupy.int)
arnet.pth sha256      6c7dba6f32e3...
iunet.pth sha256      6df927797a16...
saída usada           bokeh_pred (híbrida clássica+neural)
```

**Teste 1 — disco, não gaussiana** [M]: `edge_width_ratio = 0.143`
(disco < 0,5; gaussiana ≈ 1,43). **PASSOU.** O scatter clássico produz disco de
verdade — o oposto do gaussiano de 16 camadas que o pipeline antigo usava.

**Teste 2 — raio contra `K·|Δdisp|`** [M], ajuste `medido = slope·esperado + offset`:

| saída | slope | offset px | resíduo | leitura |
|---|---|---|---|---|
| `bokeh_classical` | **0,9619** | +1,026 | **0,0000 px (0,00%)** | reta EXATA |
| `bokeh_pred` (usada) | **0,9873** | +0,792 | 0,4168 px (1,09%) | linear; desvio vem do peso do `error_map`, que varia com K |
| `bokeh_neural` | 0,0994 | +128,2 | 1,27 px (3,30%) | sem sentido isolado — refina o clássico, não substitui |

**PASSOU** em linearidade e em escala.

Três coisas que isso ensina:

1. **O renderer é exatamente linear em K.** O clássico dá resíduo **0,0000 px** sobre
   raios de 3,2 a 38,4 px. O contrato `CoC = K·|Δdisp|` vale.
2. **`k_effective_factor = 0,9873`** — o raio renderizado é 1,3% menor que o K nominal
   prediz (3,8% no clássico puro). É pequeno, é sistemático, e **fica gravado**. Para a
   rota C não importa, porque K é ajustado por SSIM contra o alvo e o ajuste absorve a
   escala do renderer; para a rota B, cujo K é analítico pela Eq. 3, é um desvio de ~1%
   entre as rotas. Registrar, não corrigir sem medir de novo com cena real.
3. **`bokeh_neural` sozinho não é renderer.** Num ponto de luz sobre fundo preto ele
   devolve ~128 px independente de K, porque está fora da distribuição de treino.
   Confirma que `bokeh_pred` é a saída certa.

**Teste 3 — `highlight`** [M]: a função `pipeline` **não lê** `args.highlight`. O
realce acontece no corpo do `demo.py`, ANTES da chamada, alterando a imagem de entrada.
No nosso caminho a flag é **inerte** — decisão registrada, não esquecimento.

### Ambiente que o renderer exige (medido a duras penas)

| item | valor | por quê |
|---|---|---|
| `cupy-cuda12x` | **12.3.0** | 14.x é compilado contra numpy 2 e quebra com o 1.26.4 do container; 13.x removeu `cupy.cuda.compile_with_cache`, que o `scatter.py` usa |
| `fastrlock` | 0.8.3 | dependência do cupy, e `--no-deps` não a traz |
| `--no-deps` | obrigatório | sem ele o cupy arrasta numpy 2.2.6 para o `.pydeps` e sombreia o do container |
| patch `cupy.int` → `int` | 3 ocorrências | alias do builtin, removido no cupy 12; BokehMe é de 2022 |
| `torch.load(weights_only=False)` | obrigatório | torch ≥ 2.6 mudou o default; os checkpoints carregam objetos numpy |
| `cv2` | **não importar** | o `~/.local` tem opencv completo, que puxa libGL exigindo GLIBC 2.38 e o container tem 2.35 |

### Descoberta lateral [M]

`classical_renderer/scatter_ex.py` traz `ModuleRenderScatterEX` com `poly_sides` —
**abertura poligonal**. O paper afirma (§3.3) que "public implementations typically
omit this functionality". Não é PSF raster arbitrária como a Eq. 6 pede, mas é mais do
que o paper dá a entender, e é ponto de partida para o LoRA de formato de abertura.

---

## Adaptador de fonte da RealBokeh — 2026-09-10

Medido ao escrever `src/sources/realbokeh.py`. Join de **20.495** nomes do espelho
`akcit-pixel/RealBokeh` (coluna `file_name_base`, split `train`) contra os **4.400**
JSONs de `metadata/` do `timseizinger/RealBokeh_3MP` (público, baixados inteiros:
3.960 train + 220 test + 220 validation, 0 erros).

### `level` é 1-BASED [M] — três medições independentes

| discriminador | resultado |
|---|---|
| cenas com níveis exatamente `1..n`, contíguos, sem repetição | **3.959 / 3.959** |
| cenas com `level_0` | **0** |
| pares com `level == len(target_avs)` — `IndexError` sob 0-based | **3.949** |
| pares com `level > len(target_avs)` ou `level < 1` | **0** de 20.495 |

Logo o f-number do nível é `target_avs[level - 1]`. Índice fora da lista **rejeita**;
clamp gravaria a abertura errada em silêncio.

### `target_avs` é confiável em ordem [M]

Sobre os 4.400 JSONs: `len(target_avs) == len(target_images)` em **4.400/4.400**;
`target_avs` em ordem **crescente** em 4.400/4.400; e
`target_avs[i] == f-number no nome de target_images[i]` em **23.051 / 23.051** pares.
O adaptador confere a última em runtime, par a par.

### Campos do `metadata/<cena>.json` [M] — 4.400 JSONs

Presença **100%** de `id`, `source_image`, `source_av`, `target_images`, `target_avs`,
`focal_length`, `ISO`, `EV`, `focus_plane_distance`, `focus_plane_uncertainty`.

```
source_av                 : 22,0 em 4.400 / 4.400  (a AIF é f/22 sempre)
focal_length              : 28 a 70 mm
focus_plane_distance      : 0,355 a 368,63 m
focus_plane_uncertainty   : 0,0 a 315,895 m   — ZERO em 8 cenas
target_avs                : 2,0 a 20,0 (21 valores distintos), 23.051 no total
```

`focus_plane_uncertainty == 0` em 8 cenas é o motivo de o validador desse campo aceitar
zero enquanto `focal_length` e `focus_plane_distance` exigem positivo: incerteza nula é
afirmação legítima, focal nula é impossibilidade física.

### CORREÇÃO ao registro do espelho: o sufixo NÃO é sempre `aligned` [M]

A linha "`file_name_base` = `..._level_<N>_aligned`" está **incompleta**. Sobre as
20.495 linhas há **32 sufixos distintos**, e eles são uma **anotação de alinhamento**:

```
aligned          20.074   (97,94%)
misaligned          183   ( 0,89%)
shift_<X.Y>px       238   ( 1,16%)   30 valores distintos, de 2,0 px a 4,9 px
```

São **421 linhas (2,05%)** em que a origem declara que o registro geométrico entre
`image_focus` e `image_blur` não fechou, espalhadas por **228 cenas**. É o único sinal
de qualidade de par que a origem publica; o adaptador o carrega em campo
(`alignment`, `alignment_shift_px_at_mirror_hw`) em vez de descartar no parser.
O deslocamento é em pixels; **[A]** que tenha sido medido nos 2000×1500 do espelho.
`misaligned` **não** vira 0,0 px — a origem não diz quanto.

### As 59 imagens que faltavam, identificadas [M]

Era `[A]` "não identificadas". São **59 imagens em 11 cenas**, e a "1 cena inteira" é
uma das 11, não uma décima segunda:

```
918: faltam 20 de 21 (níveis 2..21)   2255: faltam  2 de  2  <- cena inteira
938: faltam 12 de 21 (10..21)           91: faltam  2 de  5
939: faltam  9 de 21 (13..21)          916: faltam  2 de  5
913: faltam  4 de  5 ( 2.. 5)          915: falta   1 de  5
 92: faltam  3 de  9 ( 7.. 9)          919: falta   1 de  3
925: faltam  3 de  5 ( 3.. 5)
```

Os níveis ausentes são **sempre a cauda**, em 11/11 — nunca buraco no meio. É o que
mantém a contiguidade `1..k` e o que torna a leitura 1-based verificável.
`2255` é a única cena com metadata e zero linhas no espelho; **0** cenas do espelho
estão sem metadata. Os 43 pares que sobrevivem nessas cenas são válidos — não são
rejeição.

### Enumeração completa [M]

```
entrada                  : 20.495 linhas do espelho
pares enumerados         : 20.495   (100,0%)
rejeitados               :      0
cenas                    :  3.959
níveis/cena              : 1->2 · 2->705 · 3->621 · 4->1 · 5->2.341 · 6->1 · 7->9 · 9->34 · 12->1 · 21->244
alinhamento              : aligned 20.074 · shift 238 · misaligned 183
split de origem          : train 20.495  (o espelho publica SÓ train)
```

O histograma de níveis/cena bate exato com o já registrado acima.

**Consequência para o split**: alimentar `split_from_source` só com os pares do espelho
dá `val_fraction == 0,0` — as 220 cenas de `test` e as 220 de `validation` da
RealBokeh_3MP **não têm linha no espelho**. Ou a validação sai de `build_scene_split`,
ou os splits `test`/`validation` entram por outra fonte.

### Slugs de rejeição que faltam em `control.contract.REJECTION_REASONS`

Seis, todos exercitados contra o dado real (corrompendo cópias do metadata). Vivem em
`sources.realbokeh.PENDING_REJECTION_REASONS` até serem registrados:
`source_name_unparseable`, `source_metadata_missing`, `source_level_out_of_range`,
`source_f_number_invalid`, `source_metadata_field_invalid`, `source_duplicate_sample`.

---

## [M] O espelho publica os TRÊS splits — e a numeração de cena reinicia em cada um

Medido em 2026-09-10 lendo a coluna `file_name_base` dos 96 shards parquet do snapshot
`fb183bfd7…` em `hf-cache/hub/datasets--akcit-pixel--RealBokeh`:

| split | shards | linhas | cenas |
|---|---|---|---|
| `train` | 85 | 20.495 | 3.959 |
| `test` | 6 | 1.257 | 220 |
| `validation` | 5 | 1.238 | 220 |
| **total** | **96** | **22.990** | **4.399** |

**Isto corrige o `[M]` anterior** que dizia "o espelho publica só `train` (20.495 de
20.495)". Aquela medição veio de um dump de coluna que cobria apenas os 85 shards de
`train` — não estava errada sobre o que mediu, estava incompleta sobre o que existe.

### A numeração de cena reinicia por split

Interseção dos números de cena, medida:

```
train ∩ test       : 220
train ∩ validation : 220
test  ∩ validation : 220
nomes idênticos entre test e validation: 0
```

As 220 cenas de `test` reusam os números `1..220`, que `train` também usa. São cenas
**físicas diferentes**: `timseizinger_realbokeh_3mp_test_f_1_level_1_aligned` e
`..._train_f_1_level_1_...` são fotos distintas com o mesmo número.

**Consequência**, e é dupla:

1. **Split furado.** Um `scene_id` cru poria a "cena 1" nos três lados ao mesmo tempo, e
   `scene_source_splits` levantaria `ValueError` — o barulhento. Pior seria o silencioso:
   `load_scene_metadata` lendo um `metadata/` só e chaveando pelo número entregaria à
   cena de `test` a distância de foco da cena de `train`, com JSON válido e completo.
2. **`sample_id` colidindo.** `c_realbokeh_1_l1` sairia igual para os três pares, e o
   gate `source_duplicate_sample` descartaria dois deles — **2.495 amostras boas** perdidas
   com motivo registrado e conclusão errada.

Conserto: `scene_key(split, numero)` é a única definição da chave (`train_1`, `test_1`,
`validation_1`), `ParsedName` guarda `scene_number` cru para achar o arquivo, e
`load_scene_metadata` lê `<raw_root>/<split>/metadata/` dos três splits.

### Efeito no split do release

Com os três splits, `split_from_source` funciona como a origem pretendeu e **não é mais
preciso sortear validação**. `test` e `validation` caem ambos em `val` (o `SceneSplit` é
binário) — os dois ficam fora do treino, que é o que importa; a distinção entre eles se
perde e está registrada aqui.

## DeblurNet — o recorte por `long_side` [M] — 2026-09-10

Medido reimplementando a aritmética de `Inference_deblurNet.py:13-49` (idêntica a
`bokehnet-preprocessing/src/model_runtime/deblurnet.py:32-66`) e propagando o
round-trip `resize → crop → cv2.resize de volta ao tamanho original` (:215-218).
Reprodutível por `model_runtime.deblurnet.plan_processing`; travado em
`tests/test_deblurnet.py::Geometria`.

**`long_side = 0` — o valor da rota B (`route_b.py:104-109`) e o default oficial
(`Inference_deblurNet.py:57`) — NÃO recorta.** O ramo é `ceil` para múltiplo de 16
(`Inference_deblurNet.py:43-49`): sobe até +8 px por eixo, e o round-trip de volta ao
tamanho original é uma identidade geométrica exata. Em 4032×3024, 5184×3456 e 1920×1080
nem redimensiona (já são múltiplos de 16).

**`long_side > 0` RECORTA o lado curto**, e o recorte é `floor` para múltiplo de 16, não
`ceil`. O lado longo nunca é recortado quando `long_side` é múltiplo de 16. Deslocamento
máximo do par AIF↔bokeh, em px da resolução ORIGINAL:

| resolução | `long_side` | processada | FOV perdida | deslocamento máx. |
|---|---|---|---|---|
| 4032×3024 (4:3) | 512 | 512×384 | — | 0,00 px |
| 1920×1080 (16:9) | 512 | 512×288 | — | 0,00 px |
| 1023×682 (3:2) | 512 | 512×336 | 1,47% em y | 6,09 px |
| 1500×1000 (3:2) | 512 | 512×336 | 1,47% em y | 8,93 px |
| 1600×1067 (3:2) | 512 | 512×336 | 1,47% em y | 9,53 px |
| 2048×1365 (3:2) | 512 | 512×336 | 1,47% em y | 12,19 px |
| 3000×2000 (3:2) | 512 | 512×336 | 1,47% em y | 17,86 px |
| **5184×3456 (3:2)** | 512 | 512×336 | 1,47% em y | **30,86 px** |
| 2048×1365 (3:2) | 768 | 768×496 | 2,94% em y | 22,02 px |
| 3000×2000 (3:2) | 1024 | 1024×672 | 1,47% em y | 14,88 px |

O gate do pipeline antigo era `--max-pair-shift-px 6.0` (`route_b.py:411-422`): 30,86 px
é **5,1× o limiar**. O deslocamento cresce linearmente com a resolução original, e é
**zero em 4:3 e 16:9** — ou seja, correlacionado com a proporção da foto, que é o pior
tipo de erro sistemático (o modelo aprende a proporção, não a física). É a mesma
assinatura do defeito B3, que erra K por `W/H` só em retrato.

Consequências práticas, e por que `long_side` não é só custo/qualidade:

1. o recorte muda o campo de visão, logo `max(H,W)` da imagem processada não descreve
   mais a mesma cena que `focal_length_35` descreve — é o `[A]` A6 disparando por
   construção nossa, não por culpa da fonte;
2. profundidade e máscara são medidas na AIF e o alvo é a bokeh: com deslocamento, os
   três vivem em geometrias diferentes (`[A]` A13, que segue **não medido** para a
   deformação que a rede em si introduz — isto aqui mede só o recorte determinístico).

Decisão implementada: `ResizePolicy.NO_CROP_MULTIPLE_OF_16` é o default e nunca recorta
(deslocamento 0,00 px por construção, para qualquer `long_side`); a política que
reproduz o recorte oficial exige `acknowledge_fov_crop=True` e grava `crop_box`,
`fov_retained`, `processed_hw` e `max_registration_shift_px` na proveniência de cada
amostra.

---

## [M] Custo do refinamento da região em foco — 2026-09-10

Medido em CPU (numpy 2.2.6, máquina local, mediana de 3 repetições) sobre imagem
sintética de 1500x2000, que é a resolução do espelho `akcit-pixel/RealBokeh`:

| operação | 1500x2000 | 384x512 |
|---|---|---|
| `qc.focus_region.detail_retention` | **291 ms** | **22 ms** |
| `renderer.calibration.resize_area_for_photo` (uma foto) | 185 ms | — |
| `routes.route_c.refine_focus_region` ponta a ponta | **391 ms** | **433 ms** |

**A resolução de trabalho não compensa nesta resolução, e a hipótese de que compensaria
está refutada.** A retenção precisa das DUAS fotos reduzidas (185 ms cada = 370 ms) para
economizar 269 ms de retenção. Por isso `RouteCConfig.focus_retention_long_side` tem
default `None` (resolução cheia) em vez do 512 do sweep da Eq. 5 — a diferença é que o
sweep reduz uma vez e reusa a redução em 14 a 18 renderizações, e a retenção é calculada
uma vez só.

Custo total da rota C por amostra a 1500x2000, com dublê de renderer barato em CPU:
**~4,2 s por amostra**, dos quais ~0,39 s são o refinamento (job de piloto real usa GPU e
BokehMe, então esta fração cai). O refinamento **não** é o gargalo.

`focus_retention_h`/`focus_retention_w` gravam a grade usada em cada amostra: a janela de
33 px do módulo cobre 1,7% do lado longo a 1500x2000 e 6,4% a 512x683, e são escalas
físicas diferentes.

## [M] Dois gates desfaziam o refinamento pela porta de trás — 2026-09-10

Achados ao plugar `qc.focus_region` na rota C, e corrigidos com teste. Os dois tinham o
mesmo mecanismo: `GateResult.passed` reprova valor não-finito **antes** de olhar o
limiar, então um gate que não conseguiu medir reprovava a amostra com um slug que
afirmava mérito.

| gate | quando | efeito |
|---|---|---|
| `mask_iou_aif_bokeh` | BiRefNet vazio nas DUAS imagens → união 0 → IoU NaN | rejeitava com `gate_mask_iou` exatamente as amostras de máscara vazia — **os mesmos 20,6% do piloto**, com outro slug |
| `focus_mask_sharpness_ratio` | região pontilhada desaparece após 3 passos de erosão → NaN | rejeitava com `gate_focus_mask_not_sharpest` regiões de retenção legítimas |

Enquanto a máscara vazia rejeitava a amostra em `focus_disparity_from_mask`, antes dos
gates, o primeiro ramo era **inalcançável** — o defeito estava latente e só apareceu
quando a amostra passou a atravessar. Agora os dois devolvem `applicable=False`: a
grandeza não existe, o metadado diz que não foi medida, e a amostra não é julgada por
ela. O sinal não se perde — está em `focus_initial_mask_was_empty` e em
`focus_agreement == 0`.

## [M] O dublê `_DISC` não serve para testar o plano de foco — 2026-09-10

`tests/test_renderer._DISC` borra a imagem inteira com um raio único, `median(coc)`. É o
dublê certo para medir linearidade em K e o errado para a rota C depois do refinamento:
numa cena uniformemente borrada **não existe plano de foco**, a retenção é igual em todo
lugar e a região devolvida é o topo de um empate numérico. Medido: com `_DISC`, a amostra
de teste saía `focus_source=retention_only` com retenção de 0,21 e região fora da
máscara, e o `focus_disparity` deixava de ser 1/3 m.

`tests/test_route_c._LAYERED` compõe 7 camadas com interpolação linear em `|K·Δdisp|`.
Com ele, a amostra cuja máscara casa com o objeto a 3 m sai `focus_source=birefnet`,
acordo 1,000, e o sweep recupera K = 18,011 com SSIM 0,9999.

## [M] O peso da nossa DeblurNet foi treinado cond-only e rodado main+cond — 2026-09-16

A proveniência fecha em quatro elos, todos conferidos:

| elo | valor |
|---|---|
| peso que a rota B usou | `outputs/deblur_docker_4gpu/deblur/deblur.safetensors` |
| sha256 | `a1ed05e48f7a3469cf825eda…` (`run_route_b_h100n1.sh:59,68`) |
| sidecar irmão `deblur.json` | `adapter_variant = 'cond-only'`, `text_adapter = None` |
| config do treino | `train_deblur_docker_4gpu.yaml:100` → `lora_on_main: false` |
| como a rota B o invocou | `main_adapter="deblurring"` (`deblurnet.py:942`, spec `:214`) |

O `deblur_best_step15500.json` traz o mesmo `adapter_variant = 'cond-only'` — a variante é
do **treino**, não do checkpoint, então vale para os dois passos.

A spec `OURS_MAIN_COND` (`deblurnet.py:214`) declara `lora_mode="main_cond"` e cita como
`evidence` o `HANDOFF_PROJECT_HISTORY.md`, que descreve um treino de **outro**
repositório. A única trava efetiva era o nome do arquivo, e `deblur.safetensors` é
idêntico nas duas famílias; a trava por sha256 existe no código e não foi acionada.

Mecanismo: `flux.py:758` monta `adapters = [main_adapter]*2 + c_adapters` e `specify_lora`
(`flux.py:146-162`) põe `scaling=1` nesse ramo. Índice 0 é texto, 1 é principal. A nossa
LoRA entrou em escala plena no ramo principal **e** no de texto, incluindo o `x_embedder`
do latente ruidoso — ramos que o treino nunca tocou. O peso oficial roda
`main_adapter=None` (`Inference_deblurNet.py:103-111`), que é o correto para ele: o
descasamento é **assimétrico** e só a nossa AIF o sofre.

**`main_adapter="deblurring"` em 11.563/11.563 não exonera o adapter.** Prova que o
roteamento foi o MESMO nos dois, não que fosse CORRETO nos dois.

Controle que fecha: o mesmo arquivo, avaliado com `main_adapter=None`
(`RESULTADOS.md:6-8`), dá RealDOF LPIPS 0,2291 contra 0,2397 do oficial — **ganha**. O
peso está são; a invocação não estava.

Proveniência dos 11.563 pares, verificada: `/tmp/jd_flat.py:19-20` leu
`b_release/generated/*_aif.jpg` contra `b_oficial_s*_de3/generated/*_aif.jpg` — os dois do
pipeline NOVO. Não é o lote antigo de 11.635 (a diferença de 72 era coincidência).

Fiel ao paper e descartado como causa: rank 128, batch 1, accum 8, 4 GPUs, batch efetivo
32, 60K passos (`paper.txt:513-515`). `--sem-identidade` só suprime a linha `Input` da
tabela de avaliação (`rodar_tabela2.py:175,213`) — zero efeito no treino.

## [M] `val_loss` da nossa DeblurNet piora depois do passo 15.500 — 2026-09-16

De `outputs/deblur_docker_4gpu/deblur/metrics.jsonl`: mínima **0,270269 no passo 15.500**,
subindo monotonamente a **0,281548 no 60.000** (+4,17%). O `best.pt` nunca foi superado em
44.500 passos.

Coerente com o alvo de treino ter menos diversidade que o paper supõe: o filtro top-3000
ranqueia **linhas**, e no RealBokeh cada cena tem ~5 aberturas com a mesma `image_focus`
⇒ **~580 alvos distintos em vez de 3.000** (`data.py:570-574`). A nitidez também é medida
num proxy 256² (`data.py:509`) e não na imagem nativa (`paper.txt:989-990`).

Secundário, não causa principal: `RESULTADOS.md:79-86` mede o 60k ganhando do 15.5k em
LPIPS e DISTS nas duas mesas.

## [M] `meta/` não mede progresso de run; a união de `sample_id` mede — 2026-09-16

Nas fatias da rota B oficial, `meta` e `mask` estavam em 4.061 enquanto `generated/*_aif.jpg`
em 2.917 e `depth/` em 2.799 — os artefatos não são escritos no mesmo instante.

Os `meta` sem AIF param todos em 13/09 19:41, enquanto os AIF seguem até 16/09 22:21: os
JPEGs daquela fase foram para `b_oficial_feito/b_oficial_release/generated/`, o lote já
mesclado. Nada se perdeu — localizado por busca recursiva do `sample_id`.

Cobertura real, sem dupla contagem: união de **12.005** `sample_id`, **12.005** com AIF em
algum lugar, **0** órfãos, sobreposição pub/fatias de 3.617 ⇒ **87,0% de 13.800**.

Lição: contar uma pasta só superestima (se ela é escrita cedo) ou subestima (se os
arquivos foram mesclados para outro lugar). Progresso se mede pela **união de
`sample_id`**, conferindo que cada um tem os artefatos em algum lugar.

## [M] Com o roteamento CERTO, a nossa DeblurNet AINDA é menos nítida — 2026-09-16

Controle sem GPU, sobre os `por_imagem.json` que já existiam em
`retreinar-deblur/saidas/tab2_*/<mesa>/modelo/`. Todas as linhas foram avaliadas com
`main_adapter=None` (o roteamento CORRETO para um peso cond-only), e as duas foram medidas
pelas MESMAS variantes de métrica, então a comparação entre elas é válida — a ressalva de
`RESULTADOS.md:121-125` é contra comparar com os números *publicados*, não entre linhas
nossas.

Nossa 60k (`a1ed05e4…`, cond-only) contra a oficial, razão das medianas por imagem:

| métrica | DPDD (75 pares) | RealDOF (50 pares) | sentido |
|---|---|---|---|
| CLIP-IQA | **0,899** (ganha 2,7%) | **0,888** (6,0%) | maior=melhor |
| MANIQA | 0,966 (36,0%) | 0,936 (26,0%) | maior=melhor |
| MUSIQ | **0,928** (2,7%) | **0,848** (4,0%) | maior=melhor |
| LPIPS | 1,119 (24,0%) | 0,931 (**84,0%**) | menor=melhor |
| DISTS | 1,246 (13,3%) | 0,898 (**76,0%**) | menor=melhor |

**Isto corrige uma afirmação minha.** Eu havia dito que "o peso está são; a invocação não
estava", apoiada no LPIPS do RealDOF (0,2291 contra 0,2397). O LPIPS realmente ganha — em
84% das imagens —, mas as métricas **sem referência** dizem que a saída é mais macia nas
DUAS mesas, com o roteamento certo: MUSIQ 0,848 e 0,928, CLIP-IQA 0,888 e 0,899, e a nossa
ganha em só 2,7–6,0% das imagens.

As duas coisas são compatíveis e é exatamente a assinatura de regressão à média: **mais
perto do gabarito em distância perceptual, e menos nítida em termos absolutos.**

Consequência para o diagnóstico: o defeito de roteamento é **real e provado** pela
proveniência (peso `cond-only` rodado com `main_adapter="deblurring"`), mas **não explica
tudo**. A razão de Laplaciano de 0,467 medida na rota B é o produto de dois fatores — o
dano do roteamento e uma maciez intrínseca do modelo — e nenhuma das medições feitas até
agora os separa.

Isso devolve à mesa a divergência dos **dados de treino** como candidata viva: ~580 alvos
distintos em vez de 3.000, porque o filtro top-3000 ranqueia linhas e não cenas
(`data.py:570-574`). Alvo com menos diversidade, e `val_loss` que piora depois do passo
15.500, são coerentes com um modelo que aprendeu a produzir macio.

O experimento de §7 da `AUDITORIA_TREINO_DEBLURNET.md` continua sendo o que separa os dois
fatores, e agora tem um valor esperado: se a razão de Laplaciano subir de 0,467 para algo
em torno de 0,85–0,93 (a faixa que MUSIQ/CLIP-IQA indicam para a maciez intrínseca), o
roteamento responde pela diferença entre 0,467 e esse patamar, e o resto é treino.

## [M] A identidade do peso da DeblurNet é o sha256, não o nome do arquivo — 2026-09-17

O `pinned_sha256` de `DeblurVariant.OURS_COND_ONLY` (`src/model_runtime/deblurnet.py`) é
**caractere a caractere** o sha medido no cluster e registrado em
`reference/AUDITORIA_TREINO_DEBLURNET.md:35` — comparado por script, não a olho: o literal
está quebrado em duas linhas no fonte, e uma concatenação truncada faria **toda** seleção
da variante certa falhar no cluster com "peso errado". O arquivo **não está nesta
máquina**: não foi re-medido aqui, foi confrontado com a medição já registrada.

`[M]` **O achado A17 está com a atribuição trocada, e a correção abriu um defeito real.**
Medido no Hub com `HfApi.repo_info(files_metadata=True)` — sha256 do LFS, sem baixar
nada. **Não medido por mim**: recebido já medido, e usado aqui como dado.

| repo | arquivo | sha256 | tamanho |
|---|---|---|---|
| `juliadollis/genrefocus-deblurnet` | **`deblurNet.safetensors`** | `a1ed05e4…` | 1.854.533.784 |
| `juliadollis/genrefocus-deblurnet-15k` | `deblurNet.safetensors` | `a941d6b5…` | 1.854.533.784 |
| `juliadollis/genrefocus-deblurnet-paper-4gpu` | `deblur.safetensors` | `deea4988…` | 1.856.139.632 |

O `deea4988…` de 688 chaves é do **`paper-4gpu`** — o main+cond antigo, isto é, a
`OURS_MAIN_COND` — e não uma "cópia divergente" do `genrefocus-deblurnet`, como diz
`ROTA_B_AUDITORIA_2.md:541`. O peso publicado em `genrefocus-deblurnet` é **bit a bit** o
`a1ed05e4…` que gerou os lotes; o `repo_id` da spec está certo.

`[M]` **O defeito é de NOME, e é o inverso de "a cópia diverge":** lá o arquivo se chama
`deblurNet.safetensors` e a spec declara `weight_filename="deblur.safetensors"` (o nome da
cópia do cluster). Duas consequências: `resolve_weights(OURS_COND_ONLY)` sem `local_dir`
pede um nome que não existe no repo; e, pior, `deblurNet.safetensors` é o nome declarado
pela `OFFICIAL_COND_ONLY` — então apontar para o NOSSO peso baixado do NOSSO repo dava
*"cruzamento de variante: este é o peso de official_cond_only"*, com o sha certo, mandando
fazer exatamente a troca errada.

Consertado em `_identifica_o_peso`: o sha256 é calculado **antes** da checagem de nome, e
`pinned_sha256` que casa encerra a identificação — dois arquivos com o mesmo sha são o
mesmo arquivo, e um `mv` muda o nome. Sem pin, nada muda: o nome continua sendo a única
evidência. A proveniência passou a gravar o nome **real** em `deblur_weight_filename`, com
o nome canônico da spec em `deblur_weight_filename_declared`.

`[M]` A suíte, medida antes e depois: `python3 -m unittest discover -s tests` saía com
**778 testes, 4 falhas e 28 erros** — `tests/test_deblurnet.py` inteiro quebrava, porque
`deblur.safetensors` deixou de identificar variante — e passa com **794 testes, 7 pulados**
(pyarrow e rede, pré-existentes), 0 falhas.

`[M]` As travas novas foram conferidas por **mutação**, não só por execução: desligar
`_check_sha_identifies_variant` derruba 3 testes de `OSha256IdentificaAVariante`; trocar
`main_adapter=self._spec.main_adapter` por `ADAPTER_NAME` fixo no call site derruba 4,
entre eles o que roteia o MESMO arquivo pelas duas variantes nossas e o de AST; e voltar a
checagem de nome para ANTES do hash derruba 3, entre eles o do peso nosso baixado do Hub
com o nome da oficial.

---

## [M] Onde a dedup do mapa de profundidade se aplica — amostras por cena nas três rotas

A profundidade é função da **cena**, não da amostra: ela sai da AIF, e a dedup economiza
exatamente onde vários `sample_id` compartilham uma AIF. Medido antes de implementar,
sobre as funções de id do próprio repositório e sobre os releases já gerados.

| rota | amostras | cenas | amostras/cena | `depth/` hoje → com dedup | economia |
|---|---|---|---|---|---|
| **A** | 69.700 | 1.700 | **41,00** | 69.700 → 1.700 | **97,6%** |
| **B** | 13.800 | 13.800 | **1,00** | 13.800 → 13.800 | **0,0%** |
| **C** | 15.423 | 4.399 | **3,51** | 15.423 → 4.399 | **71,5%** |

Origem de cada linha:

* **Rota A** — `sample_id_for(scene_id, i) = f"{scene_id}_v{i:02d}"`
  (`routes/route_a.py:1019`) com `DEFAULT_SAMPLES_PER_IMAGE = 41`
  (`routes/route_a.py:220`): 41 variantes por cena, exatamente. Confirmado em pixel pela
  usuária no release `a_release`, em 120 cenas sorteadas com semente fixa: **120/120**
  com a profundidade **byte a byte idêntica** em todas as variantes, 0 divergentes,
  0 ausentes, 444 KB por arquivo — **31,7 GB contra 0,77 GB**.
* **Rota B** — `scene_id` é o `flickr_photo_id` e `sample_id` é `b_<scene_id>`
  (`scripts/run_route_b.py:164-165`), uma função **injetiva**, e
  `scripts/run_route_b.py:289` ainda rejeita id repetido com `source_duplicate_sample`.
  São **13.800 cenas para 13.800 amostras** (`ACHADOS.md:155`). A dedup não tem o que
  economizar aqui, e **não foi ligada** — `RouteBConfig.depth_layout` é `per_sample`.
* **Rota C** — `sample_id` é `c_<fonte>_<scene_id>_l<nível>`
  (`sources/realbokeh.py:608`, `sources/lfdof.py:617`), com o teto de 4 níveis por cena
  do paper (`sources/level_selection.py`). O release publicado
  `juliadollis/bokehnet-regen-rota-c` tem **15.423 amostras em 4.399 cenas**
  (`REGISTRO.md:1320`) = 3,51 por cena. Que a AIF é a mesma entre os níveis já estava
  medido em pixel: o `image_focus` é byte a byte idêntico entre os níveis de uma cena,
  em 90 linhas de 9 cenas de 3 row groups (`sources/lfdof.py:64-71`).

### [M] `realbokeh.scene_key` e `lfdof.scene_key` produzem a MESMA string

As duas são `f"{split}_{numero}"` (`sources/realbokeh.py:242`, `sources/lfdof.py:435`),
então a cena `train_1275` existe nas duas fontes e são cenas **físicas diferentes**. No
layout por amostra isso nunca fez dano, porque o `sample_id` carrega a fonte
(`c_realbokeh_…` contra `c_lfdof_…`); com `depth/<scene_id>.png` as duas escreveriam o
mesmo arquivo. O writer recusa alto quando uma cena aparece em duas
`source_dataset` (`dataio/writer.py:_grava_depth`), e
`test_dataio.test_a_mesma_cena_em_duas_fontes_e_erro` trava a checagem.

**Não é hipótese**: o release da rota C publicado é só RealBokeh (4.399 cenas = 3.959 +
220 + 220), então a colisão ainda não ocorreu — mas ocorreria no primeiro release que
misturasse as duas fontes, que é o que o LFDOF está previsto para ser.
