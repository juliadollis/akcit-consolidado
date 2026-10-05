# Contrato canônico — `metric_disparity_official_v1`

Uma página. É a única definição válida do sinal de controle. Geração, dataloader,
avaliação e inferência usam **esta** e nenhuma outra.

Autoridade, em ordem: (1) o paper, onde ele fala; (2) `Inference_bokehNet.py`, onde
o paper cala; (3) decisão nossa, **declarada como desvio**, onde os dois calam.

## A fórmula

```python
z            = depth_pro(aif)              # METROS, métrico, sem normalizar
disp         = 1.0 / z                     # 1/m
focus_disp   = median(disp[mask])          # calcular NA disparidade
K            = k_eq3 / 1000.0              # Eq. 3 é em mm; a inferência é em 1/m
max_coc      = 100.0                       # MAX_COC oficial, congelado por release

defocus      = clip(abs(K * (disp - focus_disp)) / max_coc, 0.0, 1.0)
```

```python
k_eq3 = (f_mm**2 * z_focus_mm) / (2 * F * (z_focus_mm - f_mm)) * pixel_ratio
pixel_ratio = max(H, W) / sensor_width_mm          # px/mm
```

## As cinco regras que não se negociam

0. **O paper escreve `D` como mapa de PROFUNDIDADE; nós operamos em DISPARIDADE.**
   §3.2 (paper.txt:313-315) diz "*D é o mapa de profundidade monocular estimado por
   [7]*", e a palavra *disparity* nunca aparece aplicada a `D` no artigo. Mas a
   Eq. 3 só fecha dimensionalmente com `|Δ(1/z)|`, e a inferência oficial resolve o
   silêncio: `Inference_bokehNet.py:94` faz `disp = 1.0/depth` e `:118` tira a
   mediana disso. **Operar em `1/z` é decisão por autoridade do código oficial, não
   leitura literal do paper.** Toda a régua abaixo depende dessa troca de espaço.

1. **`focus_disp` é mediana da disparidade, não `1 / median(z)`.** A mediana só é
   invariante sob transformação monótona no caso de contagem ímpar; `np.median`
   faz a média dos dois centrais quando é par, e a média de dois recíprocos não é
   o recíproco da média. Guardar `focus_depth_m = 1/focus_disp` para leitura
   humana é permitido; **o mapa nunca usa o valor de volta.**

2. **`max_coc` é global e congelado.** Nunca por imagem, nunca por rota, nunca por
   fonte. A rota B vai ocupar `[0, 0.05]` e a rota C `[0, 0.4]` — a diferença é
   física e é o que o modelo tem que aprender. Reescalar uma rota para "ocupar
   [0,1] como a outra" é o defeito D1 com granularidade mais grossa.

3. **Toda quantidade em pixel carrega a resolução em que foi medida.** `k_eq3` está
   em pixels da imagem gravada. Se o dataloader reescala e recorta, o fator
   `512/min(H,W)` — que varia por amostra — tem que ser aplicado, ou o mapa que o
   modelo recebe não é o mapa que foi gravado.

4. **Sem fallback numérico.** Faltou EXIF, faltou sensor, falhou o modelo: rejeita
   a amostra e registra o motivo. Nunca substitui por constante.

5. **Uma implementação só.** A fórmula vive em um módulo, importado por geração,
   dataloader, avaliação e inferência. Cópias divergem — foi assim que chegamos a
   quatro interpretações de K.

## Análise dimensional, para conferir qualquer mudança

```
f, z_focus em mm    f² · z_focus / (2F(z_focus − f))   →  mm²
pixel_ratio         max(H,W)_px / sensor_mm            →  px/mm
k_eq3               mm² · px/mm                        →  px·mm
                    k_eq3 · |Δ(1/mm)|                  →  px      ✓

k_value oficial     k_eq3 / 1000                       (disparidade em 1/m)
```

## Âncoras numéricas

Se uma mudança tirar os números daqui, algo quebrou.

| grandeza | valor esperado | origem |
|---|---|---|
| `k_value` mediano, rota B | **16,6** | mediana de `k_eq3` da tabela `rota-b-kfix-eq3` ÷ 1000 |
| `k_value` mediano, rota B | **20,1** | mediana calculada da EXIF, 318 amostras com distância de foco |
| `k_value` default da inferência oficial | **15,0** | `Inference_bokehNet.py:53` |
| `K` da Fig. 12 do paper | **{0, 5, 10, 15}** | paper, Fig. 12 |
| `k_value` da rota C | **3,6 a 36** | Eq. 3 sobre o `metadata/` da RealBokeh |
| CoC p99 mediano, rota B | **4,665 px** | tabela kfix, coluna `coc_p99_px` |
| `pixel_ratio`, rota B | 22,2 a 277,3, mediana **42,6** | tabela kfix |

## Contrato do renderer

BokehMe público, `demo.py`:

```python
defocus = K * (disp_norm - disp_focus) / defocus_scale
# e dentro do pipeline:
classical_renderer(image**gamma, defocus * defocus_scale)
```

Logo o **raio de borrão é `K · Δdisp` em pixels**, e `defocus_scale` só normaliza a
entrada da rede. Para alimentar disparidade normalizada em `[0,1]`:

```python
k_renderer = K * (disp_max - disp_min)     # K na convenção 1/m
```

O paper **não publica**: `gamma`, `defocus_scale`, `highlight` e seus limiares, qual
das três saídas usa (`bokeh_pred` híbrida, `bokeh_classical`, `bokeh_neural`), a
versão de `arnet.pth`/`iunet.pth`, `K_min`/`K_max` da Eq. 5, nem o limiar de SSIM.
Congelamos os nossos, gravamos commit e hashes por amostra, e declaramos.

### Verificado em GPU — job 32212, 2026-09-10

| teste | resultado |
|---|---|
| disco, não gaussiana | `edge_width_ratio = 0,143` (disco < 0,5) ✓ |
| linearidade em K | `bokeh_classical` com resíduo **0,0000 px** — reta exata ✓ |
| escala | `slope = 0,9873` para `bokeh_pred` ✓ |

**`k_effective_factor = 0,9873`**: o raio renderizado é 1,3% menor que o K nominal
prediz. Irrelevante para a rota C, onde o ajuste por SSIM absorve a escala do
renderer; é um desvio de ~1% entre rotas para a rota B, cujo K é analítico. Gravado,
não corrigido — corrigir exige medir com cena real, não com ponto sintético.

`bokeh_neural` isolado **não é renderer**: num ponto de luz sobre fundo preto devolve
~128 px independente de K. Ele refina o clássico, não o substitui. Use `bokeh_pred`.

## Resolução — o que o paper publica e o que não

O paper **não publica a resolução de treino**. O tiling da §3.5 é explicitamente de
**inferência**: *"a tiling strategy inspired by [5] **during inference**"*
(paper.txt:481-482). A §4.1 dá batch, acumulação e steps, e cala sobre resolução.

Logo estas três são **decisão nossa**, declaradas:

| decisão | onde | custo declarado |
|---|---|---|
| profundidade gravada com lado longo **768** | `dataio/encoding.py` | `quantization_coc_error_px` por amostra; 0,000378 px com K=50 |
| treino em crop **512**, K reescalado | `control.k_at_resolution` | fator por amostra, medidos 0,892 e 0,821 |
| calibração da Eq. 5 em lado longo **512** | `renderer/calibration.py` | K convertido nas duas direções; travado por teste |

Cuidado de ergonomia: `k_at_resolution` recebe **lado menor** (`dst_short_side`) e
`encode_depth` recebe **lado longo** (`long_side`). Convenções opostas na mesma
cadeia — sempre nomeie o argumento na chamada.

## Arquitetura e currículo — §4.1

O paper publica (paper.txt:512-516), e isto é especificação, não sugestão:

| item | valor |
|---|---|
| backbone | FLUX.1-dev |
| LoRA rank | DeblurNet **128**, BokehNet **64** |
| batch | 1 por GPU × acumulação 8 × 4 GPUs = **32 efetivo** |
| DeblurNet | 60K steps |
| BokehNet | **40K sintético + 60K real** |
| passos de denoise na inferência | 28 |

## Escopo das equações

| Eq. | O que é | Onde vale |
|---|---|---|
| 2 | `D_def = K·|D − D_focus|`, **sem normalizador** | todas as rotas |
| 3 | `K ≈ f²·D_focus/(2F(D_focus−f)) × pixel_ratio` | **só rota B / ITW** |
| 4 | `D_focus = median(D[M])` | rotas B e C |
| 5 | `K* = argmax SSIM(R(...), I_real)` | **só rota C**, com limiar de SSIM que o paper afirma usar e não publica |
| 6 | `R(I_aif, D; D_focus, K, s)` com shape kernel | **só** o LoRA de formato de abertura |

Na rota A o paper diz apenas *"randomly sample a focus plane and a target bokeh level
K"* (§3.2(a), paper.txt:326-328) e **cala sobre a distribuição**. Amostrar `K` da
distribuição empírica de B/C (`k_source = "sampled_from_bc"`) é **decisão nossa**, e
existe para manter o sintético na mesma escala física do real.

A extensão privada dos autores é **só a Eq. 6**. Eq. 5 e Fig. 3(a) usam o BokehMe
público `[43]`, e o supplement B.2 confirma: *"we then optimized the parameter K
using simulator [43]"*.
