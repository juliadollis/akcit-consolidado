# Auditoria 2 da rota B — equação por equação, contra o paper, com medida

Escopo: a rota B como ela está **agora**, no código deste checkout e no dado que está
sendo gerado no cluster. Nada foi editado. Tudo o que é afirmação carrega
`paper.txt:<linha>`, `<arquivo>:<linha>`, ou a medição que a sustenta.

Data: 2026-09-14. Suíte local: **778 testes coletados, 775 passaram, 3 pulados**,
202 subtests (`pytest -q`, interpretador `AKCIT/.venv/bin/python`, `PYTHONPATH=src:scripts:tests`).
Só de rota B + contrato + DeblurNet: 202 testes, todos passando.

Medições sobre dado real, **somente leitura**, sem tocar em job nenhum:

| corpus | onde | N medido |
|---|---|---|
| variante **oficial** (`official_cond_only`) | `dgx-H100-03:.../output/b_full_official` + `b_full_s1` | **3.788** |
| variante **nossa** (`ours_main_cond`), rodando agora | `dgx-H100-01:.../output/b_full_nossa_s{0..3}_de4` | **1.644** |
| **pareado**, mesma foto nas duas variantes | interseção por `sample_id` | **1.644** |

---

## 0. A resposta direta

> *"A rota B está sendo gerada 100% correta de acordo com o paper, equações corretas etc?"*

**As equações estão corretas. O dataset, não é "100% o do paper" — e não pode ser.**
A resposta honesta é **sim com ressalvas**, em três camadas que não devem ser
confundidas:

**1. A álgebra está certa, e isso é medido, não opinado.**
As Eq. 2, 3 e 4 estão implementadas corretamente, a análise dimensional fecha, e a
Eq. 3 reproduz a fórmula física do círculo de confusão com erro relativo **1,7 × 10⁻¹⁶**
(§1.6). Nas 5.432 amostras já geradas, todas as identidades internas batem
**bit a bit**: `k_value == k_eq3_mm/1000` com desvio máximo **0,0**; a Eq. 3 recalculada
dos termos gravados bate com o gravado com desvio relativo máximo **0,0**;
`pixel_ratio == max(H,W)/sensor_mm` com desvio **0,0**. O fator 1000 acontece
**exatamente uma vez**. Nenhum dos nove defeitos catalogados em `ROTA_B_AUDITORIA.md`
sobreviveu.

**2. Há cinco desvios do paper, todos declarados, quatro deles inevitáveis.**
Operar em disparidade e não em profundidade; `MAX_COC = 100`; qual leitura de `D_focus`
alimenta a Eq. 3; treinar em crop 512 com `k_at_resolution`; e não aplicar o refinamento
do §3.2(c). Os cinco estão em §6, com a evidência de cada um e o custo medido. **Três
deles são numericamente irrelevantes e agora isso é medido, não suposto.**

**3. A ressalva que importa não é de equação — é do checkpoint que está rodando.**
Na comparação **pareada, nas mesmas 1.644 fotos**, a AIF da nossa variante tem
**2,3× menos energia de alta frequência** que a da oficial (variância de Laplaciano
mediana 359 contra 1005), e é **menos nítida que a própria foto borrada de entrada em
77,1% das amostras** (contra 22,8% na oficial). Consequência medida no rótulo: o Depth
Pro sobre essa AIF devolve um plano de fundo **15× mais próximo** (`z_max` mediano
7,2 m contra 103,6 m), o vão de disparidade **cai pela metade**, e o CoC máximo por
amostra cai de 5,33 px para 2,43 px. **O `K` quase não muda** (razão mediana 0,999,
nenhuma amostra fora de ±20%) — mas o mapa `D_def = K·|D − D_focus|`, que é o que a
BokehNet realmente consome, muda muito.

Isso **não é erro de equação**. E não é a AIF "lavada" do defeito B1: o
`main_adapter="deblurring"` está provado na proveniência de 1.644/1.644 amostras, e o
benchmark da Tabela 2 rodado pela usuária (`REGISTRO.md:1388-1395`) mostra o nosso
checkpoint **melhor que o oficial no RealDOF** — LPIPS 0,2291 contra 0,2397 —, que é o
domínio de desfoque óptico real da rota B. Uma AIF lavada não melhora LPIPS contra
gabarito. A leitura que sobra é que o modelo troca energia de alta frequência (ruído e
artefato de JPEG do rendition Flickr) por estrutura limpa.

**Fica, portanto, uma conferência e uma decisão**, não um bloqueio: (i) abrir 20–30 pares
AIF × original das duas variantes e olhar (`[A]` A16, risco médio); (ii) escolher qual dos
dois lotes vai para treino, sabendo que eles produzem mapas de defocus fisicamente
diferentes apesar do mesmo K (`[A]` A19). Ver §5.4.

Em uma frase: **a régua está certa e é auditável até o bit; o que sobra é escolher qual
dos dois lotes vai para treino, e conferir 20–30 AIFs com o olho antes de fechar.**

Um quarto ponto, que não é ressalva mas corrige um registro: `paper.txt:513` **publica** o
rank da LoRA (*"DeblurNet employs LoRA rank r=128"*), e medimos **128 no nosso** checkpoint
e **64 no oficial**. No eixo do rank é o nosso que reproduz a especificação do paper, não
o peso publicado pelos autores (§3.3).

---

## 1. A Eq. 3, termo a termo

O texto, verbatim (`paper.txt:340`):

```
K ≈ f²·D_focus / ( 2·F·(D_focus − f) ) × pixel_ratio                       (3)
```

com o parágrafo que a governa em `paper.txt:342-352`.

Escopo: a Eq. 3 vale **só na rota B / ITW**. O §3.2(b) a introduz para o ITW
(`paper.txt:336-338`) e o §3.2(c) manda a rota C para a Eq. 5
(`paper.txt:369-372`). `CONTRATO.md:156-163` registra a mesma separação.

### 1.1 `f` e `F` — de onde vêm, e o que acontece quando faltam

O paper: *"where f is the focal length and F is the aperture f-number, **both directly
obtained from EXIF metadata**"* (`paper.txt:342-343`).

No código, o caminho exato é:

| termo | origem | linha |
|---|---|---|
| `f` | coluna `focal_length` do `atfortes/BokehDiffusion` | `scripts/run_route_b.py:137` |
| `F` | coluna `f_number` do mesmo dataset | `scripts/run_route_b.py:138` |
| `focal_length_35` | coluna `focal_length_35` | `scripts/run_route_b.py:142` |

**`[I]` registrado**: nós não lemos EXIF do arquivo de imagem — lemos as **colunas EXIF
que o dataset publica**. A confiança na extração é do construtor do dataset, não nossa.
Isso é o `[A]` A1 de `ROTA_B_AUDITORIA.md:948` visto por outro ângulo, e é honesto
marcar: se o `atfortes/BokehDiffusion` errou ao extrair a EXIF, nós herdamos o erro sem
nenhum sinal.

**Quando faltam, a amostra é rejeitada — nunca completada.** Sem exceção e sem default:

* `f` ausente/inválido → `exif_focal_length_missing` (`src/control/contract.py:345-346`);
* `F` ausente/inválido → `exif_f_number_missing` (`src/control/contract.py:395-396`);
* `focal_length_35` ausente → `sensor_width_unresolvable` (`src/control/contract.py:347-348`);
* valor não-finito ou ≤ 0 na coluna → `source_metadata_field_invalid`
  (`scripts/run_route_b.py:158-160`). O `math.isfinite` está lá por um motivo concreto:
  a string `"inf"` passa por `float()` e é maior que zero.

Isso é a regra 4 do `CONTRATO.md:52-53`, e é **decisão nossa** (`[A]` A8): o paper cala
sobre EXIF ausente. Escolhemos o lado conservador.

**Medido**: `focal_length_35` está presente em **13.800/13.800** das linhas que passam o
filtro (`ACHADOS.md:155-157`), então na prática o slug `sensor_width_unresolvable` não
dispara. Nas 3.788 amostras da oficial e nas 1.644 da nossa, **zero** rejeições por
qualquer um dos três slugs de EXIF. O único motivo de rejeição observado no dado real é
`focus_mask_empty` (§4.2).

Distribuições medidas (oficial, N=3.617 — min · p05 · **p50** · p95 · max):

```
focal_length_mm      4   ·  20  ·  55   · 105  · 200
f_number           1,4  · 1,8  ·  4,5  ·  10  ·  25
focal_length_35mm   21  ·  35  ·  82   · 157  · 249
```

### 1.2 `D_focus` — qual das duas leituras entra, e a outra está gravada ao lado

O problema é real e está corretamente identificado. A Eq. 4 (`paper.txt:352`) é
`D_focus = median(D[M])` sobre um mapa de **profundidade**. O contrato produz
`focus_disparity = median(1/z[M])` (`src/control/contract.py:315`), por autoridade do
código oficial (`Inference_bokehNet.py:118`, `CONTRATO.md:28-34`). E
`1/median(1/z) ≠ median(z)`: a mediana não comuta com a inversão quando a contagem é par,
porque `np.median` faz a **média** dos dois centrais e a média de dois recíprocos não é o
recíproco da média.

**A rota usa `1/focus_disparity`.** Está declarado em três lugares independentes:

* a Decisão 1, no docstring do módulo — `src/routes/route_b.py:51-77`;
* a chamada — `src/routes/route_b.py:952`:
  `focus_depth_m=focus_diag["focus_depth_m_from_disparity"]`;
* o metadado de **toda** amostra — `focus_depth_convention: "1/median(1/z[M])"`, com a
  nota que cita `paper.txt:352` como o desvio (`src/routes/route_b.py:738-744`).
  Confirmado em **3.617/3.617** da oficial, **171/171** da s1 e **1.644/1.644** da nossa.

**A outra leitura é gravada ao lado**, por amostra: `focus_depth_m_from_disparity`,
`focus_depth_m_median_z` e `focus_depth_ratio`
(`_focus_depth_diagnostics`, `src/routes/route_b.py:717-746`). As duas são calculadas por
fórmulas **diferentes** sobre o mesmo conjunto de pixels — não é um número repetido.

**Medido, e é o fim da discussão:**

```
focus_depth_ratio = 1/median(1/z[M])  ÷  median(z[M])
  nossa   (N=1.630) : min 0,999999882  mediana 0,999999998  max 1,000000122
                      desvio máximo de 1,0 = 1,2 × 10⁻⁷
  oficial (N=3.788) : min = p01 = p50 = p95 = max = 1,0
  |razão − 1| > 0,01 : 0 amostras     > 0,05 : 0     > 0,20 : 0
```

Por que coincidem: numa máscara do BiRefNet com ~10⁵ pixels de profundidade suave, ou a
contagem é ímpar — e aí a mediana comuta **exatamente** com `x ↦ 1/x`, que é monótona —
ou os dois valores centrais diferem por ruído de float. Reproduzido localmente: `n=99`
dá razão `1,000000` exata; `n=100` dá `0,999999`.

**E o efeito sobre K é menor ainda**, por álgebra: `dK/K = −[f/(D_focus − f)] · dD/D`.
Com as medianas reais (`f = 55 mm`, `D_focus = 2,714 m`) o fator é **0,0207** — um erro
de 20% em `D_focus` move K em **0,41%**. Verificado numericamente (tabela de
sensibilidade, §1.6) e **confirmado no dado pareado**: `focus_disparity` difere em mais
de 20% entre as duas variantes em **60%** das 1.644 fotos, e mesmo assim `k_value` difere
em mais de 5% em **2%** delas e em mais de 20% em **nenhuma**.

**Veredito**: a Decisão 1 está declarada, a leitura literal do paper está gravada ao
lado em toda amostra, e o desvio é de ordem 10⁻⁷ — abaixo da precisão de qualquer coisa
que dependa dele. Fechado.

Uma ressalva de leitura, para ninguém se enganar: como `focus_depth_ratio ≡ 1`, esse
diagnóstico **não é um teste de nada**. Ele mede a diferença entre duas convenções que se
provaram idênticas; ele não valida se `D_focus` está **certo**. O validador de `D_focus`
é outro, e está em §1.7.

### 1.3 `pixel_ratio` — é `max(H,W)/sensor_mm`, e o paper fecha isso

A legenda da Fig. 16 (`paper.txt:1185-1187`), que é a que vale:

> *"Pixel ratio distribution. [...] This ratio, defined as the **image's largest edge
> length divided by the physical sensor width** (px/mm)."*

O código: `src/control/contract.py:356-371`, `return max(height, width) / float(sensor_mm)`.
Verificado localmente: `pixel_ratio((4000,6000), 36) == pixel_ratio((6000,4000), 36) == 166,67`
— idêntico sob rotação, logo é `max`, não largura.

**Isso importa em 28,3% do dataset**: das 1.630 amostras da nossa variante, **461 são
retrato** (`H > W`). Usar a largura erraria K por `H/W` nelas — o defeito B3.

`sensor_mm` vem do crop factor, `36,0 / (focal_length_35 / focal_length)`
(`src/control/contract.py:330-353`). O `36,0` é `FULL_FRAME_WIDTH_MM`, e **como o paper
obteve a largura física do sensor ele não diz** — é o `[A]` A7.

**Medido**, e as identidades fecham bit a bit nas 5.432 amostras:

```
pixel_ratio == longest_edge_px / sensor_width_mm      desvio relativo máx. 0,0
sensor_width_mm == 36,0 / crop_factor                 desvio relativo máx. 0,0
crop_factor == focal_length_35mm / focal_length_mm    desvio relativo máx. 1,4e-16 (1 ulp)
```

Distribuição (oficial, N=3.617): `pixel_ratio` 22,22 · 28,42 · **42,625** · 76,88 · 277,3.
A âncora do `CONTRATO.md:82` é *"22,2 a 277,3, mediana 42,6"*. **Bate nos três números.**

### 1.4 O fator 1000 — onde acontece, e acontece uma vez só

A Eq. 3 vive em milímetro (porque `f` está em mm); a inferência oficial opera em
disparidade `1/m` (`Inference_bokehNet.py:94,118`; `CONTRATO.md:28-34`).

A conversão está em **um lugar**: `k_official()`, `src/control/contract.py:410-418`,
`return float(k_eq3_value) / MM_PER_M`. Varredura completa de `src/` e `scripts/`:
`k_official` é chamada **em dois lugares** — `k_from_exif` (`contract.py:437`) e o
validador analítico da rota C (`route_c.py:639`). A rota B **nunca** chama `k_eq3_mm`
direto; só `k_from_exif`, em `route_b.py:948` (rótulo) e `route_b.py:875` (validador).
Não existe uma segunda divisão por 1000 em lugar nenhum do código de produção.

Há uma segunda aparição de `MM_PER_M`, em `k_eq3_mm` (`contract.py:398`):
`focus_mm = focus_depth_m * MM_PER_M`. Ela é **necessária e de sentido oposto** — `f` está
em mm, logo `D_focus` também tem que estar, ou a fração `f²·D/(2F(D−f))` não é
homogênea. As duas juntas dão a conversão líquida correta, e isso está verificado
algebricamente em §1.5 e numericamente em §1.6.

**Medido no dado**: `max |k_value − k_eq3_mm/1000|` = **0,0 exato** em 3.617 + 171 + 1.644
amostras. Se houvesse uma segunda divisão, a identidade não fecharia.

### 1.5 A análise dimensional completa, do EXIF ao `defocus`

Feita do zero, sem copiar a do `CONTRATO.md`.

```
ENTRADA (EXIF + imagem + modelos)
  f                    [mm]
  F                    [adimensional]
  focal_length_35      [mm]
  max(H,W)             [px]
  z = DepthPro(I_aif)  [m]           ← métrico, sem normalizar
  M = BiRefNet(I_aif)  [booleano]

PASSO 1 — sensor
  crop_factor  = focal_length_35 / f             [mm/mm]  → adimensional
  sensor_mm    = 36 [mm] / crop_factor           → mm

PASSO 2 — pixel_ratio                            (Fig. 16, paper.txt:1186-1187)
  pixel_ratio  = max(H,W) [px] / sensor_mm [mm]  → px·mm⁻¹

PASSO 3 — plano de foco                          (Eq. 4, paper.txt:352)
  disp         = 1/z                             → m⁻¹
  focus_disp   = median(disp[M])                 → m⁻¹
  D_focus      = 1 / focus_disp                  → m
  focus_mm     = D_focus × 1000                  → mm

PASSO 4 — Eq. 3                                  (paper.txt:340)
  f² · focus_mm / (2·F·(focus_mm − f))
      = [mm²·mm] / ([1]·[mm])                    → mm²
  × pixel_ratio  = mm² × px·mm⁻¹                 → px·mm       =: k_eq3_mm

PASSO 5 — para a convenção da inferência
  K = k_eq3_mm / 1000                            → px·m
      (porque px·mm = px·m/1000; é troca de unidade, não fator de correção)

PASSO 6 — Eq. 2                                  (paper.txt:312)
  CoC = K · |disp − focus_disp|
      = [px·m] × [m⁻¹]                           → px          ✓ FECHA

PASSO 7 — condição da BokehNet                   (Inference_bokehNet.py:20, NÃO o paper)
  defocus = clip(CoC / MAX_COC, 0, 1)            → adimensional em [0,1]
```

Duas leituras que caem fora dessa cadeia e merecem ser ditas em voz alta:

**(a) O divisor `2F` significa que a Eq. 3 é RAIO, não diâmetro.** A fórmula clássica do
círculo de confusão dá o **diâmetro** `c = f²·s/(N(s−f)) · |1/z − 1/s|`. A Eq. 3 tem um
`2` a mais no denominador, logo entrega metade — raio. O renderizador clássico do BokehMe
espalha com raio, então os dois concordam (`src/control/contract.py:392-393`). Verificado
numericamente em §1.6.

**(b) O paper escreve `D` como PROFUNDIDADE e nós operamos em DISPARIDADE.** O §3.2 diz
*"D is the monocular depth map estimated from I_aif"* (`paper.txt:313-314`), e a palavra
*disparity* nunca aparece aplicada a `D`. Mas **só disparidade fecha dimensionalmente**:
`k_eq3` tem unidade px·mm, e `px·mm × [m] `, `px·mm × [adimensional]` e
`px·mm × [mm]` nenhum deles dá px — só `px·mm × [mm⁻¹]` dá. O paper, lido ao pé da letra,
é internamente inconsistente entre a Eq. 2 e a Eq. 3; o código oficial resolve o silêncio
(`Inference_bokehNet.py:94`, `disp = 1.0/depth`). **Este é o desvio D1 de §6, e é o de
maior alcance** — mas é o único que torna as equações do paper coerentes entre si.

### 1.6 Verificação numérica independente da Eq. 3

Escrevi a fórmula física do CoC do zero e comparei com o que o `contract.py` produz,
passando pela cadeia inteira (`k_eq3_mm` → `k_official` → `signed_coc_px`):

| caso | `pixel_ratio` | `k_eq3` | `K` | CoC (código) | CoC (física, raio) | erro rel. |
|---|---|---|---|---|---|---|
| f=50 F=2,0 s=2 m z=10 m 6000×4000 FF | 166,67 | 106.838 | 106,84 | 42,7350 px | 42,7350 px | **1,7e-16** |
| f=24 F=1,8 s=0,8 m z=5 m 4032×3024 1/6" | 672,00 | 110.845 | 110,85 | 116,3876 px | 116,3876 px | **1,2e-16** |
| f=85 F=1,4 s=1,5 m z=∞ 3000×4500 **retrato** FF | 125,00 | 341.920 | 341,92 | 227,9464 px | 227,9464 px | **1,2e-16** |

A Eq. 3 como implementada **é** a fórmula física do raio do círculo de confusão, à
precisão da máquina, inclusive em retrato.

Identidade extra conferida (a que `ROTA_B_AUDITORIA.md` afirma valer em 900/900 linhas da
tabela kfix): `f_mm × pixel_ratio == max(W,H) × focal_length_35 / 36`. Reproduzida:
`24,0 × 168,0 == 4032,0 == 4032 × 36/36`. ✓

Sensibilidade de `K` a `D_focus` (o que justifica §1.2), variando `D_focus` em +20%:

```
f=4 mm   s=2 m   →  K varia −0,033%
f=24 mm  s=2 m   →  K varia −0,202%
f=55 mm  s=2,7 m →  K varia ≈ −0,4%      ← as medianas reais do ITW
f=200 mm s=0,5 m →  K varia −10,0%       ← teleobjetiva em foco próximo, o pior caso
```

### 1.7 O validador independente — e o que ele realmente prova

A rota grava, por amostra, `k_validator_exif_focus_distance`: a Eq. 3 com a
**distância de foco da EXIF**, quando a câmera publica
(`src/routes/route_b.py:849-891`). O paper **proíbe** a EXIF como fonte de `D_focus`
(*"it is frequently missing or noisy; therefore, we do not rely on EXIF for D_focus"*,
`paper.txt:344-346`), e o código respeita: o valor **nunca** entra em `k_value`.

**Medido, pareado por amostra** (nossa variante, as 32 amostras em que a EXIF publica
distância de foco — 2,0% do lote):

```
k_rótulo / k_validador :  min 0,597   p25 0,998   mediana 1,0035   p75 1,009   max 1,060
dentro de ±20% : 31 de 32
```

Concordância mediana de **0,35%**. Mas é preciso ler isso com honestidade: como K é
quase insensível a `D_focus` (§1.2), esse validador prova que **nenhum dos outros
termos — `f`, `F`, `pixel_ratio` — está grosseiramente errado**, e não prova que o plano
de foco estimado está certo. É um teste forte da Eq. 3 e um teste fraco da Eq. 4.

O teste de fato da Eq. 4 na rota B é indireto e está em §4.3.

### 1.8 A Eq. 2, e o normalizador que o paper não tem

`D_def = K·|D − D_focus|` (`paper.txt:312`), **sem normalizador**. O código
(`src/control/contract.py:472-490`) divide por `MAX_COC = 100.0`, que vem de
`Inference_bokehNet.py:20` e **não do paper**. É o desvio D2 de §6.

A defesa estrutural está certa: `MAX_COC` **não é parâmetro** de `defocus_map` — a
assinatura verificada é `(depth_m, focus_disparity, k_value)`, sem terceiro argumento. Um
override parcial não tem por onde entrar, e `RouteBConfig` não expõe o campo. Foi assim
que o defeito B4 (`--max-coc` por rota, o mecanismo do kfix) ficou inexpressável.

**Medido — o clip quase nunca morde**, e agora isso é número e não suposição. Reconstruí
o CoC máximo de cada amostra a partir de `k_value`, `focus_disparity`, `disparity_min` e
`disparity_max`:

| | CoC máx/amostra (px) p50 | p95 | p99 | máx | ocupação p50 | amostras em que `MAX_COC` corta |
|---|---|---|---|---|---|---|
| oficial (N=3.617) | **5,284** | 20,08 | 36,64 | 114,26 | 0,053 | **1 (0,03%)** |
| nossa (N=410, s0) | **2,611** | 8,59 | 12,21 | 21,14 | 0,026 | **0 (0,00%)** |

A âncora do `CONTRATO.md:83` é *"CoC p99 mediano, rota B = 4,665 px"*; medimos 5,284 px de
mediana. Mesma ordem. E a previsão do `CONTRATO.md:45-46` — *"a rota B vai ocupar
[0, 0.05]"* — **confirma-se**: ocupação mediana 0,053 na oficial e 0,026 na nossa.

**Lacuna registrada**: nenhum campo de metadado grava diretamente a ocupação do mapa
(não existe `coc_p99_px` nem `defocus_max`). Os números acima são reconstruídos, o que só
foi possível porque `disparity_min`/`disparity_max`/`focus_disparity`/`k_value` estão
todos gravados. Gravar a ocupação direto seria barato e removeria a reconstrução.

---

## 2. O alvo é a foto real, e nenhum renderizador produz rótulo

**O que o paper diz.** A tupla de supervisão é `(I_aif, I_out, D_def)`
(`paper.txt:321`). A legenda da Fig. 3 separa as três rotas com precisão
(`paper.txt:287-299`):

* **(a)** *"[...] and **feed it into a bokeh renderer [43]** to synthesize corresponding
  bokeh images"* — `paper.txt:291-292`. O renderizador está **aqui**.
* **(b)** *"**Given real bokeh images**, DeblurNet recovers an AIF image. We then
  estimate depth and extract a foreground mask [86] to define the estimated focus plane
  D_focus. The bokeh level K is computed from the EXIF metadata [...]"* —
  `paper.txt:293-296`. **Nenhum renderizador.** A imagem com bokeh é o dado de entrada e
  o alvo ao mesmo tempo.
* **(c)** *"For real pairs, we obtain D_focus as in (b), and follow Eq. (2) to estimate
  the bokeh level K"* — `paper.txt:297-298`; o renderizador reaparece só no sweep da
  Eq. 5 (`paper.txt:369-372`).

`[43]` é o BokehMe (`paper.txt:851-852`). Ele aparece em (a) e na Eq. 5 de (c). **Não
aparece em (b).**

**O que o código faz.** `src/routes/route_b.py:999`, dentro de `SampleProvenance`:

```python
renderer=None,
```

com o comentário que nomeia o defeito que isso mata (`route_b.py:996-998`): o pipeline
antigo gravava `{"name": "not_applicable_route_b", "is_final_label_renderer": True}` —
proveniência que mente na direção tranquilizadora (defeito B14). Também:

* `calibration_ssim=None` — não há Eq. 5 nesta rota (`route_b.py:983`);
* `k_analytic=None` — na rota C a Eq. 3 valida o sweep; aqui a Eq. 3 **é** o rótulo, e
  repetir o número faria passar por validador independente algo que não é
  (`route_b.py:985-989`);
* `k_effective_factor=None` — os 0,9873 medidos descrevem o BokehMe, que esta rota não
  usa (`route_b.py:990-991`, `[A]` A12);
* `provenance.extra.route_b_decisions.renderer = None` — espelhado
  (`route_b.py:1028`);
* `provenance_base["renderer"] = None` no `run_config.json` do run
  (`scripts/run_route_b.py:620-624`).

**Medido — `provenance.renderer` não-nulo:**

```
b_full_official : 0 de 3.617
b_full_s1       : 0 de 171
b_full_nossa_*  : 0 de 1.644
                  ───────────
  total         : 0 de 5.432
```

E os campos correlatos, todos de valor único nas 5.432 amostras:

```
k_source        = "eq3_exif"                       (nenhum outro valor)
is_k_censored   = false                            (nenhuma amostra censurada)
calibration_ssim = null      k_analytic = null     k_effective_factor = null
mask_source     = "birefnet"      focus_source = "birefnet"
focus_was_refined = false         depth_backend = "depth_pro"
control_version = "metric_disparity_official_v1"
max_coc         = 100.0
```

**Fechado.** Nenhum renderizador toca no rótulo da rota B, e a proveniência não afirma
nenhum.

---

## 3. A DeblurNet

### 3.1 `main_adapter="deblurring"` chega ao `generate` — provado, não suposto

A cadeia, elo a elo:

1. `DeblurVariant.OURS_MAIN_COND.spec` → `main_adapter = ADAPTER_NAME = "deblurring"`,
   `lora_mode = "main_cond"` (`src/model_runtime/deblurnet.py:210-217`).
2. `DeblurVariantSpec.__post_init__` **falha na importação do módulo** se o par
   `(lora_mode, main_adapter)` divergir (`deblurnet.py:195-206`). O mapeamento é 1-para-1.
3. `DeblurNetRuntime.infer` passa o valor **sempre, explicitamente**, direto da spec:
   `main_adapter=self._spec.main_adapter` (`deblurnet.py:942`). Não há default nosso.
4. A proveniência grava o valor **efetivamente passado** (`deblurnet.py:857`).

**Não existe caminho para cruzar os pesos de uma variante com o adapter da outra**, e
isso é estrutural, não convenção:

* `main_adapter` **não é parâmetro** de nada — nem do `__init__`, nem de `infer`, nem de
  `resolve_weights`. Varredura: as 41 ocorrências de `main_adapter` em `deblurnet.py` são
  a spec, a checagem por AST, a proveniência e a chamada. Nenhuma é um argumento
  configurável.
* Não há `--main-adapter` na CLI (`scripts/run_route_b.py:454-486`). O antipadrão exato
  — uma flag de texto livre com default — está nomeado no docstring (`deblurnet.py:50-54`).
* `resolve_weights(variant)` não aceita `repo_id` nem nome de arquivo
  (`deblurnet.py:250-296`); os dois vêm da variante.
* `_check_weights_belong_to_variant` (`deblurnet.py:747-779`) exige que o **nome do
  arquivo** seja o da variante, e levanta `DeblurVariantMismatch` — exceção própria, para
  que um `except ValueError` genérico não a engula.
* `expected_lora_sha256` é a trava contra o único cruzamento que o nome não pega: um peso
  renomeado (`deblurnet.py:683-689`).

**A guarda contra o upstream.** `generate` termina em `**params: dict` — verificado no
checkout real do cluster: a assinatura vai de `flux.py:464` a `:489`, com
`main_adapter: Optional[List[str]] = None` em `:480` e `**params: dict` em `:488`. Logo um
kwarg renomeado pelo upstream seria **engolido em silêncio** e a inferência voltaria a ser
cond-only sem uma linha de log. `_inspect_flux_generate` (`deblurnet.py:521-591`) confere
por **AST**, antes de importar torch, que `generate` ainda declara `main_adapter` — e
levanta `RuntimeError` se não declarar. O uso a jusante também está conferido:
`flux.py:758`, `:791` e `:812` fazem `adapters=[main_adapter]*2 + c_adapters`, exatamente
como o docstring descreve.

**Medido na proveniência das amostras reais:**

| | oficial (3.788) | nossa (1.644) |
|---|---|---|
| `deblur_variant` | `official_cond_only` | `ours_main_cond` |
| `main_adapter` | `null` | **`"deblurring"`** |
| `deblur_lora_mode` | `cond_only` | `main_cond` |
| `deblur_lora_sha256` | `8d4c3960…c9f75` | `a1ed05e4…8cecc` |
| `deblur_repo_id` | `nycu-cplab/Genfocus-Model` | `juliadollis/genrefocus-deblurnet-paper-4gpu` |
| `genfocus_generate_accepts_main_adapter` | `true` | `true` |
| `genfocus_generate_main_adapter_default_is_none` | `true` | `true` |
| `genfocus_generate_swallows_unknown_kwargs` | `true` | `true` |
| `num_inference_steps` | 28 | 28 |
| `long_side` / `resize_policy` | 0 / `no_crop_multiple_of_16` | 0 / `no_crop_multiple_of_16` |

`null` na oficial **é o correto** para um LoRA cond-only (`Inference_deblurNet.py:103-111`
não passa o argumento), e é por isso que existe `deblur_lora_mode`, que nunca é vazio e
pode ser validado por truthiness. **O defeito B1 está morto, e está provado no dado.**

Duas notas de higiene, ambas de severidade baixa:

* **`batch_tiles=True` é gravado na proveniência e não faz nada.** `generate` não declara
  `batch_tiles` — conferido por grep no `flux.py` do cluster: zero ocorrências. O kwarg é
  absorvido pelo `**params`. A proveniência afirma um comportamento que não existe. É a
  classe exata de defeito que este projeto persegue, em escala pequena: ou remover o campo,
  ou marcá-lo como "enviado, ignorado pelo upstream".
* **`genfocus_commit` é `null` na nossa variante** — o checkout em
  `/raid/user_juliadollis/.../third_party/Genfocus` não é um repositório git, então
  `_git_commit` devolve `None`. A oficial tem `54752a13…`. O `genfocus_pipeline_sha256`
  (`be1bf8c7…`) é **idêntico nos dois**, então o `flux.py` é o mesmo arquivo — a
  proveniência ainda pinça o que importa. Mas o campo de commit está vazio de um lado.
* `flux_backbone_fingerprint` **difere entre os dois hosts** (`ccb2404…` contra
  `7bb7568…`) com o **mesmo** snapshot do FLUX (`3de623fc…`). O fingerprint é de
  caminhos+tamanhos+configs pequenas (`flux_backbone_fingerprint_kind`), e portanto não é
  comparável entre máquinas. Está honestamente rotulado, mas quem comparar os dois
  releases vai ver "backbone diferente" onde não há.

### 3.2 As 686 chaves — medido, e o que a medição realmente diz

A auditoria anterior e o `docker/run_route_b_h100n1.sh:24-27` afirmam: os dois
`.safetensors` têm as **mesmas 686 chaves**, e a diferença main+cond/cond-only vive no
**roteamento**, não no arquivo.

**Medido**, lendo só o cabeçalho JSON de cada arquivo (prefixo u64 little-endian; nenhum
peso carregado):

| arquivo | sha256 | chaves | dtype | rank `lora_A` | tamanho |
|---|---|---|---|---|---|
| nossa, **a que os runs usaram** | `a1ed05e4…8cecc` | **686** | F32 | **128** | 1,85 GB |
| oficial | `8d4c3960…c9f75` | **686** | BF16 | **64** | 464 MB |
| nossa, **cópia publicada no HF** | `deea4988…32152` | **688** | F32 | 128 | 1,86 GB |

**A afirmação das 686 chaves confirma-se para o par que está em jogo**: o conjunto de
chaves de `a1ed05e4` e de `8d4c3960` é **idêntico**, 686/686, zero chaves de um lado só.
E o código trata isso exatamente como a auditoria descreve
(`deblurnet.py:23-27`, `:751-755`): o peso não sabe de qual convenção precisa; a defesa é
amarrar a convenção ao peso **no código**, mais o nome do arquivo e o sha opcional.

**Duas correções de precisão que a medida obriga:**

1. O docstring diz *"Um LoRA main+cond e um cond-only têm **o mesmo formato** e as mesmas
   chaves"* (`deblurnet.py:23-24`) e *"Nenhuma inspeção do peso separa os dois"*. Isso é
   verdade como afirmação sobre a **convenção** — rank e dtype são independentes de
   main+cond/cond-only, e um cond-only de rank 128 é perfeitamente possível. Mas é **falso
   como afirmação sobre estes dois arquivos concretos**: rank 128 contra 64 e F32 contra
   BF16 os separam trivialmente. A defesa do código continua correta e suficiente; o que
   está impreciso é a frase, e vale corrigi-la para não induzir alguém a concluir que o
   sha256 é supérfluo.
2. **A trava `--deblur-lora-sha256` não está sendo usada nos runs em voo.** O
   `docker/run_route_b_h100n1.sh` não a passa, e `expected_lora_sha256` fica `None`. A
   defesa efetiva é só o nome do arquivo. Recomendação: passar o sha nos próximos lotes —
   custa uma flag.

**Achado novo, e ele é de proveniência, não de rótulo**: a cópia publicada no HF
(`deea4988`, **688** chaves) **não é** o arquivo com que as amostras `_de4` foram geradas
(`a1ed05e4`, **686** chaves). As duas chaves a mais são
`transformer.proj_out.lora_A.weight` `[128, 3072]` e `transformer.proj_out.lora_B.weight`
`[64, 128]`. Se algum release, card ou paper citar o repositório HF como "os pesos usados",
a afirmação está errada. Ou se republica o arquivo usado, ou se corrige a citação.

### 3.3 O rank — e aqui o paper **publica**, e a leitura é contraintuitiva

`paper.txt:511-513`:

> *"Our backbone is FLUX-1-dev [58], fine-tuned via LoRA [25] with a conditioning scheme
> following [62, 63]. **DeblurNet employs LoRA rank r=128, while BokehNet uses r=64.**"*

O paper **publica** o rank, e o valor para a DeblurNet é **128**. Medido nos shapes de
`lora_A` (343 tensores em cada arquivo, um único valor de rank por arquivo, sem módulos de
rank misto):

| | rank medido | contra o paper |
|---|---|---|
| **nosso** checkpoint (`a1ed05e4`) | **128** | **bate** com `paper.txt:513` |
| checkpoint **oficial** (`8d4c3960`) | **64** | **não bate** com `paper.txt:513` |

Isto inverte a intuição de "oficial = fiel ao paper". No eixo do rank, o **nosso**
checkpoint é o que reproduz a especificação publicada; o peso que os autores liberaram no
`nycu-cplab/Genfocus-Model` é rank 64, que é o rank que o paper atribui à **BokehNet**, não
à DeblurNet.

**Afeta a fidelidade ao paper?** Sim, e nas duas direções:

* Gerar a rota B com a **oficial** significa que a AIF vem de um modelo cuja capacidade
  (rank 64) é metade da que o paper declara ter treinado para a DeblurNet. Isso é um desvio
  do paper — do lado em que ninguém suspeitaria.
* Gerar com a **nossa** (rank 128) casa com `paper.txt:513`, mas o treino é nosso, com
  nossos dados e nosso `step_60000`, e a §4.1 do paper especifica 60K steps
  (`paper.txt:515`) — o número bate, o dado de treino não é o mesmo. É reprodução da
  receita, não do artefato.

**`[A]` A14 permanece aberto e agora é mais agudo**: o paper não diz se os dados da rota B
foram gerados com o peso publicado ou com um intermediário do treino
(`ROTA_B_AUDITORIA.md:968`). A medida de rank acrescenta que o peso publicado **não é** o
que a §4.1 descreve, o que torna "os autores usaram o peso publicado" menos provável, não
mais.

**Correção a um registro anterior.** `REGISTRO.md:1377-1381` registra a medida de rank
corretamente (`[M]`, 128 contra 64) e conclui *"a usuária afirmou que as configs de LoRA
são iguais às do paper; neste ponto **não são**"*. A medida está certa; a conclusão está
invertida quanto a **quem** diverge. `paper.txt:513` publica `r=128` para a DeblurNet, e o
**nosso** é 128. Quem não bate com o paper, nesse eixo, é o checkpoint **oficial**. A
ressalva do registro — *"pode ser capacidade, não treino"* — continua válida e importante
para interpretar um eventual "o nosso ficou melhor": ele tem o dobro de capacidade do
peso publicado, mas é o rank que o paper especifica.

**Recomendação**: não é escolha entre certo e errado. Os dois lotes existem, com o **mesmo
`sample_id`**, e isso é o ativo. Mantenha os dois, declare o rank de cada um no card, e
deixe a §5.4 decidir qual vai para treino.

---

## 4. `D_focus` e a máscara — a Decisão 2, conferida no texto

### 4.1 A leitura do paper se sustenta

A Decisão 2 (`ROTA_B_DECISOES.md:91-157`, `route_b.py:79-115`) diz que a rota B **não**
aplica o refinamento do §3.2(c), porque o paper o reserva a (c). **Conferi no texto, e a
leitura está correta.** Quatro evidências, em ordem de força:

**(i) O que o paper especifica para (b) são três passos, e nenhum é refinamento**
(`paper.txt:347-352`):

> *"Instead, we compute the focus plane based on an in-focus mask and a monocular depth
> estimate [...]. Specifically, we estimate an in-focus mask M using BiRefNet [86] and
> produce a depth map D using a monocular depth estimator. The focus plane is then
> determined as the median depth within the masked in-focus area: D_focus = median(D[M])."*

BiRefNet → profundidade → mediana. A frase é fechada; não há um quarto passo.

**(ii) O `"Similar to (b)"` marca onde (c) ainda coincide com (b), e o `"However"` marca
onde ela deixa de coincidir** (`paper.txt:360-365`):

> *"**Similar to (b)**, we employ BiRefNet [86] to obtain an initial in-focus mask M.
> **However**, due to the increased diversity and complexity of the scenes **in these
> datasets**, the initial estimate of M is sometimes unreliable. [...] we introduce a
> manual refinement step."*

A estrutura sintática é decisiva: (b) é o caminho **até** o BiRefNet; o refinamento entra
depois do `However`, isto é, **fora** do que (b) faz.

**(iii) O `"in these datasets"` é uma afirmação comparativa**, e o comparando é o ITW — o
único outro dataset real já introduzido. O paper está dizendo que LFDOF e RealBokeh são
mais difíceis **que o ITW**, o que é a justificativa explícita para o passo extra existir
só lá.

**(iv) O suplemento B.2 confirma pelo outro lado** (`paper.txt:999-1005`):

> *"This collection comprises 13K **previously filtered and verified** images from the ITW
> dataset [19], alongside 13K images newly curated for this work. The newly collected data
> consists of focus-consistent series [...]. To establish accurate ground truth **for these
> series**, we first conducted a manual verification and refinement process on the in-focus
> masks."*

O refinamento manual descrito em B.2 é **"for these series"** — as 13K **novas**, não as
13K do ITW. E as do ITW são descritas como *"filtered and verified"*, isto é, **triadas**,
que é exatamente o que a rota B faz: máscara que não serve **rejeita a amostra**.

**Veredito: o paper sustenta a Decisão 2.** O código está alinhado ao texto, não a uma
conveniência.

### 4.2 E o gatilho declarado para revisitá-la não disparou

A Decisão 2 declarou dois números que a reabririam (`ROTA_B_DECISOES.md:141-150`,
`route_b.py:108-114`): `focus_mask_empty` acima de **5%**, ou `focus_agreement` mediano
abaixo de **0,30**.

**Medido:**

| | `focus_mask_empty` | taxa | `focus_agreement` mediano |
|---|---|---|---|
| oficial (`b_full_official`) | **8** de 3.625 | **0,22%** | 0,768 |
| oficial (`b_full_s1`) | **0** de 171 | 0,00% | — |
| nossa (4 shards `_de4`) | **22** de ~1.630 processadas | **≈1,3%** | 0,566 |

Nenhum dos dois gatilhos dispara: 0,22%–1,32% contra o teto de 5%, e 0,566–0,768 contra o
piso de 0,30. E o contraste com a rota C — **20,6%** de `focus_mask_empty` na RealBokeh
(`REGISTRO.md`, etapa 9) — confirma a previsão da Decisão 2: o BiRefNet é um segmentador de
objeto **saliente**, e no ITW (fotos de Flickr em que o fotógrafo focou um sujeito) as duas
coisas coincidem. Nos datasets de cena da rota C, não coincidem — que é literalmente o que
`paper.txt:361-363` diz.

**`focus_mask_empty` é o único motivo de rejeição observado em todo o dado real.** Zero
por EXIF, zero por profundidade, zero por K, zero por gate de qualidade (todos os gates
estão em modo medir, §5.1).

### 4.3 O que ainda **não** foi testado sobre `D_focus`

Sendo rigoroso: o que está medido acima é que o BiRefNet **devolve uma máscara**, que ela
concorda razoavelmente com a região de maior retenção, e que a região marcada é de fato a
mais nítida da foto com bokeh — `focus_mask_sharpness_ratio` mediano **2,34** na nossa
variante, isto é, a região da máscara tem 2,3× a nitidez do resto **na bokeh**, que é a
hipótese física correta.

O que **não** está medido é se esse é o plano de foco **certo**, porque não há gabarito de
distância de foco no ITW. O caso de falha que nenhum desses números pega continua sendo o
declarado em `qc/gates.py:38-40`: o fotógrafo focou o **fundo**, o BiRefNet acha o objeto
saliente em primeiro plano, as duas máscaras concordam, a IoU é alta — e o plano está
errado. **`[A]` A2 e a acurácia de `D_focus` no ITW seguem abertos.**

Indício indireto favorável: `mask_iou_aif_bokeh` mediano **0,9928** na oficial e **0,9423**
na nossa. A máscara extraída da AIF e a extraída da bokeh são praticamente a mesma região,
o que enfraquece o `[A]` A2 (*"a máscara sai da AIF"*) como risco — mas não resolve o caso
do fundo focado, porque justamente nesse caso as duas concordam.

---

## 5. O que está rodando, e o que o dado diz

### 5.1 O estado, verificado

**dgx-H100-01**, quatro containers `b4_s0` … `b4_s3`, `Up 3 hours`, imagem
`julia-genrefocus:1.0`, escrevendo em `output/b_full_nossa_s{0..3}_de4`. Config lida do
`run_config.json` da fatia 0:

```
deblur_variant  = ours_main_cond        main_adapter = "deblurring"   lora_mode = main_cond
deblur_long_side = 0                    resize_policy = no_crop_multiple_of_16
num_inference_steps = 28                seed = 42       prompt = "a sharp photo ..."
num_shards = 4                          pilot = 13800   include_pseudo_aif = false
renderer = None                         depth = depth_pro   mask = birefnet
limiares congelados = {}   ← TODOS os gates em MODO MEDIR
```

**Nenhum limiar está ligado.** Isso é legítimo e é o que o `run_route_b.py:611-614`
avisa em voz alta (*"AVISO: run completo com todos os gates em modo medir"*), e é coerente
com a política de `qc/gates.py:3-6` — limiar não medido não bloqueia. Mas a consequência
tem que ser dita: **este lote não filtra nada além de `focus_mask_empty`**. Se ele virar
treino, vira treino com as 4,3%/1,3% de amostras da cauda ruim de §5.3 dentro.

**dgx-H100-03**: `b_full_official` é a **fatia 0 de 2** de um run da variante oficial, com
3.617 de 6.874 linhas → **52,6% completo**. `b_full_s1` é a fatia 1 do **mesmo** run
(diff completo dos dois `run_config.json`: seis campos, todos de bookkeeping — `output_dir`,
`shard`, `rows_enumerated` e três do orçamento de disco), com 171 de 6.926 → **2,5%**. Os
3.788 são ~27% das 13.800.

Geometria: o `long_side = 0` leva ao ramo `NO_CROP_MULTIPLE_OF_16`, que arredonda para cima
e **nunca recorta** — round-trip é identidade geométrica exata por construção
(`deblurnet.py:444-450`). Confirmado no log do container: `Latent size: 43x64` → imagem
688×1024, `min(H,W) = 688 ≥ 512` → `NO_TILED_DENOISE = False` → tiling ligado, 6 tiles. Isso
reproduz `Inference_deblurNet.py:95` (`force_no_tile = min(w,h) < 512`) **exatamente**, e a
semântica do parâmetro está correta: `flux.py:572` faz `if NO_TILED_DENOISE: TILED_DENOISE
= False`. O tiling é de **inferência**, que é o que `paper.txt:480-481` autoriza
(*"a tiling strategy inspired by [5] **during inference**"*).

### 5.2 `k_value` contra as âncoras 16,6 / 20,1 / 15,0

```
oficial  N=3.617  min 0,834  p05 2,318  p25 8,907  p50 16,902  p75 26,31  p95 37,58  p99 39,39  max 47,62
s1       N=  171                                    p50 11,736                                   max 40,27
oficial combinada N=3.788                           p50 16,775
nossa    N=1.644  min 0,834  p05 2,229             p50 17,117  p75 26,43  p95 37,72            max 43,05
```

| âncora | origem | contra a mediana oficial (16,902) | contra a nossa (17,117) |
|---|---|---|---|
| **16,6** | mediana da tabela `rota-b-kfix-eq3` ÷ 1000 (`CONTRATO.md:76`) | **+1,8%** | **+3,1%** |
| **20,1** | mediana calculada da EXIF, 318 amostras (`CONTRATO.md:77`) | −15,9% | −14,8% |
| **15,0** | default de `Inference_bokehNet.py:53` (`CONTRATO.md:78`) | +12,7% | +14,1% |

A mediana cai **em cima da âncora 16,6** — a que foi calculada pela mesma Eq. 3 sobre o
mesmo dataset — e fica entre as outras duas. Quatro caminhos independentes na mesma faixa.
A Fig. 12 do paper varre `K ∈ {0, 5, 10, 15}` (`paper.txt:1151-1156`); nossa massa está em
`[2,3 · 37,6]` (p05–p95), portanto do mesmo lado da régua, deslocada para cima porque a Fig.
12 é uma demonstração de controlabilidade, não a distribuição de treino.

**85,6%** das amostras da nossa variante (1.396 de 1.630) caem em `[5, 40]`.
**Nenhuma amostra censurada** (`is_k_censored`
false em 5.432/5.432), porque a rota B **rejeita** K implausível em vez de censurar no teto
— que é a lição do `--k-max 300` da rota C, onde **47,0%** ficaram no teto exato sem que
nada denunciasse (`ACHADOS.md:19`). Como nenhum limiar de K está ligado, nem isso disparou.

### 5.3 `k_eq3_diagnostics` — qual termo está fora quando K é estranho

O diagnóstico por amostra responde a pergunta. Caudas da oficial (medianas; 180 amostras
em cada cauda de 5%):

| termo | cauda inferior 5% | global | cauda superior 5% |
|---|---|---|---|
| **`k_value`** | **1,75** | **16,90** | **38,57** |
| `focal_length_mm` | **18** | **55** | **85,5** |
| `f_number` | 4,0 | 4,5 | 4,0 |
| `pixel_ratio_px_per_mm` | 42,67 | 42,625 | 42,625 |
| `focus_depth_m` | 1,547 | 2,714 | 3,246 |
| `sensor_width_mm` | 24 | 24 | 24 |
| `crop_factor` | 1,5 | 1,5 | 1,5 |

**O termo que explica a cauda é a distância focal, e isso é a física, não um defeito.**
`K ∝ f²` (com `f ≪ D_focus`), e 18 → 85,5 mm sozinho dá `(85,5/18)² ≈ 22,6×`, que cobre
toda a faixa observada. O `f_number` é **plano** entre as caudas (4,0 / 4,5 / 4,0), o
`pixel_ratio`, o `sensor_width` e o `crop_factor` são planos, e o `focus_depth_m` contribui
com um empurrão secundário fraco — exatamente o esperado no regime `z ≫ f` (§1.2).

Na fatia `b_full_s1` aparece um segundo mecanismo na cauda inferior: `pixel_ratio` 110,9
contra 42,7 e `crop_factor` 3,9 contra 1,5, isto é, **câmeras compactas de sensor
pequeno** — e ali `f` chega a 7 mm. Também física correta.

**Conclusão**: quando K é estranho na rota B, é porque a lente é estranha. Não há amostra
em que o K saia da faixa por um termo errado; as identidades bit a bit de §1.3 já provam
que nenhum termo foi adulterado depois de calculado.

### 5.4 `deblur_diagnostics` — "deblurou" contra "não fez nada" contra "lavou"

Este é o eixo que o §4.3 da auditoria anterior pediu, e é o que produz a única ressalva
séria desta auditoria.

Os campos gravados por amostra: `deblur_retention_p05/p50/p95`, `deblur_retention_spread`,
`deblur_retention_measurable_fraction`, `retention_region_found`, `retention_semantics`.
O **SSIM(AIF, bokeh)** não fica aí — ele vive em
`quality.deblur_structural_ssim_min`/`_max` (mesmo valor nos dois).

**Os dois testes que o design declarou** (`route_b.py:427-447`, `:138-142`):

| caso | razão de nitidez | SSIM(AIF,bokeh) | dispersão da retenção |
|---|---|---|---|
| deblur bem-feito | baixa | intermediário | espalhada |
| LoRA não carregou | ≈ 1 | ≈ 1 | ≈ 0 |
| AIF lavada | baixa ou alta por artefato | baixo | baixa em todo lugar |

**Medido:**

| critério | oficial (3.788) | nossa (1.644) |
|---|---|---|
| SSIM mediano | 0,836 | **0,564** |
| SSIM p05 · p95 | 0,600 · 0,956 | 0,426 · 0,681 |
| SSIM > 0,95 | 271 (**7,15%**) | **0 (0,00%)** |
| SSIM > 0,90 | 976 (**25,8%**) | **0 (0,00%)** |
| `retention_spread` < 0,05 | 79 (2,09%) | **0 (0,00%)** |
| **"NÃO FEZ NADA"** (SSIM>0,95 **e** spread<0,05) | **10 (0,26%)** | **0 (0,00%)** |
| SSIM < 0,60 | 189 (4,99%) | 1.138 (70,9%) |
| `retention_p50` < 0,30 | 415 (11,0%) | 37 (2,31%) |
| **"LAVOU"** (SSIM<0,60 **e** p50<0,30) | **162 (4,28%)** | **21 (1,31%)** |
| `bokeh_over_aif_sharpness` mediano | **0,674** | **1,340** |

*Os N da coluna "nossa" oscilam entre 1.586 e 1.644 conforme o instante da leitura — os
quatro containers estão escrevendo. As frações não se movem.*

**O defeito B1 não está acontecendo em nenhuma das duas**: o caso "LoRA não carregou" é
0,26% na oficial e **zero** na nossa. E pelo teste de duas pontas do próprio projeto, a
nossa variante "lava" **menos** que a oficial (1,31% contra 4,28%).

**Mas há um sinal que os dois testes não pegam, e ele é forte.** A comparação **pareada,
nas mesmas 1.644 fotos** (só possível porque o `sample_id` é `b_<flickr_photo_id>` nas duas
variantes, e a fatia 0 de 4 é subconjunto da fatia 0 de 2):

| grandeza | oficial p50 | nossa p50 | razão nossa/oficial p05 · **p50** · p95 |
|---|---|---|---|
| `k_value` | 17,077 | 17,117 | 0,977 · **0,999** · 1,024 |
| `focus_disparity` | 0,3708 | 0,3400 | 0,467 · **0,938** · 2,326 |
| vão de disparidade | 0,5029 | 0,2290 | 0,134 · **0,494** · 1,123 |
| `z_max_m` | 103,61 | **7,24** | 0,000 · **0,064** · 0,912 |
| CoC máx (px) | 5,333 | 2,433 | 0,130 · **0,491** · 1,154 |
| SSIM(AIF,bokeh) | 0,830 | 0,564 | 0,519 · **0,697** · 0,936 |
| **`aif_laplacian_variance`** | **1005,0** | **358,9** | 0,174 · **0,438** · 0,868 |
| `bokeh/aif` laplaciano | 0,638 | **1,344** | 1,152 · **2,281** · 5,743 |
| resposta da correlação de fase | 562,5 | **83,7** | 0,098 · **0,154** · 0,226 |
| `mask_iou_aif_bokeh` | 0,9928 | 0,9423 | 0,467 · 0,962 · 1,000 |

Três leituras, em ordem de importância:

**(1) `K` é robusto à AIF, e isso é uma vitória do desenho.** Razão mediana **0,999**;
**nenhuma** amostra fora de ±20%; só 33 de 1.644 (2,0%) fora de ±5%. Era exatamente a
promessa do `sample_id` compartilhado (`run_route_b.py:42-45`: *"a diferença de K entre as
versões vira uma medida direta de quanto a AIF influencia o rótulo"*). A medida existe e o
número é: **quase nada**. Isso decorre de §1.2 — K depende de `D_focus` com fator
`f/(D_focus−f) ≈ 0,02`.

**(2) O mapa de defocus NÃO é robusto à AIF.** `focus_disparity` difere em mais de 5% em
**88,5%** dos pares, mais de 20% em **60,2%**, mais de 50% em **21,0%**. E o `z_max` cai por
um fator de **15** (103,6 m → 7,2 m): o Depth Pro, sobre a AIF da nossa variante, deixa de
enxergar fundo distante. Consequência: o vão de disparidade cai à metade e o CoC máximo cai
de 5,33 px para 2,43 px. Como `D_def = K·|D − D_focus|`, e `D_def` é o que a BokehNet
consome, **os dois lotes ensinam coisas fisicamente diferentes apesar de terem
praticamente o mesmo K**. Isto não estava medido em lugar nenhum antes.

**(3) A AIF da nossa variante tem menos alta frequência que a foto borrada de entrada.**
Variância de Laplaciano 2,3× menor que a da oficial nas mesmas fotos, e
`bokeh/aif > 1` em **77,1%** das amostras da nossa contra **22,8%** da oficial. Ou seja: em
três de cada quatro fotos, a "AIF" produzida tem **menos** energia de alta frequência que
a bokeh que ela deveria estar deblurrando.

Por que os dois testes de §5.4 não pegam isso: **a retenção satura em 1,0**. O próprio
módulo registra a limitação (`route_b.py:139-142`, `focus_region.py:210`): *"bokeh mais
detalhada que a AIF num pixel significa que a DeblurNet removeu detalhe ali — informação
real que o `clip` descarta"*. É precisamente o caso aqui, e por isso
`retention_p50`/`spread` saem quase idênticos nas duas variantes (0,750 contra 0,777;
0,739 contra 0,780) enquanto o SSIM e a razão de nitidez divergem muito. **O desenho de
dois eixos funcionou: o eixo que vê isso é o SSIM + razão de nitidez, e ele está gravado
em toda amostra.** Nenhum limiar está ligado para bloquear, porque nenhum foi medido antes
de agora.

**O que isso NÃO é**: não é o defeito B1. A proveniência prova `main_adapter="deblurring"`
com `lora_mode="main_cond"` em 1.644/1.644 amostras (§3.1). E não é o modo "lavado"
clássico, porque a estrutura sobrevive (retenção comparável, IoU de máscara 0,94, SSIM 0,56
e não 0,1).

**O que isso pode ser**, e o metadado não decide entre as três:
(a) o checkpoint de 60K steps rank 128 produz saídas mais suaves e mais "regeneradas",
trocando textura de alta frequência (inclusive ruído e artefato de JPEG do rendition
Flickr) por estrutura limpa — o que baixaria a variância de Laplaciano **sem** ser um
defeito; (b) o checkpoint está sub ou sobre-treinado e amacia de fato; (c) alguma coisa na
inferência da nossa variante difere da oficial além do adapter. A queda de 6,5× na resposta
da correlação de fase (562 → 84) favorece (a)/(b): a AIF correlaciona muito menos com a
entrada.

**Há uma evidência externa que empurra fortemente para (a), e ela muda a conclusão.**
`REGISTRO.md:1388-1395` traz o benchmark da Tabela 2 do paper, rodado pela usuária com os
dois checkpoints:

| | LPIPS ↓ | DISTS ↓ | paper, Tab. 2 (`paper.txt:456,463`) |
|---|---|---|---|
| **RealDOF** — oficial | 0,2397 | 0,1153 | 0,2408 / 0,1126 |
| **RealDOF** — nosso 60k | **0,2291** | **0,1089** | — |
| **DPDD** — oficial | **0,1596** | **0,0844** | 0,1440 / 0,0772 |
| **DPDD** — nosso 60k | 0,1744 | 0,1021 | — |

O checkpoint oficial reproduz os números publicados do paper de perto (RealDOF 0,2397
contra 0,2408 de LPIPS), o que valida o harness. E **o nosso ganha no RealDOF** — 0,2291
contra 0,2397 de LPIPS e 0,1089 contra 0,1153 de DISTS —, que é **fotografia real com
desfoque óptico, o domínio exato da rota B**. Perde no DPDD, que é desfoque de par de
pixels de sensor.

Uma AIF "lavada" não melhora LPIPS nem DISTS contra gabarito real. Portanto a queda de
2,3× na variância de Laplaciano **não é degradação perceptual** — é o modelo trocando
energia de alta frequência (ruído de sensor e artefato de JPEG do rendition do Flickr,
que a variância de Laplaciano conta como "nitidez") por estrutura limpa. A interpretação
(a) está sustentada por métrica com gabarito.

**Recomendação, rebaixada de bloqueante para conferência:** abrir **20 a 30 pares
`generated/<id>_aif.jpg` × foto de origem**, lado a lado, nas duas variantes. É barato,
os arquivos estão em disco, e fecha o assunto. **`[A]` A16**, aberto aqui com **risco
médio**, não alto.

O que **permanece** verdadeiro e importante independentemente disso é o item (2): o
**mapa de defocus difere muito entre as duas variantes** (vão de disparidade pela
metade, `z_max` 15× menor) mesmo com o mesmo K. Os dois lotes não são intercambiáveis, e
a escolha de qual vai para treino é uma decisão real — o benchmark acima favorece o
nosso para o domínio da rota B.

Consequência secundária: como a resposta da correlação de fase da nossa variante é 6,5×
menor, a afirmação "deslocamento = 0" (§5.5) é **mais fraca** do lado da nossa do que do
lado da oficial. O `pair_registration_response` existe justamente para isso
(`route_b.py:545-547`: *"baixo significa que o deslocamento medido NÃO é confiável, não
que ele é zero"*), e ele está fazendo o trabalho dele.

### 5.5 `pair_registration_shift_px` — o `[A]` A13, medido pela primeira vez

| | N | p50 | p95 | p99 | máx | > 1 px | > 3 px | > 6 px |
|---|---|---|---|---|---|---|---|---|
| oficial | 3.788 | 0 | 0 | 0 | **5,099** | 5 (0,13%) | 2 (0,05%) | **0** |
| nossa | 1.644 | 0 | 0 | 0 | **1,414** | 1 (0,06%) | **0** | **0** |

O `[A]` A13 estava marcado como **nunca medido** (`ROTA_B_DECISOES.md:433`). **Agora está
medido, e o resultado é o melhor possível**: a DeblurNet não desloca a AIF. O máximo
observado em 5.432 amostras é 5,1 px, contra o gate de 6,0 px do pipeline antigo, e a
mediana é zero exato. O recorte de campo de visão que produziria 30,86 px
(`deblurnet.py` docstring) não acontece porque a política é `no_crop_multiple_of_16` com
`long_side = 0`, em que o round-trip é identidade por construção.

Com a ressalva de §5.4: na nossa variante a resposta da correlação de fase é 83,7 mediana
contra 562,5 da oficial, então o zero é medido com muito menos margem.

### 5.6 `crop_factor == 1,0` exato — e uma boa notícia para o `[A]` A7

| corpus | exatamente 1,0 | N | % |
|---|---|---|---|
| oficial (`b_full_official`) | **699** | 3.617 | **19,33%** |
| oficial (`b_full_s1`) | 35 | 171 | 20,47% |
| nossa (`_de4`) | 240 | 1.620 | 14,81% |

A âncora de `ACHADOS.md:165` é **30,33%** (4.185 de 13.800), medida sobre a **coluna
inteira**. Os números acima são de prefixos do dataset (os runs processam em ordem de
`row_index`, não em amostra aleatória), então não contradizem a âncora — só ainda não a
alcançaram. **Não tratar 19,33% como o número final.**

**O que é novo e importa**: a tabela `make/model`, que não existia. Top 10 com
`crop_factor == 1,0` na oficial (contagem a cf=1,0 / total no dataset / taxa):

```
NIKON D810   122/123  0,992      NIKON D750   51/61  0,836
NIKON D800   103/106  0,972      NIKON D4     44/49  0,898
NIKON D700    61/62   0,984      SONY ILCE-7M2 44/47 0,936
NIKON D610    57/58   0,983      NIKON D600   34/65  0,523
NIKON D4S     52/55   0,945      SONY ILCE-7   20/20 1,000
```

**Todos são corpos full-frame de verdade.** D810, D800, D700, D610, D600, D750, D4, D4S,
ILCE-7 e ILCE-7M2 são 35 mm. Logo, nesses 10 modelos, `crop_factor == 1,0` é uma **medida
correta**, não a assinatura de uma câmera que ecoa a focal no campo de 35 mm quando não
sabe o valor — que era a hipótese de risco alto do `[A]` A7
(`ROTA_B_AUDITORIA.md:955`). O `sensor_width_mm = 36,0` está certo para elas, e o K também.

O `[A]` A7 **não fecha**, por três razões: (i) os 10 modelos cobrem uma fração dos 699; (ii)
a taxa do D600 é 0,523 e a do D750 é 0,836, isto é, o **mesmo corpo** às vezes reporta
`focal_length_35 ≠ focal_length` — ruído de EXIF por disparo, que empurra na direção segura
mas mostra que o campo não é confiável por construção; (iii) a cauda de sensores pequenos
(`crop_factor` até 9,75, `sensor_width_mm` até 3,69 mm) não foi auditada por modelo. Mas o
risco baixou de **alto** para **médio**, e a marcação está funcionando como desenhada:
`exif_crop_factor_unity` **nunca bloqueia**, por construção (`route_b.py:549-571` — não
recebe limiar, e `GATE_TO_REASON` deliberadamente não o inclui, `route_b.py:699-703`).

### 5.7 Outros números do lote, para referência

```
resolução: lado longo = 1.024 px em quase todas (oficial p05 1023 · p50 1024 · máx 1024;
           mínimo observado 556). Retrato (H>W): 461 de 1.630 = 28,3%.
depth_useful_levels (nossa): p05 35.957 · p50 55.328 · máx 64.619 de 65.535
           — contra os 24,7% do dataset antigo com menos de 256 níveis. Sem degeneração.
focus_depth_m: p05 1,04 · p50 2,71 · p95 8,60 · máx 72,36 m (oficial)
           — nenhuma amostra perto da sentinela de 10.000 m do Depth Pro, e por isso
             ZERO rejeições por focus_depth_implausible.
mask_area_ratio (nossa): p05 0,098 · p50 0,335 · p95 0,547
quantization_coc_error_px_at_depth_hw: p50 4,0e-05 px, máx 8,1e-04 px. Irrelevante.
```

O lado longo uniformemente 1.023–1.024 px **responde parcialmente o `[A]` A6**: as imagens
do `atfortes/BokehDiffusion` são renditions "Large 1024" do Flickr, **não** a resolução de
captura. Isso é benigno para a Eq. 3 **se** o rendition é um redimensionamento puro, porque
`pixel_ratio = max(H,W)/sensor_mm` é calculado na **mesma** imagem em que o CoC é medido —
a razão se ajusta sozinha. Seria maligno se houvesse **recorte**, que mudaria o campo de
visão sem mudar `focal_length_35`. O Flickr não recorta o "Large 1024"; a proporção varia
livremente no nosso dado (681×1023, 678×1024, 1024×678, …), o que é consistente com resize
puro. **`[A]` A6 baixa de alto para baixo, mas não fecha**: não verificamos contra o
original de nenhuma foto.

---

## 6. Os desvios do paper — declarados e defensáveis

| # | desvio | o que o paper diz | por que desviamos | custo **medido** |
|---|---|---|---|---|
| **D1** | `D` é **disparidade** (1/z), não profundidade | *"D is the monocular depth map"* — `paper.txt:313-314` | **é a única leitura que fecha dimensionalmente** (§1.5-b), e o código oficial resolve o silêncio: `Inference_bokehNet.py:94,118` | nenhum — a alternativa não tem unidade coerente |
| **D2** | `MAX_COC = 100` normaliza a Eq. 2 | Eq. 2 é **crua**, sem normalizador — `paper.txt:312` | `Inference_bokehNet.py:20`; a BokehNet consome `[0,1]` | o clip morde em **1 de 3.617** (0,03%); ocupação mediana 0,053 |
| **D3** | Eq. 3 recebe `1/median(1/z[M])`, não `median(z[M])` | `D_focus = median(D[M])` — `paper.txt:352` | mesmo plano que gera o mapa; regra 5 do contrato | **1,2 × 10⁻⁷** de desvio relativo, N=1.630; efeito em K ≈ 10⁻⁹ |
| **D4** | treino em crop 512, `K` reescalado | o paper **não publica** a resolução de treino; o tiling da §3.5 é de **inferência** (`paper.txt:480-481`) | decisão nossa, declarada | fator por amostra via `k_at_resolution`; medidos 0,892 e 0,821 |
| **D5** | sem refinamento da região em foco em (b) | o refinamento é de (c) — `paper.txt:360-368`; B.2 confirma (`paper.txt:999-1005`) | **sustentado pelo texto** (§4.1) | `focus_mask_empty` 0,22%–1,32%, muito abaixo do gatilho de 5% |
| **D6** | rejeitar quando a EXIF falta | o paper **cala** | regra 4, lado conservador (`[A]` A8) | **zero** rejeições por EXIF no dado real |
| **D7** | sensor via crop factor de `focal_length_35` | o paper **não diz** como obteve a largura (`[A]` A7) | única via disponível | 19,3% com cf=1,0, e os 10 modelos mais frequentes **são** full-frame |
| **D8** | `k_effective_factor = 0,9873` **não** aplicado | — | descreve o BokehMe, que esta rota não usa (`[A]` A12) | ~1% de diferença de escala entre rotas, declarada |

**D1 a D3, D5 e D6 são defensáveis sem ressalva.** D4 é decisão declarada com o fator
gravado. D7 é o desvio com mais risco residual, e §5.6 o reduziu. D8 é o certo.

---

## 7. Os `[A]` que continuam abertos

| # | item | estado depois desta auditoria |
|---|---|---|
| **A1** | ITW [19] = `atfortes/BokehDiffusion` | segue `[I]`. O paper cita [19] sem URL (`paper.txt:791-793`); autor (*Fortes, A.*, handle `atfortes`) + volume (13.800 contra "13K", `paper.txt:1000`) sustentam. Não fechável a partir do paper |
| **A2** | a máscara sai da AIF, não da bokeh | risco **reduzido**: `mask_iou_aif_bokeh` mediano **0,9928** (oficial) / 0,9423 (nossa). Mas não fecha — o caso do fundo focado é justamente aquele em que as duas concordam |
| **A3** | filtrar `pseudo_aif` | segue `[A]`. **Medido**: 3.010 linhas pré-filtradas por `source_metadata_field_invalid` com detalhe `"pseudo_aif=True"`. Exposto em `--include-pseudo-aif` |
| **A4** | quais são os "13K filtered and verified" | segue `[A]`, não endereçável a partir do paper |
| **A5** | qual `D_focus` alimenta a Eq. 3 | **FECHADO**. Decidido, declarado, e medido em 10⁻⁷ (§1.2) |
| **A6** | resolução de captura sem recorte | risco **alto → baixo**. Lado longo é 1.024 px uniformemente = rendition "Large 1024" do Flickr, com proporção livre ⇒ resize, não recorte. Não verificado contra nenhum original |
| **A7** | largura do sensor via crop factor | risco **alto → médio**. Os 10 modelos mais frequentes com cf=1,0 **são** full-frame (§5.6). A cauda de sensor pequeno e os ~500 restantes não foram auditados |
| **A8** | rejeitar quando a EXIF falta | segue `[A]` (lado conservador). Zero disparos no dado real |
| **A9** | `long_side` da DeblurNet | **FECHADO**: 0, igual ao oficial. Confirmado no `run_config.json` e no log (tiling ligado só acima de 512, igual a `Inference_deblurNet.py:95`) |
| **A10** | faixa de K plausível | segue `[A]`. `--min-k-value`/`--max-k-value` ainda `None`. **Agora há distribuição para escolher**: p01 1,14 · p99 39,4 · máx 47,6 |
| **A11** | limiares dos gates | segue `[A]` **e nenhum está ligado**. §5.4 dá, pela primeira vez, os histogramas para congelar `--min/max-deblur-structural-ssim` e `--max-bokeh-over-aif-sharpness` |
| **A12** | aplicar `k_effective_factor` à rota B | **FECHADO: não aplicar**, e está `None` em 5.432/5.432 |
| **A13** | quanto a DeblurNet desloca a AIF | **FECHADO**: p50 = 0, máx 5,1 px em 5.432 amostras, zero acima de 6 px (§5.5) |
| **A14** | qual DeblurNet os autores usaram | segue `[A]`, e **ficou mais agudo**: o peso publicado é rank **64** e o paper especifica rank **128** para a DeblurNet (`paper.txt:513`) |
| **A15** | bokeh na resolução nativa | segue `[A]`, herda D4 |
| **A16** | **NOVO** — a AIF da nossa variante tem 2,3× menos alta frequência que a da oficial nas mesmas fotos, e é menos nítida que a própria entrada em 77,1% das amostras | **aberto, risco médio**. Não é B1 (adapter provado), e o benchmark da Tab. 2 (`REGISTRO.md:1388-1395`) mostra o nosso **melhor** no RealDOF, o que refuta "lavado". Conferência visual de 20–30 pares fecha (§5.4) |
| **A19** | **NOVO** — o mapa `D_def` difere muito entre as duas variantes com o mesmo K: vão de disparidade cai 2×, `z_max` cai 15× | **aberto**. Não é defeito de nenhuma das duas; é o Depth Pro reagindo a AIFs diferentes. Decide qual lote vai para treino (§5.4) |
| **A17** | **NOVO** — a cópia no HF (`deea4988`, 688 chaves) não é o arquivo com que as `_de4` foram geradas (`a1ed05e4`, 686 chaves) | **aberto, risco de proveniência**. Ou republicar, ou corrigir a citação (§3.2) |
| **A18** | **NOVO** — `D_focus` não tem gabarito no ITW | **aberto**. `focus_mask_sharpness_ratio` mediano 2,34 apoia a hipótese física, mas não testa o caso do fundo focado (§4.3) |

---

## 8. Higiene — pequeno, verificado, e não urgente

1. **`batch_tiles=True` na proveniência não faz nada.** `generate` não declara o
   parâmetro (grep no `flux.py` do cluster: zero ocorrências); o `**params` o engole. O
   metadado afirma um comportamento inexistente em 5.432 amostras (§3.1).
2. **`--deblur-lora-sha256` não está sendo passado** nos runs em voo
   (`docker/run_route_b_h100n1.sh`). A trava existe e está desligada (§3.2).
3. **`genfocus_commit` é `null`** na nossa variante — o checkout não é repositório git. O
   `genfocus_pipeline_sha256` é idêntico ao da oficial, então o `flux.py` é o mesmo.
4. **`flux_backbone_fingerprint` difere entre hosts** com o mesmo snapshot do FLUX. Está
   honestamente rotulado como `paths_sizes_and_small_configs`, mas não é comparável entre
   máquinas (§3.1).
5. **`publish_release.py` não ramifica por rota** (§7.4 de `ROTA_B_DECISOES.md`, ainda
   aberto): `rotas_c` é o único caminho que confere ledger de bytes, o
   `generated_images.jsonl` da rota B não é validado, e `deblur_variant` **não entra na
   linha do manifesto** (`src/dataio/writer.py:140-174`), logo um release misto de variantes
   não seria reprovado por ele — só pelo `_confere_variante_uniforme` do
   `run_route_b.py:400-425`, que age antes e não no publish.
   Também: `sem_validador_analitico` contará 100% na rota B, porque `k_analytic` é `None`
   **por desenho** — é aviso, não reprovação, mas vai assustar quem ler.
6. **`_REQUIRED_PROVENANCE` ainda tem 4 campos** (`src/dataio/sample.py:380`) e não exige
   `k_eq3_diagnostics` nem `deblurnet` (§7.2 de `ROTA_B_DECISOES.md`). Na prática a rota B
   sempre os grava — medido: **0 amostras sem `k_eq3_diagnostics`** em 5.432 —, mas o
   schema não obriga.
7. **`FOCUS_DEPTH_MAX_M = 1000,0` é um `[A]` que BLOQUEIA**
   (`src/control/contract.py:283`, aplicado em `focus_disparity_from_mask:320-321`),
   contrariando a política de `qc/gates.py:3-6`. No ITW ele **não morde** — `focus_depth_m`
   máximo medido é 72,36 m —, então é inócuo hoje. Registrado para não ser esquecido caso a
   rota B rode sobre um corpus com paisagens em hiperfocal.

---

## 9. O que esta auditoria NÃO mediu

Registrado para ninguém confundir com verificação.

* **Nenhuma imagem foi aberta.** Todas as conclusões sobre qualidade da AIF vêm de
  escalares gravados. É exatamente por isso que o `[A]` A16 pede olho humano.
* **Os lotes estão incompletos**: oficial em 52,6% da fatia 0 de 2 (a fatia 1 em 2,5%),
  nossa em ~12% de 13.800. Todas as distribuições são de prefixos do dataset processados em
  ordem de `row_index`, **não** de amostras aleatórias — por isso os 19,33% de
  `crop_factor == 1,0` não podem ser comparados diretamente com a âncora de 30,33%.
* **Nada foi executado em GPU por esta auditoria.** Os quatro containers e os jobs de
  terceiros não foram tocados: só `ls`, `cat`, `wc`, `docker ps`, `docker logs` e `python3`
  de leitura sobre JSON.
* **Os pesos não foram carregados.** O rank foi lido do cabeçalho do `.safetensors`
  (prefixo u64 + JSON), ~94 KB por arquivo.
* **A acurácia de `D_focus` no ITW não foi testada** — não há gabarito (`[A]` A18).
* **A suíte foi executada num interpretador com `pytest` instalado num diretório de
  scratch**, porque o `AKCIT/.venv` indicado não tem `pytest`. Mesmo interpretador, mesmas
  bibliotecas (`numpy`, `PIL`, `cv2`); só o `pytest` veio de fora.
