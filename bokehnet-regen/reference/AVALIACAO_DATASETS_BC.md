# Avaliação estatística dos datasets gerados — rotas B e C

Data: 2026-09-15. Isto **não** é auditoria de código (essas são `ROTA_B_AUDITORIA_2.md` e
`ROTA_C_AUDITORIA_2.md`). Aqui se olha o **dado que saiu**: distribuições, correlações,
patologias, e o que ele ensina.

Tudo foi medido em modo somente-leitura, via `ssh`, com `python3` do stdlib no host e
`numpy`/`PIL` **dentro** dos containers (singularity em h100n3, docker em h100n1). Nada
foi escrito, submetido, cancelado ou apagado em nenhum cluster. Nenhum pacote instalado.

| corpus | host | caminho | n |
|---|---|---|---|
| rota C | dgx-H100-03 | `output/c_release` | 15.423 amostras · 4.399 cenas |
| rota B (nossa, DeblurNet 60k) | dgx-H100-01 | `output/b_release` | 13.615 amostras · 13.615 cenas |
| rota B (DeblurNet oficial) | dgx-H100-03 | `output/b_full_official` | 3.617 |
| rota B (oficial, fatia 1) | dgx-H100-03 | `output/b_full_s1` | 171 |

---

## 0. Veredito

**Servem para treinar — com três correções antes do treino, uma delas obrigatória.**

O que está certo, e é muito: o contrato foi cumprido sem exceção em 29.038 amostras
(`max_coc = 100,0` em 100%, `control_version` único, `depth_backend` único, zero cena nos
dois lados do split, zero duplicata de bytes de origem, zero profundidade degenerada); o
rótulo `K` da rota B bate com um validador **independente** em 95,0% das 181 amostras que
publicam distância de foco na EXIF (razão mediana 1,0071); e a mediana de `K` da rota B,
16,44, cai a 1,0% da âncora 16,6 do contrato.

O que precisa de decisão antes do treino:

1. **A ocupação do mapa de defocus é minúscula nas duas rotas** (§3). Na resolução de
   treino (lado menor 512), a mediana do **máximo** do mapa é **0,0164** na rota B e
   **0,0439** na rota C, de um alcance de [0, 1]. **71,3%** de todos os pixels da rota B
   e **41,1%** dos da rota C ficam abaixo de 0,01. Se esse mapa for codificado em 8 bits
   em qualquer ponto do caminho até o modelo, a mediana da rota B tem **5 níveis
   distintos de 256** (p5 = 2 níveis; 1% das amostras viram mapa constante). Isto é
   consequência direta e prevista do `max_coc = 100` global do contrato — mas **ninguém
   tinha medido quanto custa**, e o custo é que o sinal de controle chega ao modelo como
   uma imagem quase preta. **Este é o achado mais importante desta avaliação.**

2. **Os dois lotes da rota B ensinam física diferente com o mesmo `K`** (§4). Confirmado
   com n = 3.581 pares (a auditoria tinha 1.644): `K` é idêntico (razão mediana 0,999),
   e mesmo assim o CoC máximo cai de 5,30 px para 2,34 px (razão 0,478) porque o vão de
   disparidade cai pela metade e `z_max` cai **14,0x** (98,2 m → 6,99 m). **Um dos dois
   lotes está errado e não sabemos qual.** Misturá-los sem resolver isto ensina que o
   mesmo `K` vale dois borrões diferentes — que é exatamente o que a LVCorr da Tabela 3
   do paper mede.

3. **A rota C não aplica o limiar de SSIM que o paper exige** (§6). `paper.txt:397-399`
   condiciona o uso do `K*` a *"provided that its corresponding SSIM exceeds a predefined
   threshold"*. Não há limiar: 6,82% do release tem `calibration_ssim < 0,80` e 0,14%
   abaixo de 0,50. Pior, a qualidade do ajuste **degrada monotonicamente com K**
   (spearman −0,685), então o limiar ausente deixa passar justamente a cauda alta.

Nenhuma das patologias históricas deste projeto sobreviveu (§5): procurei todas as seis,
com método declarado, e não achei nenhuma.

---

## 1. Rota B — as distribuições

### 1.1 `k_value` (n = 13.615)

```
media=17,35  dp=10,52
p0=0,834  p1=1,32  p5=2,28  p10=3,59  p25=8,60  p50=16,44
p75=25,25  p90=32,63  p95=37,07  p99=39,61  p100=47,95
```

| faixa | n | % |
|---|---|---|
| [0, 2) | 394 | 2,89% |
| [2, 8) | 2.777 | 20,40% |
| [8, 15) | 3.030 | 22,25% |
| [15, 25) | 3.916 | 28,76% |
| [25, 40) | 3.397 | 24,95% |
| [40, 48] | 101 | 0,74% |
| > 48 | 0 | 0,00% |

**Contra as âncoras do `CONTRATO.md`:** mediana 16,44 contra **16,6** (kfix) → −1,0%;
contra **20,1** (EXIF, 318 amostras) → −18,2%; contra **15,0** (default de
`Inference_bokehNet.py:53`) → +9,6%. A âncora principal bate. Os K = {0, 5, 10, 15} da
Fig. 12 do paper (`paper.txt:1154`) ficam no primeiro terço da nossa distribuição — o
paper demonstra controlabilidade numa faixa onde nós temos 30% dos dados.

**Faixa verificada do renderizador.** O laudo do job 32212 varreu `K ∈ {8, 16, 32, 64,
96}`. **23,29% (3.171 amostras) têm K < 8** e **0,00% têm K > 96**. O lado de baixo é
benigno: K pequeno é borrão pequeno, e o laudo mediu linearidade exata (resíduo
0,0000 px em `bokeh_classical`); extrapolar para baixo de um regime linear é seguro.
A rota B **não** usa renderizador de qualquer forma (`provenance.renderer = null` em
13.615/13.615), então a faixa do laudo só importa para a comparabilidade de escala
entre rotas.

**Sem acúmulo, sem bimodalidade, sem valor grudado.** 13.610 valores distintos em
13.615 amostras; o valor mais repetido aparece **2 vezes** (0,01%); **zero** K
exatamente inteiro; **zero** K no teto ou no piso de qualquer busca. Cauda superior
truncada em 47,95 sem acúmulo na borda. Isto é o oposto exato do `k == 300` em 47% do
release v0.

**Validação independente da Eq. 3.** 181 amostras (1,33%) publicam distância de foco na
EXIF, o que permite recalcular `K` sem passar pela máscara nem pela profundidade:

```
k_rotulo / k_validador:  min=0,239  p05=0,975  p25=1,000  p50=1,0071  p75=1,015  p95=1,032  max=1,060
dentro de ±5%:  172/181 = 95,0%      dentro de ±10%:  177/181 = 97,8%
```

Isto é n = 181, contra as 32 da auditoria. É teste **forte** da Eq. 3 e da cadeia
`focal × f-number × pixel_ratio`; é teste **fraco** da Eq. 4, porque o termo `D_focus`
entra na Eq. 3 com sensibilidade `f/(D_focus−f)` ≈ 0,02.

### 1.2 `focus_disparity` e a distância de foco implicada

```
focus_disparity (1/m):  p5=0,115  p25=0,231  p50=0,346  p75=0,524  p95=1,005  p100=4,424
focus_depth_m:          p5=0,995  p25=1,91   p50=2,89   p75=4,33   p95=8,71   p99=15,5  p100=213,7
```

| faixa de distância de foco | % |
|---|---|
| < 0,5 m | 0,62% |
| 0,5 – 2 m | 26,60% |
| 2 – 5 m | 54,02% |
| 5 – 20 m | 18,31% |
| > 20 m | 0,45% |

**Fisicamente plausível.** 80,6% entre 1 e 5 m é exatamente a distribuição de retrato e
foto de rua que a ITW contém. A cauda superior tem **1 amostra a 213,7 m** e 5 acima de
50 m — implausível como plano de foco, mas 0,04% do lote e sem efeito na distribuição.
`focus_disparity` está **dentro de `[disparity_min, disparity_max]` em 100,00%** das
amostras — o plano de foco nunca cai fora da cena.

### 1.3 EXIF — comparação com o que o paper publica

O paper publica as distribuições de focal (Fig. 14), f-number (Fig. 15) e pixel ratio
(Fig. 16) da ITW. **Armadilha resolvida:** a definição válida de pixel ratio é a da
Fig. 16 — *"the image's largest edge length divided by the physical sensor width
(px/mm)"* (`paper.txt:1185-1187`) — e é exatamente a que o pipeline implementa.

```
focal_length_mm:        p5=22    p25=44    p50=60    p75=78    p95=105   p100=200
f_number:               p5=2,0   p25=3,5   p50=5,0   p75=5,9   p95=10    p100=25
pixel_ratio (px/mm):    p5=28,4  p25=28,4  p50=42,62 p75=42,67 p95=78,0  p100=277,3
sensor_width_mm:        p50=24,0 (full-frame 36 em 25%+)
```

`pixel_ratio` mediano **42,62** bate com a âncora 42,6 do `CONTRATO.md`, e a faixa
22,2–277,3 é idêntica à registrada.

**Ressalva que ninguém tinha levantado:** `pixel_ratio` é **fortemente bimodal** —
30,27% em [20, 30) e 59,29% em [40, 50) — e a razão é que o lado longo da imagem é
**1024 px em quase todas** (816 resoluções distintas, mas todas com lado longo ≈1024),
então `pixel_ratio ≈ 1024 / sensor_mm` e o sensor só assume uns poucos valores
(24 mm → 42,67; 36 mm → 28,44). A Fig. 16 do paper é sobre as imagens **originais**. Se
as originais da ITW forem maiores que 1024 px, a nossa distribuição de `pixel_ratio` é
uma versão comprimida da do paper, e com ela a de `K`. Não é defeito de implementação
— o `k_value` está gravado na escala em que foi medido, como manda a regra 3 do
contrato — mas é um **desvio de distribuição** contra a Fig. 16 que não foi conferido.

### 1.4 Profundidade

```
z_min_m:  p5=0,86   p25=1,61   p50=2,39   p75=3,49   p95=6,58   p100=33,1
z_max_m:  p5=1,31   p25=3,40   p50=6,98   p75=13,66  p95=32,4   p99=61,1   p100=10.000
vao de disparidade (dmax−dmin):  p5=0,055  p25=0,132  p50=0,219  p75=0,347  p95=0,668
```

`z_max ≥ 9999 m` (sentinela do Depth Pro) em **1 de 13.615 = 0,01%**. Guardar este
número: na variante oficial ele é 29,5% (§4).

Saúde da quantização (n = 4.539, 1 em cada 3): **zero** amostras com menos de 256 níveis
u16 úteis; mediana de **55.114 valores distintos** de 65.536; u16 mínimo mediano 0 e
máximo mediano 65.533. O defeito do v0 — 24,7% das amostras com a cena útil em menos de
256 níveis — **não existe mais**.

---

## 2. Rota C — as distribuições

### 2.1 `k_value` (n = 15.423)

```
media=36,51  dp=85,46
p0=0,5  p1=0,5  p5=1,29  p10=2,46  p25=5,69  p50=14,54
p75=35,18  p90=74,22  p95=119,0  p99=420,3  p100=960,0
```

A âncora do contrato para a rota C é **3,6 a 36**. O intervalo interquartil medido é
**5,7 a 35,2** — dentro. Mas a cauda vai a 960, que é exatamente `K_ABSOLUTE_MAX`.

| | n | % |
|---|---|---|
| K < 8 (abaixo da faixa do laudo) | 5.119 | 33,19% |
| K em [8, 96] (faixa verificada) | 9.264 | 60,07% |
| **K > 96 (acima da faixa do laudo)** | **1.040** | **6,74%** |
| K ≥ 500 | 123 | 0,80% |
| K == 960 exato (teto absoluto) | 26 | 0,17% |
| **K == 0,5 exato (piso da busca)** | **540** | **3,50%** |

**Duas patologias de borda, com gravidades opostas.**

*No piso:* 540 amostras (3,50%) com `K = 0,5` exato. Todas com
`is_k_censored = True` e portanto `is_valid_for_control = False` — **estão corretamente
marcadas**. O SSIM delas é 0,9968 mediano, o que não é sinal de qualidade: com K ≈ 0 o
render é a própria AIF, e a bokeh de origem nesses casos quase não tem borrão. A
ocupação máxima do mapa delas é **0,0038** — mapa praticamente nulo. 430 cenas
distintas, distribuídas por todos os splits.

*No teto:* 1.040 amostras com K > 96, das quais só **58 estão marcadas como
censuradas**. Ou seja, **982 amostras válidas para controle têm K numa faixa em que o
renderizador nunca foi verificado** — 6,64% do subconjunto válido. E a qualidade do
ajuste cai junto:

| faixa de K | n | SSIM p10 | SSIM p50 | % com SSIM < 0,80 |
|---|---|---|---|---|
| [0, 1) | 661 | 0,980 | 0,9967 | 0,9% |
| [4, 8) | 2.383 | 0,936 | 0,9832 | 2,6% |
| [8, 16) | 3.001 | 0,905 | 0,9683 | 2,7% |
| [16, 32) | 3.075 | 0,861 | 0,9476 | 4,4% |
| [32, 64) | 2.361 | 0,815 | 0,9279 | 8,0% |
| [64, 96) | 827 | 0,770 | 0,9046 | 15,1% |
| **[96, 200)** | **635** | 0,710 | 0,8665 | **29,3%** |
| **[200, 500)** | **282** | 0,671 | 0,7970 | **52,5%** |
| **[500, 961)** | **123** | 0,656 | 0,7599 | **63,4%** |

Correlação global: **spearman(K, calibration_ssim) = −0,685** (n = 15.423). O rótulo
perde confiabilidade exatamente onde ele é mais extremo. As amostras com K ≥ 500 têm
ocupação máxima do mapa **4,93** (ou seja, o mapa satura em 1,0 numa área grande) e vão
de disparidade **0,798** contra 1,208 global — são cenas **frontoparalelas** em que o
sweep empurra K para cima para compensar a falta de vão, exatamente o cenário C4 que a
auditoria da rota C previu e que ninguém tinha medido no lote completo.

**Acúmulo em valores de busca, não em valores redondos.** Só 2.629 valores distintos
para 15.423 amostras: 92,89% das amostras compartilham `K` com outra. Isso é a seção
áurea convergindo em pontos de grade, não K grudado num valor semântico — o valor
individual mais frequente depois de 0,5 aparece 52 vezes (0,34%). Apenas 40 amostras
(0,26%) têm K exatamente inteiro.

### 2.2 `focus_disparity` e distância de foco

```
focus_disparity (1/m):  p5=0,162  p25=0,482  p50=0,792  p75=1,205  p95=2,090  p100=8,273
focus_depth_m:          p5=0,478  p25=0,830  p50=1,263  p75=2,077  p95=6,161  p99=16,4  p100=426,6
```

35,6% das cenas focam a menos de 1 m e 72,8% a menos de 2 m — coerente com a RealBokeh
(macro e produto em estúdio), e **muito mais perto** que a rota B (mediana 2,89 m).
`focus_disparity` dentro de `[disparity_min, disparity_max]` em **100,00%**. 12 amostras
(0,08%) com plano de foco acima de 50 m; implausível, irrelevante em massa.

### 2.3 Profundidade — o problema do fundo

```
z_min_m:  p5=0,32   p25=0,53   p50=0,78   p75=1,15   p95=2,07
z_max_m:  p5=2,58   p25=7,27   p50=19,5   p75=208,4  p90=10.000  p95=10.000
```

**20,02% das amostras têm `disparity_min = 1,0e-4` exato** — isto é `z_max = 10.000 m`,
a sentinela do Depth Pro. 22,84% têm `z_max > 1.000 m`; 33,66% acima de 50 m. Não é
defeito: é céu, e o vão de disparidade é justamente o que reconstrói o mapa. O efeito
prático foi medido e é desprezível: `quantization_coc_error_px_at_depth_hw` mediano
**5,1e-5 px**, p99 **1,4e-3 px**.

Saúde da quantização (n = 5.141): **zero** amostras com menos de 256 níveis u16; mediana
de **52.038 valores distintos**.

### 2.4 `calibration_ssim` — a qualidade do ajuste da Eq. 5

```
media=0,9356  dp=0,0770
p0=0,287  p1=0,641  p5=0,771  p10=0,837  p25=0,917  p50=0,9633  p75=0,986  p95=0,997  p100=0,9997
```

| | n | % |
|---|---|---|
| SSIM ≥ 0,95 | 9.256 | 60,01% |
| SSIM < 0,90 | 3.048 | 19,76% |
| SSIM < 0,85 | 1.756 | 11,39% |
| **SSIM < 0,80** | **1.052** | **6,82%** |
| SSIM < 0,70 | 356 | 2,31% |
| SSIM < 0,50 | 22 | 0,14% |

**Nenhum limiar é aplicado.** O paper exige um (`paper.txt:397-399`). As 22 amostras com
SSIM < 0,50 são rótulos que o próprio critério do paper chamaria de não confiáveis, e
estão no release com `is_valid_for_control = True`.

### 2.5 `focus_source` — a composição e o viés do refinamento

| `focus_source` | n | % | K p25 | **K p50** | K p75 | K p90 | SSIM p50 |
|---|---|---|---|---|---|---|---|
| `birefnet` (não refinada) | 7.381 | 47,86% | 5,66 | **14,47** | 33,81 | 65,80 | 0,9644 |
| `retention_only` | 5.460 | 35,40% | 5,43 | **13,57** | 34,08 | 77,06 | 0,9658 |
| `birefnet_refined` | 2.582 | 16,74% | 6,37 | **17,45** | 42,02 | 95,49 | 0,9528 |

`focus_was_refined = True` em 8.042 = **52,14%**.

**O refinamento enviesa o rótulo, e o viés é mensurável e cresce com o quantil.** A
classe `birefnet_refined` tem mediana de K **+20,6%** acima da não refinada (17,45 vs
14,47), p75 **+24,3%** e p90 **+45,1%**. Não é ruído: são 2.582 amostras. E o SSIM delas
é *pior* (0,9528 vs 0,9644), o que exclui a leitura benigna de "o refinamento achou o
plano certo e por isso o K ficou maior" — se fosse isso, o ajuste teria melhorado.
O mecanismo é direto e está nos números: refinar **afasta** o plano de foco —
`focus_disparity` mediano cai de 0,881 (não refinada) para 0,781 (refinada), isto é a
distância de foco sobe de 1,136 m para 1,280 m —, o que reduz `|disp − focus_disp|` no
primeiro plano e exige K maior para produzir o mesmo borrão no fundo. A ocupação máxima
mediana do mapa acompanha: **0,159** nas refinadas contra **0,122** nas não refinadas
(+30%).

`retention_only` (35,40% do lote) é o caso em que o BiRefNet foi descartado inteiro e a
região saiu só da retenção de nitidez: `mask_area_ratio` dessas é **exatamente 0,05**
(o `top_fraction` da configuração) — daí o pico em p25 = p50 = 0,05 na distribuição
global de `mask_area_ratio`.

### 2.6 A estrutura por cena, e o que o paper diz dela

4.399 cenas, 3,51 amostras por cena, máximo 4:

| amostras na cena | cenas |
|---|---|
| 1 | 2 |
| 2 | 756 |
| 3 | 655 |
| 4 | 2.986 |

Isto **bate com o paper**: *"focus-consistent series captured with varying apertures,
containing 2 to 4 images per set"* (`paper.txt:1002-1004`). Os índices de abertura
usados são 1, 2, 4 e 5 na maioria (e 3, 8, 14, 21 numa minoria), de um total de até 21
que a origem oferece.

**Mas duas propriedades que a frase do paper implica não se sustentam no dado.**

*(a) O `K` não respeita a ordem das aberturas.* Dentro de uma cena, todas as amostras
compartilham a mesma AIF (verifiquei: **0 de 4.399 cenas** têm sha256 de AIF diferente
entre níveis), a mesma profundidade e o mesmo plano de foco físico; só a abertura muda.
`K` deveria crescer monotonicamente com o índice de abertura. Medido:

```
cenas com K monotonicamente crescente no indice:   718 / 4.397 = 16,3%
esperado sob ordem ALEATORIA (dado o mix 2/3/4):   612 / 4.397 = 13,9%
cenas com K monotonicamente decrescente:           539 / 4.397 = 12,3%
tau de Kendall (indice x K) dentro da cena:  media=+0,073   mediana=0,000
```

16,3% contra 13,9% de acaso. O `K*` da Eq. 5 é **praticamente não ordenado** com respeito
à abertura da foto que ele deveria descrever. A razão máx/mín de `K` dentro da mesma cena
tem mediana **7,05**, p90 **30,0** e máximo **1920**.

*(b) O plano de foco não é constante na série "focus-consistent".* Mesma AIF, mesmo
sha256, e ainda assim:

```
razao max/min de focus_disparity dentro da cena:  p50=1,0078  p90=1,594  p99=7,97  max=482,3
cenas com focus_disparity identico em todos os niveis:  1.817 / 4.397 = 41,3%
cenas com variacao > 2x:                                  329 / 4.397 =  7,5%
cenas com focus_source diferente entre niveis:            823 / 4.397 = 18,7%
```

Restringindo às 3.574 cenas em que `focus_source` é **constante** entre os níveis, o
mecanismo fica evidente:

| `focus_source` da cena | cenas | razão máx/mín p50 | % com valor idêntico |
|---|---|---|---|
| `birefnet` | 1.808 | 1,0000 | **100,0%** |
| `birefnet_refined` | 395 | 1,0250 | 1,5% |
| `retention_only` | 1.371 | 1,0431 | 0,2% |

O caminho não refinado é determinístico, como tem de ser. **É o refinamento que
introduz a variação**, e introduz porque usa a retenção de nitidez entre AIF e bokeh —
e a bokeh muda a cada nível. Isso é coerente com o código, mas o resultado é que a
mesma cena, fotografada com o mesmo foco, recebe até 482 planos de foco diferentes.
A grandeza refinada é uma propriedade da **foto borrada**, não da cena, e é usada como
se fosse da cena.

---

## 3. A ocupação do mapa de defocus — o número que faltava

Medido abrindo os PNGs de profundidade, decodificando
`disp = disparity_min + u16/65535 · (disparity_max − disparity_min)` e aplicando a
definição do contrato `defocus = clip(|K·(disp − focus_disp)| / 100, 0, 1)`. Amostragem
de 1 em cada 3: **n = 4.539 (rota B, 1.807 Mpx)** e **n = 5.141 (rota C, 2.274 Mpx)**.

Reportado em duas escalas, porque as duas existem: `k_value` está gravado na escala da
**imagem original** (regra 3 do contrato), mas o modelo vê o crop de **512** de lado
menor, com `K' = K · 512/min(H,W)`.

### 3.1 Histograma agregado de pixels

Fração de **todos os pixels** do lote em cada faixa do mapa:

| faixa | **B** (escala imagem) | **B** (escala 512) | **C** (escala imagem) | **C** (escala 512) |
|---|---|---|---|---|
| [0,00 – 0,01) | 64,93% | **71,30%** | 22,04% | **41,10%** |
| [0,01 – 0,05) | 30,17% | 26,26% | 31,54% | 36,41% |
| [0,05 – 0,10) | 4,08% | 2,19% | 15,57% | 12,72% |
| [0,10 – 0,25) | 0,81% | 0,25% | 18,65% | 8,44% |
| [0,25 – 0,50) | 0,01% | 0,00% | 8,91% | 0,97% |
| [0,50 – 1,00) | 0,00% | 0,00% | 2,70% | 0,22% |
| **== 1,00 (saturado)** | **0,00%** | **0,00%** | **0,59%** | **0,13%** |

### 3.2 Por amostra

| grandeza | **rota B**, escala 512 | **rota C**, escala 512 |
|---|---|---|
| máximo do mapa, p25 · **p50** · p75 · p95 | 0,0085 · **0,0164** · 0,0297 · 0,0604 | 0,0164 · **0,0439** · 0,0959 · 0,2634 |
| média do mapa, p50 | 0,0062 | 0,0204 |
| desvio padrão do mapa, p50 | 0,0045 | 0,0124 |
| fração de pixels < 0,01, p50 | 0,775 | 0,304 |
| fração de pixels < 0,001, p50 | 0,166 | 0,042 |
| **fração de pixels saturados em 1,0** | **0,000 em todos os percentis** | p50 = 0,000; p99 = 0,0069 |
| amostras que alcançam 1,0 em algum pixel | **0 / 4.539 = 0,00%** | 69 / 5.141 = **1,34%** |
| amostras que alcançam ≥ 0,5 | 0 = 0,00% | 118 = 2,30% |
| amostras que alcançam ≥ 0,1 | 43 = 0,95% | 1.218 = 23,69% |
| **níveis distintos se gravado em uint8**, p5 · **p50** · p95 | 2 · **5** · 16 | 2 · **12** · 68 |
| amplitude em níveis uint8 (`max·255`), p50 | **4,2** de 255 | **11,2** de 255 |

Na escala da imagem original os números sobem, mas não mudam a conclusão: rota B máximo
mediano **0,0221** (p95 = 0,081; **nenhuma** amostra chega a 0,5), rota C máximo mediano
**0,1285** (p95 = 0,772; 3,35% saturam em algum pixel).

### 3.3 O que isso significa

**Nenhum dos dois lotes está "quase todo saturado".** A rota B **nunca** satura: zero
pixels em 1,807 bilhões. A rota C satura 0,59% dos pixels na escala da imagem e 0,13%
na escala de treino. O modo de falha "o mapa é tudo 1,0 e não ensina modulação" **não
existe aqui**.

O modo de falha oposto existe, e é sério. Na escala de treino, **71,3% dos pixels da
rota B e 41,1% dos da rota C ficam no primeiro centésimo de [0, 1]**. A mediana da rota
B usa **1,6% do alcance** do sinal; a rota C, **4,4%**. Toda a informação de modulação
de CoC vive num intervalo mais estreito que o passo de quantização de um uint8.

A predição do `CONTRATO.md` (regra 2) era *"a rota B vai ocupar [0, 0,05] e a rota C
[0, 0,4]"*. Medido na escala da imagem: **a rota B cumpre** — 84,51% das amostras têm o
máximo do mapa abaixo de 0,05, e as 15,49% que passam vão só até 0,314 no pior caso.
**A rota C não chega perto de [0, 0,4]**: a mediana do máximo é 0,128, só 35,44% das
amostras alcançam 0,2 e 10,82% alcançam 0,5 — a previsão era otimista por um fator ~3.

E a razão de escala entre as rotas, que o contrato chama de "física e é o que o modelo
tem que aprender", é de **2,7x** na mediana do máximo (0,0439 / 0,0164 na escala de
treino), não a ordem de grandeza que [0,05] vs [0,4] implicava.

**Consequências operacionais, em ordem de urgência:**

1. **O mapa não pode passar por uint8 em ponto nenhum do dataloader.** Mediana de 5
   níveis distintos na rota B; 1% das amostras dá mapa **constante** (1 nível). Se ele
   for para o VAE do FLUX como imagem de 3 canais em 8 bits, o sinal de controle é
   ruído de quantização. `contract.defocus_map` devolve `float32` — o caminho até o
   modelo precisa preservar isso ponta a ponta, e isso **não foi verificado** (o
   dataloader de treino não existe neste repositório).
2. **O `max_coc = 100` global é defensável e é o custo de ter uma única convenção.** Não
   recomendo mudá-lo — normalizar por rota é o defeito D1 com granularidade mais grossa,
   e foi assim que se chegou ao `max_coc = 10,5107` do kfix. Mas a decisão precisa ser
   tomada **com este número na mesa**, e até agora não estava.
3. Se a escolha for manter, vale registrar no README dos releases que o alcance
   efetivo do canal de controle é [0, ~0,05] e não [0, 1], para que quem consumir o
   dataset não normalize por conta própria.

---

## 4. A comparação pareada entre as duas variantes da rota B

**n = 3.581 pares** com o mesmo `sample_id` em `b_release` (nossa, `ours_main_cond`,
LoRA `a1ed05e4`) e em `b_full_official` (`official_cond_only`, LoRA `8d4c3960`). A
auditoria fez isto com 1.644; este n é 2,2x maior. As 171 amostras de `b_full_s1` são
**bit a bit idênticas** às correspondentes de `b_full_official` (0 de 171 com `K` ou
`z_max` diferente) — é o mesmo corpus fatiado, não uma terceira variante.

### 4.1 `K` é o mesmo. Tudo o mais não é.

| grandeza | nossa p50 | oficial p50 | razão nossa/oficial: p25 · **p50** · p75 |
|---|---|---|---|
| `k_value` | 16,91 | 16,92 | 0,994 · **0,999** · 1,004 |
| `focus_disparity` | 0,3455 | 0,3693 | 0,725 · **0,937** · 1,243 |
| `disparity_max` (ponto mais perto) | 0,4193 | 0,5198 | 0,602 · **0,818** · 1,095 |
| `disparity_min` (ponto mais longe) | 0,1431 | 0,0102 | 3,54 · **14,9** · 625 |
| **vão de disparidade** | **0,2233** | **0,4956** | 0,295 · **0,481** · 0,703 |
| **`z_max_m`** | **6,99 m** | **98,20 m** | 0,0016 · **0,067** · 0,283 |
| **CoC máximo (px)** | **2,34** | **5,30** | 0,280 · **0,478** · 0,711 |

`|ΔK|/K` mediano = **0,51%**, p95 = 3,3%, máximo 18,3%. O `K` é o mesmo rótulo.

### 4.2 Confirmação do achado da auditoria

A auditoria mediu, com n = 1.644, `z_max` de **103,61 m → 7,24 m** (14,3x nas medianas)
e CoC máximo de **5,333 px → 2,433 px**. Com n = 3.581 eu meço **98,20 m → 6,99 m**
(14,0x) e CoC máximo **5,30 px → 2,34 px**. **Confirmado**, com desvio de 5% no valor
absoluto e desvio nulo na conclusão. A razão por amostra é ainda mais severa que a razão
das medianas: mediana das razões de `z_max` = 0,067, ou seja **15,0x**.

Média: `z_max` oficial 3.131 m contra 12,9 m da nossa — 243x na média, porque a oficial
bate na sentinela de 10.000 m com frequência e a nossa nunca bate.

### 4.3 Não é escala global, é achatamento da cena

Decodifiquei o perfil completo de profundidade (todos os pixels) de 717 pares:

| quantil de z | nossa p50 (m) | oficial p50 (m) | razão nossa/oficial p50 |
|---|---|---|---|
| z_p1 | 2,328 | 1,971 | **1,166** |
| z_p5 | 2,410 | 2,201 | 1,132 |
| z_p25 | 2,813 | 2,755 | 1,045 |
| z_p50 | 3,715 | 5,627 | 0,758 |
| z_p75 | 4,972 | 13,661 | 0,365 |
| z_p95 | 5,834 | 29,802 | 0,199 |
| z_p99 | 6,229 | 41,449 | **0,151** |
| z_max | 6,516 | 93,500 | 0,061 |

**Se o Depth Pro estivesse apenas estimando outra focal e reescalando a métrica, a razão
seria a mesma em todos os quantis.** Ela varia por um fator de **7,74x** entre p1 e p99,
e **inverte de sinal**: no primeiro plano a nossa cena está 17% mais **longe**; no fundo,
6,6x mais **perto**. Só **10,2%** dos pares (73 de 717) têm razão consistente entre p50
e p99 dentro de ±25%. Não é escala — é **compressão do alcance de profundidade**.

Fração de pixels no fundo distante:

| | nossa (média) | oficial (média) |
|---|---|---|
| z > 20 m | 2,04% | 19,49% |
| z > 50 m | 0,14% | 9,35% |
| z > 100 m | 0,00% | 6,34% |
| z > 1000 m | 0,00% | 3,59% |

E o corte categórico: **`z_max ≥ 9999 m` (a sentinela do Depth Pro) em 219 de 717 =
30,5% das oficiais, e em 0 de 717 = 0,0% das nossas.** Zero. Não é um deslocamento
gradual da distribuição; é um regime do Depth Pro que a nossa AIF nunca aciona.

### 4.4 A AIF: 2,2x menos energia de alta frequência — confirmado, e decomposto

Medi direto nos JPEGs, n = 717 pares, com a mesma métrica dos gates (luma BT.601,
Laplaciano de 4 vizinhos, variância global) **e** com decomposição espectral por FFT em
bandas radiais (janela de Hann, recorte quadrado de até 1024 px):

| métrica | nossa p50 | oficial p50 | razão p50 | inverso |
|---|---|---|---|---|
| **variância do Laplaciano** | 376,4 | 1.010 | **0,456** | **2,20x** |
| energia relativa r ≥ 0,50 Nyq | 0,00829 | 0,01582 | 0,588 | 1,70x |
| energia relativa 0,30–0,50 Nyq | 0,01435 | 0,02443 | 0,621 | 1,61x |
| energia relativa 0,15–0,30 Nyq | 0,02049 | 0,04563 | 0,454 | 2,20x |
| energia relativa 0,05–0,15 Nyq | 0,07997 | 0,1229 | 0,657 | 1,52x |
| **energia relativa r < 0,05 Nyq** | 0,8715 | 0,7837 | **1,111** | — |
| energia **absoluta** r ≥ 0,30 Nyq | 4,62e6 | 9,56e6 | 0,534 | 1,87x |
| gradiente médio \|dI\| | 5,74 | 7,81 | 0,820 | 1,22x |
| desvio padrão da luminância | 57,7 | 61,3 | 0,940 | 1,06x |
| **bytes do JPEG q95** | 365,6 kB | 357,8 kB | **1,046** | — |

**A queda de 2,3x da auditoria está confirmada: 2,20x pela mediana das razões**
(2,68x pela razão das medianas). A decomposição espectral mostra que a perda é
monotônica e concentrada acima de 0,15 Nyquist, com **ganho** de 11% de energia
relativa na banda baixa — a assinatura de um filtro passa-baixa.

Mas o JPEG q95 da nossa AIF é **4,6% maior**, não menor. Uma imagem simplesmente
borrada comprime **menos**. Isso sustenta a leitura já registrada no `[A]` A16 (e apoiada
pelo benchmark da Tabela 2, em que o nosso 60k **ganha** no RealDOF: LPIPS 0,2291 contra
0,2397 do oficial): não é "lavagem", é troca de textura de ruído/artefato de JPEG por
estrutura mais limpa. A métrica de Laplaciano não distingue as duas coisas.

### 4.5 A resposta: é o Depth Pro, e não pelo borrão

A pergunta era se a queda de `z_max` é a DeblurNet borrando ou o Depth Pro reagindo.
O teste é a correlação pareada entre as duas quedas, n = 717:

```
spearman( log(razao de lapvar) , log(razao de z_max) )                   = +0,068
spearman( log(razao de energia HF>0,5 Nyq) , log(razao de z_max) )        = +0,072
spearman( log(razao de lapvar do quarto superior) , log(razao de z_max) ) = −0,007
spearman( log(razao de lapvar) , log(razao do vao de disparidade) )       = +0,152
```

**Essencialmente zero.** As amostras em que a nossa AIF perdeu mais alta frequência
**não** são as amostras em que a profundidade encolheu mais. Confirmação por
estratificação: nos 369 pares em que o Depth Pro oficial deu `z_max ≤ 100 m` (sem
sentinela de céu, portanto sem o efeito de saturação), a razão de `z_max` ainda é
**0,276** (3,6x menor) e a razão de Laplaciano nesse subgrupo é **0,471** — idêntica à
do subgrupo com céu (0,460). A perda de nitidez é uniforme; a compressão de
profundidade é uniforme; e as duas não se correlacionam.

**Conclusão: é o Depth Pro reagindo ao conteúdo da AIF, e o canal não é o borrão.** Os
pesos do Depth Pro são os mesmos nos dois lotes (`depth_model_sha256 = 3eb35ca68168` em
ambos), a geometria de entrada é a mesma (574×1024 → 576×1024, `no_crop_multiple_of_16`,
`crop_applied: false` nos dois), e o commit do pipeline não explica nada: dentro do
próprio `b_full_official`, os dois commits presentes (`c001b5acf4a2`, n = 3.453, e
`e49f20b5f492`, n = 164) dão taxas de sentinela de **29,5% e 29,3%** — idênticas —
contra **0,01%** do `b_release` (`3509db023a07`). A variável que muda é a imagem.

O mecanismo mais provável é que a nossa DeblurNet reconstrói o fundo distante (céu,
horizonte, névoa) com textura suficiente para o Depth Pro atribuir-lhe profundidade
finita e próxima, em vez de reconhecê-lo como plano infinito. Isso é testável abrindo
20–30 pares — é a ação já registrada em A16 e ainda pendente.

### 4.6 O que isso significa para o treino

**Os dois lotes não ensinam a mesma física.** Com o mesmo `K`, o `D_def = K·|D − D_focus|`
que a BokehNet consome tem **metade** do alcance de CoC no nosso lote (2,34 px contra
5,30 px de máximo mediano). Como o alvo de supervisão é a **mesma foto real** nos dois
casos, um dos dois mapas está descrevendo errado o borrão que a foto tem. Treinar com
os dois misturados ensina que um mesmo valor de `D_def` corresponde a dois borrões
diferentes — que é literalmente a definição de perda de controlabilidade.

Não dá para decidir qual está certo sem medir o borrão real na foto de origem
(por exemplo, ajustando o raio de disco que melhor explica o fundo da bokeh e comparando
com `K·Δdisp` de cada lote). **Isso não foi feito e é o teste que falta.** Enquanto não
for feito, a recomendação é **não misturar**: escolher um dos dois lotes para o treino.

Nota lateral de proveniência: `mask_model_sha256` também difere entre as variantes
(`c38d5beb` na nossa, `b30edcc5` na oficial). Isso explica parte da divergência de
`focus_disparity` (razão p50 = 0,937, mas p05 = 0,470 e p95 = 2,43) e **não** explica
`z_max`, que não depende da máscara.

---

## 5. As patologias históricas — procurei todas, com método

| patologia | método de busca | resultado |
|---|---|---|
| **K grudado num valor exato** (`k == 300` em 47% do v0) | `Counter` sobre `round(k_value, 6)` nos 29.038 metadados | **Rota B: limpa.** 13.610 valores distintos em 13.615; valor mais repetido aparece 2x (0,01%); 0 inteiros exatos. **Rota C: 540 em `K = 0,5` (3,50%)**, que é o piso da busca, **todas marcadas** `is_k_censored = True` e `is_valid_for_control = False`; e 26 em `K = 960` (0,17%), o teto absoluto. A censura é **detectada e gravada**, que era o conserto pedido. |
| **profundidade que é disparidade normalizada disfarçada** | decodifiquei 9.680 PNGs e comparei o perfil de `z` reconstruído com `z_min_m`/`z_max_m` do metadado; contei níveis u16 úteis | **Não encontrada.** `depth_encoding = "uint16_linear_in_disparity"` em 100%; perfil de `z` natural (denso perto, esparso longe) nas duas rotas; **zero** amostras com < 256 níveis u16 (B: mediana 55.114 níveis; C: 52.038); `depth_backend = "depth_pro"` em 29.038/29.038, sem cascata para Depth Anything. |
| **`max_coc != 100`** | leitura do campo em todos os metadados | **Não encontrada.** `max_coc = 100,0` em **29.038 de 29.038**. Nenhum outro valor aparece. |
| **cena nos dois lados do split** | cruzei `scene_id` de cada amostra com o `assignment` de `split.json` | **Não encontrada.** Rota B: 0 de 13.615 cenas em mais de um split (train 12.968 / val 647). Rota C: 0 de 4.399 (train 13.796 / val 1.627). O split é **materializado por cena**, então o vazamento é estruturalmente impossível — e confirmei que nenhum `scene_id` do metadado ficou sem atribuição. |
| **amostras duplicadas** | `Counter` sobre `sample_id`, `source_sample_id` e os sha256 de `source_images.jsonl` | **Não encontrada.** Rota B: **0** sha256 de origem repetido em 13.800 linhas. Rota C: **0** `bokeh_sha256` repetido em 15.427; o `aif_sha256` repete 2–4x **por construção** (os níveis de uma cena compartilham a AIF) e **nenhuma AIF é compartilhada entre cenas diferentes** — o que também confirma que a chave de cena `scene_key(split, numero)` resolveu a colisão de numeração da RealBokeh. |
| **K correlacionado com o que não deveria influenciá-lo** | spearman e pearson de `k_value` contra resolução, área, aspect ratio, posição no shard e modelo de câmera | **Rota B, nada espúrio:** lado longo **−0,038**, lado curto **−0,048**, área **−0,050**, aspect ratio **+0,046**, posição na ordem dos arquivos **−0,023**. As correlações que **devem** existir, existem: focal **+0,781** (K ∝ f²) e f-number **−0,114**. Entre os 46 modelos de câmera com n ≥ 50, a mediana de K varia 3,8x (7,53 a 28,94) — isso é sensor e lente, não defeito. **Rota C:** resolução é constante (1500×2000 em 100%), então não há o que correlacionar; posição na ordem **+0,012**. |

**Duas correlações que existem e merecem registro** (não são patologias, são propriedades
do rótulo):

- `K` × vão de disparidade: **−0,336** (B) e **−0,329** (C). Cena com pouco relevo exige
  K maior para o mesmo borrão. Na rota C é o mecanismo que produz a cauda acima de 96.
- `K` × `calibration_ssim`: **−0,685** (C). Já discutido em §2.1.

### 5.1 Uma patologia nova, de publicação

O `README.md` do `b_release` está **trocado**. O cabeçalho diz `pretty_name: BokehNet
regen — rota C`, o título diz `# juliadollis/bokehnet-regen-rota-b`, e o corpo descreve
a **rota C**: *"§3.2(c) — rota C: pares reais (all-in-focus, bokeh), com o sinal de
controle calibrado pelo sweep da Eq. 5"*, *"a origem entrega de 2 a 21 aberturas da
mesma cena"*, e uma linha "K censurado no teto". Nenhuma dessas frases descreve o
`b_release`, que tem `k_source = "eq3_exif"` em 100%, `calibration_ssim = null` em 100%,
`renderer = null` em 100%, `is_k_censored = false` em 100% e **exatamente 1 amostra por
cena** (13.615 cenas para 13.615 amostras — a própria tabela do README diz isso, ao lado
do texto que afirma o contrário). Os números da tabela são da rota B; a prosa é da rota
C. Há uma publicação em andamento (`pub_rota_b`); este README vai junto.

### 5.2 Proveniência mista no corpus oficial

`b_full_official` foi gerado por **dois `pipeline_commit` diferentes**: `c001b5acf4a2`
(3.453 amostras) e `e49f20b5f492` (164). `b_full_s1` tem `e49f20b5f492` (164) e
`5fcde487ed8f` (7). `b_release` e `c_release` são homogêneos (1 commit cada). Verifiquei
que a mistura **não afeta** as grandezas medidas (sentinela de `z_max` em 29,5% vs 29,3%;
`K` mediano 16,94 vs 16,38), mas o corpus oficial não é um artefato de uma versão só, e
isso precisa estar declarado se ele for usado como referência.

---

## 6. O que o paper pede sobre os dados e ainda não foi verificado

Leitura de §3.2 (`paper.txt:308-402`), §3.5/§4.1 (`paper.txt:476-530`) e do suplemento
B.2 (`paper.txt:985-1011`).

| # | o que o paper afirma / exige | onde | estado |
|---|---|---|---|
| 1 | *"provided that its corresponding SSIM exceeds a predefined threshold to ensure reliable supervision"* — a Eq. 5 só vira rótulo se o SSIM passar de um limiar | `paper.txt:397-399` | **NÃO IMPLEMENTADO.** Nenhum limiar aplicado ao `c_release`. 6,82% com SSIM < 0,80, 2,31% < 0,70, 0,14% < 0,50 — todas com `is_valid_for_control = True`. O paper não publica o valor do limiar, mas publica que existe um. |
| 2 | *"approximately 26K real examples sourced from ITW dataset [19], RealBokeh [57], and **LFDOF [52]**"* | `paper.txt:527-530` e `paper.txt:356` | **COMPOSIÇÃO DIVERGENTE.** Temos 13.615 (ITW) + 15.423 (RealBokeh) = **29.038**, contra ~26K. E **LFDOF está ausente**, embora seja citada nas duas passagens. O total bate por acidente: sobra RealBokeh onde falta LFDOF. |
| 3 | *"13K previously filtered and verified images from the ITW dataset, alongside 13K images newly curated"* | `paper.txt:999-1002` | **PARCIAL.** ITW: 13.615 ≈ 13K ✓. RealBokeh: 15.423, não 13K (+19%). O que significa "filtered and verified" na ITW continua sem definição — é o `[A]` A4, aberto. |
| 4 | *"focus-consistent series captured with varying apertures, containing **2 to 4 images per set**"* | `paper.txt:1002-1004` | **A CONTAGEM BATE, A CONSISTÊNCIA NÃO.** 2 a 4 por cena em 4.397 de 4.399 (2 cenas com 1). Mas, medido em §2.6: o plano de foco varia dentro da série em 58,7% das cenas (>2x em 7,5%, máx 482x), e o `K` não respeita a ordem das aberturas (16,3% monotônico contra 13,9% de acaso, tau de Kendall +0,073). O paper trata a série como uma unidade fisicamente consistente; o nosso rótulo não a trata assim. |
| 5 | *"we first conducted a manual verification and refinement process on the in-focus masks... 4 to 8 seconds per image, approximately 8 hours"* | `paper.txt:1004-1008` | **SUBSTITUÍDO POR HEURÍSTICA, COM VIÉS AGORA MEDIDO.** Nosso refinamento é automático (`top_fraction = 0,05`, `agreement_floor = 0,3`, `window_px = 33`, todos `[A]` — *"nenhum destes valores vem do paper"*). Cobre 52,14% do lote. §2.5 mede o viés: K mediano +20,6% e p90 +45,1% nas refinadas, com SSIM **pior**. Nunca se comparou o nosso refinamento com o manual — não há gabarito. |
| 6 | Fig. 16: distribuição de pixel ratio da ITW | `paper.txt:1185-1187` | **DEFINIÇÃO CONFERIDA, DISTRIBUIÇÃO NÃO.** A fórmula `max(H,W)/sensor_mm` é a da Fig. 16 (e não a da numeração antiga das legendas da Fig. 4) ✓. Mas a nossa distribuição é bimodal em 28,4 e 42,6 porque o lado longo é ~1024 em quase tudo; a do paper é sobre originais. **Nunca comparada com a figura.** Mesma lacuna para focal (Fig. 14), f-number (Fig. 15) e modelos de câmera (Fig. 13) — temos 426 modelos, o paper mostra os 30 primeiros. |
| 7 | Eq. 2 `D_def = K·|D − D_focus|`, **sem normalizador** | `paper.txt:310-312` | **DESVIO DECLARADO, AGORA QUANTIFICADO.** O paper não tem `max_coc`; nós dividimos por 100 e saturamos. §3 mede o preço: o mapa ocupa 1,6% (B) e 4,4% (C) do alcance. O paper também não diz em que espaço `D` vive; o `CONTRATO.md` resolve por autoridade de `Inference_bokehNet.py`. |
| 8 | ~70K pares sintéticos (rota A) | `paper.txt:527-528` | **FORA DO ESCOPO DESTA AVALIAÇÃO**, mas registre-se: sem a rota A, o currículo de §4.1 (*"40K steps on synthetic data"* antes de *"60K on real"*) não pode ser reproduzido, e é nesse estágio que o modelo aprende a modular CoC. |
| 9 | Resolução de treino | ausente no paper; §3.5 fala de tiling **só na inferência** (`paper.txt:481-482`) | **DECISÃO NOSSA (crop 512), E É ELA QUE PIORA §3.** Reescalar para 512 multiplica `K` por `512/min(H,W)` < 1, encolhendo o mapa mais 25–35%. Medido: máximo mediano cai de 0,0221 para 0,0164 na rota B e de 0,1285 para 0,0439 na rota C. |
| 10 | Precisão de `D_focus` (Eq. 4) na ITW | `paper.txt:345-352` | **SEM GABARITO, SEM TESTE.** É o `[A]` A18. O caso de falha não coberto continua: fotógrafo focou o fundo, BiRefNet acha o objeto saliente, as duas concordam, IoU alta, plano errado. Os 181 validadores de EXIF de §1.1 testam a Eq. 3, não a Eq. 4. |

---

## 7. Método, e os n

- Metadados: li os 29.038 `meta/<id>.json` inteiros (não só o `manifest.jsonl`), via
  `python3` do stdlib no host, sem escrever nada no cluster. Toda estatística de
  distribuição, correlação e categoria vem daí, com n completo.
- Mapa de defocus e saúde da quantização: abri os PNGs de profundidade de **1 em cada 3**
  amostras — **n = 4.539** (rota B) e **n = 5.141** (rota C), 4.081 Mpx no total —
  dentro dos containers, com `numpy`/`PIL`. A decodificação usa
  `disp = disparity_min + u16/65535·(disparity_max − disparity_min)`, que é exatamente
  `dataio.encoding.decode_disparity`.
- Comparação pareada de metadados: **n = 3.581** (interseção de `sample_id` entre
  `b_release` e `b_full_official`).
- Comparação pareada de imagens e de perfil de profundidade: **n = 717** (1 em cada 5 dos
  pares), medidos em cada host separadamente e cruzados localmente — nenhum byte de
  imagem transitou entre clusters.
- Validador independente da Eq. 3: **n = 181** (as amostras da rota B que publicam
  distância de foco na EXIF).
- Correlações: spearman e pearson implementados em stdlib, com tratamento de empates
  por posto médio. Tau de Kendall por contagem exata de pares concordantes.
- Nenhum job foi submetido, cancelado ou consultado; nenhum container de terceiros foi
  tocado; nenhum arquivo foi criado, movido ou apagado em `dgx-H100-01` ou `dgx-H100-03`.
  Os scripts foram entregues ao interpretador por `stdin` (`ssh host 'python3 -' <
  script.py`), sem materializar arquivo em disco remoto.
