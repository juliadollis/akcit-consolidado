# Dataset NOVO contra dataset ANTIGO (v0) — comparação medida, eixo a eixo

Data: 2026-09-16. Complementa `AVALIACAO_DATASETS_BC.md` (que mede só o novo) e as
auditorias de código `ROTA_B_AUDITORIA_2.md` / `ROTA_C_AUDITORIA_2.md`. Aqui o único
objetivo é **o mesmo cálculo nos dois lotes**, com n declarado, para responder
literalmente: *está tudo correto? está melhor que o antigo?*

Tudo medido em modo somente-leitura via `ssh`, com `python3` do host e `numpy`/`PIL`/
`pyarrow` **dentro** dos containers. Nenhum arquivo criado, movido ou apagado em nenhum
cluster; nenhum job submetido, cancelado ou consultado; nenhum pacote instalado. Os
scripts foram entregues ao interpretador por `stdin` (`ssh host 'docker run -i … python3 -'
< script.py`), sem materializar arquivo em disco remoto.

---

## 0. Resposta curta

**Está melhor — sim, e com folga — no que decide o treino: o rótulo `K`, a física do
canal de profundidade, o split, a proveniência e a diversidade de cena da rota C.**

**E está pior em dois eixos, os dois medidos e pareados:**

1. **A ocupação do mapa de defocus caiu muito.** Na escala da imagem, o máximo mediano do
   mapa da rota B caiu de **0,4887** (v0) para **0,0221** (novo) — fator **22**; na rota C,
   de **1,0000** para **0,1285**. Em níveis uint8, a mediana caiu de **117 → 7** (B) e de
   **256 → 34** (C). O antigo usava a faixa toda; o novo usa 2% dela. *Mas* a faixa do
   antigo vinha de um sinal errado (§4, §5), então o eixo é uma regressão **de alcance**, não
   de informação.
2. **A AIF da rota B ficou 2,1x menos nítida.** Pareado por **a mesma foto do Flickr**
   (n = 3.815, resoluções idênticas), a variância do Laplaciano caiu de **1.026,1** para
   **404,0**; o novo é mais nítido em apenas **2,2%** dos pares. Controlei o confundidor
   de JPEG (§6.2) e ele **não explica** nada disso.

Três coisas continuaram iguais e continuam erradas: o limiar de SSIM que o paper exige
não é aplicado em nenhum dos dois lotes; a LFDOF está ausente nos dois; e a rota B tem
**1 amostra por cena** nos dois, ou seja, zero variação de nível dentro da cena.

---

## 1. Qual lote antigo eu usei, e como o identifiquei

### O que achei

| lote v0 | repositório HF | revisão | n | onde está materializado |
|---|---|---|---|---|
| rota B | `AKCITPixel3/BKXcuVXCmeRvN` | `b7ca23f062f6…4842` | **11.635** | `dgx-H100-01:/raid/user_juliadollis/julia_docker/hf-cache-julia/datasets/AKCITPixel3___bk_xcu_vx_cme_rv_n/default/0.0.0/b7ca23f0…/` — 40 GB, 76 shards `.arrow` |
| rota C | `AKCITPixel3/CMiQdveBBzNii` | `b78f1c250…c0c9` | **2.932** | `…/datasets/AKCITPixel3___c_mi_qdve_b_bz_nii/default/0.0.0/b78f1c25…/` — 37 GB, 71 shards `.arrow` |
| tabela lateral | `juliadollis/rota-b-kfix-eq3` | — | **11.635** | `…/hf-cache-julia/hub/datasets--juliadollis--rota-b-kfix-eq3/blobs/0467c5cc…` |

**Este é o dado de verdade, com imagens, não um registro.** Só a rota A do v0 não foi
localizada (fora do escopo desta comparação).

### Como cheguei nele

Os nomes são opacos (`BKXcuVXCmeRvN`) e não aparecem em nenhum documento do repositório.
Achei-os no **config de treino da fase 2**, que é quem consumiu o v0:

`dgx-H100-01:/raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper/configs/train_bokeh_fase2_4gpu.yaml`

```yaml
57:    datasets:
67:      - name: AKCITPixel3/BKXcuVXCmeRvN   # rota b (real + EXIF, ~ITW)
72:      - name: AKCITPixel3/CMiQdveBBzNii   # rota c (pares reais, K por SSIM)
```

O mesmo arquivo confirma, em comentário de 2026-08-13, o defeito registrado em
`ACHADOS.md`: *"auditamos 300 amostras e o k é 50.0 em TODAS — o valor de FALLBACK do
pipeline de dados"*, e *"a coluna `depth` foi gravada normalizada em [0,1], perdendo a
escala"*. Ele também fixa como o v0 era consumido: `defocus_source: recompute`,
`model.max_coc: 100.0`, `image_size: 512`, `min_calibration_ssim: 0.6` — parâmetros que
uso abaixo para reconstruir **o mapa que a rede realmente recebeu no v0**.

### Como confirmei que é o lote certo

Reproduzi, no dado, **sete** números publicados em `ACHADOS.md`, todos exatos:

| fato de `ACHADOS.md` | registrado | medido agora |
|---|---|---|
| rota B, amostras | 11.635 | **11.635** ✓ |
| rota B, coluna `k`, valores distintos | 1 (= 50,0) | **1 (= 50,0)** ✓ |
| rota B, `calibration_ssim` nula | 100% | **11.635 / 11.635** ✓ |
| rota C, amostras | 2.932 | **2.932** ✓ |
| rota C, `k == 300` exato | 1.379 = 47,0% | **1.379 = 47,03%** ✓ |
| rota C, `k == 0` | 51 = 1,7% | **51 = 1,74%** ✓ |
| rota C, K interior **e** ssim ≥ 0,6 | 1.483 = 50,6% | **1.483 = 50,58%** ✓ |
| rota C, `exif` nula | 100% | **2.932 / 2.932** ✓ |

Também reproduzi a linha *"`s1` implícito vs gravado: mediana 0,00308, p90 0,07927"* —
com valor idêntico, ao rodar o teste de semântica da profundidade (§3).

**Nada nesta comparação é contra registro. É dado contra dado.**

---

## 2. O quadro geral

| | **v0 rota B** | **novo rota B** | **v0 rota C** | **novo rota C** |
|---|---|---|---|---|
| amostras | 11.635 | **13.615** | 2.932 | **15.423** |
| cenas | 11.635 | 13.615 | 2.932 | **4.399** |
| amostras/cena | 1,000 | 1,000 | 1,000 | **3,506** |
| splits publicados | só `train` | `train` + `val` por cena | só `train` | `train` + `val` por cena |
| resolução | 376 distintas na amostra (1024 no lado longo) | 816 distintas (1024 no lado longo) | 1500×2000 em 100% | 1500×2000 em 100% |
| união B+C | **14.567** | **29.038** | | |

---

## 3. Eixo 1 — `K`

Mesma conta nos dois: `Counter` sobre `round(K, 6)`, quantis por interpolação linear, n
completo (sem amostragem).

| | **v0 B** (n=11.635) | **novo B** (n=13.615) | **v0 C** (n=2.932) | **novo C** (n=15.423) |
|---|---|---|---|---|
| valores distintos | **1** | **13.610** | 402 | 2.629 |
| mediana | 50,0 | 16,44 | **251,02** | 14,54 |
| média ± dp | 50,0 ± **0,0** | 17,35 ± 10,52 | 183,75 ± 124,25 | 36,51 ± 85,46 |
| p5 · p25 · p75 · p95 | 50 · 50 · 50 · 50 | 2,28 · 8,60 · 25,25 · 37,07 | 2,86 · 50,87 · 300 · 300 | 1,29 · 5,69 · 35,18 · 119,0 |
| **fração grudada num valor** | **100,00%** (K = 50,0) | 0,01% (repete 2x) | **47,03%** (K = 300) | 3,50% (K = 0,5) |
| **fração numa borda da busca** | — (constante) | **0,00%** | **48,77%** | **3,67%** |
| censura **marcada** no metadado? | **não existe campo** | n/a | **não existe campo** | **sim** — `is_k_censored` + `is_valid_for_control`, 100% |

**Quanto do defeito sobreviveu: nada.** O `k = 50` constante da rota B sumiu por completo
(1 → 13.610 valores distintos, nenhum inteiro exato, valor mais repetido aparece 2 vezes).
O `k == 300` em 47,0% da rota C caiu para **3,50%** no piso (K = 0,5) mais **0,17%** no
teto (K = 960) — e, ao contrário do v0, **todas** as amostras de borda estão marcadas como
censuradas e excluídas do controle. Em termos de "rótulo utilizável", o v0 entregava
**1.483 amostras** com K interior e SSIM ≥ 0,6 (50,58% da rota C, e zero da rota B, que
não tinha K físico nenhum); o novo entrega **29.038** com K físico, das quais **28.396** com
`is_valid_for_control = True` (13.615 da rota B + 14.781 da rota C).

Contrapartida honesta: a cauda do novo vai a **960** (0,17% no teto absoluto), e 6,74% da
rota C tem K > 96, acima da faixa em que o renderizador foi verificado — patologia que o
v0 não tinha porque o teto dele era 300. Isso está medido e discutido em
`AVALIACAO_DATASETS_BC.md` §2.1; é troca de um teto rígido por uma cauda longa.

---

## 4. Eixo 2 — o mapa de defocus, `clip(|K·Δ|/100, 0, 1)`

**Este é o eixo que mais importa, e ninguém tinha medido no antigo.** Medi agora, com o
mesmo código nos quatro lotes (mesma função de histograma, mesmas bordas, mesma definição
de saturação, mesma contagem de níveis uint8).

O que cada lado alimenta na fórmula:

- **v0** — `D` é a coluna `depth` decodificada `u16/65535` (profundidade métrica
  normalizada por imagem, §5), `D_focus` é a coluna `s1`, `K` é a coluna `k`,
  `max_coc = 100`. Isto é **literalmente** o que `train_bokeh_fase2_4gpu.yaml` fez
  (`defocus_source: recompute`), e portanto o que a rede viu na fase 2.
- **novo** — `disp = disparity_min + u16/65535·(disparity_max − disparity_min)`,
  `focus_disparity`, `k_value`, `max_coc = 100`. É `dataio.encoding.decode_disparity`.

Amostragem 1 em 3 nos quatro: **n = 3.879 (v0 B)**, **4.539 (novo B)**, **978 (v0 C)**,
**5.141 (novo C)** — 9,7 bilhões de pixels no total. A coluna "escala 512" aplica
`K' = K · 512/min(H,W)`, o crop de treino, que os dois configs usam.

### 4.1 Histograma agregado de pixels — escala da imagem

| faixa | **v0 B** | **novo B** | **v0 C** | **novo C** |
|---|---|---|---|---|
| [0,00 – 0,01) | 41,54% | **64,93%** | 16,17% | 22,17% |
| [0,01 – 0,05) | 22,21% | 30,17% | 21,28% | 31,73% |
| [0,05 – 0,10) | 10,86% | 4,08% | 13,02% | 15,67% |
| [0,10 – 0,25) | 14,41% | 0,81% | 21,60% | 18,76% |
| [0,25 – 0,50) | 10,97% | 0,01% | 14,90% | 8,96% |
| [0,50 – 1,00) | 0,00% | 0,00% | 2,67% | 2,12% |
| **== 1,00 (saturado)** | **0,00%** | **0,00%** | **10,38%** | **0,59%** |

### 4.2 Por amostra

| grandeza | **v0 B** | **novo B** | **v0 C** | **novo C** |
|---|---|---|---|---|
| máximo do mapa, p50 (escala imagem) | **0,4887** | **0,0221** | **1,0000** | **0,1285** |
| máximo do mapa, p50 (escala 512) | 0,3647 | 0,0164 | 0,5733 | 0,0438 |
| média do mapa, p50 (imagem) | 0,0678 | 0,0083 | 0,2115 | 0,0597 |
| fração de pixels < 0,01, p50 (imagem) | 0,3895 | 0,6253 | 0,0273 | 0,1292 |
| amostras que chegam a 1,0 (imagem) | 0,00% | 0,00% | **61,35%** | 3,35% |
| amostras que chegam a 0,5 (imagem) | 0,00% | 0,00% | 70,04% | 10,82% |
| **níveis uint8, p50 (imagem)** | **117** | **7** | **256** | **34** |
| **níveis uint8, p50 (escala 512)** | **88** | **5** | **135** | **12** |

### 4.3 Leitura honesta deste eixo

**Em alcance bruto, o antigo era melhor, e por muito.** O mapa do v0 ocupava 49% de
[0, 1] na rota B e saturava na rota C; o novo ocupa 2,2% e 12,9%. Em quantização de 8
bits, a mediana caiu de 117 para 7 níveis na rota B. Se o critério fosse só "o canal de
controle usa a faixa disponível", o novo é uma regressão de **22x**.

**Mas o alcance do v0 não carregava informação de controle.** Três medidas fecham isso:

1. **Rota B do v0: `K` é constante em 11.635/11.635.** Com `K = 50` fixo, o mapa
   `clip(50·|D − s1|/100)` é **função apenas da cena** — não existe nível de bokeh a
   controlar. A ocupação de 0,49 é ocupação de um mapa de profundidade, não de um sinal
   de controle. A faixa era grande e o conteúdo, zero.
2. **Rota C do v0: 10,38% de todos os pixels estão grudados em 1,0**, contra 0,59% no
   novo. Isso é informação **destruída** pelo clip, e vem direto do `K = 300` censurado
   de 47% do lote. 61,35% das amostras têm pelo menos um pixel saturado; 70,04% chegam a
   0,5. O "alcance" do v0 é, em boa parte, teto.
3. **O mapa do v0 tem a forma errada** — quantificado em §5.2: correlação de Pearson
   mediana **0,749** com o CoC fisicamente correto, e **26,03%** das amostras abaixo de
   0,5.

E há um detalhe de publicação do v0 que vale registrar: a coluna `defocus_map` gravada no
release tem `max = 65535` **exato** em **3.879/3.879** da rota B e **965/978 = 98,67%** da
rota C — ou seja, foi normalizada **por imagem**, apagando `K`. Foi exatamente por isso
que o config da fase 2 escreveu `defocus_source: recompute` e ignorou a coluna. O novo
release **não grava** mapa de defocus; ele grava profundidade em disparidade + `K` +
`max_coc` e manda recompor, que é a decisão certa.

**Veredito do eixo:** piorou em alcance, melhorou em conteúdo. A recomendação de
`AVALIACAO_DATASETS_BC.md` §3.3 continua de pé e agora tem contraste: o `max_coc = 100`
global custa 22x de alcance contra o que o v0 tinha, e o mapa **não pode** passar por
uint8 em ponto nenhum do dataloader.

---

## 5. Eixo 3 — a profundidade

### 5.1 A suspeita, testada no dado

A suspeita registrada era *"disparidade normalizada gravada como se fosse profundidade
métrica"*. **Ela está refutada — na direção em que foi formulada.** O que há é o oposto,
e é pior.

Teste, n = **11.635** (rota B inteira), juntando o release v0 à tabela
`juliadollis/rota-b-kfix-eq3` por `stem` (11.635/11.635 casam). Se `depth` for
profundidade métrica normalizada min-max, então `s1` tem de ser
`(z_foco − z_min)/(z_max − z_min)`; se for disparidade normalizada, tem de ser
`(1/z_foco − 1/z_max)/(1/z_min − 1/z_max)`. As duas previsões são comparadas com a coluna
`s1` gravada:

```
|s1_metrica  − s1|   mediana = 0,00308    p90 = 0,07927
|s1_disparidade − s1| mediana = 0,66184    p90 = 0,89669
amostras em que a hipotese METRICA fica mais perto:  11.276 / 11.635 = 96,91%
spearman(s1, s1_metrica)      = +0,8849
spearman(s1, s1_disparidade)  = −0,5036
```

**Conclusão [M]:** a coluna `depth` do v0 é **profundidade métrica, normalizada min-max
por imagem**, e `s1` vive no mesmo espaço. Confirma a linha de `ACHADOS.md`
(*"profundidade métrica min-max, não disparidade"*) e **refuta** a suspeita inversa.

O defeito real do v0 é duplo:

- **A escala absoluta foi destruída.** Cada imagem é renormalizada para [0, 1], então
  `K` não tem significado comparável entre imagens. Foi isso que forçou o `K = 50` da
  rota B: a Eq. 3 precisa de `D_focus` em metros e a coluna não tinha metros.
- **A grandeza está no espaço errado para a fórmula.** O CoC é linear em **disparidade**;
  `K·|D − D_focus|` com `D` linear em `z` não é CoC.

No novo: `depth_encoding = "uint16_linear_in_disparity"` em **29.038/29.038**, com
`disparity_min`/`disparity_max` e `z_min_m`/`z_max_m` gravados por amostra, e erro de
quantização em CoC de **5,1e-5 px** mediano (medido em `AVALIACAO_DATASETS_BC.md` §5).

### 5.2 Quanto custa a grandeza errada

Medido em **n = 388** amostras da rota B do v0 (1 em 30), reconstruindo
`z = z_min + D·(z_max − z_min)` com os limites da tabela kfix e comparando, pixel a
pixel (subamostrado 4×4), o mapa do v0 (`|D − s1|`, normalizado) com o CoC correto
(`|1/z − 1/z_foco|`, normalizado):

```
pearson(mapa v0, CoC correto):  p10=0,215  p25=0,477  p50=0,749  p75=0,870  p90=0,934
amostras com pearson < 0,80: 60,05%      amostras com pearson < 0,50: 26,03%
|mapa_v0 − CoC| medio por amostra:  p25=0,202  p50=0,292  p75=0,374
fracao de pixels abaixo de 10% do maximo — mapa v0: 0,671   CoC correto: 0,279
razao z_max/z_min mediana da cena: 35,18
```

Em uma cena de cada quatro, o mapa de controle do v0 correlacionava **menos de 0,5** com
o borrão que deveria descrever. O mecanismo é direto. Exemplo aritmético em `b_000000`
(`z_min = 1,354 m`, `z_foco = 4,423 m`, `z_max = 72,44 m`): linear em `z`, o céu a 72 m
recebe todo o alcance e um objeto a 10 m recebe **5,8%** dele; em disparidade, esse mesmo
objeto recebe **59%**. O v0 achatava o campo médio.

### 5.3 Saúde da quantização (mesma definição: níveis u16 distintos no PNG)

| | **v0 B** (n=3.879) | **novo B** (n=4.539) | **v0 C** (n=978) | **novo C** (n=5.141) |
|---|---|---|---|---|
| níveis u16 p5 · **p50** · p95 | 4.727 · **35.890** · 60.183 | 34.202 · **55.114** · 62.936 | 13.231 · **54.167** · 65.060 | 29.454 · **52.038** · 62.735 |
| amostras com < 256 níveis | **0,00%** | 0,00% | 0,00% | 0,00% |

Rota B melhorou 54% na mediana e 7,2x no p5. **Rota C ficou praticamente igual, 3,9%
pior na mediana** (54.167 → 52.038) — e isso é consequência de o novo gravar a
profundidade em 576×768 em vez da resolução da imagem; é irrelevante porque o erro de CoC
correspondente é 5,1e-5 px, mas não é melhora.

*Ressalva:* o teste decisivo de §5.1 só existe para a rota B, porque só ela tem tabela
métrica lateral. Para a rota C do v0 não há gabarito de `z` — a semântica dela é
**inferida** [I] do fato de ser o mesmo código de pipeline.

---

## 6. Eixo 4 — a AIF

Mesma métrica nos quatro lotes: luma BT.601, Laplaciano de 4 vizinhos, variância global
na imagem inteira — a métrica dos gates e a mesma de `AVALIACAO_DATASETS_BC.md` §4.4.

### 6.1 Rota C — melhorou, e a origem da AIF mudou

Pareado **pela mesma cena da RealBokeh** (v0 `c_<n>` ↔ novo `train_<n>`), n = **273**,
**mesma resolução 1500×2000 nos dois**, sem reamostragem.

| | **v0 C** | **novo C** |
|---|---|---|
| o que é a "AIF" | o **maior f-stop de dentro de `gt/`** (`source_aif`) | `image_focus` do espelho `akcit-pixel/RealBokeh` (`aif_ref`) |
| f-stop dessa AIF | mediana **f/14**, faixa f/2.2 – f/20 | — (não é uma foto de `gt/`) |
| variância do Laplaciano p25 · **p50** · p75 | 127,9 · **254,7** · 637,4 | 170,4 · **399,3** · 889,6 |
| razão novo/antigo por cena, p25 · **p50** · p75 | — | 1,001 · **1,175** · 1,763 |
| cenas em que o novo é mais nítido | — | **206 / 273 = 75,5%** |

Confirmado no dado o que `ACHADOS.md` registrava como estrutura: a "AIF" do v0 era uma
foto de abertura alta tirada de dentro de `gt/`, mediana **f/14** — e em **78 amostras
(2,66%)** essa "AIF" era f/2.8 ou mais aberta, isto é, **uma foto borrada usada como
all-in-focus**. O novo troca isso por uma fonte que é 1,18x mais nítida na mediana e
mais nítida em 3 de cada 4 cenas. Melhora modesta em magnitude, mas elimina a classe de
falha.

### 6.2 Rota B — **piorou 2,1x**, e o JPEG não explica

Pareado **pela mesma foto do Flickr** (`source_path` do v0 ↔ `sample_id` do novo),
n = **3.815**, **resoluções idênticas em 3.815/3.815**.

| | **v0 B** | **novo B** |
|---|---|---|
| variância do Laplaciano p10 · p25 · **p50** · p75 · p90 | 326,3 · 575,2 · **1.026,1** · 1.662,0 · 2.554,2 | 138,6 · 225,1 · **404,0** · 802,0 · 1.462,7 |
| razão novo/antigo por par, p10 · p25 · **p50** · p75 · p90 | — | 0,251 · 0,344 · **0,472** · 0,624 · 0,773 |
| pares em que o novo é mais nítido | — | **83 / 3.815 = 2,2%** |

**Controle do confundidor de compressão** (a AIF do v0 é PNG, a do novo é JPEG). Reabri
a AIF do v0 e a reescrevi em JPEG em três qualidades antes de medir, n = **453**:

| AIF do v0 medida em | p50 da variância | razão novo/antigo p50 | novo mais nítido em |
|---|---|---|---|
| PNG original | 849,2 | 0,476 | 2,2% |
| JPEG q95 | 876,0 | 0,466 | 1,8% |
| JPEG q90 | 882,1 | 0,456 | 1,5% |
| JPEG q85 | 882,5 | 0,454 | 1,8% |

Recomprimir **não baixa** a métrica — sobe um pouco (blocagem soma alta frequência). **A
queda de 2,1x é do conteúdo da imagem, não do formato.**

**O que isso quer dizer.** `AVALIACAO_DATASETS_BC.md` §4.4 já tinha medido que a AIF da
**nossa** DeblurNet é 2,20x menos nítida que a da **oficial** (376,4 contra 1.010, n=717).
O número que acabo de medir encaixa a peça que faltava: a AIF do **v0 está no nível da
oficial** (1.026,1) e a do `b_release` está no nosso (404,0 aqui, 376,4 lá). Ou seja, a
rota B **trocou de comportamento de deblur entre o v0 e o novo**, na direção mais suave.

Duas ressalvas obrigatórias, ambas já registradas: (a) a variância do Laplaciano não
separa "nitidez real" de "textura de ruído/artefato de JPEG", e o teste de bytes q95 de
§4.4 mostra que a nossa AIF comprime **menos**, o que é assinatura de estrutura limpa e
não de lavagem; (b) o nosso 60k **ganha** do oficial em LPIPS no RealDOF (0,2291 contra
0,2397). Mesmo assim: **por esta métrica, neste eixo, o novo é pior, em 97,8% das fotos.**
E não dá para atribuir a diferença a um checkpoint específico **pelo dado**, porque o v0
não grava hash de modelo nenhum (§9).

---

## 7. Eixo 5 — o split

| | **v0 B** | **novo B** | **v0 C** | **novo C** |
|---|---|---|---|---|
| splits no release | **só `train`** | `train` + `val` | **só `train`** | `train` + `val` |
| split materializado por cena? | **não existe** | sim (`split.json`, `assignment` por `scene_id`) | **não existe** | sim |
| cenas dos dois lados | **n/a — não há validação** | **0 / 13.615** | **n/a** | **0 / 4.399** |
| tamanho da validação | — | 659 cenas | — | 440 cenas (train 3.959) |

O `dataset_info.json` dos dois lotes v0 declara **um único split, `train`**, e o config
da fase 2 consome `split: train` nas duas fontes com `shuffle: true`. **Não havia conjunto
de validação.** Não houve vazamento porque não havia para onde vazar — o que é pior, não
melhor: o v0 foi treinado sem nenhuma cena retida.

Ressalva de justiça: como o v0 tinha **1 amostra por cena** nas duas rotas (§8), um split
aleatório por amostra também não teria vazado. O ganho do novo é ter split, e tê-lo por
cena — o que importa de verdade na rota C, onde agora existem 3,5 amostras por cena e um
split por amostra vazaria por construção.

---

## 8. Eixo 6 — diversidade real

| | **v0 B** | **novo B** | **v0 C** | **novo C** |
|---|---|---|---|---|
| amostras | 11.635 | 13.615 | 2.932 | 15.423 |
| cenas distintas | 11.635 | 13.615 | 2.932 | **4.399** |
| amostras por cena | 1,000 | 1,000 | 1,000 | **3,506** |
| histograma amostras/cena | 1 → 11.635 | 1 → 13.615 | **1 → 2.932** | 4 → 2.986 · 3 → 655 · 2 → 756 · 1 → 2 |
| metade do dataset vem de | 50,0% das cenas | 50,0% das cenas | 50,0% das cenas | **43,8%** das cenas |

**O achado mais duro da rota C do v0:** das 2.932 amostras, **2.930 usam bokeh a f/2.0**
(uma a f/2.5, uma a f/8.0). Havia **um único nível físico de bokeh no lote inteiro** — e,
mesmo assim, o rótulo `K` varria de 0 a 300 com 47% no teto. Não sobra interpretação
benigna: o `K` do v0 não estava medindo nível de bokeh, estava medindo geometria de cena
(e saturando a busca quando a cena não cooperava). O novo usa 2 a 4 aberturas por cena,
que é o que o paper descreve (`paper.txt:1002-1004`).

Contrapartida, já medida em `AVALIACAO_DATASETS_BC.md` §2.6 e que **não** conserto aqui:
ter 2–4 níveis por cena não produziu `K` ordenado com a abertura (16,3% de cenas
monotônicas contra 13,9% de acaso). A diversidade estrutural existe; a consistência
física dentro da série, não.

A rota B **não mudou neste eixo**: 1 amostra por cena nos dois. Zero variação de nível
dentro da cena, antes e agora.

---

## 9. Eixo 7 — proveniência

Contagem direta de campo presente e não-nulo, n completo nos quatro lotes.

| campo | **v0 B** (11.635) | **novo B** (13.615) | **v0 C** (2.932) | **novo C** (15.423) |
|---|---|---|---|---|
| `control_version` | **0** | **13.615 (100%)** | **0** | **15.423 (100%)** |
| hash do modelo de profundidade | **0** | **13.615 (100%)** | **0** | **15.423 (100%)** |
| hash do modelo de máscara | **0** | **13.615 (100%)** | **0** | **15.423 (100%)** |
| commit do pipeline | **0** | **13.615 (100%)** | **0** | **15.423 (100%)** |
| resolução gravada (`image_h`/`image_w`) | **0** | **13.615 (100%)** | **0** | **15.423 (100%)** |
| `depth_encoding` | **0** | 100% | **0** | 100% |
| `depth_backend` | **0** | 100% | **0** | 100% |
| `max_coc` | **0** | 100% | **0** | 100% |
| marcação de censura de `K` | **0** | 100% | **0** | 100% |
| hash do renderizador | **0** | n/a (rota B não renderiza) | **0** | 100% |
| `exif` | 100% | 100% | **0%** | 0% (a fonte não publica) |
| `calibration_ssim` | **0%** | 0% (não se aplica) | 100% | 100% |

O esquema do v0 tem **16 colunas** (`stem, route, bokeh, aif, depth, defocus_map,
foreground_mask, s1, k, exif, qc, calibration_ssim, shape_kernel, source_path, source_aif,
source_bokeh`) e **nenhuma** delas carrega versão, hash ou resolução. O bloco `qc` tem 19
chaves, todas de métrica de alinhamento (`mi`, `nmi`, `chamfer`, `boundary_precision`,
`severity`, `source_verdict`, `notes`, …) — nenhuma de proveniência. `shape_kernel` é
**nulo em 14.567/14.567**.

**0 de 14.567 amostras do v0 têm hash de modelo, resolução gravada ou `control_version`.
29.038 de 29.038 do novo têm os três.** É a diferença entre um lote irreproduzível e um
lote auditável, e é a maior melhora absoluta desta comparação.

Efeito prático imediato: é por isso que §6.2 não consegue atribuir a queda de nitidez da
AIF da rota B a um checkpoint — o v0 não registrou qual usou.

---

## 10. Eixo extra — a qualidade do ajuste da Eq. 5 (só rota C)

Não estava na lista, mas é o rótulo da rota C e mede direto se o novo `K` é mais
confiável.

| | **v0 C** (n=2.932) | **novo C** (n=15.423) |
|---|---|---|
| `calibration_ssim` p25 · **p50** · p75 | 0,808 · **0,871** · 0,915 | 0,917 · **0,9633** · 0,986 |
| SSIM < 0,90 | **65,59%** | 19,76% |
| SSIM < 0,80 | **22,41%** | **6,82%** |
| SSIM < 0,70 | 5,32% | 2,31% |
| SSIM < 0,50 | 0,24% | 0,14% |
| limiar do paper aplicado? | **não** | **não** |
| spearman(K, SSIM) | −0,171 | −0,685 |

O ajuste melhorou muito: a fração de rótulos com SSIM < 0,80 caiu de 22,41% para 6,82%,
fator 3,3. Duas ressalvas: (a) **nenhum dos dois** aplica o limiar que
`paper.txt:397-399` exige — o v0 pelo menos tinha o `min_calibration_ssim: 0.6` no config
de treino, que descartava 1,09% do lote na hora de treinar; o novo não traz limiar nem no
release nem em config; (b) a correlação negativa entre `K` e SSIM ficou **muito mais
forte** no novo (−0,685 contra −0,171), porque a faixa de `K` do novo é muito mais larga —
o rótulo do novo degrada sistematicamente na cauda alta, coisa que o v0 escondia atrás do
teto em 300.

---

## 11. O que piorou ou não mudou — lista fechada

**Piorou:**

1. **Alcance do mapa de defocus** — máximo mediano 0,4887 → 0,0221 (rota B, escala
   imagem, fator 22) e 1,0000 → 0,1285 (rota C). Níveis uint8 medianos 117 → 7 e
   256 → 34. §4.
2. **Nitidez da AIF da rota B** — variância do Laplaciano 1.026,1 → 404,0, razão mediana
   0,472, novo mais nítido em só 2,2% de 3.815 pares; confundidor de JPEG controlado e
   descartado. §6.2.
3. **Correlação K × SSIM na rota C** — −0,171 → −0,685; o rótulo do novo perde
   confiabilidade na cauda alta de um jeito que o v0 não mostrava. §10.
4. **Faixa verificada do renderizador** — o v0 não tinha amostra fora de [0, 300] por
   construção; o novo tem 6,74% da rota C com K > 96, fora do que o laudo verificou. §3.

**Não mudou:**

5. **Rota B: 1 amostra por cena.** 11.635/11.635 antes, 13.615/13.615 agora. Zero
   variação de nível de bokeh dentro da cena nas duas versões. §8.
6. **O limiar de SSIM do paper não é aplicado** em nenhum dos dois releases. §10.
7. **LFDOF ausente** nos dois, embora `paper.txt:356` e `paper.txt:527-530` a citem.
8. **Níveis u16 de profundidade da rota C** — 54.167 → 52.038 na mediana, 3,9% pior;
   efeito prático nulo (5,1e-5 px de erro de CoC), mas não é melhora. §5.3.
9. **`exif` da rota C continua ausente** — 0% nos dois; a fonte não publica.

---

## 12. O que não deu para comparar, e por quê

| eixo | por quê |
|---|---|
| **Semântica da profundidade da rota C do v0** | O teste decisivo de §5.1 precisa de `z_min`/`z_max`/`z_foco` métricos por amostra, e a tabela `rota-b-kfix-eq3` cobre **100% da rota B e 0% da rota C** (`ACHADOS.md:43`). Para a rota C a conclusão é **[I]**, herdada do fato de ser o mesmo código. |
| **Rota A** | O lote v0 da rota A (~68.000 amostras, stems só UUID) **não foi localizado** em nenhum cache dos dois clusters; e não há rota A nova para comparar. Fora do escopo. |
| **Qual DeblurNet gerou a AIF do v0 da rota B** | O v0 não grava hash de pesos (§9). A atribuição só existe em documentação (`bokehnet-preprocessing/README.md` cita `juliadollis/genrefocus-deblurnet-paper-4gpu`), não no dado. A queda de 2,1x de §6.2 fica **medida** mas **não atribuída**. |
| **Mapa de defocus *gravado* do novo** | O novo release, por decisão de projeto, **não grava** coluna `defocus_map` — grava profundidade em disparidade + `K` + `max_coc`. Comparei o mapa *recomposto* nos dois (que é o que a rede recebe) e reportei o gravado do v0 só como evidência da normalização por imagem. §4.3. |
| **Distribuições contra as figuras do paper (Figs. 13–16)** | Não comparadas em nenhum dos dois lotes; lacuna aberta em `AVALIACAO_DATASETS_BC.md` §6, item 6. |
| **Qualidade final (LPIPS, LVCorr) do novo** | Não é métrica de dataset e exigiria treinar. Os números do v0 estão em `ACHADOS.md`; não há equivalente do novo. |
| **Comparação pixel a pixel entre mapa v0 e mapa novo da mesma foto** | As profundidades vêm de execuções diferentes do Depth Pro, em resoluções diferentes (v0 na resolução da imagem, novo em 576×768 na rota C). A diferença mediria o backend, não o dataset. |

---

## 13. Método e os n

| medição | onde | n | ferramenta |
|---|---|---|---|
| metadados v0 (K, s1, SSIM, EXIF, QC, fontes) | h100n1, arrow memory-mapped, projeção de coluna | **11.635 + 2.932** (completo) | `pyarrow.ipc` + `pa.memory_map`, docker `julia-genrefocus:1.0` |
| mapa de defocus, 4 lotes, mesmo código | h100n1 (v0 B, v0 C, novo B), h100n3 (novo C) | 3.879 + 978 + 4.539 + 5.141 (1 em 3) | `numpy`/`PIL` no container |
| semântica da profundidade (v0 B) | local, sobre metadado + tabela kfix | **11.635** (completo) | stdlib |
| distorção física do mapa v0 (v0 B) | h100n1 | 388 (1 em 30) | `numpy` no container |
| nitidez da AIF, pareada rota B | h100n1 | **3.815** pares | mesma métrica dos gates |
| nitidez da AIF, controle de JPEG | h100n1 | 453 pares | reencode q85/q90/q95 em memória |
| nitidez da AIF, pareada rota C | h100n1 (v0) + h100n3 (espelho) | 273 cenas | idem, medidas em cada host e cruzadas localmente |
| proveniência/split/diversidade do novo | h100n1 + h100n3, `meta/*.json` | **13.615 + 15.423** (completo) | `python3` do host |

Validação cruzada do método: as minhas medidas do `c_release` reproduzem
`AVALIACAO_DATASETS_BC.md` §3 dentro do arredondamento (histograma em escala 512:
41,15 / 36,46 / 12,74 / 8,45 / 0,98 contra 41,10 / 36,41 / 12,72 / 8,44 / 0,97; máximo
mediano 0,0438 contra 0,0439), e as do `b_release` batem exato (64,93 / 30,17 / 4,08 /
0,81 / 0,01; máximo mediano 0,0221 e 0,0164). É o mesmo cálculo dos dois lados.

Nenhum job foi submetido, cancelado ou consultado; nenhum container de terceiros foi
tocado; nenhum arquivo foi criado, movido ou apagado em `dgx-H100-01` ou `dgx-H100-03`;
nenhum pacote foi instalado; nenhum byte de imagem transitou entre clusters.
