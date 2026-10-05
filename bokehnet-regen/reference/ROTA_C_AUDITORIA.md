# Auditoria da ROTA C — §3.2(c), LFDOF e RealBokeh

Data: 2026-09-10. Autoridade: `reference/paper.txt` (arXiv:2512.16923v3), depois o
código oficial, depois decisão nossa declarada. Toda afirmação carrega
`paper.txt:<linha>` ou `<arquivo>:<linha>`. Onde não consegui medir, está escrito
**não medido**.

**Nenhum código foi escrito nem editado.** Este documento descreve o que muda; não
aplica nada.

Etiquetas, iguais às de `ACHADOS.md`: `[M]` medido · `[I]` inferido de algo medido ·
`[A]` assumido e não verificado.

> **Emenda de 2026-09-10, etapa 9 — o item 6 / defeito D2 deste documento FOI FECHADO.**
> Onde se lê aqui "não existe refinamento" e "dez gates que descartam", o estado atual é
> outro: `src/qc/focus_region.py` faz o refinamento automático da região em foco e
> `routes.route_c` o aplica antes dos gates, com `focus_source` gravado por amostra.
> A auditoria não foi reescrita — ela é o registro do que era verdade quando foi feita.
> O que mudou está em `REGISTRO.md` (etapa 9) e a medição que motivou, em
> `reference/MEDICAO_PLANO_FOCO.md`.

Reuso declarado: as conclusões das auditorias irmãs (`ROTA_A_AUDITORIA.md`,
`ROTA_B_AUDITORIA.md`) sobre escopo da Eq. 3, `pixel_ratio` pelo maior lado, e
"nenhum renderizador produz o alvo da rota B" foram **aceitas sem re-derivação**.
Não discordo de nenhuma. Onde a rota C muda a leitura, está dito.

---

## 0. Sumário executivo

A rota C é, das três, a que está mais perto de pronta: a cadeia
AIF → Depth Pro → BiRefNet → Eq. 4 → Eq. 5 → gates → writer existe inteira, roda,
tem 239 testes passando [M] e não tem fallback numérico no caminho do rótulo.

Mas ela **não está pronta para gerar dado**, por cinco motivos de peso desigual.
Dois são defeitos de código medidos, um é uma leitura do paper que ainda não foi
feita, e dois são escolhas de faixa que hoje não têm justificativa física.

| # | achado | efeito no rótulo | evidência |
|---|---|---|---|
| **C1** | `k_analytic` (Eq. 3 sobre o alvo) **não é comparável** com `K*` (Eq. 5), e é impresso como se fosse | validador inútil; convida a "consertar" o sweep que está certo | §2.4, `route_c.py:150`, `calibrate_thresholds.py:209` |
| **C2** | `aif_f_number = None` **reprova** a amostra (`NaN` nunca passa) | **100% do LFDOF rejeitado**, com slug que diz "abertura larga" | `[M]` §4.2, `gates.py:54-59`, `gates.py:305-307` |
| **C3** | multiplicidade por cena: pegamos até **21** alvos/cena; o paper diz **2 a 4 por conjunto** | 25% do dataset sai de 6,2% das cenas | `paper.txt:1001-1003`, `[I forte]` §1, item 8 |
| **C4** | `K_ABSOLUTE_MAX = 960` é ~27× acima do fisicamente possível e **fora da faixa verificada do renderer** (K ≤ 96) | custo e regime não medido; censura sem significado | `calibration.py:38`, `output/renderer_verification.json` |
| **C5** | `FOCUS_DEPTH_MIN_M`/`MAX_M` **bloqueiam por default** e o piloto não consegue calibrá-los | contradiz a regra "limiar não medido não bloqueia"; `[A]` que já filtra | `contract.py:243-244`, `gates.py:231-253`, `calibrate_thresholds.py:64-74` |

Mais seis achados menores, em §5, incluindo um **slug de rejeição errado**, um
**gate que não pode reprovar** dentro do laudo do renderer, e um `k_min = 0` aceito
pela calibração que a contrato rejeita — três amostras do padrão clássico deste
projeto.

**Veredito**: ver §8.

---

## 1. §3.2(c) inteiro — o que o paper manda, item por item

O parágrafo vive em `paper.txt:359-399` (o corpo é cortado ao meio pela Tab. 2, que
entra em `paper.txt:371-390`; a frase de `:370` continua em `:391`). A legenda da
Fig. 3(c) é `paper.txt:296-297`, e o supplement B.2 é `paper.txt:994-1011`.

Transcrição dos itens, na ordem em que o paper os enuncia, cada um confrontado com o
código.

### Item 1 — a fonte são **dois** datasets: LFDOF [52] e RealBokeh [57]

> *"(c) **LFDOF and RealBokeh.** These datasets provide pairs but omit EXIF metadata
> or provide insufficient fields to estimate K"* — `paper.txt:359-360`

Confirmado em mais três lugares: `paper.txt:326` (a enumeração das três rotas),
`paper.txt:296` (legenda da Fig. 3), e `paper.txt:528-529` (§4.1: *"approximately 26K
real examples sourced from ITW dataset [19], RealBokeh [57], and LFDOF [52]"*).
Bibliografia: `[52]` = Ruan et al., AIFNet/LFDOF (`paper.txt:873-875`); `[57]` =
Seizinger et al., Bokehlicious/RealBokeh (`paper.txt:885-887`).

**No código**: só RealBokeh. `run_route_c.py:193` aceita `--source lfdof` e
`_carrega_fonte` (`run_route_c.py:145-150`) falha com mensagem explícita. É lacuna
**declarada**, registrada em `REGISTRO.md:768-771`. Detalhe em §4 deste relatório.

**Divergência: declarada, e bloqueante para "rota C completa".** Ver C2 — hoje o
adaptador do LFDOF, se fosse escrito, seria rejeitado por um gate.

### Item 2 — o motivo de existir a rota: **não dá para usar a Eq. 3**

> *"omit EXIF metadata or provide insufficient fields to estimate K (see Fig. 3 (c))"*
> — `paper.txt:359-360`

**No código**: `KSource.EQ5_SSIM_SWEEP` (`sample.py:67`) é o `k_source` gravado
(`route_c.py:273`), e nunca `EQ3_EXIF`. ✓ **Confere.**

### Item 3 — `D_focus` primeiro, K depois

> *"To address this, we **first** compute the focus-plane proxy Dfocus."* —
> `paper.txt:360-361`

**No código**: `process_pair` calcula `focus_disparity` em `route_c.py:241` e só então
chama `calibrate_k` em `:246`. ✓ **Confere.**

### Item 4 — a máscara inicial vem do **BiRefNet [86]**, "similar a (b)"

> *"Similar to (b), we employ **BiRefNet [86]** to obtain an initial in-focus mask M."*
> — `paper.txt:361-362`

`[86]` = Zheng et al., BiRefNet (`paper.txt:955-957`) — numeração nova, confere.

**No código**: `BiRefNetRuntime` (`segmentation.py:93-167`), único segmentador, sem
cascata, `MaskSource.BIREFNET` gravado (`route_c.py:279`). ✓ **Confere**, e melhora:
o `MaskSource` é enum fechado sem valor `AUTOMATIC` (`sample.py:52-62`), que é o
conserto do B7 da rota B.

De **qual imagem** sai a máscara: ver §5.

### Item 5 — `D_focus` é a mediana da profundidade sob a máscara (Eq. 4)

O paper não repete a Eq. 4 dentro do parágrafo (c); ele diz *"Similar to (b)"*
(`paper.txt:361`) e *"we obtain Dfocus as in (b)"* (`paper.txt:296-297`). A Eq. 4 é
`paper.txt:352`: `D_focus = median(D[M])`.

**No código**: `focus_disparity_from_mask` (`contract.py:247-284`), que calcula
`median(1/z[M])` — **na disparidade**, não na profundidade. É a regra 0 do
`CONTRATO.md:26-34`, decisão declarada por autoridade do código oficial
(`Inference_bokehNet.py:118`), não leitura literal do paper. ✓ **Confere, com desvio
declarado.**

Uma coisa boa da rota C que a rota B não tem: aqui **não há a ambiguidade A5 da rota
B**. A rota B tem que escolher entre `1/median(1/z)` e `median(z)` para alimentar a
Eq. 3; a rota C alimenta a Eq. 3 (o validador) com `focus_plane_distance` **medido na
cena**, publicado pela origem (`realbokeh.py:633`), e o mapa com `focus_disparity`.
Documentado em `route_c.py:300-309`. ✓ Concordo com a auditoria da rota B nisso.

### Item 6 — **refinamento MANUAL** da máscara, em vez de descartar

> *"However, due to the increased diversity and complexity of the scenes in these
> datasets, the initial estimate of M is sometimes unreliable. **Rather than simply
> verifying and discarding unreliable cases, we introduce a manual refinement step.**
> Specifically, we re-select a small yet reliable in-focus region to correct M [...]
> This strategy **preserves challenging samples rather than excluding them**"* —
> `paper.txt:362-368`

E o supplement quantifica: *"we first conducted a manual verification and refinement
process on the in-focus masks. This annotation step required **4 to 8 seconds per
image**, amounting to approximately **8 hours of manual effort** in total"* —
`paper.txt:1003-1006`.

**No código**: não existe refinamento. Existem dez gates que **descartam**
(`gates.py`, `route_c.py:160-216`). A substituição está declarada em três lugares —
`route_c.py:33-39`, `gates.py:15-21` e `CONTRATO.md` — com a direção do viés escrita
por extenso: descartamos onde o paper corrigia, então o viés é **contra cena
complexa**, exatamente a direção que o paper diz ter evitado de propósito.

**Divergência: DECLARADA.** Análise completa em §7.

### Item 7 — calibração **simulador-no-laço** do bokeh level (Eq. 5)

> *"With Dfocus established, we adopt a **simulator-in-the-loop calibration** of the
> bokeh level. Given an AIF image Iaif and estimated depth D, we **sweep K** and
> choose the value whose rendered result **best matches the real bokeh target Ireal**"*
> — `paper.txt:369-370` + `:391`, equação em `:393`.

**No código**: `calibrate_k` (`calibration.py:111-228`), com `render_fn` = BokehMe
in-process. ✓ **Confere.** Detalhe em §2.

### Item 8 — o volume e a multiplicidade (supplement B.2)

> *"a comprehensive dataset of 26K real bokeh images. This collection comprises 13K
> previously filtered and verified images from the ITW dataset [19], alongside **13K
> images newly curated** for this work. The newly collected data consists of
> **focus-consistent series captured with varying apertures, containing 2 to 4 images
> per set**."* — `paper.txt:999-1003`

Este item **não estava em nenhuma lista do projeto** e muda o dimensionamento da rota C.

Três aritméticas que fecham entre si — `[I forte]`, não `[M]`:

1. **8 horas ÷ 4-8 s por imagem = 3.600 a 7.200 máscaras** (`paper.txt:1004-1006`). A
   RealBokeh_3MP tem **4.400 cenas** (`ACHADOS.md`, os três splits: 3.960 + 220 + 220)
   [M], e a máscara é **por cena** (uma AIF por cena). 4.400 × 6,5 s = 7,9 h.
2. **4.400 cenas × 3 alvos = 13.200 ≈ "13K"** (`paper.txt:1001`).
3. Recontando o histograma medido de níveis/cena do espelho (`ACHADOS.md`:
   `2→705 · 3→621 · 5→2.341 · 7→9 · 9→34 · 21→244 · 1→2 · 4→1 · 6→1 · 12→1`, soma
   20.495 [M]) **com teto de 4 alvos por cena**: 1.410 + 1.863 + 4×2.631 + 2 =
   **13.799** no split `train`. Com teto 3: **11.168**. A faixa "2 a 4 por conjunto"
   do paper produz 11K–14K — e o paper diz 13K.

**No código**: `enumerate_pairs` (`realbokeh.py:672-714`) pega **todos** os níveis de
todas as cenas: 20.495 pares de 3.959 cenas no `train`, 22.990 nos três splits [M].
Consequência aritmética direta do histograma medido [I]:

```
cenas com 21 níveis :   244 de 3.959  =  6,2% das cenas
pares que elas geram : 5.124 de 20.495 = 25,0% das amostras
```

Um quarto do dataset sairia de um vigésimo das cenas. É a mesma armadilha que o
`CLAUDE.md` já nomeia — *"20.554 amostras de 3.960 cenas não são 20.554 unidades de
diversidade"* — só que agora com número.

**Divergência: ACIDENTAL** (ninguém decidiu pegar 21; foi o que a enumeração fez), e é
o **C3**. O conserto é barato: um teto de níveis por cena, sorteado com seed, aplicado
onde `sample_pairs_for_pilot` já vive (`mirror_images.py:280-305`). Nada em `route_c.py`
muda.

### Item 9 — o `K*` vira o **pseudo-rótulo**, condicionado a um **limiar de SSIM**

> *"The selected K⋆ is then used as the **pseudo-bokeh-level label** for training,
> **provided that its corresponding SSIM exceeds a predefined threshold** to ensure
> reliable supervision."* — `paper.txt:396-398`

E de novo no supplement: *"we applied a Structural Similarity (SSIM) threshold to
filter out sub-optimal results, ensuring that only reliable pairs are used for
supervision"* — `paper.txt:1008-1010`.

**No código**: o gate existe (`gates.py:313-334`), o slug existe
(`contract.py:105`), a ponte existe (`route_c.py:209`), o teste que o faz **reprovar**
existe (`test_route_c.py:166-170`, `test_gates.py:140-146`) — e o valor é `None` por
default (`route_c.py:111`). ✓ **Confere estruturalmente**, com o valor `[A]`. Ver §2.3.

### Item 10 — por fim, `D_def` pela Eq. 2

> *"Finally, the defocus map Ddef is constructed accordingly following Eq. 2."* —
> `paper.txt:398-399`

**No código**: **não é gravado**. É derivado no dataloader pela mesma função da
geração (`defocus_map`, `contract.py:433-451`), por decisão declarada em
`sample.py:13-16` — gravá-lo criaria segunda fonte de verdade (o defeito D1). O teste
`test_metadado_em_disco_reconstroi_o_mapa` (`test_route_c.py:244-265`) prova que o
mapa é reconstruível só dos escalares, e que nenhum arquivo `*defocus*` é escrito.
✓ **Confere, com desvio declarado e travado por teste.**

### Resumo do §3.2(c)

| item | paper | código | veredito |
|---|---|---|---|
| 1 | LFDOF **e** RealBokeh | só RealBokeh | divergência **declarada**, bloqueante (C2) |
| 2 | não usar Eq. 3 para o rótulo | `k_source = eq5_ssim_sweep` | ✓ |
| 3 | `D_focus` antes de K | `route_c.py:241` antes de `:246` | ✓ |
| 4 | máscara do BiRefNet | `BiRefNetRuntime`, único | ✓ |
| 5 | `D_focus = median(D[M])` | `median(1/z[M])` | ✓ desvio **declarado** (regra 0) |
| 6 | refinamento **manual** | dez gates que descartam | divergência **declarada** |
| 7 | sweep de K com simulador | `calibrate_k` + BokehMe | ✓ |
| 8 | **2 a 4 imagens por conjunto**, 13K | todos os níveis, 20.495 | divergência **ACIDENTAL** (C3) |
| 9 | limiar de SSIM | gate existe, valor `None` | ✓ estrutural, `[A]` no valor |
| 10 | `D_def` pela Eq. 2 | derivado, não gravado | ✓ desvio **declarado** |

---

## 2. Eq. 5 — a fórmula, o que é otimizado, e o critério de parada

### 2.1 O que o paper publica

```
paper.txt:393   K* = argmax_{K ∈ (K_min, K_max)}  SSIM( R(I_aif, D; D_focus, K), I_real )
```

- **O que é otimizado**: um escalar `K`, no intervalo **aberto** `(K_min, K_max)`.
- **Contra o quê**: `I_real`, a **fotografia real** com bokeh do par. Não é uma imagem
  renderizada — o alvo do argmax é o alvo da supervisão.
- **Com que função objetivo**: SSIM, imagem inteira. O paper não menciona máscara,
  recorte, canal ou escala.
- **O que é `R`**: *"our physically guided renderer"* (`paper.txt:396`). O corpo do
  artigo **não o identifica**; quem fecha é o supplement B.2: *"we then optimized the
  parameter K using **simulator [43]**"* (`paper.txt:1007`), e `[43]` = BokehMe
  (`paper.txt:851-852`).
- **Critério de parada**: o paper **não publica nenhum**. Nem algoritmo de busca, nem
  tolerância, nem número de avaliações, nem `K_min`/`K_max`.
- **Critério de aceitação**: *"provided that its corresponding SSIM exceeds a
  predefined threshold"* (`paper.txt:397-398`), **valor não publicado**.

### 2.2 O que o paper diz sobre busca em K, em outro lugar

Não é o §3.2(c), mas é o mesmo problema e é a única pista de método que os autores
dão: na avaliação, *"we conduct a **per-image binary search** over K and select the
value that **maximizes SSIM** with the target"* (`paper.txt:561-562`), e de novo em
`paper.txt:579-580`. Busca binária sobre um argmax **pressupõe unimodalidade** — a
mesma hipótese que a seção áurea faz.

### 2.3 Nosso `calibrate_k` corresponde?

**Correção de vocabulário**: o enunciado da auditoria diz "busca ternária". O código
faz **seção áurea** (`calibration.py:41`, `:177-191`), que é a variante que reaproveita
uma avaliação por iteração. A diferença importa só no custo; as duas assumem a mesma
unimodalidade. `REGISTRO.md:321-330` já registra a escolha.

O algoritmo, em duas fases:

1. **Grid grosso** de 7 pontos em `[k_min, current_max]`, **expandindo o teto ×2**
   enquanto o máximo cair na última posição, até `k_absolute_max`
   (`calibration.py:165-175`).
2. **Seção áurea** dentro do bracket `[grid[best-1], grid[best+1]]`, até
   `hi - lo <= tolerance = 0.25` ou esgotar 40 avaliações (`calibration.py:177-191`).

| aspecto | paper | nosso código | veredito |
|---|---|---|---|
| o que otimiza | `K` escalar | idem, `k_full` na escala da imagem original | ✓ |
| contra o quê | `I_real` | `target_bgr` = `image_blur` do espelho (`route_c.py:247`) | ✓ |
| objetivo | SSIM | SSIM próprio em numpy, Wang 2004, janela 11×11 σ=1,5 (`metrics.py:56-73`) | ✓ com `[A]` (§2.5) |
| renderizador | `[43]` | `BokehMeRenderer` (`bokehme.py:199-348`), commit + 4 sha256 na proveniência | ✓ |
| intervalo | `(K_min, K_max)` **não publicado** | `[0.5, 120]`, expandindo até `960` | `[A]`, e é o **C4** |
| parada | **não publicado** | `hi-lo <= 0.25` ou 40 avaliações | `[A]` declarado (`calibration.py:18-19`) |
| limiar de SSIM | existe, valor **não publicado** | gate com default `None` | ✓ honesto (§2.3 abaixo) |

**Medido nesta auditoria** `[M]`, com o SSIM substituído por um pico sintético exato
para isolar a busca dos dublês de renderer:

```
k_verdadeiro    K* recuperado    erro      avaliações   expansões
    3,6            3,606        0,17%          19           0
   20,0           20,031        0,15%          20           0
   60,0           59,988        0,02%          20           0
  150,0          150,000        0,00%          27           1
  300,0          300,001        0,00%          35           2
  700,0          699,949        0,01%          40           3
```

A busca **fecha** em toda a faixa que a rota C pode encontrar, e o orçamento de 40
avaliações é **exatamente** suficiente no pior caso (3 expansões). Não sobra margem:
qualquer aumento de `coarse_points` ou da faixa trunca a seção áurea em silêncio.
`evaluations` fica gravado em `search` (`calibration.py:220`), então é auditável — mas
nada marca "estourou o orçamento". Ver C4 e §5.3.

### 2.4 O limiar de SSIM: o paper afirma e não publica — tratamos honestamente?

**Sim, e com folga.** As duas citações (`paper.txt:397-398` e `paper.txt:1008-1010`)
estão **transcritas dentro da docstring do gate** (`gates.py:316-321`), o default é
`None`, e há **teste que faz o gate reprovar** com limiar explícito
(`test_gates.py:140-146`: `calibration_ssim_is_reliable(0.42, min_ssim=0.60).passed`
é `False`) e **teste ponta a ponta** que prova que o slug certo chega ao histograma
(`test_route_c.py:166-170`). O `CONTRATO.md:145` lista "o limiar de SSIM" entre o que
o paper não publica. `calibrate_thresholds.py:114-158` implementa o caminho
**recomendado** — painel humano revisado, corte por precisão com monotonicidade a
partir do topo — em vez de percentil cego, e diz por escrito que percentil sozinho é
ponto de partida ruim (`calibrate_thresholds.py:17-20`).

É o tratamento mais honesto de `[A]` que vi neste repositório. **Nada a mudar.**

Uma ressalva de método, e não de código: o SSIM é calculado na resolução de trabalho
de 512 (`route_c.py:105`, `calibration.py:141-146`). O valor de SSIM **depende da
resolução**. O limiar congelado a partir do piloto só vale enquanto
`calibration_long_side` não mudar. Isso está em `CONTRATO.md:130` como decisão
declarada, mas **não está gravado ao lado do limiar** — está em `search.work_long_side`
por amostra (`calibration.py:223`), o que basta para auditar. Suficiente.

### 2.5 `k_analytic` não é comparável com `K*` — **C1, e é o achado mais importante**

`route_c.py:294-323` calcula `k_analytic` pela Eq. 3 sobre o `metadata/` e o apresenta
como *"validador independente do sweep"*. `RouteCStats.summary` imprime, logo abaixo
dos percentis de `k_value`:

```
route_c.py:150     "âncora da rota C: 3,6 a 36 (Eq. 3 sobre o metadata)"
calibrate_thresholds.py:209     "âncora da rota C: 3,6 a 36"
```

**As duas grandezas não são a mesma, e a diferença é grande e sistemática.**

A AIF da rota C **não é all-in-focus**: é a fotografia `train/in/<cena>_f22.JPG`, uma
foto real a **f/22** (`ACHADOS.md`: `source_av = 22,0` em 4.400/4.400 cenas [M]). Ela
carrega o próprio desfoque. O que a Eq. 5 mede é o borrão **incremental** que leva da
foto f/22 até a foto f/F_alvo. O que a Eq. 3 dá é o borrão **absoluto** da foto f/F_alvo.

Dentro de uma cena, todos os termos da Eq. 3 são idênticos entre níveis — mesma
`focal_length`, mesma `focus_plane_distance`, mesma resolução — e `K ∝ 1/F`. Logo,
exatamente:

```
K_eq3(F_aif=22) / K_eq3(F_alvo) = F_alvo / 22
```

Compondo os borrões de forma **linear no raio**, `K* ≈ K_eq3(alvo) · (1 − F_alvo/22)`;
compondo em **quadratura**, `K* ≈ K_eq3(alvo) · √(1 − (F_alvo/22)²)`. A física do
scatter de disco fica entre as duas. A razão `K*/k_analytic` esperada, `[I]` (álgebra
da Eq. 3, não medição):

| `F_alvo` | linear | quadrática | `k_analytic` superestima por |
|---|---|---|---|
| f/2,0 | 0,909 | 0,996 | 1,0× a 1,1× |
| f/5,6 | 0,745 | 0,967 | 1,0× a 1,3× |
| f/11 | 0,500 | 0,866 | 1,2× a **2,0×** |
| **f/14** (mediana medida) | 0,364 | 0,771 | 1,3× a **2,7×** |
| f/20 | 0,091 | 0,417 | 2,4× a **11×** |

A distribuição de f-stop da RealBokeh é **fechada**: o maior f-stop por cena tem
mediana **f/14**, e só 3,0% das cenas chegam a f/2,8 ou mais aberto (`ACHADOS.md`) [M].
Logo a maioria dos alvos vive na faixa em que `k_analytic` erra por **2× ou mais**.

Duas consequências, e as duas são caras:

1. **O validador não valida.** Quem olhar o piloto e vir `K*` mediano em, digamos, 6
   contra `k_analytic` mediano em 16 vai concluir que o sweep está quebrado. Ele não
   está: a razão é justamente o que a álgebra prevê. Um "conserto" aqui reintroduz o
   defeito.
2. **A âncora impressa é a âncora errada.** `3,6 a 36` é a faixa de `k_analytic`, não
   de `K*`. `route_c.py:150` e `calibrate_thresholds.py:209` colam essa faixa embaixo
   dos percentis de `k_value`, que é exatamente o tipo de justaposição que produz
   conclusão errada.

**Conserto (não aplicado)**: gravar `k_analytic_incremental = k_analytic · (1 −
F_alvo/F_aif)` ao lado do absoluto, e trocar a linha impressa por *"esperado ≈
`k_analytic·(1 − F/22)`, entre 0,4 e 33"*. O plot que fecha a questão é
`K*/k_analytic` contra `F_alvo` no piloto: se ele cair entre as curvas linear e
quadrática, o sweep está certo e a composição de borrão fica **medida** em vez de
`[I]`. É a medição mais barata e mais informativa que o piloto pode produzir.

Nota lateral, e é boa: os valores esperados de `K*` (0,4 a 33) encaixam melhor no que
o paper mostra — `K ∈ {0, 5, 10, 15}` na Fig. 12 (`paper.txt:1151,1154`) e default 15,0
na inferência oficial — do que os 3,6 a 36 da Eq. 3.

### 2.6 Um erro do próprio paper, registrado

A legenda da Fig. 3 diz, sobre a rota (c): *"we obtain Dfocus as in (b), and **follow
Eq. (2)** to estimate the bokeh level K"* (`paper.txt:296-297`). A Eq. 2
(`paper.txt:312`) é `D_def = K·|D − D_focus|` — ela **consome** K, não o estima. O
corpo (`paper.txt:369-370`, `:391-398`) e o supplement (`paper.txt:1007`) são
inequívocos: o K da rota C vem do sweep, Eq. 5.

É a mesma regra que já governa a numeração de referências: **legenda de figura não
manda; corpo e bibliografia mandam**. Registrado aqui para não custar meia hora a
quem ler a legenda primeiro.

---

## 3. Escopo das equações — Eq. 3 não vale aqui, Eq. 5 só vale aqui

### 3.1 A Eq. 3 não vale na rota C

**Confirmado, e por dois caminhos independentes.**

1. **Posição no texto**: a Eq. 3 (`paper.txt:340`) está dentro do parágrafo
   *"(b) ITW dataset"*, que vai de `paper.txt:336` a `:352`. O parágrafo (c) começa em
   `:359`.
2. **Afirmação explícita**: o (c) abre dizendo que aqueles datasets *"omit EXIF
   metadata or provide insufficient fields to estimate K"* (`paper.txt:359-360`) — a
   Eq. 3 é justamente o que não dá para calcular ali.

Concordo integralmente com `ROTA_B_AUDITORIA.md:96-101` e com `CONTRATO.md:158`.
Medido no dado v0: `exif` 100% NULA na rota C, `fx` derivável em 0/2.932 [M]
(`ACHADOS.md`).

### 3.2 A Eq. 5 só vale na rota C

**Confirmado.** A Eq. 5 aparece uma vez no artigo, dentro do (c). Nas outras duas:

- **Rota A**: *"we randomly sample a focus plane Dfocus and a target bokeh level K"*
  (`paper.txt:329-330`) — sorteio, sem equação. Confirmado por
  `ROTA_A_AUDITORIA.md:82-104`.
- **Rota B**: Eq. 3 a partir da EXIF (`paper.txt:337-340`).

A ocorrência de busca por SSIM em `paper.txt:561-562` e `:579-580` é **avaliação**, não
geração de rótulo — outro escopo.

### 3.3 Usar a Eq. 3 como *validador* na rota C é legítimo?

**Sim, com uma condição que hoje não está satisfeita.**

O que o paper proíbe é usar a Eq. 3 como **rótulo** na rota C — e a proibição é
factual, não normativa: os campos não existem. Nada no artigo proíbe usar uma
quantidade calculada de metadados como instrumento de auditoria. A própria rota B faz
o análogo: a distância de foco da EXIF *"serve como validador [...] nunca como rótulo"*
(`ROTA_B_AUDITORIA.md:175-179`).

E aqui o validador é **mais forte que na rota B**, por um motivo real: o
`focus_plane_distance` da RealBokeh é **profundidade métrica medida na cena**, com
barra de erro (`focus_plane_uncertainty`, `ACHADOS.md`) — não estimada por modelo.
Alimentar o validador com `1/focus_disparity` o faria depender do Depth Pro e do
BiRefNet, os mesmos dois modelos que produzem o valor sendo validado, e ele deixaria de
validar. Isso está corretamente argumentado em `route_c.py:300-309`, com instrução
explícita para ninguém "harmonizar" com a rota B. **Concordo e endosso.**

**A condição que falta é a de §2.5**: um validador cuja escala difere da grandeza
validada por um fator sistemático de 1× a 11%, dependente de `F_alvo`, não é validador
— é armadilha. Corrigido o fator, é legítimo e vale a pena.

**Segunda condição, menor**: hoje `_analytic_k` devolve `None` **sempre** na RealBokeh,
porque `RealBokehPair` não tem `sensor_width_mm` (`realbokeh.py:537-597`) e
`route_c.py:315-317` sai por `getattr(..., None)`. O validador só existe se alguém
passar `--sensor-width-mm` (`run_route_c.py:216-219`, `_ComSensor` em `:67-87`). A
âncora `3,6 a 36` do `CONTRATO.md:79` foi calculada com sensor **full-frame de 36 mm**
`[I]` (confere numericamente: `pixel_ratio = 2000/36 = 55,6 px/mm`, e `f=50 mm`,
`z=3 m`, `F=2` dá `K = 35,3`). Não há 36,0 cravado no laço — o que é correto e
declarado (`REGISTRO.md:752-758`) —, mas o resultado prático é que **o validador está
desligado por default**. `publish_release.py:112,175` conta e reporta
`sem_validador_analitico`, sem reprovar. Correto.

---

## 4. A fonte — o LFDOF é exigido, e o que ele exige de diferente

### 4.1 O paper exige, em três lugares

- `paper.txt:326`: *"(c) LFDOF [52] and RealBokeh [57]"* (a enumeração das rotas)
- `paper.txt:359`: o título do parágrafo (c)
- `paper.txt:528-529`: *"26K real examples sourced from ITW dataset [19], RealBokeh
  [57], and **LFDOF [52]**"*
- `paper.txt:296`: legenda da Fig. 3(c)

**Não é opcional.** A rota C do paper são dois datasets.

### 4.2 O que o LFDOF exige de diferente — e o defeito C2

`akcit-pixel/LFDOF`: train **11.247**, test 725 [M] (`ACHADOS.md`). AIF **real**,
renderizada da light field — melhor que a AIF f/22 da RealBokeh. N desfocadas por AIF.

Sete diferenças, em ordem de impacto:

1. **`f_number` e `aif_f_number` não existem.** O LFDOF sintetiza o desfoque a partir
   da light field; não há abertura fotográfica. O `PairSource` já prevê isso
   (`route_c.py:83-84`, `:90`). **Mas o gate não.**

   **Medido nesta auditoria** `[M]`:
   ```
   aif_aperture_is_narrow(None) -> value=nan  threshold=None
                                   measured_only=True  passed=FALSE
   process_pair(par com aif_f_number=None)
                                -> REJEITADA: gate_aif_aperture_wide
   ```
   `GateResult.passed` (`gates.py:54-59`) devolve `False` para valor não-finito
   **antes** de olhar o limiar. Isso é deliberado e correto para `mask_iou`
   (`gates.py:122-126`: duas máscaras vazias não são concordância) e para
   `calibration_ssim`. Mas em `aif_aperture_is_narrow` (`gates.py:305-307`) o `NaN`
   significa literalmente *"f-stop da AIF desconhecido"* — que é a condição **normal**
   do LFDOF, não um defeito.

   **Resultado: 100% do LFDOF seria rejeitado**, com slug `gate_aif_aperture_wide`
   ("abertura larga"), que afirma algo falso sobre o dado. É o formato exato do defeito
   que este projeto está desfazendo: rejeição em massa com motivo registrado e
   conclusão errada — o mesmo padrão do `source_duplicate_sample` da Etapa 7
   (`REGISTRO.md:857-861`).

   `test_nenhum_gate_bloqueia_por_default` (`test_gates.py:130-138`) testa
   `aif_aperture_is_narrow(2.0)` e **não** testa `aif_aperture_is_narrow(None)`. O caso
   passou por dez agentes e uma suíte de 239 testes.

   **Conserto**: separar "desconhecido" de "indefinido". Ou o gate devolve
   `value=+inf`/`threshold=None` quando a origem não publica f-stop (com nota
   `"origem não publica f-number"`), ou `GateResult` ganha um terceiro estado
   `not_applicable` que não bloqueia e é contado à parte no histograma. A segunda é
   melhor: `focus_mask_is_sharpest` com área insuficiente (`gates.py:161-163`) tem o
   mesmo problema latente.

2. **Sem `focal_length` e sem `focus_plane_distance`**, logo `_analytic_k` devolve
   `None` — nenhum validador, nem com correção. Só a Eq. 5.

3. **N desfocadas por AIF ⇒ split por cena obrigatório.** Já está resolvido no
   protocolo: `scene_id` é campo do `PairSource` (`route_c.py:76`) e o split é por cena
   e materializado (`split.py:1-17`). O adaptador só precisa produzir `scene_id`
   estável — e, pela lição da Etapa 7 (`REGISTRO.md:841-870`), **qualificado pelo
   split de origem** se a numeração reiniciar.

4. **A AIF é genuinamente all-in-focus.** Some o offset sistemático de §2.5: no LFDOF,
   `K*` da Eq. 5 é o K absoluto, comparável direto com o de outras fontes. É a fonte
   que **calibra a interpretação da RealBokeh**, e por isso vale rodar as duas.

5. **`aif_aperture_is_narrow` e `bokeh_is_blurrier_than_aif`**: o primeiro não se
   aplica (item 1); o segundo se aplica e continua sendo o gate certo.

6. **Registro geométrico é perfeito por construção** (mesma light field), então a
   anotação de alinhamento que a RealBokeh publica (`alignment`, 2,05% não-`aligned`
   [M]) não tem análogo — e não deve ser inventada.

7. **Licença**: *"Sem licença explícita na página"* [M] (`ACHADOS.md`). Isso é
   bloqueante para `--store-source-images` (release autocontido), não para o rótulo.
   `SampleProvenance.source_license` existe (`sample.py:94`) e hoje entra como `None`
   (`route_c.py:286`). Antes de publicar pixels do LFDOF, resolver a licença.

**Custo de escrever o adaptador**: baixo. A rota recebe pares **por protocolo**
(`route_c.py:69-90`) e `run_route_c.py:145-150` já diz isso por escrito. O que
**não** é baixo é o item 1: sem ele, o adaptador roda e rejeita tudo.

### 4.3 A RealBokeh — o que já está certo

Vale registrar, porque é bastante: o D5 e o D6 estão **estruturalmente** consertados.
A AIF vem de `train/in/<cena>_f22.JPG` via `image_focus` do espelho, **confirmado byte
a byte por sha256** na cena 1038 [M] (`ACHADOS.md`), e não mais do maior f-stop dentro
de `gt/` (mediana f/14, 12,7% das cenas a f/5.6 ou mais aberto [M]). O `level` é
1-based por três medições independentes, índice fora da lista **rejeita** em vez de
fazer clamp (`realbokeh.py:496-501`), e a ordem de `target_avs` é conferida contra o
nome do arquivo par a par (`realbokeh.py:512-521`).

**Documentação obsoleta, sem efeito no rótulo**: `realbokeh.py:733-738` e
`run_route_c.py:158-168` ainda afirmam *"o espelho publica só `train` (20.495 de
20.495)"*, e `run_route_c.py:202` diz "85 shards". A Etapa 7 mediu **96 shards e três
splits** (`ACHADOS.md`, `REGISTRO.md:841-856`). O código está certo — `_monta_split`
decide por `counts()["val"] > 0` (`run_route_c.py:170-182`) —, só o texto envelheceu.

---

## 5. Máscara e plano de foco — de qual imagem sai a máscara?

### 5.1 O paper não diz. É `[A]`.

O §3.2(c) diz *"Similar to (b), we employ BiRefNet [86] to obtain an initial in-focus
mask M"* (`paper.txt:361-362`) — e o (b) também não diz. O mais próximo de uma
afirmação é a ordem na legenda da Fig. 3(b): *"DeblurNet recovers an AIF image. **We
then** estimate depth and extract a foreground mask [86]"* (`paper.txt:293-294`). Isso
**sugere** a AIF, e é sugestão, não afirmação.

O que o paper **afirma** é só sobre a profundidade: *"D is the monocular depth map
estimated **from Iaif**"* (`paper.txt:314`). Da máscara, nada.

`ROTA_B_AUDITORIA.md` registra isso como `[A]` A2, risco **médio**, com o argumento de
que numa bokeh a região nítida *é* a região em foco. **Concordo, e na rota C o
argumento é mais forte, não mais fraco**: aqui a bokeh real está disponível desde o
começo (é o alvo da Eq. 5), enquanto na rota B ela é a entrada da DeblurNet.

### 5.2 O que `process_pair` faz

```
route_c.py:239   depth = depth_runtime.infer(aif_rgb)      # AIF  ✓ (paper.txt:314)
route_c.py:240   mask  = mask_runtime.infer(aif_rgb)       # AIF  [A]
route_c.py:241   focus_disparity = focus_disparity_from_mask(depth.values_m, mask)
route_c.py:244   mask_bokeh = mask_runtime.infer(bokeh)    # só para MEDIR divergência
```

A escolha está comentada em `route_c.py:237-238` e justificada pelo defeito D12 — usar
a máscara da bokeh com a profundidade da AIF. Esse argumento é bom contra **misturar**
as duas, e não decide **qual das duas** usar de forma consistente. Usar as duas da
bokeh seria igualmente coerente, e é uma leitura defensável do paper.

**Divergência: nenhuma — é `[A]` legítimo.** Mas com três observações:

1. **A alternativa já está calculada.** `mask_bokeh` é inferida em toda amostra
   (`route_c.py:244`) só para alimentar `mask_iou`. Trocar a fonte de `focus_disparity`
   custa uma linha, e o piloto pode gravar **os dois** `focus_disparity` para decidir
   com número em vez de com argumento. É a medição mais barata deste `[A]`.
2. **O gate que testa a hipótese mede na bokeh, com a máscara da AIF.**
   `focus_mask_is_sharpest(bokeh_bgr, mask, ...)` (`route_c.py:181-182`,
   `gates.py:137-170`) — nitidez dentro sobre fora da máscara, **na imagem bokeh**.
   Está certo e é o gate mais importante do módulo: é o único que pega o modo de falha
   dominante (fotógrafo focou o fundo, BiRefNet marcou o objeto saliente, as duas
   máscaras concordam no objeto errado e a IoU vale 1,0). Testado nos dois sentidos
   (`test_gates.py:68-88`).
3. **Custo**: duas inferências de BiRefNet por par, 22.990 pares. Não é erro, é
   escolha; só não está no orçamento de tempo em lugar nenhum.

### 5.3 Um detalhe de fidelidade que a rota C tem e a B não

O paper diz que na rota (b) a AIF é produzida pela DeblurNet (`paper.txt:293`,
`:337-338`). Na (c) ele diz *"These datasets provide pairs"* (`paper.txt:359`) — a AIF
**vem da origem**. Nosso código não roda DeblurNet na rota C. ✓ **Confere.**

Ressalva `[A]` que o paper não trata e nós herdamos: na RealBokeh a "AIF" é uma foto
**f/22**, não all-in-focus. É a raiz de C1 (§2.5) e vale para tudo que a AIF alimenta —
Depth Pro, BiRefNet e o sweep.

---

## 6. O renderizador — o BokehMe produz o alvo da rota C?

**Não. O alvo é a fotografia real. O BokehMe aparece só dentro do sweep da Eq. 5.**

A distinção que a auditoria da rota B fez vale aqui, com uma diferença importante.

### 6.1 O que o paper diz

- A Eq. 5 compara `R(...)` **contra** `I_real` (`paper.txt:393`) — `I_real` é o alvo, e
  é a fotografia. Se o renderizador produzisse o alvo, a equação seria uma
  tautologia.
- A tupla de supervisão é `(I_aif, I_out, D_def)` (`paper.txt:321`); na Fig. 3(c) a
  entrada é literalmente *"Real bokeh image · Real AIF image"* (`paper.txt:264`,
  `:271`).
- O paper chama `R` de *"our physically guided renderer"* (`paper.txt:396`) e **não o
  nomeia no corpo**. Quem fecha é o supplement: *"we then optimized the parameter K
  using **simulator [43]**"* (`paper.txt:1007`).

**A diferença entre as rotas**, e ela é real:

| rota | quem produz o alvo | onde o `[43]` aparece |
|---|---|---|
| A | **o BokehMe** (`paper.txt:292`, `:331`) | produz o alvo, citação literal duas vezes |
| B | **ninguém** — o alvo é a foto | não aparece |
| **C** | **ninguém** — o alvo é a foto | **dentro do sweep**, e só |

Concordo com `ROTA_B_AUDITORIA.md:139-153` e `ROTA_A_AUDITORIA.md:105-118`.

### 6.2 O que o código faz

✓ **Confere, e está bem construído.**

- `calibrate_k` recebe `target_bgr = bokeh_bgr` (`route_c.py:247`), que vem da origem
  via `MirrorImageLoader.__call__` (`mirror_images.py:252-272`).
- `Sample.generated_images` fica **vazio** na rota C (`route_c.py:267-291` não o
  preenche), então `FileSampleWriter` não escreve nada em `generated/`
  (`writer.py:118-128`). Coerente com a tabela de `sample.py:7-11`.
- O renderizador é o `[43]` público, com proveniência forte: commit, sha256 do
  `pipeline` extraído por AST, sha256 do `scatter.py` **depois do patch**, e sha256 de
  `arnet.pth`/`iunet.pth` (`bokehme.py:289-306`).
- `run_route_c.py:276-284` **recusa rodar** se o laudo do renderer for de outro
  checkout, comparando três hashes. Bom.

### 6.3 Duas ressalvas sobre `is_final_label_renderer`

**(a) O nome afirma mais do que deveria.** `bokehme.py:305` grava
`is_final_label_renderer: True`. Na rota C isso é **verdade para o rótulo K** e
**falso para a imagem** — o BokehMe não produziu pixel nenhum que vá para o release.
`ROTA_B_AUDITORIA.md:152-153` já chamou o mesmo campo de "proveniência que mente na
direção tranquilizadora" no contexto da rota B. Aqui não mente, mas é ambíguo.
Sugestão barata: acrescentar `produces_target_image: False` ao lado, na rota C.

**(b) Um dos quatro critérios do laudo não pode reprovar.**

```
verify_renderer.py:123   usa_highlight = "highlight" in pipeline_src
verify_renderer.py:125   report["passed"]["highlight_decidido"] = True     # <- constante
verify_renderer.py:138   apto = all(report["passed"].values())
```

`highlight_decidido` é `True` incondicionalmente e entra no `all(...)` que autoriza
`is_final_label_renderer`. É o padrão do `check_no_leak` antigo com granularidade
menor: um critério que compõe um veredito e nunca pode contribuir com `False`. Os
outros três são medições de verdade (`disco_nao_gaussiana`, `linear_em_k`,
`escala_bate_com_contrato`, confirmados em `output/renderer_verification.json`), então
o impacto prático é nulo hoje — mas o campo pertence a `measurements`, não a `passed`.
Se ficar em `passed`, tem que assertar algo: por exemplo, que `usa_highlight` é `False`
**ou** que `BokehMeConfig.highlight` foi mexido de propósito.

**(c) A faixa verificada não cobre a faixa autorizada.** O laudo mediu linearidade em
`K ∈ {8, 16, 32, 64, 96}` (`output/renderer_verification.json`), raios de 3,2 a 38,4 px,
com resíduo 1,09% e `slope = 0,9873` [M]. `K_ABSOLUTE_MAX_DEFAULT = 960`
(`calibration.py:38`) autoriza renderizar **10× além** do que foi medido. Ver C4.

---

## 7. Dez gates automáticos no lugar da revisão humana — é substituição honesta?

**É honesta, e não é equivalente — e o código diz as duas coisas por escrito.**

### 7.1 O que o paper faz, exatamente

Duas atividades humanas distintas, e vale separá-las:

1. **Refinamento de máscara** (`paper.txt:362-368`, `:1003-1006`): re-selecionar
   manualmente uma região pequena e confiável em foco, para **corrigir** M. 4 a 8
   segundos por imagem, ~8 horas. O paper diz explicitamente que faz isso **em vez de**
   verificar-e-descartar, e diz o porquê: *"preserves challenging samples rather than
   excluding them"*.
2. **Filtro automático por SSIM** (`paper.txt:396-398`, `:1008-1010`): o limiar sobre o
   SSIM da Eq. 5. **Este é automático no paper também.**

### 7.2 O que substituímos, e o que não

| o que o paper pega | como | nós | honesto? |
|---|---|---|---|
| máscara ruim | **corrige à mão** | `mask_area_ratio`, `mask_border_coverage`, `focus_mask_is_sharpest`, `mask_iou` — **descartam** | **desvio declarado**, direção do viés escrita |
| K ruim | limiar de SSIM | `calibration_ssim_is_reliable` | ✓ **mesma coisa** |
| — | — | `pair_shape_matches`, `aif_sharpness`, `bokeh_is_blurrier_than_aif`, `aif_aperture_is_narrow`, `depth_useful_levels`, `focus_depth_plausible` | **acréscimos nossos**, sem análogo no paper |

Ou seja: dos dez gates, **um** corresponde a um passo do paper (o de SSIM), **quatro**
substituem a revisão humana por descarte, e **cinco** são gates novos que o paper não
tem. Isso é o que a substituição é, e está corretamente descrito em `gates.py:15-27`
e `route_c.py:33-39`.

**A declaração é boa**: o texto nomeia a direção do viés (*"contra cena complexa —
exatamente a direção que o paper diz ter evitado de propósito"*) em vez de afirmar
equivalência. Não vi nenhuma frase no repositório dizendo "equivale à revisão manual".

### 7.3 Algum gate mede coisa diferente do que a revisão humana pegaria?

**Sim, e é o melhor gate do módulo.**

`focus_mask_is_sharpest` (`gates.py:137-170`) mede nitidez dentro/fora da máscara **na
imagem bokeh**. A revisão humana do paper corrige a máscara quando ela está mal
desenhada. O gate pega um caso **diferente e mais grave**: a máscara está bem
desenhada, sobre o objeto errado — o fotógrafo focou o fundo e o BiRefNet marcou o
objeto saliente em primeiro plano. Nesse caso as duas máscaras automáticas concordam,
a IoU vale 1,0, e o `D_focus` inteiro está errado. Um revisor humano "re-selecionando
uma região confiável em foco" (`paper.txt:365-366`) **pegaria** esse caso, porque um
humano vê onde está o foco. Um gate de IoU **não** pega. Testado nos dois sentidos
(`test_gates.py:73-88`).

Então: o gate cobre parte do que o humano cobre, e cobre por outro caminho. Bom.

**Onde a cobertura é pior que a humana**, e vale escrever:

- Nenhum gate **corrige**. Uma cena com máscara 80% certa é descartada inteira; o
  humano do paper recuperaria.
- Nenhum gate olha a cena. `mask_border_coverage` e `mask_area_ratio` são heurísticas
  de forma; um humano distingue "objeto grande legítimo" de "máscara vazando para o
  fundo".
- **`bokeh_is_blurrier_than_aif` não pega o par f/14 contra f/2,0** — está documentado
  na própria docstring (`gates.py:274-278`) e travado por teste
  (`test_gates.py:115-121`). Para esse caso o gate certo é o de f-stop. Documentar um
  limite em vez de fingir cobertura é a coisa certa.

### 7.4 Modo medir até o piloto — mas não é bem "todos"

A regra está enunciada em `gates.py:1-6` e `run_route_c.py:230-233`: *"TODOS default
None = mede e não bloqueia"*. **Duas exceções, e uma importa** — é o **C5**:

1. **`pair_shape_matches`** tem limiar `1.0` cravado (`gates.py:263`). É estrutural, é
   declarado em `gates.py:10-11`, e na prática é código morto: `process_pair:231-233`
   já rejeitou com `resolution_invalid` antes. Defesa em profundidade. **Sem problema.**

2. **`focus_depth_plausible`** tem defaults **não-`None`** vindos de
   `FOCUS_DEPTH_MIN_M = 0.05` e `FOCUS_DEPTH_MAX_M = 1000.0` (`gates.py:231-235`,
   `contract.py:243-244`). E `focus_disparity_from_mask:281-282` já rejeita com os
   mesmos limites, **antes** dos gates. Ou seja: **dois `[A]` explicitamente marcados
   como "não medidos, calibrar no piloto" (`contract.py:238-242`) bloqueiam desde a
   primeira amostra.**

   Pior: **o piloto não consegue calibrá-los.** A amostra é rejeitada em
   `contract.py:282`, então seu `z_focus` nunca entra em `quality`;
   `calibrate_thresholds._distributions` (`calibrate_thresholds.py:77-85`) lê só as
   **aceitas**; e `_GATES` (`:64-74`) tem **nove** entradas e não inclui
   `focus_depth_m_min`/`max`. Além disso não existe flag `--focus-depth-max-m` em
   `run_route_c.py` — mudar o valor exige editar `contract.py`.

   O valor rejeitado **está** no `detail` da linha do `rejections.jsonl`
   (`contract.py:282`, `rejection.py:50-55`), então é recuperável por parsing de
   string. Nenhuma ferramenta faz isso.

   **Impacto: não medido** para a rota C. Na rota B, `z_focus_m` medido vai até
   10.000 m e `z_max == 10.000` exato ocorre em 25,7% das amostras [M]
   (`ACHADOS.md`) — mas isso é `z_max`, não `z_focus`, e a rota C tem `Δ` de cena
   diferente. O `focus_plane_distance` publicado pela RealBokeh vai de 0,355 a
   368,63 m [M], **dentro** da faixa; mas o `focus_disparity` que o gate julga vem do
   Depth Pro + BiRefNet, não do metadata. É exatamente o que o piloto precisaria medir
   e hoje não mede.

   **Conserto**: expor `--focus-depth-min-m`/`--max-m` no CLI, passá-los a
   `focus_disparity_from_mask` e ao gate a partir do `RouteCConfig`, e rodar o piloto
   com a faixa larga (por exemplo `[0.01, 1e6]`) para que a distribuição apareça.
   Congelar depois. Alternativa mínima: acrescentar as duas entradas a `_GATES` e fazer
   `calibrate_thresholds` ler os `detail` das rejeições.

---

## 8. VEREDITO

**Não. A rota C não está pronta para gerar o dado final.**

Está pronta para **rodar o piloto** — e deve rodar, porque quatro dos cinco achados só
fecham com número do piloto. O que não está pronto é o run completo de 20 mil amostras.

Justificativa, em uma linha por item:

- **C2 é bloqueante para "rota C"** no sentido do paper: com o gate como está, o LFDOF
  — que `paper.txt:326,359,528-529` exige em três lugares — é rejeitado 100%. Sem
  LFDOF, o que temos é meia rota C.
- **C1 torna o piloto ilegível**: o único instrumento de sanidade que o run imprime
  para `k_value` é uma âncora que descreve outra grandeza. Rodar o piloto sem
  consertar isso é rodar sem saber se o resultado está certo.
- **C3 não tem conserto retroativo barato**: gerar 20.495 amostras e depois descobrir
  que 25% delas vêm de 6% das cenas significa refazer o split e a contagem. Decidir
  antes custa uma função.
- **C4 é custo e regime não medido**, e o custo do sweep é o item dominante do
  orçamento de GPU.
- **C5 é honestidade de método**: dois `[A]` filtrando enquanto o documento diz que
  nenhum `[A]` filtra.

O que **está** pronto, e é muito: o contrato fecha dimensionalmente, o renderer está
verificado contra o BokehMe real com número (`ACHADOS.md`, job 32212), não há fallback
numérico no caminho do rótulo, a proveniência é forte, o split é por cena e
materializado, a censura é gravada em vez de virar medida, e 239 testes passam [M].

---

## 9. O que PRECISA mudar antes de rodar — por impacto no rótulo

### P0 — sem isto o rótulo nasce errado ou ilegível

1. **C1 · Corrigir a comparação `K*` × `k_analytic`.** Gravar
   `k_analytic_incremental = k_analytic · (1 − F_alvo/F_aif)` ao lado do absoluto;
   trocar a linha de âncora em `route_c.py:150` e `calibrate_thresholds.py:209`;
   acrescentar ao relatório do piloto o gráfico `K*/k_analytic` contra `F_alvo`.
   *Sem isso, ninguém consegue dizer se o sweep está certo.*

2. **C2 · Separar "desconhecido" de "indefinido" em `GateResult`.** Um terceiro
   estado `not_applicable` que não bloqueia e é contado à parte, aplicado a
   `aif_aperture_is_narrow` quando a origem não publica f-stop e a
   `focus_mask_is_sharpest` quando a área após erosão é insuficiente. Manter `NaN`
   bloqueando onde o `NaN` **é** o defeito (`mask_iou` com duas máscaras vazias,
   `calibration_ssim` ausente). Teste que falha hoje:
   `aif_aperture_is_narrow(None).passed is True`.
   *Sem isso, o adaptador do LFDOF roda e rejeita tudo.*

3. **C3 · Decidir e declarar a multiplicidade por cena.** O paper diz 2 a 4 por
   conjunto (`paper.txt:1001-1003`); nós pegamos até 21. Ou aplicar um teto sorteado
   com seed (onde `sample_pairs_for_pilot` já vive), ou declarar por escrito que
   rodamos 1,6× o volume do paper com concentração medida de 25%/6,2%. Não deixar
   acontecer por omissão.

### P1 — sem isto o piloto não calibra o que precisa calibrar

4. **C5 · Expor `FOCUS_DEPTH_MIN_M`/`MAX_M` como limiares de verdade.** Flags no CLI,
   passadas a `focus_disparity_from_mask` e ao gate; rodar o piloto com faixa larga;
   acrescentar as duas entradas a `_GATES` em `calibrate_thresholds.py:64-74`.

5. **`min_aif_f_number` não aparece em `_GATES`.** São 10 limiares em `RouteCConfig` e
   9 entradas em `_GATES`. Na RealBokeh o valor é 22,0 constante e o percentil não
   informa nada — mas então a ferramenta devia dizer isso, não omitir a linha.

6. **Gravar, no piloto, os DOIS `focus_disparity`** — o da máscara da AIF e o da
   máscara da bokeh. `mask_bokeh` já é calculada (`route_c.py:244`). Fecha o `[A]` da
   máscara com medição em vez de argumento, e custa uma linha.

### P2 — custo, robustez e vocabulário

7. **C4 · Baixar `K_ABSOLUTE_MAX`.** 960 é ~27× acima do que a Eq. 3 prevê para esta
   fonte (3,6 a 36 absoluto, 0,4 a 33 incremental) e **10× acima da faixa em que o
   renderer foi verificado** (`K ≤ 96`, `output/renderer_verification.json`). No pior
   caso o sweep renderiza raios de centenas de pixels numa imagem de 512, o que é
   lento e é um regime não medido. Derivar o teto de um CoC máximo em pixels em vez de
   cravar número.

8. **Rodar os gates baratos ANTES do sweep.** `process_pair` chama `calibrate_k`
   (`:246`) **antes** de `enforce_gates` (`:261`). Toda amostra reprovada por máscara,
   nitidez, f-stop ou profundidade paga ~20 a 40 renderizações de BokehMe. Só
   `calibration_ssim_is_reliable` depende do sweep; os outros nove não. É a maior
   economia disponível e não muda rótulo nenhum.

9. **Slug errado no estouro de orçamento.** `calibration.py:158` levanta
   `k_out_of_configured_range` para "orçamento de 40 avaliações esgotado", enquanto
   `k_search_budget_exhausted` está registrado em `contract.py:142` e **nunca é usado**
   [M]. Dois modos de falha diferentes agregados no mesmo bucket do histograma — que é
   justamente o instrumento que denuncia fallback novo.

10. **`k_min = 0` é aceito pela calibração e rejeitado pelo contrato.**
    `calibration.py:135` valida `0 <= k_min`, mas `signed_coc_px` rejeita `K <= 0`
    (`contract.py:425-426`). **Medido**: `calibrate_k(..., k_min=0.0)` com um
    `render_fn` que chama `signed_coc_px` — exatamente o que
    `BokehMeRenderer.__call__` faz (`bokehme.py:330`) — levanta
    `SampleRejected: k_non_positive` [M] — ou seja, `--k-min 0` rejeitaria
    **todas** as amostras e o histograma culparia a profundidade. Exigir `k_min > 0`
    em `calibration.py:135`.

11. **Nada trava `GATE_TO_REASON` contra `build_gate_report`.** Um gate novo sem
    entrada no mapa faz `enforce_gates` (`route_c.py:216`) levantar `KeyError`, que
    **não** é `SampleRejected` e portanto **derruba o run inteiro** (o `except` de
    `route_c.py:361` só pega `SampleRejected`). Teste que falta:
    `set(GATE_TO_REASON) == {r.name for r in build_gate_report(...)}` e
    `set(GATE_TO_REASON.values()) <= GATE_REJECTION_REASONS`. Hoje os dois conjuntos
    batem — 13 nomes, 11 slugs — mas por sorte, não por trava.

12. **A censura no piso pode alcançar dado legítimo.** `at_lower` marca
    `k_star <= k_min + tolerance = 0.75` (`calibration.py:206-208`). O `K*` incremental
    esperado para um alvo a f/20 é ~0,1 a 0,5 (§2.5), e `K = 0` é valor de
    condicionamento **legítimo** — a Fig. 12 do paper usa `K ∈ {0, 5, 10, 15}` e rotula
    `K=0` como "(Input)" (`paper.txt:1151,1154`). Com `is_valid_for_control =
    not is_k_censored` (`sample.py:156-157`), a cauda inferior inteira sairia da loss.
    Baixar `k_min` (mantendo `> 0`) e reavaliar a banda de censura no piso **com o
    histograma do piloto**.

### P3 — higiene

13. Documentação obsoleta sobre o espelho: `realbokeh.py:733-738`,
    `run_route_c.py:158-168`, `run_route_c.py:202`. Contradizem `ACHADOS.md` e
    `REGISTRO.md:841-856` (96 shards, três splits). Só texto.
14. `verify_renderer.py:125` — mover `highlight_decidido` de `passed` para
    `measurements`, ou fazê-lo assertar algo (§6.3b).
15. `bokehme.py:305` — acrescentar `produces_target_image: False` (§6.3a).
16. `test_sources.py:473-477` — a lista `ATRIBUTOS` é uma **cópia** do `PairSource`,
    não uma leitura dele. A docstring (`:464-471`) justifica a escolha, e o motivo é
    bom (não puxar BokehMe para um teste de adaptador); mas o resultado é um teste que
    verifica a própria cópia. Note que `sensor_width_mm` — lido por `_analytic_k` via
    `getattr` (`route_c.py:315`) — não está na lista nem no protocolo. Ler
    `PairSource.__annotations__` custa um import de `typing`, não de BokehMe.
17. `estimate_disk_budget` (`writer.py:207-225`) modela a máscara como
    `depth_px * 0.04`, mas a máscara é gravada em **resolução cheia** (`writer.py:110`,
    3 MP na RealBokeh) enquanto a profundidade vai a 768. O número final fica na ordem
    certa por acaso; a fórmula está errada. E o metadado não declara `mask_h`/`mask_w`
    — é inferível de `image_h`/`image_w`, mas a regra 3 do contrato pede que seja dito.

---

## 10. Desvios DECLARADOS e defensáveis — não mexer

Listados para não serem redecididos.

| # | desvio | onde está declarado | por que é defensável |
|---|---|---|---|
| D1 | operar em **disparidade** onde o paper escreve profundidade | `CONTRATO.md:26-34`, `contract.py:254-266` | a Eq. 3 só fecha dimensionalmente em `1/z`, e a inferência oficial (`Inference_bokehNet.py:94,118`) resolve o silêncio |
| D2 | **descartar** onde o paper **corrige à mão** | `route_c.py:33-39`, `gates.py:15-21` | a alternativa honesta a 8 h de anotação não é fingir que temos as 8 h; a direção do viés está escrita |
| D3 | **não gravar** o mapa de defocus | `sample.py:13-16`, teste em `test_route_c.py:244-265` | gravá-lo cria segunda fonte de verdade — o defeito D1 histórico |
| D4 | **não gravar** os pixels de origem (default) | `route_c.py:16-18`, `mirror_images.py:171-187` | ledger com sha256 por amostra torna o join verificável byte a byte; `publish_release.py:127-132` **reprova** amostra da rota C sem essa linha |
| D5 | **seção áurea** em vez de varredura exaustiva | `calibration.py:5-11`, `REGISTRO.md:321-330` | o próprio paper usa busca binária no problema análogo (`paper.txt:561-562`); medido, recupera K com erro ≤0,17% [M] |
| D6 | calibrar a Eq. 5 em **lado longo 512** | `CONTRATO.md:130`, `calibration.py:124-131` | custo; K é convertido nas duas direções e travado por teste (`test_renderer.py:216-229`) |
| D7 | profundidade gravada com lado longo **768** | `CONTRATO.md:128`, `encoding.py:50-53` | cota; `quantization_coc_error_px` gravado por amostra |
| D8 | `max_coc = 100,0` **global e congelado**, não parâmetro | `contract.py:43-47`, `contract.py:440-451` | é o mecanismo exato do kfix; `validate_metadata:229-233` reprova qualquer outro valor |
| D9 | `k_effective_factor = 0,9873` **gravado, não aplicado** | `ACHADOS.md`, `CONTRATO.md:110-115` | na rota C o ajuste por SSIM absorve a escala do renderer; aplicá-lo seria corrigir duas vezes |
| D10 | `highlight` **inerte** no nosso caminho | `ACHADOS.md`, `verify_renderer.py:126-131` | medido: `pipeline` não lê `args.highlight`; é decisão registrada, não esquecimento |
| D11 | `_analytic_k` usa `focus_plane_distance` **medido**, não `1/focus_disparity` | `route_c.py:300-309` | é o que torna o validador independente dos dois modelos que produzem o valor validado |
| D12 | AIF da rota C é `train/in/<id>_f22.JPG`, nunca de dentro de `gt/` | `route_c.py:23-27`, confirmado por sha256 [M] | conserto estrutural do D6, com gate (`aif_aperture_is_narrow`) que prova que está de pé |

---

## 11. `[A]` ainda em aberto

Numerados para poderem ser citados. Nenhum foi promovido a `[M]`.

| # | `[A]` | por que é `[A]` | risco | o que resolveria |
|---|---|---|---|---|
| **C-A1** | de qual imagem sai a **máscara** — AIF ou bokeh | o paper não diz; `paper.txt:293-294` sugere AIF pela ordem, e só | **médio** | gravar os dois `focus_disparity` no piloto (`mask_bokeh` já é calculada) e comparar contra `focus_plane_distance` medido |
| **C-A2** | o **limiar de SSIM** da Eq. 5 | o paper afirma usar e não publica (`paper.txt:397-398`, `:1008-1010`) | por definição | painel de 300-500 casos + `calibrate_thresholds --reviewed-csv` |
| **C-A3** | `K_min` / `K_max` da Eq. 5 | não publicados (`paper.txt:393` só diz o símbolo) | **médio** (C4) | histograma de `K*` do piloto + teto derivado de CoC máximo em px |
| **C-A4** | **largura do sensor** da RealBokeh | não publicada pela origem; `--sensor-width-mm` é opcional e marcado `[A]` | baixo (não muda o rótulo, só o validador) | a âncora `3,6 a 36` assume 36 mm `[I]`; medir com o EXIF de um JPEG bruto da origem |
| **C-A5** | a AIF f/22 **não é all-in-focus** e o paper não trata disso | a fonte é fotografia real | **médio** — é a raiz de C1 | medir `K*/k_analytic` contra `F_alvo` no piloto; fecha `[I]` de §2.5 em `[M]` |
| **C-A6** | se "13K newly curated" **é** a RealBokeh e se a subamostragem de 2-4/conjunto é o que fizeram | três aritméticas fecham (`[I forte]`, §1 item 8), mas o paper não nomeia | **médio** (C3) | nenhuma medição resolve; é decisão a declarar |
| **C-A7** | SSIM em **luma** ou por canal | o paper diz só "SSIM" (`paper.txt:393`) | baixo | rodar as duas no piloto e ver se o `argmax` desloca |
| **C-A8** | `alignment_shift_px` medido em `MIRROR_IMAGE_HW` | anotação está no nome do espelho; a resolução não é declarada | baixo | `realbokeh.py:146-149` já registra o `[A]` |
| **C-A9** | os **421 pares** (2,05%) que a origem marca `misaligned`/`shift_*px` entram no dataset | decisão nossa: o campo viaja, ninguém filtra por ele | **médio** | é um gate a mais que o piloto pode medir: `K*` e SSIM de `aligned` contra os outros |
| **C-A10** | os oito limiares de gate restantes | nenhum vem do paper | por definição | piloto + `calibrate_thresholds` |
| **C-A11** | licença dos pixels do LFDOF | *"Sem licença explícita na página"* [M] (`ACHADOS.md`) | **alto** se publicarmos pixels | resolver antes de `--store-source-images` no LFDOF |
| **C-A12** | as 59 imagens ausentes em 11 cenas e a cena `2255` | identificadas [M], causa não | baixo | `ACHADOS.md` já registra; são cauda de nível, nunca buraco no meio |

**Fechados nesta auditoria** (eram implícitos e agora têm evidência):

- ~~"a Eq. 3 vale na rota C?"~~ → **não**, `paper.txt:359-360` contra `:336-352`.
- ~~"a Eq. 5 vale em mais de uma rota?"~~ → **não**, só no (c); a busca de
  `paper.txt:561-562` é avaliação.
- ~~"o BokehMe produz o alvo da rota C?"~~ → **não**; o alvo é `I_real`
  (`paper.txt:393`, `:264`, `:321`), e o `[43]` só entra dentro do sweep
  (`paper.txt:1007`).
- ~~"o paper exige o LFDOF?"~~ → **sim**, em três lugares (`paper.txt:326`, `:359`,
  `:528-529`).
- ~~"o paper publica o limiar de SSIM?"~~ → **não**, e afirma usá-lo duas vezes.
- ~~"a rota C usa a DeblurNet?"~~ → **não**; *"These datasets provide pairs"*
  (`paper.txt:359`).
- ~~"quantos alvos por cena?"~~ → o paper diz **2 a 4 por conjunto**
  (`paper.txt:1001-1003`). Era desconhecido; agora é C3.

---

## 12. O que este relatório NÃO conseguiu medir

Registrado para ninguém confundir com verificação.

- **Nada rodou com GPU, BokehMe real, Depth Pro ou BiRefNet.** Todas as medições de
  busca em §2.3 e §9.10 usaram renderers sintéticos ou SSIM substituído por pico
  analítico. O comportamento do sweep **sobre foto real** — inclusive se `SSIM(K)` é
  de fato unimodal, que é a hipótese da seção áurea e da busca binária do paper — é
  **não medido**.
- **A distribuição de `K*` da rota C é não medida.** Toda a análise de §2.5 é álgebra
  a partir da Eq. 3 (`[I]`), não medição. O piloto é quem fecha.
- **O impacto real de C5** (quantas amostras `focus_depth_plausible` reprova na rota C)
  é **não medido**.
- **Custo por amostra** do sweep é **não medido**: nem o número real de avaliações
  sobre foto natural, nem o tempo por render de BokehMe a 512 px. É o item dominante do
  orçamento e não tem estimativa em lugar nenhum do repositório.
- **A composição de borrão** (linear no raio × quadrática) do scatter clássico do
  BokehMe é **não medida**. As duas colunas da tabela de §2.5 são limites teóricos.
- **Suíte**: 239 testes, `OK (skipped=6)` [M], com
  `/Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python`, `PYTHONPATH=src:scripts:tests`.
  Os 6 skips são o bloco que toca o HF de verdade e o que exige `pyarrow`.
