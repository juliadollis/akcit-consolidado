# Desvios do paper de referência — rascunho da seção de método

**Paper de referência**: GenRefocus, arXiv:2512.16923v3. Texto extraído, grepável e
**imutável** em `reference/paper.txt` — **1.203 linhas**, incluindo o supplement, que
começa na linha 966 (cabeçalho de página) / 968 (`A Overview`).

Este arquivo é novo. Ele não substitui `reference/CONTRATO.md` (que define o sinal de
controle) nem as três auditorias de rota (que confrontam código com paper item por item).
O que ele faz é **reunir num só lugar todo ponto em que divergimos**, com a linha do
paper conferida, o motivo, e a evidência que sustenta a escolha — e separar três coisas
que não se misturam:

- **§2 — desvios declarados e defensáveis**: decididos, justificados, e que ficam.
- **§3 — divergências ainda acidentais**: ninguém decidiu; precisam de decisão.
- **§4 — silêncios do paper**: onde tivemos de assumir, com o que mediríamos para
  resolver.

Escrito com o peso de rascunho de método: cada parágrafo da §2 é candidato a entrar no
paper como está.

---

## 1. Antes de tudo: duas armadilhas de leitura do `paper.txt`

### 1.1 O `paper.txt` tem **duas numerações de referência**

Registrada no projeto, e **confirmada nesta auditoria por prova cruzada**:

| citação | nas legendas das Figs. 4(a)/(b) | no corpo e na bibliografia |
|---|---|---|
| Restormer | `[80]` — `paper.txt:429`, `:439` | `[82]` — `paper.txt:382` |
| DiffCamera | `[67]` — `paper.txt:434`, `:444` | `[69]` — `paper.txt:225`, `:385`, bibliografia em `:914-915` |
| Generative Photography | — | `[80]` — `paper.txt:240`, `:528`, `:996`, bibliografia em `:941-943` |
| CLIP-IQA | — | `[67]` — `paper.txt:553` |

**A numeração antiga aparece só nas legendas das Figs. 4(a) e 4(b)** — linhas 429, 434,
439, 444. As legendas da Fig. 3 (287-297) e das Tabs. 3/4/6 usam a numeração **atual**.
Ou seja: a regra "legenda não vale" é forte demais; a regra correta é **estas quatro
linhas não valem**.

Isto já pegou o projeto duas vezes: `[80]` lido como DiffCamera quando é Generative
Photography.

### 1.2 A legenda da Fig. 3(c) contém um **erro do próprio paper**

`paper.txt:296-297`: *"(c) LFDOF [52] and RealBokeh. For real pairs, we obtain Dfocus as
in (b), and **follow Eq. (2) to estimate the bokeh level K**."*

A Eq. 2 (`paper.txt:312`) é `D_def = K · |D − D_focus|`: ela **consome** K, não o estima.
O corpo (`paper.txt:369-370`, `:391-398`) e o supplement (`paper.txt:1007`) são
inequívocos — o K da rota C vem do sweep da Eq. 5. Confirmado nesta auditoria, e o
diagrama da Fig. 3 (`paper.txt:279`) escreve a Eq. 2 na forma `D_def = |D − D_focus| ∗ K`,
o que reforça a leitura.

Seguimos o corpo. Não é desvio nosso; é imprecisão dele, e registrar isso é parte de ser
auditável.

### 1.3 Quatro identificações de dataset vêm de **uma frase**, não da bibliografia

Verificado literalmente: as entradas bibliográficas **não contêm** os nomes curtos.

| ref. | entrada bibliográfica | onde o nome curto aparece |
|---|---|---|
| `[27]` | Ignatov, Patel, Timofte, *"Rendering natural camera bokeh effect with deep learning"*, CVPRW 2020 — `paper.txt:814-815` | **"EBB"** só em `paper.txt:996` |
| `[86]` | Zheng et al., *"Bilateral reference for high-resolution dichotomous image segmentation"*, CAAI AIR 2024 — `paper.txt:955-957` | **"BiRefNet"** só em `paper.txt:348` e `:362` |
| `[52]` | Ruan et al., *"AIFNet: all-in-focus image restoration network using a light field-based dataset"*, IEEE TCI 2021 — `paper.txt:873-875` | **"LFDOF"** só no corpo (`:326`, `:359`, `:528-529`) |
| `[57]` | Seizinger et al., *"Bokehlicious: photorealistic bokeh rendering with controllable apertures"*, ICCV 2025 — `paper.txt:885-887` | **"RealBokeh"** só no corpo |

Consequência prática: *"o paper usa a RealBokeh"* é `[P]` — o corpo diz `RealBokeh [57]`.
Mas *"a RealBokeh é o `timseizinger/RealBokeh_3MP`"* é `[I]`, e *"o ITW `[19]` é o
`atfortes/BokehDiffusion`"* é `[I]` mais fraco ainda — sustentado por autor (`Fortes, A.`,
`paper.txt:791-793`), handle do HF, e volume (13.800 linhas medidas contra os "13K" de
`paper.txt:1000`). **Nenhum URL de dataset é publicado pelo paper.**

---

## 2. Desvios declarados e defensáveis

Doze. Cada um com o que o paper faz, o que fazemos, por quê, e a evidência.

### D1 — Operamos em **disparidade** onde o paper escreve profundidade

**O paper.** `paper.txt:314-315`: *"where D is the monocular depth map estimated from
I_aif using an off-the-shelf depth estimator [7]"* — `[7]` é Depth Pro
(`paper.txt:762-764`). A Eq. 2 (`:312`) e a Eq. 4 (`:352`) operam sobre esse `D`. A
palavra *disparity* **nunca** é aplicada a `D` no artigo: as duas únicas ocorrências são
`paper.txt:199` (*"disparity-aware techniques [32,68,77]"*, related work) e `:933` (título
de uma referência). **Verificado literalmente nesta auditoria.**

**Nós.** `disp = 1/z`, e todo o controle vive em `1/m`.

**Por quê, e a autoridade é o código, não a leitura.** Dois argumentos independentes:

1. **Dimensional.** A Eq. 3 (`paper.txt:340`) produz `K` com unidade `px·mm` (mm² × px/mm),
   e `CoC` tem que sair em pixel. Só fecha multiplicando por `|Δ(1/z)|`. Com `|ΔD|` em mm
   a unidade seria `px·mm²`. A análise completa está em `CONTRATO.md:59-68` e é travada
   por teste.
2. **Código oficial.** `Inference_bokehNet.py:94` faz `disp = 1.0/safe_depth` e `:118`
   tira `np.median(valid_disp)`.

**Evidência que sustenta.** A âncora numérica fecha por três caminhos: `k_official(16553,9)
= 16,5539` (reprodutível com o nosso código), mediana da EXIF 20,1, default oficial 15,0,
e a Fig. 12 do paper varre `K ∈ {0, 5, 10, 15}` (`paper.txt:1151`, `:1154`, conferido).
Um `k_eq3` de 16.553,9 multiplicando `Δ(1/mm)` daria CoC na casa dos milhares de pixels —
mapa saturado, K algebricamente apagado. Foi o defeito real, medido: `max(defocus_map)`
= 65535 exato em **todas** as amostras publicadas, com `k` variando de 33 a 195.

**Fraqueza a declarar no paper.** `Inference_bokehNet.py` **não está neste repositório**.
Um avaliador que receba só `bokehnet-regen/` não pode conferir a autoridade que decide o
desvio. É a lacuna `[X]` mais consequente do projeto.

### D2 — `max_coc = 100,0`, global e congelado, onde o paper não normaliza

**O paper.** A Eq. 2 é crua: `D_def = K · |D − D_focus|`, sem denominador
(`paper.txt:312`, conferido).

**Nós.** `defocus = clip(|K·(disp − disp_focus)| / 100,0, 0, 1)`, com `MAX_COC` **global,
congelado por release, e nunca parâmetro de função**.

**Por quê.** O valor vem de `Inference_bokehNet.py:20`. Adotá-lo torna o dataset
comparável com os pesos oficiais: dá para inicializar deles, avaliar no mesmo harness e
comparar K absoluto. Com o `max_coc = 10,5107` do experimento `kfix`, nada disso valia.

**Evidência, e é a mais forte do repositório.** O normalizador por rota é o mecanismo
medido do desastre: LF-Bokeh subiu para +0,8288 enquanto RealBokeh caiu para +0,4832 e
RealDOF para **−0,4599**. E `max_coc_calibrado = 10,510746` é exatamente o percentil 82,66
de `coc_p99_px` **da própria rota B** — um normalizador derivado de uma rota só, aplicado
ao lote inteiro.

**Como o desvio é travado.** `MAX_COC` não é campo de `ControlLabel` nem parâmetro de
`defocus_map`; `validate_metadata` rejeita qualquer outro valor com slug
`max_coc_invalid`; e um revisor **mediu** que, quando era parâmetro, uma amostra com
`max_coc = 10.510746` atravessava o writer até o disco sem uma linha de erro.

### D3 — `D_focus` é `median(1/z[M])`, não `1/median(z[M])`

**O paper.** Eq. 4, `paper.txt:352`: `D_focus = median(D[M])` — conferido.

**Nós.** A mediana é tirada **na disparidade**, e o mapa nunca reconstrói a partir de
`focus_depth_m`.

**Por quê.** A mediana só é invariante sob transformação monótona em contagem ímpar;
`np.median` faz a média dos dois centrais quando é par, e a média de dois recíprocos não é
o recíproco da média. Contraexemplo no teste: `z = [1, 2, 4, 8]` dá `focus_disp = 0,375`
contra `1/median(z) = 0,3333`.

**Honestidade sobre a magnitude.** A diferença é minúscula em máscara grande. O desvio
existe para **não ter a folga**, não porque a folga fosse grande. `focus_depth_m` fica no
metadado, documentado como leitura humana.

### D4 — O refinamento **manual** da máscara virou refinamento **automático**

Este é o desvio de maior consequência científica, e o único que tem uma medição própria
dedicada (`reference/MEDICAO_PLANO_FOCO.md`).

**O paper.** `paper.txt:364-368` (conferido, e é literal):

> *"**Rather than simply verifying and discarding unreliable cases**, we introduce a
> manual refinement step. Specifically, we re-select a small yet reliable in-focus region
> to correct M, thereby accurately extracting D_focus. This strategy **preserves
> challenging samples rather than excluding them**."*

E o supplement quantifica: *"This annotation step required **4 to 8 seconds per image**,
amounting to approximately **8 hours of manual effort** in total"* (`paper.txt:1005-1006`,
conferido).

**Nós.** `src/qc/focus_region.py` mede **retenção de detalhe** e propõe a região:

```
retencao(x) = media_local(|laplaciano(bokeh)|) / media_local(|laplaciano(aif)|)
```

Perto de 1 no plano de foco, perto de 0 no que borrou. Três ramos: se a máscara do
BiRefNet concorda com a retenção (cobre ≥ 30% dela), mantém a máscara — o caminho do
paper; se discorda, interseca; se não há máscara utilizável, usa só a retenção.

**Por quê o denominador importa.** Ele normaliza pela textura da própria cena, que é
exatamente onde nitidez absoluta falha: folhagem desfocada tem mais energia de alta
frequência que parede lisa em foco. `tests/test_focus_region.py` monta essa armadilha e
prova que a medida absoluta escolheria o lado errado e a razão não.

**Por quê é necessário, com número.** O piloto (job 32224, 162 amostras aceitas de 204,
29 cenas) mediu:

- a `focus_disparity` da máscara crua do BiRefNet fica dentro de ±25% da distância de foco
  **medida na captura** em apenas **35,2%** dos casos (57/162);
- razão obtida ÷ gabarito: mediana **0,579** — a máscara escolhe um plano ~1,7× mais longe;
- a origem publica essa distância com incerteza mediana de **±0,010 m**, o que elimina a
  hipótese de gabarito ruim;
- **20,6%** das amostras (42/204) eram descartadas com `focus_mask_empty`, em 10 cenas
  inteiras, com **zero** cenas misturando aceite e rejeição.

E a causa foi medida separadamente (job 32231, 12 cenas): o BiRefNet devolve
probabilidade **exatamente 0,000**, e baixar o limiar de 0,5 para 0,05 não recupera nada.
Ele **declina**, não falha — é segmentador de objeto saliente, e a RealBokeh é feita de
cenas (um tronco num parque, um muro de pedra).

**A ironia que fecha o argumento.** O paper **antecipa** este problema, nestes mesmos dois
datasets, e **rejeita explicitamente** a estratégia de descartar. Nós descartávamos. Os
nossos 35,2% são a quantificação do *"sometimes unreliable"* dele.

**O que ainda é desvio, e é estreito.** O paper corrige à mão; nós corrigimos por
retenção. Cada amostra carrega `focus_source` (`birefnet` / `birefnet_refined` /
`retention_only`) e `focus_was_refined`, no metadado **e** no manifesto, para dar para
treinar com e sem as refinadas e **medir** a diferença. Sem essa marcação, "consertamos"
seria afirmação sem teste.

**Por que não usar a distância medida como rótulo.** Ela resolveria a RealBokeh e deixaria
o LFDOF — que não publica distância nenhuma — com rótulo de outra qualidade dentro da
mesma rota. Dois níveis de qualidade de rótulo no mesmo lote é o erro do `kfix` com outra
roupa. Um método só, validado contra o gabarito onde ele existe.

**Estado**: o caminho está implementado e testado; a validação contra o gabarito
(`scripts/validate_focus_refinement.py`, régua = 35,2%) **ainda não rodou**. Enquanto não
rodar, o desvio é declarado mas **não é justificado por medição de melhoria**.

### D5 — Onde o paper corrige à mão, dez gates descartam

Distinto de D4 e vale separar: D4 substitui a **correção**; D5 é o que sobra depois dela.

**Contabilidade honesta dos onze gates atuais**, na leitura da auditoria da rota C:

| o que o paper pega | como | nós |
|---|---|---|
| K ruim | limiar de SSIM (`paper.txt:397-398`) | `calibration_ssim_is_reliable` — **mesma coisa**, e no paper também é automático |
| máscara ruim | corrige à mão | `mask_area_ratio_min/max`, `mask_border_coverage`, `focus_mask_sharpness_ratio`, `mask_iou_aif_bokeh` — **descartam** |
| — | — | `pair_shape_matches`, `aif_sharpness`, `bokeh_over_aif_sharpness`, `aif_f_number`, `depth_useful_levels`, `focus_depth_m_min/max`, `focus_region_retention` — **acréscimos nossos, sem análogo no paper** |

Ou seja: **um** gate corresponde a um passo do paper, quatro substituem revisão humana por
descarte, e seis são novos. A direção do viés está escrita no código: **contra cena
complexa**, exatamente a direção que o paper diz ter evitado de propósito.

**Um gate mede algo que a revisão humana pegaria e nenhuma IoU pega.**
`focus_mask_is_sharpest` mede nitidez dentro/fora da máscara **na imagem bokeh**. Quando o
fotógrafo focou o fundo e o BiRefNet marcou o objeto saliente em primeiro plano, as duas
máscaras automáticas concordam, a IoU vale 1,0, e o `D_focus` inteiro está errado. Um
humano "re-selecionando uma região confiável em foco" pegaria; um gate de IoU não.

**Onde a cobertura é pior que a humana, e vale dizer no paper**: nenhum gate **corrige**;
nenhum gate olha a cena (`mask_border_coverage` não distingue objeto grande legítimo de
máscara vazando); e `bokeh_over_aif_sharpness` não pega um par f/14 contra f/2,0 — isso
está na docstring e travado por teste.

### D6 — A rota C consome uma "AIF" que é uma foto **f/22**, não all-in-focus

**O paper.** `paper.txt:359`: *"These datasets provide pairs"* — a AIF vem da origem, e o
paper não discute a qualidade dela. Na (b) ele é explícito que a AIF é produzida pela
DeblurNet (`paper.txt:293`, `:337-338`); na (c), não.

**Nós.** A AIF da RealBokeh é `train/in/<cena>_f22.JPG`, confirmado **byte a byte por
sha256** (cena 1038: `59d8e910ca69…`, 2000×1500, byte-idêntico entre `level_3` e `level_4`,
e nenhuma das 5 imagens de `train/gt/1038/` bate).

**Por quê isso é o desvio certo.** A alternativa era o que o pipeline antigo fazia: pegar
o maior f-stop **dentro de `gt/`**, cuja mediana por cena é **f/14** e que em **12,7%** das
cenas é f/5.6 ou mais aberto — nessas, profundidade, máscara e sweep saíam todos de uma
foto com bokeh forte.

**A consequência que herdamos, e é `[A]`.** f/22 não é all-in-focus. Ela carrega o próprio
desfoque, e isso vale para tudo que a AIF alimenta: Depth Pro, BiRefNet e o sweep. É a raiz
do §3.1 abaixo.

### D7 — O K analítico da rota C é **validador**, nunca rótulo

**O paper.** A Eq. 3 vive dentro do parágrafo *"(b) ITW dataset"* (`paper.txt:336-352`), e
o (c) abre dizendo que aqueles datasets *"omit EXIF metadata or provide insufficient
fields to estimate K"* (`paper.txt:359-360`, conferido). Escopo fechado por dois caminhos.

**Nós.** O rótulo da rota C é o sweep da Eq. 5. O `k_analytic` calculado da Eq. 3 sobre o
`metadata/` da origem é gravado ao lado, como instrumento de auditoria.

**Por quê é legítimo.** O paper proíbe usar a Eq. 3 como **rótulo** ali, e a proibição é
factual (os campos não existem), não normativa. E aqui o validador é **mais forte que na
rota B**: o `focus_plane_distance` da RealBokeh é profundidade métrica **medida na cena**,
com barra de erro — não estimada por modelo. Alimentá-lo com `1/focus_disparity` o faria
depender do Depth Pro e do BiRefNet, os mesmos dois modelos que produzem o valor sendo
validado.

**Está desligado por default, e isso é correto.** `_analytic_k` devolve `None` sem
`--sensor-width-mm`, porque a largura do sensor da RealBokeh não é publicada. O release
conta e reporta quantas amostras ficaram sem validador, sem reprovar. Não há 36,0 mm
cravado dentro do laço.

### D8 — Seção áurea em vez de varredura exaustiva

**O paper.** `paper.txt:393` dá a Eq. 5 e o intervalo aberto `(K_min, K_max)`. **Não
publica** algoritmo de busca, tolerância, número de avaliações, nem `K_min`/`K_max`.

**Nós.** Grid grosso de 7 pontos para bracketear (expandindo o teto ×2 enquanto o máximo
cair na borda), depois seção áurea até `hi − lo ≤ 0,25`, teto de 40 avaliações,
in-process.

**Por quê.** Custo: o antigo dava ~40 `subprocess` por amostra, cada um subindo Python,
CUDA e dois checkpoints. E o paper **usa busca binária no problema análogo**:
`paper.txt:561-562` (conferido) — *"we conduct a per-image binary search over K and select
the value that maximizes SSIM with the target"*, na **avaliação**. Busca binária sobre um
argmax pressupõe unimodalidade, a mesma hipótese da seção áurea.

**Correção de vocabulário registrada:** documentação anterior dizia "busca ternária".
Ternária gasta 2 avaliações por iteração e encolhe para 2/3; a áurea gasta 1 e encolhe
para 0,618. `CLAUDE.md` ainda diz "ternária" — ver §3.6.

**Medido**: recupera K com erro ≤ 0,17% em toda a faixa (3,6 a 700), com 19 a 40
avaliações. **Ressalva honesta**: medido com o SSIM substituído por um pico sintético
exato. Se `SSIM(K)` é de fato unimodal **sobre foto real** segue **não medido** — e é a
hipótese que a seção áurea e a busca binária do paper compartilham.

### D9 — Três resoluções escolhidas por nós, porque o paper não publica a de treino

**O paper.** Ele **não publica a resolução de treino**. O tiling da §3.5 é explicitamente
de **inferência**: *"We implement a tiling strategy inspired by [5] **during inference**"*
(`paper.txt:481-482`, conferido). A §4.1 dá batch, acumulação e steps, e cala sobre
resolução.

**Registro de um erro nosso, e vale manter no paper como exemplo de método.** Uma versão
anterior deste repositório escreveu *"o paper treina em resolução nativa com tiling
(§3.5)"* — em `contract.py` e no `REGISTRO.md`. A frase é **falsa**, e foi exatamente a
construção que o nosso próprio agente `paper-fidelity` proíbe: "o paper implica" escrito
com número de seção. Corrigido.

**Nós, e cada uma com o custo declarado:**

| decisão | valor | custo declarado |
|---|---|---|
| profundidade gravada, lado longo | **768** | `quantization_coc_error_px` por amostra — **0,000378 px** com K=50, refeito nesta auditoria |
| crop de treino, lado menor | **512**, com K reescalado | fator por amostra; medidos 0,892 e 0,821 |
| calibração da Eq. 5, lado longo | **512** | K convertido nas duas direções, travado por teste |

**Armadilha de ergonomia declarada**: `k_at_resolution` recebe **lado menor** e
`encode_depth` recebe **lado longo**. Convenções opostas na mesma cadeia.

### D10 — O mapa de defocus **não** é gravado

**O paper.** `paper.txt:398-399` (conferido): *"Finally, the defocus map D_def is
constructed accordingly following Eq. 2."* E `paper.txt:321`: a tupla de supervisão é
`(I_aif, I_out, D_def)`.

**Nós.** Gravamos disparidade uint16 + `k_value` + `focus_disparity`, e o mapa é derivado
no dataloader pela **mesma função** da geração.

**Por quê.** Gravá-lo cria segunda fonte de verdade. Foi o defeito D1 histórico:
`max(defocus_map) = 65535` exato em todas as amostras, com `k` de 33 a 195 — o mapa
gravado apagou o K algebricamente, e nada denunciou. `tests/test_route_c.py` prova que o
mapa é reconstruível só dos escalares e que **nenhum arquivo `*defocus*` é escrito**.

### D11 — Amostrar K da distribuição empírica de B/C na rota A

**O paper.** `paper.txt:329-330` (conferido): *"we **randomly sample** a focus plane
D_focus and a target bokeh level K"*. E **cala sobre a distribuição**.

**Nós.** `k_source = "sampled_from_bc"` — amostrar da distribuição empírica das rotas B e
C, para manter o sintético na mesma escala física do real.

**Estado**: decisão declarada, rota A **não implementada**. E há uma dependência dura,
medida: hoje a rota B publicada tem `k = 50,0` em 11.635/11.635 e a rota C tem 47,0% no
teto exato 300. **Amostrar dessa "distribuição" hoje é amostrar de duas constantes.**

### D12 — `k_effective_factor = 0,9873` é **gravado, não aplicado**

**O paper.** Não trata de viés de escala do renderer.

**Nós.** Medimos em GPU que o raio renderizado é **1,27% menor** que o K nominal prediz
(3,81% no clássico puro), gravamos o fator em cada amostra, e **não corrigimos**.

**Por quê.** Na rota C o K é ajustado por SSIM contra o alvo real, e o ajuste absorve a
escala do renderer por construção — aplicar o fator seria corrigir duas vezes. Na rota B,
cujo K é analítico, **não há renderer**, então o fator não descreve nada daquela rota. A
recomendação registrada é não aplicar em nenhuma das duas, e declarar.

**Ressalva de escopo**: medido com fonte pontual sintética, não com cena real.

---

## 3. Divergências que ainda são acidentais — precisam de decisão

Seis, no estado de **2026-09-10 19h**. Nenhuma foi decidida; todas são o que a enumeração
ou a omissão fizeram.

> **Nota de estado, e é importante.** Este repositório estava sendo editado por outros
> agentes enquanto este arquivo era escrito. Em 24 minutos a suíte foi de 519 para
> **650 testes** e `src/` de 7.993 para **9.297 linhas**. A §3.1 abaixo **foi resolvida
> durante a redação** — deixei o diagnóstico inteiro de pé, com o conserto anotado no fim,
> porque o raciocínio é o que vai para o paper e porque um desvio consertado sem registro
> do motivo volta na revisão seguinte. Reverifique a §3 contra o código antes de citá-la.

### 3.1 CRÍTICA — Multiplicidade por cena: o paper diz **2 a 4**, nós pegávamos até **21**
### — **RESOLVIDA em 2026-09-10, entre 19:11 e 19:32**

**O paper.** `paper.txt:1001-1003` (conferido, literal): *"This collection comprises 13K
previously filtered and verified images from the ITW dataset [19], alongside **13K images
newly curated** for this work. The newly collected data consists of focus-consistent
series captured with varying apertures, containing **2 to 4 images per set**."*

**Nós.** `enumerate_pairs` pega **todos** os níveis de todas as cenas: 20.495 pares de
3.959 cenas no `train`, 22.990 nos três splits.

**A conta que mostra o problema**, aritmética direta sobre o histograma medido de
níveis/cena (refeita nesta auditoria, fecha exato):

```
cenas com 21 níveis :   244 de 3.959  =  6,2% das cenas
pares que elas geram : 5.124 de 20.495 = 25,0% das amostras
```

Um quarto do dataset sairia de um vigésimo das cenas. É a armadilha que o `CLAUDE.md` já
nomeia — *"20.554 amostras de 3.960 cenas não são 20.554 unidades de diversidade"* — agora
com número.

**Três aritméticas independentes indicam que a RealBokeh **é** os "13K newly curated"**
(`[I]` forte, não `[M]`):

1. 8 h ÷ 4-8 s por imagem = **3.600 a 7.200** máscaras, e a RealBokeh tem **4.400** cenas
   (a máscara é por cena, uma AIF por cena): 4.400 × 6,5 s = **7,9 h**.
2. 4.400 cenas × 3 alvos = **13.200 ≈ "13K"**.
3. Recontando o histograma medido **com teto de 4 alvos por cena**: 1.410 + 1.863 +
   4×2.631 + 2 = **13.799** no `train`. Com teto 3: **11.168**. A faixa "2 a 4" produz
   11K–14K, e o paper diz 13K. **As três contas foram refeitas nesta auditoria e fecham.**

**Por que é acidental.** Ninguém decidiu pegar 21; foi o que a enumeração fez. E o item
não estava em nenhuma lista do projeto antes da auditoria da rota C.

**Por que precisa de decisão agora.** Não tem conserto retroativo barato: gerar 20.495
amostras e depois descobrir que 25% vêm de 6% das cenas significa refazer o split e a
contagem. Decidir antes custa uma função — um teto de níveis por cena, sorteado com seed,
onde `sample_pairs_for_pilot` já vive.

**Duas saídas defensáveis, e a escolha é de quem escreve o paper:** (a) aplicar o teto e
reportar ~13K, alinhado ao paper; (b) rodar 20.495 e **declarar** que rodamos 1,6× o
volume do paper, com a concentração medida de 25%/6,2% escrita no texto. O que não é
defensável é deixar acontecer por omissão. Registrado como item aberto em
`PLANO_EXECUCAO.md:115-117`.

#### O conserto, aplicado durante a redação deste arquivo

`src/sources/level_selection.py` (136 linhas) e `tests/test_level_selection.py` (23 testes)
passaram a existir entre 19:11 e 19:32 de 2026-09-10, e `scripts/run_route_c.py` ganhou
`--max-levels-per-scene`, **default 4**. Verificado: o módulo corta **níveis, não cenas** —
toda cena continua representada, uma cena de 21 aberturas contribui com 4 e uma de 2
contribui com as 2 — e a docstring reproduz as duas aritméticas da §3.1 acima, incluindo o
13.799 e a nota de que sem teto dá 20.495, "58% acima do publicado".

A escolha foi a **(a)**. Isso move o item da §3 para a §2 como desvio declarado — ou, mais
precisamente, **remove** o desvio: passamos a reproduzir o volume que o paper publica. O que
resta declarar no método é que o teto de 4 é uma leitura `[I]` de *"2 to 4 images per set"*,
que o critério de seleção dos 4 níveis é nosso (sorteio com seed, não o critério dos
autores, que não é publicado), e que a escolha muda o tamanho do dataset de ~20K para ~14K.

### 3.2 CRÍTICA — Metade da rota C não roda: o LFDOF é **exigido em três lugares**

**O paper.** Exige o LFDOF em quatro lugares, todos conferidos: `paper.txt:326` (a
enumeração das três rotas), `:359` (o título do parágrafo (c)), `:528-529` (*"approximately
26K real examples sourced from ITW dataset [19], RealBokeh [57], and LFDOF [52]"*), e
`:296` (legenda da Fig. 3(c)). **A rota C do paper são dois datasets.**

**Nós.** O adaptador do LFDOF **existe** — `src/sources/lfdof.py` (879 linhas, 112 testes)
e `src/sources/lfdof_images.py` (403 linhas), com 28 medições próprias (§8.9 da
`TABELA_DE_EVIDENCIAS.md`): 11.972 pares, 840 cenas, 83 shards, alinhamento medido,
`level` 1-based confirmado.

**A divergência acidental é de fiação.** `scripts/run_route_c.py::_carrega_fonte` só trata
`"realbokeh"`, e `--source lfdof` levanta `SystemExit` com a mensagem *"fonte ainda não tem
adaptador. A rota C recebe os pares por protocolo — escrever o adaptador não exige tocar em
routes/route_c.py"* — mensagem que hoje é **falsa**. O adaptador existe; falta ligá-lo.

**E o bloqueador que existia foi removido.** O defeito C2 (`aif_f_number = None`
reprovando 100% do LFDOF com o slug `gate_aif_aperture_wide`, que afirmava algo falso
sobre o dado) está **fechado**: `GateResult.applicable` distingue "não existe para esta
fonte" de "existe e deu ruim", e `aif_aperture_is_narrow(None)` devolve
`applicable=False`.

**Consequência para o paper.** Enquanto isso não for ligado, o que temos é **meia rota C**,
e a comparação de volume com os "~26K real examples" não é aplicável.

**Ganho científico que se perde enquanto isso.** No LFDOF a AIF é **genuinamente**
all-in-focus (renderizada da light field), então o `K*` da Eq. 5 é o K **absoluto** — some
o offset sistemático do §3.1 da RealBokeh. É a fonte que **calibra a interpretação** da
RealBokeh, e por isso vale rodar as duas.

### 3.3 ALTA — `K_ABSOLUTE_MAX = 960` autoriza 10× além da faixa verificada

**O paper.** Não publica `K_min` nem `K_max` (`paper.txt:393` só dá os símbolos).

**Nós.** Busca inicial em `[0,5; 120]`, expandindo o teto ×2 até **960**.

**Por que é acidental.** O laudo do renderer mediu linearidade em **K ∈ {8, 16, 32, 64,
96}**, raios de 3,2 a 38,4 px. `960` autoriza renderizar **10× além do medido**, e ~27×
acima do que a Eq. 3 prevê para esta fonte (3,6 a 36 absoluto, 0,4 a 33 incremental). No
pior caso o sweep renderiza raios de centenas de pixels numa imagem de 512 — lento, e
regime não medido.

**Conserto proposto**: derivar o teto de um CoC máximo em pixels, em vez de cravar número.

### 3.4 ALTA — Dois limiares `[A]` bloqueiam desde a primeira amostra, e o piloto não os calibra

**A regra do módulo**, escrita em `src/qc/gates.py`: *"limiar não medido não bloqueia"*.
Onze limiares de `RouteCConfig` são `None` por default. **Duas exceções**, e uma importa:

- `pair_shape_matches` tem limiar `1.0` cravado — estrutural, declarado, e na prática
  código morto (a rota já rejeitou por `resolution_invalid` antes). Sem problema.
- **`focus_depth_plausible` tem defaults não-`None`**, vindos de `FOCUS_DEPTH_MIN_M = 0,05`
  e `FOCUS_DEPTH_MAX_M = 1000,0` — os dois marcados `[A]` no fonte, com a nota "calibrar no
  piloto". E `focus_disparity_from_mask` já rejeita com os mesmos limites **antes** dos
  gates.

**Por que o piloto não consegue calibrá-los** (reverificado no código em 2026-09-10 19h):
a amostra é rejeitada antes de entrar em `quality`;
`calibrate_thresholds._distributions` lê só as **aceitas**; `_GATES` tem 10 entradas e
não inclui `focus_depth_m_min/max`; **não existe** flag `--focus-depth-min-m`/`--max-m` em
`run_route_c.py`; e `calibrate_thresholds` **ignora o campo `detail`** do
`rejections.jsonl`, que é onde o valor rejeitado está gravado.

Resultado: dois `[A]` filtram o dataset, o documento diz que nenhum `[A]` filtra, e o
instrumento que existe para transformar histograma em número não os alcança.

**Impacto: não medido** para a rota C. O `focus_plane_distance` publicado vai de 0,355 a
368,63 m — **dentro** da faixa. Mas o valor que o gate julga vem do Depth Pro + máscara,
não do metadata.

### 3.5 MÉDIA — Um slug registrado e nunca usado, e um bucket que agrega dois modos de falha

`calibration.py` levanta **`k_out_of_configured_range`** quando o orçamento de 40
avaliações esgota. O slug **`k_search_budget_exhausted`** está registrado em
`control/contract.py` e **não é usado em nenhum lugar** de `src/`, `scripts/` ou `tests/`
(verificado por grep).

Dois modos de falha distintos — "a faixa configurada estava errada" e "a busca não
convergiu no orçamento" — caem no mesmo bucket do histograma, que é justamente o
instrumento que denuncia fallback novo.

Na mesma família: `calibration.py` valida `0 <= k_min`, enquanto `signed_coc_px` rejeita
`K <= 0` com `k_non_positive`. Ou seja, `--k-min 0` é aceito pela calibração e faz **todas**
as amostras serem rejeitadas, com o histograma culpando a profundidade.

### 3.6 BAIXA — Documentação que contradiz medição do próprio repositório

Dez casos verificados, nenhum afetando o rótulo, todos afetando quem lê. Os mais visíveis
para um avaliador externo:

| onde | diz | é |
|---|---|---|
| `README.md` | "48 testes passando", "1.386 linhas em `src/`", "14 dos 33 defeitos fechados" | 521 testes, 7.993 linhas, placar da etapa 4 é 16/3/14 |
| `CLAUDE.md`, Layout | "busca ternária" | seção áurea |
| `src/sources/mirror_images.py` | "O espelho são 85 shards" | 96 |
| `src/sources/realbokeh.py` | "o espelho publica só o split `train`" | três splits, 22.990 linhas |
| `scripts/run_route_c.py` | idem, e "(85 shards parquet)" no `--help` | idem |
| `.claude/agents/defect-regression.md` | "o catálogo dos **32** defeitos" | **33**, corrigido em `REGISTRO.md:152` |
| `src/control/contract.py` | "32 de 100 amostras conferidas da rota B são retrato" | 32 de **900** |
| 4 lugares | "a janela de 33 px cobre 6,4% do lado longo a **512x683**" | a grade é **384×512**, e 33/512 = 6,4% |
| `PLANO_EXECUCAO.md:104` | "`deblur_variant` … conferido na publicação" | `publish_release.py` não tem essa checagem |
| `slurm/route_c_pilot.slurm` | "TODOS os **dez** gates" | onze |

A mais irônica é a do espelho: `src/sources/lfdof.py` **registra a correção** ("a medição
anterior dela — 'publica só train' — estava incompleta justamente por isso"), enquanto os
dois módulos da RealBokeh, ao lado, ainda carregam a versão revogada.

---

## 4. O que o paper não publica — e o que assumimos

Vinte e quatro itens. Cada um com por que é silêncio, o risco, e **o que mediríamos** para
resolver. Nenhum foi promovido a `[M]`.

### 4.1 Silêncios sobre o renderer

| # | silêncio | nosso valor | risco | o que resolveria |
|---|---|---|---|---|
| R1 | `gamma` | 4,0 | médio | grid de `gamma` contra o alvo real, medindo o SSIM do `K*` |
| R2 | `defocus_scale` | 10,0 | **nulo** — medido: se cancela algebricamente no pipeline | nada; já fechado |
| R3 | `highlight` e seus dois limiares | `False`, 0,8627451, 0,4 | **nulo no nosso caminho** — medido: a função `pipeline` **não lê** `args.highlight`; o realce acontece no corpo do `demo.py`, antes da chamada | já fechado por medição |
| R4 | qual das três saídas | `bokeh_pred` | baixo | medido: `bokeh_classical` é reta exata mas 3,81% fora de escala; `bokeh_neural` isolado devolve ~128 px independente de K — **não é renderer** |
| R5 | versão de `arnet.pth`/`iunet.pth` | commit `8b3ed556…` + sha256 dos dois | baixo | congelado e gravado |
| R6 | `K_min`/`K_max` da Eq. 5 | 0,5 / 120 / 960 | **médio** — §3.3 | histograma de `K*` do piloto + teto derivado de CoC máximo em px |
| R7 | limiar de SSIM da Eq. 5 | `None` | por definição | painel humano de 300-500 casos + `calibrate_thresholds --reviewed-csv` |
| R8 | algoritmo e critério de parada da busca | áurea, tol. 0,25, 40 avaliações | baixo | `paper.txt:561-562` dá busca binária no problema análogo |
| R9 | SSIM em **luma** ou por canal | luma, janela 11×11 σ=1,5, Wang 2004 | baixo | rodar as duas no piloto e ver se o `argmax` desloca |
| R10 | se `SSIM(K)` é unimodal **sobre foto real** | assumido | **médio** — é a hipótese da áurea e da binária do paper | varredura densa de K em ~20 amostras reais do piloto |

**Nota de fidelidade que o paper esconde e o repositório achou.** `paper.txt:406-407`
(§3.3) afirma: *"Physics-based renderers, in principle, support arbitrary shapes, yet
**public implementations typically omit this functionality**."* Mas o BokehMe público já
traz `classical_renderer/scatter_ex.py` com `ModuleRenderScatterEX` e `poly_sides` —
**abertura poligonal**. Não é PSF raster arbitrária como a Eq. 6 pede (`paper.txt:421`),
mas é mais do que o paper dá a entender.

### 4.2 Silêncios sobre a rota A

| # | silêncio | risco | o que resolveria |
|---|---|---|---|
| A1 | **qual artefato de `[80]`** — o conjunto publicado pelos autores da Generative Photography, ou imagens que nós geramos com o modelo deles. `paper.txt:996` diz *"candidate images from [80] … collections"* e não dá revisão nem contagem | alto | baixar o release público, contar as imagens, e ver se o pool com a EBB! chega a ~1,7K (`paper.txt:998`). Se não chegar, a leitura está errada |
| A2 | **qual lado da EBB!** — ela é pareada (bokeh/nítida) e o paper não diz qual entra | médio | variância do Laplaciano dos dois lados; ver qual sobrevive ao corte que produz ~1,7K |
| A3 | o **corte** do filtro Laplaciano, e se o ranking é por fonte ou global | médio | escolher o corte **por fonte** que faz o pool somar ~1,7K, e registrar o valor de cada fonte |
| A4 | a **divisão** do pool de 1,7K entre `[80]` e EBB! | baixo | nenhuma medição resolve; escolher, declarar a quota, gravar por amostra |
| A5 | **N por imagem = 41** — é `[I]` de 70K/1,7K (`paper.txt:527-528` e `:998`, os dois conferidos), não número do paper | médio | nada resolve; o que dá é gravar N e o par `(K, D_focus)` de cada variante |
| A6 | a **distribuição** de K e de `D_focus` no sorteio | médio | o paper não publica e não vai; medir e publicar a nossa |
| A7 | se K deve ser **normalizado por resolução** ao migrar de B/C para A | **alto** — `pixel_ratio` medido vai de 22,2 a 277,3, faixa de 12× | decisão nossa com fundamento físico; teste: mesma imagem em duas resoluções tem que dar o mesmo CoC relativo |
| A8 | **resolução** das imagens das duas fontes → decide se a rota A cabe no disco | alto | `identify`/PIL sobre o pool, um `Counter` de `(H,W)` |
| A9 | **custo de GPU por render** do BokehMe | **alto** — é a única etapa sem estimativa | cronometrar o piloto |
| A10 | **sobreposição** entre o pool da rota A e os quatro benchmarks | médio | sha256 exato + hash perceptual contra os quatro conjuntos de avaliação; **nunca feito** |
| A11 | se a rota A deve gravar **máscara alguma** — o paper não usa máscara na (a) | baixo | decisão nossa. A alternativa honesta a `DEPTH_BAND` é **não gravar** o campo: ausente é mais verdadeiro que uma banda sintética |

### 4.3 Silêncios sobre a rota B

| # | silêncio | risco | o que resolveria |
|---|---|---|---|
| B1 | ITW `[19]` = `atfortes/BokehDiffusion` | baixo, mas é `[I]` | nada no paper; a coincidência de autor, handle e volume (13.800 vs "13K") é o que há |
| B2 | de qual imagem sai a **máscara** — AIF ou bokeh. `paper.txt:293-294` diz *"DeblurNet recovers an AIF image. **We then** estimate depth and extract a foreground mask [86]"*: **sugere** AIF pela ordem, e só. Do `D` o paper **afirma** (`:314-315`, conferido) que vem da AIF | **médio** — numa bokeh, a região nítida *é* a região em foco | gravar os **dois** `focus_disparity` no piloto (a máscara da bokeh já é calculada em toda amostra, para a IoU) e comparar contra a distância medida. Custa uma linha |
| B3 | qual `D_focus` alimenta a Eq. 3: `1/median(1/z)` ou `median(z)` — o paper usa o mesmo símbolo nas Eq. 3 e 4 e opera em profundidade | **médio** | decisão nossa. Recomendação registrada: `1/focus_disparity`, declarado, com `median(z[M])` gravado ao lado só para auditoria — a regra "uma implementação só" pesa mais que a literalidade |
| B4 | o que fazer quando a **EXIF falta** — não há uma frase no artigo | baixo (é o lado conservador) | a regra "rejeita, nunca substitui" é nossa |
| B5 | como obter a **largura física do sensor** — não está na EXIF padrão | **alto** — 30,33% das amostras têm crop factor exatamente 1,0, e assumir 1,0 onde o real é 5,6 erra K por 5,6× | auditar os 30% por tabela `make`/`model`; o `flickr_exif` tem os dois campos. A tabela serve para **auditar**, não para preencher lacuna — não há lacuna |
| B6 | se as imagens do BokehDiffusion estão na resolução de captura, sem recorte | **alto se falso** — `pixel_ratio` e `focal_length_35` descreveriam geometrias diferentes | não medido |
| B7 | resolução (`long_side`) em que a DeblurNet roda | médio | **medido**: `long_side = 0` não recorta (identidade exata); `long_side > 0` recorta o lado curto e desloca o par até **30,86 px** em 3:2 (5.184×3.456), contra o gate antigo de 6,0 px — e **zero** em 4:3 e 16:9. Decisão implementada: `NO_CROP_MULTIPLE_OF_16` é o default |
| B8 | quanto a DeblurNet **deforma** a AIF em relação à bokeh (além do recorte determinístico) | médio | **não medido** — correlação de fase entre AIF gerada e bokeh, num piloto |
| B9 | qual DeblurNet os autores usaram para gerar os dados | baixo (não reprodutível de qualquer forma) | o paper não diz se foi o peso publicado ou um intermediário do treino de 60K |
| B10 | faixa de K plausível para a rota B | médio | histograma do piloto. Âncoras: 16,55 (kfix), 20,1 (EXIF), 15,0 (default oficial), `{0,5,10,15}` (Fig. 12) |

**Um silêncio do paper que o repositório fechou contra o texto, e vale registrar como
ganho:** `pixel_ratio` é o **maior lado**, não a largura. O corpo é vago
(`paper.txt:343-344`: *"accounts for differences in camera sensor size and image
resolution"*); a **legenda da Fig. 16** fecha (`paper.txt:1186-1187`, conferido):
*"defined as **the image's largest edge length divided by the physical sensor width
(px/mm)**"*. E a medição concorda de forma independente: `fx_px = max(W,H) ·
focal_length_35 / 36`, identidade verificada em **900/900** linhas.

Note a ironia: a regra do projeto é *"legenda de figura não manda"* (§1.1) — e aqui a
legenda é a **única** fonte que resolve. A regra correta é mais estreita: **a numeração de
referência das legendas das Figs. 4(a)/(b) não vale**; o conteúdo das legendas vale.

### 4.4 Silêncios sobre a rota C

| # | silêncio | risco | o que resolveria |
|---|---|---|---|
| C1 | a AIF f/22 **não é all-in-focus**, e o paper não trata disso | **médio** — é a raiz de todo o resto | medir `K*/k_analytic` contra `F_alvo` no piloto |
| C2 | **largura do sensor** da RealBokeh — a origem não publica | baixo (não muda o rótulo, só o validador) | EXIF de um JPEG bruto da origem. A âncora "3,6 a 36" assume 36 mm `[I]` |
| C3 | se os **421 pares** (2,05%) que a origem marca `misaligned`/`shift_*px` entram no dataset | **médio** | é um gate a mais que o piloto pode medir: `K*` e SSIM de `aligned` contra os outros |
| C4 | em que resolução o `shift_<X>px` do nome foi medido | baixo | a origem não declara; assumimos os 2000×1500 do espelho |
| C5 | **licença** dos pixels do LFDOF — sem licença explícita na página | **alto se publicarmos pixels** | resolver antes de `--store-source-images` no LFDOF |
| C6 | se "13K newly curated" **é** a RealBokeh e se a subamostragem de 2-4/conjunto é o que fizeram | médio — §3.1 | nenhuma medição resolve; é decisão a declarar |
| C7 | se `level` do LFDOF é monótono no raio ou comparável entre cenas | baixo | **declarado como não medido** no próprio adaptador, que se recusa a afirmar |
| C8 | os oito limiares de gate restantes | por definição | piloto + `calibrate_thresholds` |

### 4.5 O que o paper publica e nós seguimos sem desvio

Vale escrever, porque é a base de comparabilidade. Tudo conferido linha a linha:

| item | valor | linha |
|---|---|---|
| backbone | FLUX.1-dev (o paper grafa **FLUX-1-dev**) | `paper.txt:511` |
| LoRA rank | DeblurNet **128**, BokehNet **64** | `paper.txt:513` |
| batch efetivo | 1 por GPU × acumulação 8 × 4 GPUs RTX A6000 = **32** | `paper.txt:514-515` |
| DeblurNet | **60K** steps | `paper.txt:515` |
| BokehNet | **40K** sintético + **60K** real | `paper.txt:516` |
| passos de denoise na inferência | **28** | `paper.txt:519-520` |
| filtro de nitidez do pool | variância do Laplaciano | `paper.txt:997` |
| estimador de profundidade | Depth Pro `[7]`, sobre a AIF | `paper.txt:314-315`, `:762-764` |
| segmentador | BiRefNet `[86]` | `paper.txt:348`, `:362`, `:955-957` |
| renderizador da rota A | BokehMe `[43]`, citação literal duas vezes | `paper.txt:292`, `:331`, `:851-852` |
| quem produz o alvo em B e C | **ninguém** — é a fotografia real | `paper.txt:271` (*"Real bokeh image … Real AIF image"*), `:283` (*"Bokeh images"*), `:321` (a tupla de supervisão), `:393` (a Eq. 5 compara **contra** `I_real`) |

**Correção de citação registrada.** A auditoria da rota C cita `paper.txt:264` para *"Real
bokeh image · Real AIF image"*. A linha 264 traz os rótulos das três rotas
(*"(a) Synthetic data / (b) ITW dataset / (c) LFDOF & RealBokeh"*). As duas frases estão
**ambas na linha 271**; *"Bokeh images"* está em `:283`. Corrigido aqui.

---

## 5. Sobre "100% igual ao paper"

Não é alcançável, e o método tem que dizer isso. O repositório oficial **não publicou**:
código de treino, dados, o benchmark LF-Bokeh, `K_min`/`K_max`, limiar de SSIM,
configuração do renderer, resolução de treino, otimizador nem scheduler.

O que é alcançável, e o que este pipeline entrega: uma reprodução **pública,
dimensionalmente coerente e auditável**, com cada desvio declarado. No estado de
2026-09-10 19h35: **12 desvios declarados** (§2), **6 divergências acidentais** (§3), das
quais uma foi resolvida durante a redação e cinco seguem abertas, e **24 silêncios
assumidos** (§4). Os três números pertencem ao texto do paper, não a um apêndice.

### As três frases que eu escreveria na seção de método

1. *"Operamos o sinal de controle em disparidade métrica (1/m), não em profundidade
   linear. O artigo escreve `D` como mapa de profundidade; a Eq. 3 só fecha
   dimensionalmente com `|Δ(1/z)|`, e a inferência oficial resolve o silêncio nessa
   direção. O desvio é declarado, e é ele que torna o dataset comparável com os pesos
   publicados."*
2. *"O passo de refinamento manual de máscara descrito em §3.2(c) — 4 a 8 s por imagem,
   ~8 h no total — foi substituído por um refinamento automático baseado em retenção de
   detalhe entre a AIF e a bokeh do mesmo par. A substituição é justificada por medição: a
   máscara de saliência concorda com a distância de foco medida na captura em 35,2% dos
   casos, e declina inteiramente em 20,6% deles. Cada amostra carrega a origem da região
   em foco, de modo que treinar com e sem as amostras refinadas seja um experimento e não
   uma suposição."*
3. *"Toda quantidade em pixel — K, CoC, janela de retenção, deslocamento de par — é
   gravada junto da resolução em que foi medida, e existe uma única função de conversão
   entre resoluções, compartilhada por geração, dataloader e avaliação."*

### A frase que eu **não** escreveria, e por quê

*"Consertar o rótulo de K recupera a controlabilidade de +0,44 para +0,83."*

É a afirmação central do projeto, e a tabela de LVCorr que a sustenta **não tem
procedência rastreável** neste repositório: nem o harness, nem a definição da métrica, nem
os checkpoints, nem a data, nem o comando. Ver §2 da `TABELA_DE_EVIDENCIAS.md`. Antes de
escrevê-la, o script de avaliação precisa estar versionado aqui.
