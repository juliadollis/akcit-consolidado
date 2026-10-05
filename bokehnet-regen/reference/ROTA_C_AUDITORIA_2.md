# Auditoria adversarial da rota C — segunda passada

**Data**: 2026-09-10. **Escopo**: `src/routes/route_c.py`, `src/control/contract.py`,
`src/qc/{gates,focus_region,rejection,metrics}.py`, `src/renderer/{calibration,bokehme}.py`,
`src/sources/{realbokeh,lfdof,level_selection,mirror_images}.py`, `src/dataio/*`,
`scripts/{run_route_c,validate_focus_refinement,calibrate_thresholds,publish_release}.py`,
`tests/` inteiro.

**Suíte**: `650 testes, OK (skipped=7)` — rodados nesta máquina.
**Versão auditada**: `src/qc/focus_region.py` foi tocado por outro agente às 19:38 durante
esta auditoria; o núcleo que este relatório ataca (linhas 205-315) está inalterado, e todas
as citações abaixo são contra o arquivo às 19:38.

**Nada em `src/`, `tests/`, `scripts/`, `REGISTRO.md` ou `ACHADOS.md` foi editado.**

---

## 0. Veredito

**NÃO.** A rota C **não pode gerar o dataset final** no estado atual.

A auditoria anterior fechou com "não pronta" por cinco defeitos de contabilidade e de
calibração. Três deles foram consertados. O que aconteceu no lugar é pior: o conserto do
achado mais caro — a substituição do refinamento manual do §3.2(c) por
`qc/focus_region.py` — **introduziu um mecanismo que, medido aqui, escolhe a região em
foco errada em cena com superfície lisa fora de foco, e o faz com a nota máxima no único
gate que deveria pegá-lo**. O parâmetro que o `REGISTRO.md` indica como "o primeiro a
revisar" (`focus_top_fraction`) é **inerte**: variei-o por um fator de 200 e a região
devolvida foi bit a bit a mesma.

Isso não é uma objeção teórica. São três medições feitas nesta máquina, com o código do
repositório, reproduzíveis em segundos (§6).

O bloqueio é **P0-1/P0-2/P0-3** abaixo. Enquanto ele existir, rodar o piloto ainda vale a
pena — ele produz exatamente o histograma que confirma ou refuta o diagnóstico —, mas
**gerar as 22.990 amostras finais, não.**

| | |
|---|---|
| **P0** (bloqueia o run final) | 3 achados, todos no refinamento da região em foco |
| **P1** (consertar antes de publicar) | 8 achados |
| **P2** (melhoria) | 11 achados |
| **C1–C5** | 1 resolvido · 2 parcialmente · 2 **não resolvidos** |

---

## 1. P0 — o refinamento da região em foco

Os três achados são faces do mesmo mecanismo e por isso vêm juntos. O ataque não é ao
princípio (a razão bokeh/AIF é de fato melhor que nitidez absoluta, e o teste
`test_nitidez_ABSOLUTA_erraria_nesta_cena` prova isso); é à **implementação**, que tem
um saturador e um guarda mal dimensionados.

### P0-1 · `detail_retention` satura em 1,0, e o saturador engole `top_fraction`

`focus_region.py:227` — `np.clip(retencao, 0.0, 1.0)`. `focus_region.py:248-249` — o corte
é `quantile(retention, 1 - top_fraction)` e a região é `retention >= corte`.

Quando mais de `top_fraction` dos pixels válidos já estão **em 1,0** (o teto do clip), o
quantil cai exatamente em 1,0 e a comparação `>= 1,0` seleciona **todos os empatados**.
`top_fraction` deixa de controlar qualquer coisa.

**Medido** (`attack4.py`, cena com primeiro plano texturizado em foco + fundo texturizado
desfocado + fundo liso desfocado, ruído σ=1,0 nas duas imagens):

```
top_fraction -> área da região / fração da região no FUNDO LISO fora de foco
   0.200: area=0.2954   no_fundo_LISO=0.536
   0.100: area=0.2954   no_fundo_LISO=0.536
   0.050: area=0.2954   no_fundo_LISO=0.536
   0.010: area=0.2954   no_fundo_LISO=0.536
   0.001: area=0.2954   no_fundo_LISO=0.536
```

Duzentas vezes menos `top_fraction` e a região é **idêntica**. E 53,6% dela está no fundo
liso, que está fora de foco.

Consequências diretas:

1. A "*small yet reliable region*" do paper (`paper.txt:365-366`) vira, na prática, "toda
   região que não perdeu detalhe" — que numa foto normal é meia imagem. Em par limpo, sem
   ruído nenhum, medi área **0,469** para um `top_fraction` pedido de 0,05.
2. `REGISTRO.md:1129` diz: *"o parâmetro a revisar primeiro é `focus_top_fraction`"*.
   **Ele não pode ser revisado**: mexer nele não muda nada no regime dominante. A
   remediação registrada é inoperante.
3. `tests/test_focus_region.py:243-254` **conhece** o empate e o declara "comportamento
   certo", com asserção `area_ratio < 0.6`. A justificativa vale para a cena de teste, onde
   o lado em foco é o único lugar com retenção 1,0. Ela **não** vale quando existe
   superfície lisa fora de foco — e a cena de teste, por construção, não tem nenhuma.
4. `tests/test_focus_region.py:256-260` (`test_fracao_e_respeitada_com_gradiente`) prova
   que `top_fraction` funciona alimentando um `np.linspace` diretamente em
   `sharpest_region_mask`. Isso testa `np.quantile`, não a retenção medida de duas fotos.

**Cenário de falha concreto.** Cena RealBokeh: tronco de árvore em foco a 1 m, muro de
concreto liso ao fundo a 10 m (as duas cenas estão nomeadas em
`MEDICAO_PLANO_FOCO.md:73-75`). O muro é liso: borrar não lhe tira detalhe, retenção 1,0.
A região "em foco" fica metade no tronco e metade no muro. `focus_disparity` sai na mediana
entre 1 m e 10 m. O sweep da Eq. 5 então otimiza K contra um plano de foco que não existe,
grava `k_source="eq5_ssim_sweep"`, `focus_source="retention_only"`, SSIM alto, e nada
denuncia.

### P0-2 · o guarda `min_aif_detail = 1e-3` não dispara em fotografia real

`focus_region.py:207,223`. O guarda existe justamente para isto — a docstring diz *"Onde a
AIF é lisa não há detalhe a reter e a pergunta não tem resposta: céu limpo, parede branca.
Esses pixels recebem `NaN`"*. Ele está **três ordens de grandeza abaixo** do necessário.

**Medido** — média em janela 33 px de `|∇²luma|`, em níveis de 8 bits (`attack2.py`):

| região da AIF | σ=0 | σ=0,5 | σ=1,0 | σ=2,0 |
|---|---|---|---|---|
| texturizada em foco | 8,43 | 8,52 | 8,76 | 9,65 |
| texturizada fora de foco | 8,31 | 8,43 | 8,70 | 9,60 |
| **lisa com gradiente suave** | **0,419** | **1,671** | **2,744** | **5,017** |
| limiar `min_aif_detail` | 0,001 | 0,001 | 0,001 | 0,001 |

Um gradiente de céu **sem ruído nenhum** já dá 0,419 — 419× o limiar. Ruído gaussiano de
σ=1 nível dá 2,74, e a teoria confirma: para ruído branco, `E[|∇²|] ≈ 3,57·σ`. O guarda só
dispara em imagem **exatamente constante**, que é precisamente o único caso que
`tests/test_focus_region.py:133-137` (`test_area_lisa_vira_nan_e_nao_zero`) exercita —
`np.full((80,80,3), 120)`.

E `min_aif_detail` **não é configurável**: não está em `RouteCConfig`, não tem flag na CLI,
não aparece na proveniência por amostra. O piloto não tem como calibrá-lo sem editar fonte.

**Medido, o caso extremo**: com ruído σ=2 nas duas imagens, a região devolvida foi **100% do
quadro** (`attack1.py`, seção B), com retenção mediana 1,000.

### P0-3 · o gate `focus_region_retention` está orientado ao contrário do modo de falha

`gates.py:219-244` mede a retenção mediana **dentro** da região final e reprova quando ela é
**baixa** (`higher_is_better=True`). Nos casos de P0-1 e P0-2 a retenção mediana da região é
**exatamente 1,000** — a nota máxima possível.

```
sigma=0.0  area=0.551  ret_mediana=1.000 | EM FOCO=0.523  fundo_LISO=0.477
sigma=1.0  area=0.330  ret_mediana=1.000 | EM FOCO=0.487  fundo_LISO=0.513
sigma=3.0  area=0.296  ret_mediana=1.000 | EM FOCO=0.443  fundo_LISO=0.557
```

Nenhum valor de `--min-focus-region-retention` pega isso. O gate que o `REGISTRO.md:1045`
descreve como *"o único que julga a região refinada com a grandeza certa"* é, para o modo de
falha dominante, **um gate que não pode reprovar** — a mesma família do `check_no_leak`
tautológico que o projeto já pagou uma vez (`split.py:184-189`).

O que faltaria e não existe: um gate sobre a **dispersão de profundidade dentro da região**
(uma região que cobre 1 m e 10 m não é um plano de foco), ou sobre a **área** da região com
limiar congelado — `mask_area_ratio_max` existe (`gates.py:131`) mas está `None` e nada em
`calibrate_thresholds._GATES` o liga à região refinada com um valor defensável.

> Nota de honestidade: `src/routes/route_b.py:126-131` **já descreve este modo de falha**
> palavra por palavra — *"a retenção fica ≈ 1 em todo lugar, e `sharpest_region_mask` devolve
> o topo de 5% de um empate numérico. O refinamento produziria uma região plausível [...]
> para uma amostra completamente quebrada"* — e conclui, em `route_b.py:140`, *"Na rota C
> isso é correto"*. As medições acima mostram que não é: o empate numérico chega na rota C
> por ruído de sensor e por superfície lisa, não só por AIF gerada.

---

## 2. P1 — consertar antes de publicar

### P1-1 · `focus_agreement` é limitado pela ÁREA, não pela concordância

`focus_region.py:296` — `acordo = |inicial ∩ regiao| / |regiao|`. O denominador é a região de
retenção, não a máscara. Logo, por álgebra:

```
acordo  ≤  área(inicial) / área(regiao_retencao)
```

Com `top_fraction = 0,05` e piso 0,30, **uma máscara 100% correta com menos de 1,5% do
quadro nunca alcança o piso**, por mais certa que esteja.

**Medido** (`attack3.py`, retenção sem empate, máscara inicial inteiramente contida na região):

```
mascara 100% correta com   100 px (0.250% do quadro): acordo=0.050  -> birefnet_refined
mascara 100% correta com   300 px (0.750% do quadro): acordo=0.150  -> birefnet_refined
mascara 100% correta com   600 px (1.500% do quadro): acordo=0.300  -> birefnet
mascara 100% correta com  2000 px (5.000% do quadro): acordo=1.000  -> birefnet
```

O ramo 1 da política ("manter o que está certo") é **estruturalmente inalcançável** para
objeto pequeno. `tests/test_focus_region.py:189-194` só testa com uma máscara de meia
imagem, e por isso nunca vê isso.

**Cenário concreto**: retrato com o rosto ocupando 1% do quadro, BiRefNet acertando o rosto.
Acordo ≤ 0,20, a amostra é marcada `birefnet_refined`, a região vira `rosto ∩ retenção`, e a
amostra entra no grupo "refinadas" — poluindo a própria estatística com que
`validate_focus_refinement.py` vai julgar o refinamento.

### P1-2 · o piso de acordo é ultrapassado POR ACASO quando a máscara é grande

O espelho da mesma álgebra. Se a máscara inicial é grande, ela cobre a região de retenção
por sorteio, e o ramo "manter" dispara sem que a máscara tenha relação com o foco.

**Medido** (`attack5.py`, máscara retangular em posição **aleatória**, sem relação com o
plano de foco, região de retenção de 30% do quadro):

| área da máscara inicial | mantida como `birefnet` |
|---|---|
| 0,200 | 19,5% das posições |
| 0,300 | 34,0% |
| 0,400 | 41,5% |
| **0,526** | **62,0%** |
| 0,600 | 78,5% |

0,526 não é um número inventado: é `area@0.50` da cena `test_100` medida no job 32231
(`MEDICAO_PLANO_FOCO.md:66`). `test_101` dá 0,369.

**Consequência que derruba a premissa da etapa 9**: nas cenas em que o BiRefNet **não**
declina — que são exatamente as 162 amostras em que os 35,2% foram medidos —, as máscaras
cobrem 37% a 53% do quadro e o refinamento vai, na maioria dos casos, **não fazer nada** e
gravar `focus_source="birefnet"`, `focus_was_refined=False`. O refinamento tende a atuar só
onde não havia máscara (os 20,6%), e a deixar intocado o grupo onde o erro de 64,8% foi
medido. `validate_focus_refinement.py` vai então reportar um bloco pareado quase vazio.

Caso extremo, medido: máscara inicial = quadro inteiro → acordo **1,000** → ramo `birefnet`
→ região final = **quadro inteiro** → `focus_disparity` = mediana da cena toda,
`was_refined=False`, `mask_source="birefnet"`. **A pior região possível recebe a melhor
proveniência.** E os dois gates que poderiam pegá-la ficam mudos: `mask_area_ratio_max` está
`None` por default e `focus_mask_is_sharpest` devolve `applicable=False` porque o
complemento da máscara erodido tem 0 px (`gates.py:197-209`).

### P1-3 · C5 **não foi resolvido** — o gate continua bloqueando por default e ninguém consegue calibrá-lo

Três lugares, todos verificados:

* `contract.py:264-265` — `FOCUS_DEPTH_MIN_M = 0.05`, `FOCUS_DEPTH_MAX_M = 1000.0`, ambos
  marcados `[A] ASSUMIDOS, não medidos` na própria docstring.
* `contract.py:302-303` — `focus_disparity_from_mask` **rejeita** fora da faixa, e
  `route_c.py:537` a chama **sem override**.
* `gates.py:305-327` — `focus_depth_plausible(min_m=FOCUS_DEPTH_MIN_M, max_m=...)`, e
  `route_c.py:343` a chama **sem override**.

E o caminho de calibração não existe:

* `RouteCConfig` (`route_c.py:126-183`) **não tem campo** para esses limiares;
* `build_parser` (`run_route_c.py:343-353`) **não tem flag** — a lista de `--min-*/--max-*`
  não os inclui;
* `calibrate_thresholds._GATES` (`calibrate_thresholds.py:64-78`) **não tem entrada** para
  `focus_depth_m_min`/`focus_depth_m_max`, então nem a distribuição é proposta.

É literalmente o que a auditoria anterior escreveu no P1 item 4, e nada mudou. O documento
continua dizendo *"limiar não medido não bloqueia"* (`gates.py:3`) enquanto dois `[A]`
bloqueiam.

**Cenário concreto**: cena de macrofotografia com plano de foco medido a 0,04 m
(`MEDICAO_PLANO_FOCO.md:32` registra mediana **medida** de 0,410 m, e a distribuição tem
cauda). A amostra é rejeitada com `focus_depth_implausible` antes de qualquer gate, o
histograma diz "implausível", e não há nenhum número medido que sustente 0,05 m como
fronteira do plausível.

### P1-4 · a aritmética do teto de níveis só fecha se você usar meio dataset

`level_selection.py:23-29` justifica `max_levels=4` com: cap 4 → **13.799 ≈ "13K"**; cap 3 →
11.168, *"longe demais"*; sem teto → 20.495.

Refiz as contas de forma independente e **os três números estão corretos** — para o
histograma do split **`train`**, 3.959 cenas, 20.495 pares.

Mas `ACHADOS.md:431` mede que o espelho `akcit-pixel/RealBokeh` publica **os três splits**:
**22.990 pares em 4.399 cenas**, em 96 shards. E `run_route_c.py:149-159` indexa o espelho
inteiro e enumera tudo o que ele tem. **O run não gera 13.799.**

Extrapolando `test` (220 cenas / 1.257 pares) e `validation` (220 / 1.238) com a mesma forma
por cena do `train`:

| teto | total no espelho INTEIRO | distância de "13K" | razão |
|---|---|---|---|
| **3** | **~12.409** | **591** | **0,955** |
| **4** | **~15.333** | **2.333** | **1,179** |

**A regra de decisão do próprio docstring se inverte**: no dataset que o código realmente
processa, cap **3** é quem cai em cima de "13K", e cap 4 estoura em 18%. O "≈ 13K" que
sustenta o 4 é um artefato de ter feito a conta só no `train`. Isso é numerologia — não
porque a soma esteja errada, mas porque o recorte que faz a soma fechar não é o recorte que
o pipeline usa.

Duas ressalvas, na direção contrária, que mantenho registradas:

* `max_levels=4` **também** sai direto do texto (`paper.txt:1002-1003`, *"containing 2 to 4
  images per set"*). O teto é defensável **pelo texto**; é a aritmética confirmatória que
  não sustenta.
* A conta das máscaras (8 h ÷ 4–8 s → 3.600 a 7.200 anotações, e a RealBokeh tem 4.399
  cenas) fecha e é bonita — mas ela restringe o número de **cenas**, e **não diz nada sobre
  o teto por cena**. O docstring a chama de uma das *"duas contas independentes"* que
  fecham o teto. Ela não é. É uma conta só.

E a identificação de fundo continua frágil: se a RealBokeh **fosse** os "13K newly curated"
descritos como *"2 to 4 images per set"*, o histograma dela não teria 2.341 cenas com 5
níveis e 244 com 21. O teto não reproduz uma estrutura nativa; ele **impõe** uma.

### P1-5 · o teto de 4 é aplicado ao LFDOF, citando um trecho do paper que não fala do LFDOF

`run_route_c.py:230` chama `_aplica_teto(pares, args)` também em `_carrega_lfdof`, com o
default `--max-levels-per-scene 4`. `cap_summary` (`level_selection.py:122`) imprime então
`teto de níveis por cena: 4 (paper.txt:1001-1003)` — e `paper.txt:1001-1003` fala dos "13K
newly curated [...] focus-consistent series captured with varying **apertures**", que não é
o LFDOF (um dataset de light field, citado como `[52]`, existente e não curado por eles).

Impacto: o LFDOF tem ~11.972 linhas em 840 cenas, a maioria com 15 níveis
(`lfdof.py:186-197`). Cap 4 corta para ~3.350 — **descarta ~72% do LFDOF** com uma citação
que não o cobre.

Pior, a justificativa do critério contradiz o módulo que é dono do campo.
`level_selection.py:44-50` diz que espaçar uniformemente preserva *"a amplitude de borrão
dentro da mesma cena [...] a única coisa que varia entre níveis é K"*. Mas `lfdof.py:216-220`
declara: *"O que o nível **não** é: quantidade física. [...] **Não medimos**, e portanto não
afirmamos, que ele seja monótono no raio do desfoque"*. Espaçar uniformemente um ordinal
opaco não garante amplitude de K nenhuma.

### P1-6 · C1 sobrevive no documento canônico

`CONTRATO.md:80`:

```
| `k_value` da rota C | **3,6 a 36** | Eq. 3 sobre o `metadata/` da RealBokeh |
```

A faixa 3,6–36 é `k_analytic` (Eq. 3, borrão **absoluto**), não `k_value` (Eq. 5, borrão
**incremental** sobre uma AIF f/22). O próprio `route_c.py:270-272` diz isso com todas as
letras: *"A âncora '3,6 a 36' é de `k_analytic`"*. E `CONTRATO.md:72` instrui: *"Se uma
mudança tirar os números daqui, algo quebrou"*.

Ou seja: o piloto vai produzir `k_value` mediano legitimamente abaixo de 3,6, e a **única
página canônica do contrato** vai dizer que algo quebrou. É exatamente o C1 — "impresso como
se fosse comparável" — sobrevivendo no artefato mais autoritativo, depois de corrigido no
console.

### P1-7 · a régua de 35,2% não é reproduzível por nenhum código do repositório

`validate_focus_refinement.py:76` congela `BASELINE_WITHIN_25_PCT = 0.352` e a docstring
(`:75`) avisa: *"um número novo mais alto que este só vale se vier do mesmo cálculo"*. O
cálculo que produziu 35,2% foi feito sobre o piloto do job 32224 e **não existe no
repositório**. `validate_focus_refinement.py` reimplementa a comparação do zero.

Confirmei que as convenções batem onde dá para conferir (`razao = focus_disparity ×
focus_plane_distance_m`; `MEDICAO_PLANO_FOCO.md:30` dá mediana 0,579 e `:36` conclui "1,7×
mais longe", e `1/0,579 = 1,727` ✓). O que **não** dá para conferir é a regra de "dentro de
±25%": `_fracao_dentro` (`:246-249`) usa `|r − 1| ≤ 0,25`, que é assimétrico em escala
logarítmica (+25% contra −25% = fator 1,333). Se o script original usou `|log r| ≤ log 1,25`,
os dois números não são comparáveis.

Mitigação real, e ela existe: o bloco **pareado** (`comparacao_pareada`, `:287-330`) não
depende da régua de 35,2%. Ele é o número que decide, e está bem construído. A régua
histórica deveria ser rebaixada a contexto, não a critério.

### P1-8 · C4 não foi resolvido — `K_ABSOLUTE_MAX = 960` continua 10× além do verificado

`calibration.py:38` — `K_ABSOLUTE_MAX_DEFAULT = 960.0`. `output/renderer_verification.json`
mede o renderer em `k_values = [8, 16, 32, 64, 96]`. A CLI expõe `--k-absolute-max`
(`run_route_c.py:381`) com default `None` → usa 960.

O achado está **declarado** em `DESVIOS_DO_PAPER.md:479-486` ("ALTA"), o que é honesto, mas
declarar não é consertar. E o custo é dobrado por uma interação que ninguém registrou: a
expansão do teto (120→240→480→960) consome ~25 das 40 avaliações de orçamento, e a seção
áurea sai com `hi − lo ≈ 0,41` em vez da tolerância pedida de 0,25 — **sem nenhum campo que
registre "a tolerância não foi atingida"**. `search["evaluations"]` permite deduzir, mas
ninguém vai deduzir.

**Cenário concreto**: cena praticamente frontoparalela (`z_max/z_min` pouco acima de 1,02, o
piso de `MIN_DEPTH_RANGE_RATIO`). `Δdisp` minúsculo, o SSIM cresce monotonicamente até o
teto, K* sai em ~960, `is_censored=True`, e o renderer rodou com CoC de centenas de pixels
numa grade de 512 — regime que nunca foi medido, com dois checkpoints neurais fora de
distribuição. A amostra é gravada.

---

## 3. P2 — custo, vocabulário, documentação

| # | achado | evidência |
|---|---|---|
| P2-1 | **O sweep caro roda ANTES dos gates baratos.** `route_c.py:542` chama `calibrate_k` (17 a 42 renders BokehMe) e só em `:560` chama `enforce_gates`. No piloto é indiferente (todos os limiares `None`); no **run final**, com limiares congelados, toda amostra que um gate barato (`pair_shape`, `aif_f_number`, `aif_laplacian_variance`, `mask_area_ratio`) reprovar já pagou o item dominante do orçamento. Só `calibration_ssim` precisa do sweep. | `route_c.py:542-560` |
| P2-2 | **Segunda inferência de BiRefNet por amostra, só para diagnóstico.** `route_c.py:540` roda o segmentador na bokeh apenas para `mask_iou_aif_bokeh`, um gate com limiar `None`, documentado em `gates.py:38-40` como **não** pegando o modo de falha dominante, e `applicable=False` quando as duas máscaras são vazias. São 22.990 inferências. | `route_c.py:540` |
| P2-3 | **Slug morto e desvio do caminho único de rejeição.** `k_search_budget_exhausted` está registrado (`contract.py:156`) e **nunca é usado**; `calibration.py:169` levanta `k_out_of_configured_range` para orçamento esgotado. Além disso `calibration.py:145,147,169` fazem `raise SampleRejected(...)` direto, contornando `reject()` — que é a função que valida o vocabulário fechado (`contract.py:176-178`). | idem |
| P2-4 | **Docstring afirma um valor que o código não produz.** `level_selection.py:67` diz `total=21, keep=4 -> [0, 6, 13, 20]`. Medido: `[0, 7, 13, 20]` (`round(20/3·1) = 7`). | `level_selection.py:67` |
| P2-5 | **`top_fraction` é fração dos pixels VÁLIDOS, não do quadro.** `focus_region.py:76` diz "Fração da imagem"; `:248` calcula o quantil sobre `retention[validos]`. Numa cena com muita área NaN, a região absoluta é bem menor que o anunciado. | `focus_region.py:76,248` |
| P2-6 | **`meta/<id>.json` justapõe `k_value` e `k_analytic` sem marcador semântico.** `sample.py:313-314`. C1 foi consertado no console (`route_c.py:277-282`) e em `calibrate_thresholds.py:215-231`, mas o artefato durável — o JSON que alguém vai ler daqui a seis meses — continua colocando dois números incomparáveis lado a lado, sem nota. | `sample.py:313-314` |
| P2-7 | **O teto de níveis sempre mantém o extremo mais fechado.** `_uniform_indices` inclui sempre `0` e `total-1`, e `target_avs` está em ordem **crescente** em 4.400/4.400 JSONs (`realbokeh.py:56`). Logo toda cena com >4 níveis contribui com a abertura mais fechada — o alvo mais parecido com a AIF f/22. São 2.630 das 3.959 cenas do `train` → ~19% do lote é o alvo de bokeh mais fraco de cada cena, com previsão testável de K* encostando no piso e saindo `is_censored` (`calibration.py:219`). | `level_selection.py:62-74` |
| P2-8 | **Dois `[A]` sem flag na CLI**: `focus_min_region_area_ratio` (existe em `RouteCConfig:180`, sem flag) e `min_aif_detail` (nem campo nem flag, ver P0-2). | `run_route_c.py:363-376` |
| P2-9 | **A janela de 33 px é aplicada a duas fontes com resoluções 2,9× diferentes** sem justificativa por fonte: RealBokeh 1500×2000 (1,65% do lado longo) e LFDOF 688×1008 (3,3%). É gravada por amostra (`focus_retention_h/w`, `sample.py:255-257`), então é auditável — mas continua sendo o mesmo `[A]` significando duas coisas. | `focus_region.py:73`, `lfdof.py` |
| P2-10 | **Gate com `threshold=None` que ainda assim bloqueia.** `GateResult.passed` (`gates.py:79-80`) reprova valor não-finito antes de olhar o limiar. `bokeh_is_blurrier_than_aif` devolve `inf` quando `laplacian_variance(aif) == 0` (`gates.py:356`) e `focus_mask_is_sharpest` devolve `inf` quando a nitidez fora da máscara é 0 (`gates.py:213`) — os dois com `applicable=True`, logo **rejeitam** com um slug de mérito. É o mesmo mecanismo de C2, sobrevivendo em dois pontos que não foram convertidos. | `gates.py:213,356` |
| P2-11 | **`focus_agreement_floor = 0.0` seria uma bomba.** Com o piso em zero, o ramo 1 dispara com `acordo = 0` e `_empacota(inicial, ...)` pode devolver `retention_in_region = NaN` (`focus_region.py` `_empacota`), que o gate `focus_region_retention` transforma em rejeição `gate_focus_region_retention_low` — um slug afirmando "reteve pouco" sobre uma medida que não existe. Com o default 0,30 o ramo é inalcançável; a CLI expõe `--focus-agreement-floor` sem validação de faixa (`run_route_c.py:375`). | `run_route_c.py:375` |

---

## 4. Estado de C1 a C5

| | achado anterior | estado | evidência |
|---|---|---|---|
| **C1** | `k_analytic` incomparável com `K*` e impresso como se fosse | **parcialmente resolvido** | Console (`route_c.py:270-289`) e `calibrate_thresholds.py:215-231` agora separam as duas grandezas, imprimem a razão e explicam a expectativa `k_value < k_analytic`. **Não resolvido** em `CONTRATO.md:80` (P1-6) nem no `meta/*.json` (P2-6). |
| **C2** | `aif_f_number=None` reprovando, matando o LFDOF inteiro | **RESOLVIDO** | `GateResult.applicable` (`gates.py:61-83`); `aif_aperture_is_narrow` devolve `applicable=False` para `None` e **reprova** para f-number inválido (`gates.py:379-390`) — a distinção certa. Adaptador do LFDOF existe (`sources/lfdof.py`, `lfdof_images.py`) e `run_route_c.py --source lfdof` funciona. Ressalva: o mesmo mecanismo NaN→reprova sobrevive em dois gates (P2-10). |
| **C3** | multiplicidade por cena: 21 alvos/cena contra "2 a 4" do paper | **parcialmente resolvido** | `sources/level_selection.py` existe, default `--max-levels-per-scene 4`, aplicado **antes** da amostragem do piloto (`run_route_c.py:163,187-192` — e essa ordem está **certa**, pelo motivo que o docstring dá). Mas a aritmética que justifica o **valor 4** não sustenta no dataset que o código processa (P1-4), e o teto é estendido ao LFDOF com citação imprópria (P1-5). |
| **C4** | `K_ABSOLUTE_MAX = 960`, 10× além da faixa verificada | **NÃO resolvido** — declarado | `calibration.py:38` inalterado. Registrado como severidade ALTA em `DESVIOS_DO_PAPER.md:479-486` e como `R6` na tabela de riscos (`:572`). Ver P1-8. |
| **C5** | `FOCUS_DEPTH_MIN/MAX_M` bloqueando por default sem o piloto calibrá-los | **NÃO resolvido** | Nada mudou em `contract.py:264-265,302-303`, `gates.py:305-327`, `route_c.py:343,537`. Sem campo em `RouteCConfig`, sem flag na CLI, sem entrada em `calibrate_thresholds._GATES`. Ver P1-3. |

**Placar: 1 resolvido, 2 parcialmente, 2 não resolvidos.**

---

## 5. As perguntas do briefing, respondidas

### 5.1 A retenção mede o que promete?

Mede, **onde a AIF tem detalhe de alta frequência a perder**. Fora daí, não. O denominador
normaliza pela textura da cena — e essa é a parte boa, provada por `test_focus_region.py:118`.
Mas a razão `borrado/nítido` só distingue foco de desfoco quando o numerador **pode** cair, e
ele não pode em:

* **Superfície lisa** (parede, água, névoa, asfalto, céu com gradiente). Borrar não muda nada.
  Retenção 1,0 no fundo fora de foco. **Medido, é o P0-1/P0-2.**
* **Ruído de sensor.** O borrão óptico acontece **antes** do sensor; o ruído é somado
  **depois**. Em região de baixa textura, `|∇²|` das duas imagens é o ruído, e a razão é 1
  independentemente do foco. **Medido: com σ=2 níveis, a região vira o quadro inteiro.**
* **Especular / ponto de luz.** O disco de bokeh de um ponto de luz tem borda dura: `|∇²|` na
  bokeh pode igualar ou superar o do ponto na AIF. Cena noturna com luzes ao fundo →
  a retenção aponta para o fundo. **Não medido** (exigiria PSF real), mas é o mesmo mecanismo
  do clip em 1,0 e é consistente com `focus_region.py:226` ("acima de 1 é ruído ou
  desalinhamento").
* **AIF f/22.** A difração amolece a AIF, o denominador encolhe, a razão sobe. **Medido**
  (`attack1.py`, seção C): amolecendo a AIF com raio 1–3 px, a área da região sobe de 0,469
  para 0,550 e a fração dela que cai no lado em foco cai de 1,000 para 0,910. O efeito é na
  direção prevista e é de segunda ordem comparado ao P0-1.
* **Desalinhamento.** Nos 421 pares que a origem anota `misaligned`/`shift_<X>px`
  (`realbokeh.py:70-73`), o deslocamento cria `|∇²|` na bokeh onde a AIF é lisa → razão > 1 →
  clipada em 1 → **entra na região**. **Medido** (`attack1.py`, seção D): com shift de 2 a 5
  px a área da região cai para ~0,27 mas 28% dos pixels finitos continuam clipados em 1,0.
  Na cena sintética o efeito não moveu a região de lado; numa borda de oclusão real ele
  moveria, e ali a profundidade é ambígua por definição. `REGISTRO.md:1150` já registra isto
  como não medido.

### 5.2 A política de três ramos

* **Ramo 1 (manter)** — alcançável, mas por **área**, não por concordância: inalcançável para
  máscara < 1,5% do quadro (P1-1) e alcançado **por acaso** em 62% das posições para máscara
  de 52,6% (P1-2).
* **Ramo 2 (intersectar)** — alcançável. É onde caem as máscaras pequenas corretas, por
  engano.
* **Ramo 3 (só retenção)** — alcançável, e é o único ramo que faz o que o §3.2(c) manda
  (preservar amostra em vez de descartar). É também o ramo mais exposto ao P0.

### 5.3 Os `[A]` — o piloto consegue calibrá-los?

| parâmetro | flag na CLI | gravado por amostra | calibrável na prática |
|---|---|---|---|
| `focus_retention_window_px` (33) | sim | sim | sim |
| `focus_top_fraction` (0,05) | sim | sim | **NÃO — é inerte (P0-1)** |
| `focus_agreement_floor` (0,30) | sim | sim | sim, mas mede área (P1-1/P1-2) |
| `focus_min_region_area_ratio` (0,001) | **não** | sim | só editando fonte |
| `min_aif_detail` (1e-3) | **não** | **não** | só editando fonte — **e é o que está errado (P0-2)** |
| `FOCUS_DEPTH_MIN/MAX_M` | **não** | não | **não (C5)** |

Justificativa dos valores além de "escolhemos": **nenhuma**, e o código diz isso
honestamente em quatro lugares (`focus_region.py:72,75,80,84`; `route_c.py:480`;
`MEDICAO_PLANO_FOCO.md:142-143`). A honestidade está correta; o problema é que dois deles não
têm caminho de calibração e um não tem efeito.

### 5.4 A retenção significa a mesma coisa na rota B?

**Não**, e a rota B já documenta isso melhor do que a rota C: `route_b.py:111-143` explica que
ali a AIF é produzida pela DeblurNet a partir da própria bokeh, então a razão mede "onde a
DeblurNet acrescentou alta frequência", não óptica; e por isso a rota B usa a retenção **só
como diagnóstico** (`route_b.py:748-806`), nunca como rótulo. Essa decisão está certa.

O que a rota B erra é a conclusão simétrica (`route_b.py:140`: *"Na rota C isso é correto"*)
— ver a nota ao fim do §1.

### 5.5 Ordem de operações

* **Teto de níveis antes da amostragem do piloto** — `run_route_c.py:163,230`. **Certo**, e
  pelo motivo declarado em `:187-192`: com o teto depois, o piloto calibraria limiares para
  uma distribuição que não é a que vai ser gerada.
* **Gates caros antes ou depois do sweep** — **errado**: o sweep vem primeiro (P2-1).
* **`focus_disparity_from_mask` antes do sweep** — certo (`route_c.py:537` antes de `:542`):
  uma amostra sem plano de foco não paga o sweep.

### 5.6 Caça sistemática — resultado

* **Fallback numérico**: não achei nenhum novo. `_analytic_k` devolve `None` sem inventar
  sensor (`route_c.py:622-632`), `_baseline_focus_disparity` devolve `None` e distingue
  ausência de defeito (`route_c.py:494-506`), `_ComSensor` obriga a passar `--sensor-width-mm`
  explicitamente e marca `[A]` na proveniência (`run_route_c.py:107-127`). A disciplina está
  de pé.
* **Unidade errada**: fiz a análise dimensional de `k_eq3_mm` / `k_official` /
  `k_at_resolution` / `k_for_bokehme` / `signed_coc_px` / `defocus_map` e da conversão de K na
  resolução de trabalho de `calibrate_k` (`:170`, `k_full * scale`, e o `k_star` devolvido na
  escala original). **Nenhum erro de unidade.** Em particular `_analytic_k` passa
  `focus_plane_distance_m` em **metros** para `k_eq3_mm`, cujo terceiro parâmetro se chama
  `focus_depth_m` e converte internamente (`contract.py:380`) — está certo, apesar do nome da
  função sugerir milímetro.
* **Gate que não pode reprovar**: `focus_region_retention` para o modo de falha dominante
  (P0-3).
* **Gate que reprova o que não devia**: P2-10 (dois gates com `inf` e `applicable=True`) e
  P1-3 (`focus_depth_plausible`).
* **`applicable=False` virou passe livre?** Em `mask_iou`, **não** — o uso está certo e é o
  conserto do C2. Em `focus_mask_is_sharpest`, **sim, na margem**: o gate que testa a hipótese
  física da máscara fica mudo precisamente quando a região refinada sai pontilhada ou quase
  integral — que são os dois casos produzidos pelo caminho novo (`gates.py:197-209`,
  `REGISTRO.md:1063-1066`). Não é passe livre criado de má-fé; é passe livre correlacionado
  com o defeito.
* **Teste que testa a si mesmo**: `test_fracao_e_respeitada_com_gradiente` (testa
  `np.quantile`), `test_area_lisa_vira_nan_e_nao_zero` (testa o único caso em que o guarda
  dispara), e as cenas sintéticas `_cena_com_armadilha` / `_scene_images` que são
  **uniformemente texturizadas** — a condição sob a qual a retenção funciona. Não achei
  asserção tautológica nem dublê medindo a coisa errada; o `_LAYERED`
  (`test_route_c.py:110-146`) é honesto e sua docstring registra por que o `_DISC` não servia.
* **Proveniência que mente**: não achei. `mask/<id>.png` é a região final, `mask_source` sai
  da ponte única `FOCUS_SOURCE_TO_MASK_SOURCE`, e `_validate_focus_region`
  (`sample.py:420-474`) rejeita a contradição com slug próprio. Os parâmetros `[A]` do refino
  vão **na amostra**, não só no `run_config.json` (`route_c.py:591-597`) — isso está bem
  feito. A única "mentira" que achei é de outra natureza: no caso P1-2 a proveniência diz a
  verdade sintática (`birefnet`, `was_refined=False`) sobre um resultado semanticamente
  péssimo.
* **`check_no_leak`**: consertado de verdade (`split.py:181-199` compara o `split` gravado em
  cada linha contra o `split.json`), com o defeito antigo documentado no próprio docstring.
* **`publish_release.py`**: sólido. Verifica arquivo por amostra, `control_version` único,
  `depth_backend` único, ledger de sha256 dos pixels de origem, cenas fora do split,
  vazamento, e **reprova** com censura acima de 20% (`:172-177`). O aviso de "refinadas > 50%"
  (`:157-165`) é bom. Não achei porta dos fundos.

---

## 6. O que medi, e como reproduzir

Todas as medições deste relatório usaram **o código do repositório sem modificação**, via
scripts avulsos que só importam `src/qc/focus_region.py` e chamam suas funções públicas.
Os cinco scripts estão em
`/private/tmp/claude-501/-Users-juliadollis-Projects-Code-repositorio-ref---c-pia-5/465081fe-4eea-4f3e-8c14-538ece4427ed/scratchpad/`
e rodam com `/Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python` em segundos, sem GPU:

| script | o que prova |
|---|---|
| `attack1.py` | par limpo: região de 0,469 para `top_fraction=0,05`. Ruído σ≥2 no alvo → região = 100% do quadro. AIF amolecida (difração) → região cresce e vaza. Shift 2–5 px → 28% dos pixels clipados em 1,0. |
| `attack2.py` | cena com fundo **liso** fora de foco: 44–56% da região cai nele, retenção mediana 1,000, NaN = 0,000. Tabela de `|∇²|` da AIF por região contra `min_aif_detail`. |
| `attack3.py` | limite estrutural do acordo; máscara correta de 0,25%–1,5% do quadro nunca alcança 0,30; máscara de quadro inteiro → `birefnet`, área final 1,000. |
| `attack4.py` | `top_fraction` de 0,20 a 0,001 → região **idêntica**. |
| `attack5.py` | piso de acordo ultrapassado por acaso: 62% das posições aleatórias para máscara de 52,6%. |

Suíte do repositório: `650 testes, OK (skipped=7)`.

---

## 7. O que continua sem poder ser verificado sem GPU

Separando rigorosamente:

**Rodou em GPU** (e é `[M]`):
* job **32212** — verificação do renderer: disco não gaussiana (`edge_width_ratio = 0,143`),
  linearidade em K com resíduo 0,0000 px em `bokeh_classical`, `k_effective_factor = 0,9873`.
  Faixa medida: **K ∈ {8, 16, 32, 64, 96}**, e só ela.
* job **32224** — piloto da rota C, 162 amostras aceitas de 204, 29 cenas: os **35,2%** de
  concordância, a razão mediana 0,579, os **20,6%** de `focus_mask_empty`.
* job **32231** — `diagnose_empty_masks.py`: probabilidade **exatamente 0,000** do BiRefNet em
  5 de 12 cenas, e `area@0.50` de 0,526 / 0,369 nas duas que não declinaram.

**Nunca rodou** — e isto é tudo o que segue:

1. **`scripts/validate_focus_refinement.py` nunca foi executado.** A régua de 35,2% não tem
   contraparte. Sem ela, **não se sabe se o refinamento melhora ou piora** a concordância com
   o gabarito. Minhas medições dizem que ele tem um modo de falha grave; elas **não** dizem
   qual a frequência dele nas 22.990 amostras reais. Isso só o piloto responde.
2. **A composição do lote por `focus_source`.** Se o P1-2 estiver certo, o piloto vai mostrar
   `birefnet` dominante nas cenas em que o BiRefNet não declina e `retention_only` nas outras,
   com o bloco pareado de `validate_focus_refinement.py` quase vazio — porque o ramo pareado
   só existe onde havia máscara inicial **e** ela foi refinada. Se esse bloco vier com poucas
   dezenas de amostras, **o refinamento não terá sido validado**, independentemente do que o
   relatório imprimir.
3. **A distribuição real de `focus_region_retention`.** Previsão desta auditoria: mediana
   muito próxima de 1,000 e p05 alto. Se for isso, o gate não separa nada e `calibrate_thresholds`
   vai propor um corte que descarta ~5% escolhidos essencialmente ao acaso.
4. **A fração de área da região refinada.** Previsão: bem acima dos "small yet reliable" do
   paper — na casa de 0,3 a 0,6 do quadro. `focus_region_area_ratio` está gravado
   (`sample.py:254`), então o piloto responde isso direto.
5. **A taxa de censura de K** e se ela passa dos 20% que `publish_release.py:174` usa para
   reprovar o release. Duas fontes de pressão para cima: o teto de níveis mantendo sempre a
   abertura mais fechada (P2-7, piso) e as cenas frontoparalelas (P1-8, teto).
6. **O impacto de C5** — quantas amostras `focus_depth_implausible` e
   `gate_focus_depth_implausible` reprovam de fato. O histograma responde; o limiar continua
   sem ter como ser mexido depois (P1-3).
7. **Se `K* < k_analytic` na massa**, que é a previsão da hipótese "a AIF f/22 já carrega parte
   do borrão" (C-A5 da auditoria anterior). Exige `--sensor-width-mm`, que é ele próprio um
   `[A]`.
8. **O efeito do desalinhamento** nos 421 pares anotados, e se `--focus-retention-long-side`
   o atenua. `REGISTRO.md:1150` já registra como não medido.
9. **O regime do renderer acima de K=96**, que `K_ABSOLUTE_MAX=960` autoriza.
10. **Custo real do sweep por amostra** e, portanto, o custo de P2-1.

---

## 8. Ordem sugerida de conserto

**Antes do próximo piloto** (barato, muda o que o piloto mede):

1. Elevar `min_aif_detail` para um valor derivado do ruído da fonte — ou, melhor, substituir o
   guarda absoluto por um **relativo** (ex.: percentil da própria distribuição de
   `detalhe_aif`), expondo-o como `[A]` com flag e gravando-o na proveniência. **P0-2.**
2. Trocar a seleção `retention >= corte` por uma que não empate no teto do clip: usar a razão
   **antes** do clip para ordenar, ou desempatar por `detalhe_aif` (mais textura = mais
   informativo). **P0-1.**
3. Acrescentar um gate sobre a **dispersão de disparidade dentro da região final** — uma
   região que cobre 1 m e 10 m não é um plano de foco. É o gate que falta e que pegaria os
   três P0 de uma vez. **P0-3.**
4. Redefinir `focus_agreement` para não ser uma medida de área: normalizar por
   `min(|inicial|, |regiao|)` ou usar IoU. **P1-1/P1-2.**
5. Expor `FOCUS_DEPTH_MIN/MAX_M` como limiar de verdade (campo, flag, entrada em `_GATES`) e
   pôr o default em `None`. **P1-3/C5.**

**Antes de publicar**: P1-4 (decidir o teto com a conta feita no espelho inteiro, e declarar),
P1-5 (não aplicar o teto do §B.2 ao LFDOF, ou justificá-lo por outro caminho), P1-6 (corrigir
`CONTRATO.md:80`), P1-8 (baixar `K_ABSOLUTE_MAX` para a faixa verificada, ou re-verificar o
renderer até 960).

**Depois**: os P2, com P2-1 primeiro por ser o único com impacto de custo de GPU.
