# Rota B — decisões declaradas

Data: 2026-09-10. Acompanha `src/routes/route_b.py`, `scripts/run_route_b.py`,
`tests/test_route_b.py` (104 testes) e `slurm/route_b_{pilot,full}.slurm`.

Autoridade, na ordem do projeto: (1) o paper (arXiv:2512.16923v3), onde ele fala;
(2) `third_party/Genfocus/Inference_*.py`, onde o paper cala; (3) decisão nossa,
**declarada como desvio**, onde os dois calam.

Etiquetas: `[M]` medido · `[I]` inferido de algo medido · `[A]` assumido, não verificado.

O que a auditoria (`reference/ROTA_B_AUDITORIA.md`) definiu está implementado; o que ela
deixou aberto está decidido aqui, com a razão e com o mecanismo que torna a decisão
auditável por amostra. **Nenhum limiar foi congelado**: todos os defaults são `None`.

---

## 0. O que a rota B é, em uma linha

Foto com bokeh real do ITW → DeblurNet gera a AIF → Depth Pro e BiRefNet rodam **sobre a
AIF** → Eq. 4 dá `D_focus` → Eq. 3 dá K a partir da EXIF → **o alvo é a própria
fotografia**. Nenhum renderizador (paper.txt:271, 283, 292, 321).

---

## 1. Decisão 1 — qual `D_focus` alimenta a Eq. 3

### O problema

O contrato produz `focus_disparity = median(1/z[M])` (`control/contract.py:297`), por
autoridade do código oficial (`Inference_bokehNet.py:118`; `CONTRATO.md:28-34`). A Eq. 3
(`k_eq3_mm`) pede uma **distância**. E `1/median(1/z) ≠ median(z)`: a mediana só é
invariante sob transformação monótona em contagem ímpar; `np.median` faz a **média** dos
dois valores centrais quando é par, e a média de dois recíprocos não é o recíproco da
média.

### A decisão

**A Eq. 3 recebe `1/focus_disparity`.** `median(z[M])` é calculado ao lado e gravado, e
nunca entra em conta nenhuma.

### Por quê, em ordem de peso

1. **É o mesmo plano de foco que gera o mapa.** `defocus_map` usa `focus_disparity`. Se a
   Eq. 3 usasse `median(z[M])`, a amostra teria **dois** planos de foco discordantes — que
   é literalmente o defeito B16: `ACHADOS.md:58` mede a divergência entre o `s1` gravado e
   o implícito em mediana 0,00308, **p90 0,07927**, máximo **0,48821**, com a conclusão
   *"são planos de foco diferentes"* `[M]`.
2. **Regra 5 do `CONTRATO.md`**: uma implementação só. `focus_disparity_from_mask` é a
   única porta para a Eq. 4, e ela devolve disparidade. Um segundo caminho para a mesma
   grandeza é como o projeto chegou a quatro interpretações de K.
3. A magnitude do desvio é a do defeito B5 — *"pequeno e sistemático"*.

### O desvio, e como ele fica auditável

O paper escreve `D_focus = median(D[M])` sobre um mapa de **profundidade**
(paper.txt:352), então a leitura literal é `median(z[M])`. O desvio é declarado, e cada
amostra grava, em `provenance.extra.focus_depth_diagnostics`:

| campo | o que é |
|---|---|
| `focus_depth_m_from_disparity` | `1/median(1/z[M])` — **o que entra na Eq. 3** |
| `focus_depth_m_median_z` | `median(z[M])` — a leitura literal do paper |
| `focus_depth_ratio` | a razão entre as duas; `1,0` = coincidem |
| `focus_depth_convention` | a string `"1/median(1/z[M])"` |

O resumo do run imprime p05/mediana/p95 de `focus_depth_ratio`. Assim a frase *"a
diferença é pequena"* deixa de ser suposição: ela vira um histograma. E quem quiser
refazer o rótulo pela leitura literal tem o número por amostra, sem reprocessar imagem.

### Medido nos testes

`tests/test_route_b.py` prova as duas metades da afirmação, e isso importa para não
ler a Decisão 1 como maior do que ela é:

* num plano de profundidade **contínuo**, as duas leituras coincidem em 5 casas
  (`test_a_divergencia_e_PEQUENA_num_plano_continuo`) — porque os dois valores centrais da
  mediana são vizinhos, e média harmônica e aritmética de vizinhos diferem em
  O((z_a − z_b)²);
* numa máscara que cruza **dois planos** com número igual de pixels, a razão cai abaixo de
  0,5 (`test_as_duas_leituras_DIVERGEM_quando_a_mascara_cruza_DOIS_planos`).

Isto é: a escolha só tem consequência quando a máscara é ruim — e é exatamente aí que ela
tem que estar decidida em vez de acidental.

**Item da auditoria fechado**: `[A]` A5 → decisão nossa declarada, com diagnóstico por
amostra. A escala do desvio segue `[A]` até o piloto medir `focus_depth_ratio` no ITW.

---

## 2. Decisão 2 — a rota B **não** aplica o refinamento da região em foco

### O que o paper diz, e onde

Duas passagens governam:

> *"Specifically, we estimate an in-focus mask M using BiRefNet [86] and produce a depth
> map D using a monocular depth estimator. The focus plane is then determined as the
> median depth within the masked in-focus area"* — paper.txt:348-350, **dentro do
> parágrafo (b)**.

> *"(c) LFDOF and RealBokeh. [...] **Similar to (b)**, we employ BiRefNet [86] to obtain
> an initial in-focus mask M. However, **due to the increased diversity and complexity of
> the scenes in these datasets**, the initial estimate of M is sometimes unreliable.
> Rather than simply verifying and discarding unreliable cases, we introduce a manual
> refinement step."* — paper.txt:361-365

Duas coisas nessa segunda passagem decidem a questão:

* o **"Similar to (b)"** identifica (b) como o passo **sem** refinamento — o refinamento é
  o que (c) acrescenta;
* o **"in these datasets"** é uma afirmação **comparativa**: o refinamento existe porque
  LFDOF e RealBokeh são mais difíceis **que o ITW**.

Para (b) o paper especifica exatamente três passos, e nenhum é refinamento.

### A decisão

A rota B usa a máscara do BiRefNet **como ela veio** — `focus_source = "birefnet"`,
`focus_was_refined = false`, `mask_source = "birefnet"` — e **rejeita** a amostra com
`focus_mask_empty` quando ela não serve. Não há cascata para RMBG nem GrabCut (defeito
B7): um segmentador só.

### Por que os 35,2% não transferem

`reference/MEDICAO_PLANO_FOCO.md` mediu que a máscara do BiRefNet acerta o plano de foco
em **35,2%** dos casos e declina (probabilidade exatamente 0) em **20,6%** `[M]`. Esses
números são da **RealBokeh**, e a própria medição explica a causa: *"a RealBokeh é feita de
**cenas**, não de fotos de objeto — um tronco de árvore num parque, um muro de pedra. O
BiRefNet é um segmentador de objeto **saliente**, e está sendo usado fora do domínio
dele."*

O ITW é feito de fotos do Flickr em que o fotógrafo compôs e focou um **sujeito**. É
justamente o domínio em que saliência e plano de foco coincidem — e é a mesma razão que o
paper dá para separar (b) de (c). Transferir os 35,2% para cá seria decidir por analogia
onde o paper decidiu pelo contrário.

**Não temos gabarito de distância de foco no ITW**, então isso é `[A]`, não `[M]`. Por
isso a decisão vem com gatilho de revisão.

### Gatilho declarado para revisitar

Os dois números saem do resumo do piloto:

| sinal | limite | o que significa passar dele |
|---|---|---|
| `focus_mask_empty` | **> 5%** das amostras processadas | o BiRefNet está declinando no ITW como declinava na RealBokeh |
| `focus_agreement` mediano | **< 0,30** (o `DEFAULT_AGREEMENT_FLOOR`) | a máscara discorda da física na maioria das cenas |

Qualquer um dos dois reabre a Decisão 2. E a reabertura é barata: `qc.focus_region` já
existe, testado, e a mudança é chamar `refine_focus_mask` em vez de aceitar a máscara —
com a ressalva grave da §4 abaixo.

**Item da auditoria fechado**: a pergunta *"a rota B usa o refinamento?"* → **não**, com
linha do paper e gatilho de revisão medido no piloto.

---

## 3. Decisão 3 — a variante da DeblurNet, e o default

Decisão da usuária, registrada: **apenas a DeblurNet oficial do paper** por agora.

| | valor |
|---|---|
| default da CLI | `official_cond_only` |
| repositório | `nycu-cplab/Genfocus-Model` |
| arquivo | `deblurNet.safetensors` |
| `main_adapter` | `None` — e isto é o **correto** para um LoRA cond-only |
| evidência | `Inference_deblurNet.py:11,88-89,103-111`; `download_models.py:11` |

`--deblur-variant ours_main_cond` existe e funciona, e é o que será usado quando o treino
main+cond terminar — **em `--output-dir` novo**, regerando a rota B. A infraestrutura das
duas variantes vive em `src/model_runtime/deblurnet.py` (65 testes) e não foi tocada.

Cinco mecanismos impedem a mistura das duas versões, e nenhum é convenção verbal:

1. **Um `--output-dir` por variante.** `scripts/run_route_b.py::_confere_variante_uniforme`
   **recusa** continuar uma pasta cuja metade gravada é de outra variante. Ele lê o
   `generated_images.jsonl` que a rota escreve, porque a linha do manifesto ainda não
   carrega `deblur_variant` — ver §7.
2. **`main_adapter` não é parâmetro de nada.** Não há `--main-adapter` na CLI, e o
   `DeblurNetRuntime` não aceita o argumento: ele vem da spec da variante. O antipadrão
   exato é `deblurnet-eval-pipeline/infer_and_eval.py:98-101`, uma flag de texto livre com
   default — um default do lado errado **produz** o defeito B1, e um do lado certo o
   **esconde**.
3. **`deblur_variant` e `deblur_lora_sha256` na proveniência de toda amostra.**
4. **O mesmo `sample_id` nas duas versões** (`b_<flickr_photo_id>`), sem sufixo de
   variante. É o que torna as duas comparáveis par a par: mesma foto, mesma EXIF, mesmo
   `scene_id` — e a diferença de K entre as versões vira uma **medida direta** de quanto a
   AIF influencia o rótulo.
5. **`--deblur-lora-sha256`** como trava opcional: um peso renomeado passa pela checagem de
   nome de arquivo, o hash não passa.

O que **não** diverge entre as duas versões, e não pode divergir sob pena de elas
deixarem de ser comparáveis: `MAX_COC`, `CONTROL_VERSION`, `DEPTH_LONG_SIDE`, a
codificação da profundidade, o split, e os limiares dos gates.

---

## 4. Decisão 4 — o que `detail_retention` significa na rota B

Esta é a conclusão que a tarefa pediu, e ela muda a leitura de um número que já existe.

Na **rota C**, `retencao(x) = média_local(|∇²bokeh|) / média_local(|∇²aif|)` compara
**duas fotografias reais e independentes**. Onde a razão é alta, a óptica preservou o
detalhe que a AIF tem — logo aquele pixel estava no plano de foco. É **evidência física**,
e é o que autoriza a retenção a *propor* a região em foco.

Na **rota B a AIF é produzida pela DeblurNet a partir da própria bokeh.** A razão deixa de
medir a óptica e passa a medir **onde a DeblurNet acrescentou alta frequência**. Três
consequências, e a terceira é a útil:

1. **Não é evidência independente.** Usá-la para definir `D_focus` faria o rótulo depender
   da DeblurNet **duas vezes** — uma na AIF que alimenta o Depth Pro, outra no plano de
   foco. O rótulo inteiro colapsaria na crença de um único modelo.
2. **Ela esconderia o defeito de maior impacto da rota.** Se o LoRA não carregar, a AIF sai
   ≈ igual à bokeh; a retenção fica ≈ 1 em **todo lugar**; e `sharpest_region_mask` devolve
   o topo de 5% de um **empate numérico**. O refinamento produziria uma região plausível,
   marcada `focus_source = "retention_only"`, com aparência inteiramente normal no
   metadado, para uma amostra **completamente quebrada**. Aplicar o refinamento aqui seria
   construir o mecanismo que mascara o defeito B1 — o defeito que a auditoria coloca em
   primeiro lugar. Isto é um argumento **independente** da §2: mesmo se o paper aplicasse o
   refinamento em (b), ele precisaria de uma salvaguarda que o paper não descreve.
3. **Por isso ela é um bom diagnóstico.** A auditoria pede em §4.3 um *"segundo eixo"*
   porque *"uma razão de nitidez sozinha não separa"* "não deblurou" de "lavou". A
   **dispersão** da retenção é esse eixo:

   | caso | dispersão da retenção | SSIM(AIF, bokeh) |
   |---|---|---|
   | deblur bem-feito | larga: alta no sujeito, baixa no fundo | intermediário |
   | LoRA não carregou | **≈ 0**, concentrada em 1,0 | **≈ 1** |
   | AIF lavada / alucinada | baixa em todo lugar | **baixo** |

Então: **medida sempre, gravada sempre, rótulo nunca.** Cada amostra carrega
`deblur_retention_p05/p50/p95`, `deblur_retention_spread`,
`deblur_retention_measurable_fraction` e a string `retention_semantics`, que diz no próprio
dado que aquela razão não é o que ela é na rota C.

**Limitação registrada.** `detail_retention` satura em 1,0 (`focus_region.py:210`,
*"acima de 1 é ruído ou desalinhamento"*). Na rota C isso é correto. Aqui, bokeh **mais**
detalhada que a AIF num pixel significa que a DeblurNet **removeu** detalhe ali —
informação real que o `clip` descarta. É por isso que a dispersão é lida junto com
`deblur_structural_ssim`, que não satura.

**Consequência para `focus_agreement`.** Ele continua sendo gravado e continua sendo o
gatilho da §2, mas o que ele mede aqui **não** é o que mede na rota C: lá é concordância
entre saliência e física; aqui é concordância entre saliência e **a crença da DeblurNet
sobre onde havia borrão**. Continua informativo — as duas discordam quando alguma das duas
erra — mas não é evidência independente.

---

## 5. As outras decisões, curtas

| # | decisão | razão, com a linha |
|---|---|---|
| 5.1 | **Nenhum renderizador.** `provenance.renderer = None` | o alvo é a foto real: paper.txt:271,283,321; o renderizador [43] aparece só em (a) (paper.txt:292) e no sweep de (c) (paper.txt:369-370). O antigo gravava `is_final_label_renderer: True` num dict literal — defeito B14 |
| 5.2 | **`k_effective_factor` não é aplicado** | os 0,9873 medidos (`CONTRATO.md:114`) descrevem o BokehMe, que esta rota não usa. `[A]` A12, recomendação da auditoria: não aplicar, e declarar |
| 5.3 | **`is_k_censored = false` sempre**; K fora da faixa **rejeita** | não há varredura, logo não existe "K no teto". O `--k-max 300` da rota C virou **47,0%** de amostras censuradas no teto exato (`ACHADOS.md:19`) sem que nada denunciasse |
| 5.4 | **`k_analytic = None`** | na rota C ele é a Eq. 3 validando a Eq. 5. Aqui a Eq. 3 **é** o rótulo: repetir o número neste campo o faria passar por validador independente sem ser um |
| 5.5 | **O validador é a distância de foco da EXIF** | independente por construção: não passa pelo Depth Pro nem pelo BiRefNet. O paper proíbe a EXIF como **rótulo** (paper.txt:344-346), não como validador. Âncora: mediana 20,1 em 318 amostras (`CONTRATO.md:77`) |
| 5.6 | **Máscara do rótulo sai da AIF**; máscara da bokeh é medida e descartada | `[A]` A2: o *"We then"* de paper.txt:293-294 vem depois do passo da DeblurNet, o que **sugere** AIF. Como A2 é risco **médio** — numa bokeh, a região nítida *é* a em foco —, `mask_iou_aif_bokeh` é calculada em toda amostra. Se a IoU do piloto for baixa, A2 merece revisão, e a revisão é trocar duas linhas |
| 5.7 | **Retenção medida na resolução da imagem, sempre. Sem knob** | a rota C tem `focus_retention_long_side` porque lá a retenção produz o **rótulo**; aqui é diagnóstico, e um knob que muda a grade de um diagnóstico só acrescenta um eixo de incomparabilidade entre metades do release. A grade vai para o metadado de qualquer forma |
| 5.8 | **Crop factor 1,0 MARCA, nunca rejeita** | `ACHADOS.md:169-173`: *"a tabela `make/model → sensor_width_mm` serve para **auditar** esses 30%, não para preencher lacuna — não há lacuna"*. `exif_crop_factor_suspect` **não recebe parâmetro de limiar**, então ninguém pode transformar a marcação em descarte de 30,33% do dataset por um valor que ninguém mediu |
| 5.9 | **`make`/`model` na proveniência de TODA amostra**, não só nas suspeitas | sem o denominador, "quais modelos ecoam a focal no campo de 35 mm?" não tem resposta |
| 5.10 | **Filtro `pseudo_aif` mantido, e exposto em `--include-pseudo-aif`** | `[A]` A3 — coluna do dataset, não do paper; herdado de `route_b.py:83-89`. Mantido para a contagem ser comparável com as 13.800 `[M]`, e exposto para a decisão ficar visível em vez de embutida |
| 5.11 | **A AIF é gravada; a bokeh é referência** | `dataio/sample.py:7-11`. A AIF é produto desta rota; a bokeh já existe em `atfortes/BokehDiffusion`, e regravá-la duplicaria bytes. `refs.aif_ref = None`, porque apontar para a origem afirmaria que ela já existia lá |
| 5.12 | **Dois ledgers de bytes** | `source_images.jsonl` com o sha256 dos bytes da bokeh (referência) e `generated_images.jsonl` com o sha256 do **JPEG em disco** da AIF (produto). Hashear o array em memória provaria uma coisa e o release conteria outra: JPEG q95 é lossy |

---

## 6. Gates da rota B — o que entra, o que não, e o que é novo

Dos dez gates de `qc/gates.py`, **oito** se aplicam. Dois não, e a ausência é decisão:

* `aif_aperture_is_narrow` — a AIF da rota B é **gerada**, não fotografada: ela não tem
  f-stop. Registrá-la como `applicable=False` poria a nota daquele gate (*"a origem correta
  é `train/in/<id>_f22.JPG`"*, que descreve a RealBokeh) no metadado de 13 mil amostras.
* `calibration_ssim_is_reliable` — não há Eq. 5 nesta rota (`CONTRATO.md:160`).

E `focus_mask_is_sharpest` é aqui **mais forte** que na rota C: ele exige a imagem com
bokeh, e a rota B **tem** a bokeh real — é ela o alvo.

Três gates novos, definidos em `src/routes/route_b.py` (e não em `qc/gates.py`, que está
sob edição de outro agente — ver §7):

| gate | o que mede | defeito que fecha |
|---|---|---|
| `deblur_structural_ssim_{min,max}` | SSIM(AIF, bokeh), com **piso e teto** | B13/§4.3-1: o piso pega a AIF LAVADA (estrutura perdida, variância de Laplaciano possivelmente ALTA); o teto pega a AIF IDÊNTICA à entrada. Um eixo só não separa os dois |
| `pair_registration_{shift_px,response}` | correlação de fase AIF↔bokeh | B17 e `[A]` A13. `ProcessingPlan` mede o que o **redimensionamento** faz (zero por construção com `NO_CROP_MULTIPLE_OF_16` e `long_side=0`); só a correlação de fase mede o que o **modelo** faz. A resposta viaja junto porque um `argmax` sobre ruído devolve uma coordenada com toda a cara de medida |
| `exif_crop_factor_unity` | crop factor exatamente 1,0 | B15. Marcação sem limiar — ver 5.8 |

Todos os limiares `None`. `k_value_{min,max}` também é novo, e mapeia para o slug
**registrado** `k_out_of_configured_range`.

---

## 7. O que precisa mudar em módulos que esta tarefa não podia editar

Nada abaixo foi aplicado. Cada item traz o defeito, o arquivo e o conserto.

### 7.1 `src/control/contract.py` — três slugs novos (P1)

`GATE_REJECTION_REASONS` (:94-111) é `frozenset` fechado, e três gates da rota B não têm
slug próprio. Enquanto isso eles usam **slugs provisórios**, e o nome do gate preserva a
distinção no `quality` de cada amostra — mas o histograma agrega errado:

| gate | slug provisório em uso | slug que ele pede |
|---|---|---|
| `deblur_structural_ssim_min` | `gate_aif_sharpness` | `gate_deblur_aif_unfaithful` |
| `deblur_structural_ssim_max` | `gate_bokeh_not_blurrier` | `gate_deblur_aif_is_input` |
| `pair_registration_shift_px` / `_response` | `gate_pair_shape_mismatch` | `gate_pair_registration_shift` |

E o adaptador do ITW precisa de um quarto: o filtro `pseudo_aif` usa hoje
`source_metadata_field_invalid`, e o certo é `source_pseudo_aif_excluded` — a linha não tem
metadado inválido, ela é de outra natureza.

Enquanto os quatro não existirem, o histograma do piloto vai somar causas distintas sob o
mesmo slug. `tests/test_route_b.py::test_todo_slug_do_mapa_de_gates_esta_no_conjunto_FECHADO`
garante que nenhum slug inventado escapa; a tabela `GATE_TO_REASON` marca cada provisório
com o comentário `PROVISÓRIO`.

**Higiene, no mesmo arquivo** (item 16 do §6 da auditoria): o comentário de
`pixel_ratio` (:342-343) diz *"32 de 100 amostras conferidas da rota B são retrato"*;
`ACHADOS.md:63` diz *"900/900 linhas, 32 delas em retrato"*, isto é **3,6%**. Um dos dois
está errado, e o número muda o peso que a frase carrega numa revisão.

### 7.2 `src/dataio/sample.py` — `k_diagnostics` e `deblurnet` obrigatórios (P1)

1. **`ControlLabel` não tem onde guardar o diagnóstico da Eq. 3.** `k_from_exif` devolve
   `sensor_width_mm`, `crop_factor`, `pixel_ratio_px_per_mm`, `longest_edge_px` e os três
   termos da EXIF (`contract.py:420-431`), e sem eles K é um número sem como auditar de
   qual sensor veio. **Conserto**: `k_diagnostics: Optional[dict]`, **exigido** quando
   `k_source == KSource.EQ3_EXIF` — simétrico ao `calibration_ssim` exigido para
   `EQ5_SSIM_SWEEP` (:411-412). Enquanto não existir, a rota grava em
   `provenance.extra["k_eq3_diagnostics"]`, que chega ao JSON mas **não é validado**.
2. **`SampleProvenance.deblurnet` é `Optional` e não entra em `_REQUIRED_PROVENANCE`**
   (:361-362). Para a rota B, `deblurnet` sem `deblur_variant` e sem `deblur_lora_sha256`
   tem que ser **erro de gravação**: são os dois campos que provam depois se a AIF de uma
   amostra saiu lavada.
3. **`FocusRegionRecord.retention_in_region` significa coisas diferentes por rota.** Na C é
   retenção contra uma AIF real; na B, contra uma AIF gerada (§4). O campo é o mesmo e o
   schema não distingue. **Conserto sugerido**: um campo
   `retention_reference: Literal["real_aif", "generated_aif"]` no registro, ou o
   `route` já disponível no metadado sendo o discriminante documentado no schema. Hoje a
   distinção vive só em `provenance.extra.deblur_diagnostics.retention_semantics` — texto,
   não schema.

### 7.3 `src/dataio/writer.py` — `deblur_variant` na linha do manifesto (P1)

`writer.py:139-174` carrega justamente os campos que denunciariam fallback: `split`,
`mask_source`, `depth_backend`, `control_version`, `max_coc`, `focus_source`.
`deblur_variant` pertence a essa lista — é o campo que, numa varredura de uma linha,
denunciaria um release meio nosso, meio oficial. Sem ele, a checagem de uniformidade que
`scripts/run_route_b.py::_confere_variante_uniforme` faz tem que ler o
`generated_images.jsonl` da rota, que é um arquivo que só a rota B escreve.

### 7.4 `scripts/publish_release.py` — ramificar por rota (P1)

Três coisas, todas em `_valida`:

1. **O ledger de bytes é conferido só para a rota C** (:130-137, `rotas_c`). Para a rota B o
   par é `source_images.jsonl` (bokeh, referência) **mais** `generated_images.jsonl` (AIF,
   produto). Hoje uma rota B sem ledger nenhum passa sem aviso.
2. **`sem_validador += meta.get("k_analytic") is None`** (:117) contaria 100% da rota B como
   "sem validador", o que é uma mensagem **falsa**: na rota B o validador é
   `provenance.extra.k_validator_exif_focus_distance` (§5.5). O contador precisa ramificar
   por `k_source`.
3. **Checagem de uniformidade de variante.** `:112-119` já reprova `control_version`
   divergente no mesmo release, com a frase certa (*"Foi assim que…"*). Precisa da **mesma
   checagem** para `provenance.deblurnet.deblur_variant` **e** `deblur_lora_sha256`. Sem
   ela, um `rsync` de duas metades produz um release misto e nada denuncia.

### 7.5 `scripts/calibrate_thresholds.py` — os gates novos

Escrito para a rota C. Os gates novos da rota B (`deblur_structural_ssim_*`,
`pair_registration_*`, `exif_crop_factor_unity`, `k_value_*`) precisam entrar na lista de
grandezas cuja distribuição ele resume. O `route_b_pilot.slurm` chama o script com
`|| echo` para que uma incompatibilidade não derrube o job — os números continuam no resumo
do run e em `meta/*.json`, campo `quality`.

### 7.6 `src/sources/bokehdiffusion.py` — o lugar definitivo do adaptador

O adaptador do ITW (`ItwRow`, `ItwImageLoader`, `_enumera_itw`) está em
`scripts/run_route_b.py` porque esta tarefa não podia criar arquivo em `src/sources/`. O
lugar certo é `src/sources/bokehdiffusion.py`, no molde de `sources/realbokeh.py`, com um
índice de leitura sequencial no molde de `MirrorIndex`/`MirrorImageLoader`
(`sources/mirror_images.py:69-273`, hoje amarrado a `file_name_base`, `image_focus` e
`image_blur`). **A rota não muda quando isso acontecer**: ela recebe o protocolo
`BokehSource`, não a classe.

### 7.7 `src/qc/gates.py` — se os gates novos passarem a valer para outra rota

Os três gates novos vivem em `routes/route_b.py` por duas razões: `qc/gates.py` está sob
edição de outro agente, e os três só fazem sentido quando a AIF é **produto de modelo**. Se
a rota A (que também gera imagem) passar a precisar deles, o lugar é `qc/gates.py` e a
seção de `route_b.py` some.

### 7.8 `third_party/` — o checkout do Genfocus

`third_party/README.md` documenta hoje só o BokehMe (:3-8). A rota B precisa de
`Genfocus/pipeline/flux.py` (`Condition`, `generate`, `seed_everything`), clonado com
commit congelado e hash — a mesma disciplina do BokehMe (:46-48). O código vendorizado
equivalente existe em `bokehnet-preprocessing/src/vendor/genfocus_flux.py`, e
`DeblurNetRuntime.load()` **recusa** cair nele por escrito: aquele repositório é referência
histórica e é onde o defeito B1 vive.

---

## 8. Itens `[A]` da rota B — estado depois desta etapa

Numeração de `ROTA_B_AUDITORIA.md` §7, com o que mudou.

| # | item | estado |
|---|---|---|
| A1 | ITW [19] = `atfortes/BokehDiffusion` | segue `[I]`. Gravado na proveniência como `source_dataset_evidence` |
| A2 | a máscara sai da AIF | segue `[A]`, risco **médio**. Agora **medido por amostra**: `mask_iou_aif_bokeh` compara as duas máscaras em toda amostra, e o piloto decide |
| A3 | filtrar `pseudo_aif` | segue `[A]`. Exposto em `--include-pseudo-aif` e gravado em `source_filter_evidence` |
| A4 | qual subconjunto são os "13K filtered and verified" | segue `[A]`, não endereçado |
| A5 | qual `D_focus` alimenta a Eq. 3 | **decidido** (§1), com as duas leituras gravadas e a razão medida por amostra |
| A6 | as imagens do ITW estão na resolução de captura | segue `[A]`, risco **alto**, **não medido**. O piloto grava `image_h`/`image_w` de toda amostra, o que permite cruzar com a distribuição de `pixel_ratio` da Fig. 16 do paper (22,2 a 277,3, mediana 42,6 — `CONTRATO.md:82`). Se a nossa distribuição divergir da publicada, é sinal de recorte |
| A7 | largura do sensor via crop factor | segue `[A]`, risco **alto**. `exif_crop_factor_unity` marca os 30,33%, e `make`/`model` vão na proveniência de toda amostra — a auditoria por câmera passa a ser possível |
| A8 | rejeitar quando a EXIF falta | segue `[A]` (lado conservador), com teste que dispara cada um dos três slugs |
| A9 | resolução em que a DeblurNet roda (`long_side`) | **decidido**: `--deblur-long-side 0`, default. É o do oficial (`Inference_deblurNet.py:57`) e o do antigo. `long_side > 0` com a política que recorta exige `acknowledge_fov_crop=True` no runtime |
| A10 | faixa de K plausível | segue `[A]`. `--min-k-value`/`--max-k-value` default `None`; o piloto mede contra 16,6 / 20,1 / 15,0 |
| A11 | limiares dos gates | segue `[A]` por definição. **Todos `None`** |
| A12 | aplicar `k_effective_factor` à rota B | **decidido: não aplicar** (5.2) |
| A13 | quanto a DeblurNet desloca a AIF | segue `[A]`, mas **agora é medido**: `pair_registration_shift_px` em toda amostra. O piloto é a primeira medição |
| A14 | qual DeblurNet os autores usaram | segue `[A]`, não reprodutível de qualquer forma |
| A15 | a rota B usa a bokeh na resolução nativa | segue `[A]`. Herda a decisão de crop 512 no treino, com `k_at_resolution` como única conversão |

**Novo `[A]` desta etapa**: a estrutura interna de `flickr_exif`. `ACHADOS.md` registra que
o campo *"tem make e model"*, sem publicar o schema, e o nome do campo de distância de foco
varia por fabricante. O adaptador aceita `dict` e JSON em texto e procura uma lista de
chaves; nenhum valor daí entra em K — `make`/`model` são auditoria e a distância de foco é
validador.

---

## 9. O que esta etapa NÃO mediu

Registrado para ninguém confundir com verificação.

* **Nada rodou em GPU.** Sem cluster, sem `sbatch`, sem `ssh`. Os 104 testes de
  `tests/test_route_b.py` usam dublês: a DeblurNet, o Depth Pro e o BiRefNet são classes de
  teste, e a bokeh é renderizada por uma composição de 7 camadas em numpy.
* **O dataset não foi baixado.** As colunas usadas (`image`, `focal_length`, `f_number`,
  `focal_length_35`, `pseudo_aif`, `flickr_photo_id`, `flickr_exif`) vêm de
  `ACHADOS.md:145-175` `[M]` e de `bokehnet-preprocessing/src/pipelines/route_b.py:83-89`.
  O adaptador **falha alto** se alguma faltar, em vez de assumir default.
* **A resolução do ITW é desconhecida** (`[A]` A6), então o `--image-megapixels 6.0` do
  orçamento de disco é um chute declarado — ele afeta o **aviso** de disco, nunca o rótulo.
* **A unidade do campo de distância de foco da EXIF** é assumida metro. Ela só alimenta o
  validador; um erro de unidade ali não toca em `k_value`.
* **A taxa de `focus_mask_empty` no ITW** é o número que decide a §2, e ele só existe depois
  do piloto.
