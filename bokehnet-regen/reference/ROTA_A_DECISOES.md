# Rota A — decisões declaradas

Data: 2026-09-13. Acompanha `src/routes/route_a.py`, `src/sources/genphoto_ebb.py`,
`src/sources/k_distribution.py`, `scripts/run_route_a.py`,
`scripts/build_k_distribution.py`, `tests/test_route_a.py` (96 testes) e
`slurm/route_a_{pilot,full}.slurm`.

Autoridade, na ordem do projeto: (1) o paper (arXiv:2512.16923v3), onde ele fala;
(2) `third_party/Genfocus/Inference_*.py`, onde o paper cala; (3) decisão nossa,
**declarada como desvio**, onde os dois calam.

Etiquetas: `[M]` medido · `[I]` inferido de algo medido · `[A]` assumido, não verificado.

O que `reference/ROTA_A_AUDITORIA.md` definiu está implementado; o que ela deixou aberto
está decidido aqui. **Nenhum limiar foi congelado**: todos os defaults são `None`.

---

## 0. O que a rota A é, em uma linha

Imagem AIF real de `[80]` + EBB! → Depth Pro dá `D` → **sorteia** `D_focus` e `K` → Eq. 2
dá `D_def` → **BokehMe `[43]` renderiza o alvo**. Uma equação, nenhuma máscara, nenhum
segmentador, nenhum SSIM (paper.txt:328-334).

É a **única** rota em que o alvo é renderizado, e a única em que
`provenance.renderer.is_final_label_renderer` é `True` porque o renderizador produziu o
rótulo. O paper autoriza isso literalmente, duas vezes no corpo: *"use a **simulator
[43]** to render the corresponding target bokeh image"* (paper.txt:331) e *"feed it into a
**bokeh renderer [43]**"* (paper.txt:292).

E carrega a ressalva que o próprio paper escreve: a rota A *"is constrained by **renderer
bias** and may introduce unrealistic artifacts"* (paper.txt:333-334). É pré-treino de
**geometria de CoC**, não de aparência — 40K passos em sintético contra 60K em real
(paper.txt:515-516).

---

## 1. Decisão 1 — a distribuição de K vive num JSON, na grandeza `k_per_long_side`

### O problema

A rota A não tem equação para K. O paper diz *"we **randomly sample** a focus plane
D_focus and a target bokeh level K"* (paper.txt:329-330) e **cala sobre a distribuição**.
Amostrar da distribuição empírica de B e C (`k_source = "sampled_from_bc"`) é decisão
nossa, já registrada em `CONTRATO.md:163-166`, e continua `[A]`.

Mas transportar K de B/C para A **cru** é erro de unidade, não de gosto. `K = k_eq3/1000` e
`k_eq3 ∝ pixel_ratio = max(H,W)/sensor_mm` (`CONTRATO.md:22-23`), logo **K escala
linearmente com a resolução**. O `pixel_ratio` medido na rota B vai de 22,2 a 277,3, com
mediana 42,6 — uma faixa de **12x** (`ACHADOS.md:36` `[M]`). É o defeito A5, e é a terceira
aparição do mesmo erro de convenção de escala neste projeto.

### A decisão

A distribuição é armazenada e sorteada na grandeza **livre de resolução**

    k_per_long_side = k_value / max(image_h, image_w)

e a rota A remultiplica pelo `max(H,W)` **da imagem dela**. Um JSON versionado
(`schema: bokehnet_k_distribution_v1`) fica entre as rotas: `scripts/build_k_distribution.py`
o produz dos `manifest.jsonl` de B e C, e o sha256 dele entra na proveniência de cada
amostra da rota A.

### Por quê um arquivo no meio, e não leitura direta dos manifestos

Porque a distribuição é **entrada auditável** do run, não efeito colateral dele. Lendo
direto, a distribuição usada num release ficaria implícita: ninguém conseguiria dizer
depois de quais linhas ela saiu, e regerar a rota A depois da rota B regerada produziria
outra distribuição sem que nada denunciasse.

### O bloqueador F3 da auditoria **não existe mais**

A auditoria registrava que a linha do manifesto não carrega `image_h`/`image_w` e que o
amostrador precisaria abrir 70 mil `meta/`. Conferido: o writer **já grava os dois**
(`dataio/writer.py:150`), com a justificativa escrita ao lado (*"K é um número EM PIXEL;
sem a resolução ao lado dele, a linha do manifesto não diz o que o K significa"*). Nada a
mudar.

### O formato

```json
{
  "schema": "bokehnet_k_distribution_v1",
  "quantity": "k_per_long_side_px",
  "quantity_definition": "k_value / max(image_h, image_w) — ...",
  "control_version": "metric_disparity_official_v1",
  "created_utc": "2026-09-13T16:21:22+00:00",
  "n": 1372,
  "distinct_values": 101,
  "widen_fraction_recommended": 0.25,
  "stats": {"min":…, "p01":…, "p50":…, "p99":…, "max":…,
            "widened_support_min":…, "widened_support_max":…},
  "quantiles": {"q": [0.00, 0.01, …, 1.00], "value": [ … 101 valores … ]},
  "per_route": {"b": {"n":…, "distinct":…, "p01":…, "p50":…, "p99":…,
                      "censored_share":…}, "c": {…}},
  "sources": [{"release_dir":…, "lines_total":…, "lines_used":…,
               "excluded": {"not_valid_for_control":…, "is_k_censored":…,
                            "missing_image_hw":…, "k_value_invalid":…},
               "routes": {…}}],
  "degenerate_reason": null
}
```

Três coisas que o formato obriga:

* **a função quantil, não a amostra.** 70 mil valores de K não precisam viajar num arquivo
  de configuração, e a função quantil é tudo que o sorteio consome. 101 nós = passo de 1%,
  que resolve p01 e p99 sem interpolar entre extremos distantes.
* **os contadores de exclusão.** Sem eles, uma distribuição montada de 3% das linhas
  pareceria idêntica a uma montada de 100%.
* **`control_version`.** Manifestos de convenções diferentes na mesma distribuição é o modo
  de falha do kfix (`ACHADOS.md:192`); o script recusa a mistura.

### O que é excluído, e por quê

| exclusão | razão |
|---|---|
| `is_valid_for_control == False` | a amostra não serve de rótulo, o K dela não descreve o fenômeno |
| `is_k_censored == True` | valor de **teto** é borda de configuração, não medida. 47,0% da rota C publicada está no teto exato de 300 (`ACHADOS.md:19` `[M]`) — incluir criaria uma moda artificial no limite |
| `image_h`/`image_w` ausentes | sem resolução não há como tirar a resolução do K. Dividir por um lado longo assumido seria **inventar** a escala |

---

## 2. Decisão 2 — o alargamento de 25% é ancorado na MEDIANA

### O problema

A rota B vai ser **regerada** com a nossa DeblurNet (`PLANO_EXECUCAO.md`, passo 5), e a
hipótese declarada do plano é que a distribuição de K não muda muito entre as duas
variantes — *"se não vale, a rota A precisa ser regerada e o pré-treino refeito"*. Sortear
exatamente `[p01, p99]` observado deixaria a cobertura no limite: se o K se mover um pouco,
a faixa nova cai fora do que o pré-treino viu.

### A decisão

    v' = p50 + (1 + f)·(v − p50)          f = 0,25 `[A]`

Cada lado da faixa observada se estende 25% em torno da mediana:
`p01 ↦ p50 − 1,25·(p50 − p01)` e `p99 ↦ p50 + 1,25·(p99 − p50)`. **A mediana fica parada.**

### A alternativa, recusada por medição

O mapa afim em torno do **centro** de `[p01, p99]` (`v' = c + (1+2f)·(v − c)`) produz a
mesma faixa alargada, e é a leitura mais literal de "25% para cada lado". Medido numa
lognormal com σ = 0,5 e 5.000 pontos — que é a forma que uma razão de grandezas ópticas
positivas tem:

| grandeza | observado | afim pelo centro | ancorado na mediana |
|---|---|---|---|
| mediana | 0,019725 | **0,012286 (−37,7%)** | 0,019725 (0,0%) |
| p25 | 0,014201 | **0,004000 (−71,8%)** | 0,012820 (−9,7%) |

O mapa pelo centro **desloca a distribuição inteira para baixo** quando ela é assimétrica à
direita. A rota A passaria a ter um K típico que não é o K típico do real — contra a razão
de existir do `sampled_from_bc`, que *"existe para manter o sintético na mesma escala
física do real"* (`CONTRATO.md:163-166`).

### Positividade

K tem que ser > 0 (`control/contract.py:464-465`). Quando o alargamento levaria a zero ou
abaixo, o valor é preso num piso **derivado da própria distribuição** —
`min_observado · (1 − f)`, estritamente positivo e abaixo de tudo que foi observado — e o
evento é **contado** por amostra (`k_clamped_to_positive_floor`) e no resumo do run. Nem
clamp silencioso, nem rejeição em massa da cauda baixa: truncar a cauda sem contar seria
uma seleção invisível.

### O valor observado viaja ao lado do alargado

Cada amostra grava `k_per_long_side` **e** `k_per_long_side_observed`. Sem o segundo não dá
para responder *"este K existia nos dados ou saiu do alargamento?"* — que é exatamente a
pergunta que o alargamento cria.

---

## 3. Decisão 3 — `D_focus` é um QUANTIL da disparidade da própria imagem

### O problema

`D_focus` sorteado fora da faixa de disparidade da imagem degenera o mapa: tudo saturado ou
tudo zero. E mesmo dentro da faixa, um plano num intervalo **sem conteúdo** produz um rótulo
que não descreve cena nenhuma.

### A decisão

    focus_disparity = quantile( pool,  q_low + u·(q_high − q_low) )

com `pool` = disparidade dos pixels cuja profundidade é fisicamente plausível, e
`q_low, q_high = 0,05, 0,95` `[A]`.

Um quantil dos dados **é** um valor entre o mínimo e o máximo observados — dentro da faixa
por definição, sem clamp — e existe **massa de cena** naquele plano.

### O que foi medido nos extremos

Cena sintética de **retrato** (sujeito a 1,2 m ocupando ~12% do quadro, fundo de 8 a 30 m),
41 variantes, `MAX_COC = 100`:

| regra | K | banda em foco VAZIA | área mediana da banda | saturação mediana |
|---|---|---|---|---|
| uniforme na faixa de disparidade | 5 | **56,1%** | 0,0000 | 0,0000 |
| uniforme na faixa de disparidade | 15 | **73,2%** | 0,0000 | 0,0000 |
| uniforme na faixa de disparidade | 50 | **80,5%** | 0,0000 | 0,0000 |
| uniforme na faixa de disparidade | 300 | 80,5% | 0,0000 | **0,8892** |
| **quantil (rota A)** | 5 | 0,0% | 0,8892 | 0,0000 |
| **quantil (rota A)** | 15 | 0,0% | 0,6154 | 0,0000 |
| **quantil (rota A)** | 50 | 0,0% | 0,1937 | 0,0000 |
| **quantil (rota A)** | 300 | 0,0% | 0,0322 | 0,1108 |

Banda vazia = nenhum pixel com `|CoC| ≤ 0,5 px` = **a variante morre com
`focus_mask_empty`**. Uniforme na faixa perde de 56% a 80% das variantes num retrato,
porque o plano cai no vazio entre o sujeito e o fundo. Por quantil, nenhuma.

Cena sintética **natural** (90% do conteúdo em 1–50 m, 10% de céu a 10.000 m), K = 300:
saturação mediana **0,9320** com uniforme contra **0,0326** por quantil.

### Uma correção medida à auditoria: A6 item 1 está parcialmente errado

A auditoria diz que sortear no quantil da **profundidade métrica** é "o espaço errado".
Medido: **o quantil é invariante a transformação monótona.**

| q | `1/quantil_z(1−q)` | `quantil_disp(q)` | erro relativo |
|---|---|---|---|
| 0,05 | 0,00010000 | 0,00010000 | 0,0 |
| 0,25 | 0,02388129 | 0,02388129 | 1,0e-11 |
| 0,50 | 0,03542072 | 0,03542072 | 5,8e-12 |
| 0,95 | 0,27461424 | 0,27461424 | 1,8e-09 |

As duas regras produzem **os mesmos planos de foco** (mediana 28,23 m nas duas). O erro
residual vem só da interpolação linear entre vizinhos — interpolar `1/z` não é o inverso de
interpolar `z` — e vai de 1e-12 a 1e-07 conforme a densidade.

O que **continua valendo** de A6 é a outra metade, e é a que importa: a **unidade
primária**. O campo gravado é `focus_disparity`; `focus_depth_m` é leitura humana. O defeito
antigo era produzir `focus_depth_m` e reconstruir a disparidade **duas vezes** a jusante, em
duas linhas diferentes (`renderer.py:71` e `route_a.py:100`).

O que de fato muda o resultado é **quantil contra uniforme**, não `z` contra `1/z`.

### A população do sorteio: planos FISICAMENTE PLAUSÍVEIS

Um pixel no teto de 10.000 m do Depth Pro não é candidato a plano de foco — é sentinela, e
25,7% das amostras medidas têm `z_max == 10.000` (`ACHADOS.md:54` `[M]`). A população é
restrita a `[FOCUS_DEPTH_MIN_M, FOCUS_DEPTH_MAX_M]` do contrato, e a fração que sobrou vai
gravada em `focus_sampling_pool_fraction`.

Medido, cena natural com 10% de céu, 41 variantes:

| população | variantes perdidas em `focus_depth_implausible` |
|---|---|
| crua, `q = [0, 1]` | **9,8%** |
| crua, `q = [0,05, 0,95]` | **4,9%** |
| restrita aos planos plausíveis | **0,0%** |

A perda é proporcional à **área** de céu, e nada no rótulo a denunciaria. Isto não é clamp
nem fallback — nenhum valor é substituído por constante; é a definição da população. O gate
`focus_depth_plausible` continua ligado como arame de tropeço: se ele passar a disparar, a
regra de amostragem mudou.

Imagem **inteira** no teto é rejeitada com `focus_depth_implausible`, e custa as N
variantes — contadas em `variants_lost_to_image`, separado das rejeições por variante.

### Uma consequência declarada: platô colapsa planos de foco

Numa cena com um platô grande (um sujeito plano ocupando 25% do quadro), a função quantil é
constante naquela faixa e **estratos diferentes caem no mesmo plano de foco**. Não é
defeito: é o sorteio seguindo a massa da cena, que é o que impede o plano de cair no vazio.
O K continua diferente em todas as variantes, então nenhuma amostra é duplicata de outra.
Travado em `test_um_plato_na_cena_colapsa_planos_de_foco_e_isso_e_declarado`.

---

## 4. Decisão 4 — 41 variantes por imagem, em hipercubo latino

### Quantas — `[A]`, não número do paper

O paper publica os dois lados da conta e não o resultado: ~70K pares sintéticos
(paper.txt:527-528) de um pool de ~1,7K AIFs nítidas (paper.txt:998). `70.000/1.700 = 41,2`
`[I]`. O `41` do código é `[A]`, escolhido para fechar as duas âncoras, e é flag
(`--samples-per-image`).

O histórico bate: o dataset antigo tinha ~68.000 amostras de ~1.700 AIFs (`ACHADOS.md:23`
`[M]`). **O dataset publicado já tinha a multiplicidade certa; o pipeline v2 é que a
perdeu** — uma variante por imagem (`route_a.py:138`), que é o defeito A2.

E A2 é pior que "40x menos dado": com uma variante por imagem, cada conteúdo aparece com
**um** K e **um** plano de foco, e o sinal que o pré-treino existe para ensinar — *"helps
the network modulate the circle of confusion according to D_def"* (paper.txt:332-333) — não
existe no dado. K vira confundido com conteúdo.

### Como — hipercubo latino 2D

41 sorteios i.i.d. por imagem deixam imagens inteiras sem K alto **por azar**: a chance de
nenhum dos 41 cair no decil superior é `0,9^41 ≈ 1,3%`, uma imagem em cada 75, ou ~23
imagens do pool de 1,7K sem nenhuma variante de borrão forte. O que ensina controle é
**cobertura por imagem**.

No hipercubo latino, `[0,1)` é dividido em N estratos e cada variante cai num estrato
**diferente** de K e num estrato **diferente** de plano de foco, com o pareamento sorteado.
Cobertura perfeita nas duas margens.

Custo declarado: as N variantes de uma imagem deixam de ser independentes entre si (é o
ponto), e **trocar `samples_per_image` muda todos os sorteios daquela imagem**, porque os
estratos mudam de largura. Por isso `samples_per_image` vai no `run_config.json` **e** na
proveniência de cada amostra.

### Determinismo

`variant_seed = sha256(f"{seed}:{scene_id}")[:8]`. Nunca `hash()`, que é randomizado por
processo (PYTHONHASHSEED) — a mesma razão e a mesma construção de `dataio/split.py:29-35`.
`draw_for_variant(scene_id, i, n, seed)` refaz o sorteio de **uma** amostra a partir do
`sample_id`, sem processar a imagem: é o que "reprodutível" significa em auditoria.

---

## 5. Decisão 5 — `scene_id` é a imagem, `sample_id` é `<scene>_v<NN>`

O antigo era `a_{...}_{index:06d}` com **o índice na lista** do manifesto
(`route_a.py:139`): reordenar o manifesto renomeava todas as amostras. E **não existia
`scene_id`** (defeito A9). Com uma variante por imagem isso passava; com 41 é vazamento
garantido — a mesma imagem cairia dos dois lados de um split por amostra, e a validação
mediria memorização.

* `scene_id` = `<slug da fonte>_<12 primeiros dígitos do sha256 do arquivo>`. Estável sob
  reordenação do diretório, e não colide entre fontes.
* `sample_id` = `<scene_id>_v<NN>`, com a largura crescendo com N.
* o split é por **cena**, materializado no release (`CLAUDE.md:50`), e as N variantes ficam
  sempre do mesmo lado — travado em
  `test_as_variantes_de_uma_imagem_ficam_do_MESMO_lado_do_split`.

---

## 6. Decisão 6 — a fonte é `[80]` + EBB!, e o corte é POR FONTE

### A fonte

`paper.txt:996`: *"We draw candidate images from **[80]** and the **EBB [27]**
collections"*. `paper.txt:527-528`: *"∼70K synthetic pairs derived from **[27, 80]**"*.
`[80]` é Generative Photography (paper.txt:941-943); `[27]` é o EBB!
(paper.txt:814-815).

**DiffCamera é `[69]` e nunca aparece como fonte de dado**: paper.txt:115 (baseline da
Tab. 1), 225, 385, 498, 587, 971, 1019, 1050 (*"Additional Comparison with DiffCamera"*),
bibliografia em 914. Zero ocorrências em §3.2, na Fig. 3 ou no B.2. O pipeline antigo tinha
`PAPER_A_SOURCES = {"DiffCamera", "EBB!"}` e docstrings afirmando fidelidade ao paper — a
divergência nº 1 da auditoria.

Fonte fora do conjunto é **erro de configuração** (`SystemExit`), não rejeição de amostra:
um dataset errado não polui o histograma, ele cancela o run.

Cuidado de leitura, repetido aqui porque custa caro: o `paper.txt` tem **duas numerações de
citação**. As legendas das Figs. 4 e 6 usam a antiga (`DiffCamera [67]`); o corpo e a
bibliografia usam a atual. Nas linhas que importam — 527-528 e 996 — é a atual.

### O corte

`paper.txt:997` diz *"we use Laplacian variance to filter out blurry examples"* e
`paper.txt:998` dá o resultado (~1,7K). **Não diz** se o ranking é por fonte.

O pipeline antigo ranqueava **global** (`build_route_a_manifest.py:45-46`). Isso é
demonstravelmente enviesado, e o nosso próprio gate já diz por quê
(`qc/gates.py:279-284`): a variância do Laplaciano **escala com resolução e compressão**,
então um ranking único entre EBB! e Generative Photography seleciona pela fonte de maior
resolução, não pela mais nítida.

Aqui: ranking e corte **dentro de cada fonte**, com quota declarada por fonte e sem
default — a divisão do pool de 1,7K não é publicada (`[A]` A4). O **corte efetivo** de cada
fonte (`cut_variance`) vai para o `run_config.json`, que é justamente o número que o
histórico não gravou.

E a variância vai com a grade em que foi medida (`sharpness_hw`), pela regra 3 do contrato.
Se uma fonte tiver mais de uma resolução de medição, o resumo avisa em voz alta.

### Dedup por conteúdo

Duas cópias do mesmo arquivo com nomes diferentes gerariam 2 × 41 variantes da mesma imagem
em **cenas diferentes**, e o split as separaria como se fossem cenas distintas. Dedup é por
sha256 do conteúdo, com slug `source_duplicate_sample`.

---

## 7. Decisão 7 — a máscara que a rota A grava, e a que ela não tem

A rota A **não tem máscara** (paper.txt:329-330). Mas o writer grava `mask/<id>.png` para
toda amostra e `validate_metadata` exige `mask_source` e os campos de região em foco
(`dataio/sample.py:352-373`). A alternativa honesta — não gravar campo de máscara — exige
mudança em `dataio/`, descrita na §9.

O que a rota grava é a **banda em foco derivada do próprio rótulo**:

    banda = |CoC| ≤ focus_band_coc_px          # 0,5 px `[A]`

Ela não afirma nada sobre a cena: é função determinística de `(D, D_focus, K)`, e a regra
completa vai em `provenance.extra.mask_rule` — tolerância, grade, `focus_disparity`,
`k_value`, seed, índice da variante. `test_a_mascara_no_disco_e_a_banda_da_regra_gravada`
reconstrói a máscara do disco a partir **só** do metadado.

Isso conserta o defeito A7, em que **um** array era gravado em **três** chaves com
semânticas diferentes (`foreground_mask`, `focus_mask_auto`, `focus_mask_final`) com a
única proveniência sendo a string `"depth_band_q12"` num dict livre.

`focus_band_coc_px = 0,5` **não é limiar de rejeição** — é a definição da banda, e 0,5 px de
raio de CoC é sub-pixel, isto é, indistinguível de nítido.

### O vocabulário: extensão, nunca reuso

`FocusSource` não tem valor para "sorteado" e `MaskSource.DEPTH_BAND` descreve **a banda no
quantil 0,12 da rota A antiga**, que é o defeito A7. Reusar qualquer um seria proveniência
que mente — `focus_source = "birefnet"` numa rota que nunca roda BiRefNet é o
`mask_source="automatic"` da cascata antiga com outra roupa.

A rota **falha alto** enquanto os dois valores novos não existirem
(`VocabularyExtensionRequired`, que **não** é `SampleRejected` e portanto não pode se
disfarçar de caso a calibrar no histograma). O patch está na §9.1.

---

## 8. Gates da rota A — o que entra, o que não, e o que é novo

| gate | origem | papel na rota A |
|---|---|---|
| `aif_laplacian_variance` | `qc/gates.py` | nitidez da AIF; comparável só dentro da fonte |
| `bokeh_over_aif_sharpness` | `qc/gates.py` | **muda de papel**: aqui pergunta *"o renderer fez alguma coisa?"* |
| `rendered_coc_p99_px` | **novo, local** | o rótulo pede borrão visível? Âncora: mediana 4,665 px na rota B (`ACHADOS.md:37` `[M]`) |
| `defocus_saturation_ratio` | **novo, local** | fração com `|CoC| ≥ MAX_COC = 100`. Na rota B foram 17,3% **com `max_coc = 10,5107`** (`ACHADOS.md:57`); com 100 é outro regime e **não foi medido** |
| `mask_area_ratio_min/max` | `qc/gates.py` | área da banda |
| `depth_useful_levels` | `qc/gates.py` | cena degenerada |
| `focus_depth_m_min/max` | `qc/gates.py` | arame de tropeço da regra de amostragem (§3) |

**Não entram**, e cada ausência é decisão:

* `calibration_ssim_is_reliable` — não há Eq. 5 (`CONTRATO.md:160`);
* `aif_aperture_is_narrow` — a fonte não publica f-stop por imagem, e a nota do gate
  descreve a RealBokeh: registrá-la poria uma frase falsa em 70 mil metadados;
* `mask_iou` / `mask_border_coverage` — não há duas máscaras, e a heurística de borda
  descreve segmentador de objeto saliente; a banda toca a borda legitimamente quando o
  plano sorteado é o fundo;
* `focus_mask_is_sharpest` — **tautológico**: a banda é, por definição, onde o CoC é
  sub-pixel. Gate que não pode reprovar é pior que gate nenhum (`REGISTRO.md:695-696`);
* `pair_shape_matches` — `process_variant` já rejeita shape divergente com
  `resolution_invalid` antes de medir, e com mensagem melhor. Redundância dentro da mesma
  função vira gate que não reprova.

**Não há gate de faixa de K**, e a ausência é decisão. Na rota B "K implausível" é uma
amostra a descartar porque o K vem da EXIF daquela foto. Aqui o K é **imposto** de um
suporte já validado uma vez, em voz alta, por `KDistribution.degenerate_reason()`: um K fora
do plausível é defeito **da distribuição**, e rejeitar amostra por isso esconderia um erro
de entrada atrás de um histograma de 70 mil linhas.

### O guarda da distribuição — pré-requisito duro, não preferência

`degenerate_reason()` recusa: menos de 32 valores distintos, `p99/p01 < 1,05`, rota
contribuinte com menos de 8 valores distintos, ou rota contribuinte com mais de 25% de
amostras censuradas. Todos `[A]`.

Ele existe por dois fatos medidos: a rota B publicada tem `k = 50,0` em **11.635/11.635**
(`ACHADOS.md:14` `[M]`) e a rota C tem **47,0%** no teto exato de 300 (`ACHADOS.md:19`
`[M]`). **Amostrar dessa "distribuição" hoje é amostrar de duas constantes.** A rota A não
pode rodar antes de B e C regeradas (`PLANO_EXECUCAO.md`), e `--allow-degenerate-k-distribution`
existe só para quem quiser furar isso por escrito.

---

## 9. O que precisa mudar em módulos que esta tarefa não podia editar

Nada abaixo foi aplicado. Cada item traz o patch e o que ele quebra se for feito errado.

### 9.1 `src/qc/focus_region.py` e `src/dataio/sample.py` — dois valores de vocabulário (P0, BLOQUEADOR)

Sem isto a rota A **não grava uma única amostra**, por construção.

```python
# qc/focus_region.py, em FocusSource:
    #: Rota A: o plano de foco foi SORTEADO (paper.txt:329-330). Não há máscara, não há
    #: BiRefNet e não há retenção — não existe região medida na cena.
    SAMPLED_PLANE = "sampled_plane"

# dataio/sample.py, em MaskSource:
    #: Rota A: a banda |CoC| <= tolerância, DERIVADA do rótulo sorteado. NÃO é
    #: DEPTH_BAND, que era a banda no quantil 0,12 da rota A antiga (defeito A7).
    SAMPLED_PLANE = "sampled_plane"

# dataio/sample.py, em FOCUS_SOURCE_TO_MASK_SOURCE:
    FocusSource.SAMPLED_PLANE: MaskSource.SAMPLED_PLANE,
```

A entrada no dicionário é obrigatória: `test_dataio.test_toda_fonte_de_foco_tem_fonte_de_mascara`
exige que ele seja total, e é por isso que ele existe.

### 9.2 `src/dataio/sample.py` — `mask_model_sha256` **ou** `mask_rule` (P0, BLOQUEADOR)

`_REQUIRED_PROVENANCE` exige `mask_model_sha256` não vazio (`:375-376`, `:421-423`). A rota
A não roda segmentador nenhum. É o item F6 da auditoria, com as duas saídas que ele lista, e
a segunda é melhor:

```python
_REQUIRED_PROVENANCE = ("pipeline_commit", "depth_model_sha256", "image_h", "image_w")

# em validate_metadata, depois da checagem de proveniência:
    if not (prov.get("mask_model_sha256") or (prov.get("extra") or {}).get("mask_rule")):
        raise ValueError(
            "proveniência sem modelo de máscara E sem `mask_rule`: a máscara gravada não "
            "tem origem declarada. Um dos dois é obrigatório — a regra sintética é o "
            "análogo do hash, é o que permite reconstruir a máscara.")
```

Não afrouxa a rota C: ela continua reprovando se o hash do BiRefNet faltar, porque ela não
grava `mask_rule`.

### 9.3 `src/dataio/sample.py` — `was_refined` está errado para fonte nova (P1)

```python
    @property
    def was_refined(self) -> bool:
        return FocusSource(self.source) in _REFINED_FOCUS_SOURCES   # {BIREFNET_REFINED,
                                                                    #  RETENTION_ONLY}
```
e o `esperado` de `_validate_focus_region` (`:448`) na mesma regra.

Hoje é `is not BIREFNET`, então **qualquer** fonte nova sai marcada como "refinada". A rota
A grava `focus_was_refined = True` sem ter refinado nada, e `focus_was_refined` é justamente
o campo por onde se monta o treino com e sem as amostras refinadas
(`dataio/writer.py:165`). O comportamento atual está **pinado** em
`test_focus_was_refined_sai_True_e_isso_e_o_defeito_DESCRITO`: quando o patch entrar, o teste
passa a esperar `False` e é por ele que alguém lembra.

### 9.4 `src/dataio/sample.py` — `FocusRegionRecord` assume duas fotografias (P1)

`agreement`, `precision`, `iou`, `retention_in_region`, `retention_hw` e
`retention_window_px` descrevem o refinamento do §3.2(c), que precisa de **par**. A rota A
tem uma foto. Hoje ela preenche `1,0 / 1,0 / 1,0 / NaN / grade da imagem / 1` e **declara**
em `focus_region_semantics` que os três primeiros são 1,0 **por construção, não por
medição** — mesmo raciocínio que a rota B registra para `precision`/`iou`
(`route_b.py:812-815`).

O conserto honesto: tornar os campos de retenção `Optional`, renomear `retention_hw` →
`region_hw`, e acrescentar `focus_region_semantics` como campo de primeiro nível em vez de
texto na proveniência. É mudança de schema e merece uma etapa própria — aplicá-la pela
metade deixaria o release com dois formatos.

### 9.5 `src/control/contract.py` — `validate_focus_disparity` pública (P1, item F2)

Hoje só existe `focus_disparity_from_mask` (`:286-323`), que **exige máscara**. Extrair o
corpo de `:316-321`:

```python
def validate_focus_disparity(focus_disparity, *, min_focus_depth_m=FOCUS_DEPTH_MIN_M,
                             max_focus_depth_m=FOCUS_DEPTH_MAX_M) -> float:
    ...
```
e chamá-la dos dois lados. **Zero slug novo, zero constante nova** — `focus_disparity_invalid`
e `focus_depth_implausible` já estão registrados (`:163-164`). Enquanto isso a função mora
em `routes/route_a.validate_sampled_focus_disparity`, importando os limites do contrato:
não há como divergir em valor, só em endereço.

### 9.6 `src/control/contract.py` — dois slugs de gate (P2)

`GATE_REJECTION_REASONS` (`:94-125`) precisa de:

```python
    #: Rota A: o p99 de |CoC| ficou abaixo do piso — o alvo renderizado é
    #: indistinguível da entrada, e a amostra não carrega sinal de controle.
    "gate_rendered_coc_too_small",
    #: Rota A: fração grande da imagem com |CoC| >= MAX_COC. Acima do teto o mapa de
    #: condição não distingue mais QUANTO borrar.
    "gate_defocus_saturated",
```

Enquanto não existirem, `GATE_TO_REASON` agrega sob `gate_bokeh_not_blurrier` e
`k_out_of_configured_range`, com a marca `PROVISÓRIO` no código. A distinção não se perde: o
**nome do gate** está no `quality` de cada amostra. Mesmo padrão e mesma razão dos três
slugs provisórios da rota B (`route_b.py:667-686`).

### 9.7 `src/dataio/writer.py` — profundidade compartilhada por cena (P2, item A13/F5)

As N variantes de uma imagem compartilham o **mesmo** array de profundidade —
`encode_depth` roda uma vez por imagem, e recomputá-lo 41 vezes seria 41x de GPU jogada
fora. Mas o writer grava `depth/<sample_id>.png` por amostra, então o disco recebe **41
cópias idênticas**.

O conserto é `depth/<scene_id>.png` mais um `depth_ref` na linha do manifesto. Ele muda o
layout do release e o dataloader, então é decisão de armazenamento a tomar **antes** do run
completo. A conta está na §10.

### 9.8 `src/qc/gates.py` — se os gates novos valerem para outra rota

`rendered_coc_p99_px` e `defocus_saturation_ratio` vivem em `routes/route_a.py` porque só
fazem sentido quando o alvo é renderizado a partir de um K imposto. Se a rota C passar a
medir saturação, eles migram para `qc/gates.py` — que é o lugar certo — e a seção local
some. Mesma regra dos três gates locais da rota B.

### 9.9 `scripts/publish_release.py` — ramificar por rota (P2)

`sem_validador += meta.get("k_analytic") is None` (`:117`) conta **todas** as amostras da
rota A como "sem validador analítico", porque a rota A não tem Eq. 3 e não pode ter. E a
exigência de `source_images.jsonl` é aplicada só a `route == "c"` (`:131`) — a rota A
**também** referencia a AIF e o loader já escreve o ledger, então a checagem deveria valer
para ela.

### 9.10 `scripts/calibrate_thresholds.py` — os gates novos

Ele lê o `quality` do piloto; os nomes novos (`rendered_coc_p99_px`,
`defocus_saturation_ratio`) precisam entrar na tabela dele para a proposta sair com os dois.

---

## 10. Orçamento de disco — a rota A é a que aperta (item F8)

Pela fórmula de `estimate_disk_budget` (`dataio/writer.py:224-242`), 70K amostras, lado
longo 768, **contando as 41 cópias de profundidade** (a fórmula é por amostra):

| resolução da imagem | controle | bokeh JPEG | total |
|---|---|---|---|
| 0,5 MP | 38,4 GB | 22,1 GB | **60,4 GB** |
| 1,0 MP | 38,4 GB | 44,1 GB | **82,5 GB** |
| 2,0 MP | 38,4 GB | 88,2 GB | **126,6 GB** |

Folga medida: ~84 GB (115 GB menos os 31 GB de B+C, `REGISTRO.md:585`). **A rota A cabe a
≤1 MP e não cabe a ≥2 MP.** A resolução das imagens de `[80]` e da EBB! **não foi medida**
(`[A]` A8) — `run_route_a.py` imprime o `Counter` de resoluções e a conta **antes** de
gerar, e `enumeration_summary` mostra as cinco resoluções mais comuns.

Duas coisas fora da conta, as duas para o mesmo lado: a máscara é gravada em resolução de
**imagem** (`writer.py:111-113`) e não entra na fórmula; e o `depth_ref` da §9.7
economizaria a maior parte dos 38,4 GB de controle.

---

## 11. Itens `[A]` da rota A — estado depois desta etapa

| id | `[A]` | estado |
|---|---|---|
| A1 | qual artefato de `[80]` | **aberto.** `--source-revision` grava a resposta quando existir; ausente fica visível como `None` |
| A2 | qual lado da EBB! | **aberto.** `--source-note` grava a escolha por release; resolve medindo a variância dos dois lados |
| A3 | corte do Laplaciano, e se o ranking é por fonte | **decidido**: por fonte, com quota declarada. O corte efetivo de cada fonte vai gravado |
| A4 | divisão do pool de 1,7K | **aberto por natureza** — o paper não publica. Quota é argumento obrigatório |
| A5 | N por imagem = 41 | **decidido `[A]`**, com o par `(K, D_focus)` e o estrato de cada variante gravados |
| A6 | distribuição de K e de `D_focus` | **decidido `[A]`**: K da empírica de B+C alargada 25%; `D_focus` quantil `[0,05, 0,95]` da disparidade plausível. Medido e publicado aqui |
| A7 | K normalizado por resolução ao migrar | **decidido: sim**, com fundamento físico e teste (`test_mesma_imagem_em_duas_resolucoes_da_o_MESMO_CoC_relativo`) |
| A8 | resolução das duas fontes | **aberto**, e é bloqueante para o orçamento. `run_route_a.py` mede antes de gerar |
| A9 | custo de GPU por render | **aberto.** É o que o piloto mede; `route_a_pilot.slurm` roda sob `/usr/bin/time -v` |
| A10 | sobreposição do pool com os benchmarks | **aberto.** Dedup por sha256 existe **dentro** do pool; contra os quatro benchmarks não foi feito |
| A11 | se a rota A deve gravar máscara | **decidido**: grava a banda derivada do rótulo, com a regra completa. A alternativa (campo ausente) exige a §9.4 |
| A12 | limiares dos gates novos | **abertos por decisão.** Todos `None`; saem do piloto |
| A13 | 41 cópias de profundidade ou uma | **aberto**, e é decisão de armazenamento. Ver §9.7 e §10 |

---

## 12. O que esta etapa NÃO mediu

* **Nada em dado real.** Não há acesso ao cluster nesta etapa: as medições da §3 são em
  cena **sintética**, e estão rotuladas como tal. Elas decidem entre duas regras de
  amostragem, não descrevem a distribuição de nenhuma fonte.
* **A distribuição de K de verdade.** Ela só existe depois de B e C regeradas. O que existe
  aqui é o formato, o sorteio, o alargamento e o guarda que recusa a distribuição de hoje.
* **O throughput do BokehMe.** `ACHADOS.md:245-290` mediu o **contrato** do renderer, não o
  custo. 70 mil renders continua sendo a única etapa do projeto sem estimativa.
* **A resolução e a contagem das duas fontes**, porque os diretórios não estão nesta
  máquina. O adaptador é construído contra um `Protocol` e enumera qualquer árvore de
  arquivos; o que falta é apontá-lo para os dados.
