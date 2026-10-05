# Tabela de evidências — a procedência de cada número

**Escrita em 2026-09-10, 19h (horário local, UTC−3).** Este arquivo é novo e não
substitui nada: `reference/ACHADOS.md` continua sendo o registro primário de medições, e
`REGISTRO.md` o diário de decisões. O que este arquivo faz é **cruzar** os dois com o
código e com `reference/paper.txt`, e responder, para cada afirmação factual que o
projeto faz: **quem mediu, com que comando, em que amostra, e quando.**

## Para que serve

Duas perguntas, e a segunda é a que dói:

1. Um avaliador externo pergunta: *"de onde vem esse número?"*
2. Nós, escrevendo o paper, perguntamos: *"posso escrever isto?"*

A resposta é a coluna **origem**. Se ela diz `[SP]`, a resposta é **não** — não sem
medir primeiro.

## Etiquetas

| etiqueta | significa |
|---|---|
| `[M]` | **medido**, com origem rastreável neste repositório ou num job registrado |
| `[I]` | **inferido** de algo medido (aritmética, ou leitura forte de duas medições) |
| `[A]` | **assumido**, não verificado — e declarado como tal no código |
| `[P]` | **lido do paper**, com linha conferida em `reference/paper.txt` |
| `[SP]` | **SEM PROCEDÊNCIA** — o número é afirmado em algum lugar do repositório e **não** existe medição, comando, data ou arquivo que o sustente. Não entra em paper. |
| `[X]` | procedência **externa a este repositório** — existe, mas um avaliador que receba só `bokehnet-regen/` não consegue conferir |

As três primeiras são a convenção herdada de `ACHADOS.md`. `[P]`, `[SP]` e `[X]` são
distinções que este arquivo acrescenta, porque são exatamente as que separam "número que
entra em paper" de "número que não entra".

**Contagem final**: **264 afirmações catalogadas**, das quais **18 marcadas `[SP]`** (sem
procedência) e **14 marcadas `[X]`** (procedência fora deste repositório). As `[SP]`
estão na §2 (8), §4 (1), §10 (1), §12 (2) e §17 (6); as `[X]` estão listadas uma a uma na
§16.2.

Fontes cruzadas: `REGISTRO.md` (1.156 linhas, 9 etapas), `reference/ACHADOS.md`
(584 linhas), `reference/CONTRATO.md`, `reference/MEDICAO_PLANO_FOCO.md`, as três
auditorias de rota, `PLANO_EXECUCAO.md`, `README.md`, `CLAUDE.md`,
`output/renderer_verification.json`, e os 27 arquivos de `src/`, 14 de `tests/`, 6 de
`scripts/` e 4 de `slurm/`.

---

## Aviso de método: o repositório estava sendo editado enquanto isto foi escrito

Três outros agentes trabalhavam em `src/`, `tests/`, `scripts/`, `REGISTRO.md` e
`reference/ACHADOS.md` durante a redação. Consequências concretas, medidas:

| observação | evidência |
|---|---|
| a suíte foi de **519** → **521** → **650** testes em 24 minutos | três execuções: 19:07:39, 19:11:37, 19:31:45 |
| `src/` foi de **7.993** para **9.297** linhas na mesma janela | `wc -l` em dois momentos |
| `src/routes/route_c.py` foi de 648 para **692** linhas | idem |
| `REGISTRO.md` foi de 967 para **1.155** linhas (a etapa 9 apareceu) | idem |
| `reference/ACHADOS.md` foi de 523 para **583** linhas | idem |
| passaram a existir, entre 19:11 e 19:32 | `src/routes/route_b.py` (1.168 linhas), `scripts/run_route_b.py` (623), `src/sources/level_selection.py` (136), `tests/test_route_b.py` (104 testes), `tests/test_level_selection.py` (23), `reference/ROTA_C_ANTES_E_DEPOIS.md` (207) |

Por isso **toda citação `arquivo:linha` para `src/`, `tests/` e `scripts/` neste
documento é datada de 2026-09-10 19h** e pode ter deslizado. Onde o alvo é estável eu
cito o **nome do símbolo** (função, constante, campo) em vez da linha — nome não desliza.
Isto é uma lição, não uma desculpa: ver §16.

---

## 1. O dataset v0 publicado — o objeto que estamos substituindo

Todas as linhas desta seção vêm de leitura de footer/coluna de parquet no Hugging Face,
com o padrão descrito em `ACHADOS.md` ("Como medir sem baixar imagem": token em
`~/.cache/huggingface/token`, file-like seekable sobre HTTP `Range`, `pyarrow.parquet`
com projeção de coluna). **O comando exato não está gravado em nenhum script deste
repositório** — o script que produziu estas medições não foi versionado. Por isso a
coluna "origem" diz `[M]` (o número foi medido) mas a coluna "reproduzível" diz **não**.

| # | afirmação | etiqueta | valor | origem | reproduzível hoje? | registrada em |
|---|---|---|---|---|---|---|
| 1.1 | rota B, amostras publicadas | `[M]` | 11.635 | footer parquet | não (script não versionado) | `ACHADOS.md:13` |
| 1.2 | rota B, coluna `k`: valores distintos | `[M]` | **1**, igual a 50,0 | footer parquet | não | `ACHADOS.md:14` |
| 1.3 | rota B, `s1` | `[M]` | 2,9e-6 a 0,974, mediana 0,0234 | footer parquet | não | `ACHADOS.md:15` |
| 1.4 | rota B, `calibration_ssim` nula | `[M]` | 100% | footer parquet | não | `ACHADOS.md:16` |
| 1.5 | rota C, amostras | `[M]` | 2.932, zero do LFDOF | footer parquet | não | `ACHADOS.md:17` |
| 1.6 | rota C, `k` | `[M]` | 0 a 300, mediana 251,65 | footer parquet | não | `ACHADOS.md:18` |
| 1.7 | rota C, `k == 300` exato | `[M]` | 1.379 / 2.932 = **47,0%** | footer parquet | não | `ACHADOS.md:19` |
| 1.8 | rota C, `k == 0` | `[M]` | 51 = 1,7% | footer parquet | não | `ACHADOS.md:20` |
| 1.9 | rota C, K interior **e** SSIM ≥ 0,6 | `[M]` | 1.483 / 2.932 = 50,6% | footer parquet | não | `ACHADOS.md:21` |
| 1.10 | rota C, `exif` nula | `[M]` | 100% | footer parquet | não | `ACHADOS.md:22` |
| 1.11 | rota A, amostras | `[M]` | ~68.000, nomes só UUID | — a origem é só "[M]", sem comando | não | `ACHADOS.md:23` |
| 1.12 | união real B+C | `[M]` | 14.567 (= 11.635 + 2.932) | aritmética sobre 1.1 e 1.5 — logo é `[I]`, não `[M]` | sim (aritmética) | `ACHADOS.md:24` |
| 1.13 | coluna `depth` é profundidade métrica min-max, não disparidade | `[M]` | erro mediano 0,011% contra 3,008% | teste de pixels | não | `ACHADOS.md:25` |
| 1.14 | `max(defocus_map)` | `[M]` | 65535 exato em **todas**, com k de 33 a 195 | footer parquet | não | `ACHADOS.md:26` |

**Achado de higiene (1.12).** `ACHADOS.md:24` etiqueta a união B+C como `[M]`. Ela é
aritmética de duas medições, ou seja `[I]` pela própria convenção do arquivo. Não muda
nada, e é exatamente o tipo de deslize que a etiqueta existe para não ter.

---

## 2. Controlabilidade — o número que justifica o projeto inteiro

Esta é a seção mais importante para o paper e a de procedência mais fraca. A tabela de
LVCorr é a base da afirmação central (*"o rótulo era o gargalo"*), e **não há, neste
repositório, o comando, o harness, a data nem o checkpoint de cada linha.**

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 2.1 | LVCorr fase 1 (só sintético), LF-Bokeh | `[M]` | +0,9059 | `[SP]` — nenhum comando, harness ou data | `ACHADOS.md:189` |
| 2.2 | LVCorr fase 1, RealBokeh / RealDOF | `[M]` | +0,8609 / +0,9247 | `[SP]` | `ACHADOS.md:189` |
| 2.3 | LVCorr pesos oficiais do paper | `[M]` | +0,8868 / +0,8498 / +0,9644 | `[SP]` | `ACHADOS.md:190` |
| 2.4 | LVCorr nossa fase 2 (a+b+c) | `[M]` | +0,4365 / +0,8257 / +0,1324 | `[SP]` | `ACHADOS.md:191` |
| 2.5 | LVCorr kfix (K da rota B pela Eq. 3) | `[M]` | +0,8288 / +0,4832 / **−0,4599** | `[SP]` | `ACHADOS.md:192` |
| 2.6 | LVCorr só-rota-c (a+c) | `[M]` | +0,7296 / +0,8245 / −0,2061 | `[SP]` | `ACHADOS.md:193` |
| 2.7 | degradação monótona no tempo de treino (só-rota-c, RealDOF) | `[M]` | +0,2509 @10K · +0,0281 @20K · −0,1776 @30K · −0,2061 @60K | `[SP]` | `ACHADOS.md:195-196` |
| 2.8 | LPIPS da só-rota-c bate os pesos oficiais nas três mesas | `[M]` | 0,1804 / 0,1134 / 0,1251 contra 0,2047 / 0,3454 / 0,2677 | `[SP]` | `ACHADOS.md:202-203` |
| 2.9 | *"consertar só o K da rota B leva o LF-Bokeh de +0,4365 para +0,8288"* | `[I]` | — | leitura de 2.4 e 2.5; a inferência causal é nossa | `ACHADOS.md:198-200`, `README.md:14-19` |

**O que isto significa para o paper.** As oito primeiras linhas são as únicas medições
*de modelo treinado* que o projeto tem, e são o argumento inteiro. Elas foram feitas
antes deste repositório existir, em outro pipeline, e **nada aqui permite refazê-las**.
Um avaliador externo que peça "mostre como você mediu +0,4365" não recebe resposta deste
repositório.

**Recomendação, e é a mais importante deste documento:** antes de escrever qualquer uma
dessas linhas num paper, versionar aqui o script de avaliação (o `LVCorr`), a definição
exata da métrica, a lista de checkpoints com sha256, e o comando. Enquanto isso não
existir, tratar 2.1–2.8 como `[SP]` no texto, não como `[M]`.

---

## 3. A tabela `rota-b-kfix-eq3` — reconstrução numérica, não código

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 3.1 | `k_eq3`: min / mediana / max / distintos | `[M]` | 833,77 / **16.553,9** / 48.514 / 11.631 | footer parquet | `ACHADOS.md:32` |
| 3.2 | `z_focus_m` | `[M]` | 0,2191 / 3,1101 / 10.000 / 11.564 | idem | `ACHADOS.md:33` |
| 3.3 | `z_min_m` | `[M]` | 0,1052 / 1,9767 / 26,594 | idem | `ACHADOS.md:34` |
| 3.4 | `z_max_m` | `[M]` | 0,3621 / 70,800 / 10.000 / 8.647 | idem | `ACHADOS.md:35` |
| 3.5 | `pixel_ratio` | `[M]` | 22,22 / **42,625** / 277,33 / 542 distintos | idem | `ACHADOS.md:36` |
| 3.6 | `coc_p99_px` | `[M]` | 0,1002 / **4,665** / 112,0 | idem | `ACHADOS.md:37` |
| 3.7 | `k_antigo` constante | `[M]` | 50,0, 1 valor distinto | idem | `ACHADOS.md:38` |
| 3.8 | `max_coc_calibrado` constante | `[M]` | **10,510746** | idem | `ACHADOS.md:39` |
| 3.9 | `k_eq3` reconstruído pela Eq. 3 com erro relativo 0 | `[M]` | 0,000e+00 em 11.635/11.635 | reconstrução numérica | `ACHADOS.md:41-42` |
| 3.10 | `coc_px == k_eq3 · |Δ(1/z_mm)|` | `[M]` | erro < 1e-6 | idem | `ACHADOS.md:43` |
| 3.11 | `max_coc_calibrado` é o percentil 82,66 de `coc_p99_px` | `[M]` | 10,510746 | idem | `ACHADOS.md:44` |
| 3.12 | cobertura do kfix | `[M]` | 100% da rota B, **0% da rota C** | idem | `ACHADOS.md:46` |
| 3.13 | `scripts/rotab_kfix.py` não existe no repositório | `[M]` | — | ausência verificável | `ACHADOS.md:46-48` |
| 3.14 | logo "mediana vs média" e a regra do percentil são **reconstrução**, não código lido | `[I]` | — | declarado em `ACHADOS.md` | `ACHADOS.md:47-48` |
| 3.15 | `k_official(16553.9) = 16,5539` | `[M]` | 16,5539 | execução do nosso `contract.k_official`; **reproduzível aqui** | `ACHADOS.md:229` |

**3.15 é o modelo de como uma linha deveria ser.** É a única âncora numérica desta seção
que um avaliador refaz em três segundos com o código deste repositório:

```
PYTHONPATH=src /Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python \
  -c "from control.contract import k_official; print(k_official(16553.9))"
```

---

## 4. Âncoras de K — quatro caminhos, e um sem procedência

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 4.1 | `k_value` mediano da rota B pela kfix | `[M]` | **16,55** (= 16.553,9 / 1000) | `[M]` em 3.1 + `[I]` na divisão | `CONTRATO.md:76` |
| 4.2 | `k_value` mediano calculado da EXIF, "318 amostras com distância de foco" | `[M]` afirmado | **20,1** | **`[SP]`** — não há linha em `ACHADOS.md`; nem as 318 amostras, nem o comando, nem a data | `CONTRATO.md:77`, ecoado em `ROTA_B_AUDITORIA.md:168` |
| 4.3 | default de K da inferência oficial | `[X]` | **15,0** | `Inference_bokehNet.py:53` — arquivo **fora** deste repositório | `CONTRATO.md:78` |
| 4.4 | K varrido na Fig. 12 do paper | `[P]` | **{0, 5, 10, 15}**, com `K=0` rotulado "(Input)" | **conferido**: `paper.txt:1151` (rótulos) e `:1154` (legenda) | `CONTRATO.md:79` |
| 4.5 | `k_value` esperado da rota C | `[I]` | 3,6 a 36 | Eq. 3 sobre o `metadata/` da RealBokeh **assumindo sensor de 36 mm** | `CONTRATO.md:80` |
| 4.6 | o cálculo de 4.5 fecha numericamente | `[I]` | `pixel_ratio = 2000/36 = 55,6`; `f=50`, `z=3`, `F=2` → `K = 35,3` | refeito nesta auditoria e **confere** | `ROTA_C_AUDITORIA.md:470-471` |

**4.2 é o caso mais claro de "número sem procedência que já virou âncora".** Ele está no
`CONTRATO.md`, é citado como `[M]` pela auditoria da rota B, e sustenta a frase "três
caminhos independentes na mesma faixa" que aparece três vezes no repositório (inclusive
em `src/control/contract.py`, na docstring de `k_official`). A medição em si pode muito
bem ter sido feita; o que não existe é o registro dela.

---

## 5. RealBokeh bruta (`timseizinger/RealBokeh_3MP`)

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 5.1 | arquivos totais | `[M]` | 31.853 | listagem da API do HF | `ACHADOS.md:70` |
| 5.2 | `train/in/` | `[M]` | 3.960 imagens, **todas** `_f22.JPG`, uma por cena | idem | `ACHADOS.md:73` |
| 5.3 | `train/gt/` | `[M]` | 20.554 imagens em 3.960 pastas de cena | idem | `ACHADOS.md:74` |
| 5.4 | `train/metadata/` | `[M]` | 3.960 JSONs | idem | `ACHADOS.md:75` |
| 5.5 | `test` / `validation` | `[M]` | 220 / 1.257 / 220 e 220 / 1.240 | idem | `ACHADOS.md:76` |
| 5.6 | `parse_aperture` casa com 100% dos nomes de `gt/` | `[M]` | 20.554 / 20.554 | idem | `ACHADOS.md:81-82` |
| 5.7 | logo as 1.028 cenas perdidas **não** são falha de parser | `[M]` | — | consequência de 5.6 | `ACHADOS.md:82-83` |
| 5.8 | imagens por cena em `train/gt/` | `[M]` | 2→704 · 3→620 · 5→2.346 · 7→9 · 9→34 · 21→247 | idem | `ACHADOS.md:85` |
| 5.9 | f-stop máximo por cena | `[M]` | ≤f/2.8 → 3,0% · ≤f/5.6 → 12,7% · ≤f/8 → 22,1% · ≤f/11 → 38,6%; mediana **f/14** | idem | `ACHADOS.md:88` |
| 5.10 | `metadata/<id>.json` traz `focal_length`, `target_avs`, `focus_plane_distance` (m) e `focus_plane_uncertainty` | `[M]` | — | leitura dos JSONs | `ACHADOS.md:90-93` |
| 5.11 | 1.028 cenas = 26% são rejeição de QC (`qc.severity=="red"`) | `[I]` **hipótese aberta** | 3.960 − 2.932 = 1.028 = 25,96% | aritmética; a **causa** não foi medida. Checagem proposta: `grep -c rejected_qc route_c_log.jsonl` — nunca executada | `ACHADOS.md:95-98` |
| 5.12 | os 4.400 JSONs de metadata, baixados inteiros, 0 erros | `[M]` | 3.960 + 220 + 220 | ao escrever `sources/realbokeh.py` | `ACHADOS.md:314-317` |
| 5.13 | presença 100% de 10 campos em `metadata/` | `[M]` | `id`, `source_image`, `source_av`, `target_images`, `target_avs`, `focal_length`, `ISO`, `EV`, `focus_plane_distance`, `focus_plane_uncertainty` | 4.400 JSONs | `ACHADOS.md:340-341` |
| 5.14 | `source_av` é 22,0 sempre | `[M]` | 4.400 / 4.400 | idem | `ACHADOS.md:344` |
| 5.15 | `focal_length` | `[M]` | 28 a 70 mm | idem | `ACHADOS.md:345` |
| 5.16 | `focus_plane_distance` | `[M]` | 0,355 a 368,63 m | idem | `ACHADOS.md:346` |
| 5.17 | `focus_plane_uncertainty` | `[M]` | 0,0 a 315,895 m — **zero em 8 cenas** | idem | `ACHADOS.md:347` |
| 5.18 | `target_avs` | `[M]` | 2,0 a 20,0 (21 valores distintos), 23.051 no total | idem | `ACHADOS.md:348` |
| 5.19 | `len(target_avs) == len(target_images)` | `[M]` | 4.400 / 4.400 | idem | `ACHADOS.md:333` |
| 5.20 | `target_avs` em ordem crescente | `[M]` | 4.400 / 4.400 | idem | `ACHADOS.md:334` |
| 5.21 | `target_avs[i]` bate com o f-number no nome de `target_images[i]` | `[M]` | 23.051 / 23.051 | idem | `ACHADOS.md:335` |
| 5.22 | **`level` é 1-based**, por três discriminadores independentes | `[M]` | cenas com `1..n` contíguos: 3.959/3.959 · cenas com `level_0`: 0 · pares com `level == len(target_avs)`: 3.949 · fora de faixa: 0 de 20.495 | idem | `ACHADOS.md:323-326` |

---

## 6. Espelho `akcit-pixel/RealBokeh` — e a medição que se corrigiu a si mesma

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 6.1 | ~~o espelho publica só `train` (20.495/20.495)~~ | **`[M]` REVOGADO** | — | medição incompleta: o dump de coluna cobria só os 85 shards de `train` | `ACHADOS.md:402`, revogado em `:433-435` |
| 6.2 | o espelho publica **três** splits, 96 shards | `[M]` | `train` 85 / 20.495 / 3.959 · `test` 6 / 1.257 / 220 · `validation` 5 / 1.238 / 220 · **total 96 / 22.990 / 4.399** | leitura da coluna `file_name_base` dos 96 shards do snapshot `fb183bfd7…` | `ACHADOS.md:421-431` |
| 6.3 | a numeração de cena **reinicia** em cada split | `[M]` | `train ∩ test` = 220 · `train ∩ validation` = 220 · `test ∩ validation` = 220 · nomes idênticos entre `test` e `validation`: **0** | idem | `ACHADOS.md:441-445` |
| 6.4 | custo evitado de 6.3, em amostras | `[I]` | **2.495** (= 1.257 + 1.238) descartadas por `source_duplicate_sample` | aritmética sobre 6.2; refeita aqui e **confere** | `ACHADOS.md:459-460`, `REGISTRO.md:854` |
| 6.5 | schema do espelho | `[M]` | `image_pre_deblur`, `image_blur`, `image_focus`, `file_name_base` | leitura | `ACHADOS.md:103` |
| 6.6 | forma de `file_name_base` | `[M]` | `timseizinger_realbokeh_3mp_<split>_f_<cena>_level_<N>_<sufixo>` | leitura | `ACHADOS.md:104` |
| 6.7 | **`image_focus` É o `train/in/<id>_f22.JPG`, byte a byte** | `[M]` | sha256 `59d8e910ca69…`, 2000×1500, 1927 KB, byte-idêntico entre `level_3` e `level_4` da cena 1038; nenhuma das 5 imagens de `train/gt/1038/` bate | comparação de sha256 | `ACHADOS.md:122-141` |
| 6.8 | faltam 59 imagens e 1 cena contra o `gt/` bruto | `[M]` | 59 imagens em **11 cenas**; a "cena inteira" (`2255`) é uma das 11 | comparação de listas | `ACHADOS.md:373-391` |
| 6.9 | os níveis ausentes são **sempre a cauda** | `[M]` | 11/11 cenas | idem | `ACHADOS.md:387-388` |
| 6.10 | 0 cenas do espelho sem metadata | `[M]` | 0 | idem | `ACHADOS.md:389-390` |
| 6.11 | o sufixo do nome é uma **anotação de alinhamento**, não sempre `aligned` | `[M]` | 32 sufixos distintos: `aligned` 20.074 (97,94%) · `misaligned` 183 (0,89%) · `shift_<X.Y>px` 238 (1,16%), 30 valores de 2,0 a 4,9 px | contagem sobre 20.495 linhas | `ACHADOS.md:355-370` |
| 6.12 | total de pares que a origem declara mal registrados | `[M]` | **421 = 2,05%**, em 228 cenas | aritmética sobre 6.11; refeita e **confere** (183+238=421; 421/20.495=2,054%) | `ACHADOS.md:364-366` |
| 6.13 | o deslocamento anotado foi medido nos 2000×1500 do espelho | **`[A]`** | — | a origem não declara a resolução | `ACHADOS.md:370` |
| 6.14 | `misaligned` **não** significa 0,0 px | `[M]` (ausência) | — | a origem não diz quanto | `ACHADOS.md:371` |
| 6.15 | enumeração completa do espelho pelo nosso adaptador | `[M]` | 20.495 entradas → 20.495 pares (100%), 0 rejeitados, 3.959 cenas | `sources.realbokeh.enumerate_pairs` | `ACHADOS.md:395-403` |
| 6.16 | níveis por cena no espelho | `[M]` | 1→2 · 2→705 · 3→621 · 4→1 · 5→2.341 · 6→1 · 7→9 · 9→34 · 12→1 · 21→244 | idem | `ACHADOS.md:400` |
| 6.17 | 6.16 fecha nas duas somas | `[I]` | Σcenas = 3.959 ✓ · Σpares = 20.495 ✓ | **refeito nesta auditoria**, confere exato | — |
| 6.18 | `_aligned` sugere registro geométrico na construção do espelho | **`[A]`** | não confirmado | — | `ACHADOS.md:147` |

**6.1 é o melhor item deste arquivo, e vale copiar o método.** Uma medição `[M]`
publicada foi depois **revogada por outra medição mais completa**, e o registro diz por
quê: *"não estava errada sobre o que mediu, estava incompleta sobre o que existe."* É a
única forma honesta de retratar um `[M]`.

**Consequência que ainda está viva:** três lugares do código **repetem a afirmação
revogada**. Ver §13.

---

## 7. BokehDiffusion / rota B — a EXIF

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 7.1 | linhas totais | `[M]` | 15.305 | leitura da coluna inteira, 1,11 MB de `train.parquet` | `ACHADOS.md:153-155` |
| 7.2 | linhas que passam o filtro da rota B | `[M]` | 13.800 | idem | `ACHADOS.md:155` |
| 7.3 | `focal_length_35` presente | `[M]` | 13.800 / 13.800 = **100,0%** | idem | `ACHADOS.md:156` |
| 7.4 | crop factor implícito | `[M]` | min 1,00 · mediana 1,50 · p95 2,75 · max 9,75 | idem | `ACHADOS.md:163` |
| 7.5 | largura de sensor implícita | `[M]` | 3,69 a 36,00 mm, mediana 24,00 (APS-C) | idem | `ACHADOS.md:164` |
| 7.6 | crop factor **exatamente 1,0** | `[M]` | 4.185 = **30,33%** | idem | `ACHADOS.md:165` |
| 7.7 | crop factor > 8 (sensor < 4,5 mm) | `[M]` | 35 | idem | `ACHADOS.md:166` |
| 7.8 | *"trinta por cento de full-frame num dataset do Flickr é alto"* | `[I]` | — | juízo declarado como tal | `ACHADOS.md:169-171` |
| 7.9 | custo de errar o crop factor | `[I]` | assumir 1,0 onde o real é 5,6 subestima K por **5,6×** | álgebra: `pixel_ratio ∝ 1/sensor_width` | `ACHADOS.md:175`, `contract.py` docstring de `sensor_width_mm` |
| 7.10 | `fx_px = f_mm · pixel_ratio = max(W,H) · focal_length_35 / 36` | `[M]` | identidade verificada em **900/900** linhas, 32 delas em retrato | reconstrução numérica | `ACHADOS.md:62-64` |
| 7.11 | logo `max(H,W)` é o correto, não a largura | `[M]` + `[P]` | — | 7.10, **e** `paper.txt:1186-1187` conferido | `ACHADOS.md:64`, `ROTA_B_AUDITORIA.md:84-93` |
| 7.12 | `fx` derivável: rota B 100%, rota C 0% | `[M]` | 11.635/11.635 e 0/2.932 | footer parquet | `ACHADOS.md:59-60` |
| 7.13 | ITW `[19]` = `atfortes/BokehDiffusion` no HF | **`[I]`** | — | autor (`Fortes, A.`, `paper.txt:791`) + handle + volume (13.800 contra "13K", `paper.txt:1000`). O paper **não publica URL nem repositório** | `ROTA_B_AUDITORIA.md:62-67`, `:949` |

---

## 8. LFDOF — 30 medições que existem **só** numa docstring

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 8.1 | `akcit-pixel/LFDOF`: train / test | `[M]` | 11.247 / 725 | leitura | `ACHADOS.md:180` |
| 8.2 | AIF **real**, renderizada da light field | `[M]` | — | schema | `ACHADOS.md:180-181` |
| 8.3 | fonte original pública | `[M]` | `sweb.cityu.edu.hk/miullam/AIFNET/` → `LFDOF.zip`, ~11 GB | leitura da página | `ACHADOS.md:182` |
| 8.4 | **sem licença explícita na página** | `[M]` (ausência) | — | idem — bloqueante para publicar pixels | `ACHADOS.md:182`, `ROTA_C_AUDITORIA.md:994` |
| 8.5 | N desfocadas por AIF ⇒ split obrigatoriamente por cena | `[I]` | — | consequência do schema | `ACHADOS.md:183` |
| 8.6 | o LFDOF **não publica óptica** (sem `metadata/`, sem f-number, sem distância de foco) | `[M]` | — | `sources/lfdof.py`, cabeçalho, e o schema | `ROTA_C_AUDITORIA.md:532` |
| 8.7 | 100% do LFDOF seria rejeitado pelo gate `aif_aperture_is_narrow` | `[M]` — **e já corrigido** | `value=nan`, `passed=False`, slug `gate_aif_aperture_wide` | medido na auditoria da rota C; o conserto (`applicable=False`) está em `qc/gates.py` e tem teste | `ROTA_C_AUDITORIA.md:501-530` |

### 8.9 As medições do adaptador do LFDOF — todas `[M]`, e nenhuma em `ACHADOS.md`

Feitas em 2026-09-10 lendo **só a coluna `file_name_base`** dos 83 shards por projeção de
coluna sobre HTTP `Range` — 1,1 MB por shard de ~500 MB, ~90 MB no total contra os 37 GB
do repositório. **Todas estão registradas apenas no cabeçalho de
`src/sources/lfdof.py`.** Nenhuma tem linha em `reference/ACHADOS.md`.

| # | afirmação | etiqueta | valor |
|---|---|---|---|
| 8.10 | shards | `[M]` | **83** = 78 de `train` + 5 de `test` |
| 8.11 | linhas | `[M]` | **11.972** = 11.247 + 725 |
| 8.12 | row groups por shard | `[M]` | 5 em 83/83 |
| 8.13 | linhas por shard | `[M]` | 145 em 20 shards, 144 em 63 |
| 8.14 | cenas distintas | `[M]` | **840** = 790 (`train`) + 50 (`test`) |
| 8.15 | `file_name_base` repetidos | `[M]` | **0 / 11.972** |
| 8.16 | `(split, cena, nível)` repetidos | `[M]` | **0 / 11.972** |
| 8.17 | o regex do nome casa | `[M]` | 11.972 / 11.972 |
| 8.18 | número de cena é de 4 dígitos, numérico | `[M]` | 840 / 840 |
| 8.19 | `train ∩ test` (números de cena) | `[M]` | **0** de 840 — ao contrário da RealBokeh |
| 8.20 | `image_focus` é byte-idêntico entre os níveis de uma cena | `[M]` | 90 linhas de 9 cenas, 3 row groups de 3 shards; **0 cenas** com mais de um sha256; cena 1275 → `d358a3847630…` |
| 8.21 | 15 desfocadas por AIF | `[M]` | em **682 das 840** cenas |
| 8.22 | anotação de alinhamento | `[M]` | `aligned` 11.528 (96,29%) · `misaligned` 204 (1,70%) · `shift_<X.Y>px` 240 (2,00%), 31 valores de 2,0 a 5,0 px |
| 8.23 | total não-alinhado | `[M]` | **444 = 3,71%** — quase o dobro dos 2,05% da RealBokeh |
| 8.24 | cenas com alinhamento **misto** | `[M]` | 195 / 840 |
| 8.25 | cenas 100% não-alinhadas | `[M]` | 4 / 840 |
| 8.26 | `level` é 1-based | `[M]` | menor nível 1; cenas com nível 0: **0** |
| 8.27 | níveis exatamente `1..n` contíguos | `[M]` | **837 / 840 = 99,64%** |
| 8.28 | as 3 cenas com buraco no meio | `[M]` | `train 2022` sem o 10 · `train 3758` sem o 8 · `train 4863` sem o 8 |
| 8.29 | maior nível | `[M]` | 17 (1 cena); distribuição 2 (3 cenas) … 15 (682) … 17 (1) |
| 8.30 | resolução e formato | `[M]` em 90 linhas / **`[A]`** para as 11.972 | `(H,W) = (688, 1008)` em 90/90; PNG, modo **RGBA**, alpha `min = max = 255` em 90/90 |
| 8.31 | o `path` da célula bate com o papel da coluna | `[M]` | **180 / 180** células conferidas |
| 8.32 | `source_duplicate_sample` hoje | `[M]` | **0 em 11.972** |
| 8.33 | cenas espalhadas por mais de um shard | `[M]` | 75 de 840 |
| 8.34 | blocos contíguos de cena na ordem `(shard, linha)` | `[M]` | **840 para 840 cenas** |
| 8.35 | economia do memo de decodificação da AIF | `[I]` | decodificar 15× por cena é 15× desperdiçado; gravar a AIF por `sample_id` custaria ~13 GB contra ~1 GB |
| 8.36 | `level` é ordinal **opaco** | **`[A]` declarado** | *"Não medimos, e portanto não afirmamos, que ele seja monótono no raio do desfoque, nem que seja comparável entre cenas"* |
| 8.37 | as 11.972 amostras do LFDOF **não têm validador analítico** | `[M]` | `_analytic_k` devolve `None` para todas; `enumeration_summary` imprime "AUSENTE em 100%" |

**O problema de método, e é o mesmo da §2 ao contrário.** Estas 28 medições são de boa
qualidade: comando descrito, amostra declarada, `[M]` e `[A]` separados, e três delas
derrubam explicitamente um `[A]` de auditoria anterior. Mas vivem numa docstring de
`src/sources/lfdof.py` — um arquivo que outro agente edita. `reference/ACHADOS.md` é o
registro que o `CLAUDE.md` manda usar (*"Cite de lá em vez de re-medir. Acrescente linha
quando medir coisa nova"*), e nenhuma delas foi acrescentada. **Se `lfdof.py` for
reescrito, 28 medições somem.**

---

## 9. Renderer verificado em GPU — job SLURM 32212, h100n3, 2026-09-10

Esta é a seção com a **melhor procedência do repositório**: número medido em GPU,
relatório JSON versionado (`output/renderer_verification.json`), proveniência
criptográfica, e um script que refaz a medição (`scripts/verify_renderer.py`).

| # | afirmação | etiqueta | valor | origem | conferido contra o JSON? |
|---|---|---|---|---|---|
| 9.1 | job, nó, estado | `[M]` | **32212**, `h100n3`, `COMPLETED`, exit 0 | `REGISTRO.md:428` | o JSON não carrega o número do job — ver nota |
| 9.2 | commit do BokehMe | `[M]` | `8b3ed556dc146207c9bb39fccd078a113f8d1bd8` | `renderer_verification.json` → `provenance.renderer_commit` | ✓ |
| 9.3 | sha256 do `pipeline` extraído por AST | `[M]` | `b1274e2bed088d82653b87ec19a50e67692aef54f…` | idem → `demo_pipeline_sha256` | ✓ |
| 9.4 | sha256 do `scatter.py` **pós-patch** | `[M]` | `cc56904e9cede5b9bcc814d3933f36d55dd5696…` | idem → `scatter_py_sha256` | ✓ |
| 9.5 | sha256 de `arnet.pth` / `iunet.pth` | `[M]` | `6c7dba6f32e3afc8…` / `6df927797a1657ba…` | idem | ✓ |
| 9.6 | saída usada | `[M]` | `bokeh_pred` (híbrida) | idem → `config.output` | ✓ |
| 9.7 | **teste 1 — disco, não gaussiana** | `[M]` | `edge_width_ratio = 0,1433675` (disco < 0,5) | idem → `measurements.edge_width_ratio` | ✓ |
| 9.8 | o discriminador de gaussiana vale 1,43 independente de σ | `[I]` derivado analiticamente | r90=0,459σ, r50=1,177σ, r10=2,146σ → (2,146−0,459)/1,177 = **1,4333** | **refeito nesta auditoria, confere** | `REGISTRO.md:307-309` |
| 9.9 | **teste 2 — `bokeh_pred`** | `[M]` | slope **0,9872884**, intercept +0,7916804 px, resíduo máx 0,4168 px (1,085%) | idem | ✓ |
| 9.10 | **teste 2 — `bokeh_classical`** | `[M]` | slope 0,9618547, intercept +1,0259784 px, resíduo máx **8,88e-15 px** | idem | ✓ |
| 9.11 | **teste 2 — `bokeh_neural`** | `[M]` | slope 0,0994, intercept +128,17 px, resíduo 1,268 px (3,30%) | idem | ✓ |
| 9.12 | faixa de K verificada | `[M]` | K ∈ {8, 16, 32, 64, 96}, raios esperados 3,2 a 38,4 px | idem → `k_values`, `expected_px` | ✓ |
| 9.13 | `k_effective_factor` | `[M]` | **0,9872884** — o raio renderizado é 1,27% menor que o K nominal prediz | idem | ✓ |
| 9.14 | o desvio no clássico puro | `[I]` | 1 − 0,9619 = **3,81%** | aritmética; refeita, confere | `ACHADOS.md:280` |
| 9.15 | `bokeh_neural` isolado devolve ~128 px independente de K | `[M]` | 127,2 a 131,3 px para K de 8 a 96 | JSON → `radius_response_bokeh_neural.measured_px` | ✓ |
| 9.16 | **teste 3 — `pipeline` não lê `args.highlight`** | `[M]` | `pipeline_usa_highlight: false` | idem | ✓ |
| 9.17 | os sete parâmetros do BokehMe estão congelados e gravados | `[M]` | `gamma=4.0`, `defocus_scale=10.0`, `highlight=false`, `highlight_rgb_threshold=0,8627451`, `highlight_enhance_ratio=0.4`, `output=bokeh_pred`, `gamma_min/max=1.0/5.0` | idem → `provenance.config` | ✓ |
| 9.18 | `scatter_ex.py` traz `ModuleRenderScatterEX` com `poly_sides` (abertura poligonal) | `[M]` | — | leitura do checkout | `ACHADOS.md:305-308` |

### Duas ressalvas que um avaliador vai levantar sobre a §9

**(a) O laudo não carrega o número do job nem a data.** `output/renderer_verification.json`
tem `renderer_dir` apontando para `/raid/user_danielpedrozo/…`, o que prova *onde* rodou,
mas não *quando* nem *sob qual job*. O vínculo "job 32212, 2026-09-10" existe só em prosa
(`REGISTRO.md:428`, `ACHADOS.md:245`, `CONTRATO.md:106`), e **não há log de SLURM
versionado neste repositório** (`logs/` está no `.gitignore`). Conserto barato: gravar
`SLURM_JOB_ID`, hostname e timestamp UTC dentro do JSON.

**(b) `bokeh_classical` e `bokeh_pred` têm raios medidos IDÊNTICOS em 4 dos 5 K.**
Comparando as duas listas no JSON:

| K | `bokeh_classical` | `bokeh_pred` |
|---|---|---|
| 8 | 4,103913408340617 | 4,103913408340617 |
| 16 | 7,181848464596079 | 7,181848464596079 |
| 32 | 13,337718577107005 | 13,337718577107005 |
| 64 | 25,649458802128855 | 25,649458802128855 |
| **96** | **37,961199027150705** | **38,98717737923586** |

Ou seja: a afirmação *"o renderer clássico é uma reta exata, resíduo 0,0000 px"* (9.10) e
a afirmação *"o desvio do `bokeh_pred` vem do peso do `error_map`, que varia com K"*
(`ACHADOS.md:270`) repousam, juntas, sobre **um único ponto diferente**, o de K=96. Com
4 pontos iguais e 1 diferente, o resíduo zero do clássico e o resíduo 0,4168 px do
híbrido são a mesma informação vista de dois lados. Isso **não invalida** o teste — a
linearidade continua medida sobre 5 pontos — mas a frase "o clássico dá resíduo 0,0000 px
sobre raios de 3,2 a 38,4 px" (`ACHADOS.md:277-278`) é mais forte do que a evidência
sustenta, porque em 4 dos 5 raios as duas saídas são o mesmo número. **Etiqueta correta:
`[M]` para os números, `[I]` para a interpretação causal do `error_map`.**

---

## 10. Piloto da rota C — job 32224 — e o diagnóstico de máscara — job 32231

Esta é a medição que **mudou o método**. Documento primário:
`reference/MEDICAO_PLANO_FOCO.md`.

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 10.1 | job, amostra, nó | `[M]` | job **32224**, 162 aceitas de 204 pares, 29 cenas (`train` e `validation`), h100n3 | `MEDICAO_PLANO_FOCO.md:3-4` | `[SP]` no log: nenhum log de SLURM nem `run_config.json` do piloto está versionado |
| 10.2 | concordância da máscara do BiRefNet com o plano de foco medido, ±25% | `[M]` | **57 de 162 = 35,2%** | comparação `focus_disparity` vs `1/focus_plane_distance_m` | `MEDICAO_PLANO_FOCO.md:28` |
| 10.3 | razão obtida ÷ gabarito | `[M]` | p05 0,165 · **mediana 0,579** · p95 1,505 | idem | `MEDICAO_PLANO_FOCO.md:29-31` |
| 10.4 | a máscara escolhe um plano ~1,7× mais longe | `[I]` | 1 / 0,579 = 1,727 | aritmética sobre 10.3; refeita, confere | `MEDICAO_PLANO_FOCO.md:37-38` |
| 10.5 | distância de foco mediana medida / implicada pela máscara | `[M]` | 0,410 m / 1,020 m | idem | `MEDICAO_PLANO_FOCO.md:32-33` |
| 10.6 | incerteza mediana publicada pela origem | `[M]` | **±0,010 m** | `metadata/` | `MEDICAO_PLANO_FOCO.md:34` |
| 10.7 | maiores divergências | `[M]` | `c_realbokeh_train_91_l1..l3`: 10,230 m (±2,06) vs 1,649 m, razão 6,20 · `c_realbokeh_validation_70_l1,_l10`: 0,405 m (±0,00) vs 2,456 m, razão 0,16 | idem | `MEDICAO_PLANO_FOCO.md:42-46` |
| 10.8 | 10.7 é aritmeticamente coerente com a definição de "razão" | `[I]` | 10,230/1,649 = 6,20 ✓ · 0,405/2,456 = 0,165 ✓ | **refeito nesta auditoria** | — |
| 10.9 | descarte por `focus_mask_empty` | `[M]` | **42 de 204 = 20,6%**, em 10 cenas inteiras, **zero** cenas com aceite e rejeição misturados | histograma de rejeição do piloto | `MEDICAO_PLANO_FOCO.md:49-52` |
| 10.10 | a causa: o BiRefNet **declina**, não falha | `[M]` | probabilidade máxima **0,000 exata** em 5 de 7 cenas; baixar o limiar para 0,05 não recupera nada | `scripts/diagnose_empty_masks.py`, **job 32231**, 12 cenas | `MEDICAO_PLANO_FOCO.md:57-72` |
| 10.11 | tabela de probabilidade crua por cena | `[M]` | `train_102`, `train_1527`, `train_1800`, `train_1873`, `test_102` → 0,000 · `test_100` → max 1,000, área@0,50 = 0,526 · `test_101` → max 1,000, área@0,50 = 0,369 | idem | `MEDICAO_PLANO_FOCO.md:60-67` |
| 10.12 | a razão é de domínio: a RealBokeh é de **cenas**, não de fotos de objeto | `[I]` | `train_102` = tronco de árvore num parque; `test_100` = muro de pedra | inspeção visual | `MEDICAO_PLANO_FOCO.md:73-77` |
| 10.13 | o risco estava documentado antes de ter número | `[M]` | cabeçalho de `src/model_runtime/segmentation.py` | verificável no repositório | `MEDICAO_PLANO_FOCO.md:79-82` |

### Uma inconsistência interna na §10, pequena e real

`MEDICAO_PLANO_FOCO.md` afirma, na mesma tabela, "mediana da razão = 0,579" (⇒ plano
1,7× mais longe, 10.4) **e** "distância mediana medida 0,410 m contra 1,020 m implicada"
(⇒ 2,49× mais longe). As duas coisas podem coexistir — mediana de razões não é razão de
medianas, e a distribuição é assimétrica —, mas o texto usa só o 1,7× e deixa o leitor
com dois fatores diferentes para a mesma frase. Para o paper, usar **um** e dizer qual
estatística é.

### A contradição de ordenação entre a §10 e a auditoria da rota C

`ROTA_C_AUDITORIA.md:1019-1030` afirma, na seção "o que este relatório NÃO conseguiu
medir": *"**Nada rodou com GPU, BokehMe real, Depth Pro ou BiRefNet**"* e *"a
distribuição de `K*` da rota C é não medida"*. `MEDICAO_PLANO_FOCO.md` documenta um
piloto de 204 pares que rodou **exatamente isso** (job 32224, h100n3).

Não é um erro de fato: as duas coisas foram escritas no mesmo dia, e a auditoria
precedeu o piloto. É um erro de **arquivo sem carimbo de ordem**. Um avaliador que leia
os dois na ordem alfabética conclui que o repositório se contradiz. Conserto: datar com
hora, ou marcar a seção da auditoria como superada. **Registrado aqui, não corrigido lá**
— aquele arquivo é de outro agente.

---

## 11. DeblurNet — o recorte por `long_side`

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 11.1 | `long_side = 0` (default oficial e da rota B antiga) **não recorta** | `[M]` | round-trip é identidade geométrica exata; em 4032×3024, 5184×3456 e 1920×1080 nem redimensiona | reimplementação da aritmética de `Inference_deblurNet.py:13-49`, reprodutível por `model_runtime.deblurnet.plan_processing`, travada em `tests/test_deblurnet.py::Geometria` | `ACHADOS.md:473-486` |
| 11.2 | `long_side > 0` **recorta** o lado curto, com `floor` para múltiplo de 16 | `[M]` | — | idem | `ACHADOS.md:488-489` |
| 11.3 | deslocamento máximo do par AIF↔bokeh, 4:3 e 16:9 | `[M]` | **0,00 px** | idem | `ACHADOS.md:493-494` |
| 11.4 | deslocamento em 3:2, `long_side=512` | `[M]` | 6,09 px (1023×682) · 8,93 (1500×1000) · 9,53 (1600×1067) · 12,19 (2048×1365) · 17,86 (3000×2000) · **30,86 (5184×3456)**; FOV perdida 1,47% em y | idem | `ACHADOS.md:495-500` |
| 11.5 | o gate antigo era `--max-pair-shift-px 6.0` | `[M]` | 30,86 px é **5,1×** o limiar | leitura de `bokehnet-preprocessing/.../route_b.py:411-422` — `[X]`, arquivo fora deste repositório | `ACHADOS.md:504-505` |
| 11.6 | o deslocamento é **zero em 4:3 e 16:9** e cresce com a resolução em 3:2 | `[M]` | — | 11.3 + 11.4 | `ACHADOS.md:506-508` |
| 11.7 | logo é erro correlacionado com a proporção da foto | `[I]` | — | juízo declarado | `ACHADOS.md:506-508` |
| 11.8 | a deformação que a **rede** introduz segue **não medida** | `[A]` A13 | — | declarado | `ACHADOS.md:516-517`, `ROTA_B_AUDITORIA.md:961` |

---

## 12. Custo e desempenho medidos localmente

| # | afirmação | etiqueta | valor | origem | registrada em |
|---|---|---|---|---|---|
| 12.1 | `detail_retention` a 1500×2000 / 384×512 | `[M]` | **291 ms** / **22 ms** | CPU, numpy 2.2.6, máquina local, mediana de 3 repetições, imagem sintética | `ACHADOS.md:529-534` |
| 12.2 | `resize_area_for_photo` de uma foto de 3 MP | `[M]` | 185 ms | idem | `ACHADOS.md:535` |
| 12.3 | `refine_focus_region` ponta a ponta | `[M]` | **391 ms** cheio vs **433 ms** reduzido | idem | `ACHADOS.md:536` |
| 12.4 | logo a resolução de trabalho **não** compensa a 1500×2000 | `[I]` | 2×185 = 370 ms para economizar 269 ms | aritmética; refeita, confere | `ACHADOS.md:538-543` |
| 12.5 | custo total da rota C por amostra, com dublê barato em CPU | `[M]` | **~4,2 s**, dos quais ~0,39 s de refinamento | idem | `ACHADOS.md:545-547` |
| 12.6 | **custo por amostra com GPU e BokehMe real** | **`[SP]` / não medido** | — | é o item dominante do orçamento e **não tem estimativa em lugar nenhum** | `ROTA_A_AUDITORIA.md:561-565`, `ROTA_C_AUDITORIA.md:1028-1030` |
| 12.7 | o comando que produziu 12.1–12.5 | — | — | **`[SP]`**: nenhum script de benchmark está versionado; a medição é reprodutível em princípio (as funções são públicas), não em prática (não há harness) | — |

### A inconsistência aritmética do "512x683"

Quatro lugares afirmam: *"a janela de 33 px cobre **1,7%** do lado longo a 1500×2000 e
**6,4%** a 512×683"* — `ACHADOS.md:550`, `REGISTRO.md:1023`, `src/routes/route_c.py`
(docstring de `refine_focus_region`) e `src/dataio/sample.py` (comentário de
`retention_hw`).

Refazendo a conta:

| grade | lado longo | 33 px / lado longo |
|---|---|---|
| 1500×2000 | 2000 | 33/2000 = **1,65%** ✓ bate com "1,7%" |
| 512×683 | **683** | 33/683 = **4,83%** ✗ não bate com "6,4%" |
| **384×512** | **512** | 33/512 = **6,45%** ✓ bate com "6,4%" |

E `_work_hw((1500,2000), 512)` devolve **(384, 512)** — não (512, 683). A própria
`ACHADOS.md:532`, na tabela de tempos, mede a **384×512**.

**Veredito:** o número 6,4% está certo; o rótulo `512x683` está errado em quatro lugares,
e contradiz `ACHADOS.md:532` **dentro do mesmo arquivo**. Efeito no rótulo: nenhum — é
texto. Efeito numa revisão: o leitor calcula 33/683 e conclui que a conta está errada.

---

## 13. Documentação que contradiz uma medição do próprio repositório

Todas verificadas em 2026-09-10 19h. **Nenhuma afeta o rótulo**; todas afetam a
credibilidade de quem lê.

| # | afirmação obsoleta | onde ainda está | o que a medição diz | evidência |
|---|---|---|---|---|
| 13.1 | "o espelho publica só `train` (20.495 de 20.495)" | `scripts/run_route_c.py`, docstring de `_monta_split`; `src/sources/realbokeh.py` (final do arquivo) | três splits, 22.990 linhas, 4.399 cenas | `ACHADOS.md:421-435` |
| 13.2 | "85 shards parquet" | `scripts/run_route_c.py`, help de `--mirror-dir` | **96** shards | idem |
| 13.3 | "**48 testes** passando" | `README.md:115`; `slurm/verify_renderer.slurm` | **650** em 2026-09-10 19:31 | esta auditoria |
| 13.4 | "14 dos 33 defeitos fechados com teste, 3 no contrato, 16 abertos" | `README.md:141` | 16 / 3 / 14 na etapa 4 | `REGISTRO.md:681-685` |
| 13.5 | "Etapas 1 e 2 concluídas… 1.386 linhas em `src/`, 640 de teste" | `README.md:115` | **9.297** linhas em `src/`, **7.827** em `tests/`, 2.694 em `scripts/` | `wc -l`, esta auditoria |
| 13.6 | "TODOS os **dez** gates em modo medir" | `slurm/route_c_pilot.slurm`, cabeçalho | **onze** limiares em `RouteCConfig` (e `scripts/run_route_c.py` já diz "onze") | contagem dos campos |
| 13.7 | "o catálogo dos **32** defeitos" | `.claude/agents/defect-regression.md:3` e `:17` | **33**, corrigido explicitamente: *"Contagem exata: 33, não 32"* | `REGISTRO.md:152` |
| 13.8 | "`src/model_runtime/deblurnet.py` — **NÃO EXISTE** — é a maior lacuna" | `ROTA_B_AUDITORIA.md:637` | existe, 967 linhas, 65 testes | `wc -l`, esta auditoria |
| 13.9 | "**239 testes** passando" (4 ocorrências) | `ROTA_C_AUDITORIA.md:25`, `:523`, `:845`, `:1033` | 521 | esta auditoria |
| 13.10 | "busca ternária" | `CLAUDE.md`, seção Layout | é **seção áurea**; já corrigido em `REGISTRO.md:321-330` | `src/renderer/calibration.py` |
| 13.11 | "32 de 100 amostras conferidas da rota B são retrato" | `src/control/contract.py`, docstring de `pixel_ratio` | 32 de **900** (3,6%) | `ACHADOS.md:64`; já apontado em `ROTA_B_AUDITORIA.md:934` e **não corrigido** |
| 13.12 | rótulo `512x683` | 4 lugares (§12) | a grade é 384×512 | `ACHADOS.md:532` |
| 13.13 | "O espelho são **85 shards** parquet" | `src/sources/mirror_images.py`, cabeçalho | 96 shards | `ACHADOS.md:423,431` |
| 13.14 | "20.495 pares", "41.000 imagens" como se fosse o espelho inteiro | `src/sources/mirror_images.py`, cabeçalho | 22.990 linhas, ~46.000 imagens | idem |
| 13.15 | todos os números de `src/sources/realbokeh.py` são do split `train` e o arquivo não sinaliza isso | cabeçalho do arquivo e `enumeration_summary` | 3.959 cenas e 20.495 linhas são de `train`; o espelho tem 4.399 cenas | idem |
| 13.16 | "slugs NOVOS — precisam ser registrados em `control.contract`" / "`contract.py` não foi editado" | `src/sources/lfdof.py` (dois blocos) e `src/sources/realbokeh.py` (um bloco) | os 8 slugs **já estão** registrados em `control/contract.py`; o mesmo arquivo diz as duas coisas em blocos diferentes, e `PENDING_REJECTION_REASONS` ficou como alias residual reexportado por `sources/__init__.py` | leitura de `contract.py` |
| 13.17 | "`deblur_variant` gravado por amostra e **conferido na publicação**" | `PLANO_EXECUCAO.md:104` | `scripts/publish_release.py` **não tem nenhuma** checagem de `deblur_variant`; a pendência está registrada em `REGISTRO.md:953` e `ROTA_B_AUDITORIA.md:899` | `grep deblur_variant scripts/publish_release.py` → vazio |
| 13.18 | o `--help` de `--mirror-dir` diz "(85 shards parquet)" | `scripts/run_route_c.py` | 96 | idem a 13.2 |

---

## 13-bis. Proveniência e gates que afirmam mais do que mediram

Três casos do padrão que o projeto inteiro persegue — *"gate que não pode reprovar"* e
*"campo de proveniência que afirma sem medir"* — encontrados **no código novo**, em
2026-09-10 19h. Nenhum deles é novo como classe; todos os três estão nomeados em
`REGISTRO.md:694-699` como o padrão a evitar.

| # | achado | evidência | por que importa |
|---|---|---|---|
| B1 | **`is_final_label_renderer: True` é literal na proveniência de cada amostra.** `BokehMeRenderer.provenance()` grava a chave sem consultar laudo nenhum. | `src/renderer/bokehme.py`, `provenance()` | Contradiz frontalmente o texto de `scripts/verify_renderer.py`: *"Enquanto não passar, `is_final_label_renderer` não vale nada — é declaração, não medição."* Quem barra o renderer não verificado é o **entrypoint** (`run_route_c.py` recusa rodar se o laudo não passou e compara 3 hashes), o que funciona — mas o campo que **viaja dentro do dado** é declarativo. Um release remontado de um run que burlasse o entrypoint carregaria `True` sem laudo. |
| B2 | **`highlight_decidido` nunca pode ser `False`.** `report["passed"]["highlight_decidido"] = True`, atribuição incondicional, e entra no `all(...)` que decide `is_final_label_renderer`. | `scripts/verify_renderer.py:125` | O laudo anuncia 4 critérios e tem 3 medidos. `src/renderer/verification.py` **já admite isso** no cabeçalho: o teste 3 é *"registrada no config, não medida aqui"*. Então o campo pertence a `measurements`, não a `passed`. Já apontado em `ROTA_C_AUDITORIA.md:691-706`, ainda aberto. |
| B3 | **`--slope-tolerance` default 0,05 afrouxa o contrato do módulo, que é 0,02.** | `scripts/verify_renderer.py` (CLI) contra `RadiusResponse.scale_matches_contract(tolerance=0.02)` em `src/renderer/verification.py` | O laudo em `output/renderer_verification.json` passou com `slope = 0,9873`, ou seja 1,27% — **passaria nos dois**. Mas o número que autorizou o renderer foi julgado contra 5%, não contra os 2% que o módulo declara como contrato. A diferença nunca foi exercida; a assimetria não está declarada. |
| B4 | **`verify_renderer.py` instancia 4 renderers** (um fora do laço e um por saída testada), contra a promessa de "modelo carregado uma vez" do adaptador. | `scripts/verify_renderer.py` | Não afeta o número medido; afeta o custo do laudo e a afirmação de arquitetura. |
| B5 | **O card do release traz 35,2% e 20,6% *hardcoded*.** `escreve_card` escreve os números do piloto 32224 em qualquer release, e `tests/test_publish_release.py` **exige** a string `"35,2%"` no card. | `scripts/publish_release.py` | O card é o documento público do dataset. Publicar um lote cujo refinamento mediu outra coisa e ainda dizer 35,2% é proveniência que mente na direção tranquilizadora — a mesma classe do `mask_source="automatic"`. Conserto: computar do próprio release, como o card já faz com as outras contagens. |
| B6 | **O validador de release afirma no card duas coisas que ele não verifica.** O card diz que `rejections.jsonl` "faz parte do release" e que "o renderer foi verificado em GPU antes de gerar rótulo … sem esse laudo o pipeline recusa rodar" — e `valida()` não checa a existência de `rejections.jsonl` nem de nenhum artefato de laudo. | `scripts/publish_release.py` | As duas afirmações são verdadeiras na prática (o `RejectionLog` grava; o entrypoint recusa), mas o release não as **prova**. |

**Contraponto justo, e vale registrar:** das 11 checagens que reprovam um release,
**todas as 11 têm teste que as faz reprovar** — verificado contra
`tests/test_publish_release.py`. Isso é exatamente a lição do `check_no_leak` antigo,
aplicada. As quatro que só avisam também têm teste. O módulo é o melhor exemplo do método
no repositório; os itens B5 e B6 são falhas de conteúdo do card, não de estrutura da
validação.

---

## 14. Suíte de testes — medição desta auditoria

Comando exato, executado três vezes:

```bash
cd "/Users/juliadollis/Projects_Code/repositorio_ref - cópia 5/bokehnet-regen"
PYTHONPATH=src:scripts:tests \
  /Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python \
  -m unittest discover -s tests -p "test_*.py"
```

| # | afirmação | etiqueta | valor | quando |
|---|---|---|---|---|
| 14.1 | suíte completa | `[M]` | `Ran 519 tests` — `OK (skipped=7)` | 2026-09-10 19:07:39 −03 |
| 14.2 | suíte completa, 2ª execução | `[M]` | `Ran 521 tests` — `OK (skipped=7)` | 2026-09-10 19:11:37 −03 |
| 14.3 | suíte completa, 3ª execução — **é este o número a citar** | `[M]` | **`Ran 650 tests in 6,584s` — `OK (skipped=7)`** | 2026-09-10 19:31:45 −03 |
| 14.4 | **zero falhas e zero erros** nas três execuções | `[M]` | — | idem |
| 14.5 | testes por arquivo (na execução de 650) | `[M]` | `test_lfdof` 112 · `test_route_b` 104 · `test_sources` 73 · `test_deblurnet` 65 · `test_dataio` 45 · `test_route_c` 37 · `test_contract` 33 · `test_gates` 33 · `test_validate_focus_refinement` 26 · `test_focus_region` 24 · `test_level_selection` 23 · `test_publish_release` 23 · `test_mirror_images` 20 · `test_model_runtime` 15 · `test_renderer` 12 · `test_rejection` 5 | 19:32:05 (soma = 650 ✓) |
| 14.6 | motivo dos 7 skips | `[M]` | **4** por `pyarrow não instalado nesta máquina` (3 em `test_mirror_images`, 1 em `test_lfdof`); **3** por `toca a rede; ligue com BOKEHNET_HF_TESTS=1` (2 em `test_mirror_images`, 1 em `test_sources`) | idem |
| 14.7 | `REGISTRO.md:1155` afirma **518** testes, `OK (skipped=7)` | `[M]` na afirmação | 518 | a afirmação era verdadeira quando escrita; a suíte cresceu depois. **Não é erro** — é o custo de contar testes em prosa |
| 14.8 | linhas de código, na medição de 19:32 | `[M]` | `src/` **9.297** · `tests/` **7.827** · `scripts/` **2.694** | 2026-09-10 19:32:05 −03 |

**Nota de honestidade sobre o enunciado da tarefa.** Fui avisado de que ~20 testes
estariam quebrados neste momento, porque outro agente editava `src/dataio/sample.py`.
**Não observei nenhuma falha**: as três execuções deram `OK`. Reporto o que medi. É
plausível que a janela de quebra tenha fechado antes das 19:07, e o número de testes
mudando entre execuções mostra que o arquivo estava mesmo sendo editado.

---

## 15. Constantes, limiares e valores `[A]` no código

Contagem por varredura: **25 ocorrências de `[A]`, 10 de `[M]` e 1 de `[I]`** em `src/`
(`grep -rho "\[A\]\|\[M\]\|\[I\]" src/ | sort | uniq -c`, 2026-09-10 19h).

### 15.1 Constantes congeladas com justificativa

| constante | valor | etiqueta | origem | onde |
|---|---|---|---|---|
| `CONTROL_VERSION` | `metric_disparity_official_v1` | — | decisão nossa | `control/contract.py` |
| `MAX_COC` | **100,0** | `[X]` | `Inference_bokehNet.py:20` — arquivo fora deste repositório | `control/contract.py` |
| `MM_PER_M` | 1000,0 | `[M]` dimensional | análise dimensional; travada por teste | idem |
| `FULL_FRAME_WIDTH_MM` | 36,0 | `[M]` físico | usada **só** com crop factor medido | idem |
| `UINT16_MAX` | 65535 | — | definição | idem |
| `DEPTH_LONG_SIDE` | 768 | **decisão nossa declarada** | orçamento de disco (§17) | `dataio/encoding.py` |
| `GAUSSIAN_EDGE_RATIO` | 1,43 | `[I]` derivado analiticamente | 9.8, refeito e confere | `renderer/verification.py` |
| `LFDOF_IMAGE_HW` | (688, 1008) | `[M]` afirmado, **`[SP]`** na medição | — | `sources/lfdof.py:304` |
| `MIRROR_IMAGE_HW` | (1500, 2000) | `[M]` | 6.7 (2000×1500 confirmado por sha256) | `sources/realbokeh.py` |

### 15.2 Limiares `[A]` — assumidos, declarados, e ainda sem número medido

| limiar | valor | risco declarado | pode bloquear hoje? |
|---|---|---|---|
| `FOCUS_DEPTH_MIN_M` | 0,05 | baixo | **SIM** — ver 15.4 |
| `FOCUS_DEPTH_MAX_M` | 1000,0 | **médio** — a faixa medida de `z_focus_m` vai a 10.000 | **SIM** |
| `MIN_DEPTH_RANGE_RATIO` | 1,02 | baixo, permissivo de propósito | sim |
| `K_MIN_DEFAULT` | 0,5 | — | sim (define a banda de censura no piso) |
| `K_MAX_DEFAULT` | 120,0 | — | não (expande) |
| `K_ABSOLUTE_MAX_DEFAULT` | **960,0** | **alto** — 10× acima da faixa verificada do renderer (K ≤ 96, 9.12) e ~27× acima do que a Eq. 3 prevê | sim (parada dura) |
| `tolerance` do sweep | 0,25 | — | sim (escala da censura) |
| `max_evaluations` do sweep | 40 | — | sim (rejeita ao esgotar) |
| `coarse_points` do sweep | 7 | — | sim |
| `DEFAULT_WINDOW_PX` | 33 | — | não (parâmetro) |
| `DEFAULT_TOP_FRACTION` | 0,05 | — | não |
| `MIN_REGION_AREA_RATIO` | 0,001 | — | sim (`focus_mask_empty`) |
| `DEFAULT_AGREEMENT_FLOOR` | 0,30 | — | não (escolhe o ramo) |
| os 11 limiares de gate de `RouteCConfig` | **todos `None`** | por definição | **não** — é a regra do módulo |

### 15.3 Slugs de rejeição — vocabulário fechado

`REJECTION_REASONS` = `GATE_REJECTION_REASONS` (12 slugs) ∪ `SOURCE_REJECTION_REASONS`
(10 slugs) ∪ 20 slugs próprios do contrato. `reject()` levanta `KeyError` para slug não
registrado, e usa `if`, não `assert` — porque `python -O` desliga `assert`. **Verificado
lendo `src/control/contract.py`.**

### 15.4 Itens abertos que a auditoria da rota C levantou e que ainda estão de pé

Reverificados um a um em 2026-09-10 19h, no código:

| item da `ROTA_C_AUDITORIA.md` | estado hoje | evidência |
|---|---|---|
| **C1** — `k_analytic` não é comparável com `K*` | **parcialmente fechado**: `RouteCStats.summary` agora imprime os dois separados, com a explicação de que um é absoluto e o outro incremental. **Não** existe campo `k_analytic_incremental` em `ControlLabel` | leitura de `routes/route_c.py` e `dataio/sample.py` |
| **C2** — `aif_f_number=None` reprovava 100% do LFDOF | **FECHADO**: `GateResult.applicable` existe, `aif_aperture_is_narrow(None)` devolve `applicable=False`, e `mask_iou`/`focus_mask_is_sharpest` também | `qc/gates.py` |
| **C3** — multiplicidade de até 21 alvos por cena | **FECHADO entre 19:11 e 19:32**: `src/sources/level_selection.py` (136 linhas, 23 testes) corta **níveis, não cenas**, e `scripts/run_route_c.py` ganhou `--max-levels-per-scene` com **default 4** | `grep -n max_levels_per_scene scripts/run_route_c.py` |
| **C4** — `K_ABSOLUTE_MAX = 960` | **ABERTO**: valor inalterado | `renderer/calibration.py` |
| **C5** — `FOCUS_DEPTH_MIN_M`/`MAX_M` bloqueiam por default e o piloto não os calibra | **ABERTO**: não há flag `--focus-depth-min-m`/`--max-m` em `scripts/run_route_c.py`; `focus_disparity_from_mask` é chamada sem os parâmetros; `_GATES` de `calibrate_thresholds.py` tem 10 entradas e não inclui `focus_depth_m_min/max` | grep, esta auditoria |
| #5 — `min_aif_f_number` ausente de `_GATES` | **ABERTO**: 11 limiares em `RouteCConfig`, 10 entradas em `_GATES` | idem |
| #8 — gates baratos **antes** do sweep | **ABERTO**: `process_pair` chama `calibrate_k` antes de `enforce_gates` | `routes/route_c.py` |
| #9 — slug errado no estouro de orçamento | **ABERTO**: `calibration.py` levanta `k_out_of_configured_range`; `k_search_budget_exhausted` está registrado em `contract.py` e **não é usado em nenhum lugar de `src/`, `scripts/` ou `tests/`** | `grep -rn k_search_budget_exhausted` |
| #10 — `k_min = 0` aceito pela calibração | **ABERTO**: `if not (0 <= k_min < k_max <= k_absolute_max)` | `renderer/calibration.py` |
| #12 — censura no piso pode alcançar dado legítimo | **ABERTO**: `at_lower = k_star <= k_min + edge` | idem |
| #14 — `highlight_decidido` não pode reprovar | **ABERTO**: `report["passed"]["highlight_decidido"] = True` atribuído incondicionalmente, e entra no `all(...)` que autoriza `is_final_label_renderer` | `scripts/verify_renderer.py:125` |
| #15 — `produces_target_image: False` | **ABERTO**: a chave não existe em nenhum arquivo | `grep -rn produces_target_image` → vazio |
| P3 #13 — documentação obsoleta do espelho | **ABERTO** | 13.1, 13.2 |

---

## 16. Onde a rastreabilidade falha por construção

Três problemas de método que aparecem em todo o repositório e que valem mais que
qualquer linha individual desta tabela.

### 16.1 Citações `arquivo:linha` para arquivos vivos apodrecem em silêncio

Amostra medida nesta auditoria, comparando o que a auditoria da rota C cita com o que
está lá hoje:

| citação | o que a auditoria afirma estar lá | o que está lá em 19h |
|---|---|---|
| `contract.py:243-244` | `FOCUS_DEPTH_MIN_M`/`MAX_M` | `_reject("depth_non_positive", …)`. As constantes estão em **264-265** |
| `contract.py:433-451` | `defocus_map` | `signed_coc_px`. `defocus_map` está em **454-472** |
| `contract.py:317-332` | `pixel_ratio` | corpo de `sensor_width_mm`. `pixel_ratio` está em **338-353** |
| `contract.py:458-482` | `k_at_resolution` | fim de `defocus_map`. Está em **479-503** |
| `contract.py:247-284` | `focus_disparity_from_mask` | fim de `validate_metric_depth`. Está em **268-305** |
| `contract.py:142` | slug `k_search_budget_exhausted` | `"depth_non_positive"`. O slug está em **156** |
| `sample.py:229-233` | `validate_metadata` rejeitando `max_coc` | corpo de `FocusRegionRecord.from_region`. Está em **391-395** |
| `sample.py:104-125` | `ControlLabel` | campos de `SampleProvenance`. `ControlLabel` está em **117-138** |
| `sample.py:156-157` | `is_valid_for_control` | comentário de `_REFINED_MASK_SOURCES`. Está em **300-301** |
| `calibration.py:135` | validação de `k_min` | `work_long_side`. Está em **146** |
| `calibration.py:158` | slug do orçamento | linha vazia. Está em **169** |
| `calibration.py:206-208` | cálculo de `at_lower` | comentário. Está em **217-219** |
| `gates.py:54-59` | `GateResult.passed` | campos da dataclass. `passed` está em **63-70** |
| `gates.py:305-307` | `aif_aperture_is_narrow` | `focus_depth_plausible`. Está em **349-380** |
| `route_c.py:150` | a âncora impressa | comentário de `min_focus_region_retention`. O resumo está em **~256-272** |
| `route_c.py:246` / `:261` | `calibrate_k` / `enforce_gates` | outro código. Estão em **~522** e **~540** |

**Nenhuma dessas citações estava errada quando foi escrita.** Todas estão erradas agora.
Como as constantes citadas ainda existem com o mesmo nome, o conteúdo da auditoria
continua válido — mas um avaliador que confira as linhas conclui que a auditoria é
descuidada, e não é.

**Recomendação para o paper e para o repositório:** citar `arquivo::símbolo` para código
próprio, e reservar `arquivo:linha` para arquivos **imutáveis** (`reference/paper.txt`,
checkouts de terceiros com commit congelado).

### 16.2 Catorze afirmações têm procedência fora deste repositório

Um avaliador que receba só `bokehnet-regen/` não consegue conferir nenhuma delas.

| # | afirmação | onde a evidência vive |
|---|---|---|
| X1 | `MAX_COC = 100.0` | `Genfocus/Inference_bokehNet.py:20` |
| X2 | default `k = 15,0` da inferência oficial | `Inference_bokehNet.py:53` |
| X3 | `disp = 1.0/depth` e `disp_focus = median(valid_disp)` | `Inference_bokehNet.py:94`, `:118` |
| X4 | `No_preprocess=True` na inferência | `Inference_bokehNet.py` |
| X5 | defaults da DeblurNet oficial (`adapter_name`, prompt, 28 steps, `long_side=0`, `NO_TILED_DENOISE`) | `Inference_deblurNet.py:11,30-57,88-111` |
| X6 | a aritmética de recorte reimplementada em 11.1–11.4 | `Inference_deblurNet.py:13-49`, `:215-218` |
| X7 | `generate(main_adapter=None)` como default | `Genfocus/pipeline/flux.py:485`, `:783` |
| X8 | a variante `main+cond` exige `main_adapter="deblurring"`, senão sai **lavada** | `../HANDOFF_PROJECT_HISTORY.md:102,109,161,171` |
| X9 | o pipeline de avaliação passa `main_adapter` explicitamente | `../deblurnet-eval-pipeline/infer_and_eval.py:98-101,137` |
| X10 | **o catálogo dos 33 defeitos** | `../genrefocus_deblurnet_paper/PLANO_REGERACAO_BOKEHNET.txt` |
| X11 | todo o diagnóstico do pipeline antigo (rotas A, B, C) | `../bokehnet-preprocessing/src/**` |
| X12 | "o dataset de ~68K não saiu deste código" | `../bokehnet-preprocessing/docs/ERROS_GERACAO_ORIGINAL_BOKEHNET.md:193` |
| X13 | o alvo gaussiano de 16 camadas com kernel de 51 px | idem, `:174-189`; `docs/VEREDITO_FINAL_BOKEHNET.md:128-130` |
| X14 | as regras do cluster (GPU4 defeituosa, QOS `onejob`, `/raid` vs `/home`) | `../INSTRUCOES_H100.md` |

**X10 é o mais sério.** Todo o placar de defeitos — "16 fechados com teste, 3 no
contrato, 14 abertos", que aparece em cinco lugares e é o principal indicador de
progresso do projeto — pende de um arquivo `.txt` que não está aqui. O próprio
repositório já discorda de si mesmo sobre o total (33 em `REGISTRO.md`, 32 em
`.claude/agents/defect-regression.md`). Para o paper: ou o catálogo entra em `reference/`,
ou o placar não é citável.

### 16.3 Nenhum log de execução em GPU está versionado

`logs/` e `output/` estão no `.gitignore`. O único artefato de execução que sobreviveu é
`output/renderer_verification.json`, e ele está no repositório **apesar** do
`.gitignore`, não por causa dele.

Consequência: dos três jobs que produziram medição (**32212**, **32224**, **32231**), só
o primeiro deixou artefato. Os números do piloto (§10) existem **só como prosa** em
`MEDICAO_PLANO_FOCO.md`. Um avaliador não pode conferir 35,2% nem 20,6% contra nada.

Conserto barato e de alto valor: versionar, por job, o `run_config.json`, o
`rejections.jsonl` agregado e o resumo de stdout — sem os pixels.

---

## 17. Orçamento de disco — a família de números sem procedência

Cinco números governam a decisão de armazenamento mais consequente do projeto
(*"guardamos o que geramos, referenciamos o que já existe"*), e **nenhum tem medição
registrada**.

| # | afirmação | onde aparece | procedência |
|---|---|---|---|
| 17.1 | "**115 GB** de cota livre" | `REGISTRO.md:581`, `src/dataio/sample.py:4`, `src/routes/route_c.py:19`, `src/dataio/encoding.py:31` e `:52` | **`[SP]`** — nenhuma saída de `quota`/`df` em lugar nenhum |
| 17.2 | regravar as imagens de origem custaria "**365 GB**" | `REGISTRO.md:581`, `sample.py:4`, `route_c.py:19` | **`[SP]`** — sem derivação |
| 17.3 | profundidade em resolução cheia custaria "**207,8 GB** só de controle" | `encoding.py:31` | **`[SP]`** — derivável de `estimate_disk_budget`, mas a derivação não está gravada |
| 17.4 | rotas B+C somam "**31 GB**" | `REGISTRO.md:585`, `run_route_c.py`, `publish_release.py` | **`[SP]`** — `REGISTRO.md:798` chama de "custo medido por estimativa" |
| 17.5 | release autocontido "~**49 GB**" | `REGISTRO.md:798`, `run_route_c.py` | **`[SP]`** — idem |
| 17.6 | "a cota é **500 GB soft, 600 GB hard**" | `src/dataio/writer.py:228` | **`[SP]`**, e **contradiz 17.1**: 500 GB de cota não é 115 GB de folga sem uma medição de ocupação que não existe. `.claude/agents/runtime-smoke.md:75` diz "500G soft" e manda rodar `quota -s` — nunca rodado e registrado |
| 17.7 | `/raid` livre: "**7,0 TB**" | `REGISTRO.md:876` | `[M]` afirmado (conferência do cluster na etapa 7), sem comando registrado |
| 17.8 | erro de quantização da profundidade a 768 | "**0,000378 px** com K=50" — `REGISTRO.md:585`, `CONTRATO.md:132` | **`[M]`, e eu refiz** — ver 17.10 |
| 17.9 | fator do crop de treino | 0,892 (1024×574) e 0,821 (624×1024) | `[M]`, `ACHADOS.md:65-66` — **e nenhum código o aplicava** |

### 17.10 O único número de orçamento que eu consegui refazer — e ele aparece duas vezes com valores diferentes

Refeito nesta auditoria, com o código do repositório:

```bash
PYTHONPATH=src /Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python -c "
import numpy as np
from dataio.encoding import encode_depth, quantization_coc_error_px
z = np.linspace(1.0, 100.0, 256).reshape(16,16).astype(np.float32)
e = encode_depth(z, image_hw=(16,16), long_side=768)
print(quantization_coc_error_px(e, 50.0))"
# -> 0.0003776607919432364
```

Confere com os **0,000378 px** de `CONTRATO.md:132` e `REGISTRO.md:585`. **Sai de `[SP]`
para `[M]` reprodutível** — é o único número da §17 nessa condição.

Mas a docstring de `src/dataio/encoding.py` diz, para a **mesma** grandeza e o mesmo caso
(`z ∈ [1, 100] m`, `K = 50`): *"`erro_coc = K * passo` … isso dá **7,5e-4 px**"*. Refeito:
`step = 1,5106e-5`, `K·step = 7,553e-4`, `K·step·0,5 = 3,777e-4`.

**As duas estão certas e diferem por exatamente 2×**: a docstring reporta o **passo
inteiro**, e `quantization_coc_error_px` computa o **meio passo** (`* 0.5`, o erro máximo
de arredondamento). Nenhum dos dois lugares diz qual convenção usa. Para o paper: escolher
uma, dizer qual, e citar o comando.

**17.1 contra 17.7 é a contradição operacional mais visível do repositório:** a decisão de
não gravar pixels foi tomada contra "115 GB de folga", e a conferência do cluster de duas
etapas depois mediu **7,0 TB livres** em `/raid`, concluindo que "o release autocontido
cabe com folga". As duas frases falam de sistemas de arquivos diferentes (cota
compartilhada vs disco local do nó), mas **nenhuma das duas diz de qual**. Para o paper,
a decisão de armazenamento precisa de uma frase que diga qual sistema, qual comando, e
qual data.

---

## 18. Contagem de defeitos ao longo das etapas

| etapa | fechados com teste | fechados no contrato | abertos | total | onde |
|---|---|---|---|---|---|
| 1 | 10 | 3 | 20 | 33 | `REGISTRO.md:154-190` |
| 2 | 13 | 3 | 17 | 33 | `REGISTRO.md:384-388` |
| 3 | 14 | 3 | 16 | 33 | `REGISTRO.md:548-552` |
| 4 | **16** | 3 | **14** | 33 | `REGISTRO.md:681-685` |
| 5 a 9 | — | — | — | — | **não recontado** |

**Duas observações.** (a) A soma fecha em 33 nas quatro etapas — consistência interna
verificada. (b) O placar **para na etapa 4**: as etapas 5 a 9 acrescentaram ~5.500 linhas
de `src/` e ~280 testes sem atualizar a contagem, e `README.md:141` congelou o número da
etapa 3. Como o catálogo é `[X]` (16.2), ninguém pode recontar a partir deste
repositório.

---

## 19. Código que existe e não está em nenhuma etapa do `REGISTRO.md`

Medido comparando o índice de etapas do `REGISTRO.md` (1 a 9) com o conteúdo de `src/` e
`tests/` em 2026-09-10 19h.

| arquivo | linhas | testes | mencionado em qual etapa? |
|---|---|---|---|
| `src/model_runtime/deblurnet.py` | 967 | 65 (`test_deblurnet.py`) | **nenhuma**. Há medição em `ACHADOS.md:473-523`, e `ROTA_B_AUDITORIA.md:637` ainda diz que o arquivo "NÃO EXISTE" |
| `src/sources/lfdof.py` | 879 | 112 (`test_lfdof.py`) | **nenhuma**. Nem etapa no `REGISTRO.md`, nem seção de medição em `ACHADOS.md` — e ele carrega **28 medições próprias** (§8.9) |
| `src/sources/lfdof_images.py` | 403 | (idem) | **nenhuma** |
| `src/routes/route_b.py` | 1.168 | 104 (`test_route_b.py`) | **nenhuma** — apareceu entre 19:11 e 19:32 |
| `scripts/run_route_b.py` | 623 | (idem) | **nenhuma** — idem |
| `src/sources/level_selection.py` | 136 | 23 (`test_level_selection.py`) | **nenhuma** — idem |
| `src/model_runtime/depth.py` | 143 | 15 (`test_model_runtime.py`) | citado de passagem na etapa 8 |
| `src/model_runtime/segmentation.py` | 176 | (idem) | idem |

Somados, **4.176 linhas de `src/` + `scripts/` e 304 testes** não têm registro de decisão
no `REGISTRO.md`. A etapa 9 foi escrita (`REGISTRO.md:971-1155`) e é detalhada; ela cobre o
refinamento da região em foco, e não estes arquivos.

Isto é consistente com haver etapas em redação por outros agentes — os três últimos itens
apareceram **durante esta auditoria**. Mas, no estado de 19h35, é código de produção sem
justificativa registrada, que é exatamente o que a regra de abertura do `REGISTRO.md`
proíbe: *"nada entra sem justificativa"*. **Reverifique esta seção antes de citá-la; ela
tem alta probabilidade de estar desatualizada por construção.**

**E há uma consequência funcional:** `src/sources/lfdof.py` existe e enumera pares, mas
`scripts/run_route_c.py::_carrega_fonte` só trata `"realbokeh"` — `--source lfdof` ainda
levanta `SystemExit` com a mensagem "fonte ainda não tem adaptador", que hoje é **falsa**.
O adaptador existe; o que falta é ligá-lo.

---

## 20. Observação sobre o único lugar onde `K*` e `k_analytic` aparecem juntos

Rodando a suíte, o resumo da rota C imprime (fixture de teste, não medição):

```
k_value    p05 18.01  mediana 18.01  p95 18.01
k_analytic p05  1.63  mediana  1.63  p95  1.63
```

E, duas linhas abaixo, o próprio resumo diz: *"Espera-se `k_value` < `k_analytic`."*

O fixture faz `k_value ≈ 11 × k_analytic` — o oposto. **Isto não refuta a física**: a
fixture usa uma imagem de lado longo 96 px com um renderer sintético de 7 camadas que
impõe K=18 por construção, enquanto `k_analytic` é a Eq. 3 com `f=49 mm`, `F=2,0`,
`z=3,0 m`, `sensor=36 mm` naquela mesma imagem minúscula (`pixel_ratio = 96/36 = 2,67
px/mm` ⇒ `K = 1,63`). São dois números que não descrevem a mesma cena.

Mas é o **único** lugar do repositório em que os dois são impressos lado a lado, e
qualquer avaliador que rode a suíte vai vê-los contradizendo a legenda logo abaixo.
Conserto de meia linha: dar ao fixture uma resolução e um K coerentes, ou não computar
`k_analytic` nele.

---

## 21. Como usar esta tabela para escrever o paper

Regra prática, em três linhas:

1. **`[M]` com "reproduzível: sim"** → escreva o número, cite o comando.
2. **`[I]`, `[A]`, `[P]`** → escreva o número **com a etiqueta em voz alta**: "inferimos",
   "assumimos", "o paper reporta".
3. **`[SP]` ou `[X]`** → **não escreva**. Meça primeiro, ou traga a evidência para dentro
   do repositório.

Pelo estado de 2026-09-10 19h, a regra 3 elimina do texto: a tabela de LVCorr inteira
(§2), a âncora de K de 20,1 (4.2), todos os números de orçamento de disco (§17), o custo
de GPU por amostra (12.6), e o placar de defeitos (§18) enquanto o catálogo não estiver
aqui.

Isso não é pouco — mas é exatamente o conjunto de números que um revisor pediria para
ver, e é melhor descobrir agora.
