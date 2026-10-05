# Rota A — vigilância do job 32695 (`dgx-H100-03`)

Relatório vivo. Atualizado enquanto o job roda. Todas as medições são `[M]` medidas,
somente leitura, com os scripts em `$P/.cache_raid/medir_rota_a.py`, `medir2.py`,
`medir3.py` (enviados por `rsync`, nada editado no cluster).

`$P = /raid/user_danielpedrozo/projects/julia/bokehnet-regen`

---

## 1. Estado do job

| carimbo | estado | amostras | ritmo | ETA |
|---|---|---|---|---|
| 23:30 (elapsed 00:10) | RUNNING | 7.168 | — | — |
| 23:36 (elapsed 00:16) | RUNNING | 8.800 (12,6%) | 16.722/h | 3,6 h |
| 23:42 (elapsed 00:21) | RUNNING | 10.120 (14,5%) | 16.755/h | 3,6 h |

- Início: `2026-09-16T23:20:34`, `--time=20-00:00:00` (folga enorme, ok).
- Retomada: **5.371 amostras** do run anterior (2.829 em s0 + 2.542 em s1), puladas por
  `sample_id`. O total cresce a partir daí — confirmado.
- Alvo: 34.235 (s0) + 35.465 (s1) = **69.700** amostras, 1.700 cenas × 41 variantes.
- Busca por `No space left|Errno 28|Traceback|ModuleNotFound|CUDA out of memory|Killed`
  nos quatro logs (`rota_a_s{0,1}_32695.log`, `bokeh-a-full-32695.{out,err}`):
  **nenhuma ocorrência**. `.err` vazio.
- `/raid`: 7,8 T livres (69% usado). Cache do CuPy dentro do `/raid`, `HOME` intocado,
  `depth_pro` importa de `/home/user_juliadollis/.local/...` — as três causas de morte
  de 32691/32692 não se repetiram.
- Suíte local passou antes de gerar: 679 testes, OK (4 skipped).

---

## 2. Quais artefatos a rota A precisa gravar

Lidos de `src/dataio/writer.py:55-180` (`FileSampleWriter.write`) e
`scripts/publish_release.py:87-96` (`valida`). Para a rota A são **seis** por amostra,
não os quatro da rota B:

| artefato | caminho | escrito por | `valida` confere? |
|---|---|---|---|
| disparidade uint16 | `depth/<id>.png` | `writer.py:107-109` | **sim** (`publish_release.py:91`) |
| banda em foco uint8 | `mask/<id>.png` | `writer.py:111-113` | **sim** |
| metadado | `meta/<id>.json` | `writer.py:132-133` | **sim** |
| bokeh renderizada | `generated/<id>_bokeh.jpg` | `writer.py:120-130` | **NÃO** |
| linha do manifesto | `manifest.jsonl` | `writer.py:139-174` | é a base da varredura |
| linha do ledger de bytes | `generated_images.jsonl` | `routes/route_a.py:1373-1375` | **NÃO** |

Diferença de contrato em relação à rota B (`src/dataio/sample.py:7-11`): na rota A a
**bokeh é gerada** e a AIF é referência; na rota B é o inverso. Então o pixel que a rota A
produz — o alvo do treino — mora em `generated/<id>_bokeh.jpg`, e é justamente o que
`valida` **não confere**.

Ordem de gravação: `depth` → `mask` → `generated` → `meta` → linha do manifesto → linha
do ledger. Logo `manifest ⊆ meta ⊆ generated ⊆ mask ⊆ depth`, e um `sample_id` que esteja
só em `depth/` é a amostra **em voo** no instante da medição, não uma falha.

---

## 3. Completude — medida pela UNIÃO de `sample_id`

Medição às 23:37 (união de `manifest ∪ meta ∪ depth ∪ mask ∪ generated ∪ generated_images.jsonl`).

| | s0 | s1 |
|---|---|---|
| união de `sample_id` | 4.579 | 4.266 |
| **completos (6 de 6)** | **4.578** | **4.265** |
| falta `manifest` | 1 | 1 |
| falta `meta` | 1 | 1 |
| falta `depth` | 0 | 0 |
| falta `mask` | 0 | 1 |
| falta `generated` | 0 | 1 |
| falta `generated_images.jsonl` | 1 | 1 |
| `sample_id` repetidos no manifesto | 0 | 0 |
| linhas de manifesto corrompidas | 0 | 0 |

O único faltante em cada fatia (`ebb_65e5bb013e9e_v27`, `ebb_a5fe59967616_v01`) é a amostra
**em voo**: os artefatos já escritos estão presentes e os posteriores na ordem ainda não.
Não há buraco real.

**Nada do desencontro da rota B** (`meta/` 4.061 contra `generated/` 2.917) aparece aqui:
as seis superfícies andam juntas, com defasagem de no máximo uma amostra.

### `rejections.jsonl` — o desfecho de tudo que entrou

O card trata `rejections.jsonl` como parte do release ("não é só a lista de recusas: é o
desfecho de tudo que entrou"). Auditado em s0: **6.527 linhas, 6.527 `sample_id`
distintos, 6.527 linhas de manifesto — correspondência 1:1 exata.** Nenhum `sample_id`
repetido, nenhum órfão dos dois lados. A retomada não duplicou nada: os jobs 32691/32692
escreveram no mesmo arquivo e o denominador continua honesto.

`status` é `ok` em **100%** das linhas, `reason` sempre `null`. É o esperado no modo medir
(§8), e é a mesma evidência que `rejeitadas: 0 (0,0%)` na enumeração de fonte.

### Disco

| | |
|---|---|
| `/raid` | 27 T, 18 T usados, **7,8 T livres** (69%) |
| `a_synthetic_s0_de2` | 7,6 G |
| `a_synthetic_s1_de2` | 7,0 G |
| `output/` inteiro (A+B+C) | 36 G |

Consumo medido: ~1,30 MB por amostra, contra os ~1,18 MB/amostra que
`estimate_disk_budget` previu (82,2 GB para as duas fatias). Projeção para as 69.700
amostras: **~91 GB**, cerca de 11% acima do orçamento impresso. Com 7,8 T livres no
`/raid` isso não é risco — a nota de "~84 GB de folga" no log vem de `REGISTRO.md:585` e é
uma cota assumida, não o que o `df` mede hoje. A morte do 32691 foi por `/home` cheio, não
por `/raid`, e o `HOME` segue intocado neste job.

### Integridade dos bytes

- **Arquivos de 0 byte: 0** em `depth/`, `mask/`, `meta/`, `generated/` (todos, não amostra).
- 800 amostras completas sorteadas, conferindo assinatura **e trailer** de cada arquivo
  (PNG `IHDR`+`IEND`, JPEG `SOI`+`EOI`, `json.load`): **0 ilegíveis, 0 truncados**.
- Todos os 800 `depth/*.png` são 16 bits, tipo de cor 0 (escala de cinza) — o
  `uint16_linear_in_disparity` do contrato, não um PNG de 8 bits disfarçado.

---

## 4. Proveniência — o que o `CLAUDE.md` exige, e o que está lá

Exigência (`CLAUDE.md`, "Regras de código"): commit, `control_version`, hash de **todo**
modelo que influenciou o rótulo (DeblurNet, DepthPro, **BiRefNet**, commit do BokehMe +
hashes de `arnet.pth`/`iunet.pth`), mais seed e resolução processada.

Amostrados 600 `meta/*.json` (300 por fatia). Resultado uniforme, sem exceção:

| exigido | campo | estado |
|---|---|---|
| `control_version` | `control_version` | ✅ `metric_disparity_official_v1` em 100% |
| commit | `provenance.pipeline_commit` | ⚠️ **`sha256tree:10aadb46…`, não um commit git** |
| `source_sha256` | `provenance_base.pipeline_source_sha256` | ✅ em `run_config.json` |
| DepthPro | `provenance.depth_model_sha256` | ✅ `3eb35ca6…` |
| BiRefNet | `provenance.mask_model_sha256` | ➖ `""` — **correto**: a rota A não tem segmentador |
| DeblurNet | `provenance.deblurnet` | ➖ `null` — **correto**: não influenciou o rótulo |
| BokehMe commit | `provenance.renderer.renderer_commit` | ✅ `8b3ed556dc14…` |
| `arnet.pth` | `provenance.renderer.arnet_sha256` | ✅ `6c7dba6f…` |
| `iunet.pth` | `provenance.renderer.iunet_sha256` | ✅ `6df92779…` |
| seed | `provenance.seed` | ✅ `0` (+ `variant_seed` por amostra em `extra.sampling`) |
| resolução processada | `image_h/w`, `depth_h/w`, `focus_retention_h/w` | ✅ nos três níveis |

Extras que a rota A grava e que passam do exigido: `demo_pipeline_sha256`,
`scatter_py_sha256`, config completa do renderizador, `renderer_report_sha256`
(`c2ca17da…`), `k_distribution_sha256` (`c21c4cbe…`), e `mask_rule` por amostra com a
regra da banda, o quantil sorteado e a semente da variante.

### O que falta

1. **Não há commit git.** `$P/.git` não existe no cluster, então
   `run_route_a.py:425` cai no fallback `f"sha256tree:{_source_sha256(raiz)}"`. É um hash
   da árvore de fontes — reproduzível e honesto, mas **não** aponta para um ponto da
   história do repositório. Quem baixar o release não consegue fazer `git checkout` do
   código que produziu o rótulo. *(O fallback é declarado, não silencioso: o prefixo
   `sha256tree:` diz o que é.)*
2. **`provenance.source_license` é `null` em 100% das amostras.** As fontes são `EBB!` e
   `GenerativePhotography`. Publicar no Hugging Face um derivado sem campo de licença é
   problema de publicação, não de rótulo — mas é problema.
3. **O sha256 da AIF não está no `meta/`.** Ele está em `source_images.jsonl`, chaveado
   por `scene_id` (ver §5). O join é verificável, mas exige o segundo arquivo.

Nada mais falta. A proveniência da rota A é, para o que ela usa, **completa**.

---

## 5. O `KeyError: 'sample_id'` — de quem é o erro

**É do `publish_release.py`.** A rota A grava o que deve; o validador lê com a chave errada.

Os dois lados, com arquivo e linha:

```
LADO QUE LÊ (errado)
  scripts/publish_release.py:138
      ledger = {l["sample_id"]: l for l in _linhas(release / "source_images.jsonl")}

LADO QUE ESCREVE (certo, mas outra coisa)
  src/sources/genphoto_ebb.py:565-569   (classe AifImageLoader, linha 539)
      self._handle.write(json.dumps({
          "scene_id": image.scene_id,
          "role": "aif_reference",
          **image.to_provenance(),
      }, ensure_ascii=False) + "\n")
```

`source_images.jsonl` da rota A é um ledger **por cena**, não por amostra: uma linha por
imagem de origem, com `aif_sha256`. Chaves reais medidas no disco:

```
scene_id, role, source_dataset, source_sample_id, source_revision, source_note,
aif_ref, aif_sha256, aif_image_h, aif_image_w,
aif_laplacian_variance, aif_sharpness_h, aif_sharpness_w
```

Não há `sample_id` em nenhuma linha (114 linhas em s0, 107 em s1 — 41 amostras por linha).
Isso é correto e economiza 41× espaço: as 41 variantes de uma cena vêm dos **mesmos bytes**
de AIF.

O ledger **por amostra** da rota A existe e se chama `generated_images.jsonl`
(`src/routes/route_a.py:1257-1289`, `_ledger_line`). Ele tem `sample_id` em 100% das
linhas, e carrega `bokeh_jpeg_sha256` — o sha256 do JPEG **em disco**, que é exatamente a
prova de "contra quais bytes o rótulo foi produzido" que o §2 do docstring do
`publish_release.py` pede.

### Por que a rota B publicou e a rota A não

A rota B escreve o mesmo `generated_images.jsonl` (`route_b.py:1045-1076`) e **não escreve
`source_images.jsonl` nenhum**. Sem o arquivo, `_linhas` devolve `[]`
(`publish_release.py:60-63`), o dict-comprehension da linha 138 nunca itera, e o `KeyError`
não acontece. A rota A é a primeira que de fato grava esse arquivo — e por isso é a
primeira a bater no defeito.

### Correção proposta (NÃO aplicada)

Trocar a linha 138 por um ledger que aceita as duas chaves, e passar a exigir do ledger
**por amostra** da rota que gera pixel:

```python
# publish_release.py:137-145, substituindo
ledger_cena = {l["scene_id"]: l for l in _linhas(release / "source_images.jsonl")
               if "scene_id" in l}
ledger_amostra = {l["sample_id"]: l for l in _linhas(release / "generated_images.jsonl")
                  if "sample_id" in l}

# rota C: os dois lados são referência, o join é por amostra no source_images.jsonl
rotas_c = [l for l in manifesto if l.get("route") == "c"]
sem_ledger = [l["sample_id"] for l in rotas_c
              if l["sample_id"] not in ledger_amostra
              and l["sample_id"] not in {x.get("sample_id") for x
                                         in _linhas(release / "source_images.jsonl")}]

# rotas A e B: o pixel gerado é NOSSO, então o sha256 que prova o rótulo está no
# generated_images.jsonl; a AIF de referência da rota A é provada por CENA.
geram_pixel = [l for l in manifesto if l.get("route") in ("a", "b")]
sem_sha = [l["sample_id"] for l in geram_pixel if l["sample_id"] not in ledger_amostra]
if sem_sha:
    problemas.append(f"{len(sem_sha)} amostras sem linha em generated_images.jsonl "
                     f"(ex.: {sem_sha[:3]}) — sem o sha256 do pixel que nós geramos, o "
                     "release não prova o que publicou.")
sem_cena = [l["scene_id"] for l in manifesto if l.get("route") == "a"
            and l["scene_id"] not in ledger_cena]
if sem_cena:
    problemas.append(f"{len(set(sem_cena))} cenas da rota A sem linha em "
                     f"source_images.jsonl (ex.: {sorted(set(sem_cena))[:3]}).")
```

A mudança mínima que **destrava** a publicação é só o `if "sample_id" in l` da linha 138.
O resto acima é o que faz a validação da rota A **medir alguma coisa** em vez de só não
quebrar — hoje as linhas 139-145 só olham `route == "c"`, então mesmo sem o `KeyError` o
release da rota A passaria sem nenhuma checagem de proveniência de pixel.

---

## 6. `publish_release.py` em modo validação — todas as falhas

Confirmado no código que **nada sobe sem `--yes`**: `main()` linhas 784-789 retornam
antes de `publica()`; `publica()` é a única função que fala com o Hub. E sem `--repo-id`
nem o card é escrito (linha 780), então a execução é 100% somente-leitura.
`_agrupa_por_cena` (linha 567) só **conta** arquivos, não move nada.

Rodado contra uma fatia congelada (4.853 linhas de manifesto de `a_synthetic_s0_de2`,
com `depth/`/`mask/`/`meta/`/`generated/` por symlink; a fatia vive em
`$P/.cache_raid/fatia_valida_233921/`, fora de `output/`).

**Execução 1 — com `source_images.jsonl` como está no disco:**

```
Traceback (most recent call last):
  File "scripts/publish_release.py", line 801, in <module>
  File "scripts/publish_release.py", line 766, in main
    resumo = valida(release)
  File "scripts/publish_release.py", line 138, in valida
    ledger = {l["sample_id"]: l for l in _linhas(release / "source_images.jsonl")}
KeyError: 'sample_id'
```

Reproduzido exatamente como descrito. **Não é uma `Problema` tratada** — é uma exceção
não capturada, então nem o relatório parcial é impresso.

**Execução 2 — mesma fatia sem `source_images.jsonl`, para ver o que vem depois da linha 138:**

```
  release: .../fatia_valida_233921/sem_ledger
    amostras                   4853
    cenas                      119
    por_split                  {'train': 4566, 'val': 287}
    por_rota                   {'a': 4853}
    control_version            metric_disparity_official_v1
    depth_backend              depth_pro
    k_censurado                0
    sem_validador_analitico    4853
    focus_sources              {'sampled_plane': 4853}
    refined                    4853
    autocontido                False
  [aviso] 4853 de 4853 amostras tiveram a região em foco REFINADA …
  Nada foi enviado (falta --yes).
```

### Lista completa

**Reprovações (`problemas`, impedem a publicação): 1**

1. `KeyError: 'sample_id'` em `publish_release.py:138`. Único bloqueio dentro de `valida`.
   Removido ele, **a fatia passa com zero `problemas`**: nenhum metadado inválido,
   `control_version` único, `depth_backend` único, nenhum `sample_id` repetido, nenhum
   arquivo faltando em `depth`/`mask`/`meta`, nenhuma cena fora do `split.json`, nenhum
   vazamento de split, `k_censurado = 0`.

**Avisos (não bloqueiam, mas são o relatório): 1**

2. `refined 4853 de 4853` → dispara o aviso de `publish_release.py:165-171`, que manda
   rodar `scripts/validate_focus_refinement.py` antes de virar treino. **É falso positivo
   na rota A**: `focus_source = sampled_plane` para 100% das amostras, e
   `FocusRegionRecord.was_refined` devolve `True` para tudo que não é `BIREFNET`
   (`sample.py:228`). Na rota A não houve refinamento nenhum — o plano de foco foi
   *sorteado*, e o próprio metadado diz isso em
   `provenance.extra.focus_region_semantics`. O laudo que o aviso pede não existe para
   esta rota e não faria sentido. Ou o aviso ganha um `and route != "a"`, ou o card vai
   sair afirmando que 69.700 amostras foram refinadas.

**O que `valida` NÃO confere e vai doer depois (não aparece na execução acima): 4**

3. **`generated/` não é conferido.** `publish_release.py:91` percorre só
   `("depth", ".png"), ("mask", ".png"), ("meta", ".json")`. Na rota A o alvo do treino é
   o JPEG em `generated/` — um release com 10.000 bokehs faltando passaria na validação.
   *(Medido: hoje não falta nenhum. O ponto é que a validação não garante isso.)*
4. **`generated_images.jsonl` não é conferido**, pelo mesmo motivo — ver §5.
5. **Duas fatias, dois releases.** `a_synthetic_s0_de2` e `a_synthetic_s1_de2` são dois
   diretórios completos e independentes, cada um com seu `manifest.jsonl`, `split.json`,
   `run_config.json`, `source_images.jsonl`. `publish_release.py --release-dir` recebe
   **um**. Não existe script de mesclagem em `scripts/` (conferido: só
   `build_k_distribution`, `calibrate_thresholds`, `diagnose_empty_masks`,
   `publish_release`, `run_route_{a,b,c}`, `validate_focus_refinement`,
   `verify_renderer`). Ou se publica dois repositórios, ou alguém tem que mesclar — e a
   mesclagem tem uma armadilha: `run_config.json` **colide** (os `args` diferem em
   `shard`, `output_dir`), enquanto `manifest.jsonl`, `split.json` e os dois ledgers são
   concatenáveis sem conflito (§7).
6. **`NaN` literal em 100% das linhas** de `manifest.jsonl` e dos `meta/*.json`
   (`focus_retention_in_region: NaN`, que é semanticamente correto — a rota A tem uma foto,
   não um par, então não existe retenção a medir). `NaN` **não é JSON válido** pela
   especificação: `json.loads(..., parse_constant=raise)` rejeita. Testado no container do
   job: `pyarrow 22.0.0` **aceita**, e `datasets 4.3.0` usa pyarrow — então o caminho do
   Hub funciona. Quebra em `JSON.parse` do navegador e em qualquer parser estrito. É risco
   de consumidor, não bloqueio de upload.

### 6-bis. O card é FIXO NA ROTA C — e isso já aconteceu com a rota B

Achado fora do escopo pedido, mas é o pior dos que encontrei, porque não dá erro: sai
calado e vira a página do dataset.

`escreve_card` (`publish_release.py:218-492`) monta o README a partir de uma f-string
**com o texto da rota C escrito à mão**. Só os números vêm do `resumo`. O que ele vai
publicar se alguém rodar `--repo-id ... --yes` na rota A:

| linha do card | o que sai | a rota A é |
|---|---|---|
| `publish_release.py:275` | `pretty_name: BokehNet regen — rota C` | rota A |
| `:281-283` | "a **rota C**: pares reais (all-in-focus, bokeh), calibrado pelo sweep da Eq. 5" | pares **sintéticos**, K **sorteado**, alvo **renderizado**; a Eq. 5 não é usada |
| `:346-405` | seção inteira sobre refinamento da máscara do BiRefNet, os 35,2%, os 20,6% de máscara vazia | não há BiRefNet nem máscara: o plano de foco é sorteado |
| `:407-419` | "teto de 4 aberturas por cena", RealBokeh, `train/in/<id>_f22.JPG` | as fontes são `EBB!` e `GenerativePhotography`, 41 variantes por imagem |
| `:421-441` | lista de arquivos **sem `generated/` e sem `generated_images.jsonl`** | são exatamente os dois artefatos que a rota A produz |
| `:462` | `pipeline` = `sha256tree:…` | ok, mas sem contexto |
| `:464` | `máscara: None · …` | `mask_backend` é `None` e `mask_model_sha256` é `""` na rota A |
| `:487` | "Os pixels seguem a licença da origem (`{fonte}`)" | **`fonte` sai como `(não registrada)`** |

O último merece detalhe: `escreve_card:223` faz
`prov.get("source_dataset", "(não registrada)")`, e o `provenance_base` da rota A **não tem
essa chave**. Conferido no `run_config.json` do disco — as chaves são
`source_roots`, `source_cuts`, `source_quota_evidence`, e nenhuma `source_dataset`. O card
sairia dizendo que os pixels seguem a licença de "(não registrada)".

**E isto não é hipótese.** O card já gravado da rota B, em `output/b_oficial_release/README.md`,
começa assim:

```
pretty_name: BokehNet regen — rota C
# juliadollis/bokehnet-regen-rota-b-oficial
Dados de treino ... §3.2(c) — a **rota C**: pares reais (all-in-focus, bokeh), com o
sinal de controle calibrado pelo sweep da Eq. 5.
```

Um release da rota B descrito como rota C. Se ele foi ao Hub, a página está errada lá.

Correção proposta: o card tem que ser **por rota**. O mínimo é derivar as três coisas que
mudam — nome, parágrafo de abertura e lista de arquivos — de `resumo["por_rota"]`, e fazer
o `provenance_base` de cada rota gravar `source_dataset` (na rota A, a lista de
`source_roots`). Enquanto isso não existir, **publicar a rota A exige escrever o card à
mão** e não passar `--repo-id` ao validador.

---

## 7. Split por cena

| | s0 | s1 |
|---|---|---|
| `split.json` materializado | sim | sim |
| `val_fraction` declarada | 0,05 | 0,05 |
| `salt` | `bokehnet-regen-v1` | `bokehnet-regen-v1` |
| cenas no split | 835 | 865 |
| train / val (cenas) | 790 / 45 | 829 / 36 |
| cenas do manifesto fora do split | **0** | **0** |
| cenas com split ambíguo no manifesto | **0** | **0** |
| linhas divergindo do split materializado | **0** | **0** |

**Vazamento entre as fatias: nenhum.**

- `sample_id` em comum entre s0 e s1: **0**
- `scene_id` em comum entre s0 e s1: **0**
- cenas com atribuição de split conflitante entre as fatias: **0**

O fatiamento é por `sha256(scene_id) % 2` (`run_route_a.py:184-187`), então as 41 variantes
de uma imagem ficam sempre na mesma fatia **e** do mesmo lado do split — o defeito A9 não
reabre. As duas fatias usam o mesmo `salt`, então concatenar os dois `assignment` produz um
split coerente de 1.700 cenas (790+829 train, 45+36 val = 1.619/81, ou 4,8% em val).

---

## 8. Sanidade do rótulo

Base: 10.314 amostras no manifesto das duas fatias (medição às 23:45); qualidade medida
sobre 2.400 `meta/` sorteados.

### K

| medida | valor |
|---|---|
| valores distintos de `k_value` | 7.758 em 8.843 (88%) — contínuo, não catálogo |
| mínimo | 0,4266 |
| p05 | 0,4449 |
| p25 | 4,689 |
| **mediana** | **18,60** |
| p75 | 39,22 |
| p95 | 73,32 |
| p99 | 213,9 |
| **máximo** | **876,68** |

### A faixa verificada do renderizador é [8, 96] — e 35% das amostras estão fora

`output/renderer_verification.json` mediu a resposta do BokehMe em
`k_values = [8, 16, 32, 64, 96]`. Nada abaixo de 8, nada acima de 96.

| faixa | amostras | fração |
|---|---|---|
| **K < 8** (abaixo do menor K verificado) | 3.334 | **32,3%** |
| 8 ≤ K ≤ 96 (verificado) | 6.672 | 64,7% |
| **K > 96** (acima do maior K verificado) | 308 | **3,0%** |

O lado de baixo é o problema maior, e não estava na pergunta:

- **13,7% das amostras têm K preso no piso** (`k_clamped_to_positive_floor = true` em
  409 de 3.000 `meta/` sorteados). `k_per_long_side` mínimo é `0,000285756` em ambas as
  fatias — é literalmente o piso, não uma cauda.
- Nas amostras com K < 8, a p99 do CoC renderizado tem **mediana de 0,27 px** — sub-pixel.
  A razão de nitidez bokeh/AIF tem **mediana 0,989**, e a banda em foco cobre o quadro
  inteiro (`area_ratio` mediana = 1,0).
- **21,8% de todas as amostras sorteadas (524 de 2.400) têm `bokeh_over_aif_sharpness > 0,98`**:
  o alvo é, na prática, a própria AIF. Extrapolando, ~15.000 das 69.700 amostras finais
  serão pares quase-identidade.

O lado de cima:

- Das 64 amostras sorteadas com K > 96, a p95 da `defocus_saturation_ratio` é **0,66** —
  dois terços dos pixels no teto de `MAX_COC = 100`, onde o mapa de controle não distingue
  mais quanto borrar. 9 das 2.400 sorteadas passam de 10% de saturação.
- CoC renderizado chega a **298 px** medidos, contra 38,4 px no maior ponto verificado.

**Isto não é um defeito do run — é o run fazendo o que foi mandado.** O `.slurm` foi
submetido com `THRESHOLDS='--seed 0'`, e o próprio log diz:

```
[rota-a] limiares congelados: nenhum — modo medir
[rota-a] AVISO: run completo com todos os gates em modo medir.
         Intencional só se você quer o lote bruto para calibrar depois.
```

Os quatro gates que pegariam exatamente isso — `--min-rendered-coc-p99-px`,
`--max-bokeh-over-aif-sharpness`, `--max-mask-area-ratio`,
`--max-defocus-saturation-ratio` — estão todos medindo e nenhum bloqueando. Por isso
`rejeitadas: 0 (0,0%)` na enumeração e `is_valid_for_control = False` em **0** amostras.

A consequência para a publicação: **`is_valid_for_control` não filtra nada.** Quem baixar
o release e confiar nesse campo leva as 15 mil quase-identidades e as 2 mil saturadas
junto. O filtro precisa ser feito na publicação, a partir de `quality.*` do `meta/`, e
`publish_release.py` não tem gate nenhum para isso.

### Outros campos

| campo | distribuição |
|---|---|
| `is_valid_for_control = False` | **0** de 8.843 |
| `is_k_censored = True` | **0** de 8.843 (a rota A sorteia K, não o busca) |
| `focus_source` | `sampled_plane`: 100% |
| `mask_source` | `sampled_plane`: 100% |
| `control_version` | `metric_disparity_official_v1`: 100% |
| `depth_backend` | `depth_pro`: 100% |
| `max_coc` | `100.0`: 100% |
| `k_source` | `sampled_from_bc`: 100% |
| amostras por split | train 8.433 · val 410 (4,6%) |

### `source_dataset` — só `EBB!` até agora, e é ordem de processamento

**Confirmado, com as duas evidências:**

1. **Código.** `rank_and_cut_per_source` (`src/sources/genphoto_ebb.py:467-490`) itera
   `for fonte in sorted(por_fonte)` e faz `mantidas.extend(selecionadas)`. `"EBB!"` <
   `"GenerativePhotography"` em ordem ASCII, então **toda** a EBB! sai antes de qualquer
   GenerativePhotography, e `run_route_a` consome a lista nessa ordem.
2. **Dados.** O `split.json` de cada fatia — que é materializado **antes** de gerar, sobre
   todas as cenas da fatia — já contém as duas fontes:

   | fatia | cenas `ebb_` | cenas `gp_` | total |
   |---|---|---|---|
   | s0 | 418 | **417** | 835 |
   | s1 | 432 | **433** | 865 |

   850 de cada fonte, como o log de enumeração disse (`EBB!: 850/4694 mantidas`,
   `GenerativePhotography: 850/1000 mantidas`).

As 850 cenas da GenerativePhotography **estão no plano** e entram quando a EBB! acabar.
Faltavam 292 cenas EBB! em s0 e 314 em s1 na hora da medição — ≈24.800 amostras, ou
cerca de **1,5 h de relógio** antes da primeira amostra `gp_`. A fonte deve aparecer por
volta de 01:15.

*(Risco a vigiar: se o job morrer antes disso, o lote inteiro é metade do dataset — só
EBB!. Isso vai na seção de acompanhamento.)*

---

## 9. O que ainda não consegui verificar

- **Se `GenerativePhotography` de fato entra.** Está provado que está no plano e na ordem
  certa, mas **nenhuma amostra `gp_` foi gravada ainda**. Só a vigilância até depois do
  cruzamento (≈01:15) transforma isso de `[I]` inferido em `[M]` medido. É o item aberto
  mais importante.
- **Se o job termina.** Ele está em 14,5% com 3,6 h de ETA.
- **`publish_release.py` sobre o release COMPLETO.** A validação acima é sobre uma fatia
  de 4.853 linhas de uma das duas fatias. Sobre 69.700 amostras, `_agrupa_por_cena`
  (`publish_release.py:567`) vai disparar — as quatro pastas passam de 10.000 arquivos — e
  o layout vira `depth/<scene_id>/<id>.png`. Li o caminho e ele trata o
  `<sample_id>_bokeh.jpg` da rota A corretamente (`_sample_id_do_arquivo:617-620` corta o
  `_bokeh`), mas isso é leitura de código, não execução.
- **Corretude numérica do rótulo.** Conferi presença, integridade de bytes, vocabulário
  fechado e coerência interna. **Não** re-derivei K, disparidade ou CoC contra
  `src/control/contract.py` para verificar se o número está certo — isso é trabalho do
  `unit-contract` e de `verify_renderer.py`, não desta vigilância.
- **Se a bokeh renderizada corresponde ao mapa de defocus.** Nenhum pixel foi inspecionado
  visualmente; a única evidência é `bokeh_over_aif_sharpness` e `rendered_coc_p99_px` do
  próprio `meta/`.
- **O card gerado de fato.** Li `escreve_card` linha a linha (§6-bis) e comparei com os
  cards já no disco da rota B e da rota C, mas **não executei** `--repo-id` na rota A —
  isso escreveria `README.md` dentro de `output/a_synthetic_s0_de2/`, que é o release vivo,
  e a regra é somente leitura.
- **Se a rota B já está publicada no Hub com o card errado.** Só o `README.md` local foi
  lido; não consultei o Hugging Face.
- **O upload em si.** Nada foi enviado ao Hugging Face, e não tentei — falta `--yes`, por
  desenho, e a regra do `CLAUDE.md` manda perguntar antes de escrever em destino
  compartilhado.

---

## 10. Acompanhamento (checagens a cada 10 min)

| hora | estado | amostras | ritmo | ETA | `gp_` |
|---|---|---|---|---|---|
| 23:46 | RUNNING | 11.197 (16,1%) | 16.667/h | 3,5 h | 0 |
| 23:56 | RUNNING | 13.983 (20,1%) | 16.709/h | 3,3 h | 0 |
| 00:06 | RUNNING | 16.823 (24,1%) | 16.788/h | 3,1 h | 0 |
| 00:16 | RUNNING | 19.617 (28,1%) | 16.820/h | 3,0 h | 0 |
| 00:26 | RUNNING | 22.429 (32,2%) | 16.869/h | 2,8 h | 0 |

---

## 11. Mesclagem das duas fatias — auditoria e receita verificável

A rota A produz **dois releases completos** (`a_synthetic_s0_de2`, `a_synthetic_s1_de2`) e
`publish_release.py --release-dir` recebe **um**. Não existe script de merge no projeto
(conferido: `scripts/` tem só `build_k_distribution`, `calibrate_thresholds`,
`diagnose_empty_masks`, `publish_release`, `run_route_{a,b,c}`,
`validate_focus_refinement`, `verify_renderer`).

Auditoria feita com `$P/.cache_raid/auditoria_merge.py`, sobre 22.936 amostras
(medição às 00:31, com o job ainda rodando). **Somente leitura.**

### 11.1 As cinco perguntas, respondidas com medição

**(1) Quais arquivos existem, e como cada um se compõe.** As duas fatias têm exatamente
o mesmo conjunto de entradas — nenhuma existe só em uma:

| entrada | o que é | composição na mesclagem | por quê |
|---|---|---|---|
| `depth/` | 1 PNG por amostra | **unir arquivos** | nomes são `sample_id`, disjuntos entre fatias |
| `mask/` | 1 PNG por amostra | **unir arquivos** | idem |
| `meta/` | 1 JSON por amostra | **unir arquivos** | idem |
| `generated/` | 1 JPEG por amostra | **unir arquivos** | nome é `<sample_id>_bokeh.jpg`, disjunto |
| `manifest.jsonl` | 1 linha por amostra | **concatenar** | chave `sample_id`, disjunta |
| `generated_images.jsonl` | 1 linha por amostra, sha256 da bokeh gerada | **concatenar** | chave `sample_id`, disjunta |
| `rejections.jsonl` | 1 linha por amostra processada (`ok`/`rejected`) | **concatenar** | chave `sample_id`, disjunta |
| `source_images.jsonl` | 1 linha por **cena**, sha256 da AIF | **concatenar** | chave `scene_id`, disjunta |
| `source_rejections.jsonl` | histograma da enumeração de fonte | **manter UM** | ver abaixo |
| `split.json` | `scene_id → train/val` | **unir `assignment` + RECALCULAR `counts`** | ver (4) |
| `run_config.json` | args + `provenance_base` | **um mesclado, com os campos divergentes preservados** | ver (3) |

O caso que engana é o `source_rejections.jsonl`: ele é **byte-idêntico** nas duas fatias
(mesmo sha256 `dc53a3f5db31de2e…`, 17.082 linhas em cada). A enumeração das fontes roda
**antes** do fatiamento, então cada processo escreveu o mesmo arquivo. **Concatenar
dobraria o denominador do histograma de motivos de rejeição de fonte** — 34.164 linhas para
17.082 imagens avaliadas. Tem que ficar um só, e a asserção P4.3 da receita prova isso.

**(2) Colisão de `sample_id` entre as fatias: nenhuma, em nenhuma superfície.**

| superfície | chave | s0 | s1 | interseção | união |
|---|---|---|---|---|---|
| `manifest.jsonl` | `sample_id` | 11.803 | 11.133 | **0** | 22.936 |
| `generated_images.jsonl` | `sample_id` | 11.803 | 11.133 | **0** | 22.936 |
| `rejections.jsonl` | `sample_id` | 11.804 | 11.133 | **0** | 22.937 |
| `source_images.jsonl` | `scene_id` | 288 | 272 | **0** | 560 |
| arquivos `depth/` | `sample_id` | 11.804 | 11.134 | **0** | 22.938 |
| arquivos `mask/` | `sample_id` | 11.804 | 11.134 | **0** | 22.938 |
| arquivos `meta/` | `sample_id` | 11.804 | 11.134 | **0** | 22.938 |
| arquivos `generated/` | nome do arquivo | 11.804 | 11.134 | **0** | 22.938 |
| `scene_id` do manifesto | `scene_id` | 288 | 272 | **0** | 560 |

Medido, não assumido. É a consequência de `sha256(scene_id) % 2` (`run_route_a.py:184-187`)
com `sample_id = <scene_id>_v<NN>`: cenas disjuntas ⇒ amostras disjuntas.

**(3) `run_config.json`: 91 campos achatados, 8 divergentes, e nenhuma divergência
inesperada.**

| campo divergente | s0 | s1 | esperado? |
|---|---|---|---|
| `args.shard` | 0 | 1 | sim |
| `args.output_dir` | `…_s0_de2` | `…_s1_de2` | sim |
| `images_enumerated` | 835 | 865 | sim — é a fatia |
| `samples_expected` | 34.235 | 35.465 | sim |
| `disk_budget.*` (4 campos) | 40,4 GB | 41,8 GB | sim — derivado do anterior |

**`3_divergencias_INESPERADAS` = `{}`.** Todos os outros 83 campos — incluindo as fontes,
quotas, limiares, `samples_per_image`, `seed`, `val_fraction` — são idênticos. As fatias
**são** comparáveis.

**(4) `split.json`: cada fatia tem só as suas cenas, e a união é o split correto.**

| | s0 | s1 | mesclado |
|---|---|---|---|
| cenas | 835 | 865 | **1.700** |
| cenas em comum | — | — | **0** |
| `salt` | `bokehnet-regen-v1` | `bokehnet-regen-v1` | mesmo |
| `val_fraction` | 0,05 | 0,05 | mesmo |
| `counts` | train 790 / val 45 | train 829 / val 36 | **train 1.619 / val 81** |

Os arquivos **não** são idênticos, e não deveriam ser: cada fatia materializou só as suas
cenas. A mesclagem tem que **unir os `assignment`** e **recalcular `counts`** — copiar o
`counts` de uma fatia publicaria uma contagem errada, e `counts` é campo gravado, não
derivado na leitura (`split.py:57-63`).

A garantia que fecha o caso está medida: **concatenar é equivalente a recalcular.**
`build_scene_split` atribui por `sha256(salt:scene_id) % 10000 < 500`
(`split.py:29-36, 86-90`) — função pura da cena e do salt, sem dependência do conjunto.
Conferi as 1.700 cenas: o lado que cada uma tem no `split.json` da sua fatia é **exatamente**
o que `build_scene_split` daria sobre a união. A mesclagem não inventa split nenhum.

**(5) Proveniência: idêntica.** `provenance_base` dos dois `run_config.json` tem **zero**
campos divergentes. E, por amostra, 800 `meta/*.json` (400 de cada fatia) têm **um único
valor distinto** em cada um dos 15 campos que definem o pipeline:

`control_version`, `max_coc`, `depth_backend`, `k_source`, `mask_source`,
`pipeline_commit`, `depth_model_sha256`, `mask_model_sha256`, `seed`,
`renderer_commit`, `arnet_sha256`, `iunet_sha256`, `demo_pipeline_sha256`,
`scatter_py_sha256`, `is_final_label_renderer`.

O release mesclado **não** seria uma mistura de dois pipelines. É um pipeline só, rodado
em duas GPUs.

### 11.2 A receita

Implementada e **testada em miniatura** em `$P/.cache_raid/testa_merge.py`: duas
mini-fatias de 60 amostras cada, copiadas de `output/` (leitura), mescladas em
`$P/.cache_raid/merge_test_003136/merged`. **36 asserções, 36 OK, 0 falhas.**

> **Não execute sobre o release de verdade antes de o job 32695 terminar.** As fatias estão
> sendo escritas agora; mesclar um alvo móvel congela metade.

**PASSO 0 — pré-condições.** Se qualquer uma falhar, a receita para e não mescla nada.

| | asserção |
|---|---|
| P0.1 | nenhum `sample_id` em comum entre as fatias |
| P0.2 | nenhum `scene_id` em comum entre as fatias |
| P0.3 | `provenance_base` idêntico entre as fatias |
| P0.4 | `salt` e `val_fraction` iguais nos dois `split.json` |
| P0.5 | `source_rejections.jsonl` byte-idêntico nas duas fatias |

**PASSO 1 — destino novo.** `output/a_release_merged/` (ou nome equivalente), criado do
zero. Nunca por cima de uma das fatias — regra 5 do `CLAUDE.md`: preferir destino novo a
sobrescrever.

| | asserção |
|---|---|
| P1.1 | o destino não existia antes |

**PASSO 2 — arquivos por amostra, por hardlink.** `os.link()` de cada arquivo das quatro
pastas. As duas fatias e o destino estão no mesmo filesystem (device `2305`, conferido com
`stat`), então o hardlink custa **0 byte** — contra ~91 GB de uma cópia. Um hardlink não é
um alias frágil: apagar um lado não afeta o outro, e nada neste pipeline reescreve arquivo
no lugar. Se a intenção for arquivar independentemente, trocar por `shutil.copy2`.
**Abortar na primeira colisão de nome**, nunca sobrescrever.

| | asserção |
|---|---|
| P2.1 | cada pasta tem exatamente a soma das duas fatias, sem colisão |

**PASSO 3 — os quatro ledgers por amostra: concatenar.**
`manifest.jsonl`, `generated_images.jsonl`, `rejections.jsonl`, `source_images.jsonl`.

| | asserção |
|---|---|
| P3.1 | nº de linhas = soma das fatias, arquivo por arquivo |
| P3.2 | nenhuma chave repetida depois de concatenar (`sample_id`, ou `scene_id` no `source_images.jsonl`) |

**PASSO 4 — `source_rejections.jsonl`: manter UM.**

| | asserção |
|---|---|
| P4.1 | sha256 igual nas duas fatias (se não for, PARE: a enumeração divergiu) |
| P4.2 | o arquivo no destino tem o mesmo sha256 |
| P4.3 | nº de linhas igual ao de **uma** fatia — não dobrado |

**PASSO 5 — `split.json`: unir e recalcular.** União dos `assignment`, abortando se uma
cena aparecer com dois lados; `salt` e `val_fraction` preservados; `counts` **recalculado**.

| | asserção |
|---|---|
| P5.1 | o split mesclado tem a soma das cenas das duas fatias |
| P5.2 | `counts` recalculado (não copiado de uma fatia) |
| P5.3 | **concatenar == recalcular**: `build_scene_split` sobre a união daria o mesmo split |
| P5.4 | toda cena do manifesto mesclado está no split mesclado |
| P5.5 | nenhuma linha do manifesto diverge do split mesclado |

**PASSO 6 — `run_config.json`: mesclar sem apagar o que diverge.** Campos comuns no topo;
`images_enumerated` e `samples_expected` viram a **soma**; os campos divergentes de cada
fatia vão inteiros para uma lista `merged_from_shards`. Esconder a divergência seria
exatamente o que a pergunta (3) existe para impedir.

| | asserção |
|---|---|
| P6.1 | `provenance_base` preservado intacto |
| P6.2 | os campos divergentes estão em `merged_from_shards`, não apagados |
| P6.3 | `samples_expected` é a soma, não o de uma fatia |

**PASSO 7 — asserções finais do release mesclado.**

| | asserção |
|---|---|
| P7.1 | todo `sample_id` do manifesto é único |
| P7.2 | todo `sample_id` tem os **quatro** arquivos — incluindo `generated/`, que o `valida` não confere |
| P7.3 | todo `sample_id` tem linha em `generated_images.jsonl` (sha256 do pixel gerado) |
| P7.4 | toda cena tem linha em `source_images.jsonl` (sha256 da AIF) |
| P7.5 | todo `sample_id` do manifesto aparece em `rejections.jsonl` |
| P7.6 | `control_version` único no release mesclado |
| P7.7 | nenhum arquivo órfão em `meta/` sem linha no manifesto |

### 11.3 O release mesclado de teste, validado

Rodado `publish_release.py` (sem `--yes`, sem `--repo-id`) sobre o mesclado de 120 amostras:

- **com** `source_images.jsonl`: mesmo `KeyError: 'sample_id'` na linha 138. A mesclagem
  **não** conserta o defeito do §5, e não deveria: são coisas independentes.
- **sem** `source_images.jsonl`: passa com **zero `problemas`** —
  `amostras 120 · cenas 4 · por_rota {'a': 120} · control_version metric_disparity_official_v1 ·
  depth_backend depth_pro · k_censurado 0`, só o aviso falso-positivo de `refined`.

Ou seja: **a receita produz um release que valida.** O que sobra para destravar a
publicação são os dois itens já relatados — a linha 138 (§5) e o card fixo na rota C
(§6-bis) —, nenhum deles causado pela mesclagem.

### 11.4 O que a receita NÃO faz, de propósito

- **Não filtra por qualidade.** As ~32% de amostras com K < 8 (pares quase-identidade) e
  as ~3% com K > 96 entram no mesclado. O filtro é decisão de quem publica, sai de
  `quality.*` do `meta/`, e é reversível sem regerar porque o `sample_id` é determinístico
  (§8).
- **Não reescreve `NaN`.** Mexer no conteúdo das linhas quebraria o sha256 do ledger.
- **Não gera o card.** Ver §6-bis: o card atual descreveria a rota A como rota C.
- **Não sobe nada.**
