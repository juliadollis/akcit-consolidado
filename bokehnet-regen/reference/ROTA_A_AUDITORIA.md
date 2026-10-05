# Auditoria da ROTA A — §3.2(a), dados sintéticos

**2026-09-10.** Rodada de auditoria, sem código. Toda afirmação carrega evidência:
`paper.txt:<linha>` ou `<arquivo>:<linha>`. Onde não houve medição, está escrito
**não medido** — não há estimativa disfarçada.

Convenção herdada de `ACHADOS.md`: `[M]` medido · `[I]` inferido de algo medido ·
`[A]` assumido e não verificado.

---

## 1. O que a rota A é, segundo o paper

### 1.1 O parágrafo, inteiro

`paper.txt:328-334`, §3.2(a):

> *(a) Synthetic data. We pretrain with synthetic data: starting from real all-in-focus
> images and their estimated depth map D, we randomly sample a focus plane Dfocus and a
> target bokeh level K (Fig. 3 (a)). We then construct Ddef using Eq. 2 and use a
> simulator [43] to render the corresponding target bokeh image consistent with Ddef.
> This synthetic pretraining helps the network modulate the circle of confusion
> according to Ddef; however, it is constrained by renderer bias and may introduce
> unrealistic artifacts.*

A legenda da Fig. 3 diz o mesmo em outra ordem, `paper.txt:290-292`:

> *(a) Synthetic data. Given real AIF images and depth maps D, we compute a defocus map
> Ddef parameterized by randomly selected bokeh level K and focus plane Dfocus, and feed
> it into a bokeh renderer [43] to synthesize corresponding bokeh images.*

Cinco fatos saem daí, e só cinco. Tudo além disso é `[A]` ou `[I]`.

### 1.2 A fonte de imagens — **CONFIRMADO: Generative Photography [80] + EBB! [27]**

A afirmação do nosso trabalho está **confirmada contra o texto**, e a troca do pipeline
original está **refutada**.

| evidência | linha | o que diz |
|---|---|---|
| supplement B.2 | `paper.txt:996` | *"We draw candidate images from **[80]** and the **EBB [27]** collections"* |
| §4.1, Datasets | `paper.txt:527-528` | *"∼70K synthetic pairs derived from **[27, 80]**"* |
| bibliografia | `paper.txt:941-943` | `80. Yuan, Y., ...: **Generative photography**: Scene-consistent camera control for realistic text-to-image synthesis. CVPR (2025)` |
| bibliografia | `paper.txt:814-815` | `27. Ignatov, A., Patel, J., Timofte, R.: **Rendering natural camera bokeh effect with deep learning**. CVPRW (2020)` — é o **EBB!** |

**DiffCamera é `[69]`, e nunca aparece como fonte de dado.** A citação `[69]` ocorre em
`paper.txt:115` (Tab. 1, linha de baseline), `225` (related work, *"requires
fixed-resolution inputs (512 × 512)"*), `385`, `498`, `587`, `1019` (tabelas de
comparação), `971` e `1050` (supplement D, *"Additional Comparison with DiffCamera"*).
Zero ocorrências em §3.2, na Fig. 3 ou no B.2. Bibliografia em `paper.txt:914`.

Cuidado de leitura, registrado para quem vier depois: **o `paper.txt` tem duas
numerações de citação.** As legendas das Figs. 4 e 6 usam uma numeração antiga
(`DiffCamera [67]` em `paper.txt:434,444`; `Restormer [80]` em `paper.txt:429,439`),
enquanto o corpo e a bibliografia usam a atual (`DiffCamera [69]` em `paper.txt:385`;
`Restormer [82]` em `paper.txt:382`). **Só a bibliografia e o corpo valem.** Nas linhas
que importam — `527-528` e `996` — a numeração é a atual, e `[80]` bate com
`paper.txt:941` (Generative Photography) e `[27]` com `paper.txt:814` (EBB!).

Confirmação lateral: `paper.txt:240` cita `[80]` como *"uses temporal modeling [80]"* no
related work de controle de câmera — coerente com Generative Photography, que modela a
variação de parâmetro de câmera como sequência temporal. Não é contradição.

### 1.3 Quantas amostras por imagem — **~41, e é `[I]`, não publicado**

O paper não publica o número. Ele publica os dois lados da conta:

| grandeza | valor | linha |
|---|---|---|
| pares sintéticos totais | **∼70K** | `paper.txt:527-528` |
| pool de AIF nítidas após o filtro | **∼1,7K** | `paper.txt:998` |

`70.000 / 1.700 = 41,2` **[I]**. O plano dizia "~40" e o número **se sustenta** — mas
como inferência aritmética de duas quantidades publicadas, nunca como número do paper.
O `41` que entrar no código é `[A]` escolhido para fechar as duas âncoras.

O histórico bate: o dataset antigo tinha ~68.000 amostras de ~1.700 AIFs, ou seja ~40
variações por imagem (`ACHADOS.md:23` `[M]`; `bokehnet-preprocessing/docs/ERROS_GERACAO_ORIGINAL_BOKEHNET.md:170`).
Ou seja: **o dataset publicado já tinha a multiplicidade certa; o pipeline v2 no
repositório é que a perdeu** (§2.2 abaixo).

### 1.4 Como K é obtido — **por sorteio, sem equação, escopo confirmado**

`paper.txt:329-330`: *"we **randomly sample** a focus plane Dfocus and a target bokeh
level K"*. Não há equação para K na rota A.

O escopo das equações, conferido linha a linha e coerente com `CONTRATO.md:153-161`:

| Eq. | linha | escopo | vale na rota A? |
|---|---|---|---|
| 2 — `Ddef = K·|D − Dfocus|` | `paper.txt:312` | todas as rotas | **SIM** — `paper.txt:330` manda usar |
| 3 — `K ≈ f²Dfocus/(2F(Dfocus−f))·pixel_ratio` | `paper.txt:340` | *"(b) ITW dataset"*, `paper.txt:336-338` | **NÃO** |
| 4 — `Dfocus = median(D[M])` | `paper.txt:352` | (b), e (c) por *"Similar to (b)"* `paper.txt:361` | **NÃO** — na A o plano é sorteado, não medido |
| 5 — `K* = argmax SSIM(...)` | `paper.txt:393` | *"(c) LFDOF and RealBokeh"*, `paper.txt:369-370` | **NÃO** |

Consequência que muda o desenho da rota: **a rota A não tem máscara, não tem BiRefNet e
não tem SSIM de calibração.** O plano de foco é um sorteio, não uma mediana sobre região
em foco. Qualquer máscara que a rota A grave é invenção nossa e tem que ser declarada
como tal.

A **distribuição** do sorteio o paper **não publica**. Amostrar K da distribuição
empírica de B/C (`k_source = "sampled_from_bc"`) é decisão nossa já registrada em
`CONTRATO.md:163-166`. Continua `[A]`.

### 1.5 O renderizador — **BokehMe [43], explicitamente autorizado**

`paper.txt:331`: *"use a **simulator [43]** to render the corresponding target bokeh
image consistent with Ddef"*. `paper.txt:292`: *"feed it into a **bokeh renderer [43]**"*.
E `paper.txt:851-852`: `43. Peng, J., ...: **Bokehme**: When neural rendering meets
classical rendering. CVPR (2022)`.

**Sim, o paper autoriza o BokehMe aqui, e de forma mais direta do que na rota C.** Na
rota C ele chama `R` de *"our physically guided renderer"* (`paper.txt:396`) e só o
supplement fecha a identificação (`paper.txt:1007`: *"we then optimized the parameter K
using **simulator [43]**"*). Na rota A a citação é literal, duas vezes, no corpo.

Isso resolve a hesitação do pipeline v2, que marcava todo BokehMe como
`is_final_label_renderer=False` (`bokehnet-preprocessing/src/preprocessing/renderer.py:59`).
Para a rota A não há ambiguidade a resolver: é `[43]`, e o `[43]` é público. A extensão
privada dos autores é **só a Eq. 6** (formato de abertura, §3.3) — ver `CONTRATO.md:168-170`.

Nossa autorização não é só leitura: o renderer foi **medido** contra o BokehMe real no
job 32212 (`ACHADOS.md:245-290` `[M]`) — disco e não gaussiana
(`edge_width_ratio = 0,143`), linearidade exata em K (resíduo `0,0000 px` no
`bokeh_classical`), escala `slope = 0,9873`.

Vale registrar a ressalva que o próprio paper escreve, `paper.txt:333-334`: a rota A *"is
constrained by **renderer bias** and may introduce unrealistic artifacts"*. O paper trata
a fase sintética como pré-treino de **geometria de CoC**, não de aparência — e o currículo
confirma: 40K passos em sintético, depois 60K em real (`paper.txt:515-516`,
`CONTRATO.md:150`).

### 1.6 O papel do EBB!

`[27]` é **uma das duas fontes de AIF candidata** da rota A, ao lado de `[80]`
(`paper.txt:996`, `527-528`). Não tem outro papel: não é benchmark do paper, não é fonte
das rotas B ou C, não aparece nas Tabs. 2–7. A única outra menção é no related work,
`paper.txt:214` (*"from fixed apertures [27] to variable f-stops"*), como marco
histórico.

Duas coisas que o paper **não** diz sobre o EBB!, e que viram `[A]` (§5):
a EBB! é um dataset **pareado** (bokeh / nítida) e o paper não diz **qual lado** entra no
pool; e não diz a **divisão** do pool de 1,7K entre as duas fontes.

### 1.7 O resto do pipeline da rota A, na ordem

1. **Pool de imagens**: `[80]` + EBB! `[27]`, filtrado por **variância do Laplaciano**
   (`paper.txt:997`: *"similar to the DeblurNet stage, we use Laplacian variance to
   filter out blurry examples"*), resultando em **~1,7K** (`paper.txt:998`).
2. **Profundidade**: *"their **estimated** depth map D"* (`paper.txt:328-329`), pelo
   estimador monocular `[7]` = **Depth Pro** (`paper.txt:314-315`, bibliografia em
   `paper.txt:762-764`).
3. **Sorteio** de `Dfocus` e `K` (`paper.txt:329-330`).
4. **Eq. 2** sobre os dois (`paper.txt:330`, equação em `paper.txt:312`).
5. **BokehMe** produz o alvo (`paper.txt:331`).

Adjacente, e **não é rota A**: o `PointLight-1K` (`paper.txt:1024-1044`) é um conjunto
de 1k imagens noturnas geradas com FLUX + LoRA AntiBlur e passadas pela DeblurNet,
usado **só** para o LoRA de formato de abertura (`paper.txt:1043-1044`). Não entra nos
70K nem no pool de 1,7K.

---

## 2. O que o pipeline original fez — divergências, com os dois lados

Código auditado: `bokehnet-preprocessing/src/pipelines/route_a.py` (231 linhas),
`bokehnet-preprocessing/scripts/build_route_a_manifest.py` (67 linhas),
`bokehnet-preprocessing/src/preprocessing/renderer.py`.

Aviso de escopo, que o próprio repositório histórico registra: **o dataset de ~68K
publicado não saiu deste código.** `bokehnet-preprocessing/docs/ERROS_GERACAO_ORIGINAL_BOKEHNET.md:193`
— *"Não havia uma implementação de rota A no commit original auditado. O dataset
existente foi gerado fora desse caminho"*. Então há **dois** objetos a auditar: o v2 no
working tree (auditável linha a linha) e o dataset histórico (auditável só pelo que
ficou gravado). Os itens abaixo dizem qual é qual.

Ordenados por **impacto no rótulo**.

### A1 — CRÍTICO · A fonte está trocada. `[M]` no código, `[M]` no paper

- **Defeito**: a rota A aceita `DiffCamera` e `EBB!`, e recusa qualquer outra coisa.
- **Código**: `route_a.py:32` — `PAPER_A_SOURCES = {"DiffCamera", "EBB!"}`, aplicado em
  `route_a.py:47-50` levantando `ValueError` para o resto. Repetido em
  `build_route_a_manifest.py:13` (`ALLOWED = {"DiffCamera", "EBB!"}`) e
  `build_route_a_manifest.py:34-35`. A docstring afirma a fidelidade:
  `build_route_a_manifest.py:2` — *"Rank only **paper-approved** DiffCamera/EBB! AIF
  inputs"*; `route_a.py:1` — *"sharp **DiffCamera**/EBB AIFs"*; `route_a.py:201` no
  `--help`.
- **Paper**: `paper.txt:996` e `527-528` dizem `[80]` + `[27]`. `[80]` é Generative
  Photography (`paper.txt:941`). DiffCamera é `[69]`, baseline (`paper.txt:914`, `115`,
  `385`, `1050`).
- **Impacto**: metade do pool é de um dataset que o paper nunca usou como dado. Não
  muda a matemática do rótulo, muda **o que o modelo vê no pré-treino inteiro**, e torna
  a ablação da Tab. 6 (`paper.txt:676-690`) incomparável na coluna (a).
- **Conserto**: trocar as duas constantes para `{"GenerativePhotography", "EBB!"}`,
  corrigir as três docstrings e o `--help`, e gravar a **revisão** de cada fonte na
  proveniência (hoje não é gravada em lugar nenhum). O corte de `--top-n 1700` está
  certo (`paper.txt:998`).

### A2 — CRÍTICO · Uma amostra por imagem, quando o paper tira ~41. `[M]`

- **Defeito**: o laço percorre o manifesto e renderiza **uma** variante por linha.
- **Código**: `route_a.py:138` — `for index, sample in enumerate(tqdm(samples, ...))`,
  com um único `focus_depth_m` (`route_a.py:157`), um único `k` (`route_a.py:158`) e um
  único `renderer(...)` (`route_a.py:164`) por iteração. Não existe parâmetro de
  multiplicidade em `build_parser()` (`route_a.py:200-223`).
- **Paper**: ~70K pares de ~1,7K imagens (`paper.txt:527-528`, `998`) → ~41 `[I]`.
- **Impacto**: **é o pior defeito depois da fonte, e é pior do que "40× menos dado".**
  Com uma variante por imagem, cada conteúdo aparece com **um** K e **um** plano de
  foco. O sinal que o pré-treino existe para ensinar — *"a mesma cena, com K diferente,
  borra diferente"* (`paper.txt:332-333`: *"helps the network modulate the circle of
  confusion according to Ddef"*) — **não existe no dado**. K vira confundido com
  conteúdo.
- **Conserto**: `--samples-per-image`, default 41 `[A]`, com sorteio **estratificado**
  (§4, F4) e `scene_id` compartilhado pelas N variantes (§A9).

### A3 — CRÍTICO · O alvo era gaussiano; e o gaussiano continua selecionável. `[M]`

- **Defeito histórico**: `_render_bokehme` levantava `ImportError` **incondicional**, e
  o fluxo caía num blur gaussiano de 16 camadas, `sigma = layer_defocus × 0.5`, kernel
  limitado a 51 px, sem oclusão e sem scattering.
  Evidência: `bokehnet-preprocessing/docs/ERROS_GERACAO_ORIGINAL_BOKEHNET.md:174-189`;
  `docs/VEREDITO_FINAL_BOKEHNET.md:128-130`. **Os ~70K alvos da fase 1 ensinaram
  gaussiana, não bokeh.**
- **Defeito remanescente no v2**: o gaussiano deixou de ser fallback e virou **opção de
  linha de comando dentro do mesmo caminho de produção** —
  `route_a.py:208` (`--renderer choices=["bokehme","smoke_gaussian"]`),
  `renderer.py:128-129` (`build_renderer("smoke_gaussian")`),
  `renderer.py:94-112` (`GaussianSmokeRenderer`). A única barreira é a string
  `is_final_label_renderer=False` (`renderer.py:100`), que apenas marca a amostra
  (`route_a.py:169-170`) — ela **é gravada assim mesmo**.
- **E o BokehMe do v2 também está bloqueado**: `renderer.py:59` marca o BokehMe público
  como `is_final_label_renderer=False`, então `route_a.py:126-129` exige
  `--allow-unverified-renderer` para rodar, e `route_a.py:169` grava
  `is_valid_for_control = False` em **toda** amostra. Não existia caminho para gerar
  dado final válido.
- **Paper**: `paper.txt:331` e `292` autorizam `[43]` sem ressalva. Nosso renderer novo
  já está verificado em GPU (`ACHADOS.md:245-290` `[M]`).
- **Conserto**: usar `renderer/bokehme.py` (in-process, verificado), com o laudo
  obrigatório como já faz a rota C (`scripts/run_route_c.py:12`,
  `--renderer-report`). **Não portar `smoke_gaussian` para `src/`** — renderer sintético
  vive no teste, decisão já registrada em `REGISTRO.md:301-303`.

### A4 — CRÍTICO · `max_coc` entra por linha de comando e é gravado por amostra. `[M]`

- **Defeito**: `route_a.py:207` — `parser.add_argument("--max-coc", required=True,
  type=float)`; usado em `route_a.py:166` (`compute_defocus_map(..., args.max_coc)`) e
  **gravado como campo** em `route_a.py:177` (`"max_coc": args.max_coc`).
- **Paper / contrato**: a Eq. 2 é crua, sem normalizador (`paper.txt:312`); o
  normalizador vem da inferência oficial e é **global e congelado** —
  `CONTRATO.md:16` (`max_coc = 100.0`) e `CONTRATO.md:42-45`.
- **Impacto**: é exatamente o mecanismo do desastre do `kfix` — uma rota numa convenção,
  outra noutra, no mesmo lote (`ACHADOS.md:191-192`: LF-Bokeh +0,8288 enquanto RealDOF
  ia a −0,4599). Um revisor **mediu** uma amostra com `max_coc = 10.510746` atravessando
  o writer sem erro (`REGISTRO.md:589-600`).
- **Conserto**: nada a fazer no código novo — `MAX_COC` já não é parâmetro de
  `defocus_map` (`src/control/contract.py:433-451`) nem campo de `ControlLabel`
  (`src/dataio/sample.py:104-120`), e `validate_metadata` rejeita qualquer outro valor
  (`src/dataio/sample.py:229-233`). O item existe para que a rota A **não reintroduza**
  o botão.

### A5 — ALTO · K é transportado de B/C sem corrigir a resolução. `[M]` no código

- **Defeito**: `route_a.py:158` — `k = float(rng.choice(k_values))`, com `k_values` lido
  das saídas das rotas B e C (`route_a.py:67-94`). O K é aplicado cru à imagem da rota
  A, que tem outra resolução.
- **Por que é erro de unidade, não de gosto**: `K = k_eq3/1000` e
  `k_eq3 ∝ pixel_ratio = max(H,W)/sensor_mm` (`CONTRATO.md:22-23`;
  `src/control/contract.py:317-332`). Logo **K escala linearmente com a resolução da
  imagem**. O `pixel_ratio` medido na rota B vai de 22,2 a 277,3, mediana 42,6
  (`ACHADOS.md:36` `[M]`) — uma faixa de **12×**. Transportar K de uma imagem de
  BokehDiffusion para uma imagem de Generative Photography sem normalizar aplica um CoC
  em pixel que não corresponde a óptica nenhuma.
- **Contrato**: regra 3 — *"Toda quantidade em pixel carrega a resolução em que foi
  medida"* (`CONTRATO.md:47-50`). A função existe: `k_at_resolution`
  (`src/control/contract.py:458-482`).
- **Impacto**: erro de escala de até uma ordem de grandeza no rótulo, variando por
  amostra, invisível no JSON. É a terceira aparição do mesmo defeito de convenção de
  escala (as outras duas: `REGISTRO.md:95-102` e `REGISTRO.md:633-640`).
- **Conserto**: amostrar numa grandeza **livre de resolução** — `k_value / max(image_h,
  image_w)` — e remultiplicar pelo `max(H,W)` da imagem da rota A. Exige que o
  manifesto carregue a resolução (§4, F3).

### A6 — ALTO · O plano de foco é sorteado no espaço errado, com faixa não justificada. `[M]`

- **Defeito**: `route_a.py:157` —
  `focus_depth_m = float(np.quantile(metric_depth.values_m, rng.uniform(q_low, q_high)))`
  com `q_low, q_high` default `0.10, 0.90` (`route_a.py:213-214`).
- **Dois problemas distintos**:
  1. **Espaço.** O sorteio é uniforme no quantil da **profundidade métrica**, e o
     controle vive em **disparidade** (`CONTRATO.md:28-34`). Numa cena com céu, 25,7%
     das amostras medidas têm `z_max == 10.000 m` (`ACHADOS.md:54` `[M]`): boa parte da
     massa de quantis de `z` cai no fundo, onde `1/z ≈ 0` e onde `|Δdisp|` — logo o CoC
     — quase não varia. O resultado é uma distribuição de planos de foco concentrada
     justamente onde o sorteio não produz sinal.
  2. **Unidade primária.** O código só produz `focus_depth_m`, e a disparidade é
     reconstruída duas vezes a jusante: `renderer.py:71`
     (`focus_norm = ((1.0/focus_depth_m) - d_min)/(d_max-d_min)`) e `route_a.py:100`
     (`delta = np.abs(disparity - (1.0/focus_depth_m))`). O contrato manda o campo
     primário ser `focus_disparity` e `focus_depth_m` existir *só para leitura humana*
     (`CONTRATO.md:36-40`; `src/dataio/sample.py:122-125`).
- **Paper**: `paper.txt:329-330` diz *"randomly sample a focus plane"* e **cala sobre a
  distribuição**. Então `[0.10, 0.90]` é `[A]` — e nada no repositório histórico o mede.
- **Conserto**: sortear em **disparidade**, em estratos, e gravar `focus_disparity` como
  campo primário. A faixa continua `[A]`, declarada.

### A7 — ALTO · Máscara sintética gravada em três campos, como se fosse medida. `[M]`

- **Defeito**: `route_a.py:97-102` constrói `_focus_band_mask` — uma banda de
  disparidade com tolerância no **quantil 0,12**, constante mágica sem justificativa
  (`route_a.py:101`). O mesmo array é gravado em **três** chaves com semânticas
  diferentes: `foreground_mask`, `focus_mask_auto`, `focus_mask_final`
  (`route_a.py:175-176`). A única proveniência é a string
  `quality["synthetic_focus_mask"] = "depth_band_q12"` (`route_a.py:168`), dentro de um
  dict de qualidade livre.
- **Paper**: na rota A **não há máscara**. O plano de foco é sorteado (`paper.txt:329-330`);
  a Eq. 4 (`paper.txt:352`) e o BiRefNet `[86]` pertencem a (b) e (c)
  (`paper.txt:348-350`, `361-362`).
- **Impacto**: baixo no rótulo (a máscara não entra em `Ddef`), **alto na auditabilidade**
  — três campos afirmando três coisas para um único array é proveniência que mente, e a
  própria docstring do código admite que a máscara existe *"only for auditability"*
  (`route_a.py:98`).
- **Conserto**: ou não gravar máscara na rota A, ou gravar **uma** com
  `mask_source = MaskSource.DEPTH_BAND` (que já existe: `src/dataio/sample.py:62`) e a
  regra completa — quantil, tolerância, seed — na proveniência. Ver F6.

### A8 — MÉDIO · O filtro Laplaciano ranqueia globalmente entre fontes (D13). `[M]`

- **Defeito**: `build_route_a_manifest.py:45-46` — todos os candidatos das duas fontes
  vão para uma lista só, ordenada por `laplacian_variance` decrescente, e o corte é
  `candidates[:args.top_n]` **global**.
- **Por que enviesa**: a variância do Laplaciano escala com resolução e com compressão,
  então um ranking único entre EBB! e Generative Photography seleciona pela fonte de
  maior resolução, não pela mais nítida. Isso já está escrito no nosso gate novo:
  `src/qc/gates.py:206-210` — *"Só é comparável DENTRO de uma mesma fonte [...]
  ranquear EBB! contra Generative Photography num ranking único enviesa a seleção"*.
- **Paper**: `paper.txt:997` diz apenas *"we use Laplacian variance to filter out blurry
  examples"* e dá o resultado (`paper.txt:998`, ~1,7K). **Não diz** se o ranking é por
  fonte. Ranquear por fonte é decisão nossa; ranquear globalmente é o que é
  demonstravelmente enviesado.
- **Detalhe menor, mesmo arquivo**: `build_route_a_manifest.py:38` passa `top_n=10**12`
  para `filter_by_laplacian` a fim de "ranquear tudo", o que faz o CSV de ranking marcar
  **todas** as linhas como `selected` (`src/preprocessing/genrefocus.py:174`). O CSV
  auxiliar mente sobre a seleção; o manifesto final está certo.
- **Conserto**: ranquear e cortar **dentro de cada fonte**, com quota por fonte
  declarada `[A]` (§5), e gravar o corte resultante por fonte.

### A9 — MÉDIO/ALTO · Proveniência e ausência de `scene_id` (P2). `[M]`

- **Histórico**: o dataset publicado gravava **só UUID** como nome
  (`ACHADOS.md:23` `[M]`), e o repositório lista o que ficou impossível certificar —
  origem exata das AIFs, revisão de cada fonte, corte do Laplaciano, se o ranking foi
  por fonte, sobreposição com benchmark, e quais `(K, Dfocus)` geraram cada variante
  (`bokehnet-preprocessing/docs/ERROS_GERACAO_ORIGINAL_BOKEHNET.md:193-201`).
- **v2**: melhora — grava `source_path` e `source_sample_id` (`route_a.py:182-183`) —
  mas o identificador da amostra é `stem = f"a_{...}_{index:06d}"` (`route_a.py:139`),
  ou seja **o índice na lista**: reordenar o manifesto renomeia todas as amostras.
- **E não existe `scene_id` em lugar nenhum** do `components` (`route_a.py:171-186`).
  Com uma variante por imagem isso passava despercebido; **com 41 variantes por imagem
  é vazamento garantido** — a mesma imagem, com 41 K diferentes, cairia dos dois lados
  de um split por amostra, e a validação mediria memorização.
- **Contrato do projeto**: *"Split por cena, materializado no dataset"*
  (`CLAUDE.md:50`); *"Contagens em cenas E em amostras"* (`CLAUDE.md:51-52`).
- **Conserto**: `scene_id` = id da **imagem de origem**, compartilhado pelas N
  variantes; `sample_id` = `<scene_id>_v<NN>`; sha256 do arquivo de origem e revisão da
  fonte na proveniência. Sem isso o split é ficção.

### A10 — MÉDIO · Sem vocabulário de rejeição; o histograma não distingue nada. `[M]`

- **Defeito**: `route_a.py:190-192` — `except Exception as exc:` incrementa
  `stats["error"]` e grava a string do erro. O resumo final é
  `print(f"[route-a] finished: {stats}")` com três contadores (`route_a.py:135, 197`):
  `ok`, `rejected`, `error`.
- **Regra**: *"Todo run termina imprimindo histograma de motivos de rejeição. Sem ele
  não dá para calibrar limiar nenhum, e é ele que denuncia fallback novo"*
  (`CLAUDE.md:44-45`).
- **Conserto**: `qc.rejection.RejectionLog` + slugs do conjunto fechado, como a rota C
  (`src/routes/route_c.py:328, 350-355, 361-363`).

### A11 — MÉDIO · Grava profundidade normalizada ao lado da métrica. `[M]`

- **Defeito**: `route_a.py:173` — `"depth": metric_depth.normalized` **e**
  `"depth_metric_m": metric_depth.values_m`. Duas representações da mesma grandeza no
  disco, uma delas normalizada por imagem.
- **Medido no dataset v0**: a coluna `depth` é profundidade métrica min-max, não
  disparidade (`ACHADOS.md:25` `[M]`), e 24,7% das amostras tinham a cena útil em menos
  de 256 níveis de 65535 (`ACHADOS.md:56` `[M]`).
- **Conserto**: já resolvido no código novo — `dataio/encoding.py` grava **uma** coisa,
  uint16 linear em disparidade (`src/dataio/encoding.py:1-38, 104-147`). Item existe
  para não regredir.

### A12 — BAIXO · Gate de nitidez com limiar que nunca reprova. `[M]`

- `route_a.py:215` — `--min-aif-laplacian-variance` default **0.0**, consumido por
  `evaluate_aif_sharpness` (`route_a.py:145`). Variância do Laplaciano é sempre ≥ 0, logo
  o gate **não pode reprovar**. É o padrão "gate que não pode reprovar" já catalogado em
  `REGISTRO.md:695-696`.
- **Conserto**: o `GateResult` novo usa `threshold=None` para *medir sem bloquear*
  (`src/qc/gates.py:57-59`), que é honesto e distingue os dois casos. Usar isso.

### A13 — BAIXO · Um `subprocess` por render, 70K vezes. `[M]`

- `renderer.py:74-86` — cada chamada escreve dois PNGs num `TemporaryDirectory`, sobe
  `python demo.py` e relê um JPEG. Para 70K renders isso é o mesmo defeito de custo já
  medido na rota C (~40 processos por amostra, `REGISTRO.md:396`).
- Agravante de fidelidade: o disco vai e volta por PNG **uint8 de disparidade**
  (`renderer.py:80`), quantizando o sinal de controle em 256 níveis antes de renderizar.
- **Conserto**: já existe — `renderer/bokehme.py` é in-process e monta o tensor
  `defocus` direto do CoC canônico, sem PNG (`src/renderer/bokehme.py:329-331`, e a
  justificativa em `src/renderer/bokehme.py:13-16`).

---

## 3. O que já existe no código novo, e serve

Auditado: `src/control/contract.py` (548), `src/renderer/` (4 arquivos),
`src/qc/gates.py` (334), `src/dataio/` (4 arquivos), `src/routes/route_c.py` (366),
`src/sources/realbokeh.py` (732) + `src/sources/mirror_images.py` (305).

### 3.1 Serve como está — reuso direto

| peça | onde | por que serve na rota A |
|---|---|---|
| `MAX_COC` congelado, fora de assinatura | `src/control/contract.py:47, 433-451` | fecha A4 por construção |
| `signed_coc_px` / `defocus_map` | `src/control/contract.py:418-451` | é a Eq. 2 (`paper.txt:312`), que vale na A |
| `k_at_resolution` | `src/control/contract.py:458-482` | é o conserto de A5 |
| `k_for_bokehme` | `src/control/contract.py:485-502` | para quem usar a CLI; o adaptador in-process não precisa |
| `validate_metric_depth` | `src/control/contract.py:196-231` | Depth Pro sobre a AIF, com `backend` obrigatório |
| `reject` + `REJECTION_REASONS` | `src/control/contract.py:126-157` | vocabulário fechado; fecha A10 |
| `BokehMeRenderer` + `render_fn_from` | `src/renderer/bokehme.py:199-355` | **é o `[43]` de `paper.txt:331`**, in-process, saída float, proveniência com commit + 4 sha256 |
| harness de verificação | `src/renderer/verification.py` | o laudo que autoriza o renderer; já passou em GPU (`ACHADOS.md:245`) |
| `encode_depth` | `src/dataio/encoding.py:104-147` | uint16 linear em disparidade, `image_hw` obrigatório, extremos da profundidade cheia |
| `KSource.SAMPLED_FROM_BC` | `src/dataio/sample.py:68` | **já existe, criado para a rota A** |
| `MaskSource.DEPTH_BAND` | `src/dataio/sample.py:62` | **já existe**, com o comentário `# rota A: banda de profundidade sintética` |
| `generated_images` + `channel_order` | `src/dataio/sample.py:139-142`, `src/dataio/writer.py:63, 118-128` | o caminho de gravar a **bokeh renderizada** já está previsto (`src/dataio/sample.py:11`) |
| `validate_metadata` | `src/dataio/sample.py:208-254` | exige `calibration_ssim` **só** quando `k_source == EQ5` (linha 253) — a rota A passa sem |
| `build_scene_split` + `check_no_leak` | `src/dataio/split.py:77-91, 181` | a rota A não tem split de origem → sorteio determinístico por cena |
| `RejectionLog` | `src/qc/rejection.py:24-121` | histograma obrigatório |
| `laplacian_variance` | `src/qc/metrics.py` | é o filtro do paper (`paper.txt:997`) |
| `aif_sharpness` | `src/qc/gates.py:202-210` | com o aviso de comparabilidade por fonte **já escrito** — é o conserto de A8 documentado |
| `depth_useful_levels` | `src/qc/gates.py:217-228` | cena degenerada |
| `focus_depth_plausible` | `src/qc/gates.py:231-253` | plausibilidade do plano sorteado |
| `bokeh_is_blurrier_than_aif` | `src/qc/gates.py:267-285` | na A vira o gate de *"o render fez alguma coisa?"* |

### 3.2 Serve como **modelo estrutural**, a copiar em forma e não em conteúdo

- **`src/routes/route_c.py`**: o `Protocol` de fonte (`:69-91`) em vez de import
  concreto; `RouteCConfig` com **todo limiar `None`** e o porquê (`:98-123`);
  `GATE_TO_REASON` como ponte entre nome de gate e slug fechado (`:196-210`);
  `enforce_gates` (`:213-216`); o laço com retomada por `completed_ids` (`:330-340`) e o
  `finally` que imprime **três** resumos, sempre (`:360-365`). A rota A deve ter a mesma
  esqueleto com `AifImage` no lugar de `PairSource`.
- **`src/sources/realbokeh.py` + `mirror_images.py`**: a fronteira que faltava —
  enumerar/validar (`realbokeh.py`) separado de carregar pixels (`mirror_images.py`), com
  a justificativa em `src/sources/__init__.py:6-10`. Mais: slugs `source_*` registrados
  no contrato e não localmente (`REGISTRO.md:724-729`); `sample_pairs_for_pilot` que
  **sorteia cenas, não pares** (`mirror_images.py:280`, e o porquê em `REGISTRO.md:736-740`)
  — na rota A isso é literalmente obrigatório, porque 41 variantes de uma imagem são uma
  cena só.
- **`scripts/run_route_c.py`**: a ordem do entrypoint — laudo do renderer → modelos →
  fonte → split → run (`:1-36`), e o `_ComSensor` (`:67-88`) como padrão de *"assumir um
  valor e marcá-lo como assumido, em vez de cravá-lo no laço"*.
- **`scripts/calibrate_thresholds.py`**: lê o piloto e **propõe** limiares sem aplicar
  nenhum (`REGISTRO.md:715`).

### 3.3 O que falta — descrito, não escrito

Nenhum destes arquivos foi tocado nesta rodada. `src/control/`, `src/dataio/`,
`src/routes/`, `src/sources/`, `src/renderer/` e `src/qc/` estão **intocados**.

**F1 · `src/sources/genphoto_ebb.py`** — adaptador de fonte, novo arquivo.
Enumera imagens de dois diretórios locais, uma fonte por vez; calcula
`laplacian_variance` **por fonte**; ranqueia e corta **dentro de cada fonte** com quota
declarada; emite um registro por imagem com `scene_id`, `source_dataset`,
`source_sample_id`, `image_ref`, `sha256`, `image_hw`, `laplacian_variance`,
`source_revision`. Fecha A1, A8 e metade de A9. Reusa `qc.metrics.laplacian_variance`.
Fonte fora do conjunto é **erro de configuração** (`SystemExit` no entrypoint), não
rejeição de amostra — não polui o histograma.

**F2 · `validate_focus_disparity` em `src/control/contract.py`** — *mudança descrita, não
aplicada.* Hoje só existe `focus_disparity_from_mask` (`:247-284`), que exige máscara. A
rota A **sorteia** o plano e não tem máscara (`paper.txt:329-330`). Falta uma função que
receba um `focus_disparity` já sorteado e aplique as mesmas validações — finitude,
positividade, e a faixa `FOCUS_DEPTH_MIN_M`/`MAX_M` (`:243-244`). Os dois slugs
necessários **já estão registrados**: `focus_disparity_invalid` e
`focus_depth_implausible` (`:131-132`). É extrair o corpo de `:276-284` para uma função
pública e chamá-la dos dois lados — nenhum slug novo, nenhuma constante nova.

**F3 · Amostrador de K livre de resolução** — módulo novo. Lê os `manifest.jsonl` das
rotas B e C, filtra `is_valid_for_control and not is_k_censored`, converte cada K para
`k_por_lado_longo = k_value / max(image_h, image_w)`, e reamostra multiplicando pelo
`max(H,W)` da imagem da rota A. Fecha A5.
**Bloqueador descoberto**: a linha do manifesto **não carrega `image_h`/`image_w`**
(`src/dataio/writer.py:137-151`) — eles existem só no `meta/<id>.json`
(`src/dataio/encoding.py:80-81`). *Mudança descrita*: acrescentar os dois campos à linha
do manifesto (é a superfície de varredura rápida, e um K sem resolução é exatamente o
que ela deveria denunciar), ou o amostrador lê os `meta/`, ao custo de 70K arquivos.
Recomendo o primeiro.

**F4 · Estratificação do sorteio** — dentro da rota A. 41 sorteios i.i.d. por imagem
deixam imagens inteiras sem K alto; o que ensina controle é a **cobertura por imagem**.
Proposta: estratos em quantis da distribuição empírica de K (B+C) × estratos de plano de
foco em **disparidade**, um sorteio dentro de cada célula, seed derivada do `scene_id`
para reprodutibilidade. O paper diz *"randomly sample"* (`paper.txt:329-330`) e cala →
**é decisão nossa, declarada**, como o `sampled_from_bc` já é (`CONTRATO.md:163-166`).

**F5 · `src/routes/route_a.py`** — a rota. Por imagem: Depth Pro → N sorteios de
`(focus_disparity, K)` → `signed_coc_px` → BokehMe → gates → writer com
`generated_images={"bokeh": ...}`, `scene_id` compartilhado pelas N variantes.
`encode_depth` roda **uma vez por imagem** e é reusado pelas N variantes — a
profundidade não depende do sorteio, e recomputá-la 41 vezes é 41× de GPU jogada fora.
*Atenção*: isso significa que 41 amostras compartilham o mesmo PNG de profundidade;
gravar 41 cópias idênticas é o dobro do orçamento de disco (§F8). Decidir e declarar.

**F6 · Proveniência sem modelo de máscara** — *mudança descrita.*
`SampleProvenance.mask_model_sha256` é **obrigatório** (`src/dataio/sample.py:88`) e
`validate_metadata` rejeita string vazia (`src/dataio/sample.py:204-205, 249-251`, via
`not prov.get(f)`). **A rota A não roda BiRefNet** — não há hash de modelo de máscara
para gravar. Duas saídas, e a segunda é melhor:
(i) tornar o campo opcional quando `mask_source == DEPTH_BAND` — afrouxa a regra para
todo mundo;
(ii) exigir **um dos dois**: `mask_model_sha256` **ou** `mask_rule` — um dict com a regra
sintética completa (quantil, tolerância, seed). A regra é o análogo exato do hash: é o
que permite reconstruir a máscara. Não afrouxa a rota C.
Fecha A7.

**F7 · Gates específicos da rota A** — nenhum existe hoje.
- **`render_is_not_noop`**: com K pequeno e plano de foco cobrindo a cena, o BokehMe
  devolve quase a AIF, e a amostra não carrega sinal. Medir
  `bokeh_is_blurrier_than_aif` (`src/qc/gates.py:267`) já dá o número; falta o limiar
  `[A]`.
- **`rendered_coc_p99_px`**: p99 de `|CoC|` em pixel. Abaixo de ~1 px o alvo é
  indistinguível da entrada. Âncora existente: `coc_p99_px` mediano da rota B é
  **4,665 px** (`ACHADOS.md:37` `[M]`). Limiar `[A]`.
- **`defocus_saturation_ratio`**: fração de pixels com `defocus == 1.0`, isto é
  `|CoC| ≥ MAX_COC = 100 px`. Medido na rota B com `max_coc = 10,5107`: 17,3%
  (`ACHADOS.md:57` `[M]`) — **com 100 é outro regime, e não foi medido**. Na rota A o K
  é sorteado, então a saturação é escolha nossa e tem que ser vista no piloto.
- **Não se aplicam à rota A**: `focus_mask_is_sharpest`, `mask_iou`,
  `aif_aperture_is_narrow` (não há máscara real nem f-stop) e
  `calibration_ssim_is_reliable` (não há Eq. 5). O `GATE_TO_REASON` da rota A é um
  subconjunto diferente do da C.

**F8 · Orçamento de disco — a rota A é a que aperta.** Pela fórmula do próprio
`estimate_disk_budget` (`src/dataio/writer.py:201-219`), com 70K amostras e lado longo
768:

| resolução da imagem | controle | bokeh JPEG | total |
|---|---|---|---|
| 0,5 MP | 38,4 GB | 22,1 GB | **60,4 GB** |
| 1,0 MP | 38,4 GB | 44,1 GB | **82,5 GB** |
| 2,0 MP | 38,4 GB | 88,2 GB | **126,6 GB** |

Contra **115 GB de folga**, dos quais as rotas B+C já ocupam **31 GB**
(`REGISTRO.md:585`) → sobram ~84 GB. **A rota A cabe a ≤1 MP e não cabe a ≥2 MP.**
A resolução das imagens de `[80]` e da EBB! **não foi medida** → `[A5]` da §5.
Duas coisas fora da conta: a máscara é gravada em resolução de **imagem**
(`src/dataio/writer.py:109-112`) e não entra na fórmula; e as 41 cópias de profundidade
por imagem (F5) dobrariam a coluna de controle se gravadas.

**F9 · Custo de GPU — não medido.** O job 32212 mediu o **contrato** do renderer
(`ACHADOS.md:245-290`), não o throughput. 70K renders in-process do BokehMe é a única
etapa do projeto sem estimativa, e o plano já registra isso
(`genrefocus_deblurnet_paper/PLANO_REGERACAO_BOKEHNET.txt:746-748`). Medir no piloto,
antes de dimensionar `--time`.

---

## 4. O que temos que mudar — lista priorizada

Cada item: **defeito · evidência · conserto**. Ordem por impacto no rótulo.

| # | prioridade | item | conserto |
|---|---|---|---|
| 1 | CRÍTICO | fonte trocada (A1) | novo `sources/genphoto_ebb.py`; `{GenerativePhotography, EBB!}`; revisão da fonte na proveniência |
| 2 | CRÍTICO | uma variante por imagem (A2) | `samples_per_image = 41` `[A]`, estratificado (F4) |
| 3 | CRÍTICO | alvo gaussiano / BokehMe bloqueado (A3) | `renderer/bokehme.py` verificado, laudo obrigatório; **não** portar `smoke_gaussian` para `src/` |
| 4 | CRÍTICO | vazamento por falta de `scene_id` (A9) | `scene_id` = imagem; `sample_id = <scene>_v<NN>`; `build_scene_split` sobre cenas |
| 5 | ALTO | K transportado sem resolução (A5) | amostrar `k/max(H,W)`; **exige** `image_h/w` no manifesto (F3) |
| 6 | ALTO | plano de foco sorteado em `z`, não em disparidade (A6) | sortear em disparidade, estratificado; `focus_disparity` como campo primário |
| 7 | ALTO | falta `validate_focus_disparity` no contrato (F2) | extrair o corpo de `contract.py:276-284`; zero slug novo |
| 8 | ALTO | `mask_model_sha256` obrigatório e inexistente na A (F6) | exigir `mask_model_sha256` **ou** `mask_rule` |
| 9 | MÉDIO | ranking Laplaciano global entre fontes (A8) | ranquear e cortar por fonte, com quota declarada |
| 10 | MÉDIO | máscara sintética em três campos (A7) | uma máscara, `MaskSource.DEPTH_BAND`, regra na proveniência |
| 11 | MÉDIO | sem histograma de rejeição (A10) | `RejectionLog` + `GATE_TO_REASON` próprio da rota A |
| 12 | MÉDIO | gates específicos ausentes (F7) | `render_is_not_noop`, `rendered_coc_p99_px`, `defocus_saturation_ratio`, todos `threshold=None` no piloto |
| 13 | MÉDIO | orçamento de disco no limite (F8) | medir a resolução das fontes **antes** de rodar; decidir sobre as 41 cópias de profundidade |
| 14 | BAIXO | `max_coc` por CLI (A4) | já fechado por construção no código novo; não reintroduzir |
| 15 | BAIXO | profundidade normalizada gravada (A11) | já fechado por `dataio/encoding.py` |
| 16 | BAIXO | limiar que nunca reprova (A12) | `threshold=None` do `GateResult` |
| 17 | BAIXO | subprocess por render (A13) | já fechado pelo adaptador in-process |

Ordem de execução sugerida: **1 → 4 → 2 → 5 → 6/7 → 3 → resto**. Os itens 1 e 4 são de
identidade do dado e não têm conserto retroativo; o 2 depende do 4 para não virar
vazamento; o 5 depende do 3 (manifesto de B/C corrigido) para ter distribuição de K que
valha alguma coisa.

**Pré-requisito duro**: o amostrador de K lê as rotas B e C. Hoje a rota B publicada tem
`k = 50,0` em 11.635/11.635 (`ACHADOS.md:14` `[M]`) e a rota C tem 47,0% no teto exato
300 (`ACHADOS.md:19` `[M]`). **Amostrar dessa "distribuição" hoje é amostrar de duas
constantes.** A rota A não pode rodar antes de B e C regeradas — isso é uma dependência
de ordem, não uma preferência.

---

## 5. `[A]` em aberto — o que o paper não publica

Nenhum destes tem número no paper. Cada um traz o que **mediria** para resolver.

| id | `[A]` | o que resolveria |
|---|---|---|
| A1 | **Qual artefato de `[80]`** — o conjunto de imagens publicado pelos autores da Generative Photography, ou imagens que nós geramos com o modelo deles. `paper.txt:996` diz *"candidate images from [80] ... collections"* e não dá revisão nem contagem. | Baixar o release público, contar as imagens, e ver se o pool sobrevivente com a EBB! chega a ~1,7K (`paper.txt:998`). Se não chegar, a leitura está errada. |
| A2 | **Qual lado da EBB!** A EBB! é pareada (bokeh/nítida) e o paper não diz qual entra no pool. | Medir a variância do Laplaciano dos dois lados; ver qual sobrevive ao corte que produz ~1,7K. |
| A3 | **O corte do filtro Laplaciano**, e se o ranking é por fonte ou global. `paper.txt:997` só diz *"filter out blurry examples"*. | Escolher o corte **por fonte** que faz o pool somar ~1,7K, e registrar o valor resultante de cada fonte. Não inventar limiar antes de ver o histograma. |
| A4 | **A divisão do pool de 1,7K entre `[80]` e EBB!** Não publicada. | Nenhuma medição resolve — o paper não publica. Escolher, declarar a quota, e gravar por amostra qual fonte a gerou. |
| A5 | **N por imagem = 41.** É `[I]` de `70K/1,7K` (`paper.txt:527-528`, `998`), não número do paper. | Nada resolve; o que dá é gravar N e o par `(K, Dfocus)` de cada variante, que é o que o histórico não fez (P2). |
| A6 | **A distribuição de K e de `Dfocus`.** `paper.txt:329-330` diz só *"randomly sample"*. `sampled_from_bc` é decisão nossa (`CONTRATO.md:163-166`). | O paper não publica e não vai publicar. O que dá é **medir e publicar a nossa**: a distribuição de K das rotas B e C **depois de corrigidas** vira o anexo que declara o desvio. |
| A7 | **Se K deve ser normalizado por resolução ao migrar de B/C para A.** O paper cala; a regra 3 do contrato (`CONTRATO.md:47-50`) diz que sim. | Decisão nossa, com fundamento físico. Declarar. O teste que a trava: mesma imagem em duas resoluções tem que dar o mesmo CoC relativo. |
| A8 | **Resolução das imagens das duas fontes** → decide se a rota A cabe no disco (F8). | `identify`/PIL sobre o pool, um `Counter` de `(H,W)`. Custa minutos e é bloqueante para dimensionar o run. |
| A9 | **Custo de GPU por render do BokehMe.** Não medido: o job 32212 mediu contrato, não throughput. | Cronometrar o piloto. É a única etapa do projeto sem estimativa (`PLANO_REGERACAO_BOKEHNET.txt:746-748`). |
| A10 | **Sobreposição entre o pool da rota A e os benchmarks.** Os quatro benchmarks (LF-Bokeh, RealBokeh, RealDOF, DPDD) não incluem EBB! nem `[80]`, mas **dedup por hash nunca foi feito** — é o P2 (`PLANO_REGERACAO_BOKEHNET.txt:530-534`). | sha256 exato + hash perceptual do pool contra os quatro conjuntos de avaliação. Enquanto não for feito, a ressalva vai no **texto**, não só no repositório. |
| A11 | **Se a rota A deve gravar máscara alguma.** O paper não usa máscara na (a). | Decisão nossa. A alternativa honesta a `DEPTH_BAND` é não gravar campo de máscara e deixá-lo ausente — ausente é mais verdadeiro que uma banda sintética. |
| A12 | **Limiares dos três gates novos da rota A** (F7): `render_is_not_noop`, `rendered_coc_p99_px`, `defocus_saturation_ratio`. | Piloto com `threshold=None` nos três, depois `scripts/calibrate_thresholds.py`. Nenhum entra com número antes do histograma. |
| A13 | **Se as 41 variantes compartilham um PNG de profundidade** ou gravam 41 cópias (F5). | Não é medição, é decisão de armazenamento — mas muda o orçamento por um fator 2 e tem que ser declarada antes do run. |

Herdados e **já resolvidos** em outra etapa, listados para não serem redecididos: os sete
parâmetros do BokehMe estão congelados em `BokehMeConfig` e gravados
(`REGISTRO.md:558-560`); `highlight` é inerte no nosso caminho, com o porquê medido
(`ACHADOS.md:288-290`); `k_effective_factor = 0,9873` está gravado e não corrigido
(`ACHADOS.md:279-283`) — na rota A ele importa como na rota B, porque o K é imposto e
não ajustado por SSIM.

---

## 6. Uma pergunta que fica em aberto, e não é `[A]`

O paper trata a rota A como pré-treino de **geometria**, e diz na cara que ela carrega
viés de renderer (`paper.txt:333-334`). A ablação da Tab. 6 (`paper.txt:676-690`) diz que
(a) sozinha *"provides a foundational performance but leaves room for improvement"*.

Nossa medição contradiz a ablação: hoje `a+c` ganha de `a+b+c` nas quatro mesas
(`ACHADOS.md:191-193`), e a fase 1 só-sintético tem a **melhor** LVCorr medida
(+0,9059 no LF-Bokeh, `ACHADOS.md:189`) — melhor que os pesos oficiais. A hipótese
registrada é que a rota B atrapalhava por causa do `k = 50` constante. **Se a rota A
sozinha já dá +0,9059 com alvo gaussiano, o ganho de rerenderizar 70K alvos com BokehMe
é exatamente a quantidade não medida** — e é a decisão 05 do plano, "segurar a rota A
até medir se ela é o gargalo" (`PLANO_REGERACAO_BOKEHNET.txt:180-186`).

Esta auditoria não muda essa ordem. Ela diz **o que a rota A tem que ser quando chegar a
vez dela**, e diz que quatro dos itens — fonte, multiplicidade, `scene_id` e escala de K
— não têm conserto retroativo: se rodar errado, roda de novo inteiro.
