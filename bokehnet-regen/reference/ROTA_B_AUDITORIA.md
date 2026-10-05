# Auditoria da ROTA B — §3.2(b), ITW dataset

Data: 2026-09-10. Autoridade: `reference/paper.txt` (arXiv:2512.16923v3), depois
`third_party/Genfocus/Inference_bokehNet.py`, depois decisão nossa declarada.
Toda afirmação carrega `paper.txt:<linha>` ou `<arquivo>:<linha>`. Onde não consegui
medir, está escrito **não medido**.

**Nenhum código foi escrito nem editado.** Este documento descreve as mudanças; não as
aplica. `bokehnet-preprocessing/` e `src/{control,dataio,routes,sources,renderer,qc}/`
foram apenas lidos.

Convenção de etiqueta, igual a `ACHADOS.md`: `[M]` medido · `[I]` inferido de algo
medido · `[A]` assumido e não verificado.

---

## 0. Sumário executivo

A cadeia causal do enunciado **confirma-se inteira**, e a auditoria acrescenta um
defeito de impacto maior que os três já catalogados: a rota B rodava a DeblurNet
**com a convenção de adapter errada**, o que corrompe a AIF — e a AIF é a raiz de
tudo o que vem depois (profundidade, máscara, `D_focus`, K, e o próprio condicionamento
da BokehNet).

Ordem de impacto no rótulo, resumida (detalhe em §3):

| # | defeito | efeito no rótulo |
|---|---|---|
| B1 | DeblurNet sem `main_adapter` com checkpoint main+cond | AIF LAVADA → **todo** o rótulo contaminado |
| B2 | fator 1000× (`k_eq3` em px·mm contra disparidade em 1/m) | CoC 1000× maior → mapa satura → K apagado |
| B3 | `pixel_ratio` com a largura em vez de `max(H,W)` | K subestimado por `W/H` em retrato |
| B4 | `max_coc` é argumento de CLI, um por rota | **é o mecanismo exato do kfix** |
| B5 | `focus` mediana da profundidade, não da disparidade | viés pequeno e sistemático em `D_focus`, logo em K |
| B6 | sensor 36 mm assumido quando falta `focal_length_35` | K errado por até 5,6× quando dispara |
| B7 | cascata de máscara com `except` nu, `mask_source="automatic"` | `D_focus` de um GrabCut, proveniência mentindo |
| B8 | profundidade gravada como métrica min-max | 24,7% da cena útil em < 256 níveis [M] |
| B9 | `import json` ausente | **NameError na 1ª amostra** — o run não roda |

---

## 1. O que o paper determina sobre a rota B

### 1.1 A fonte de imagens

O paper chama de **"ITW dataset [19]"** (paper.txt:325, 336). A referência [19] é:

> `19. Fortes, A., Wei, T., Zhou, S., Pan, X.: Bokeh diffusion: Defocus blur control in
> text-to-image diffusion models. In: SIGGRAPH Asia (2025)` — paper.txt:791-793

O que o paper afirma sobre esse dado:

- *"we leverage ITW dataset [19] contains **real bokeh images**"* — paper.txt:336-337.
- *"all data consists of **real optical bokeh images captured directly by digital
  cameras**, rather than artificial defocus synthesized through computational aperture
  techniques"* — paper.txt:1110-1112.
- *"This dataset is utilized to estimate the image bokeh level K **via EXIF metadata**"*
  — paper.txt:1109.
- Volume: *"13K previously filtered and verified images from the ITW dataset [19]"*
  dentro dos ~26K reais — paper.txt:999-1001. E §4.1: *"approximately 26K real examples
  sourced from ITW dataset [19], RealBokeh [57], and LFDOF [52]"* — paper.txt:528-529.

**O paper não publica URL nem nome de repositório.** Que o ITW seja
`atfortes/BokehDiffusion` no HF é `[I]` — forte, por três razões: o autor de [19] é
*Fortes, A.* (paper.txt:791), o handle é `atfortes`, e as **13.800 linhas** que passam
o filtro da rota B [M, `ACHADOS.md:151-158`] casam com os "13K" do paper.txt:1000. É a
mesma escolha que o pipeline antigo já fazia (`route_b.py:39-43`), e não encontrei
alternativa melhor. Fica registrado como inferência, não como leitura.

### 1.2 Como K é obtido — Eq. 3, e o escopo dela

```
paper.txt:340   K ≈ f² · D_focus / ( 2 · F · (D_focus − f) ) × pixel_ratio      (Eq. 3)
```

Termos, com as unidades que o paper dá:

| termo | o que é | onde o paper diz |
|---|---|---|
| `f` | distância focal, **da EXIF** | paper.txt:342-343 |
| `F` | f-number da abertura, **da EXIF** | paper.txt:342-343 |
| `D_focus` | plano de foco, **da Eq. 4**, NÃO da EXIF | paper.txt:344-352 |
| `pixel_ratio` | **maior lado da imagem ÷ largura física do sensor, em px/mm** | paper.txt:1185-1187 |

A definição de `pixel_ratio` é publicada, e é explícita, na legenda da Fig. 16:

> *"Fig. 16: Pixel ratio distribution. The percentage distribution of the
> pixel-to-sensor ratio across ITW dataset. This ratio, defined as **the image's largest
> edge length divided by the physical sensor width (px/mm)**."* — paper.txt:1185-1187

Isto **refuta diretamente** o uso da largura no pipeline antigo (§3, defeito B3) e
**confirma** a implementação nova (`src/control/contract.py:317-332`). No corpo do
artigo a frase é vaga — *"accounts for differences in camera sensor size and image
resolution"* (paper.txt:343-344) —; a legenda da figura é que fecha.

**Escopo: a Eq. 3 vale só na rota B.** O paper coloca a Eq. 3 dentro do parágrafo
"(b) ITW dataset" (paper.txt:336-352), e abre o parágrafo (c) dizendo que aqueles
datasets *"omit EXIF metadata or provide insufficient fields to estimate K"*
(paper.txt:359-360) — por isso a rota C usa a Eq. 5. Confirma `CONTRATO.md:158`.

O paper **não publica**:
- as unidades de `f` e `D_focus` na Eq. 3 (mm e mm é a única combinação que fecha
  dimensionalmente com `pixel_ratio` em px/mm — ver §4.2);
- se o resultado é raio ou diâmetro (o `2F` no denominador indica **raio** — e é o que
  o scatter clássico do BokehMe espalha);
- o que fazer quando a EXIF falta.

### 1.3 O papel da DeblurNet, e a ordem

Sim, **a DeblurNet gera a AIF a partir da imagem com bokeh**, e é o primeiro passo:

> *"(b) ITW dataset [19]. **Given real bokeh images, DeblurNet recovers an AIF image.**
> We then estimate depth and extract a foreground mask [86] to define the estimated
> focus plane D_focus. The bokeh level K is computed from the EXIF metadata and the
> estimated D_focus following the formulation in [19]."* — paper.txt:293-296

E no corpo: *"For each real image, we use DeblurNet to produce the AIF input and compute
an approximate bokeh level K following [19]"* — paper.txt:337-338.

A profundidade sai **da AIF**, não da bokeh: *"where D is the monocular depth map
estimated **from I_aif** using an off-the-shelf depth estimator [7]"* — paper.txt:314.
[7] é Depth Pro (paper.txt:762-764); [86] é BiRefNet (paper.txt:955-957).

Ordem canônica da rota B, então:

```
bokeh real  ──DeblurNet──▶  I_aif
                             ├── Depth Pro [7]  ─▶ D (profundidade métrica)
                             └── BiRefNet [86]  ─▶ M (in-focus mask)
                                                      │
                              Eq. 4 (paper.txt:352) ──┴─▶ D_focus = median(D[M])
                              Eq. 3 (paper.txt:340) ────▶ K  (com f, F da EXIF)
                              Eq. 2 (paper.txt:312) ────▶ D_def = K·|D − D_focus|
```

De qual imagem sai a **máscara** o paper não diz literalmente. O "We then" de
paper.txt:293-294 vem depois de "DeblurNet recovers an AIF image", o que sugere AIF —
mas é sugestão, não afirmação. **`[A]`**, ver §7.

### 1.4 Qual renderizador produz o alvo

**Nenhum.** Na rota B o alvo é a **própria fotografia real**.

- A tupla de supervisão é `(I_aif, I_out, D_def)` — paper.txt:321.
- Na Fig. 3(b) a entrada da rota é literalmente "Real bokeh image" (paper.txt:271), e a
  fileira de saída "Bokeh images" (paper.txt:283) é alimentada por ela.
- O renderizador [43] (BokehMe, paper.txt:851-852) aparece **só** em (a) — *"feed it
  into a bokeh renderer [43] to synthesize corresponding bokeh images"*, paper.txt:292 —
  e em (c), dentro do sweep da Eq. 5 (paper.txt:369-370, 391-396).

Consequência prática: na rota B **não há `k_effective_factor` a aplicar**, não há SSIM
de calibração, e o campo `renderer` da proveniência deve ser `None` — não um dicionário
com `is_final_label_renderer: True`, que é o que o pipeline antigo grava
(`route_b.py:121`, ver defeito B14).

### 1.5 De onde vêm focal, f-number e distância de foco — e o que fazer quando faltam

- **`f` e `F`: da EXIF, diretamente.** *"where f is the focal length and F is the
  aperture f-number, both directly obtained from EXIF metadata"* — paper.txt:342-343.
- **Distância de foco: NÃO da EXIF, por decisão explícita do paper.**

> *"Although some devices may provide a focus-distance field in EXIF, it is **frequently
> missing or noisy; therefore, we do not rely on EXIF for D_focus**. Instead, we compute
> the focus plane based on an in-focus mask and a monocular depth estimate, following
> the approach of [19]."* — paper.txt:344-347

  Ou seja: a EXIF fornece dois dos três termos da Eq. 3; o terceiro vem da Eq. 4.
  A distância de foco da EXIF, quando existe, serve como **validador** — a mediana de
  `k_value` calculada dela em 318 amostras dá 20,1 [M, `CONTRATO.md:77`], contra 16,6
  da tabela kfix — nunca como rótulo.

- **O que fazer quando falta: o paper CALA.** Não há uma frase no artigo sobre EXIF
  ausente. A regra 4 do `CONTRATO.md:52-53` (*rejeita, nunca substitui por constante*)
  é **decisão nossa**, declarada. `[A]`, item A8 em §7.

A largura física do sensor — necessária para `pixel_ratio` — **também não está na
EXIF padrão**. O caminho usado (crop factor via `focal_length_35`) é reconstrução
nossa; o paper não descreve como obteve a largura do sensor. `[A]`, item A7.

---

## 2. O estado real do dataset v0 da rota B

Para não confundir camadas: **o dado publicado v0 NÃO foi gerado pelo código que está
hoje em `bokehnet-preprocessing/src/pipelines/route_b.py`.** O HEAD daquele arquivo
morre na primeira amostra (defeito B9). O que está no `ACHADOS.md` é o resultado de uma
versão anterior:

| fato | valor | origem |
|---|---|---|
| rota B, amostras | 11.635 | `ACHADOS.md:13` [M] |
| coluna `k` | **1 valor distinto: 50,0** | `ACHADOS.md:14` [M] |
| `calibration_ssim` | 100% NULA | `ACHADOS.md:16` [M] |
| coluna `depth` | profundidade **métrica** min-max | `ACHADOS.md:25` [M] |
| `max(defocus_map)` | **65535 exato em todas** | `ACHADOS.md:26` [M] |

O `k = 50,0` constante e o `max(defocus) = 65535` em todas dizem a mesma coisa por dois
caminhos: **o mapa entregue não continha K**. Como o mapa mediano é a profundidade
normalizada (enunciado da tarefa, coerente com `ACHADOS.md:25-26`), a fase 2 aprendeu a
ignorar o canal de controle nas rotas reais — e é isso que a LVCorr de +0,4365 mede
(`ACHADOS.md:191`).

O `kfix` recalculou K pela Eq. 3 **só na rota B** (cobertura 100% da B, **0% da C**,
`ACHADOS.md:46`) e usou `max_coc_calibrado = 10,510746` — um percentil da própria rota B
(`ACHADOS.md:39,44`). LF-Bokeh subiu para +0,8288 e RealBokeh/RealDOF desabaram
(`ACHADOS.md:192`). A explicação estrutural está no código: **`--max-coc` é um argumento
de linha de comando, um por rota**, e nada obriga os três a coincidirem —
`route_a.py:207`, `route_b.py:350-355`, `route_c.py:466`, todos `required=True`,
independentes. Duas convenções de normalizador no mesmo lote não é acidente de
operação; é o que o código permite por construção.

---

## 3. Auditoria do pipeline original — divergência a divergência

Arquivos lidos: `bokehnet-preprocessing/src/pipelines/route_b.py` (460 linhas),
`src/preprocessing/control_contract.py` (202), `src/preprocessing/genrefocus.py`,
`src/model_runtime/deblurnet.py` (222), `src/model_runtime/runtime.py` (52),
`src/vendor/genfocus_flux.py`, `src/dataio/writers.py`, `scripts/run_route_b.py`.

### B1 — A DeblurNet roda com a convenção de adapter ERRADA (CRÍTICO, novo)

**Defeito.** `build_deblurnet_fn` chama `generate(...)` **sem passar `main_adapter`**:

```
bokehnet-preprocessing/src/model_runtime/deblurnet.py:201-211
    result = generate(pipe, height=..., width=..., prompt=...,
                      num_inference_steps=num_steps, conditions=[cond0],
                      NO_TILED_DENOISE=no_tiled_denoise, ...)      # sem main_adapter
```

O default é `None`:

```
bokehnet-preprocessing/src/vendor/genfocus_flux.py:485   main_adapter: Optional[List[str]] = None
bokehnet-preprocessing/src/vendor/genfocus_flux.py:783   adapters=[main_adapter] * 2 + (c_adapters if use_cond else [])
```

Logo o LoRA age **só nas condições** (cond-only). Mas a rota B **exige o nosso
checkpoint** e proíbe o oficial:

```
bokehnet-preprocessing/src/pipelines/route_b.py:331-335
    "--deblurnet-weights", required=True,
    help="... This route never defaults to the official checkpoint."
bokehnet-preprocessing/src/model_runtime/deblurnet.py:78-80
    """... publicado como ``juliadollis/genrefocus-deblurnet-paper-4gpu``."""
```

E esse checkpoint é a variante **main+cond**:

> `| main+cond (original) | genrefocus_deblurnet/ | main+cond | COMPLETO, step_60000 |
> SIM: juliadollis/genrefocus-deblurnet-paper-4gpu | main_adapter="deblurring" |`
> — `HANDOFF_PROJECT_HISTORY.md:109`

> *"um modelo treinado main+cond PRECISA de `main_adapter="deblurring"` na inferência;
> se rodar com `main_adapter=None` (o default oficial), sai **LAVADO**."*
> — `HANDOFF_PROJECT_HISTORY.md:102`; e o mesmo bug já custou LPIPS ~0,85,
> `HANDOFF_PROJECT_HISTORY.md:171`

**Evidência do lado certo**: o pipeline de avaliação passa o argumento explicitamente,
com default `"deblurring"` — `deblurnet-eval-pipeline/infer_and_eval.py:98-101, 137`.
A inferência oficial, por sua vez, também não passa
(`genrefocus_deblurnet_paper/third_party/Genfocus/Inference_deblurNet.py:103-111`) — e
está **certa**, porque o peso oficial é cond-only.

**Impacto.** É o defeito de maior alcance da rota B, porque contamina a **AIF**, e a
AIF é entrada de tudo: Depth Pro, BiRefNet, `D_focus`, K pela Eq. 3, e o próprio
condicionamento `I_aif` que a BokehNet vê no treino. Nenhum gate do pipeline antigo
pegava: `--min-aif-laplacian-variance` tem default **0,0** (`route_b.py:405-410`), que
o próprio help descreve como "0 logs only".

**Não medido**: se o run que gerou o v0 usou este caminho de código. O v0 é anterior
(§2). O que está medido é que o caminho **atual** produz AIF lavada com o peso que ele
próprio exige.

**Conserto.** O `main_adapter` tem que ser **função declarada da variante do
checkpoint**, nunca default. Ver §5.

### B2 — Fator de 1000× (CRÍTICO, confirmado)

**Defeito.** `compute_k_eq3` devolve `k` na convenção **milímetro**:

```
bokehnet-preprocessing/src/preprocessing/control_contract.py:117   focus_mm = float(focus_depth_m) * 1000.0
bokehnet-preprocessing/src/preprocessing/control_contract.py:125   k = (f**2 * focus_mm) / (2.0*aperture*(focus_mm - f)) * pixel_ratio
```

`f²·z/(2F(z−f))` em mm² vezes `pixel_ratio` em px/mm dá **px·mm** — um `k` que só fecha
contra `Δ(1/mm)`. Mas o consumidor usa disparidade em **1/m**:

```
bokehnet-preprocessing/src/preprocessing/control_contract.py:85-87
    disparity = 1.0 / depth_metric_m.astype(np.float32)      # 1/m
    focus_disparity = 1.0 / float(focus_depth_m)             # 1/m
    return float(k) * (disparity - focus_disparity)
```

`px·mm × 1/m = px/1000` na conta certa, ou seja **o valor calculado é 1000× maior que o
CoC físico**. É exatamente o que `CONTRATO.md:15,67` corrige com `K = k_eq3 / 1000`, e
o que `ACHADOS.md:229` ancora: `k_official(16553.9) = 16,5539`, batendo com o default
15,0 da inferência oficial (`Inference_bokehNet.py:53`, verificado byte a byte).

**Impacto.** Com `k_eq3` mediano de 16.553,9 (`ACHADOS.md:32`) e um `Δdisp` típico de
décimos de 1/m, o CoC calculado sai na casa dos milhares de pixels. Dividido por
qualquer `max_coc` razoável e `clip`ado em 1,0 (`control_contract.py:99`), o mapa vira
uma máscara binária "em foco / fora de foco" — **K desaparece do rótulo pela segunda
vez**, por um mecanismo diferente do D1.

**Conserto.** `src/control/contract.py:371-379` (`k_official`) já resolve. A rota B nova
tem que chamar `k_from_exif`, que já compõe `sensor_width_mm → pixel_ratio → k_eq3_mm →
k_official` (`contract.py:382-411`), e nunca `k_eq3_mm` direto.

### B3 — `pixel_ratio` com a largura, não com o maior lado (confirmado)

**Defeito.**
```
bokehnet-preprocessing/src/pipelines/route_b.py:187-193
    k = compute_k_eq3(row["focal_length"], row["f_number"], focus_depth_m,
                      bokeh_bgr.shape[1],            # <- LARGURA
                      row.get("focal_length_35"))
bokehnet-preprocessing/src/preprocessing/control_contract.py:106,124
    image_width_px: int
    pixel_ratio = float(image_width_px) / sensor_width_mm
```

**Contra o paper**: paper.txt:1186-1187 define `pixel_ratio` como *"the image's largest
edge length divided by the physical sensor width"*. **Contra a medição**:
`ACHADOS.md:62-63` reconstrói `fx_px = f_mm · pixel_ratio = max(W,H) · focal_length_35 / 36`
com identidade verificada em 900/900 linhas — *"Logo `max(H,W)` é o correto, não a largura"*.

**Impacto.** Em paisagem `max(H,W) == W` e não há erro. Em retrato, `K` sai subestimado
por `W/H` (0,75 num 3:4, 0,667 num 2:3). O erro é silencioso e correlacionado com a
orientação da foto, o que é pior que ruído: o modelo aprende que retratos precisam de
menos borrão.

**Conserto.** `src/control/contract.py:317-332` já implementa certo, recebendo
`image_hw` e usando `max(height, width)`. A rota B nova só precisa passar
`(H, W)` da imagem, não `shape[1]`.

**Nota de higiene documental** — divergência entre nossos próprios documentos:
`src/control/contract.py:322` diz *"32 de 100 amostras conferidas da rota B são
retrato"*; `ACHADOS.md:63` diz *"900/900 linhas, 32 delas em retrato"*, isto é 3,6%.
Um dos dois está errado. Não muda comportamento; muda o peso que a frase carrega numa
revisão. Corrigir o comentário para citar o número do `ACHADOS.md`.

### B4 — `max_coc` é argumento de CLI, um por rota (CRÍTICO)

**Defeito.**
```
bokehnet-preprocessing/src/pipelines/route_b.py:350-355   parser.add_argument("--max-coc", required=True, type=float, ...)
bokehnet-preprocessing/src/pipelines/route_a.py:207       parser.add_argument("--max-coc", required=True, type=float)
bokehnet-preprocessing/src/pipelines/route_c.py:466       "--max-coc", required=True, type=float,
```
e o valor viaja para dentro da amostra: `route_b.py:234, 257`.

**Contra o contrato**: `CONTRATO.md:42-45` — *"`max_coc` é global e congelado. Nunca por
imagem, nunca por rota, nunca por fonte."* **Contra a medição**:
`max_coc_calibrado = 10,510746` constante na tabela kfix (`ACHADOS.md:39`), que é o
percentil 82,66 de `coc_p99_px` **da própria rota B** (`ACHADOS.md:44`) — um
normalizador derivado de uma rota só.

**Impacto.** É o mecanismo estrutural do resultado do kfix: LF-Bokeh +0,8288 com
RealBokeh +0,4832 e RealDOF −0,4599 (`ACHADOS.md:192`). Três `--max-coc` independentes
tornam a mistura de convenções o caminho padrão, não a exceção.

**Conserto.** Já está fechado por construção no código novo: `MAX_COC = 100.0` em
`src/control/contract.py:47`, e `defocus_map` **não aceita** `max_coc` como parâmetro
(`contract.py:433-451`, com o porquê escrito em :440-448). A rota B nova não pode expor
flag nenhuma que encoste nisso.

### B5 — `D_focus` mediana da PROFUNDIDADE, não da disparidade

**Defeito.**
```
bokehnet-preprocessing/src/preprocessing/control_contract.py:66-74
    """Eq. (4): median metric depth inside the final in-focus mask."""
    focus_depth_m = float(np.median(pixels))
bokehnet-preprocessing/src/preprocessing/control_contract.py:86
    focus_disparity = 1.0 / float(focus_depth_m)
```

**Contra a autoridade.** O paper escreve `D_focus = median(D[M])` com `D` sendo o mapa
de profundidade (paper.txt:352, 314) — literalmente, o código antigo está certo. Mas a
inferência oficial resolve o silêncio na direção oposta:

```
Inference_bokehNet.py:94    disp = 1.0 / safe_depth
Inference_bokehNet.py:118   disp_focus = float(np.median(valid_disp))
```
(ambas verificadas neste checkout, em
`genrefocus_deblurnet_paper/third_party/Genfocus/Inference_bokehNet.py`).

`CONTRATO.md:28-40` já registra que operar em `1/z` é decisão por autoridade do código
oficial, e a regra 1 exige mediana **na disparidade**. `1/median(z) ≠ median(1/z)` em
contagem par (`np.median` faz a média dos dois centrais, e a média de recíprocos não é o
recíproco da média).

**Impacto.** Pequeno em máscara grande, sistemático, e é folga que o contrato existe
para não ter. Não é o gargalo.

**Conserto.** `src/control/contract.py:247-284` (`focus_disparity_from_mask`) já faz
`np.median(1.0 / values)`.

**Mas fica uma pergunta aberta que a rota B levanta e a C não** — ver §4.2 e o item
`[A]` A5: a Eq. 3 precisa de `D_focus` como **profundidade em mm**. Se `focus_disparity`
é `median(1/z)`, então `focus_depth_m = 1/focus_disparity` **não é** `median(z)`. Qual
das duas alimenta a Eq. 3 é escolha nossa, e precisa ser declarada.

### B6 — Sensor de 36 mm assumido quando falta `focal_length_35`

```
bokehnet-preprocessing/src/preprocessing/control_contract.py:120-123
    if focal_length_35_mm is not None and float(focal_length_35_mm) > 0:
        sensor_width_mm = 36.0 / (float(focal_length_35_mm) / f)
    else:
        sensor_width_mm = 36.0                       # <- fallback numérico
```

Viola `CONTRATO.md:52-53` (regra 4). **Na prática nunca dispara**: `focal_length_35`
está presente em 13.800/13.800 das amostras que passam o filtro da rota B
(`ACHADOS.md:155-157`) [M]. Mas o custo se disparasse é grande — `contract.py:297-298`
registra: assumir crop 1,0 onde o real é 5,6 subestima `pixel_ratio`, logo K, por 5,6×.

**Conserto.** `src/control/contract.py:291-314` (`sensor_width_mm`) rejeita com
`sensor_width_unresolvable`, sem fallback. Já resolvido; travado por teste
(`tests/test_contract.py:189`, `test_sensor_sem_focal_35_rejeita_em_vez_de_assumir_36mm`).

### B7 — Cascata de máscara com `except` nu, e proveniência que mente

```
bokehnet-preprocessing/src/preprocessing/genrefocus.py:420-428
    try:      mask = _segment_birefnet(img, model, device)
    except Exception:
        try:  mask = _segment_rmbg(img, device)
        except Exception:
              mask = _segment_grabcut(img)
```
e a proveniência continua afirmando "automatic":
```
bokehnet-preprocessing/src/preprocessing/control_contract.py:144   return automatic_mask.astype(np.float32), "automatic", None
bokehnet-preprocessing/src/pipelines/route_b.py:223                quality_dict["mask_source"] = mask_source
```

O paper especifica **BiRefNet [86]** nominalmente (paper.txt:348, 955-957). Um GrabCut
no lugar dele muda `D_focus`, que multiplica direto no K da Eq. 3 — e nada no dado
registra que aconteceu.

**Conserto.** `src/model_runtime/segmentation.py` tem um backend só, com `FileNotFoundError`
explícito (:86-91) e `provenance()` com sha256 do snapshot (:107-114); `MaskSource` é
enum fechado **sem** valor `AUTOMATIC` (`dataio/sample.py:52-62`). Já resolvido.

Observação secundária no mesmo arquivo: `genrefocus.py:442` calcula
`s1 = mean(foreground_pixels)` — média onde a Eq. 4 pede mediana (defeito D3). Na rota B
esse valor é **descartado** (`route_b.py:172`, `_, mask_auto = estimate_focus_plane(...)`),
então D3 não contamina o K da rota B; contaminou a geração anterior. Registro para não
se re-medir.

### B8 — Profundidade gravada como métrica min-max

```
bokehnet-preprocessing/src/pipelines/route_b.py:166-170
    depth = metric_depth.normalized                       # (z − z_min)/(z_max − z_min)
    cv2.imwrite(str(tmp_depth), (np.clip(depth,0,1)*65535).astype(np.uint16))
bokehnet-preprocessing/src/preprocessing/control_contract.py:32-37   @property normalized
```

Medido: 24,7% das amostras da rota B com a cena útil em **menos de 256 níveis** de
65535, mediana de **23 níveis** no subgrupo `z_max ≥ 1000 m` (`ACHADOS.md:56`) [M].
E `ACHADOS.md:25` confirma por teste de pixel que a coluna `depth` publicada é
profundidade métrica min-max.

**Conserto.** `src/dataio/encoding.py` grava **uint16 linear em disparidade**
(:104-147), com o span vindo da profundidade **cheia**, não da reduzida (:120-137) —
detalhe que já custou um K de renderer 3,4× menor numa auditoria (:126-129).

### B9 — `import json` ausente: NameError na primeira amostra (confirmado)

```
bokehnet-preprocessing/src/pipelines/route_b.py:1-12    (nenhum `import json`)
bokehnet-preprocessing/src/pipelines/route_b.py:136-137 json.loads / json.JSONDecodeError
bokehnet-preprocessing/src/pipelines/route_b.py:298     log_file.write(json.dumps(log_entry, ...) + "\n")
```

A linha 298 está no bloco `finally:` do `try` por amostra (:297-298), então roda
**sempre** — inclusive quando o corpo é bem-sucedido. `NameError` levantado dentro de um
`finally` não é capturado pelo `except Exception` que já terminou: o run morre na
amostra 0.

`route_c.py:1-13` tem `import json` (linha 3). É regressão específica da rota B — não um
padrão do repositório. Confirma `CLAUDE.md` do regen: *"`compileall` e parsing AST não
pegam `NameError`"*.

### B10 — `except Exception` largo, sem vocabulário fechado de motivo

```
bokehnet-preprocessing/src/pipelines/route_b.py:292-296
    except Exception as exc:
        stats["error"] += 1
        log_entry["error"] = str(exc)
```

Todas as falhas viram um balde `error` (`route_b.py:124-129`). Sem slug agregável não há
histograma de motivos, e sem histograma não há como calibrar limiar nem detectar
fallback novo (`CLAUDE.md` do regen; `qc/rejection.py:1-9` no código novo).

**Conserto.** `SampleRejected` + `REJECTION_REASONS` fechado
(`src/control/contract.py:71-157`) e `RejectionLog.summary()`
(`src/qc/rejection.py:93-111`). Já existe; falta a rota B usar.

### B11 — K gravado sem a resolução em que foi medido

Nenhum campo da amostra da rota B guarda a resolução processada
(`route_b.py:239-278`), e `ACHADOS.md:65-66` registra: o fator do crop de treino
`512/min(W,H)` vale 0,892 e 0,821 em duas amostras medidas, e **nenhum código o aplica**
[M]. Viola `CONTRATO.md:47-50` (regra 3).

**Conserto.** `EncodedDepth` carrega `image_hw` **e** `depth_hw`
(`dataio/encoding.py:56-83`), `Sample.metadata()` grava os quatro
(`dataio/sample.py:170` via `to_metadata`), e `k_at_resolution`
(`control/contract.py:458-482`) é a única conversão. Risco residual já registrado em
`REGISTRO.md:196` — nada **força** o chamador a usar; mitigação é o `data-contract`
conferir no release.

### B12 — Sem split por cena; identificadores frágeis

```
bokehnet-preprocessing/src/pipelines/route_b.py:144   stem = f"b_{index:06d}"
bokehnet-preprocessing/src/pipelines/route_b.py:262   "source_sample_id": str(row.get("flickr_photo_id", index))
```

`index` é a posição **no dataset já filtrado** (`route_b.py:82-90`): mudar o filtro
renumera tudo, e a retomada por `stem` (:131-139) passa a pular as amostras erradas.
Não há `scene_id` nem split materializado.

Na rota B cada linha é uma foto independente, então o vazamento treino/val é menos
agudo que na C — mas o `flickr_photo_id` é a única chave estável, e nada garante
unicidade. `source_duplicate_sample` já existe como slug
(`control/contract.py:118`).

**Conserto.** `scene_id = flickr_photo_id`, `sample_id` derivado dele (não de posição),
`build_scene_split` + `SceneSplit.save` materializado pelo `FileSampleWriter`
(`dataio/writer.py:74-78`).

### B13 — Sem gate de qualidade sobre a AIF (o B6 do REGISTRO)

`--min-aif-laplacian-variance` tem default **0,0** e o help diz "0 logs only"
(`route_b.py:405-410`). É o único gate que olharia a saída da DeblurNet, e ele está
desligado. Combinado com B1, é como o defeito de maior impacto passou.

`REGISTRO.md:187` lista exatamente isto como o único item aberto da rota B:
*"Rota B (1): B6 (gate de qualidade da AIF do DeblurNet)"*.

### B14 — Proveniência da rota B afirma coisas falsas

```
bokehnet-preprocessing/src/pipelines/route_b.py:121
    renderer={"name": "not_applicable_route_b", "is_final_label_renderer": True},
```

Dicionário literal, não `renderer.info` — nas rotas A e C esse campo vem de um objeto
real (`route_a.py:126`, `route_c.py:192`, `preprocessing/renderer.py:27,58,100`, onde o
default é `False`). Na rota B **não existe renderer** (§1.4), e afirmar
`is_final_label_renderer: True` é proveniência que mente na direção tranquilizadora.

E a proveniência **omite o BiRefNet**:
```
bokehnet-preprocessing/src/preprocessing/control_contract.py:196-200
    "model_hashes": { "deblurNet.safetensors": ..., "depth_pro.pt": ... }
```
O BiRefNet define a máscara, que define `D_focus`, que multiplica no K da Eq. 3. Sem o
hash dele não dá para reproduzir o rótulo. `CLAUDE.md` do regen exige explicitamente
*"hash de todo modelo que influenciou o rótulo — DeblurNet, DepthPro, **BiRefNet**"*.

### B15 — Nenhuma auditoria dos 30,33% com crop factor exatamente 1,0

`ACHADOS.md:160-173` [M]: crop factor mediano 1,50, e **4.185 amostras = 30,33%** com
crop factor **exatamente 1,0**. *"Trinta por cento de full-frame num dataset do Flickr é
alto"* — é ou full-frame de verdade, ou a assinatura de uma câmera que ecoa a focal no
campo de 35 mm quando não sabe. O `flickr_exif` traz `make` e `model`.

O pipeline antigo aceita todos sem marcar. Como `pixel_ratio` é linear em
`1/sensor_width`, errar aqui erra K na mesma proporção.

### B16 — `s1` legado: uma segunda convenção de plano de foco na mesma amostra

```
bokehnet-preprocessing/src/pipelines/route_b.py:236-238
    s1 = (focus_depth_m - metric_depth.min_m) / max(metric_depth.max_m - metric_depth.min_m, 1e-8)
bokehnet-preprocessing/src/pipelines/route_b.py:252     "s1": float(s1),
```

`ACHADOS.md:58` mede a divergência entre o `s1` implícito e o gravado: mediana 0,00308,
p90 **0,07927**, máximo 0,48821 — *"são planos de foco diferentes"* [M]. Dois campos
descrevendo a mesma grandeza, discordando.

`REGISTRO.md:199` já registra que `s1` **não existe** no contrato novo. A rota B nova não
pode reintroduzi-lo, nem por compatibilidade.

### B17 — A AIF pode não estar registrada com a bokeh

A DeblurNet é um modelo de difusão. O caminho com `long_side > 0` **recorta** até
múltiplo de 16 depois de redimensionar
(`bokehnet-preprocessing/src/model_runtime/deblurnet.py:55-60`, idêntico ao oficial
`Inference_deblurNet.py:30-41`) e depois volta ao tamanho original por `cv2.resize`
(`deblurnet.py:215-218`) — o que é um **zoom**, não uma identidade. A rota B chama sem
`long_side` (`route_b.py:104-109`, default 0), caindo no caminho sem crop
(`deblurnet.py:62-66`), então hoje o risco não se materializa — mas basta alguém passar
`--long-side` para a AIF virar um recorte reescalado da bokeh, com a profundidade e a
máscara medidas numa geometria e o alvo em outra.

O pipeline antigo tem `--max-pair-shift-px 6.0` por correlação de fase
(`route_b.py:411-422`), o que é a mitigação certa. **O código novo não tem equivalente**:
`qc/gates.py` só oferece `pair_shape_matches` (:260-264), que compara shapes.

**Não medido**: quanto a DeblurNet desloca de fato. É item `[A]` A13.

---

## 4. O que já existe no código novo, e o que falta

### 4.1 Inventário

| módulo | serve à rota B como está? | observação |
|---|---|---|
| `src/control/contract.py` | **sim, inteiro** | `k_from_exif` é literalmente o caminho da Eq. 3 (:382-411) |
| `src/dataio/encoding.py` | **sim** | disparidade uint16, `image_hw` obrigatório |
| `src/dataio/sample.py` | **quase** | tem `KSource.EQ3_EXIF` (:66) e `SampleProvenance.deblurnet` (:93), mas nada os exige |
| `src/dataio/writer.py` | **sim** | `generated_images` + `channel_order` já previstos (:114-128) — a AIF da rota B é o caso citado no comentário |
| `src/dataio/split.py` | **sim** | `build_scene_split`; `split_from_source` não se aplica (BokehDiffusion só tem `train`) |
| `src/model_runtime/depth.py` | **sim** | Depth Pro único, hash, resample para `image_hw` |
| `src/model_runtime/segmentation.py` | **sim** | BiRefNet único, hash de snapshot |
| `src/qc/gates.py` | **parcial** | 6 dos 10 gates se aplicam; faltam 3 específicos da B |
| `src/qc/rejection.py` | **sim** | histograma fechado |
| `src/renderer/` | **não se aplica** | a rota B não tem renderer (§1.4) |
| `src/routes/route_c.py` | **modelo de estrutura** | copiar a forma, não o conteúdo |
| `src/sources/` | **não serve** | é RealBokeh-específico; a B precisa do seu adaptador |
| `src/model_runtime/deblurnet.py` | **NÃO EXISTE** | é a maior lacuna |
| `scripts/run_route_b.py` | **NÃO EXISTE** | |

### 4.2 Checagem dimensional da Eq. 3 no nosso código

Percorrendo `src/control/contract.py` função a função, com a unidade de cada saída:

| # | função | linha | entrada | saída | unidade |
|---|---|---|---|---|---|
| 1 | `sensor_width_mm(f_mm, f35_mm)` | :291-314 | mm, mm | `36.0 / (f35/f)` | **mm** |
| 2 | `pixel_ratio(image_hw, sensor_mm)` | :317-332 | px, mm | `max(H,W)/sensor` | **px/mm** |
| 3 | `k_eq3_mm(f, F, z_focus_m, px_per_mm)` | :335-368 | mm, —, m, px/mm | `f²·z_mm/(2F(z_mm−f)) · px_mm` | **px·mm** |
| 4 | `k_official(k_eq3)` | :371-379 | px·mm | `k_eq3 / 1000` | **px·m** |
| 5 | `signed_coc_px(depth_m, focus_disp, k)` | :418-430 | m, 1/m, px·m | `k·(1/z − disp_f)` | **px** ✓ |
| 6 | `defocus_map` | :433-451 | — | `|coc|/MAX_COC` | adimensional em [0,1] ✓ |
| 7 | `k_at_resolution(k, src_hw, dst_short)` | :458-482 | px·m, px, px | `k · dst/min(H,W)` | **px·m**, nova escala ✓ |

**Fecha.** Três verificações independentes:

1. **Álgebra**: `px·mm × (1/m)` = `px·mm / (1000 mm)` = `px/1000`; dividir `k_eq3` por
   1000 antes cancela exatamente. `MM_PER_M = 1000.0` (:52) é o único lugar onde o
   número aparece.
2. **`FULL_FRAME_WIDTH_MM = 36.0`** (:56) é usado só em `sensor_width_mm` (:314) e para
   reportar `crop_factor` no diagnóstico (:403) — nunca como default. Confere com a
   regra 4.
3. **Âncora numérica**: `ACHADOS.md:229` [M] registra `k_official(16553.9) = 16,5539`,
   contra a mediana da EXIF 20,1 (`CONTRATO.md:77`) e o default oficial 15,0
   (`Inference_bokehNet.py:53`, **verificado neste checkout**). Três caminhos
   independentes na mesma faixa.

**Uma ambiguidade real, que a rota B expõe e a C não.** `k_eq3_mm` recebe
`focus_depth_m` — uma **profundidade**. Mas o contrato produz `focus_disparity =
median(1/z[M])` (:276), e `1/focus_disparity ≠ median(z[M])`. A Eq. 3 do paper usa o
mesmo símbolo `D_focus` da Eq. 4 (paper.txt:340 e :352), que é definida sobre
profundidade. Então há duas leituras defensáveis:

- **(i)** alimentar a Eq. 3 com `1/focus_disparity` — coerente internamente, o mesmo
  plano que gera o mapa;
- **(ii)** alimentar com `median(z[M])` — literal ao paper, mas cria um segundo plano de
  foco na mesma amostra, que é o defeito B16 de volta.

**Recomendação: (i), declarada como desvio**, com `median(z[M])` gravado ao lado só
para auditoria. Motivo: a regra 5 do contrato (uma implementação só) pesa mais que a
literalidade, e a diferença é da ordem da medida em B5. É `[A]` A5.

**Um segundo ponto de atenção**, já sinalizado em `CONTRATO.md:136-138`:
`k_at_resolution` recebe **lado menor** e `encode_depth` recebe **lado longo**.
Convenções opostas na mesma cadeia; a rota B tem que nomear o argumento em toda chamada.

### 4.3 Gates: o que se aproveita e o que falta

De `src/qc/gates.py`, aplicando à rota B (AIF gerada, bokeh real, sem Eq. 5):

| gate | linha | rota B? |
|---|---|---|
| `pair_shape_matches` | :260-264 | **sim** — AIF gerada vs bokeh fonte |
| `bokeh_is_blurrier_than_aif` | :267-285 | **sim, e vira o gate B6/B13** — mede se a DeblurNet fez alguma coisa |
| `aif_sharpness` | :202-210 | **sim** |
| `mask_area_ratio` / `mask_border_coverage` | :99-115 | **sim** |
| `focus_mask_is_sharpest` | :137-170 | **sim, e é o mais importante** — a rota B **tem** a bokeh real, que é exatamente a imagem que esse gate exige |
| `mask_iou` | :118-130 | **sim** — AIF vs bokeh, com o cuidado de :123-126 |
| `depth_useful_levels` | :217-228 | **sim** |
| `focus_depth_plausible` | :231-253 | **sim** |
| `aif_aperture_is_narrow` | :288-310 | **não** — a AIF da B é gerada, não tem f-stop |
| `calibration_ssim_is_reliable` | :313-334 | **não** — não há Eq. 5 na rota B |

**Faltam três, específicos da rota B:**

1. **`deblur_actually_deblurred`** — a DeblurNet produziu algo diferente e mais nítido
   que a entrada? `bokeh_is_blurrier_than_aif` cobre a direção; falta pegar o caso
   *"saída ≈ entrada"* (LoRA não carregou) e o caso *"saída LAVADA"* (defeito B1), que
   pode ter **variância de Laplaciano alta** por artefato e ainda assim ser lixo. Uma
   razão de nitidez sozinha não separa os dois; precisa de um segundo eixo
   (ex.: PSNR/SSIM contra a entrada, com faixa de aceitação nos dois extremos).
2. **`exif_crop_factor_suspect`** — marcar (não necessariamente rejeitar) crop factor
   exatamente 1,0, cruzando `make`/`model` (defeito B15, `ACHADOS.md:160-173`).
3. **`pair_registration_shift`** — deslocamento AIF↔bokeh por correlação de fase, com
   confiança mínima (defeito B17). O pipeline antigo tinha
   (`route_b.py:411-422`); o novo não.

Cada um exige um slug novo em `GATE_REJECTION_REASONS`
(`src/control/contract.py:94-106`), que é `frozenset` fechado — a mudança é em
`control/`, que **não** editei.

### 4.4 O que falta escrever, em ordem de dependência

1. **`src/sources/bokehdiffusion.py`** — enumera as linhas de `atfortes/BokehDiffusion`,
   aplica o filtro (`not pseudo_aif`, `f > 0`, `F > 0`), devolve uma dataclass no
   feitio de `RealBokehPair`, com `scene_id`, `source_sample_id` (o
   `flickr_photo_id`), `focal_length_mm`, `f_number`, `focal_length_35mm`, `make`,
   `model`, e rejeição com slug. Padrão de leitura barata em `ACHADOS.md:206-216`.
2. **`src/sources/bokehdiffusion_images.py`** (ou estender `mirror_images.py` para
   receber o nome da coluna) — índice `sample_id → (shard, linha)` e leitura sequencial.
   O `MirrorIndex`/`MirrorImageLoader` (`src/sources/mirror_images.py:69-273`) já é a
   forma certa; hoje está amarrado a `file_name_base`, `image_focus` e `image_blur`.
3. **`src/model_runtime/deblurnet.py`** — ver §5. É o bloqueador.
4. **`src/routes/route_b.py`** — no molde do `route_c.py`, sem `calibrate_k` e sem
   `render_fn`; com `k_from_exif` no lugar do sweep.
5. **`scripts/run_route_b.py`** — no molde do `run_route_c.py`, com `--pilot` e todos os
   limiares em `None`.
6. **Extensões em `control/` e `dataio/`** (descritas, não aplicadas — §6).

---

## 5. As duas versões da DeblurNet

Está decidido gerar **duas versões da rota B**: uma com a nossa DeblurNet e outra com a
oficial. Isto é uma escolha de convenção de inferência, não só de arquivo de peso, e
misturar as duas num release é a mesma classe de erro do `max_coc` por rota.

### 5.1 O que cada uma é, com evidência

| | **nossa** | **oficial** |
|---|---|---|
| repositório | `juliadollis/genrefocus-deblurnet-paper-4gpu` | `nycu-cplab/Genfocus-Model` |
| arquivo | `deblur.safetensors` | `deblurNet.safetensors` |
| LoRA | **main + cond** | **cond-only** |
| `main_adapter` | **`"deblurring"`** | **`None`** |
| evidência | `HANDOFF_PROJECT_HISTORY.md:92,109,161`; `deblurnet-eval-pipeline/infer_and_eval.py:98-101` | `Inference_deblurNet.py:11,88-89,103-111`; `bokehnet-preprocessing/src/setup/download_models.py:11` |
| treino | step 60000, completo | pesos publicados pelos autores |

Comum às duas, e verificado no `Inference_deblurNet.py` oficial:
`adapter_name="deblurring"` (:88), `pipe.set_adapters(["deblurring"])` (:89),
prompt `"a sharp photo with everything in focus"` (:107), `num_inference_steps=28`
(:56 default; bate com `paper.txt:519-520`), `long_side=0` (:57),
`NO_TILED_DENOISE = min(w,h) < 512` (:95).

Existe ainda uma terceira variante no repositório — a cond-only **nossa**
(`genrefocus_deblurnet_paper/`), que é a fiel ao paper mas **parou em ~step 13500** por
um blocker de `peft` (`HANDOFF_PROJECT_HISTORY.md:110,113`). Ela **não** entra: um
checkpoint a 22% do treino não é uma versão, é um artefato incompleto.

### 5.2 O que isso exige do código

**Uma enum fechada, não uma string.** No feitio de `MaskSource`
(`dataio/sample.py:52-62`):

```
DeblurVariant.OURS_MAIN_COND     -> (repo, "deblur.safetensors",    main_adapter="deblurring")
DeblurVariant.OFFICIAL_COND_ONLY -> (repo, "deblurNet.safetensors", main_adapter=None)
```

A tripla é **indivisível**. O que não pode existir: um `--main-adapter` livre de CLI, e
um default para `main_adapter` no nosso wrapper. Passar sempre, explicitamente — porque
`generate` tem default `None` (`genfocus_flux.py:485`) e foi esse default que produziu
o defeito B1.

**Um `DeblurNetRuntime` no molde do `DepthProRuntime`** (`model_runtime/depth.py:68-143`):
`__post_init__` calcula o sha256 do `.safetensors` **antes** de carregar; `load()` é
separado; `provenance()` devolve
`{deblur_variant, deblur_lora_sha256, main_adapter, adapter_name, prompt, num_steps, long_side, backbone_sha_ou_id, genfocus_pipeline_sha256}`;
`infer(bokeh_rgb) -> aif_rgb`.

**`third_party/Genfocus` clonado, como o BokehMe.** Hoje `third_party/README.md` só
documenta o BokehMe (:3-8). A rota B precisa de `Genfocus/pipeline/flux.py`
(`Condition`, `generate`, `seed_everything`). O código vendorizado equivalente existe em
`bokehnet-preprocessing/src/vendor/genfocus_flux.py`, mas aquele repositório é referência
histórica e não deve ser importado por `bokehnet-regen`. Clonar, congelar o commit,
hashear `pipeline/flux.py` — mesma disciplina do BokehMe (`third_party/README.md:46-48`).

**Proveniência obrigatória.** `SampleProvenance.deblurnet` existe
(`dataio/sample.py:93`) mas é `Optional` e não entra em `_REQUIRED_PROVENANCE`
(`sample.py:204-205`). Para a rota B, `deblurnet` sem `deblur_variant` e
`deblur_lora_sha256` tem que ser erro de gravação — a assimetria certa é a mesma de
`calibration_ssim` para a Eq. 5 (`sample.py:253-254`).

### 5.3 Como as duas coexistem sem se misturar

Cinco regras, e nenhuma delas é convenção verbal:

1. **Um `--release-dir` por variante.** Nunca a mesma pasta com sufixo no `sample_id`.
   O `FileSampleWriter` grava `split.json` e `manifest.jsonl` no diretório
   (`dataio/writer.py:71-81`) — duas variantes ali dentro compartilhariam o manifesto.
2. **Um repo HF por variante**, e `publish_release.py` já se recusa a publicar por cima
   de repo com arquivos sem `--allow-existing` (:297-316).
3. **Checagem de uniformidade no release.** `publish_release.py:112-119` já reprova
   `control_version` divergente dentro do mesmo release, com a frase certa
   (*"Foi assim que…"*). Precisa da **mesma checagem** para
   `provenance.deblurnet.deblur_variant` **e** `deblur_lora_sha256`. Sem ela, um `rsync`
   de duas metades produz um release misto e nada denuncia.
4. **A variante entra na linha do manifesto.** `writer.py:137-151` carrega hoje
   `split`, `mask_source`, `depth_backend`, `control_version`, `max_coc` — precisamente
   os campos que denunciariam fallback. `deblur_variant` pertence a essa lista.
5. **O `sample_id` é o mesmo nas duas versões.** É o que torna as duas comparáveis
   par a par: mesma foto, mesma EXIF, mesmo `scene_id`, mesmo K analítico *se* a máscara
   coincidir — e a diferença de K entre as versões vira uma medida direta de quanto a
   AIF influencia o rótulo. Isso é um experimento barato que só existe se os ids
   baterem.

**O que NÃO diverge entre as duas versões**: `MAX_COC`, `CONTROL_VERSION`,
`DEPTH_LONG_SIDE`, a codificação da profundidade, o split, e os limiares dos gates.
Se algum desses divergir, as duas versões deixam de ser comparáveis e viram dois
datasets diferentes.

---

## 6. Lista priorizada do que mudar

Ordem por impacto no rótulo. Cada item: **defeito → evidência → conserto**.

### P0 — sem isto, o rótulo nasce errado

**1. `main_adapter` amarrado à variante do checkpoint.**
*Defeito*: `generate` chamado sem `main_adapter` com peso main+cond → AIF lavada
(`bokehnet-preprocessing/src/model_runtime/deblurnet.py:201-211`;
`src/vendor/genfocus_flux.py:485`; `HANDOFF_PROJECT_HISTORY.md:102,171`).
*Conserto*: `DeblurVariant` fechada, tripla indivisível, `main_adapter` sempre
explícito, gravado na proveniência. §5.2.

**2. A rota B usa `k_from_exif`, nunca `k_eq3_mm` direto.**
*Defeito*: fator 1000× (`control_contract.py:117,125` contra :85-87).
*Conserto*: `src/control/contract.py:382-411` já compõe o caminho inteiro e devolve o
diagnóstico. Chamar só ele.

**3. `pixel_ratio` recebe `image_hw`, não a largura.**
*Defeito*: `route_b.py:191` passa `bokeh_bgr.shape[1]`.
*Evidência contra*: `paper.txt:1186-1187`; `ACHADOS.md:62-63`.
*Conserto*: `contract.py:317-332`, já correto. Passar `(H, W)` da imagem **fonte**,
que é a resolução em que o K vive.

**4. `MAX_COC` nunca vira flag.**
*Defeito*: três `--max-coc` independentes (`route_a.py:207`, `route_b.py:350-355`,
`route_c.py:466`); é o mecanismo do kfix (`ACHADOS.md:39,44,192`).
*Conserto*: já fechado por construção (`contract.py:47,433-451`;
`dataio/sample.py:229-233`). A rota B não expõe botão nenhum aqui.

**5. `import json` — e o smoke de 2 amostras antes de qualquer run.**
*Defeito*: `route_b.py` sem `import json`, usado em :136,137,298, dentro de um `finally`.
*Conserto*: além do import, a disciplina do `CLAUDE.md`: smoke de 2 amostras primeiro,
porque `compileall` e AST não pegam `NameError`.

### P1 — sem isto, o rótulo é inauditável

**6. Slug fechado e histograma de rejeição.**
*Defeito*: `except Exception` largo virando um balde `error` (`route_b.py:292-296`).
*Conserto*: `SampleRejected` + `RejectionLog` (`qc/rejection.py`), no molde de
`route_c.run_route_c` (`routes/route_c.py:334-365`), com `print(log.summary())` no
`finally`.

**7. Proveniência completa e honesta.**
*Defeito*: sem hash do BiRefNet (`control_contract.py:196-200`);
`renderer={"is_final_label_renderer": True}` literal (`route_b.py:121`).
*Conserto*: `SampleProvenance` com `mask_model_sha256` obrigatório (já é,
`sample.py:204-205`), `renderer=None` na rota B, e `deblurnet` **obrigatório** —
mudança a fazer em `_REQUIRED_PROVENANCE`.

**8. Os diagnósticos da Eq. 3 vão para o disco.**
*Defeito*: `k_from_exif` devolve `(k, diagnostics)` com `sensor_width_mm`,
`crop_factor`, `pixel_ratio_px_per_mm`, `longest_edge_px`, `focal_length_mm`,
`f_number`, `focal_length_35mm`, `focus_depth_m` (`contract.py:399-410`) — e **nenhuma
estrutura do `dataio` tem onde guardá-los**. `ControlLabel` (`sample.py:104-125`) não
tem campo.
*Conserto*: um campo `k_diagnostics: Optional[dict]` em `ControlLabel`, exigido quando
`k_source == KSource.EQ3_EXIF` — simétrico ao `calibration_ssim` exigido para
`EQ5_SSIM_SWEEP` (`sample.py:253-254`). Sem isso, K vira um número sem como auditar de
qual sensor veio.

**9. `scene_id`, `sample_id` estável e split materializado.**
*Defeito*: `stem = f"b_{index:06d}"` sobre o dataset filtrado (`route_b.py:144`).
*Conserto*: `scene_id = flickr_photo_id`, `sample_id` derivado dele,
`build_scene_split` + `FileSampleWriter` (`dataio/writer.py:74-78`).

**10. `deblur_variant` no manifesto e checagem de uniformidade no release.**
*Defeito*: `publish_release.py:112-119` confere `control_version` uniforme, mas nada
impede um release meio nosso, meio oficial.
*Conserto*: §5.3, regras 3 e 4.

**11. Ledger de bytes da rota B.**
*Defeito*: `publish_release.py:125-137` exige `source_images.jsonl` com sha256 da AIF e
da bokeh — escrito para a rota C, onde **as duas** são referência. Na rota B a AIF é
**gerada** (`dataio/sample.py:7-11`).
*Conserto*: ledger com sha256 dos bytes da **bokeh fonte** (referência) e da **AIF
gravada** (produto), e a validação ramificando por `route`.

### P2 — gates e limiares (medir no piloto, congelar depois)

**12. Gate de qualidade da AIF** — o B6 do `REGISTRO.md:187`. `bokeh_is_blurrier_than_aif`
(`gates.py:267-285`) mais um segundo eixo que separe "não deblurou" de "lavou".
Default `None`, como todo gate do módulo.

**13. Gate de registro do par AIF↔bokeh** — correlação de fase com confiança mínima; o
antigo tinha (`route_b.py:411-422`), o novo não. Defeito B17.

**14. Marcação de crop factor suspeito** — os 30,33% exatamente 1,0
(`ACHADOS.md:160-173`), cruzando `make`/`model`. Marcar, não rejeitar: a tabela serve
para **auditar**, não para preencher lacuna — não há lacuna
(`ACHADOS.md:169-173`).

**15. Faixa de K plausível para a rota B** — o antigo tinha `--min-physical-k 1e-6` e
`--max-physical-k None` (`route_b.py:356-363`), isto é, nenhum limite útil.
Âncoras: kfix mediana 16,55 (`ACHADOS.md:229`), EXIF 20,1 e default oficial 15,0
(`CONTRATO.md:76-78`), Fig. 12 `K ∈ {0,5,10,15}` (`paper.txt:1151-1156`). Medir a
distribuição do piloto antes de congelar — e lembrar do `--k-max 300` da rota C, que
virou 47,0% de amostras censuradas no teto exato (`ACHADOS.md:19`).

### P3 — higiene

**16.** Corrigir o comentário `src/control/contract.py:322` ("32 de 100") para o número
do `ACHADOS.md:63` ("32 de 900").
**17.** Nunca reintroduzir `s1` (defeito B16; `REGISTRO.md:199`).
**18.** Rodar `estimate_disk_budget(n, DEPTH_LONG_SIDE, generates_image=True, ...)`
(`dataio/writer.py:201-219`) **antes** do run — a rota B grava a AIF em JPEG, ao
contrário da C, e são duas versões.

---

## 7. Itens `[A]` — o que o paper não publica e teremos que assumir

Numerados para poder ser referenciados. Nenhum foi promovido a `[M]`.

| # | item | por que é `[A]` | risco |
|---|---|---|---|
| **A1** | ITW [19] = `atfortes/BokehDiffusion` no HF | o paper cita só a publicação (paper.txt:791-793), sem URL. Sustentado por autor + volume (13.800 [M] contra "13K", paper.txt:1000) | baixo, mas é `[I]` e não `[M]` |
| **A2** | a máscara sai da **AIF**, não da bokeh | paper.txt:293-294 diz "We then estimate depth and extract a foreground mask" depois de "DeblurNet recovers an AIF image" — sugere, não afirma | **médio**: numa bokeh, a região nítida *é* a região em foco; a máscara da bokeh pode ser melhor que a da AIF |
| **A3** | filtrar `pseudo_aif` | coluna do dataset, não do paper; o filtro é do pipeline antigo (`route_b.py:83-89`) | baixo |
| **A4** | qual subconjunto são os "13K previously filtered and verified" | o critério de filtragem não é publicado (paper.txt:1000) | baixo |
| **A5** | qual `D_focus` alimenta a Eq. 3: `1/median(1/z)` ou `median(z)` | o paper usa o mesmo símbolo nas Eq. 3 e 4 (paper.txt:340,352) e opera em profundidade; nós operamos em disparidade por autoridade do código oficial (`CONTRATO.md:28-34`) | **médio** — ver §4.2 |
| **A6** | as imagens do BokehDiffusion estão na resolução de captura, sem recorte | não medido. Se houve recorte, `pixel_ratio` e `focal_length_35` descrevem geometrias diferentes | **alto** se falso: erra K por fator desconhecido |
| **A7** | largura do sensor via crop factor de `focal_length_35` | o paper não diz como obteve a largura física; e 30,33% têm crop factor exatamente 1,0 (`ACHADOS.md:165`) | **alto**: crop 1,0 assumido onde o real é 5,6 erra K por 5,6× (`ACHADOS.md:175`) |
| **A8** | rejeitar quando a EXIF falta, em vez de completar | o paper cala; a regra 4 é nossa (`CONTRATO.md:52-53`) | baixo (é o lado conservador) |
| **A9** | resolução em que a DeblurNet roda na rota B (`long_side`) | o paper cala; o oficial tem default 0 (`Inference_deblurNet.py:57`); o antigo também (`route_b.py:104-109`) | médio: `long_side > 0` **recorta** (`deblurnet.py:55-60`) e quebra o registro |
| **A10** | faixa de K plausível para a rota B | o paper não publica `K_min`/`K_max` (e os da Eq. 5 são da rota C) | médio — item 15 do §6 |
| **A11** | os limiares dos gates da rota B | nenhum vem do paper; todos default `None` (`gates.py:1-6`) | por definição, calibrar no piloto |
| **A12** | aplicar ou não `k_effective_factor = 0,9873` à rota B | medido no cluster (`ACHADOS.md:279-283`), mas na rota B **não há renderer** (§1.4), logo o fator não descreve nada da rota | baixo; a recomendação é **não aplicar**, e declarar |
| **A13** | quanto a DeblurNet desloca/deforma a AIF em relação à bokeh | **não medido** | médio: se houver deslocamento, `D`, `M` e o alvo vivem em geometrias diferentes |
| **A14** | qual DeblurNet os autores usaram para gerar os dados da rota B | o paper não diz se foi o peso publicado ou um intermediário do treino de 60K (paper.txt:515) | baixo (não reprodutível de qualquer forma) |
| **A15** | a rota B usa a bokeh na resolução nativa | o paper diz que o backbone suporta resolução arbitrária e que o tiling é de **inferência** (paper.txt:478-482, 503-504), e cala sobre a resolução de treino (`CONTRATO.md:122-126`) | médio; herda a decisão já declarada de crop 512 |

Fechados nesta auditoria (eram implícitos e agora têm evidência):

- ~~"o `pixel_ratio` usa a largura ou o maior lado?"~~ → **maior lado**, paper.txt:1186-1187.
- ~~"a Eq. 3 vale em mais de uma rota?"~~ → **só na B**, paper.txt:336-352 contra :359-360.
- ~~"que renderizador produz o alvo da rota B?"~~ → **nenhum**, paper.txt:271,283,321,292.
- ~~"a distância de foco vem da EXIF?"~~ → **não, por decisão explícita**, paper.txt:344-347.

---

## 8. O que este relatório NÃO conseguiu medir

Registrado para ninguém confundir com verificação:

- **Suíte de testes não rodou localmente**: `ModuleNotFoundError: No module named 'numpy'`
  nesta máquina. O último resultado conhecido é 36/36 (`ACHADOS.md:225`), da etapa 1 —
  antes de `dataio`, `qc`, `renderer` e `sources` existirem. **Não medido** se os
  ~10 arquivos de teste atuais passam.
- **Não medido** se o run que gerou o v0 usou o `route_b.py` de HEAD (§2). O HEAD não
  roda (B9), e o `git log` do arquivo tem um commit só (`94d9e15`, um flatten de
  pacote), então o histórico anterior não está neste checkout.
- **Não medido**: A6 (resolução de captura), A13 (deslocamento da DeblurNet), e a
  distribuição de `make`/`model` nos 30,33% de crop factor 1,0 (A7).
