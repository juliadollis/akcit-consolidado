# Auditoria externa — comece por aqui

**Estado congelado em 2026-09-10, 19h35 (UTC−3).** Este é o ponto de entrada para quem
chega sem contexto: um avaliador externo que vai ler o paper de referência, ler o código,
rodar os testes e decidir se o que afirmamos se sustenta.

Três arquivos novos acompanham este:

| arquivo | responde a |
|---|---|
| `reference/TABELA_DE_EVIDENCIAS.md` | *"de onde vem esse número?"* — 264 afirmações com origem, etiqueta e data. Inclui as **18 que não têm procedência** |
| `reference/DESVIOS_DO_PAPER.md` | *"onde vocês fizeram diferente do paper, e por quê?"* — 12 desvios declarados, 6 divergências acidentais, 24 silêncios do paper |
| `reference/REPRODUZIR.md` | *"como eu refaço isso?"* — do clone ao release, com o que **não** funciona e por quê |

---

## 1. O que este pipeline é

**O que ele é.** `bokehnet-regen` é a regeração dos dados de treino de uma reprodução da
**BokehNet** do paper GenRefocus (arXiv:2512.16923v3) — um modelo de difusão que
re-desfoca uma fotografia com controle explícito sobre o nível de bokeh `K` e sobre o plano
de foco. O repositório **não treina** o modelo. Ele produz o **rótulo de controle**: para
cada par (imagem nítida, imagem com bokeh), um mapa de defocus `D_def = K·|D − D_focus|`
que diz, pixel a pixel, quanto borrão aquela amostra representa. O paper obtém esse rótulo
por três caminhos, e o repositório reproduz os três: (a) sintético, sorteando `K` e
renderizando o alvo; (b) fotos reais com EXIF, calculando `K` pela óptica; (c) pares reais
sem EXIF, calibrando `K` por SSIM contra a fotografia real.

**Por que ele existe.** Um pré-processamento anterior construiu o mapa de defocus em
**profundidade linear normalizada por imagem**, enquanto a inferência oficial o constrói em
**disparidade métrica absoluta**. As duas fórmulas se parecem e não são a mesma. Toda a
fase 2 de treino foi feita assim, e a controlabilidade medida caiu de +0,91 (fase 1,
sintética) para +0,44 (fase 2, nas rotas reais), degradando **monotonicamente com o tempo
de treino** — quanto mais treinava, pior controlava. Consertar só o `K` de uma das rotas já
recuperou boa parte da distância até os pesos oficiais, o que aponta o rótulo como gargalo,
não a arquitetura. **Ressalva de procedência, e ela é grave:** esses números de
controlabilidade foram medidos antes deste repositório existir, e o harness que os produziu
**não está aqui**. Ver §6, pergunta 9.

**Como ele é construído.** O princípio organizador é que **um defeito de rótulo é
silencioso**: um `K` constante, um normalizador por rota, uma profundidade invertida por um
`except` — nada disso levanta erro, e o dataset sai com 100% de sucesso aparente. Então o
repositório troca sucesso silencioso por rejeição registrada. Não existe fallback numérico:
faltou EXIF, faltou sensor, falhou o modelo, a amostra é rejeitada com um slug de um
vocabulário fechado, e **todo run imprime o histograma de motivos**. Toda quantidade em
pixel carrega a resolução em que foi medida. `max_coc` é global, congelado, e não é
parâmetro de função nenhuma. A fórmula do sinal de controle vive num módulo só, importado
por geração, dataloader, avaliação e inferência — porque cópias divergem, e foi assim que o
projeto chegou a quatro interpretações de `K`.

---

## 2. O mapa — o que ler, em que ordem

### Se você tem 15 minutos

1. **`reference/CONTRATO.md`** (170 linhas) — a definição canônica do sinal de controle,
   com a análise dimensional e as âncoras numéricas. É a única página que **precisa** estar
   certa.
2. **§0 de `reference/REPRODUZIR.md`** — rode a suíte. Seis segundos.
3. **§6 deste arquivo** — as perguntas céticas, com resposta e ponteiro.

### Se você vai avaliar de verdade

| ordem | arquivo | o que é | linhas |
|---|---|---|---|
| 1 | `reference/paper.txt` | o paper extraído, grepável, **imutável**. 1.203 linhas; o supplement começa em 966 | 1.203 |
| 2 | `reference/CONTRATO.md` | a definição do sinal de controle | 170 |
| 3 | `reference/DESVIOS_DO_PAPER.md` | onde divergimos, com a linha do paper conferida | ~700 |
| 4 | `reference/ACHADOS.md` | tudo que foi **medido**, com `[M]`/`[I]`/`[A]` | 583 |
| 5 | `reference/TABELA_DE_EVIDENCIAS.md` | a procedência de cada número, inclusive a ausência dela | ~850 |
| 6 | `REGISTRO.md` | o diário de decisões, 9 etapas, com o porquê de cada uma | 1.155 |
| 7 | `reference/MEDICAO_PLANO_FOCO.md` | a medição que mudou o método da rota C | 150 |
| 8 | `reference/ROTA_{A,B,C}_AUDITORIA.md` | o confronto código × paper, item por item, por rota | 651 / 986 / 1.043 |
| 9 | `PLANO_EXECUCAO.md` | a ordem de execução e a hipótese que ela aposta | 118 |
| 10 | `reference/REPRODUZIR.md` | o caminho operacional | ~630 |

### Onde está o código

```
src/control/contract.py     a fórmula canônica — K, disparidade, defocus, rejeição
src/qc/                     gates (medem sempre, bloqueiam só com limiar), métricas,
                            registro de rejeição, refinamento da região em foco
src/renderer/               adaptador in-process do BokehMe, harness de verificação,
                            calibração da Eq. 5 por seção áurea
src/dataio/                 codificação da profundidade, contrato de amostra, writer,
                            split por cena materializado
src/model_runtime/          Depth Pro, BiRefNet, DeblurNet — um backend cada, sem cascata
src/sources/                adaptadores de fonte: RealBokeh, LFDOF, e os carregadores
                            de pixels correspondentes
src/routes/                 route_b.py, route_c.py
scripts/                    entrypoints e ferramentas de calibração/validação
slurm/                      os quatro jobs
tests/                      16 arquivos de teste
```

### Para verificar uma afirmação específica

| afirmação | leia | e rode |
|---|---|---|
| "o sinal de controle é dimensionalmente coerente" | `CONTRATO.md:59-68` | `tests/test_contract.py` |
| "o renderer é disco, não gaussiana" | `ACHADOS.md`, seção do job 32212 | `output/renderer_verification.json` |
| "o raio renderizado é linear em K" | idem | idem, `radius_response_*` |
| "não há fallback numérico" | `src/control/contract.py`, `REJECTION_REASONS` | `grep -rn "except:" src/` → vazio |
| "`max_coc` não pode variar por amostra" | `src/dataio/sample.py`, `validate_metadata` | passe `10.510746` e veja rejeitar |
| "o mapa de defocus é reconstruível dos escalares" | `src/dataio/sample.py:13-16` | `tests/test_route_c.py` |
| "o split não vaza por cena" | `src/dataio/split.py`, `check_no_leak` | `tests/test_dataio.py` |
| "os gates podem reprovar" | `tests/test_publish_release.py` | uma reprovação por checagem |
| "a máscara do BiRefNet erra o plano de foco" | `reference/MEDICAO_PLANO_FOCO.md` | não reproduzível sem GPU |

---

## 3. Como rodar os testes

```bash
cd "<raiz>/bokehnet-regen"
PYTHONPATH=src:scripts:tests \
  /Users/juliadollis/Projects_Code/AKCIT/.venv/bin/python \
  -m unittest discover -s tests -p "test_*.py"
```

**O comando do `README.md` não funciona.** Ele usa `PYTHONPATH=src` e `python3`; o python
do sistema desta máquina não tem numpy, e vários testes importam de `scripts/` e de
fixtures em `tests/`.

### O número real, e ele se mexeu enquanto eu media

| momento | resultado |
|---|---|
| 19:07:39 | `Ran 519 tests` · `OK (skipped=7)` |
| 19:11:37 | `Ran 521 tests` · `OK (skipped=7)` |
| **19:31:45** | **`Ran 650 tests in 6,6s` · `OK (skipped=7)`** |

**Zero falhas e zero erros nas três execuções.** Fui avisado de que ~20 testes estariam
quebrados neste momento; não observei nenhuma falha, e reporto o que medi.

O crescimento não é ruído: entre 19:11 e 19:32, `src/` foi de 7.993 para **9.297 linhas**,
e passaram a existir `src/routes/route_b.py` (1.168 linhas), `scripts/run_route_b.py`
(623), `src/sources/level_selection.py` (136), `tests/test_route_b.py` (104 testes) e
`tests/test_level_selection.py` (23). Três outros agentes trabalhavam no repositório em
paralelo. **Reverifique qualquer número de estado deste documento antes de citá-lo.**

### O que os 650 testes cobrem, por arquivo

| arquivo | testes | o que trava |
|---|---|---|
| `test_lfdof.py` | 112 | adaptador do LFDOF: parse de nome, chave de cena por split, `level` 1-based, alinhamento, enumeração |
| `test_route_b.py` | 104 | a rota B recém-escrita |
| `test_sources.py` | 73 | adaptador da RealBokeh: join com `metadata/`, `target_avs[level-1]`, slugs de fonte |
| `test_deblurnet.py` | 65 | a tripla indivisível `(repo, arquivo, main_adapter)`, e a geometria de resize/crop |
| `test_dataio.py` | 45 | schema da amostra, codificação da profundidade, manifesto, split, vazamento |
| `test_route_c.py` | 37 | rota C ponta a ponta com dublês, censura, reconstrução do mapa a partir dos escalares |
| `test_contract.py` | 33 | invariantes do contrato: unidades, mediana na disparidade, resolução, ausência de fallback |
| `test_gates.py` | 33 | cada gate nos **dois** sentidos — passa e reprova |
| `test_validate_focus_refinement.py` | 26 | inclusive o caso "PIOROU" e o aviso de "mais amostras pioraram" |
| `test_focus_region.py` | 24 | a armadilha da nitidez absoluta: prova que a medida absoluta erra e a razão não |
| `test_level_selection.py` | 23 | o teto de níveis por cena |
| `test_publish_release.py` | 23 | **uma reprovação por checagem do publicador** |
| `test_mirror_images.py` | 20 | índice do espelho, leitura sequencial, amostragem de piloto por cena |
| `test_model_runtime.py` | 15 | proveniência e hash dos modelos |
| `test_renderer.py` | 12 | harness de verificação e calibração da Eq. 5 |
| `test_rejection.py` | 5 | o registro que substitui o fallback |

### Os 7 skips, e não são maquiagem

**4** por `pyarrow não instalado nesta máquina` — os testes que leem parquet de verdade.
**3** por `toca a rede; ligue com BOKEHNET_HF_TESTS=1` — os que batem no espelho privado.
Os dois grupos rodam no cluster. Nenhum skip esconde falha.

---

## 4. O que foi executado de verdade em GPU

Três jobs, todos em 2026-09-10, partição `h100n3`. Rodaram **ao lado** do job 32186
(`deblur-n2-4gpu`), que não é nosso e não foi tocado — a QOS `onejob` permite 2 jobs
simultâneos.

| job | o que rodou | resultado | artefato versionado? |
|---|---|---|---|
| **32212** | `scripts/verify_renderer.py` contra o BokehMe real | `COMPLETED`, exit 0. Os três testes passaram | **sim** — `output/renderer_verification.json` |
| **32224** | piloto da rota C, 204 pares, 162 aceitas, 29 cenas | produziu a medição que mudou o método | **não** |
| **32231** | `scripts/diagnose_empty_masks.py`, 12 cenas | mostrou que o BiRefNet **declina**, não falha | **não** |

### O que o job 32212 mediu

```
edge_width_ratio        0,1434     disco (< 0,5); gaussiana seria ≈ 1,43
bokeh_classical         slope 0,96185  intercept +1,0260 px  resíduo 8,9e-15 px
bokeh_pred (usada)      slope 0,98729  intercept +0,7917 px  resíduo 0,4168 px (1,09%)
bokeh_neural            slope 0,0994   intercept +128,2 px   — não é renderer isolado
K testado               {8, 16, 32, 64, 96}, raios esperados 3,2 a 38,4 px
pipeline_usa_highlight  false
```

Com proveniência criptográfica: commit do BokehMe, sha256 do `pipeline` extraído por AST,
sha256 do `scatter.py` **pós-patch**, e sha256 dos dois checkpoints.

### O que o job 32224 mediu — e por que mudou o método

A `focus_disparity` tirada da máscara do BiRefNet fica dentro de ±25% da distância de foco
**medida na captura** em **35,2%** dos casos (57 de 162). A razão mediana é 0,579 — a
máscara escolhe um plano ~1,7× mais longe. A origem publica essa distância com incerteza
mediana de **±0,010 m**, o que elimina a hipótese de gabarito ruim. E **20,6%** das
amostras (42 de 204) eram descartadas com `focus_mask_empty`, em 10 cenas inteiras.

O job 32231 achou a causa: probabilidade **exatamente 0,000**, e baixar o limiar de 0,5
para 0,05 não recupera nada. O BiRefNet **declina** — é segmentador de objeto saliente, e a
RealBokeh é feita de cenas.

O paper **antecipa** este problema, nestes mesmos dois datasets, e rejeita explicitamente
a estratégia de descartar (`paper.txt:364-368`). A resposta foi
`src/qc/focus_region.py` — refinamento automático por retenção de detalhe. Detalhe em
`reference/MEDICAO_PLANO_FOCO.md` e em `DESVIOS_DO_PAPER.md`, D4.

### O que **nunca** rodou

O piloto com o refinamento novo; `validate_focus_refinement.py`; `calibrate_thresholds.py`;
o lote completo da rota C; `publish_release.py`; a rota B; a rota A. **Nenhum dataset foi
gerado.** Nenhum limiar de gate tem número. O quadro completo está na §9 de
`reference/REPRODUZIR.md`.

### E dois avisos sobre o que sobreviveu

`logs/` e `output/` estão no `.gitignore`. Dos três jobs, **só o 32212 deixou artefato**;
os números do 32224 e do 32231 existem apenas como prosa. E o próprio
`renderer_verification.json` não carrega o número do job nem a data — o vínculo
"job 32212, 2026-09-10" está só em texto.

---

## 5. O que já está de pé, sem rodeio

Vale dizer, porque é substancial e um documento que só lista problemas engana tanto quanto
um que só lista acertos.

- **O contrato fecha dimensionalmente**, e a conversão está travada por teste nas duas
  direções. Três âncoras numéricas independentes caem na mesma faixa: 16,55 (reconstrução
  da tabela `kfix`), 20,1 (EXIF) e 15,0 (default oficial), com a Fig. 12 do paper varrendo
  `K ∈ {0, 5, 10, 15}`.
- **O renderer foi medido, não declarado.** Disco e não gaussiana, linear em K, escala
  0,9873, com proveniência criptográfica de qual renderer produziu o número.
- **Não há fallback numérico no caminho do rótulo.** Vocabulário de rejeição fechado,
  `reject()` levanta `KeyError` para slug não registrado, e usa `if` e não `assert` —
  porque `python -O` desliga `assert` justamente no run de produção.
- **`max_coc` é inexpressável como variável.** Não é campo, não é parâmetro; um revisor
  mediu que, quando era, uma amostra com o valor exato do experimento `kfix` atravessava o
  writer até o disco sem erro.
- **O split é por cena e materializado no dataset**, e o gate que o confere **pode
  reprovar** — a versão anterior era tautológica, e o teste chamado "detecta vazamento"
  afirmava `clean is True`.
- **As 11 checagens que reprovam um release têm, cada uma, um teste que a faz reprovar.**
- **A censura é gravada em vez de virar medida.** `k == 300` exato em 47% das amostras do
  release anterior não significava "K físico é 300"; significava "o ótimo ainda crescia na
  borda".
- **Erros do próprio projeto estão registrados como erros**, inclusive um erro de método
  que teria feito descartar um renderer correto, e uma frase falsa sobre o paper que o
  próprio agente de fidelidade proíbe escrever.

---

## 6. As perguntas que um cético faria

### 1. "O paper escreve `D` como profundidade. Vocês operam em disparidade. Isso não é uma mudança de modelo?"

É, e está declarado. Dois argumentos independentes: a Eq. 3 (`paper.txt:340`) produz `K` em
`px·mm`, e só fecha em pixel multiplicando por `|Δ(1/z)|`; e a inferência oficial faz
`disp = 1.0/depth` e tira a mediana disso. A palavra *disparity* de fato nunca é aplicada a
`D` no artigo — **conferido literalmente**: as duas ocorrências são no related work e num
título de referência.

→ `CONTRATO.md:26-34`, `DESVIOS_DO_PAPER.md` D1.

**A fraqueza, e é real:** o código oficial que decide o desvio **não está neste
repositório**. Ver pergunta 12.

### 2. "`max_coc = 100` não está no paper. De onde saiu?"

De `Inference_bokehNet.py:20`. A Eq. 2 do paper é crua, sem normalizador — **conferido**.
Adotar o valor oficial torna o dataset comparável com os pesos publicados. A alternativa
medida foi pior: um normalizador derivado de uma rota só levou o LF-Bokeh a +0,8288
enquanto o RealDOF ia a −0,4599.

→ `DESVIOS_DO_PAPER.md` D2.

### 3. "Vocês substituíram 8 horas de anotação humana por um filtro automático. Isso é equivalente?"

**Não, e o repositório não afirma que é.** O paper corrige a máscara à mão; nós corrigimos
por retenção de detalhe. O desvio é justificado por medição — 35,2% de concordância da
máscara crua, 20,6% de descarte — e cada amostra carrega `focus_source` e
`focus_was_refined`, para dar para treinar com e sem as refinadas e **medir** a diferença.

**Mas a validação ainda não rodou.** A régua é bater os 35,2% no bloco pareado de
`scripts/validate_focus_refinement.py`. Enquanto não rodar, o desvio é declarado e **não
está justificado por medição de melhoria**.

→ `MEDICAO_PLANO_FOCO.md`, `DESVIOS_DO_PAPER.md` D4 e D5.

### 4. "Os gates só medem, ou eles reprovam?"

Onze limiares de gate, **todos `None` por default** — medem e não bloqueiam, porque nenhum
tem número medido ainda. As 11 checagens do publicador **bloqueiam**, e cada uma tem um
teste que a faz reprovar.

**Duas exceções, e uma é um defeito aberto:** `focus_depth_plausible` tem defaults
não-`None` vindos de dois `[A]` explicitamente marcados como "calibrar no piloto" — e o
piloto **não consegue calibrá-los**, porque a amostra é rejeitada antes de entrar nas
estatísticas, a ferramenta de calibração não tem entrada para eles, e não há flag de CLI.

E há um gate que **não pode reprovar**: `highlight_decidido`, no laudo do renderer, é
`True` incondicional e entra no `all(...)` que autoriza o renderer. Impacto prático hoje:
nulo — os outros três critérios são medições reais. Mas é o padrão que o projeto persegue.

→ `DESVIOS_DO_PAPER.md` §3.4, `TABELA_DE_EVIDENCIAS.md` §13-bis.

### 5. "O renderer é mesmo o BokehMe, ou é um gaussiano com outro nome?"

É o BokehMe, e foi **medido em GPU**: `edge_width_ratio = 0,1434` (disco < 0,5; uma
gaussiana daria 1,43 independentemente de σ, valor derivado analiticamente e refeito nesta
auditoria). A pergunta é legítima porque o pipeline antigo **nunca instalou o BokehMe** —
caía sempre num gaussiano de 16 camadas com kernel travado em 51 px, que renderizou ~70K
alvos e calibrou os K de uma rota inteira.

**Uma ressalva que um avaliador atento vai encontrar:** `bokeh_classical` e `bokeh_pred`
têm raios medidos **idênticos em 4 dos 5 K**; só K=96 difere. O "resíduo 0,0000 px do
clássico" e o "0,4168 px do híbrido" são, portanto, a mesma informação vista de dois lados.
Não invalida o teste; enfraquece a frase.

→ `TABELA_DE_EVIDENCIAS.md` §9.

### 6. "Vocês afirmam 33 defeitos fechados e abertos. Como eu confiro?"

**Você não consegue.** O catálogo dos 33 defeitos vive em
`../genrefocus_deblurnet_paper/PLANO_REGERACAO_BOKEHNET.txt`, **fora deste repositório**. E
o próprio repositório discorda de si mesmo sobre o total: `REGISTRO.md:152` diz *"contagem
exata: 33, não 32"*, e `.claude/agents/defect-regression.md` ainda diz "32".

O placar também **para na etapa 4** (16 fechados / 3 no contrato / 14 abertos), enquanto as
etapas 5 a 9 acrescentaram ~5.500 linhas de `src/`. Para o paper: ou o catálogo entra em
`reference/`, ou o placar não é citável.

→ `TABELA_DE_EVIDENCIAS.md` §16.2 e §18.

### 7. "Quanto do dataset foi realmente gerado?"

**Zero.** Nenhum lote completo rodou. O que existe é o caminho inteiro, testado, mais um
piloto de 204 pares com uma versão anterior do método. Isto está dito em todos os
documentos e não é escondido em nenhum.

### 8. "A rota C do paper são dois datasets. Vocês só têm um?"

O paper exige LFDOF **e** RealBokeh em quatro lugares. O adaptador do LFDOF **existe** —
879 + 403 linhas, 112 testes, 28 medições próprias — mas
`scripts/run_route_c.py::_carrega_fonte` só trata `"realbokeh"`, e `--source lfdof` falha
com uma mensagem que hoje é **falsa** ("fonte ainda não tem adaptador"). Falta ligar.

O bloqueador que existia — um gate que rejeitaria **100% do LFDOF** com um slug afirmando
"abertura larga", sobre uma fonte que legitimamente não publica f-number — está **fechado**:
`GateResult.applicable` distingue "não existe para esta fonte" de "existe e deu ruim".

→ `DESVIOS_DO_PAPER.md` §3.2.

### 9. "A afirmação central é que consertar o rótulo recupera a controlabilidade. Mostre a medição."

**Não consigo mostrar, e esta é a resposta mais importante deste documento.**

A tabela de LVCorr (fase 1 +0,9059; oficial +0,8868; nossa fase 2 +0,4365; `kfix` +0,8288;
etc.) é o argumento inteiro do projeto — e **não tem, neste repositório, o harness de
avaliação, a definição da métrica, os checkpoints, a data nem o comando**. Foram oito
medições feitas antes deste repositório existir, em outro pipeline.

Enquanto isso não for versionado aqui, esses números são `[SP]` — sem procedência — e **não
devem entrar em paper como medição nossa**.

→ `TABELA_DE_EVIDENCIAS.md` §2.

### 10. "Vocês dizem que o dataset não cabe em disco e por isso não gravam os pixels. Mostre a medição de espaço."

Também não existe. Cinco números governam a decisão de armazenamento — "115 GB de folga",
"365 GB para regravar", "207,8 GB só de controle", "31 GB" e "~49 GB" — e **nenhum tem
`quota`, `df` ou derivação registrada**. Pior: `src/dataio/writer.py` diz "a cota é 500 GB
soft, 600 GB hard", e a conferência do cluster de duas etapas depois mediu **7,0 TB livres**
em `/raid`. As duas frases falam de sistemas de arquivos diferentes, e **nenhuma diz de
qual**.

O único número de orçamento que consegui refazer é o erro de quantização da profundidade:
**0,0003777 px** com K=50, e ele confere com os "0,000378 px" documentados. (A docstring do
módulo reporta `7,5e-4 px` para a mesma grandeza — está certo também: é o passo inteiro
contra o meio passo. Nenhum dos dois lugares diz qual convenção usa.)

→ `TABELA_DE_EVIDENCIAS.md` §17.

### 11. "As citações `arquivo:linha` conferem?"

**As de `paper.txt`, quase todas.** Conferi 59 citações uma a uma contra o arquivo. Nenhuma
está completamente errada; **seis** estão deslocadas, e uma aponta para uma linha que não
contém nada do que se afirma. A lista está na §7.

**As de `src/`, não mais.** Amostrei 16 citações das auditorias de rota e **todas as 16
estão deslocadas** — não porque estivessem erradas quando escritas, mas porque os arquivos
cresceram. As constantes citadas ainda existem com o mesmo nome, então o **conteúdo** das
auditorias continua válido; um avaliador que confira as linhas conclui que a auditoria é
descuidada, e não é.

→ `TABELA_DE_EVIDENCIAS.md` §16.1.

### 12. "O que eu, recebendo só esta pasta, não consigo verificar de jeito nenhum?"

Catorze afirmações têm procedência fora deste repositório. As quatro que mais doem:

| não verificável | vive em |
|---|---|
| `MAX_COC = 100.0` e o default `K = 15,0` | `Genfocus/Inference_bokehNet.py` |
| `disp = 1/depth` e `median(disp[mask])` na inferência oficial | idem |
| **o catálogo dos 33 defeitos** | `../genrefocus_deblurnet_paper/PLANO_REGERACAO_BOKEHNET.txt` |
| que a variante main+cond sai **lavada** sem `main_adapter` | `../HANDOFF_PROJECT_HISTORY.md` |

As três primeiras sustentam, respectivamente, o normalizador, a troca de espaço
profundidade→disparidade, e o placar de progresso. **São as três coisas mais citadas do
projeto.**

→ `TABELA_DE_EVIDENCIAS.md` §16.2.

### 13. "Achei uma contradição interna. Vocês sabem?"

Provavelmente. Dezoito estão catalogadas. As que um avaliador encontra primeiro:

- `README.md` diz "48 testes passando" e "1.386 linhas em `src/`" — são 650 e 9.297.
- `src/sources/mirror_images.py` diz "o espelho são 85 shards"; são 96.
- `src/sources/realbokeh.py` diz "o espelho publica só o split `train`"; publica três. E
  `src/sources/lfdof.py`, ao lado, **registra a correção**.
- `CLAUDE.md` diz "busca ternária"; é seção áurea.
- `PLANO_EXECUCAO.md` afirma que `deblur_variant` é "conferido na publicação";
  `publish_release.py` não tem essa checagem.
- Quatro lugares dizem que a janela de 33 px cobre "6,4% do lado longo a 512x683"; a grade
  que o código produz é 384×512, e 33/512 = 6,4%. O rótulo está errado, o número está certo
  — e `ACHADOS.md` contém as duas versões.

Nenhuma afeta o rótulo. Todas afetam quem lê.

→ `TABELA_DE_EVIDENCIAS.md` §13 e §12.

### 14. "Rodei a suíte e o resumo da rota C imprime `k_value = 18,01` contra `k_analytic = 1,63`, mas o texto logo abaixo diz que se espera `k_value < k_analytic`. Qual das duas está errada?"

Nenhuma. É um artefato de fixture: a imagem de teste tem lado longo de 96 px e o dublê de
renderer impõe K=18 por construção, enquanto `k_analytic` é a Eq. 3 sobre aquela mesma
imagem minúscula. São dois números que não descrevem a mesma cena.

Mas é o **único** lugar do repositório em que os dois aparecem lado a lado, e qualquer um
que rode a suíte vai vê-los contradizendo a legenda. Conserto de meia linha; ainda não
feito.

→ `TABELA_DE_EVIDENCIAS.md` §20.

### 15. "Por que eu deveria acreditar num repositório que outros agentes estão editando enquanto ele é auditado?"

Não deveria acreditar — deveria conferir, e é para isso que os quatro documentos existem.
Toda medição de estado deste conjunto está datada, e a deriva está medida: 519 → 521 → 650
testes em 24 minutos, `src/` de 7.993 para 9.297 linhas.

Uma divergência que eu diagnostiquei como aberta (a multiplicidade de até 21 níveis por
cena, contra os "2 a 4" do paper) **foi consertada durante a redação**, com
`--max-levels-per-scene` default 4. Deixei o diagnóstico inteiro de pé com o conserto
anotado, porque o raciocínio é o que vai para o paper — e porque um desvio consertado sem
registro do motivo volta na revisão seguinte.

---

## 7. Correções às citações de `paper.txt`

Conferi as 59 citações que o repositório faz ao paper, uma a uma, contra
`reference/paper.txt` (1.203 linhas; supplement a partir de 966).

### A armadilha das duas numerações — confirmada, e mais estreita do que se dizia

Prova cruzada: `[67]` é DiffCamera nas legendas (429, 444) e **CLIP-IQA** no corpo (553);
`[80]` é Restormer nas legendas (429, 439) e **Generative Photography** no corpo (240, 528,
996).

**Mas a numeração antiga aparece só nas legendas das Figs. 4(a) e 4(b)** — linhas 429, 434,
439, 444. As legendas da Fig. 3 e das Tabs. 3/4/6 usam a numeração **atual**. A regra
registrada no projeto ("legenda de figura não vale") é forte demais; a regra correta é
**estas quatro linhas não valem**. Isso importa porque a definição de `pixel_ratio` — que
resolve um silêncio do corpo — está justamente numa **legenda de figura**
(`paper.txt:1186-1187`), e ela **vale**.

### As citações erradas ou deslocadas

| citação | onde é usada | correto |
|---|---|---|
| `paper.txt:264` para *"Real bokeh image · Real AIF image"* | `ROTA_C_AUDITORIA.md` | **errada**. A linha 264 traz os rótulos das três rotas. As duas frases estão **ambas em `:271`** |
| `paper.txt:326-328` para *"randomly sample a focus plane and a target bokeh level K"* | `CONTRATO.md:163-164` | a frase está em **`:329-330`**; a linha 326 ainda é o parágrafo anterior |
| `paper.txt:326-334` para o §3.2(a) inteiro | vários | o parágrafo começa em **328** |
| `paper.txt:313-315` para *"D é o mapa de profundidade… [7]"* | `CONTRATO.md:29` | a frase é **314-315**; a 313 é vazia, e o `[7]` está na **315** |
| `paper.txt:512-516` para "backbone FLUX.1-dev" | `CONTRATO.md:142` | o backbone está em **511**, fora do intervalo. E o paper grafa **"FLUX-1-dev"** |
| `paper.txt:1050` para *"Additional Comparison with DiffCamera"* | `ROTA_A_AUDITORIA.md:49` | o título está em **1048** |
| `paper.txt:676-690` para a ablação da Tab. 6 | `ROTA_A_AUDITORIA.md:638` | a tabela é **670-680** e a prosa **684-692**; o intervalo corta as duas. A frase citada existe, em **688-689** |
| `paper.txt:579-580` para *"maximizes SSIM"* | `ROTA_C_AUDITORIA.md:267` | a linha **não menciona SSIM** — diz busca binária *"following [43, 87]"*. Só **561-562** sustenta o SSIM |
| `paper.txt:1043-1044` para "usado **só** para o LoRA de formato de abertura" | `ROTA_A_AUDITORIA.md:160` | o paper diz *"pretrain and refine our learning strategy for controllable bokeh shape"*; o **"só"** é nosso |

### Duas linhas que o repositório citava sem número, e agora têm

- *"public implementations typically omit this functionality"* (§3.3, sobre formato de
  abertura) → **`paper.txt:406-407`**.
- A Eq. 6, `I_syn = R(I_aif, D; D_focus, K, s)` → **`paper.txt:421`**, com o shape kernel
  definido em 418-419.

### Uma nuance que muda a etiqueta de quatro afirmações

As entradas bibliográficas de `[27]`, `[86]`, `[52]` e `[57]` **não contêm** os nomes
curtos "EBB", "BiRefNet", "LFDOF" e "RealBokeh". A ligação é feita no corpo, em uma frase
cada. Então *"o paper usa a RealBokeh"* é leitura do paper; *"a RealBokeh é o
`timseizinger/RealBokeh_3MP`"* é **inferência nossa**. O paper não publica URL de dataset
nenhum.

### Um erro do próprio paper, registrado

A legenda da Fig. 3(c) (`paper.txt:296-297`) diz que o K da rota (c) é estimado *"following
Eq. (2)"*. A Eq. 2 **consome** K, não o estima; o corpo e o supplement são inequívocos de
que o K vem do sweep da Eq. 5. Seguimos o corpo.

### E as citações que **conferem**

A esmagadora maioria. As cinco equações, o parágrafo do refinamento manual, as 4-8 s por
imagem e as 8 h, os "2 a 4 images per set", os "13K newly curated", a definição de
`pixel_ratio` na Fig. 16, a decisão explícita de **não** usar a EXIF para `D_focus`, a
`K ∈ {0,5,10,15}` da Fig. 12, o tiling *"during inference"*, todo o §4.1 de treino, e as
nove entradas bibliográficas — **todas conferidas literalmente**.

---

## 8. O que um avaliador externo ainda **não** consegue verificar

Em ordem de gravidade. Esta lista é o produto mais útil desta auditoria.

1. **A afirmação central do projeto.** Os oito números de controlabilidade que justificam
   o trabalho inteiro não são refazíveis: o harness de LVCorr não está aqui.
2. **A autoridade que decide o desvio maior.** `Inference_bokehNet.py` — que fixa
   `max_coc = 100`, o default `K = 15` e a operação em disparidade — não está aqui.
3. **O placar de progresso.** O catálogo dos 33 defeitos não está aqui, e o repositório se
   contradiz sobre o total.
4. **Os números do piloto.** 35,2% e 20,6% mudaram o método da rota C e existem só como
   prosa: nem log de SLURM, nem `run_config.json`, nem `rejections.jsonl` foram versionados.
5. **O orçamento de disco.** Cinco números sem medição, e dois deles em tensão aparente.
6. **O custo de GPU por amostra.** Não medido, e é o item dominante do orçamento.
7. **Se `SSIM(K)` é unimodal sobre foto real.** É a hipótese da seção áurea e da busca
   binária do paper, e foi verificada só com um pico sintético.
8. **Se o refinamento automático melhora.** A ferramenta existe e nunca rodou; a régua
   (35,2%) está fixada e não foi batida nem refutada.
9. **As 28 medições do adaptador do LFDOF.** São boas, e vivem numa docstring de um arquivo
   que outros agentes editam, em vez de em `reference/ACHADOS.md`.
10. **~2.250 linhas de `src/` sem etapa no `REGISTRO.md`** — o adaptador do LFDOF e a
    DeblurNet —, contra a regra de abertura daquele arquivo: *"nada entra sem
    justificativa"*.

**Os quatro consertos de maior retorno**, todos baratos: versionar o harness de LVCorr;
trazer `Inference_bokehNet.py` e o catálogo de defeitos para `reference/`; gravar
`SLURM_JOB_ID`, hostname e timestamp dentro dos JSONs de laudo; e versionar, por job, o
`run_config.json` e o `rejections.jsonl` — sem os pixels.
