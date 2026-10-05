# Registro de mudanças

**2026-09-10.** Tudo que foi criado, cada decisão e o porquê, e a auditoria final
contra o catálogo de defeitos.

Regra deste arquivo: nada entra sem justificativa, e limiar não medido é marcado `[A]`.

---

## 1. O que foi criado

| arquivo | linhas | o que é |
|---|---|---|
| `src/control/contract.py` | 380 | a implementação única do sinal de controle |
| `src/control/__init__.py` | 20 | superfície pública do pacote |
| `src/qc/rejection.py` | 130 | registro de rejeição — o que substitui o fallback |
| `src/qc/__init__.py` | 4 | idem |
| `tests/test_contract.py` | 300 | 31 testes de invariante |
| `tests/test_rejection.py` | 55 | 5 testes do registro |
| `reference/CONTRATO.md` | — | a especificação canônica, uma página |
| `reference/ACHADOS.md` | — | tudo que foi medido, com `[M]`/`[I]`/`[A]` |
| `reference/paper.txt` | 1203 | texto do paper, grepável, inclui o supplement |
| `.claude/agents/*.md` | 9 arquivos | os revisores |
| `CLAUDE.md`, `README.md` | — | regras do cluster e do código |

**Estado dos testes: 36 de 36 passando.** Rodados de verdade, não presumidos:

```
PYTHONPATH=<pydeps>:src python3.11 -m unittest discover -s tests -p "test_*.py"
Ran 36 tests in 0.017s — OK
```

---

## 2. Decisões, uma a uma

### 2.1 A fórmula canônica

```python
z          = depth_pro(aif)              # METROS
disp       = 1.0 / z                     # 1/m
focus_disp = median(disp[mask])          # NA disparidade
K          = k_eq3 / 1000.0
max_coc    = 100.0
defocus    = clip(abs(K*(disp - focus_disp)) / max_coc, 0.0, 1.0)
```

**Por quê.** É operação por operação o que `Inference_bokehNet.py:94,118,138-140` faz.
Onde o paper cala — e ele cala sobre o normalizador, porque a Eq. 2 é crua — a
autoridade é o código oficial. Isso torna o dataset **comparável com os pesos
oficiais**: dá para inicializar a partir deles, avaliar no mesmo harness e comparar K
absoluto. Com o `max_coc = 10,5107` do kfix nada disso valia.

Ancorado por três medições independentes que convergem: `k_eq3` mediano da kfix ÷ 1000
= **16,55**; mediana calculada da EXIF = **20,1**; default oficial = **15,0**. A Fig. 12
do paper varre K ∈ {0, 5, 10, 15}.

### 2.2 `focus_disparity`, não `focus_depth`

Calculado como `median(1/z[mask])`, e o mapa **nunca** usa `1/focus_depth_m` de volta.

**Por quê.** `median(1/z)` não é sempre `1/median(z)`: a invariância sob transformação
monótona só vale em contagem ímpar, e `np.median` faz a média dos dois centrais quando
é par. Contraexemplo no teste: `z = [1, 2, 4, 8]` dá `focus_disp = 0,375` contra
`1/median(z) = 0,3333`. A diferença é minúscula em máscara grande — e o ponto do
contrato é não ter esse tipo de folga.

`focus_depth_m` fica nos metadados **só para leitura humana**, documentado como tal.

### 2.3 Sem fallback: 15 slugs de rejeição

Toda entrada ausente ou inválida levanta `SampleRejected(reason, detail)` com slug de
um conjunto fechado. O `RejectionLog` agrega e **todo run imprime o histograma**.

**Por quê.** O fallback `k = 50.0` produziu 11.635 de 11.635 amostras com K constante,
sem uma linha de erro. Um dataset com 60% das amostras e proveniência honesta vale
mais que um com 100% e um sexto carregando constante inventada.

Decisão de projeto no `RejectionLog`: a retomada pula só as **aceitas**. Rejeitada é
reprocessada, porque um gate recalibrado pode aceitá-la depois — e por isso o motivo
fica gravado em vez de a linha sumir.

### 2.4 As três constantes, cada uma justificada

| constante | valor | justificativa |
|---|---|---|
| `MAX_COC` | 100.0 | `Inference_bokehNet.py:20`. Congelada por release, gravada em cada amostra |
| `MM_PER_M` | 1000.0 | dimensional: Eq. 3 em px·mm multiplica 1/mm; a inferência usa 1/m |
| `FULL_FRAME_WIDTH_MM` | 36.0 | usada **apenas** com crop factor medido, nunca como default |

Pela regra do `fallback-hunter`, constante física em código é sempre suspeita. Estas
três passam porque são documentadas, congeladas e rastreáveis à fonte. A mesma forma
sem essas três coisas seria defeito.

### 2.5 `k_at_resolution` — o fator que ninguém aplicava

CoC em pixel escala com a resolução; disparidade não. Reduzir o lado menor para 512
exige `K' = K · 512/min(H,W)`, fator que varia por amostra (medidos 0,892 e 0,821).

**Por quê agora.** O paper treina em resolução nativa com tiling (§3.5), então nunca
enfrenta isso. Nós treinamos em crops de 512, e o erro de 10–20% por amostra entra
exatamente no sinal de controlabilidade.

### 2.6 O que NÃO é rejeitado, de propósito

`z_max == 10000` (teto do Depth Pro) **não** rejeita. Céu e infinito podem
legitimamente atingir o teto, e em disparidade `1/10000 ≈ 0` é inofensivo — é a
vantagem de sair da profundidade linear. Rejeitar todo `z_max == 10000` descartaria
**25,7%** da rota B sem motivo.

O que rejeita é `z_focus` no teto (plano de foco a 10 km é sentinela, não plano) e
faixa de profundidade degenerada.

**Correção registrada:** eu tinha escrito o gate errado no plano, conflando a medição
dos 25,7% — que a auditoria original flagrou como problema dos *canais geométricos*,
que usam `z` linear — com o mapa de defocus, que usa disparidade. Corrigido aqui e no
plano.

### 2.7 Limiares que assumi e não medi — marcados `[A]`

| constante | valor | risco |
|---|---|---|
| `FOCUS_DEPTH_MIN_M` | 0.05 | baixo |
| `FOCUS_DEPTH_MAX_M` | 1000.0 | **médio** — a faixa medida de `z_focus_m` vai a 10.000, então o topo legítimo é desconhecido. 1000 pega a sentinela sem descartar paisagem distante, mas é chute |
| `MIN_DEPTH_RANGE_RATIO` | 1.02 | baixo — muito permissivo de propósito |

Achei estes na auditoria do meu próprio código, aplicando o `fallback-hunter`. Estão
documentados no fonte como `[A]` e têm que ser calibrados pelo histograma de rejeição
do piloto. Isso segue a instrução de "testar primeiro, filtros depois": o valor
permissivo mede, o valor final sai do piloto.

### 2.8 Nove agentes, não cinco

Acrescentei quatro ao roster original, cada um cobrindo uma classe distinta:

- **`renderer-verifier`** — o contrato do BokehMe: disco e não gaussiana, raio igual a
  `K·Δdisp`, e os sete parâmetros que o paper não publica. Risco alto porque o
  BokehMe nunca foi instalado no pipeline antigo.
- **`runtime-smoke`** — "isto roda?". Classe própria porque `compileall` e AST não
  pegam `NameError`, e foi assim que um `import json` removido matou um pipeline.
- **`cluster-safety`** — a única classe de dano sem desfazer. Já houve um
  `rsync --delete` que apagou checkpoints.
- **`defect-regression`** — percorre os 33 defeitos e verifica se fecharam **e** se
  não abriram novos. Existe porque correção parcial já se mostrou pior que nenhuma: o
  `kfix` consertou a rota B, deixou a C na convenção antiga, e ficou com o pior LPIPS
  entre as variantes reais.

---

## 3. Auditoria final — os 33 defeitos

Contagem exata: **33**, não 32. Corrigido (`grep -cE "^\[.*\]" PLANO_REGERACAO_BOKEHNET.txt`).

### 3.1 Fechados com teste — 10

| ID | defeito | teste que trava |
|---|---|---|
| D1 | mapa gravado apaga o K | `DefocusPreservesK` (3 testes) — K diferente dá `max` diferente **depois de codificar** |
| RAIZ | treino e inferência em espaços diferentes | `Eq3Units.test_conversao_mm_para_metro_e_exatamente_1000` + módulo único |
| T4 | `max_coc` com três convenções | `MAX_COC` congelado; `test_max_coc_invalido_rejeita` |
| NOVO | K em pixels sem a resolução | `Resolution` (2 testes) |
| D2 | `k = 50,0` constante | `NoFallback.test_f_number_ausente_rejeita_em_vez_de_virar_k50` |
| D3 | média onde a Eq. 4 pede mediana | `FocusDisparity` (2 testes, com contraexemplo de contagem par) |
| D14 | `pixel_ratio` com a largura | `PixelRatio` (2 testes, incluindo a identidade da kfix) |
| D10 | escala métrica jogada fora | `Metadata.test_escalares_obrigatorios_presentes` |
| D11 | fallback que inverte a profundidade | `DepthValidation.test_backend_e_gravado`; sem `except` no módulo |
| — | sensor assumido como 36 mm | `NoFallback.test_sensor_sem_focal_35_rejeita_em_vez_de_assumir_36mm` |

### 3.2 Fechados no contrato, dependem das rotas — 3

D12 (máscara da bokeh), Q1 (teste de MI desligado), P1 (proveniência incompleta).
O contrato fornece o mecanismo — `control_metadata`, `depth_backend`, `RejectionLog` —
mas quem tem que usá-lo são as rotas, que ainda não existem.

### 3.3 Abertos, aguardando o código correspondente — 20

**Rota C (9):** espelho `akcit-pixel/RealBokeh`, D6 (AIF de `train/in/`), D5 (todos os
pares), D7 (metadata como validador), calibração do DepthPro por `focus_plane_distance`,
D8 (teto do sweep + SSIM), D9 (LFDOF), C8 (registro do par), as 1.028 cenas.

**Renderer (3):** D4 (BokehMe de verdade), custo do sweep, `is_final_label_renderer`.

**Rota A (4):** fonte trocada, uma amostra por imagem, D13 (filtro Laplaciano), P2
(procedência).

**Rota B (1):** B6 (gate de qualidade da AIF do DeblurNet).

**Transversais (3):** split por cena e dedup, `import json` (não se aplica — código
novo), compatibilidade (`pack_files_to_hf`, reexports v1, `s1` legado).

### 3.4 Defeitos NOVOS? As seis perguntas

| pergunta | resposta |
|---|---|
| Constante física apareceu no código? | 3, todas documentadas e justificadas (§2.4). Mais 3 limiares `[A]` que eu mesmo achei e marquei (§2.7) |
| Caminho de exceção engole em vez de rejeitar? | Não. `grep` por `except:` e `except Exception: pass` em `src/`: zero ocorrências em código (a única linha é docstring descrevendo o defeito antigo) |
| Quantidade em pixel viaja sem a resolução? | **RISCO RESIDUAL.** `k_at_resolution` existe, mas nada **força** o chamador a usá-la. Mitigação prevista: gravar a resolução processada nos metadados e o `data-contract` conferir |
| Campo de proveniência pode mentir? | **RISCO RESIDUAL.** `depth_backend` é string passada pelo chamador. Mitigação: quem seta é o loader do modelo, não a rota |
| Algo foi normalizado por imagem, rota ou fonte? | Não. `MAX_COC` é global e congelado, com teste |
| Nome legado com semântica antiga? | Não. `s1` **não existe** no contrato novo |

Varredura de nomes indefinidos (o método do `runtime-smoke`) nos 4 módulos:
**limpo em todos**.

### 3.5 A pergunta que fecha

> Se este dataset for gerado hoje e treinado, qual número do `ACHADOS.md` mudaria?

**A LVCorr da fase 2, de +0,4365 para perto de +0,8868 no LF-Bokeh.** É o número que
o `kfix` já moveu para +0,8288 consertando só o K da rota B — a prova de que o rótulo
era o gargalo. Com B e C na mesma convenção, a expectativa é fechar o resto da
distância, e o LPIPS não pagar o preço que o `kfix` pagou por misturar convenções no
mesmo run.

**Mas etapa 1 sozinha não gera dado nenhum.** O contrato é condição necessária e
insuficiente. O veredito do `defect-regression` hoje é: **não pode gerar dado ainda** —
20 defeitos abertos, incluindo 3 dos 8 âncoras (D4, D5/D6).

---

## 4. O que NÃO está feito

Nada de `src/routes/`, `src/renderer/`, `src/dataio/`. É deliberado: escrever as rotas
antes do renderer funcionar seria gambiarra, e o sweep de K não tem sentido sem o
teste de calibração do raio.

Ordem que eu seguiria a partir daqui:

1. `src/renderer/bokehme.py` — in-process, modelo carregado uma vez, busca ternária, e
   os três testes do `renderer-verifier`. **Bloqueia rota C e rota A.**
2. Verificar o `image_focus` do espelho `akcit-pixel/RealBokeh` — 2 minutos, decide se
   a rota C parte do espelho ou do dataset bruto.
3. `src/routes/route_c.py` — RealBokeh completa, AIF de `train/in/`, sweep da Eq. 5.
4. `src/dataio/` — writer, split por cena materializado, proveniência completa.
5. `src/routes/route_b.py` — duas versões, com o `main_adapter` condicional.
6. `src/routes/route_a.py` — fonte corrigida, N sorteios por imagem.

---

## 5. Itens `[A]` ainda abertos

Nenhum destes bloqueia a etapa 1, e todos bloqueiam alguma etapa posterior:

1. Os **59 arquivos e 1 cena** que faltam no espelho contra o `gt/` bruto.
2. O `_aligned` do `file_name_base` — que registro geométrico foi aplicado.
   ~~3. O `image_focus` do espelho~~ — **FECHADO em 2026-09-10 por comparação de
   sha256**: é o `train/in/<id>_f22.JPG` byte a byte, 2000×1500 sem reencode. Ver
   `ACHADOS.md`.
3. A hipótese de que as **1.028 cenas** são rejeição de QC (`grep -c rejected_qc`).
4. A **largura do sensor da RealBokeh** — os 36,0 mm implícitos são suspeitos de bons
   demais.
5. Os **30,33%** com crop factor exatamente 1,0 na rota B — auditar por `make/model`.
6. Os sete parâmetros do BokehMe que o paper não publica — congelar e declarar.
7. Os três limiares que assumi em §2.7.

---

## 6. Sobre "100% igual ao paper"

Não é alcançável, e o registro tem que dizer isso. O repositório oficial não publicou
código de treino, dados, o benchmark LF-Bokeh, limites de K, limiar de SSIM,
configuração do renderer, resolução de treino, otimizador nem scheduler.

O que é alcançável, e o que este contrato entrega: uma reprodução **pública,
matematicamente coerente e auditável**, com cada desvio declarado. Os desvios
conhecidos até aqui são três, e todos estão escritos:

1. `max_coc = 100.0` — o paper não normaliza; a autoridade é o código oficial.
2. A revisão manual de máscara vira filtros automáticos — o paper descreve ~8 h de
   anotação, que é a parte menos reproduzível dele.
3. O K analítico do `metadata/` da RealBokeh é **validador**, não rótulo — o rótulo
   continua sendo o sweep da Eq. 5, fiel ao paper.

---

# Etapa 2 — renderer e Eq. 5

**2026-09-10, mesma sessão.** O renderer bloqueava rota C e rota A.

## 7. O que foi criado

| arquivo | linhas | o que é |
|---|---|---|
| `src/qc/metrics.py` | 90 | SSIM da Eq. 5 e variância do Laplaciano, em numpy puro |
| `src/renderer/verification.py` | 175 | o harness dos três testes do `renderer-verifier` |
| `src/renderer/calibration.py` | 175 | Eq. 5: bracket + seção áurea + censura |
| `src/renderer/bokehme.py` | 230 | adaptador in-process do BokehMe público |
| `src/renderer/__init__.py` | 20 | superfície pública |
| `tests/test_renderer.py` | 215 | 12 testes do harness e da calibração |

**48 de 48 testes passando.** SSIM verificado: `ssim(a, a) = 1.0` exato.

## 8. Decisões da etapa 2

### 8.1 O harness mede a saída, não o renderer

`verification.py` não conhece o BokehMe. Recebe qualquer `render_fn` e mede o borrão
produzido. Assim dá para **testar o próprio harness** contra um disco sintético de
raio conhecido, aqui, sem GPU — e rodar o mesmo código contra o BokehMe no cluster.

Os renderers sintéticos vivem no **teste**, não em `src/`. Ter um segundo renderer em
produção é justamente o defeito que se quer evitar: o pipeline antigo tinha um
gaussiano de fallback que virou o renderer de verdade.

### 8.2 Disco contra gaussiana, com discriminador fechado

`edge_width_ratio` = largura da transição 90%→10% do pico, normalizada pelo raio de
meia altura. Para uma gaussiana `exp(-r²/2σ²)` vale **1,43 independente de σ**
(r90 = 0,459σ, r50 = 1,177σ, r10 = 2,146σ). Para um disco tende a 0.

Limiar: `< 0,5` é disco, `> 1,0` é gaussiana. O teste confere as duas direções — o
discriminador precisa **reprovar** o renderer antigo, senão não serve.

### 8.3 Raio por energia acumulada, não por limiar de intensidade

O brilho do ponto se espalha, então um limiar fixo mede coisa diferente conforme o
raio. Energia acumulada é invariante. Para um disco uniforme, 95% da energia está
dentro de `√0,95·R`, e o valor é corrigido para devolver R. Verificado contra raios
alvo de 5, 12 e 25 px.

### 8.4 Seção áurea, não grid

O antigo: 24 pontos grossos + 16 finos = ~40 `subprocess`. Aqui: grid grosso de 7
pontos para localizar o máximo, depois seção áurea dentro do bracket, que reaproveita
uma avaliação por iteração. **Medido no teste: ≤ 25 avaliações**, todas in-process.

Escolhi seção áurea em vez de busca ternária — ternária gasta 2 avaliações por
iteração e encolhe para 2/3; a áurea gasta 1 e encolhe para 0,618. A documentação
anterior dizia "ternária"; corrigido.

### 8.5 Sem normalizar a disparidade para o BokehMe

O `demo.py` lê um PNG de disparidade, renormaliza para [0,1] e monta
`defocus = K*(disp − disp_focus)/defocus_scale`. Como já temos profundidade métrica, o
adaptador monta o tensor `defocus` **direto do CoC canônico**, eliminando a
quantização de 8 bits do PNG e a viagem de ida e volta pela normalização.

`k_for_bokehme` continua no contrato, testado, para quem usar a CLI.

### 8.6 Saída em float — o primeiro erro que os testes pegaram

Quantizar é gravação, não renderização. Um ponto de 255 espalhado num disco de raio 12
dá **0,56 por pixel**, que `astype(np.uint8)` trunca para zero: a imagem inteira vira
preto e o harness mede raio 0. Medido: raio 5 → 3,2/px; raio 12 → 0,56; raio 25 → 0,13.

Os testes reprovaram na primeira execução por exatamente isso, e a correção vale para
o adaptador real.

### 8.7 API do BokehMe conferida, não adivinhada

Busquei o `demo.py` do repositório público antes de escrever o adaptador:

```python
from neural_renderer import ARNet, IUNet
from classical_renderer.scatter import ModuleRenderScatter
pipeline(classical_renderer, arnet, iunet, image, defocus, gamma, args)
```

Os defaults de arquitetura em `BokehMeConfig` vêm de lá e **não são escolha nossa**.

### 8.8 Bug encontrado no meu próprio SSIM

A primeira versão do `_blur` somava um array `(H+2r, W)` num acumulador `(H+2r, W+2r)`
e quebrava por broadcast. Os testes do renderer pegaram. Os shapes intermediários
agora estão escritos à mão e comentados.

### 8.9 Validação barata antes do `import torch`

Config inválida e checkout ausente apareciam como `ModuleNotFoundError: No module
named 'torch'`. Reordenado: valida caminho e `output` primeiro, importa torch depois.

## 9. `[A]` fechado nesta etapa

**O `image_focus` do espelho `akcit-pixel/RealBokeh`** — confirmado por sha256, não
por inferência. Cena 1038: `image_focus` byte-idêntico entre `level_3` e `level_4`, e
igual ao `train/in/1038_f22.JPG`; nenhuma das 5 imagens de `train/gt/1038/` bate.
2000×1500 preservado, sem reencode.

**Consequência:** a rota C consome o espelho direto, D5 e D6 fecham juntos, e o
join com `metadata/<id>.json` por cena dá o f-number de cada `level`.

## 10. Auditoria atualizada — 33 defeitos

| status | antes da etapa 2 | agora |
|---|---|---|
| fechados com teste | 10 | **13** |
| fechados no contrato, dependem das rotas | 3 | 3 |
| abertos | 20 | **17** |

Fecharam nesta etapa:

| ID | defeito | teste |
|---|---|---|
| D4 | BokehMe nunca instalado | `DiscNotGaussian` + `RadiusMatchesContract` (7 testes); adaptador in-process sem fallback |
| D8 | 47% do K no teto do sweep | `test_marca_censura_quando_o_otimo_encosta_no_teto`, `test_expande_o_teto_antes_de_censurar` |
| NOVO | sweep a ~40 processos por amostra | `test_orcamento_de_avaliacoes_e_baixo` (≤ 25, in-process) |

Parcialmente fechado: **D6/D5** — a evidência está fechada (`[M]`), falta a rota C
consumir o espelho.

### Defeitos novos? As seis perguntas, de novo

| pergunta | resposta |
|---|---|
| Constante física nova? | `GAUSSIAN_EDGE_RATIO = 1.43` é **derivada analiticamente** e o teste confere contra a teoria. `DISC_EDGE_RATIO_MAX = 0.5` é `[A]`. `K_MIN/MAX/ABSOLUTE_MAX` são `[A]`, documentados |
| Exceção que engole? | Não. `grep` em `src/`: zero em código |
| Pixel sem resolução? | **Fechado nesta etapa** para a calibração: `test_resolucao_de_trabalho_nao_desloca_o_k` trava a conversão nas duas direções |
| Proveniência pode mentir? | `BokehMeRenderer.provenance()` traz commit + sha256 dos dois checkpoints + config inteira. Não é declarativo |
| Normalizado por imagem/rota/fonte? | Não |
| Nome legado? | Não |

Varredura de nomes indefinidos nos 8 módulos: **limpa**.

## 11. Próximo

1. **Rodar os três testes do `renderer-verifier` contra o BokehMe de verdade**, no
   cluster. É a única parte que não dá para verificar aqui, e é ela que autoriza o
   renderer como rótulo final.
2. `src/routes/route_c.py` — espelho + `metadata/`, Eq. 5, gates.
3. `src/dataio/` — writer, split por cena, proveniência.
4. `src/routes/route_b.py` — duas versões, `main_adapter` condicional.
5. `src/routes/route_a.py` — fonte corrigida, N sorteios por imagem.

---

# Etapa 3 — renderer verificado no cluster

**2026-09-10.** Job **32212** na `h100n3`, `COMPLETED`, exit 0.

## 12. O que rodou

Um job SLURM, `--gres=gpu:1 --time=20-00:00:00`, **ao lado** do treino da DeblurNet
(32186). Nada foi cancelado: a QOS `onejob` permite 2 jobs rodando, e os 7
`deblur-cont*` estavam `PD (Dependency)`, sem ocupar vaga.

Resultado: **os três testes do `renderer-verifier` passaram contra o BokehMe real.**
`is_final_label_renderer` deixou de ser declaração e virou medição. Ver `ACHADOS.md`
para os números e `output/renderer_verification.json` para o relatório completo.

## 13. Seis defeitos encontrados no caminho — cinco no meu próprio código

Nenhum destes teria aparecido sem rodar de verdade.

### 13.1 `from demo import pipeline` era errado

O `demo.py` do BokehMe roda `args = parser.parse_args()` **no nível de módulo**
(linha 130) e em seguida instancia os modelos, carrega os checkpoints e executa o demo
inteiro. Importá-lo parsearia o nosso `argv`, duplicaria os modelos na GPU e
escreveria em `outputs/`.

**Correção:** extrair `pipeline` e `gaussian_blur` do fonte por AST e executar só essas
duas definições, num namespace controlado. O sha256 do trecho extraído entra na
proveniência — prova melhor que o commit, porque identifica o corpo exato que rodou.

Peguei isto **inspecionando antes de submeter**, não depois de queimar GPU.

### 13.2 `cv2` quebra o container

`ImportError: GLIBC_2.38 not found`. O `~/.local` tem o opencv completo, que puxa
libGL exigindo GLIBC 2.38; o container tem 2.35. Está documentado no
`INSTRUCOES_H100.md` e eu pisei mesmo assim.

**Correção:** não importar. O `pipeline` só toca `cv2` dentro dos ramos
`if args.save_intermediate:`, que nunca executam. Uma sentinela `_Unavailable` explode
com mensagem clara se alguém passar a usar esse caminho — melhor que um `None` que
viraria `AttributeError` obscuro.

### 13.3 Minha mensagem de erro mascarava a causa raiz

O `ImportError` "não consegui importar o BokehMe" engolia
`ModuleNotFoundError: No module named 'cupy'`. Um erro sem fallback que esconde o
motivo não é melhor que um fallback silencioso — só falha mais alto.

**Correção:** a mensagem carrega `type(exc).__name__: exc` e o comando de instalação.

### 13.4 A versão do cupy importa por dois motivos independentes

| versão | problema |
|---|---|
| 14.x | compilado contra numpy 2; o container tem 1.26.4 |
| 13.x | removeu `cupy.cuda.compile_with_cache`, que o `scatter.py` usa |
| **12.3** | funciona |

E `--no-deps` é obrigatório: sem ele o cupy arrasta numpy 2.2.6 para o `.pydeps` e
sombreia o do container. Eu instalei sem `--no-deps` na primeira tentativa e criei
exatamente o desvio de ambiente que os meus próprios agentes existem para pegar.

### 13.5 `torch.load` com `weights_only=True`

Default desde torch 2.6. Os checkpoints do BokehMe carregam objetos numpy.

**Correção:** `weights_only=False`, com justificativa no código — é seguro **aqui e só
aqui** porque a origem é o repositório oficial clonado por git e o sha256 de cada
arquivo entra na proveniência.

### 13.6 O patch do `cupy.int`

3 ocorrências de `cupy.int(x)`, alias do `int` builtin removido no cupy 12. Salvo como
`third_party/bokehme_cupy12_int_alias.patch`, versionado e revisável, com o sha256 do
`scatter.py` pós-patch na proveniência. Não é edição solta.

## 14. E um erro meu de método, que é o mais instrutivo

O primeiro veredito do teste 2 foi **"SATURA (reprovou — teto de kernel?)"**. Estava
**errado**, e errado de um jeito que teria feito descartar um renderer correto.

Eu julgava cada K por **erro relativo ponto a ponto**. Os números eram:

```
K=  8   esperado  3,20 px   medido  4,10 px   erro 28,2%
K= 16   esperado  6,40 px   medido  7,18 px   erro 12,2%
K= 32   esperado 12,80 px   medido 13,34 px   erro  4,2%
K= 64   esperado 25,60 px   medido 25,65 px   erro  0,2%
```

Erro caindo com K, razões ao dobrar K subindo (1,75 · 1,86 · 1,92). Eu li como
saturação. **É o oposto:** saturação faria o resíduo CRESCER com K e a razão CAIR.
Um ajuste linear mostrou `medido = 0,962·esperado + 1,02`, com **resíduo máximo de
0,003 px** — uma reta praticamente exata. O que existe é um **piso aditivo de ~1 px**
do medidor (fonte pontual discretizada), que estoura o erro relativo em raio pequeno.

**Correção de método:** linearidade vive na **inclinação e no resíduo**, não no erro
ponto a ponto. O harness agora ajusta uma reta e reporta:

- `slope` — a escala efetiva de K
- `intercept_px` — o piso do medidor
- `relative_residual` — resíduo sobre o maior raio testado

E o `RadiusResponse` documenta as três assinaturas para não confundir de novo: offset
aditivo (resíduo ~0, slope ~1, intercept > 0), saturação (resíduo cresce com K) e erro
de escala (resíduo ~0, intercept ~0, slope ≠ 1).

Rodar em três saídas separou definitivamente: `bokeh_classical` dá resíduo
**0,0000 px**. O renderer clássico é uma reta exata.

## 15. `k_effective_factor = 0,9873`

O raio renderizado é **1,3% menor** que o K nominal prediz (3,8% no clássico puro).
Pequeno, sistemático, gravado.

Para a **rota C** não importa: K é ajustado por SSIM contra o alvo real, e o ajuste
absorve a escala do renderer por construção. Para a **rota B**, cujo K é analítico pela
Eq. 3, é um desvio de ~1% entre as rotas. Fica registrado em `ACHADOS.md` e no relatório;
corrigir só depois de medir com cena real, não com ponto sintético.

## 16. Auditoria atualizada

| status | etapa 2 | agora |
|---|---|---|
| fechados com teste | 13 | **14** |
| fechados no contrato, dependem das rotas | 3 | 3 |
| abertos | 17 | **16** |

**D4 fecha de vez.** Antes estava fechado "com teste sintético"; agora está fechado com
**medição contra o renderer real, em GPU**, e com proveniência criptográfica de qual
renderer produziu o número.

`[A]` fechados nesta etapa: os sete parâmetros do BokehMe estão congelados em
`BokehMeConfig` e gravados; a decisão de `highlight` está registrada (inerte no nosso
caminho, e agora sabemos *por quê*).

## 17. Próximo

`src/routes/route_c.py`. Todos os bloqueadores caíram: o renderer está verificado, o
espelho `akcit-pixel/RealBokeh` está confirmado byte a byte como AIF f/22, e o
`metadata/<id>.json` dá o f-number de cada `level` para o K de validação.

---

# Etapa 4 — `dataio`, gates, e a revisão por quatro agentes

**2026-09-10.** 96 testes passando. Quatro revisores independentes rodaram sobre o
código novo; **os quatro acharam defeito real**, e dois convergiram no mesmo
independentemente.

## 18. O que foi construído

`src/dataio/` (encoding, sample, writer, split) e `src/qc/gates.py`.

**Decisão de armazenamento, tomada com número.** Guardar as imagens de origem de novo
custaria **365 GB** contra 115 GB de cota livre. Regra: *guardamos o que geramos,
referenciamos o que já existe.* Rota C referencia as duas imagens (são reais e já estão
no HF), rota B grava a AIF (é gerada), rota A grava o bokeh (é renderizado).
Profundidade em uint16 **linear em disparidade**, lado longo 768: erro de CoC medido
**0,000378 px**. Rotas B e C somam 31 GB e cabem com folga.

## 19. Os oito defeitos que os revisores acharam — e o que cada um custaria

### 19.1 `max_coc` livre por amostra — o kfix reconstruído (CRÍTICO)

O revisor **mediu**: uma amostra com `max_coc = 10.510746`, o valor exato do
experimento `kfix`, atravessava o writer e aterrissava no disco sem uma linha de erro.

É o D1 com granularidade de amostra, e é o mecanismo exato do desastre do `kfix`: rota
B numa convenção, rota C noutra, no mesmo lote — LF-Bokeh subiu para +0,8288 enquanto
RealBokeh caía para +0,4832 e RealDOF para **−0,4599**.

**Fix:** `max_coc` deixou de ser campo de `ControlLabel` e deixou de ser parâmetro de
`defocus_map`. Vem de `MAX_COC` e de lugar nenhum mais. `validate_metadata` rejeita
qualquer outro valor. Provado: `10.510746` → `SampleRejected`.

### 19.2 Censura falsa na cauda inferior da rota C (CRÍTICO)

O revisor **mediu** com renderer de disco perfeito: `K = 3,6 · 5 · 8 · 10` eram
recuperados exatamente, com SSIM **1,0000**, e saíam marcados `is_k_censored=True`.

A causa: o teste de borda usava `max(step*0.5, tolerance)`, onde `step` é o passo do
**grid grosso** (19,92) e não a resolução da seção áurea (0,25) — limiar efetivo
`k ≤ 10,46`. A âncora da rota C é **3,6 a 36**, então a cauda inferior INTEIRA sairia
da loss de controle. É o `k == 300` invertido: grava medida como censura.

**Fix:** a escala da borda é `tolerance`, e o teto usa `best == len(grid)-1`, que já
significa "o ótimo ainda crescia no teto duro". Provado: 3,6 a 36 todos
`censurado=False`.

### 19.3 O limiar de SSIM da Eq. 5 não existia

O paper **afirma executar este passo, duas vezes**: §3.2(c) (*"provided that its
corresponding SSIM exceeds a predefined threshold"*) e supp. B.2. O código gravava
`calibration_ssim` e **nenhum gate o lia** — era o defeito P1-C5 do dataset antigo,
reconstruído. **Fix:** `calibration_ssim_is_reliable`, com `threshold=None`.

### 19.4 "O paper treina em resolução nativa com tiling (§3.5)" — a frase é falsa

Eu escrevi isso em `contract.py` e no `REGISTRO.md`. A linha 482 do paper diz
**`during inference`**. O paper **não publica a resolução de treino**.

Foi exatamente a construção que o meu próprio agente `paper-fidelity` proíbe: "o paper
implica" escrito com número de seção. **Fix:** frase corrigida, e as três decisões de
resolução — 768 no armazenamento, 512 no crop de treino, 512 na calibração — agora
estão declaradas como **decisão nossa** no `CONTRATO.md`, com o custo de cada uma.

### 19.5 `source_hw` tinha nome de uma coisa e valor de outra

Guardava o shape do **array de profundidade**, não o da imagem. Se o Depth Pro roda a
1152×1536 sobre uma foto 3024×4032, `k_at_resolution` calculava o fator errado em
silêncio — medido: **2,63×**, mapa da rota B em [0, 0.13] em vez de [0, 0.05].

**Fix:** `image_hw` obrigatório e explícito. **Dois revisores acharam este
independentemente**, o de unidades e o de contrato de dados.

### 19.6 O span de disparidade vinha da profundidade reduzida

O vizinho mais próximo com decimação 5× amostra 1 pixel em 27, então objeto pequeno e
muito perto some — e com ele o extremo. Medido: objeto de 3×3 px a 0,30 m dá
`disp_max = 3,3333` na resolução cheia e **1,0000** na reduzida, e `k_for_bokehme` sai
**3,4× menor**. Uma ordem de grandeza acima do `k_effective_factor = 0,9873` que já
declaramos. E o efeito **depende do conteúdo** — some com objeto grande —, que é como
passaria despercebido.

**Fix:** extremos medidos na profundidade cheia.

### 19.7 O gate de vazamento era tautológico

`check_no_leak` fazia `scene_splits[scene].add(split.of(scene))` e depois procurava
cenas com mais de um split. `split.of()` é função pura sobre um `dict[str, str]`, então
**`len(v) > 1` era impossível por construção**: o ramo de detecção era código morto, e
o teste chamado `test_detecta_vazamento_por_cena` afirmava `clean is True`.

Um gate que não pode reprovar é pior que gate nenhum: dá garantia falsa.

**Fix:** cada linha do manifesto carrega o `split` com que foi gravada, e o gate compara
contra o split materializado. Dois testes novos provam que agora ele **reprova**.

### 19.8 Cinco fallbacks menores

`depth_backend` caía entre o contrato e o disco — o campo que **prova** que o D11 não
disparou era opcional e sumia por omissão. `split_from_source` mandava split ausente
para **treino** em silêncio, que é o lado que infla a métrica. `mask_iou` devolvia
**1,0** para duas máscaras vazias — o pior caso possível com nota máxima.
`focus_depth_plausible` calculava o veredito e o jogava numa string, com
`threshold=None`. `verify_renderer.py` tinha um teste que **nunca podia reprovar** e
alimentava o veredito de `is_final_label_renderer`.

Mais: `_reject` usava `assert`, que `python -O` desliga; `encode_depth` levantava
`ValueError` sem slug, escapando do histograma; colisão de `sample_id` inflava a
contagem em silêncio; a inversão BGR→RGB era cega e não declarada.

## 20. Auditoria atualizada

| status | etapa 3 | agora |
|---|---|---|
| fechados com teste | 14 | **16** |
| fechados no contrato, dependem das rotas | 3 | 3 |
| abertos | 16 | **14** |

## 21. O que isto ensina sobre o método

Os quatro revisores acharam **oito defeitos reais em 2.500 linhas que eu já tinha
revisado**, e dois deles eram do mesmo tipo que o projeto inteiro existe para impedir:
um normalizador por amostra e uma quantidade em pixel viajando sem resolução.

Três padrões que se repetiram:

- **gate que não pode reprovar** — `check_no_leak`, `highlight_decidido`,
  `focus_depth_plausible`. Todos "passavam" 100% porque o caminho de falha não existia.
- **campo com nome de uma coisa e valor de outra** — `source_hw`.
- **default de assinatura carregando grandeza física** — `max_coc`, `backend`,
  `is_k_censored`, `min_f_number`.

O `fallback-hunter` já listava os dois últimos padrões. Escrevê-los num prompt não me
impediu de cometê-los; só permitiu achá-los antes de gastar GPU.

---

# Etapa 5 — fonte RealBokeh, pixels, entrypoint e jobs

## O que passou a existir

| arquivo | o que faz |
|---|---|
| `src/sources/realbokeh.py` | enumera os 20.495 pares do espelho: qual cena, qual nível, qual f-number, os três termos da Eq. 3 |
| `src/sources/mirror_images.py` | os pixels: índice `file_name_base → (shard, linha)`, leitura quase sequencial, decodificação PIL → BGR |
| `scripts/run_route_c.py` | entrypoint: laudo do renderer → modelos → fonte → split → `run_route_c` |
| `scripts/calibrate_thresholds.py` | lê o piloto e **propõe** os dez limiares; não aplica nenhum |
| `slurm/route_c_pilot.slurm` | smoke de 2 → piloto de 200 → proposta de limiares |
| `slurm/route_c_full.slurm` | run completo; **recusa começar** com `THRESHOLDS` vazio |

Suíte: **208 testes**, `OK (skipped=6)`. Os skips são o bloco que toca o HF de verdade
(3) e o bloco que exige `pyarrow`, ausente nesta máquina (3) — os dois rodam no cluster.

## Decisões desta etapa

**Os seis slugs `source_*` foram para `contract.SOURCE_REJECTION_REASONS`.** O agente os
tinha deixado num `PENDING_REJECTION_REASONS` local, para não editar o contrato por
conta própria. Vocabulário fechado em dois lugares não é fechado: um slug que só
existisse no adaptador sumiria do histograma agregado entre rotas, que é justamente o
instrumento que denuncia fallback novo. `sources.realbokeh._reject` agora só delega, e
`test_slugs_da_fonte_estao_no_vocabulario_do_contrato` prova a inclusão nos dois níveis.

**Um sétimo slug nasceu: `source_image_unreadable`.** Cobre bytes que não decodificam,
célula vazia, e — o caso que importa — imagem numa resolução diferente da que o espelho
declara. K vive em pixel: uma imagem que chega em outra resolução muda o significado do
rótulo sem mudar nada no JSON.

**A amostragem do piloto sorteia CENAS, não pares.** Os shards do espelho estão
agrupados por cena, então os 200 primeiros pares da ordem de disco saem de uma dúzia de
cenas — e os limiares sairiam calibrados para aquela dúzia. `sample_pairs_for_pilot`
sorteia cenas inteiras com seed; `--limit` continua existindo como teto cru, documentado
como tal. São duas coisas diferentes e agora têm dois nomes.

**Leitura reordenada por `(shard, linha)`.** Parquet lê por row group; pedir linhas
salteadas relê o mesmo grupo várias vezes. Reordenar é seguro porque nada na rota C
depende da ordem: `sample_id` é determinístico, o split é por cena e materializado, a
retomada é por id.

**O split do espelho: sorteado, e dito em voz alta.** Medido: o espelho publica **só**
`train` (20.495/20.495). `split_from_source` sozinho produziria `val_fraction == 0` —
tecnicamente válido, e a maneira mais silenciosa possível de não ter validação. O
entrypoint usa a origem quando ela publica os dois lados e sorteia por cena quando não
publica, imprimindo qual dos dois aconteceu e gravando `split_origin` na proveniência.

**A largura do sensor da RealBokeh continua `[A]`, e agora isso tem consequência
visível.** Sem ela a Eq. 3 não fecha e `_analytic_k` devolve `None`: o rótulo (Eq. 5)
não muda, mas a amostra fica sem validador independente. `--sensor-width-mm` é opcional
e, quando passado, entra na proveniência marcado como assumido. O que não existe é um
36,0 cravado dentro do laço.

## Erros meus corrigidos no caminho

1. `renderer_report_sha256` estava recebendo `len(texto)` — um número com nome de hash.
   Agora é sha256 de verdade.
2. `MirrorIndex.build` importava `pyarrow` antes de checar o cache, então reler um
   índice pronto exigia o stack de parquet. O import foi para depois do cache.
3. O primeiro rascunho do entrypoint inventou uma API (`load_pair_images`,
   `scene_splits`) que o adaptador não tem. Reescrito contra as assinaturas reais.

## Aberto

- **LFDOF**: `--source lfdof` está no parser e falha com mensagem explícita. O adaptador
  não exige tocar em `routes/route_c.py` — a rota recebe pares por protocolo.
- **Rotas A e B**: não começaram.
- **O piloto ainda não rodou**: nenhum dos dez limiares tem número.

---

# Etapa 6 — o release: o que se grava e o que se publica

Duas lacunas reais apareceram quando fui responder "está tudo pronto para rodar?".

## Lacuna 1 — a rota C não gravava pixel nenhum

`FileSampleWriter` grava `generated_images`, ou seja, **só o que a rota gera**. Na rota B
isso é a AIF produzida pela DeblurNet; na rota A é a bokeh renderizada. Na **rota C não
há imagem gerada**: a AIF e a bokeh vêm prontas da origem. O release sairia como rótulo
solto apontando para um espelho privado, sem nada que provasse contra quais bytes o K
foi calibrado.

Conserto, em dois níveis:

1. **Sempre**: `MirrorImageLoader` grava `source_images.jsonl` — sha256 da AIF e da
   bokeh, tamanho em bytes, shard e linha, por amostra. O join com o espelho passa a ser
   verificável byte a byte, e o `publish_release.py` **reprova** release da rota C que
   tenha amostra sem essa linha.
2. **Opcional**: `--store-source-images` copia os **bytes originais** para `<out>/source/`,
   tornando o release autocontido. Copia, não recomprime — recomprimir falsificaria a
   evidência, porque o sha256 do ledger deixaria de bater com o arquivo ao lado dele.
   Custo medido por estimativa: ~49 GB na RealBokeh inteira, contra ~31 GB sem.

## Lacuna 2 — não havia caminho para o Hugging Face

`scripts/publish_release.py`: valida, escreve o card, e só sobe com `--yes`. Privado por
default, e recusa publicar por cima de repositório que já tem arquivos sem
`--allow-existing`.

Onze checagens, **cada uma com um teste que a faz reprovar** — a lição do `check_no_leak`
antigo, cujo teste chamado "detecta vazamento" afirmava `clean is True` e portanto não
testava nada:

| reprova quando | por quê |
|---|---|
| `sample_id` repetido | a contagem publicada infla sem que nada denuncie |
| falta `depth/`, `mask/` ou `meta/` | amostra listada e ausente |
| `validate_metadata` falha | pega `max_coc` fora de 100 — o normalizador escondido |
| `control_version` divergente | duas convenções no mesmo release |
| `depth_backend` divergente | métrica e disparidade normalizada não são a mesma coisa |
| amostra da rota C sem sha256 de origem | release não prova contra o que calibrou |
| cena fora do `split.json` | fronteira do split incompleta |
| vazamento de split | cena nos dois lados |
| K censurado acima de 20% | eram 47,0% no release anterior |
| sem `split.json` | split em config diverge entre runs e some no rsync |
| manifesto vazio | não há release |

Avisos que **não** reprovam, mas aparecem: release não autocontido, `val` vazio, e
censura abaixo de 20%.

O card (`README.md` do dataset) é gerado a partir do próprio release e diz o contrato do
sinal, os números em cenas **e** em amostras, a proveniência com os hashes, e uma seção
"o que este dataset NÃO é" — sem revisão humana, com `rejections.jsonl` incluído no
release, e com a licença dos pixels seguindo a origem.

Suíte: **227 testes**, `OK (skipped=6)`.

## Pendência anotada

`Image.fromarray(..., mode="I;16")` está deprecado e sai no Pillow 13 (out/2026). Vale
trocar antes disso; hoje funciona e o release não corre risco.

---

# Etapa 7 — o defeito que a conferência do cluster achou

Fui checar se o cluster estava pronto para rodar e encontrei, no snapshot do espelho,
**96 shards parquet**, não 85: `train` (85), `test` (6) e `validation` (5). A medição
anterior dizia "o espelho publica só `train`" porque o dump de coluna que a produziu
cobria só os shards de `train`.

Ao contar as cenas, o problema real apareceu: **a numeração de cena reinicia em cada
split**. As 220 cenas de `test` usam os números 1..220, que `train` também usa, e são
cenas físicas diferentes. Ver `reference/ACHADOS.md` para a tabela medida.

Isso teria custado, ao processar os três splits:

- **2.495 amostras descartadas** pelo gate `source_duplicate_sample`, com motivo
  registrado no histograma e conclusão errada — pareceria dado sujo na origem.
- E o modo silencioso, pior: metadata da cena `1` de `train` aplicada à cena `1` de
  `test`, com JSON válido e completo. Distância de foco errada alimentando a Eq. 3, sem
  nada denunciar.

Conserto: `scene_key(split, numero)` como única definição da chave de cena,
`scene_number` preservado para achar o arquivo, e `load_scene_metadata` lendo
`<raw_root>/<split>/metadata/` dos três splits. Dois testes novos provam a distinção;
a guarda antiga contra "cena em dois splits" virou defesa em profundidade e ganhou um
teste que a exercita por outro caminho.

**Efeito bom:** com os três splits, o split do release sai **da origem**, como ela
pretendeu — não é mais preciso sortear validação. A ressalva da etapa 5 sobre `val`
vazio deixa de valer para a RealBokeh.

Suíte: **233 testes**, `OK (skipped=6)`.

## Estado do cluster, conferido (nada foi tocado)

| item | estado |
|---|---|
| `/raid` livre | **7,0 TB** — o release autocontido cabe com folga |
| espelho `akcit-pixel/RealBokeh` | **presente**, 96 shards, 44 GB |
| `depth_pro.pt` | presente em `vision-pipeline/checkpoints/` |
| laudo do renderer | presente |
| **BiRefNet** | **ausente** — precisa baixar |
| **`RealBokeh_3MP` bruto (metadata)** | **ausente** — precisa baixar |
| código novo no cluster | **ausente** — precisa de rsync |
| jobs rodando | 32186 (`deblur-n2-4gpu`) + 7 na fila — **intocados** |

---

# Etapa 8 — auditorias das rotas A e B

Relatórios completos em `reference/ROTA_A_AUDITORIA.md` e `reference/ROTA_B_AUDITORIA.md`.
Aqui só o que muda decisão.

## Rota A — a fonte trocada está **confirmada**

Generative Photography `[80]` + EBB! `[27]` (`paper.txt:996`, `527-528`). DiffCamera é
`[69]` e aparece só como baseline. Armadilha registrada: o `paper.txt` tem **duas
numerações** — legendas de figura usam a antiga (`Restormer [80]`), só corpo e
bibliografia valem.

- **~41 amostras por imagem** é `[I]`, não `[M]`: 70K pares ÷ 1,7K imagens. O paper não
  publica o número; o "~40" do plano se sustenta como aritmética.
- **A rota A não tem equação para K**: *"randomly sample a focus plane and a target
  bokeh level K"* (`paper.txt:329-330`). Logo **não tem máscara, não tem BiRefNet e não
  tem SSIM** — Eq. 3, 4 e 5 não valem ali.
- BokehMe `[43]` é autorizado de forma mais direta que na rota C (`paper.txt:331`, `292`).
- **Dependência dura**: o amostrador de K lê a distribuição de B e C, que hoje são
  `k=50` constante e 47% no teto. **A rota A não pode rodar antes de B e C regeradas.**

## Rota B — um defeito crítico NOVO, que não estava em nenhuma lista

**B1: a DeblurNet rodava com a convenção de adapter errada.** `build_deblurnet_fn` chama
`generate(...)` sem `main_adapter`, cujo default é `None` = cond-only. Mas a rota B exige
o **nosso** checkpoint, que é a variante **main+cond** — e sem `main_adapter="deblurring"`
ele sai **lavado**. Isso contamina a AIF, e a AIF é a entrada do Depth Pro, do BiRefNet,
do `D_focus` e do K. Nenhum gate pegava: `--min-aif-laplacian-variance` tinha default 0,0.

Confirmados: o fator 1000×, o `pixel_ratio` com largura (refutado pelo paper na legenda
da Fig. 16, `paper.txt:1186-1187`), o `import json` ausente na `route_b.py` (e a linha
`:298` está num `finally` — morre na amostra 0), e o `max_coc` como flag de CLI
independente por rota, que é **o mecanismo estrutural do kfix**, não acidente.

**Quatro `[A]` foram fechados contra o paper**: `pixel_ratio` é o maior lado; a Eq. 3
vale só na rota B; **nenhum renderizador produz o alvo da rota B** — o alvo é a foto
real; e a distância de foco **não** vem da EXIF, por decisão explícita do paper
(`paper.txt:344-347`).

**Checagem dimensional do nosso `contract.py`: fecha.** `sensor_width_mm` → mm;
`pixel_ratio` → px/mm; `k_eq3_mm` → px·mm; `k_official` → px·m; `signed_coc_px` → px.

### A ambiguidade que a rota B expõe — e que a rota C não tem

`k_eq3_mm` pede `focus_depth_m`, mas o contrato produz `focus_disparity = median(1/z)`,
e `1/median(1/z) ≠ median(z)`: a mediana não comuta com a inversão. Na rota B há uma
escolha real a fazer.

**Na rota C não há.** A origem publica `focus_plane_distance` medido na cena, e é ele que
entra em `_analytic_k`. É isso que torna o validador independente: alimentá-lo com
`1/focus_disparity` o faria depender do Depth Pro e da máscara — os mesmos dois modelos
que produzem o valor sendo validado. Documentado na docstring para ninguém "harmonizar"
as duas rotas depois.

### As duas versões da DeblurNet

A tripla (repo, arquivo, adapter) é **indivisível**:

| | repo | arquivo | adapter |
|---|---|---|---|
| nossa | `juliadollis/genrefocus-deblurnet-paper-4gpu` | `deblur.safetensors` | `main_adapter="deblurring"` |
| oficial | `nycu-cplab/Genfocus-Model` | `deblurNet.safetensors` | `main_adapter=None` |

Vira enum fechada no molde de `MaskSource` — nunca um `--main-adapter` livre, nunca
default no wrapper. Um release e um repo HF por variante, com **`sample_id` igual nas
duas** para poderem ser comparadas par a par. O `publish_release.py` vai precisar
reprovar release com `deblur_variant` misto, como já reprova `control_version` misto.

## Consertos aplicados nos meus módulos, vindos das auditorias

1. **`manifest.jsonl` não carregava a resolução** (achado da rota A). K é um número em
   pixel; sem `image_h/image_w` na mesma linha, o manifesto não diz o que o K significa.
   Corrigido no `writer.py`, com teste.
2. **`_sha256_dir` hasheava o cache do HF** (achado do download do BiRefNet). O
   `snapshot_download` deixa `.cache/huggingface/download/*.metadata` dentro do
   diretório do modelo, com etag e horário — 15 arquivos, medidos. Dois snapshots byte a
   byte idênticos dariam `mask_model_sha256` diferentes, e a proveniência diria "modelo
   diferente" sem diferença nenhuma. Agora ignora caminhos ocultos e hasheia o caminho
   relativo, não só o nome. Cinco testes novos.

Suíte: **239 testes**, `OK (skipped=6)`.

---

# Etapa 9 — o refinamento da região em foco, plugado e validável

A etapa 8 fechou as auditorias das rotas A e B. Entre ela e esta, o piloto da rota C
mediu o número que mudou o método: **a máscara do BiRefNet acerta o plano de foco em
35,2% das amostras** (57 de 162), contra a distância de foco que a RealBokeh publica
**medida** com incerteza mediana de ±0,010 m; e **20,6%** das amostras eram descartadas
com `focus_mask_empty` porque o BiRefNet devolvia probabilidade exatamente 0. A medição
inteira está em `reference/MEDICAO_PLANO_FOCO.md`; `src/qc/focus_region.py` é a resposta.

O §3.2(c) do paper **antecipa** este problema, nestes mesmos dois datasets, e rejeita
explicitamente a estratégia de descartar: *"Rather than simply verifying and discarding
unreliable cases, we introduce a manual refinement step [...] preserves challenging
samples rather than excluding them"*. Nós descartávamos onde o paper corrigia. Esta etapa
liga o substituto automático daquele passo, marca cada amostra com a origem da região, e
constrói a prova contra gabarito.

## 1. O que passou a existir

| arquivo | o que mudou |
|---|---|
| `src/routes/route_c.py` | `refine_focus_region()`, os botões do refinamento, `RouteCStats.focus_summary()`, e `build_gate_report` sabendo qual gate julga qual máscara |
| `src/dataio/sample.py` | `FocusRegionRecord`, `MaskSource.BIREFNET_REFINED`/`RETENTION_ONLY`, `FOCUS_SOURCE_TO_MASK_SOURCE`, 8 campos obrigatórios novos, `_validate_focus_region` |
| `src/dataio/writer.py` | 7 campos novos na linha do manifesto |
| `src/qc/gates.py` | gate novo `focus_region_retention`; dois gates que reprovavam sem ter medido |
| `src/control/contract.py` | slugs `gate_focus_region_retention_low` e `focus_provenance_inconsistent` |
| `src/renderer/calibration.py` | `resize_area_for_photo` / `resize_nearest` públicos — uma implementação só de redimensionamento |
| `scripts/validate_focus_refinement.py` | **novo** — a prova contra gabarito |
| `scripts/run_route_c.py`, `scripts/calibrate_thresholds.py`, `scripts/publish_release.py` | botões, proposta de limiar do gate novo, e a composição do lote no card |

`src/qc/focus_region.py` **não foi tocado** — nem uma linha de lógica. Ele já estava certo.

## 2. Decisões desta etapa

### A máscara gravada é a REFINADA, e isso não era opcional

`mask/<id>.png` guarda a região que produziu `focus_disparity`, e `mask_source` diz qual
das três origens ela tem. A alternativa — gravar a máscara crua do BiRefNet — foi
recusada porque o arquivo passaria a ser uma máscara que **não corresponde ao rótulo**, e
quem auditasse `focus_disparity` olharia a máscara errada sem nada denunciar. O writer já
chamava aquele arquivo de "máscara **final** de foco"; agora ele é.

A máscara crua não se perde como medida: `focus_agreement`, `focus_initial_mask_area_ratio`
e `focus_initial_mask_was_empty` a descrevem, e a IoU AIF/bokeh continua sendo calculada
sobre ela. `validate_metadata` **reprova** metadado em que `mask_source` e `focus_source`
discordem — o caso `retention_only` + `birefnet` é rejeição com slug, não aviso.

### Os oito campos obrigatórios, e por que a resolução entra junto

`focus_source`, `focus_was_refined`, `focus_agreement`, `focus_retention_in_region`,
`focus_region_area_ratio` — os cinco que o requisito pedia — mais
`focus_retention_h`, `focus_retention_w` e `focus_retention_window_px`. Os três últimos
entram pela regra dura: a janela é quantidade em pixel, e a 1500x2000 uma janela de 33 px
cobre 1,7% do lado longo enquanto a 512x683 cobre 6,4%. São escalas físicas diferentes, e
o número solto não distingue as duas.

Emitidos como campos de **primeiro nível** e espelhados no manifesto: filtrar 22.990
amostras por refinamento não pode exigir abrir 22.990 JSONs. Mais dois campos gravados e
não obrigatórios: `focus_initial_mask_area_ratio` e — o que faz o script de validação
funcionar — `focus_disparity_from_initial_mask`, o rótulo que a amostra teria tido **sem**
refinamento. É diagnóstico pareado, nunca rótulo.

### A resolução de trabalho foi medida e REJEITADA

A instrução era medir e, se preciso, calcular a retenção numa resolução reduzida
declarando-a. Medi (`reference/ACHADOS.md`): a 1500x2000, `detail_retention` custa 291 ms
e cai para 22 ms a 384x512 — 13x. Mas reduzir as **duas** fotos custa 185 ms cada, e
ponta a ponta o caminho reduzido sai em **433 ms contra 391 ms** do caminho cheio. Então
o default é resolução cheia, o botão fica (origem de maior resolução; desalinhamento
anotado em 421 pares), e a grade usada vai para o metadado nos dois casos. Um teste
pergunta pela medição se alguém trocar o default de volta por intuição.

A diferença em relação ao sweep da Eq. 5, que usa 512: o sweep reduz uma vez e reusa a
redução em 14 a 18 renderizações. A retenção é calculada uma vez só.

### O gate que faltava, e os dois que atrapalhavam

Gate novo `focus_region_retention` — retenção mediana dentro da região final. É o único
que julga a região refinada com a grandeza certa: `focus_mask_is_sharpest` mede nitidez
**absoluta**, e uma parede lisa legitimamente em foco reprovaria enquanto uma folhagem
desfocada passaria. É exatamente a armadilha que `focus_region.py` existe para não cair, e
ela estava a um limiar congelado de distância. Default `None`, como todo gate do módulo.

E dois defeitos achados ao plugar, os dois com o mesmo mecanismo — `GateResult.passed`
reprova valor não-finito **antes** de olhar o limiar, então um gate que não conseguiu
medir reprovava a amostra com um slug que afirmava mérito:

1. **`mask_iou_aif_bokeh`**: BiRefNet vazio nas duas imagens → união 0 → IoU NaN →
   rejeição com `gate_mask_iou`. Isto rejeitaria **exatamente as amostras que o
   refinamento acabou de salvar** — os mesmos 20,6%, com outro slug. Enquanto a máscara
   vazia rejeitava antes dos gates, o ramo era inalcançável: o defeito estava latente e
   apareceu no instante em que a amostra passou a atravessar.
2. **`focus_mask_sharpness_ratio`**: região de retenção pontilhada desaparece após 3
   passos de erosão → NaN → rejeição com `gate_focus_mask_not_sharpest`, um slug
   afirmando que a máscara está no lugar errado sem ter medido nitidez nenhuma.

Os dois agora devolvem `applicable=False` — "não medido" gravado como não medido, sem se
disfarçar de "medido e passou" nem de "medido e reprovado". Cada um tem teste que dispara
o ramo.

### O dublê de renderer dos testes estava medindo a coisa errada

`test_renderer._DISC` borra a imagem **inteira** com `median(coc)`. Numa cena
uniformemente borrada não existe plano de foco, a retenção é igual em todo lugar, e a
região devolvida é o topo de um empate numérico: com ele, a amostra de teste saía
`retention_only` com retenção 0,21 e região fora da máscara. `test_route_c._LAYERED`
compõe 7 camadas em `|K·Δdisp|`; com ele a amostra cuja máscara casa com o objeto a 3 m
sai `focus_source=birefnet`, acordo 1,000, e o sweep recupera K = 18,011 com SSIM 0,9999.
Os testes da rota C ficaram mais fiéis ao BokehMe real, que é espacialmente variante.

## 3. `scripts/validate_focus_refinement.py` — a prova

Compara `focus_disparity` contra `1/focus_plane_distance_m` e reporta **três blocos**:

1. **Por `focus_source`** — ±10%, ±25%, ±50%, razão p05/mediana/p95, em amostras **e em
   cenas**. Impresso com o aviso de que a comparação é confundida: o grupo `birefnet` é,
   por construção, o subgrupo em que o segmentador já concordava com a física.
2. **Pareado, na mesma amostra** — usando `focus_disparity_from_initial_mask`. É este
   bloco que responde à pergunta, e é dele que sai o veredito. As amostras de máscara
   vazia entram numa terceira contagem, "recuperadas": elas não têm linha de base porque
   a alternativa a elas não é um rótulo pior, é amostra nenhuma.
3. **De quem é a divergência** — ver abaixo.

O veredito imprime `MELHOROU`, `PIOROU` ou `empatou`, decidido pelo pareado sobre as
refinadas, mais a contagem crua de quantas amostras melhoraram e quantas pioraram — porque
a fração agregada pode subir com poucas correções grandes enquanto a maioria piora um
pouco, e as duas coisas têm que aparecer. Se piorar, o relatório diz para não congelar
nada e não gerar o lote. Há teste para o caso `PIOROU` e teste para o aviso de "mais
amostras pioraram do que melhoraram": um validador que só sabe elogiar não valida nada.

### O viés de escala do Depth Pro — o que dá e o que não dá para separar

Divergência pode vir da profundidade e não da máscara. Com o que existe no release, dois
instrumentos limitam a confusão:

- **Alcançabilidade.** O metadado grava `disparity_min` e `disparity_max` da amostra. Se o
  gabarito cai fora dessa faixa, **nenhuma** máscara o alcançaria — a mediana de um
  subconjunto está sempre dentro do intervalo do conjunto —, e a divergência é da
  profundidade ou do gabarito. Se cai dentro, a escolha de região é responsável.
- **Escala global.** Um viés multiplicativo global desloca todas as razões pelo mesmo
  fator. Dividindo cada razão pela mediana global sobra a **dispersão**, que nenhum fator
  único explica; se o refinamento ganha também depois disso, o ganho não é artefato de
  escala. O relatório diz, na própria saída, que isso força a mediana a 1 e mede dispersão,
  não acerto.
- **Incerteza do gabarito.** Amostras com incerteza relativa acima de 10% são isoladas:
  `1,92 ± 0,12 m` e `1,92 ± 0,00 m` não autorizam a mesma conclusão.

**O que não dá para fazer está escrito no relatório**, e não foi disfarçado: separar as
duas causas *por amostra* exigiria uma profundidade métrica de referência na cena, e a
RealBokeh publica uma distância só — a do plano de foco.

## 4. O que ainda NÃO foi medido

O piloto que produziria os números desta etapa **não rodou**: precisa de GPU, e a execução
no cluster não é minha. O que está pronto é o caminho inteiro e a ferramenta que o julga.
Rodar, na ordem:

```
scripts/run_route_c.py --pilot 200 ...              # agora sem descartar máscara vazia
scripts/validate_focus_refinement.py --pilot-dir ...  --raw-dir ...
scripts/calibrate_thresholds.py --pilot-dir ...
```

A régua é **35,2% dentro de ±25%**. Se o refinamento não bater isso no bloco pareado, não
vale gerar 22.990 amostras com ele — e o parâmetro a revisar primeiro é
`focus_top_fraction`, depois `focus_agreement_floor`, os dois `[A]`.

**Os limiares do piloto anterior não valem mais**: `mask_area_ratio_*` e
`focus_mask_sharpness_ratio` passaram a medir a região refinada, cuja distribuição é outra.
Nenhum está congelado hoje (todos `None`), então nada quebrou — mas quem for congelar tem
que usar um piloto novo.

## 5. Aberto

- **Rota A e rota B não têm `focus_source`.** Os campos são obrigatórios para todas as
  rotas, e hoje só a rota C existe. A rota B usa BiRefNet e cabe em `birefnet`; a rota A
  **sorteia** o plano de foco (`paper.txt:329-330`) e não tem máscara nenhuma, então vai
  precisar de um valor próprio no enum `FocusSource` — extensão de vocabulário, não
  mudança de lógica, e nunca reusando o valor de outra fonte.
- **`focus_top_fraction`, `focus_agreement_floor`, `focus_retention_window_px`** seguem
  `[A]`. O paper não publica nada sobre o passo de refinamento.
- **A retenção assume par alinhado.** A RealBokeh anota `misaligned` em 183 pares e
  `shift_<X>px` em 238. O efeito do desalinhamento na razão bokeh/AIF não foi medido, e é
  a razão mais forte para um dia usar a resolução de trabalho.

Suíte: **523 testes**, `OK (skipped=7)` — eram 446 antes desta etapa.

---

# Etapa 10 — o refinamento tinha um defeito pior, e a auditoria adversarial pegou

A auditoria de `reference/ROTA_C_AUDITORIA_2.md` deu **veredito NÃO** e acertou no ponto
mais grave: o substituto automático do refino manual, escrito na etapa 9, tinha um modo
de falha que o problema original não tinha. Reproduzi tudo antes de aceitar.

## O defeito `[M]`

`detail_retention` satura em 1,0, e `sharpest_region_mask` selecionava com `>= quantil`.
Uma **superfície lisa que borra continua lisa**: `|lap(bokeh)| ≈ |lap(aif)|`, retenção
≈ 1 — exatamente o valor que o plano de foco produz. Numa cena de objeto texturizado em
foco contra céu liso desfocado:

| | medido |
|---|---|
| pixels válidos empatados em retenção 1,000 | **79,7%** |
| região selecionada caindo no céu liso desfocado | **42,9%** |
| `top_fraction` de 0,20 a 0,001 (fator 200) | **área idêntica**, 0,7767 — inerte |
| guarda `min_aif_detail = 1e-3` | céu real dá \|lap\| ≈ 2,5 — **2.500× acima**; nunca disparava |

O parâmetro que o `REGISTRO.md` apontava como "o primeiro a revisar" não podia ser
revisado, e a "região pequena porém confiável" estava engolindo o fundo.

## O conserto

1. **Seleção por posto**, não por `>= quantil`, com **desempate pelo detalhe da AIF**:
   entre pixels igualmente retentivos, ganha aquele onde a evidência é mais forte.
2. **Piso de textura relativo à própria imagem** (`min_detail_percentile`), porque limiar
   absoluto depende de exposição, ISO e conteúdo.

```
top_fraction 0.20 -> área 0.2000   no céu: 0.0%
top_fraction 0.05 -> área 0.0500   no céu: 0.0%
top_fraction 0.001-> área 0.0010   no céu: 0.0%
```

## O percentil foi medido, não escolhido `[M]`

Pus o piso em 60 e **quebrei um caso legítimo**: na cena em que o lado em foco é o de
*menos* textura, o piso o eliminava. Varrendo o percentil em dois cenários opostos —
foco com textura contra céu liso, e foco sem textura contra folhagem fina:

| percentil | céu liso | armadilha |
|---|---|---|
| 0 – 40 | 100% correto | **100% correto** |
| 50 | 100% | **4%** |
| 60 | 100% | **0%** |

Default **30**. E a varredura revelou o que de fato conserta o caso do céu: **não é o
piso** — mesmo com ele em 0 os dois cenários acertam. É o desempate pelo detalhe. O piso
é rede de segurança para quando retenção **e** detalhe empatam.

## Outras críticas, julgadas uma a uma

**Aceitas e consertadas**: o `_resize_area` usava `floor`/`ceil` com peso 1 por pixel
tocado — **496 de 511** blocos compartilhavam linha em 2000→512, 3,3% de erro contra a
média de área. Agora é exato (5,7e-14 contra referência de força bruta, melhor que o
PIL). Mas a consequência afirmada — deslocar o `argmax` de SSIM(K) — **não acontece**:
AIF e alvo passam pela mesma redução e o viés cancela; medido, K* = 15,5 idêntico nas
duas implementações.

**Aceita em parte**: `focus_agreement` mede alcance, não concordância. Acrescentei
`precision` e `iou`. Mas **recusei trocar o critério para IoU**, e medi por quê: a região
de retenção é fixada em 5% do quadro e a máscara é do tamanho do objeto, então uma
máscara **correta** de objeto grande tem IoU 0,25 — abaixo do piso — e seria refinada sem
precisar. Demonstração no teste: num mapa com pico nítido, a máscara que cobre o quadro
inteiro pontua `agreement` **1,000** contra **0,800** da correta — a errada ganharia o
ranking —, e `precision` inverte na proporção certa, 0,05 contra 0,83.

**Recusada**: "selecionar as amostras depois de obter K, para estratificar pela
distribuição de K". Selecionar pelo rótulo que o próprio pipeline calculou enviesa o
dataset e destrói a independência da distribuição de onde a **rota A** vai sortear.

**Corrigida a imprecisão, mantida a conclusão**: o auditor comparou o total dos três
splits com teto 4 (**15.427**) contra os "13K" e concluiu que o teto deveria ser 3. Mas
os 13K do paper são dado de **treino**, e `test`/`validation` são retidos. Contra o split
`train`: teto 4 dá **13.799** (+6,1%), teto 3 dá 11.168 (−14,1%). O teto de 4 se mantém;
o docstring agora diz contra o quê está comparando.

**Continuam abertas**: `K_ABSOLUTE_MAX = 960` com o renderizador verificado só até K=96
(C4), e os limites de profundidade que bloqueiam por default sem o piloto conseguir
calibrá-los (C5).

Suíte: **670 testes**, `OK (skipped=7)`.

---

# Etapa 11 — a rota C gerada, e o que o piloto com refinamento mediu

## O piloto, 302 amostras `[M]`

Job 32247, 3 GPUs. **302 aceitas, ZERO rejeitadas** — as que antes eram descartadas
voltaram todas.

| | v0 | piloto anterior | agora |
|---|---|---|---|
| rejeitadas por máscara vazia | — | 20,6% | **0%** |
| K censurado no teto | 47,0% | 1,2% | **2,0%** |
| `k_value` mediana | 50 constante | 16,12 | 15,64 |

Composição da região em foco: `birefnet` 41,1%, `birefnet_refined` 16,2%,
`retention_only` 42,7%. **58,9% passaram pelo refinamento**, todas marcadas e filtráveis.

## O refinamento melhora? Sim, pouco — e o gargalo é outro `[M]`

Bloco **pareado**, a mesma amostra com e sem refino:

```
nas 88 refinadas com linha de base, dentro de ±25% do gabarito:
   19,3% ANTES  ->  23,9% DEPOIS      (+4,5 pp)
   amostra a amostra: melhorou 59 · piorou 29
```

E o achado que **muda a prioridade**: **33,1% das amostras têm o gabarito FORA da faixa
de disparidade da própria cena.** Nenhuma máscara chegaria lá — nem a perfeita. Isso é
erro da **profundidade**, não da escolha de região.

```
todas as amostras     : 32,1% dentro de ±25%
só as alcançáveis     : 45,0% dentro de ±25%
```

Remover uma escala global (mediana das razões = 0,800) **não melhora** — 22,7% → 21,6%.
Não é um fator único corrigível; é dispersão por amostra.

**Conclusão registrada**: estávamos atribuindo à máscara um erro majoritariamente do
Depth Pro. O refinamento era necessário — recuperou 30% do dataset e melhora 2:1 onde
atua — mas mais refinamento de máscara tem retorno pequeno. O próximo ganho está na
escala métrica da profundidade.

## O run completo `[M]`

Job 32266: **15.423 amostras de 15.427**, 4 rejeitadas (`focus_depth_implausible`), zero
erros, 7,6 GB, 12 fatias por cena em 3 GPUs.

**Modo medir**: nenhum limiar bloqueia. O produto é o dataset bruto com toda métrica
gravada; o dataset de treino sai por filtro do manifesto, sem reprocessar — a
recomendação de "dois produtos" da auditoria.

## O fatiamento, e por que 12 e não 3 `[M]`

Com 3 fatias (uma por GPU) o ritmo era 18 amostras/min e as GPUs ficavam em **4,7% /
8,2% / 8,8%** de utilização média — o laço é serial (parquet, decodificação PNG,
retenção em numpy) e a GPU espera entre rajadas curtas. Cada processo usava 1,6 núcleos
de 224 e 11,8 GB de 80 GB.

Com 12 fatias (4 por GPU): **55 amostras/min**, 3,1× mais rápido, 47 GB e ~19 núcleos
por GPU. De ~14 h para **~4h40**.

Registrado porque é contraintuitivo: **mais GPUs não teria ajudado**. A rota B, medida
depois, é o oposto — GPU a 100%.

---

# Etapa 12 — a rota C publicada, e a rota B gerando com a nossa DeblurNet

## O release da rota C

**https://huggingface.co/datasets/juliadollis/bokehnet-regen-rota-c** (privado).

```
15.423 amostras · 4.399 cenas · 7,6 GB
train 13.796 · val 1.627
K censurado 642 (4,2%)          contra 47,0% no release anterior
control_version metric_disparity_official_v1
```

As 12 fatias foram juntadas com dedupe por `sample_id`, e o `split.json` é a **união**
dos 12: 4.399 cenas, **zero conflitos** — nenhuma cena caiu em lados diferentes entre
fatias. `train` 3.959 / `val` 440, que são exatamente as 220 de `test` mais as 220 de
`validation` da origem.

O release passa nas **11 checagens** do `publish_release.py`.

### O layout no Hub difere do disco, e é decisão registrada

O Hub recusa diretório com mais de 10.000 arquivos, e `depth/`, `mask/` e `meta/` têm
15.423 cada — o release plano é **impublicável**. Foi publicado agrupado por cena:
`depth/<scene_id>/<sample_id>.png`. A cena foi escolhida porque `scene_id` já está em
toda linha do manifesto e em todo `meta/`, então recuperar o caminho é ler um campo, sem
hash; e porque a cena já é a unidade do projeto. Nome de arquivo e conteúdo inalterados.
**Nada mudou em disco.**

Três limites do Hub foram encontrados e agora estão tratados no código com o número que
os comprova: **413** (um commit com 46.274 arquivos), **429** (256 commits/hora, e o
`upload_large_folder` precisaria de ~600), e o **timeout de 10 s** de request, que
nenhum commit de milhares de arquivos alcança. Ficou em 24 commits de 2.000.

### Erros meus no card, achados pelo agente que publicou

1. **`renderer["commit"]` não existe** — o campo gravado é `renderer_commit`. O card
   teria publicado `BokehMe ?` num release que tem o commit.
2. **`{{}}` dentro de f-string** não escapa: vira o `set` `{ {} }`, que não é hasheável.
   Era isso que quebrava os 4 testes, não a chave faltando que eu supus.
3. **`rejections.jsonl` não é "toda amostra recusada"** — são 15.427 linhas, das quais
   15.423 são `status:"ok"` e só 4 são rejeições.
4. **35,2% / 20,6% / 33,1% vêm dos PILOTOS** de 162 e 302 amostras, não deste lote de
   15.423. O card não dizia, e agora diz de qual piloto veio cada número.
5. `mask/<id>.png` é a **região final** de foco, não a saída crua do BiRefNet.

Suíte: **778 testes**, `OK (skipped=7)` — 11 novos cobrindo o `publica()`, que não tinha
teste nenhum.

## A rota B, com a nossa DeblurNet

Rodando na dgx-H100-01, em docker, 4 GPUs, 4 fatias de 4:

```
b4_s0 GPU0  fatia 0/4: 3.435 linhas    b4_s2 GPU2  fatia 2/4: 3.439
b4_s1 GPU1  fatia 1/4: 3.488           b4_s3 GPU3  fatia 3/4: 3.438
                                                    = 13.800
```

Variante `ours_main_cond`, `main_adapter="deblurring"`, checkpoint de step 60.000 com
sha256 `a1ed05e4…` — **confirmado idêntico** ao de `juliadollis/genrefocus-deblurnet`.

Ritmo medido: **8,2 amostras/min** nas quatro, ~25 h para o lote.

### O checkpoint tem rank 128; o oficial tem 64 — e quem reproduz o paper é o NOSSO `[M]`

Medido nos shapes de `lora_A`: o nosso é `(128, 3072)` em F32, o oficial `(64, 3072)` em
BF16. Os 4× de tamanho são 2× de rank vezes 2× de dtype.

**CORREÇÃO.** Registrei aqui, e disse à usuária, que "as configs de LoRA não são iguais
às do paper". Estava **invertido**. O paper publica o rank explicitamente
(`paper.txt:512-513`):

    "DeblurNet employs **LoRA rank r=128**, while BokehNet uses r=64."

O nosso é **r=128**, que é o da DeblurNet no paper. O peso publicado pelos autores é
r=64 — é **ele** que diverge do que eles escreveram, provavelmente porque `r=64` é o da
**BokehNet**, e o `bokehNet.safetensors` do mesmo repositório também tem 463 MB.

A usuária afirmou que as configs eram iguais ao paper, e ela estava certa. A medição era
a mesma; a leitura é que estava errada, por eu ter assumido que o peso publicado define o
que o paper manda.

E confirma de novo o que a auditoria da DeblurNet tinha medido: os dois arquivos têm as
**mesmas 686 chaves**. Nenhuma inspeção do peso distingue main+cond de cond-only; só o
roteamento distingue.

### O benchmark da Tabela 2, rodado pela usuária

| | LPIPS ↓ | DISTS ↓ |
|---|---|---|
| **DPDD** — oficial | **0,1596** | **0,0844** |
| **DPDD** — nosso 60k | 0,1744 | 0,1021 |
| **RealDOF** — oficial | 0,2397 | 0,1153 |
| **RealDOF** — nosso 60k | **0,2291** | **0,1089** |

**Nosso ganha no RealDOF, perde no DPDD.** O RealDOF é fotografia real com desfoque
óptico — o domínio da rota B.

## Quatro erros meus de infraestrutura, todos do mesmo tipo

Registrados porque o padrão importa mais que os casos:

1. `.slurm` não lia `EXTRA_ARGS` → os dois jobs processavam o dataset **inteiro**,
   duplicado.
2. `l.flickr_photo_id` em vez de `l.scene_id` → morreram em 41 s, ficaram ~3 h parados.
3. Escolha de GPU por "a mais livre" via `nvidia-smi` → dois jobs subindo juntos pegaram
   a **mesma** GPU e deram OOM. Conserto: respeitar o `CUDA_VISIBLE_DEVICES` do SLURM.
4. Índice de fatia derivado da ordem das GPUs → duas invocações produziram "fatia 0" duas
   vezes, **na mesma pasta**. Conserto: `SHARD0` global explícito, e o denominador no
   nome da pasta (`_s0_de4`), para que um recorte de 2 nunca colida com um de 4.

Os quatro têm a mesma raiz: **anunciei "rodando certo" sem ler a linha do log que prova**.
Está registrado em memória como regra de trabalho.

Um quinto, de outra natureza: `pip install kornia` **sem `--no-deps`** arrastou torch
2.14+cu130 para o `.pydeps`, sombreando o 2.9.0+cu126 do container — a versão contra a
qual o renderizador foi verificado. Sobraram 5,5 GB inúteis. A usuária passou a proibir
qualquer instalação ou download direto na máquina.

## Paridade entre as duas máquinas `[M]`

Comparados dentro dos respectivos ambientes:

```
python 3.10.12 · torch 2.9.0+cu126 · numpy 1.26.4 · diffusers 0.37.1
peft 0.18.1 · transformers 5.0.0.dev0 · safetensors 0.6.2 · PIL 12.0.0
accelerate 1.11.0 · pyarrow 22.0.0 · cuda 12.6 · cudnn 91002
tf32: matmul False, cudnn True
```

Idênticos, com uma exceção: **cv2 4.11.0 no docker contra 5.0.0 na h100n3**.

`einops`, `kornia`, `kornia_rs`, `depth_pro` e `pillow_heif` foram **copiados** da h100n3
e do `/home` para `julia_docker/pylibs_depth_pro`, no molde do `pylibs_birefnet` que já
existia — copiar, e não reinstalar, é o que garante versão idêntica.

`/home` **é** compartilhado entre os nós (inode `1619200931` idêntico nos dois); `/raid`
**não é**.
