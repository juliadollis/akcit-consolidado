# Panorama do que já foi feito

Fechamento em 09/09/2026. Cobre a reprodução do GenRefocus (arXiv 2512.16923v3,
"Generative Refocusing: Flexible Defocus Control from a Single Image") feita
neste repositório, com foco no que foi medido, no que foi conferido no código e
no que ainda está aberto.

Documentos vizinhos, que este aqui não substitui: `REGISTRO.md` diz onde cada
resultado mora, `PLANO_REPRODUCAO_PAPER.md` lista o que do paper ainda dá para
reproduzir, `TABELA_EQ4.md` é a tabela gerada, `historico-ultimo.md` guarda o
histórico longo das campanhas anteriores.

---

## 1. Estado da máquina hoje

- **dgx-H100-01**, 09/09 às 20:28 UTC: as 7 GPUs disponíveis estão rodando
  trabalho seu (`julia_geo_A2linha`, `julia_geo_Blinha`, `julia_geo_B`), entre
  82% e 89% de uso. A GPU 4 segue com 78 GB presos por um vLLM de terceiro.
- **Nada nosso da campanha de bokeh está no ar.** Tudo foi retirado em 05/09,
  com as flags `PARAR_KEEPER_*` postas antes de encerrar keepers e containers,
  para nenhuma volta de fila retomar placa.
- **Cota:** 456G de 500G (limite duro 600G). Começou esta rodada em 575G, acima
  do limite e com 4 dias de carência correndo.

---

## 2. A campanha da Eq. 4, que é o corpo do trabalho

Oito modelos medidos em quatro mesas, todos pelo mesmo pipeline, com o plano de
foco vindo do BiRefNet (a Eq. 4 do paper) e não do pixel central, que era o
defeito silencioso das campanhas antigas.

Os modelos: `oficial do paper`, `fase 1 (só sintético)`, `nosso fase 2 (a+b+c)`,
`só rota c (a+c)`, `sem filtro de SSIM`, `kfix (K da rota b pela Eq. 3)`,
`identidade` e `FLUX.1-dev cru`.

As mesas: **EBB400** (400 cenas), **RealBokeh test v2** (217), **RealDOF** (50) e
**LF-Bokeh reproduzido / BLB** (500 imagens de 10 cenas).

Resultado consolidado em `TABELA_EQ4.md`, com IC95 por bootstrap de 10.000
reamostras e teste pareado cena a cena contra os pesos oficiais. Os dados por
imagem estão em `juliadollis/bokeh-eq4-por-imagem` e
`juliadollis/bokeh-eq4-riemann` (9336 linhas, 32 repos), e a tabela em
`juliadollis/genrefocus-tabela-eq4`.

### O quadro em LPIPS, quanto menor melhor

| Modelo | EBB400 | RealBokeh | RealDOF | LF-repro |
|---|---|---|---|---|
| só rota c (a+c) | **0.1155** | **0.1115** | **0.1178** | **0.1719** |
| nosso fase 2 (a+b+c) | 0.1174 | 0.1248 | 0.2157 | 0.2072 |
| sem filtro de SSIM | 0.1242 | 0.1289 | 0.2243 | 0.2103 |
| kfix (Eq. 3) | 0.1548 | 0.1397 | 0.2508 | 0.2261 |
| oficial do paper | 0.1687 | 0.2867 | 0.1900 | 0.1812 |
| fase 1 (só sintético) | 0.1735 | 0.2676 | 0.2087 | 0.1968 |
| linha de identidade | 0.2785 | 0.3587 | 0.3279 | 0.2371 |
| FLUX cru | 0.7693 | 0.7661 | 0.8520 | 0.8342 |

Três leituras que essa tabela sustenta:

1. **A rota c sozinha ganha em todas as quatro mesas.** Treinar com LFDOF e
   RealBokeh, sem a rota b, dá o melhor modelo, inclusive contra o nosso
   original que usa as três rotas. Isso não é o que o paper prevê na ablação
   dele (Tab. 6), onde as três juntas vencem.
2. **Os pesos oficiais vão mal na RealBokeh e bem na RealDOF.** 0.2867 contra
   0.1900. A RealBokeh test v2 é a mesa onde a distância entre eles e nós é
   maior, e é justamente a mesa cuja procedência do treino não conseguimos
   certificar (ver seção 7).
3. **Comparar LPIPS entre mesas não significa nada**: o piso de cada uma é
   diferente (identidade dá 0.2371 no LF-repro e 0.3587 na RealBokeh). Por isso
   a tabela tem a coluna de margem sobre a identidade, que é a única comparação
   válida entre mesas.

---

## 3. O estudo de sensibilidade ao `k_escala`

Motivo: toda a campanha fixou `k_escala=3.0`, e esse fator multiplica a **faixa
da busca binária por K**. O comentário do próprio `bokeh_net.py` registra que,
com `k_escala=0.01`, os três modelos escolheram K no piso da faixa e a
comparação ficou enviesada. Ou seja, uma constante de protocolo escolhida por
nós podia estar mexendo no ranking que publicamos.

Rodamos as mesmas mesas com fator 1.0 e 5.0. Todos os repos completos: 4 modelos
x 2 fatores na RealBokeh (217 cenas cada) e 2 modelos x 2 fatores na EBB400 (400
cada), conferidos contando linha por repo.

LPIPS:

| Mesa | Modelo | k=1.0 | k=3.0 | k=5.0 |
|---|---|---|---|---|
| RealBokeh | rotac-only-60k | 0.1161 | 0.1122 | 0.1131 |
| RealBokeh | nosso fase 2 | 0.1350 | 0.1248 | 0.1233 |
| RealBokeh | fase 1 | 0.2473 | 0.2676 | 0.2873 |
| RealBokeh | oficial | 0.2570 | 0.2867 | 0.3578 |
| EBB400 | nosso fase 2 | 0.1350 | 0.1174 | 0.1182 |
| EBB400 | oficial | 0.1600 | 0.1687 | 0.1879 |

**Conclusão: o ranking nunca vira, mas a margem cresce com o fator.** Os pesos
oficiais pioram monotonicamente quando a faixa da busca aumenta; os nossos
melhoram e depois estabilizam. Na EBB400 a distância entre nós e o oficial vale
0.025 com fator 1.0, 0.051 com 3.0 e 0.070 com 5.0. Na RealBokeh vai de 0.141 a
0.245.

Consequência prática para o paper: a vantagem existe em toda a faixa testada,
inclusive no fator **menos** favorável a nós, mas o tamanho dela depende de uma
constante de protocolo. O número honesto de reportar é o do fator 1.0, com a
ressalva explícita.

---

## 4. A Tabela 2 do paper (deblurring), já medida

Medida em agosto, com a linha `Input` fechada em 03/09. Vive em
`juliadollis/tab2-metricas`. É a única tabela do paper reproduzível na íntegra,
porque RealDOF e DPDD são públicos.

| Mesa / linha | LPIPS | DISTS | CLIP-IQA | MANIQA | MUSIQ |
|---|---|---|---|---|---|
| RealDOF, oficial (nossa medida) | 0.2385 | 0.1161 | 0.5004 | 0.3532 | 42.46 |
| RealDOF, **publicado** | 0.2408 | 0.1126 | 0.4595 | 0.2884 | 43.52 |
| RealDOF, nosso DeblurNet | 0.2433 | 0.1117 | 0.4534 | 0.3266 | 35.30 |
| RealDOF, Input (nossa medida) | 0.5280 | 0.2865 | 0.3576 | 0.1356 | 23.64 |
| RealDOF, Input **publicado** | 0.5241 | 0.2865 | 0.3562 | 0.2213 | 28.71 |
| DPDD, oficial (nossa medida) | 0.1537 | 0.0779 | 0.6481 | 0.4360 | 65.08 |
| DPDD, **publicado** | 0.1440 | 0.0772 | 0.4755 | 0.3452 | 49.41 |
| DPDD, Input (nossa medida) | 0.3752 | 0.1853 | 0.5303 | 0.2708 | 54.98 |
| DPDD, Input **publicado** | 0.3485 | 0.1827 | 0.4337 | 0.3325 | 45.54 |

**A reprodução fecha nas métricas de referência e não fecha nas sem referência.**
LPIPS e DISTS batem: no RealDOF o DISTS da linha `Input` dá 0.2865 nos dois
lados, dígito a dígito. CLIP-IQA, MANIQA e MUSIQ ficam deslocados, e o
deslocamento aparece **também na linha `Input`**, que não passa por modelo
nenhum. Mesma imagem, sem tratamento, pontuando diferente dos dois lados: a
diferença é de preprocessamento, provavelmente escala ou recorte, e métrica sem
referência é sensível a resolução. O teste que resolve isso é barato: recalcular
MANIQA e MUSIQ da linha `Input` na resolução original.

---

## 5. Duas afirmações minhas que o código desmentiu

Registro para ninguém gastar GPU refazendo o que já existe.

**"Falta implementar a busca binária por K."** Errado. O
`inference/src/pipelines/bokeh_net.py` já faz a bissecção por imagem escolhendo
o K que maximiza SSIM contra o alvo (linhas 485 a 508 na versão anterior à
correção), exatamente a Sec. 4.1 do paper. A vencedora vai para `image_best_k` e
é sobre ela que a fidelidade é medida.

**"O nosso LVCorr não é o do paper."** Errado. Todo repo de saída grava o sweep
`image_k01/k05/k10/k15`, e a métrica é Pearson entre os K e a variância do
Laplaciano da imagem inteira desse sweep, que é a definição do paper. A anotação
antiga do `REGISTRO.md` sobre "variância no fundo" descreve a métrica riemanniana
espacial do `metricas_riemannianas.py`, que é outra coisa.

O que **sobra** de dúvida real é o **sinal**. O paper publica LVCorr +0.9368 para
o GenRefocus, e o nosso Pearson cru dá negativo quando o modelo obedece o K
(mais K, mais desfoque, menos variância do Laplaciano). Por isso existe a coluna
`LVCorr_convencao_paper`, que guarda o simétrico. Se a definição deles for
Pearson cru, o sinal publicado implicaria variância crescendo com o K, o que não
fecha fisicamente. Isso é pergunta para os autores, não experimento.

---

## 6. Defeitos encontrados e corrigidos

**Upload de lote que falha e trava o gatilho para sempre.** No `bokeh_net.py`, o
gatilho era `len(lista) == tamanho_lote`. Quando um upload ao Hub falhava, a
exceção era engolida pelo `except` por imagem e a lista **não** era zerada. Com
11 linhas na lista, a igualdade exata nunca mais fechava, e a rodada seguia
gerando por horas sem gravar, até a última imagem. Aconteceu com o `kfix` no
LF-repro em 04/09 e com dois itens do `k_escala`, sempre em janelas de
instabilidade do Hub (`read operation timed out`, `503 Service Unavailable`).

Nada se perdeu, porque no fim tudo o que estava em memória subiu de uma vez.
Mas era risco: um processo morto levaria tudo junto. Corrigido em duas partes:
`>=` no lugar de `==`, e uma variável `nome_parquet_pendente` que preserva o
nome do parquet do lote que falhou, para a retentativa não abrir buraco na
numeração (o pipeline pula lote pelo nome do arquivo, então um buraco faria uma
rodada futura regerar imagens já feitas).

**Guardião de memória frouxo demais.** O `fila_gpu_eq4.sh` exigia 40 GB livres
para subir um item. Com uma inferência de 34 GB já rodando, sobravam 47 GB e o
guardião deixava passar, colocando dois jobs na mesma placa. Aconteceu no
lançamento dos keepers em 03/09; corrigido subindo o piso para 70 GB via
`MIN_LIVRE_MIB`.

**Rodada incompleta que o keeper não conserta sozinho.** O `run_3models.py` pula
um item que já está na tabela de métricas. Se a inferência terminou incompleta,
a volta seguinte do keeper não completa nada. Contornado relançando o item com
um sufixo no `--nome` e o mesmo `--saida`: a guarda não dispara, e o pipeline
gera só os lotes cujo parquet não existe. Foi assim que `bokeh-kesc1-rb-nosso` e
`-fase1` saíram de 207 para as 217 cenas.

**Variável de ambiente que não pega.** O `fila_gpu_eq4.sh` exporta
`BOKEHNET_METRICS_REPO`, mas o `run_3models.py` lê `BOKEH_METRICS_REPO`. A
intenção era mandar a campanha da Eq. 4 para uma tabela própria; na prática tudo
foi para o log de append de sempre. É inofensivo, porque o log é append e o
`monta_listas_eq4.py` filtra pelo prefixo `bokeh-eq4-`, mas continua lá.

---

## 7. O que sabemos que não sabemos

- **Procedência da rota a.** O pipeline gravou só um UUID como nome, sem caminho
  de origem. Consequência permanente: não conseguimos certificar que um
  benchmark qualquer está fora do treino da fase 1. Qualquer paper nosso carrega
  essa ressalva.
- **LF-Bokeh e LF-Refocus não foram liberados.** Conferido na página do projeto:
  o código (`github.com/rayray9999/Genfocus`) e a demo estão no ar, os datasets
  não. A mesa `LF-Bokeh reproduzido` é reconstrução de protocolo; os pesos
  oficiais medem 0.1812 nela contra 0.0833 publicado. Comparação entre modelos
  vale, valor absoluto não.
- **O sinal do LVCorr**, descrito na seção 5.

---

## 8. Infraestrutura que ficou pronta

Tudo em `scripts/`, versionado:

- `fila_gpu_eq4.sh`: roda uma fila de itens numa GPU, com guardião de memória
  livre. Injeta o `PYTHONPATH` do BiRefNet, sem o qual o plano de foco cai no
  pixel central e a rodada deixa de ser da Eq. 4.
- `keeper_eq4.sh`: mantém a placa ocupada dando voltas na fila. Lê o arquivo a
  cada volta, então dá para injetar trabalho editando o arquivo. Para com
  `touch PARAR_KEEPER_<gpu>`.
- `consolida_lfrepro.sh`: encadeamento em duas passadas. Espera as filas
  fecharem pelo `FILA COMPLETA` dos wrappers, e não por `docker ps`, porque
  entre um item e outro existe um intervalo de segundos sem container e quem
  olhasse o `docker ps` consolidaria com a fila pela metade.
- `pipeline_noite.sh`, `consolida_final.sh`, `vigia_noite.sh`: a cadeia da noite
  de 03/09, da inferência até a tabela, sem intervenção.

Uma lição que vale registrar: volta de keeper em fila já concluída **não é
grátis**. Ela carrega o FLUX e o Depth Pro na placa para depois pular tudo. Serve
para segurar a GPU contra outros usuários, mas não produz nada, então não deve
ficar rodando por dias.

---

## 9. Armazenamento

A cota estava em **575G** de 500G, acima do limite, com carência de 4 dias
correndo. Hoje está em **456G**.

Apagado, tudo derivado e reprodutível:

- cache Arrow dos dois datasets da rota A, 76G. Os parquets de origem seguem em
  `hub/`, e o Arrow se refaz sozinho na próxima leitura.
- `data/spring_zips`, 23G. Os três zips já estavam extraídos nas 37 cenas de
  `data/spring`, conferido antes de apagar.

Descoberta útil: os 54G do `hf-cache` antigo **não contam na sua cota**. Todos os
arquivos são do root, criados por containers que rodaram como root em junho, e
cota é por dono. Apagar aquilo não devolve cota nenhuma, só espaço no `/raid`
compartilhado, e exige um container com privilégio.

Ainda disponível para liberar, se precisar: 77G dos parquets da rota A em
`hub/datasets--AKCITPixel3--*` (download puro, só necessário se formos retreinar
a fase 2), 23G de `data/spring_split` (intermediário entre o `spring` extraído e
o `spring_prep`, que é o que o treino riemanniano lê), e 8,8G de
`vision-pipeline/temp_depth_maps` (mapas derivados, regerados na FASE 1).

---

## 10. Pendências, em ordem de custo

1. **MANIQA e MUSIQ em resolução original** na linha `Input` da Tab. 2, para
   fechar o diagnóstico da seção 4. Barato, quase não usa GPU.
2. **Bokehlicious como baseline externo.** Código e pesos públicos em
   `github.com/TimSeizinger/Bokehlicious`, com controle por f-stop. Hoje as
   nossas tabelas só têm identidade e FLUX cru como referência de fora. É uma
   das linhas das Tabs. 2 e 3 do paper.
3. **Refocusing ponta a ponta**, análogo da Tab. 4. Hoje avaliamos o BokehNet
   recebendo a imagem nítida e o DeblurNet em separado; o paper avalia a cadeia
   inteira a partir da imagem borrada. É a única avaliação do paper que ainda
   não temos, e a mais cara.
4. **Treino (a+b)**, sintético mais ITW, que é a única célula que falta para a
   Tab. 6 do paper ficar reproduzida por inteiro. É treino, não avaliação.
5. **Tab. 5** (refocusing direto contra duas etapas). Exige treinar um baseline
   novo e a conclusão dele já é o desenho que adotamos.
