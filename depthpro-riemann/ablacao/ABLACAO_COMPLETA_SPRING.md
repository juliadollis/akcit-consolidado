# Ablação completa no Spring — 39 configurações, Optuna e estatística

Runbook da ablação completa no Spring, no mesmo formato da que foi feita no Hypersim:
todas as combinações de termos, busca de hiperparâmetros e o aparato estatístico.

A diferença em relação ao que já rodou no Spring: aquilo foram quatro configurações
selecionadas à mão. Aqui é a varredura completa, com os termos geométricos na
**formulação métrica**.

---

## 1. O que mudou no código

A curvatura passou a ser calculada sobre a superfície 3D retroprojetada, em unidades
físicas, e não sobre o gráfico em coordenadas de imagem normalizadas. Isso exige a
distância focal.

| arquivo | mudança |
|---|---|
| `riemann/geometry.py` | movido da raiz para dentro do pacote |
| `scripts/test_geometry_metrica.py` | movido para `scripts/` |
| `riemann/losses.py` | novas `gauss_loss_metrica` e `normal_loss_metrica`; `RiemannWeights` ganha `usar_metrica` e `gauss_clamp_metrico`; `forward` aceita `focal` |
| `riemann/dataset.py` | aceita `focal_px` ou lê `intrinsics.json`; devolve `fx` e `fy` por amostra |
| `riemann/trainer.py` | repassa a focal do lote para a perda |
| `scripts/run_ablation.py` | flags `--focal` e `--gauss-clamp-metrico` |
| `scripts/run_optuna.py` | idem, e o teto entra no espaço de busca |

**Compatibilidade.** Sem focal, os termos caem na formulação antiga e a coluna
`metrica_ativa` do CSV sai 0. Nada quebra, mas o resultado não é comparável.

### O ponto que mais oferece risco

A focal é dada para a resolução **original**. Ao redimensionar, as escalas horizontal e
vertical podem **diferir**: o Spring é 1920×1080 e o pipeline trabalha em 512×512, então

```
fx = f · 512/1920 = 0,267·f        fy = f · 512/1080 = 0,474·f
```

O dataset faz isso automaticamente. O que **você** precisa fornecer é a focal original.
Confira no log: a primeira época imprime os dois valores.

---

## 2. Antes de começar

### 2.1 Validar a geometria (2 min)

```bash
python scripts/test_geometry_metrica.py
```

Todos os testes devem passar. Se falhar, para: é implementação, não dado.

### 2.2 Registrar a focal no dataset

Crie um `intrinsics.json` na raiz de cada partição, para não depender de passar a flag
toda vez:

```bash
for P in train val test; do
  echo '{"fx": 2586.0}' > /data/spring_prep/$P/intrinsics.json
done
```

Substitua 2586.0 pela focal que o `prepare_spring.py` reportou. Alternativamente use
`--focal 2586` nos comandos.

### 2.3 Conferir que a métrica ativou

Rode uma configuração por uma época e olhe duas coisas no log: os valores de `fx` e `fy`,
e a coluna `metrica_ativa`, que precisa sair **1**. Se sair 0, a focal não chegou.

---

## 3. O desenho do estudo ablativo

Antes dos comandos, o que a varredura é. As 39 configurações não são arbitrárias: estão
organizadas em blocos, e cada bloco responde a uma pergunta diferente. Entender isso é o
que permite ler o CSV.

Os seis termos disponíveis são `berhu` (erro de profundidade, o único não geométrico),
`grad` (primeira ordem), `normal` (orientação da superfície), `gauss` (curvatura
Gaussiana, intrínseca), `geod` (distância geodésica) e `metric` (tensor métrico).

| bloco | n | conteúdo | pergunta que responde |
|---|---|---|---|
| **B0** | 1 | berHu puro | Qual é o piso? É o controle contra o qual todo ganho geométrico precisa ser medido. |
| **B1** | 5 | berHu + um termo geométrico | Qual termo, **isolado**, contribui? É o bloco mais interpretável. |
| **B2** | 10 | berHu + todos os pares | Há **interação** entre termos? Dois termos juntos podem render mais ou menos que a soma. |
| **B3** | 10 | berHu + todos os trios | A interação se sustenta em três? |
| **B4** | 5 | berHu + quádruplos | Há retorno decrescente ao empilhar? |
| **B5** | 1 | todos os termos | O empilhamento completo ajuda ou atrapalha? |
| **B6** | 3 | só geometria, **sem** berHu | A geometria sozinha sustenta o treino? É um controle: se B6 colapsa, os termos geométricos são regularizadores e não objetivos autônomos. |
| **B7** | 4 | peso alto em gauss e normal | Testa a **hipótese central** do trabalho de forma direta. |

Duas observações sobre a leitura. O **B1 é o bloco de maior valor informativo por
configuração**: com termos isolados, qualquer efeito é atribuível. Nos blocos de
combinação, um ganho pode vir de qualquer um dos termos ou da interação, e separar isso
exige comparar com os B1 correspondentes.

O **B6 é controle, não candidato**. Espera-se que vá mal. Se for bem, é um achado
inesperado e importante; se colapsar, confirma que os termos geométricos funcionam como
regularização em cima de um objetivo de profundidade, o que é uma afirmação mais modesta
mas defensável.

### Como ler o CSV resultante

O `ablation_results.csv` tem uma linha por configuração, mais duas linhas especiais:

- `ZERO_SHOT_sem_finetune` — o modelo de prateleira no mesmo protocolo. **Primeira linha
  do arquivo.** É a referência para "o treino ajuda?".
- linhas com `status = DIVERGIU` — configurações cuja loss virou não-finita e foram
  abortadas. Não são resultado, são falha de estabilidade numérica. Registre quais
  divergiram e com quais termos, porque isso é informação (em geral aponta para `metric`
  ou `geod` com peso alto), mas **não as inclua na comparação**.

As colunas que importam são `boundary_fmax` (a métrica de borda a reportar, porque varre o
limiar e não penaliza modelo conservador), `abs_rel`, `d1`, e `boundary_precision` com
`boundary_recall` para entender a natureza de um eventual ganho de borda.

### Como escolher a vencedora, e o erro a evitar

A tentação é pegar a linha com o melhor `boundary_fmax` e chamar de campeã. Isso está
errado por dois motivos.

Primeiro, **a ablação roda uma semente por configuração**. Com 39 sorteios, o máximo está
enviesado para cima por construção: parte da vantagem da primeira colocada é sorte. Foi
exatamente o que aconteceu no Hypersim, onde um `bF` de 0,772 numa semente virou 0,758 com
três sementes e o intervalo passou a conter o zero-shot.

Segundo, **a seleção usa a validação**, e reportar o número da mesma validação que
selecionou é vazamento.

O procedimento correto tem três etapas. A ablação **triage**: ela identifica candidatas,
não vencedoras. Escolha as duas ou três melhores, não a melhor. Depois o Optuna refina os
pesos dentro da região promissora. Só então as candidatas vão para o `train_single` com
múltiplas sementes e o conjunto de teste held-out, e é dali que sai o número a reportar.

Ao escolher as candidatas, olhe o **padrão** e não só o ranking: se as cinco melhores
compartilham um termo, isso é mais informativo que a primeira colocada isolada. E compare
sempre contra o `B0`, não apenas contra o zero-shot — foi o que revelou, na rodada
anterior, que a curvatura entrega menos que o berHu puro.

## 4. Fase A — Ablação completa, 39 configurações

```bash
python scripts/run_ablation.py \
  --train-root /data/spring_prep/train \
  --val-root   /data/spring_prep/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/spring_ablacao_completa \
  --variant heads_final \
  --filter '' \
  --focal 2586 --gauss-clamp-metrico 5.0 \
  --epochs 30 --lr 1e-5 --batch-size 2 --grad-accum 4 --size 512 \
  --monitor boundary_fscore --seed 42
```

O `--filter ''` é o que dispara as 39. O zero-shot é medido como primeira linha do CSV,
no mesmo protocolo, então a tabela nasce autocontida.

**Custo.** Cerca de 2 h por configuração no ritmo observado, então algo entre 3 e 4 dias.
Rode em sessão persistente.

**Ponto de decisão, cerca de 15 min após o início.** Confira na primeira linha do CSV que
o zero-shot bate os valores já conhecidos (AbsRel ≈ 0,36 e bF ≈ 0,54 no teste). Se
divergir muito, algo mudou no protocolo e vale parar.

---

## 5. Fase B — Optuna

```bash
python scripts/run_optuna.py \
  --train-root /data/spring_prep/train \
  --val-root   /data/spring_prep/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/spring_optuna \
  --variant heads_final \
  --focal 2586 \
  --trials 40 --epochs-per-trial 12 \
  --batch-size 2 --grad-accum 4 --size 512 \
  --monitor boundary_fscore --seed 42
```

O espaço de busca agora inclui o **teto da curvatura** entre 1 e 50 em escala
logarítmica. Calibrá-lo à mão se mostrou difícil: 50 cortava 40% dos pixels na formulação
antiga e 1000 deixava ruído dominar. Em unidades métricas a faixa útil corresponde a raios
de curvatura entre 1 m e 14 cm, e a busca escolhe.

### Como ler a saída do Optuna

A busca é **multiobjetivo**: minimiza AbsRel e maximiza boundary F-score ao mesmo tempo.
Por isso não existe "um melhor", e sim uma **fronteira de Pareto** — o conjunto de
configurações em que não se pode melhorar uma métrica sem piorar a outra. O
`pareto_front.json` traz essa fronteira.

O `term_presence.json` responde a pergunta que interessa: quais termos aparecem nas
soluções Pareto-ótimas? E aqui há uma armadilha que já nos pegou uma vez. Dizer que
"gauss aparece em 100% do Pareto" parece forte, mas se o peso de gauss é sorteado
uniformemente em [0, 1] e o critério de presença é peso acima de 0,05, então **95% de
todos os trials** têm gauss presente. Cem por cento contra uma taxa-base de 95% é
indistinguível do acaso.

Por isso o relatório traz três números por termo: a presença na fronteira, a **taxa-base**
na população de trials, e o **lift**, que é a razão entre as duas. Só lift claramente acima
de 1 conta como evidência. Um lift de 1,05 não é nada.

### Escolha das candidatas na fronteira

Da fronteira saem duas candidatas, não uma: o extremo de **melhor borda** e o de **melhor
AbsRel**. As duas vão para a fase de sementes. Isso é deliberado — o trabalho já mostrou
que o escopo controla um compromisso entre as duas métricas, e reportar os dois pontos de
operação é mais honesto que escolher o que favorece a narrativa.

---

## 6. Fase C — Campeãs com sementes e teste held-out

Para a vencedora da ablação e para a do Optuna, e **obrigatoriamente para o `B0_berhu`**:

```bash
python scripts/train_single.py \
  --train-root /data/spring_prep/train \
  --val-root   /data/spring_prep/val \
  --test-root  /data/spring_prep/test \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/spring_campea_<nome> \
  --variant heads_final --focal 2586 \
  --berhu ... --grad ... --normal ... --gauss ... --geod ... --metric ... \
  --epochs 60 --seeds 5 --lr 1e-5 --batch-size 2 --grad-accum 4 \
  --monitor boundary_fscore --export-curvature
```

O `B0_berhu` não é opcional. Sem ele não se separa "a geometria ajudou" de "fine-tuning no
domínio certo ajudou", e essa é a pergunta central. A rodada anterior mostrou que o berHu
puro sozinho entrega o maior ganho contra o zero-shot, então ele é o adversário real.

**Use 5 sementes.** Com 3 o multiplicador do t de Student é 4,30; com 5 cai para 2,78. Os
efeitos aqui são de centésimos e 3 sementes não os separam.

---

## 7. Fase D — Avaliação e consolidação

```bash
# comparacao emparelhada por cena contra o zero-shot, para cada campea
python scripts/evaluate_paired.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --seeds-dir  /workspace/runs/spring_campea_<nome> \
  --data-root  /data/spring_prep/test \
  --out-dir    /workspace/runs/spring_campea_<nome>/aval \
  --variant heads_final --nivel cena

# tabela geral com todas as campeas e o zero-shot
python scripts/make_results_table.py \
  --entrada berhu=/workspace/runs/spring_campea_b0/aval \
            ablacao=/workspace/runs/spring_campea_ablacao/aval \
            optuna=/workspace/runs/spring_campea_optuna/aval \
  --out-dir /workspace/relatorio_spring_completo \
  --titulo "Spring — ablacao completa, teste held-out"
```

---

## 8. Erros que já cometemos neste projeto

Lista curta do que deu errado antes, para não repetir. Cada item custou dias.

**Reportar o número da ablação como resultado.** A ablação tem uma semente por
configuração e seleciona na validação. No Hypersim um `bF` de 0,772 virou 0,758 com três
sementes, e o intervalo passou a conter o zero-shot. O número do paper sai da fase de
sementes no teste held-out, nunca da ablação.

**Usar três sementes.** O multiplicador do t de Student com n=3 é 4,30; com n=5 cai para
2,78. Os efeitos aqui são de centésimos e três sementes não os separam. Foi o que deixou o
ganho de borda inconclusivo por duas rodadas.

**Confundir intervalo entre sementes com efeito contra o baseline.** O primeiro mede
reprodutibilidade do treino. O segundo exige comparação emparelhada contra o zero-shot.
São perguntas diferentes e os dois números precisam aparecer, rotulados.

**Tratar imagens como amostras independentes.** Imagens da mesma cena são fortemente
correlacionadas. O tamanho efetivo de amostra é o número de **cenas**, e no Spring é o
número de **sequências**. O `evaluate_paired --nivel cena` já faz isso; o que não pode é
reportar intervalo calculado sobre imagens.

**Esquecer o `B0_berhu`.** Sem ele, um ganho não distingue "a geometria ajudou" de
"fine-tuning no domínio certo ajudou". A rodada anterior mostrou que o berHu puro sozinho
entrega o maior ganho contra o zero-shot, então ele é o adversário real.

**Chamar significância de relevância.** Com n grande o teste detecta efeito trivial. Um
delta de 0,0004 em AbsRel é 0,5% relativo e sair como significativo diz apenas que a
medição é precisa. Declare o limiar de relevância prática antes de olhar os resultados.

**Assumir que o termo está ativo.** A curvatura rodou três datasets em unidades erradas
sem ninguém notar, porque o teste de sanidade usava unidades consistentes. Confira a
coluna `metrica_ativa` e rode o `test_geometry_metrica.py` antes de cada campanha.

## 9. Como reportar

O resultado principal é a **comparação emparelhada por cena no teste**, com ganho médio,
intervalo da diferença e os dois p-valores. O intervalo entre sementes mede
reprodutibilidade do treino, não o efeito contra o baseline, e entra como informação
secundária.

**Adote um limiar de relevância prática declarado antes de olhar os resultados.** Com n
grande o teste detecta efeitos triviais: um delta de 0,0004 em AbsRel é 0,5% relativo e
sair como significativo diz apenas que a medição é precisa. Sugestão: só chamar de ganho o
que for significativo **e** acima de 1% relativo.

Duas comparações precisam aparecer, e elas respondem perguntas diferentes:

- **contra o zero-shot** — o fine-tuning ajuda?
- **contra o `B0_berhu`** — a geometria acrescenta sobre o berHu puro?

Na rodada anterior a primeira deu sim e a segunda deu não, com sete de oito comparações
significativas na direção negativa. A ablação completa testa se alguma combinação escapa
disso.

---

## 10. Custo total

| fase | tempo |
|---|---|
| A. Ablação 39 configs | 3 a 4 dias |
| B. Optuna, 40 trials | ~20 h |
| C. Campeãs, 3 braços × 5 sementes | ~30 h |
| D. Avaliação e tabelas | ~1 h |

Se o tempo apertar, a ordem de corte é: reduzir os trials do Optuna, depois reduzir as
sementes de 5 para 4. **Não corte** o `B0_berhu` da Fase C nem o zero-shot da Fase A.
