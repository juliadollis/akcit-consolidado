# Plano de execução — ordem, dependências e a hipótese que ele aposta

Decidido pela Julia em 2026-09-10. Este documento existe para que a ordem seja
explícita, para que a **hipótese** embutida nela seja testável em vez de presumida, e
para que o avaliador externo entenda por que as coisas foram feitas nesta sequência.

---

## A restrição que dita tudo

A rota A **sorteia** o plano de foco e o nível de bokeh K (`paper.txt:329-330`). Ela não
tem equação para K: precisa de uma **distribuição** de onde sortear, e essa distribuição
vem das rotas B e C, que calculam K de verdade — B pela Eq. 3 sobre EXIF, C pelo sweep da
Eq. 5.

Ou seja, a ordem natural seria **B e C antes de A**, e a rota A ficaria bloqueada até as
duas terminarem. A DeblurNet nova, que a rota B precisa, ainda está treinando.

## A jogada

Não esperar.

```
    agora                     depois do treino da DeblurNet
    ─────                     ────────────────────────────
 1. rota B  com a DeblurNet          5. rota B REGERADA com
    OFICIAL do paper                    a nossa DeblurNet
 2. rota C  (a nossa)                       │
        │                                   ▼
        ├──► distribuição de K        6. treino da BokehNet
        │                                em B (nossa) + C
 3. rota A, sorteando dessa
    distribuição
        │
        ▼
 4. treino da BokehNet em A
    (pré-treino sintético)  ◄── roda EM PARALELO com o treino da DeblurNet
```

O ganho: o pré-treino sintético da BokehNet — que é a fase 1 e leva tempo — acontece
**enquanto** a DeblurNet treina, em vez de depois. E a rota C, que não depende da
DeblurNet em nada, fica pronta e disponível o tempo todo.

## A hipótese que isso aposta

> **A distribuição de K não muda muito entre a rota B feita com a DeblurNet oficial e a
> mesma rota feita com a nossa.**

Se a hipótese vale, a rota A gerada no passo 3 continua válida depois do passo 5, e o
pré-treino do passo 4 não se perde. Se não vale, a rota A precisa ser regerada e o
pré-treino refeito.

### Por que ela é plausível

K vem da **Eq. 3**, que consome focal length e f-number do EXIF, e `D_focus`. Desses
três, a DeblurNet só influencia `D_focus` — e indiretamente, via a AIF que ela produz,
que alimenta Depth Pro e BiRefNet. Os dois termos de EXIF são idênticos nas duas
variantes, porque vêm do arquivo, não do modelo.

### Por que ela pode falhar

Foi medido que a nossa DeblurNet, rodada **sem** `main_adapter="deblurring"`, sai
**lavada** — e essa AIF ruim contamina Depth Pro, BiRefNet e portanto `D_focus`. Ou seja,
a DeblurNet **pode sim** mexer no K de forma relevante; foi exatamente isso que o defeito
B1 provocou. A diferença entre a oficial (cond-only, usada do jeito dela) e a nossa
(main+cond, usada do jeito certo) é menor que a diferença entre "certa" e "lavada", mas
não é obviamente zero.

### Como testar, e é barato

Quando a nossa DeblurNet ficar pronta, **antes** de regerar a rota B inteira: rodar as
duas variantes sobre **as mesmas ~200 amostras** e comparar as distribuições de K —
p05 / mediana / p95, e a razão pareada por amostra. Os `sample_id` são iguais nas duas
variantes de propósito, justamente para permitir comparação pareada.

Critério de decisão a fixar **antes** de olhar o resultado: se a mediana da razão pareada
ficar dentro de ±X%, a rota A é preservada; fora disso, é regerada. O valor de X é
`[A]` e precisa ser escolhido — sugestão a discutir: ±10%.

## Prioridade declarada

> **A rota C tem que estar 100% correta.**

Ela é a única que não depende da DeblurNet, é a que tem pares reais dos dois lados, e é
contra ela que as outras duas vão ser comparadas. Erro nela contamina a comparação
inteira.

## Consequência para a rota B

A rota B usa **só a DeblurNet oficial** por ora (`nycu-cplab/Genfocus-Model`,
`deblurNet.safetensors`, cond-only, `main_adapter=None`). A infraestrutura das duas
variantes já existe e está testada em `src/model_runtime/deblurnet.py`; trocar é mudar um
valor de enum. A tripla (repositório, arquivo, adapter) é indivisível por construção,
então não há como regerar com os pesos de uma e o adapter da outra.

Um detalhe medido que torna isso obrigatório e não apenas cuidadoso: os dois
`.safetensors` têm **as mesmas chaves de LoRA**. A diferença entre main+cond e cond-only
vive no **roteamento**, não no arquivo. Nenhuma inspeção do peso detecta um cruzamento —
só o vínculo no código.

## Um release por variante

Cada variante da rota B vira um `--output-dir` e um repositório HF próprios, com
`deblur_variant` gravado por amostra e conferido na publicação. Um release meio de uma
variante e meio de outra é exatamente o modo de falha do kfix, que rodou duas convenções
no mesmo lote — e o `publish_release.py` já reprova `control_version` misto pelo mesmo
motivo.

## O que fica em aberto

- **`[A]`** o critério de ±X% para decidir se a rota A é preservada.
- **`[A]`** se o pré-treino da BokehNet na rota A pode de fato correr em paralelo com o
  treino da DeblurNet, dado o orçamento de GPU da h100n3 e a QOS `onejob` (2 jobs
  simultâneos por usuário).
- A decisão sobre o teto de níveis por cena — o paper diz *"2 to 4 images per set"*
  (`paper.txt:1001-1003`) e nós pegamos todos os níveis, com 25% das amostras vindo de
  6,2% das cenas. Muda o tamanho do dataset de ~20K para ~13K.
