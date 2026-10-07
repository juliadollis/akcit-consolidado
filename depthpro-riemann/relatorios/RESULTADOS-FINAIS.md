# Resultados

Split de **teste** do Spring: 485 imagens de **13 cenas**. Todo peso está
publicado em `akcit-dephpro/depthpro-riemann-modelos`.

**n = 6** (seeds 0 a 5) em todos os braços, exceto o de 768 px, que só teve
3 seeds treinadas.

---

## 1. Tabela

| braço | perda | n | F-borda | fmax | f_auc | AbsRel | delta1 | RMSE |
|---|---|---|---|---|---|---|---|---|
| zero-shot | sem fine-tune | - | 0.5402 | 0.7674 | 0.5641 | 0.3602 | 0.6594 | 5.4002 |
| **Bgrad** | berHu 0,7 + grad 0,3 | 6 | 0.6091 | 0.7882 | 0.6253 | 0.2561 | 0.6937 | 4.3492 |
| **Bgeod** | berHu 0,7 + geod 0,1 | 6 | 0.5984 | 0.7790 | 0.6308 | 0.2481 | 0.6950 | 4.2336 |
| **B0 berHu (controle)** | berHu 0,7 | 6 | 0.5954 | 0.7791 | 0.6254 | 0.2509 | 0.6946 | 4.2433 |
| **B0 berHu a 768 px** | berHu 0,7, treinado a 768 px | 3 | 0.5918 | 0.7463 | 0.6041 | 0.2500 | 0.7000 | 4.1258 |
| **Bmetric** | berHu 0,7 + metric 0,2 | 6 | 0.5904 | 0.7681 | 0.6202 | 0.2489 | 0.6958 | 4.2604 |
| **B3 normal+curvatura, teto 50** | berHu 0,7 + normal 0,9 + curvatura 0,45 | 6 | 0.5765 | 0.7713 | 0.5906 | 0.2733 | 0.6960 | 4.4016 |
| **B3 normal+curvatura, teto 5** | berHu 0,7 + normal 0,9 + curvatura 0,45 | 6 | 0.5723 | 0.7690 | 0.5875 | 0.2778 | 0.6916 | 4.4110 |
| **B3 normal+curvatura, teto 1000** | berHu 0,7 + normal 0,9 + curvatura 0,45 | 6 | 0.5696 | 0.7628 | 0.5804 | 0.3104 | 0.6750 | 4.7995 |
| **B1 curvatura só, teto 5** | berHu 0,7 + curvatura 0,45 | 6 | 0.5523 | 0.7560 | 0.5872 | 0.2687 | 0.6928 | 4.5229 |
| **B1 curvatura só, teto 1000** | berHu 0,7 + curvatura 0,45 | 6 | 0.5378 | 0.7161 | 0.5697 | 0.2520 | 0.7057 | 4.3756 |

---

## 2. Contraste pareado contra o controle

Diferença seed a seed contra o B0 berHu. A seed fixa inicialização e ordem
dos dados nos dois braços, então o pareamento remove a variação do sorteio.

| braço | n | delta F-borda | desvio | seeds a favor | delta fmax |
|---|---|---|---|---|---|
| **Bgrad** | 6 | +0.0137 | 0.0084 | 6/6 | +0.0091 |
| **Bgeod** | 6 | +0.0030 | 0.0072 | 4/6 | -0.0000 |
| **B0 berHu a 768 px** | 3 | -0.0026 | 0.0094 | 2/3 | -0.0344 |
| **Bmetric** | 6 | -0.0050 | 0.0038 | 1/6 | -0.0109 |
| **B3 normal+curvatura, teto 50** | 6 | -0.0190 | 0.0263 | 2/6 | -0.0078 |
| **B3 normal+curvatura, teto 5** | 6 | -0.0231 | 0.0223 | 1/6 | -0.0101 |
| **B3 normal+curvatura, teto 1000** | 6 | -0.0258 | 0.0168 | 1/6 | -0.0163 |
| **B1 curvatura só, teto 5** | 6 | -0.0431 | 0.0183 | 0/6 | -0.0231 |
| **B1 curvatura só, teto 1000** | 6 | -0.0576 | 0.0042 | 0/6 | -0.0630 |

**17 vitórias em 51 comparações pareadas.**

Só os braços com curvatura: **4 em 30**.

---

## 3. A leitura

**Existe headroom no Spring.** Todo braço treinado bate o zero-shot, e em
AbsRel a diferença é grande (0,3602 contra ~0,25). É o oposto do Hypersim,
que está na lista de treino do DepthPro e por isso não tinha espaço a tomar.

**A curvatura piora a borda.** O braço B1, que a isola, perde mais que o B3,
que a mistura com o termo normal. O normal **mascara parte do estrago** em
vez de ajudar.

**O `grad` é o único ganho.** É também a única intervenção de toda a campanha
que moveu o `fmax`, o F-score no melhor limiar de cada modelo. Esse teto de
~0,77 não cedeu a seis configurações de perda nem a 768 px de resolução.

---

## 4. O que estes números NÃO sustentam

**O n efetivo é 13 cenas, não 485 imagens.** Quadros consecutivos da mesma
sequência são quase o mesmo dado. O desvio entre cenas é 0,2147 e o erro
padrão da média é **0,0595**, da mesma ordem do maior efeito medido.

**Duas cenas dominam.** A `seq0020` (77 quadros, 16% do teste) tem
`d1 = 0,05`, e a `seq0043` tem `AbsRel = 2,28`. Excluindo as duas, o AbsRel
do zero-shot cai de 0,3591 para 0,1637.

**Os braços com curvatura não eram reprodutíveis** até a correção do
`F.pad(mode="replicate")`, cujo backward na CUDA usa `atomicAdd`. Medido:
|delta| médio de 0,0148 no F-borda entre duas execuções da mesma seed,
contra 0,0000 no berHu puro.

**Não há teste de significância.** O que sustenta a leitura é a consistência
do sinal, não um p-valor.

