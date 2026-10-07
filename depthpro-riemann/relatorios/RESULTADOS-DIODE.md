# DIODE — generalização fora do domínio

Avaliação dos modelos treinados no Spring contra o **DIODE val indoors**: 325
imagens, 3 cenas, **10 scans**. O n efetivo da estatística é 10, não 325 —
recortes do mesmo scan saem da mesma posição do scanner.

Profundidade `float32` em metros, máscara de validade binária, 99–100% de
pixels válidos. Faixa p5–p95 de **4,57 a 8,35 m**, bem mais estreita que o
Hypersim (~1–10 m).

Varredura completa: 65 checkpoints + zero-shot × 1 mesa.

---

## 1. Tabela

| braço | n | AbsRel | delta1 | RMSE | F-borda | fmax |
|---|---|---|---|---|---|---|
| zero-shot | - | **0,0647** | **0,9400** | **0,4189** | 0,5036 | 0,6714 |
| Bgeod | 6 | 0,1093 | 0,8907 | 0,5737 | **0,5573** | 0,6830 |
| Bmetric | 6 | 0,1170 | 0,8823 | 0,6044 | 0,5503 | 0,6779 |
| B0 berHu a 768 px | 3 | 0,1185 | 0,8780 | 0,6122 | 0,5456 | 0,6639 |
| B3 normal+curv, teto 5 | 6 | 0,0931 | 0,9098 | 0,5161 | 0,5455 | 0,6913 |
| B0 berHu (controle) | 6 | 0,1121 | 0,8871 | 0,5853 | 0,5439 | 0,6695 |
| Bgrad | 6 | 0,0966 | 0,9044 | 0,5266 | 0,5437 | 0,6806 |
| B3 normal+curv, teto 50 | 6 | 0,0959 | 0,9071 | 0,5311 | 0,5427 | 0,6916 |
| B3 normal+curv, teto 1000 | 6 | 0,0940 | 0,9069 | 0,5235 | 0,5416 | 0,6932 |
| B1 curv, teto 5 | 6 | 0,0951 | 0,9111 | 0,5200 | 0,4960 | 0,6296 |
| B1 curv, teto 1000 | 6 | 0,1181 | 0,8780 | 0,5990 | 0,4689 | 0,5952 |

## 2. Contraste pareado contra o controle

| braço | n | delta F-borda | desvio | seeds a favor |
|---|---|---|---|---|
| Bgeod | 6 | **+0,0134** | 0,0089 | 5/6 |
| Bmetric | 6 | +0,0065 | 0,0083 | 4/6 |
| B0 berHu a 768 px | 3 | +0,0071 | 0,0167 | 2/3 |
| B3 normal+curv, teto 5 | 6 | +0,0017 | 0,0078 | 3/6 |
| Bgrad | 6 | −0,0002 | 0,0087 | 3/6 |
| B3 normal+curv, teto 50 | 6 | −0,0012 | 0,0158 | 2/6 |
| B3 normal+curv, teto 1000 | 6 | −0,0023 | 0,0136 | 4/6 |
| B1 curv, teto 5 | 6 | **−0,0479** | 0,0258 | 0/6 |
| B1 curv, teto 1000 | 6 | **−0,0750** | 0,0140 | 0/6 |

---

## 3. A leitura

**O fine-tune no Spring piora a profundidade fora do domínio.** O zero-shot vence
todo braço treinado em AbsRel (0,0647 contra 0,093–0,119), delta1 e RMSE. Não é
margem pequena: o melhor braço treinado erra ~44% mais que o modelo intocado. É
esquecimento catastrófico — o Spring é sintético, de rua, com profundidade longa;
o DIODE é real e interno.

**E melhora a borda.** Na direção oposta, todo braço treinado bate o zero-shot em
F-borda (0,5416–0,5573 contra 0,5036) e quase todos em fmax. As duas coisas
convivem: o fine-tune ensinou onde ficam as descontinuidades e desaprendeu a
escala métrica.

**A curvatura sozinha é o único efeito grande e inequívoco.** B1 teto 5 perde
0,048 e B1 teto 1000 perde 0,075 em F-borda, **0 de 6 seeds a favor nos dois**,
com desvio bem menor que o efeito. É o mesmo sinal do Spring, mais forte aqui.

**O termo normal cancela o estrago da curvatura.** Os braços B3, que somam normal
à curvatura, ficam entre −0,002 e +0,002 do controle — empate — enquanto os B1,
que isolam a curvatura, despencam. A separação B1 vs B3 é de 0,05 a 0,08. Não é
que o normal ajude: ele neutraliza.

**O `grad` não generaliza.** No Spring era o único ganho real e a única coisa que
movia o fmax. Aqui dá −0,0002 e 3 de 6 seeds. O ganho era específico do Spring.

**O `geod` é o que sobra.** +0,0134 com 5 de 6 seeds e desvio 0,0089 — o único
braço cujo efeito é maior que a própria dispersão. Com n=10 grupos isso não é
prova, mas é o único candidato que o DIODE não derrubou.

---

## 4. O que estes números NÃO sustentam

**n = 10, e desbalanceado.** Os grupos vão de 27 a 73 amostras. A barra para
afirmar qualquer coisa é mais alta que no Spring, que já tinha n=13.

**Não sei se o DIODE está no treino do DepthPro.** Se estiver, a vantagem do
zero-shot em AbsRel está inflada e a comparação com os braços treinados não é
limpa. Isso precisa ser checado antes de escrever a conclusão em qualquer lugar.

**A faixa de profundidade é estreita** (4,6–8,4 m). O `gauss_clamp` foi calibrado
noutra faixa, então o desempenho ruim da curvatura aqui pode ser em parte
desajuste de escala, não só o problema de cauda que já conhecíamos.

**Não há teste de significância.** O que sustenta a leitura é consistência de
sinal entre seeds e entre datasets.

**A ablação não entrou.** Os 10 checkpoints em `runs_ablacao_spring/` não têm
componente `seed_` no caminho e o fiscal não os enxerga.

---

# DIODE outdoor — o mesmo teste na faixa larga

446 imagens, 3 cenas, **10 scans**. Medido, não citado: **80,1% de pixels válidos
na mediana** (mínimo 32,8%), contra 99,8% do interno. Profundidade p50 de 9,99 m,
p99 de 36,09 m, máximo 290 m — contra p50 5,35 m e p99 11,09 m no interno.

Varredura de 75 checkpoints + zero-shot, agora **incluindo a ablação**.

## 5. Campanha de seeds — outdoor (n = 6)

| braço | AbsRel | delta1 | F-borda out | F-borda indoors | vs controle |
|---|---|---|---|---|---|
| zero-shot | 0,4073 | 0,6873 | 0,4139 | 0,5036 | — |
| Bgrad | 0,4045 | 0,6698 | **0,4604** | 0,5437 | +0,0026 |
| B0 berHu a 768 px | 0,4491 | 0,6439 | 0,4602 | 0,5456 | +0,0024 |
| B0 berHu (controle) | 0,4444 | 0,6472 | 0,4579 | 0,5439 | — |
| Bmetric | 0,4480 | 0,6452 | 0,4560 | 0,5503 | −0,0018 |
| B3 normal+curv, teto 50 | 0,4116 | 0,6676 | 0,4478 | 0,5427 | −0,0100 |
| Bgeod | 0,4497 | 0,6462 | 0,4462 | **0,5573** | −0,0116 |
| B3 normal+curv, teto 1000 | 0,4155 | 0,6619 | 0,4443 | 0,5416 | −0,0136 |
| B3 normal+curv, teto 5 | 0,4180 | 0,6640 | 0,4382 | 0,5455 | −0,0197 |
| B1 curv, teto 5 | 0,4367 | 0,6526 | 0,4335 | 0,4960 | −0,0244 |
| B1 curv, teto 1000 | 0,4422 | 0,6408 | 0,4137 | 0,4689 | −0,0441 |

## 6. Pareado por seed — outdoor

| braço | n | delta F-borda | desvio | a favor |
|---|---|---|---|---|
| Bgrad | 6 | +0,0026 | 0,0052 | 4/6 |
| B0 berHu a 768 px | 3 | +0,0040 | 0,0137 | 1/3 |
| Bmetric | 6 | −0,0018 | 0,0090 | 3/6 |
| B3 normal+curv, teto 50 | 6 | −0,0100 | 0,0315 | 3/6 |
| Bgeod | 6 | −0,0116 | 0,0086 | 1/6 |
| B3 normal+curv, teto 1000 | 6 | −0,0136 | 0,0535 | 2/6 |
| B3 normal+curv, teto 5 | 6 | −0,0197 | 0,0284 | 1/6 |
| B1 curv, teto 5 | 6 | **−0,0244** | 0,0194 | **0/6** |
| B1 curv, teto 1000 | 6 | **−0,0441** | 0,0323 | 1/6 |

## 7. Ablação — outdoor (n = 1)

| braço | AbsRel | F-borda out | F-borda indoors | vs controle |
|---|---|---|---|---|
| abl +geod | 0,4769 | 0,5409 | 0,5101 | +0,0610 |
| abl +grad | 0,4844 | 0,5070 | 0,4761 | +0,0272 |
| abl +metric | 0,4402 | 0,4939 | 0,5412 | +0,0141 |
| abl B0 controle | 0,4525 | 0,4798 | 0,5061 | — |
| abl gauss dom | 0,4087 | 0,4713 | 0,5128 | −0,0085 |
| abl +gauss | 0,4444 | 0,4610 | 0,4444 | −0,0188 |
| abl +normal | 0,4631 | 0,4251 | **0,5927** | −0,0547 |
| abl gauss pesado | 0,4492 | 0,4228 | 0,5181 | −0,0570 |
| abl normal dom | 0,4680 | 0,4089 | 0,5215 | −0,0709 |
| abl campeão | 0,4598 | 0,3841 | 0,5293 | −0,0958 |

---

## 8. O que o outdoor muda

**Corrige uma afirmação anterior.** A seção 3 diz que "o fine-tune no Spring piora
a profundidade fora do domínio". Isso vale para o **interno**, onde o zero-shot
faz 0,0647 de AbsRel contra 0,093–0,119 dos treinados. No externo o zero-shot faz
**0,4073 contra 0,4045–0,4497** — está no meio do grupo, e o Bgrad até o supera.
O esquecimento catastrófico era específico do regime estreito, não geral.

**A curvatura perde nas três mesas.** B1 teto 5 dá −0,0244 com **0 de 6 seeds a
favor**; B1 teto 1000 dá −0,0441. Somado ao Spring e ao DIODE interno, é o único
efeito que não muda de sinal em nenhum dataset, domínio ou faixa de profundidade.
É o resultado mais sólido da campanha inteira.

**O `geod` era específico do interno.** Lá era o melhor braço (+0,0134, 5 de 6
seeds); aqui é −0,0116 com 1 de 6. Um termo que inverte de sinal entre domínios do
mesmo dataset não é um ganho, é uma interação com o dado.

**O `grad` é o único que nunca prejudica.** Spring positivo, interno −0,0002,
externo +0,0026 com desvio de 0,0052, o menor da tabela. O efeito é pequeno demais
para valer sozinho, mas é o único consistente.

**A ablação contradiz a campanha de seeds outra vez.** O `abl +geod` é o melhor
braço externo (+0,0610) enquanto o `Bgeod` da campanha é negativo; o `abl +normal`
era o melhor interno (+0,0866) e é dos piores externos (−0,0547). Com n = 1 por
braço, a ablação não sustenta ranking de termo.

**80,1% de pixels válidos, mínimo 32,8%.** Como a métrica principal é qualidade de
borda, buraco no ground truth cai justamente onde há descontinuidade. Metade dos
braços externos está dentro de 0,02 do controle, o que é da ordem desse ruído.
