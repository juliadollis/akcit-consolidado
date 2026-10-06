# O que ainda dá para reproduzir do paper

Leitura das 15 páginas de corpo do `paper.pdf` (Generative Refocusing, arXiv
2512.16923v3, 18 mar 2026), cruzada com o que a nossa campanha já mediu.

O paper tem seis tabelas. Nós reproduzimos, até agora, o formato de UMA delas
(a Tab. 3), e mesmo assim num protocolo diferente do original. Abaixo, o estado
de cada uma e o custo de fechar o que falta.

---

## Tabela por tabela

| Tabela | O que é | Dá para reproduzir? |
|---|---|---|
| Tab. 1 | matriz de capacidades entre métodos | não tem número, nada a medir |
| Tab. 2 | **deblurring** em RealDOF e DPDD, 5 baselines + linha `Input` | **sim, inteira e barato** |
| Tab. 3 | bokeh synthesis em LF-Bokeh | dataset não liberado; temos a reconstrução, mas em protocolo diferente |
| Tab. 4 | refocusing em LF-Refocus, 400 pares origem-alvo | dataset não liberado; reconstruível com esforço médio |
| Tab. 5 | ablação refocusing direto vs. duas etapas | precisa treinar um baseline novo |
| Tab. 6 | **ablação de dados do BokehNet** (a / a+b / a+c / a+b+c) | **temos 3 das 4 células** |

Conferi a página do projeto: o código e a demo estão no ar
(`github.com/rayray9999/Genfocus`, `hf.co/spaces/nycu-cplab/Genfocus-Demo`),
mas **LF-Bokeh e LF-Refocus continuam não liberados**. A ressalva que já está no
cabeçalho da nossa tabela permanece válida.

---

## 1. Tab. 2, o deblurring: JA MEDIDA, falta so ler o resultado

Conferido no `juliadollis/tab2-metricas`: a Tab. 2 foi rodada em agosto e a
linha `Input` entrou hoje de manha. Nao ha GPU a gastar aqui, ha leitura a
fazer.

| Mesa / linha | LPIPS | DISTS | CLIP-IQA | MANIQA | MUSIQ |
|---|---|---|---|---|---|
| RealDOF, oficial (nosso) | 0.2385 | 0.1161 | 0.5004 | 0.3532 | 42.46 |
| RealDOF, **publicado** | 0.2408 | 0.1126 | 0.4595 | 0.2884 | 43.52 |
| RealDOF, nosso DeblurNet | 0.2433 | 0.1117 | 0.4534 | 0.3266 | 35.30 |
| RealDOF, Input (nosso) | 0.5280 | 0.2865 | 0.3576 | 0.1356 | 23.64 |
| RealDOF, Input **publicado** | 0.5241 | 0.2865 | 0.3562 | 0.2213 | 28.71 |
| DPDD, oficial (nosso) | 0.1537 | 0.0779 | 0.6481 | 0.4360 | 65.08 |
| DPDD, **publicado** | 0.1440 | 0.0772 | 0.4755 | 0.3452 | 49.41 |
| DPDD, Input (nosso) | 0.3752 | 0.1853 | 0.5303 | 0.2708 | 54.98 |
| DPDD, Input **publicado** | 0.3485 | 0.1827 | 0.4337 | 0.3325 | 45.54 |

**A reproducao fecha nas metricas de referencia e nao fecha nas sem
referencia.** LPIPS e DISTS batem em tudo: no RealDOF o DISTS da linha `Input`
da 0.2865 nos dois lados, digito a digito. Ja CLIP-IQA, MANIQA e MUSIQ ficam
sistematicamente deslocados, e o deslocamento aparece **tambem na linha
`Input`**, que nao passa por modelo nenhum. Se a mesma imagem, sem tratamento,
pontua diferente dos dois lados, a diferenca nao e do nosso modelo: e de
preprocessamento, provavelmente escala ou recorte, e metrica sem referencia e
sensivel a resolucao.

Isso vale um teste barato, de CPU/GPU leve: recalcular MANIQA e MUSIQ da linha
`Input` na resolucao original, sem redimensionar, e ver se encosta no
publicado. Se encostar, esta explicado, e passamos a reportar as tres
sem-referencia so com a ressalva de escala.

## 2. Tab. 6: falta exatamente UMA célula

A ablação de dados do paper cruza três fontes: (a) sintético, (b) ITW, (c)
LFDOF + RealBokeh. As nossas rotas batem uma a uma com essas três, o que dá
para confirmar pelo `kfix`, que usa a Eq. 3 do paper, e a Eq. 3 é justamente a
fórmula do K para a rota ITW.

| Célula do paper | Nosso experimento | Status |
|---|---|---|
| (a) | `fase1, so sintetico` | medido |
| (a) + (b) | ausente | **falta** |
| (a) + (c) | `so rota c` | medido |
| (a) + (b) + (c) | `nosso original, fase 2` | medido |

Ou seja: um treino a mais, o de sintético + ITW sem LFDOF/RealBokeh, e a
Tab. 6 fica reproduzida na íntegra, nas nossas mesas.

Vale registrar o inverso também: o nosso braço `sem filtro de SSIM` **não** é
reprodução, é experimento novo. Ele ablaciona o limiar de SSIM da Eq. 5, que o
paper usa para aceitar ou descartar supervisão, e que o paper nunca ablaciona.

## 3. Os dois "furos de protocolo" NAO existem (conferido no codigo)

Na primeira leitura do paper eu apontei duas diferencas de protocolo. Fui ao
codigo depois e as duas estao erradas. Fica o registro para ninguem gastar GPU
com isso:

**A busca binaria por K ja e feita.** O `inference/src/pipelines/bokeh_net.py`
implementa a bisseccao por imagem escolhendo o K que maximiza SSIM contra o
alvo (linhas 485-508), exatamente a Sec. 4.1 do paper. A imagem vencedora vai
para a coluna `image_best_k`, e o `metricas_por_imagem.py` mede fidelidade
sobre ela. Nao rodamos com K fixo.

**O LVCorr ja e o do paper.** O mesmo pipeline grava o sweep `image_k01`,
`image_k05`, `image_k10`, `image_k15` em todo repo de saida, e o
`metricas_por_imagem.py` calcula Pearson entre os K e a variancia do Laplaciano
da IMAGEM INTEIRA desse sweep. E a definicao da Sec. 4.1. A anotacao do
`REGISTRO.md` sobre "variancia do Laplaciano no fundo" descreve a metrica
riemanniana espacial do `metricas_riemannianas.py`, que e outra coisa, com
outro proposito.

O que **sobra** de duvida real e so o **sinal**. O paper publica LVCorr
+0,9368 para o GenRefocus, e o nosso Pearson cru da negativo quando o modelo
obedece o K (mais K, mais borrado, menos variancia do Laplaciano). Por isso a
coluna `LVCorr_convencao_paper` guarda o simetrico. Se a definicao deles for
Pearson cru, o sinal publicado implicaria variancia CRESCENDO com o K, o que
nao fecha fisicamente. Isso e uma linha de e-mail para os autores, nao um
experimento.

**O que isso muda no custo**: o sweep de K ja esta gravado em todos os repos de
saida da campanha. Qualquer reanalise de controlabilidade custa CPU, nao GPU.

## 4. Tab. 4 e Tab. 5: caras, e nesta ordem

- **Tab. 4 (LF-Refocus)**: 400 pares origem-alvo tirados do mesmo material de
  light field do LF-Bokeh. Como já reconstruímos o LF-Bokeh, a mesma fonte
  serve; falta definir os pares. É a tabela que ainda não tem nenhum análogo
  nosso, mas é trabalho de dataset antes de ser trabalho de GPU.
- **Tab. 5 (refocusing direto vs. duas etapas)**: exige treinar um baseline de
  refocusing direto no backbone do BokehNet. É o item mais caro da lista e o
  menos informativo para nós, porque a conclusão dele já é o desenho que
  adotamos.

---

## Baselines externos que dá para rodar de verdade

O paper compara contra métodos que, em parte, são públicos. Hoje as nossas
tabelas só têm dois pontos de referência externos: a linha de identidade e o
FLUX cru. Dá para melhorar isso sem treinar nada:

- **Bokehlicious** (ICCV 2025 Highlight, Seizinger et al.): código e pesos
  públicos em `github.com/TimSeizinger/Bokehlicious`, com `predict.py` que
  aceita um f-stop de 2.0 a 20.0 como controle. É a linha `Bokehlicious` das
  Tabs. 2 e 3 do paper, e roda direto nas nossas mesas.
- **RealBokeh**: o dataset original está público em
  `timseizinger/RealBokeh_3MP` (23 mil imagens, 24 MP, Canon EOS R6 II). Serve
  para checar a procedência da nossa `bokeh-bench-realbokeh-test-v2`.
- **NTIRE 2026, Controllable Bokeh Rendering Challenge** (arXiv 2605.05510):
  44 inscritos, 8 times com solução válida, quase todos partindo do
  Bokehlicious, com pista quantitativa e estudo qualitativo com especialistas.
  É o conjunto de pontos de comparação externos mais recente da área.

---

## Ordem que eu recomendo

Depois de conferir o codigo e as tabelas ja medidas, a lista de GPU encolheu
bastante: quase tudo o que eu tinha listado como "reproduzir" ja estava feito.
O que sobra, em ordem:

1. **Estudo de sensibilidade ao `k_escala`** (avaliacao, nao ablacao de modelo).
   Toda a campanha fixou `k_escala=3.0`, e esse fator multiplica a FAIXA da
   bisseccao de K. O proprio comentario do `bokeh_net.py` registra que, com
   `k_escala=0.01`, os tres modelos escolheram o K no piso da faixa e a
   comparacao ficou enviesada. Ou seja: a escolha do fator pode mexer no
   ranking, e isso e um risco em cima de numeros que ja publicamos. Medir com
   1.0 e 5.0 responde. Nao precisa de codigo novo, so de linhas de fila.
2. **Bokehlicious como baseline externo.** Hoje as tabelas so tem identidade e
   FLUX cru como referencia de fora. Codigo e pesos publicos.
3. **Refocusing ponta a ponta (analogo da Tab. 4).** Hoje avaliamos o BokehNet
   recebendo a imagem NITIDA e o DeblurNet separado. O paper avalia a cadeia
   inteira a partir da imagem borrada. E a avaliacao que falta de verdade, e a
   mais cara.
4. **MANIQA/MUSIQ em resolucao original**, para fechar a leitura da Tab. 2.
5. Treino (a+b) da Tab. 6 e, por ultimo, a Tab. 5. Os dois sao treino, entao
   ficam fora da fila de avaliacao.
