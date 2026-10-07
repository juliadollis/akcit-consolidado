# Adequações — rodada pós-experimento do termo de escala

O experimento do termo de escala **não confirmou** a hipótese 4, e a revisão expôs dois
erros de desenho meus. Esta rodada corrige o que estava errado, remove o que não funcionou
e reorienta o objetivo. Nada aqui é "a solução" — são correções de higiene e um repivot.

---

## 1. O experimento anterior era INOBSERVÁVEL (erro meu)

**Crítica:** com `normalize_terms=True`, cada termo é dividido pela sua própria EMA, então
o total fica ancorado perto de 1.0 por construção. Os números do teste (`train_berhu`
A=1.0493 vs B=1.0446) não dizem nada sobre "a loss desceu?".

**Procede.** Medi: com a EMA acompanhando o regime, uma queda real de 2.8× no erro aparecia
como 1.5× no total — e, pior, cada run normaliza pela **própria** EMA, então A e B **não são
comparáveis** em valor absoluto. Desenhei um experimento cuja pergunta o próprio código não
conseguia responder — e a normalização que causa isso fui eu que escrevi.

**Correções:**
- **Valores CRUS sempre expostos.** Todo forward da loss agora devolve `raw_<termo>` e
  `raw_total` além dos normalizados; o log de época imprime `train_raw=` ao lado de
  `train_total=`. Para julgar progresso, use SEMPRE os crus.
- **Congelamento da normalização** (`norm_freeze_steps`, padrão 200): as escalas são
  estimadas nos primeiros passos e depois **congeladas**. Os termos seguem balanceados, mas
  o total volta a ser sinal legítimo e comparável entre épocas/runs. Medido: sem congelar,
  0.84→0.30 no erro real virava 1.00→0.67 no total; com congelar, 0.88→0.31 vira 1.00→0.43.
- Regressão em `test_geometry.py::test_observability`.

## 2. O termo de escala era logicamente incompatível com a avaliação (erro meu)

**Crítica:** a avaliação usa alinhamento afim (invariante a escala). Um termo que otimiza
escala **absoluta** não pode, por definição, melhorar o AbsRel reportado — e empiricamente
piorou bordas feio (F 0.645→0.363, recall 0.699→0.287), puxando para predições mais chapadas.

**Procede, e é decisivo.** Propus recuperar "o sinal que o afim descarta" sem notar que a
**métrica descarta o mesmo sinal**. Não havia razão para esperar ganho no alvo; havia razão
para esperar dano em bordas. Falha de raciocínio minha, não do experimento.

**Correção:** o termo `scale` fica **desativado por padrão**, marcado como NÃO RECOMENDADO
no `--help`, e **removido do espaço de busca do Optuna**. Permanece no código apenas como
instrumento de diagnóstico.

## 3. Learning rate: o suspeito mais provável

**Crítica:** o default de `train_single` era `2e-4` — alto. Degradar 0.11→0.21 em 1 época com
LR alto é esperado.

**Procede.** A varredura da própria B200 mostrava 1e-5 como o melhor (0.1150 vs baseline
0.1098) e 5e-5 já bem pior (0.1571) — e mesmo assim a faixa do Optuna estava em **5e-5 a
5e-4**, inteiramente na região ruim.

**Correções:** default de `--lr` passa a **1e-5** em `train_single` e `run_ablation`; a faixa
do Optuna passa a **1e-6 – 3e-5**, condizente com a evidência.

## 4. Repivot para boundary F-score

**Recomendação acatada.** O DepthPro já faz AbsRel ~0.11 zero-shot no Hypersim: quase não há
headroom, e com validação em cenas disjuntas qualquer especialização vira regressão. Bordas
é onde o método promete (a curvatura é sobre forma local) e onde há espaço (F ~0.645, e o
`gauss` teve o melhor F no smoke: 0.8435).

**Correção:** `--monitor` novo nos três scripts, **padrão `boundary_fscore`** — define o
`best.pt` e o early-stop. `abs_rel` continua disponível para quem quiser precisão métrica.
O Optuna já era multiobjetivo (AbsRel + boundary), então segue coerente.

---

## 5. O que rodar agora (ordem sugerida pela revisão)

1. **Baseline com LR baixo, antes de qualquer outra coisa** — testa a causa mais provável:
   ```
   python scripts/train_single.py ... --berhu 1.0 --lr 1e-5 --epochs 1 --seeds 1
   ```
   Olhar `train_raw` (não `train_total`) e comparar val com o zero-shot (~0.11 AbsRel /
   ~0.645 bF). Se a degradação sumir, era LR.

2. **Ver o "desce?" de verdade** — com os crus expostos isso já funciona por padrão; se
   quiser o sinal totalmente sem normalização, rode com `normalize_terms=False`
   (via config) para comparar.

3. **Se o baseline com LR baixo estabilizar**, seguir para a ablação com `--monitor
   boundary_fscore` e avaliar se os termos de 2ª ordem (gauss/normal) ganham em bordas —
   que é a hipótese central do projeto.

---

## 6. Ressalva honesta sobre o alcance destas mudanças

Nada aqui foi validado com o DepthPro real. O que está validado (dummy + testes numéricos):
os crus aparecem, o congelamento restaura a observabilidade, o monitor de boundary funciona,
os defaults novos parseiam e o pipeline roda sem NaN.

O que **não** está resolvido e pode não ter solução via código: se o gargalo for headroom
zero-shot (hip. 3 do report anterior), nenhuma mudança de loss resolve — a saída é
reposicionar a pergunta científica (bordas em vez de AbsRel, que é o que o item 4 faz) ou
mudar o protocolo (mais cenas, ou avaliar em domínio onde o DepthPro seja mais fraco).
Duas propostas minhas seguidas falharam nesta frente; o próximo passo deve ser guiado pelo
baseline com LR baixo, não por mais uma mudança especulativa na loss.

---

# Rodada: estatística correta e artefatos visuais

## 6. Intervalos de confiança: t de Student no nível-semente

**Problema:** o `bootstrap_ci` era usado tanto para itens quanto para sementes. Com n=3,
o bootstrap percentil não consegue ultrapassar o menor e o maior valor da amostra, e
entrega cobertura real de cerca de 75%, não os 95% anunciados. Isso fez um ganho de borda
de +0.005 parecer significativo quando não estava.

**Correção:** `riemann/repro.py` ganhou `mean_ci`, que usa t de Student (com n=3 o
multiplicador é 4.303, não 1.96). O `bootstrap_ci` continua disponível para amostras
grandes e agora emite aviso quando n < 10. A função `resumo_sementes` imprime os dois lado
a lado, deixando explícita a diferença.

Regra adotada: variação entre **sementes** usa `mean_ci`; variação entre **itens/cenas**
pode usar qualquer um dos dois; comparação contra baseline usa **sempre** `compare_paired`.

## 7. Comparação emparelhada por cena

**Problema:** confrontar dois intervalos independentes é a análise mais fraca possível
aqui. A variância entre cenas (cena fácil vs cena difícil) é muito maior que o efeito, e
mascara ganhos reais. Além disso, imagens de uma mesma cena são fortemente correlacionadas,
então tratar 300 imagens de 4 cenas como 300 amostras independentes produz intervalos
otimistas: o tamanho efetivo de amostra é o número de CENAS.

**Correção:** `compare_paired` calcula a diferença modelo menos baseline na MESMA cena,
e reporta ganho médio, IC da diferença, teste t pareado, Wilcoxon e taxa de vitórias.
Em simulação com ganho real de +0.005, os ICs independentes se sobrepõem (concluiriam
"inconclusivo") enquanto o pareado detecta o efeito com 100% de vitórias.

Novo script `scripts/evaluate_paired.py` executa a avaliação final completa: mede o
zero-shot e cada semente, salva métricas por imagem com a coluna de cena, e roda as duas
análises (t entre sementes e pareada por cena).

## 8. Mapas de profundidade e curvatura

**Problema:** os resultados foram apresentados sem os artefatos visuais.

**Correção:** novo script `scripts/make_figures.py`, que gera painéis comparativos
(RGB, GT, zero-shot, treinado, curvatura Gaussiana), mapas individuais em PNG colorizado e
`.npy` cru, e uma figura resumo com várias cenas empilhadas, em PNG e PDF.

Decisões que importam para a honestidade da figura: a predição é alinhada ao GT pela mesma
transformação afim usada nas métricas, para que a cor não varie por escala; zero-shot e
treinado compartilham a mesma escala de cor dentro de cada linha; a curvatura usa colormap
divergente com escala simétrica em torno de zero, porque o sinal tem significado geométrico
(elíptico versus hiperbólico), com limite por percentil robusto para um outlier não achatar
o mapa. A opção `--zoom` recorta e amplia uma região de borda, marcando o recorte na
imagem cheia.

---

# Rodada: artefato na métrica de borda

## 9. O limiar de borda era normalizado pelo MÁXIMO (frágil)

**Problema encontrado ao analisar os resultados do teste.** O detector de bordas usava
limiar relativo ao **gradiente máximo da imagem**. Isso torna a métrica refém de um único
pixel: um outlier de gradiente eleva a referência, o limiar sobe, e quase nenhuma borda é
detectada.

Demonstração controlada: uma predição **idêntica ao ground truth**, acrescida de um único
pixel outlier, caiu de F=1.000 para **F=0.596**, com precisão 0.98 e recall 0.43.

**Por que isso é grave aqui:** essa é exatamente a assinatura observada no teste para o
`heads_final` (precisão sobe, recall despenca, F piora). A conclusão de "piora a borda"
pode ser artefato da normalização, e não achatamento real da geometria.

**Correções em `riemann/metrics.py`:**

1. `_depth_edges` passa a usar limiar por **percentil** (padrão 99) em vez do máximo. No
   mesmo teste, o F com outlier vai de 0.596 para **0.996**. O modo `"max"` continua
   disponível apenas para reproduzir números antigos.
2. Nova função `boundary_fscore_sweep`: varre o limiar e reporta **F-max**, o melhor F
   alcançável por cada modelo, mais a precisão e o recall nesse ponto e a média ao longo
   da curva (`boundary_f_auc`). Comparar dois modelos num único limiar confunde qualidade
   de borda com agressividade de detecção; o F-max é o ponto de operação ótimo de cada um
   e portanto a comparação justa.
3. `all_metrics` e `scripts/evaluate_paired.py` já incluem as novas chaves.

**Consequência prática:** os números de borda do teste precisam ser **recalculados** antes
de qualquer reporte. O resultado de `abs_rel`, `rmse` e `d1` não é afetado.

## 10. Mapas de curvatura dominados por ruído (visualização)

**Problema.** Os mapas de curvatura gerados no passo 5 estão ilegíveis: medindo o PNG,
**52% dos pixels aparecem saturados** e a energia de alta frequência é altíssima. Uma
parede plana, cuja curvatura verdadeira é zero, aparece coberta de pontos vermelhos e
azuis.

**Duas causas somadas, ambas de parâmetro de visualização:**

1. `smooth_sigma=0.5` é pequeno demais. A segunda derivada amplifica ruído por 1/h², que
   a 384 px vale ~150 mil. Medido em cena sintética com 4 mm de ruído: numa parede plana o
   ruído de curvatura fica em ~18000 com sigma=0.5, e cai para ~50 com sigma=2.0. A razão
   sinal/ruído contra a borda real sobe de 3 para 10.
2. `clamp_val=50` está muito abaixo do sinal real. Na mesma cena, a borda verdadeira
   produz curvatura de ~530. Com teto em 50, sinal e ruído saturam no mesmo valor e a
   figura perde toda a discriminação.

**A distinção que faltava:** o clamp existe para a LOSS, onde limita o gradiente e evita
divergência. Na VISUALIZAÇÃO ele destrói o mapa, porque a escala de cor já é definida por
percentil robusto sobre os dados. São usos diferentes e precisam de parâmetros diferentes.

**Correção.** Nos dois scripts de figura, `--curv-sigma` passa a 2.0 e `--curv-clamp` a 0,
que agora significa "sem clamp". Efeito medido na mesma cena: a fração de pixels saturados
cai de 95% para 2%, a parede plana fica branca (curvatura zero, como deve ser) e as bordas
aparecem como linhas finas nítidas. A loss segue com o clamp de 50, inalterada.

Observação geométrica que ajuda a ler o mapa: na forma de Monge a curvatura é dividida por
(1+|∇z|²)², então superfícies inclinadas têm o ruído naturalmente atenuado, enquanto
regiões frontoparalelas mostram mais granulação. Isso é propriedade da parametrização, não
defeito da predição.

## 11. Elemento de área riemanniano e painel de apresentação

**Motivação.** A curvatura Gaussiana é de segunda ordem e, mesmo com a suavização
corrigida no item 10, produz mapas granulados quando a profundidade predita tem ruído.
Para comunicar "onde estão as bordas 3D" existe um mapa muito mais estável, e que já havia
sido usado em trabalho anterior do grupo: o **elemento de área riemanniano**.

**Definição.** Com g_ij = δ_ij + ∂_i D · ∂_j D (primeira forma fundamental de uma
superfície de Monge), o elemento de área é √det(g) = √(1 + |∇D|²). Ele é o fator que
converte área no plano da imagem em área sobre a superfície em R³: vale exatamente 1 em
regiões frontoparalelas e cresce onde a superfície se inclina ou quebra.

**Por que é mais limpo:** é de PRIMEIRA ordem, então o ruído entra linearmente, em vez de
ser amplificado por 1/h² como na curvatura. Não precisa de suavização pesada.

**O que entrou:**
- `riemann/geometry_maps.py`: função `area_element`, incluída em `compute_all` e no
  `VIS_INFO` com colormap `hot`.
- `scripts/make_riemannian_figure.py`: painel de apresentação em tema escuro, 2×3 —
  RGB, profundidade e √det(g) na primeira linha; RGB⊕profundidade, RGB⊕métrica e vetores
  normais na segunda. Aceita `--modo curvatura` para trocar √det(g) por K.

**Três detalhes de renderização que fazem a figura funcionar:**
1. **Subtrair a linha de base 1.** Em superfície frontoparalela √det(g) vale exatamente 1;
   usar (√det(g) − 1) coloca o plano em zero, que no colormap `hot` é preto.
2. **Percentil inferior alto** (`--piso`, padrão 60). A maior parte da cena é superfície
   suave, então isso joga o piso de ruído para o preto em vez de deixá-lo avermelhado.
3. **Mistura *screen*** na sobreposição, 1 − (1−base)(1−brilho). Diferente da mistura
   linear, ela só clareia: as regiões escuras do mapa deixam a foto intacta e apenas as
   bordas 3D acendem sobre ela.

Se numa cena real o fundo ainda ficar avermelhado, suba `--piso` (ex.: 80); se as bordas
sumirem, baixe. O `--gama` (padrão 1.8) controla o contraste entre piso e bordas.

## 12. Zero-shot nas figuras e tabela geral

Dois pedidos da liderança, ambos com a mesma motivação: sem a referência do zero-shot não
é possível julgar se os modelos treinados estão de fato ganhando.

**Sinais geométricos do zero-shot.** O `visual_signals.py` já aceitava omitir `--weights`
para usar o modelo base; o que faltava era isso estar documentado e a garantia de que as
cenas coincidem entre as execuções. Como os índices são escolhidos de forma determinística
por espaçamento, basta usar o mesmo `--n-imagens` nas três chamadas (zero-shot, `heads`,
`heads_final`) para as cenas baterem.

**Modo comparativo no painel riemanniano.** Nova flag `--comparar` em
`make_riemannian_figure.py`: zero-shot na linha de cima, treinado na de baixo, mesmas
colunas. O ponto crítico é que as **escalas de cor são compartilhadas entre as linhas** —
tanto a faixa de profundidade quanto os limites do mapa de área ou de curvatura são
calculados sobre os dois modelos juntos. Sem isso cada painel se auto-normalizaria e a
diferença visual viria da normalização, não da geometria. Cada linha traz também AbsRel e
bF-max daquele modelo no título, para a figura ser autoexplicativa.

**Tabela geral.** Novo `scripts/make_results_table.py`, que lê os
`comparacao_pareada.json` de cada variante e produz `tabela_geral.md`, `.csv` e `.txt`.
São duas tabelas: valores absolutos com o zero-shot como coluna de referência, e ganho
contra o zero-shot com intervalo da diferença, os dois p-valores, taxa de vitórias e
veredito. O veredito exige que t pareado e Wilcoxon concordem.

**Runbook.** `RUNBOOK.md` reúne os cinco passos em ordem, com os comandos completos, o que
enviar de volta e uma lista de verificação. Nenhum passo exige treinar novamente.
