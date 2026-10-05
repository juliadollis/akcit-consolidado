# Condicionamento geométrico na BokehNet — campanha v2

> Do zero, sobre o contrato `metric_disparity_official_v1`. Nada da campanha de
> 03–09/09/2026 é reaproveitado como número: ver `referencia_antiga/LEIA.md`.
>
> Base: `../retreinar-bokeh`, a árvore com as 13 correções do treino.
> Alvo: BokehNet apenas. Objetivo: publicação.
> Criado em 2026-09-22.

---

## 0. A pergunta, e o que faria ela morrer

**Pergunta.** Fornecer à BokehNet sinais geométricos derivados da profundidade —
e concentrar a supervisão nas descontinuidades — reduz o vazamento de cor
através de bordas de profundidade, sem degradar o resto da imagem?

**O argumento que sustenta.** Os sinais não são informação nova. São a forma
explícita dos termos que aparecem na expansão do operador de desfoque: a oclusão
é o termo de visibilidade `V(x,y)`; o elemento de área e as normais são o termo
de primeira ordem, que governa a anisotropia do núcleo; a curvatura é o de
segunda ordem. Fornecê-los prontos tira da rede o encargo de reconstruir por
regressão algo que se calcula em fechado.

**O que faria a campanha morrer, e cedo.** Três coisas, nesta ordem:

1. **A fase 0 mostrar que não há espaço.** Se o modelo retreinado já errar na
   borda quase tanto quanto fora dela, não há artefato a consertar. Medimos isso
   ANTES de escrever qualquer treino.
2. **O controle B′ empatar com o B.** Se trocar os canais geométricos por ruído
   de mesma estatística produzir o mesmo ganho, o ganho veio de capacidade e de
   mais tokens, não de informação geométrica. Isso não sustenta publicação.
3. **A perda ponderada repetir o resultado de setembro.** Ela já piorou o modelo
   uma vez, e o motivo está medido. Se a correção não resolver, ela sai.

Nenhuma dessas é hipotética: as três têm instrumento definido abaixo, e o limiar
de cada uma é declarado **antes** de olhar o resultado.

---

## 1. O que mudou, e por que recomeçar era obrigatório

### 1.1 A campanha anterior mediu contra um modelo quebrado

O pré-processamento antigo montava o mapa de defocus em profundidade linear
normalizada por imagem; a inferência oficial monta em disparidade métrica
absoluta. Não é diferença de escala — nenhuma constante em `K` converte uma na
outra (erro relativo residual 0,926, Pearson 0,341, razão do CoC 4408× a 1,5 m
e 0,66× a 10 km). O condicionamento geométrico modula exatamente esse sinal.

Consequência: a tabela de `E_bleed` dos 7 modelos, o veredito do par A/A′ e as 8
constantes calibradas mediram outra coisa. Todos saem.

### 1.2 O dado novo simplifica a metade mais chata

Conferido no Hub hoje (2026-09-22), e o job `confere-rota-a` fechou a contagem:
**210.809 arquivos, 0 faltando, 0 sobrando** contra o manifesto.

| release | amostras | cenas | estado |
|---|---|---|---|
| `bokehnet-regen-rota-a` (sintético, §3.2a) | 69.700 | 1.700 | completo, split 1.619/81 cenas |
| `bokehnet-regen-rota-b-oficial` (~ITW, §3.2b) | 13.761 | 13.761 | completo |
| `bokehnet-regen-rota-c` (LFDOF+RealBokeh, §3.2c) | 15.423 | 4.399 | completo, 4,2% de K censurado |

O PNG de profundidade é **disparidade** `uint16` linear entre `disparity_min` e
`disparity_max`, que vêm **por amostra** no `meta/`. Isso mata, de uma vez:

- o job `f0b_escala_metrica` e a tabela `geo_escalares` — a escala vem nativa;
- a sentinela do `z_max == 10.000,0` (o teto do Depth Pro, que afetava 25,7% da
  rota b antiga) — o release novo não tem o problema;
- a inversão para obter `u`: o canal de profundidade inversa **é o array lido**,
  reescalado. Não é mais uma conta, é uma leitura.

O que **não** vem: `focallength_px`. Conferido campo a campo no `meta/`. A
curvatura retroprojetada exige focal em pixels, então sem ela o canal `K~` sai
neutro e a flag `segunda_ordem_valida` sai `False`. Isso é deliberado — inventar
um `fx` produziria um canal plausível e errado.

### 1.3 O que sobrevive, e por que não é contradição

Três coisas da campanha anterior não são resultado de experimento:

1. **A física.** `‖∇u‖` e não `‖∇D‖`; a decomposição polar de `∇u`; o branch de
   tokens em vez de soma no patch embedding; o agrupamento `G1=[s,n_x,n_y]` /
   `G2=[u,O,K~]` imposto pelo `group_mask`. É leitura de papel e de código.
2. **A curvatura retroprojetada**, verificada contra esfera (`K=1/R²`, erro <2%),
   plano e cilindro. Teste de propriedade fechada, independente de dado.
3. **O diagnóstico da densidade de token.** É geometria do VAE, não do dado, e é
   o achado mais caro do conjunto. Está na seção 4.

Tudo isso é **reimplementado** em `geocond/` e requalificado pelos testes novos.
Nada é importado da quarentena.

---

## 2. Como o pipeline realmente condiciona

Duas premissas da §5.2 da proposta não correspondem ao código, e a correção é
favorável.

**Não existe patch embedding convolucional.** `x_embedder` é `nn.Linear(64→3072)`;
a patchificação já aconteceu no `_pack_latents`. O condicionamento não é soma:
cada condição é um **branch de tokens separado**, e o `S_t = [X_t ; E(I_in)]` do
paper se realiza como concatenação de key/value dentro da atenção.

**Isso é melhor do que a proposta assumia.** Um branch extra custa **zero
parâmetro novo** — o adapter LoRA é compartilhado entre os branches — e o
checkpoint resultante roda na inferência oficial sem alteração. É literalmente o
mecanismo que o próprio paper usa para forma de abertura na §3.3: *"we append its
tokens directly to the unified sequence"*.

O custo real é comprimento de sequência:

| configuração | tokens | atenção rel. |
|---|---|---|
| hoje: texto + main + AIF + defocus | 3.584 | 1,00× |
| + 1 branch geométrico | 4.608 | ~1,65× |
| + 2 branches geométricos | 5.632 | ~2,5× |

Com `gradient_checkpointing: true` a campanha anterior mediu 29,4 GB e
33,8 s/step em 2 GPUs. **Esse número é de bancada, não de dado, então vale como
ordem de grandeza — mas será remedido no smoke desta campanha.**

**Por que o agrupamento não é livre.** O `group_mask` faz cada condição atender
só a si mesma, ao texto e ao branch principal. Dois branches geométricos são
**mutuamente cegos**. O que precisa ser lido junto fica junto: `G1` é o termo de
primeira ordem completo (a decomposição polar de `∇u`), `G2` é escala,
visibilidade e segunda ordem. A combinação acontece no branch principal, que
enxerga tudo.

---

## 3. Os canais

```
G(x) = [ u , O , s , n_x , n_y , K~ ](x)   ∈ [0,1]^6

u   = clip( disp / u_max , 0, 1 )            disp vem do PNG + meta, em 1/m
O   = min( ‖∇u‖ / τ , 1 )                    τ FIXO, calibrado no train set
s   = clip( log√(det g) / s_max , 0, 1 )     det g = 1 + ‖∇u‖²
n_x , n_y                                    normais de Monge, (n+1)/2
K~  = sgn(K)·log(1+|K|/K0) , levado a [0,1]  curvatura RETROPROJETADA
```

Três decisões que precisam ficar registradas:

**`∇` em coordenadas de imagem normalizadas**, `∂/∂(x/L)`. É o que torna `O`,
`s`, `n` invariantes a redimensionamento — o treino roda a 512² e a inferência
do paper usa a resolução original com tiling. A 512² é no-op; a diferença
aparece no tiling.

**`τ` é constante, não percentil por imagem.** Sob crop aleatório o percentil do
recorte não bate com o da imagem inteira, e treino e inferência veriam
normalizações diferentes. Pior: numa cena sem descontinuidade real o percentil é
ruído, e o mapa de oclusão vira ruído saturado de quadro cheio.

**`s`, `n_x`, `n_y` são reparametrização, não complemento.** Numa superfície de
Monge, `n = (−u_x, −u_y, 1)/√(1+‖∇u‖²)` e `s = log√(1+‖∇u‖²)`: os três canais
codificam **dois** graus de liberdade, que são `∇u`. É a decomposição polar —
magnitude e direção. A parametrização é boa e bem condicionada, mas a redação da
proposta ("juntos, determinam completamente o termo de primeira ordem") sugere
complementaridade que não há, e um revisor vai apontar. **Corrigir no texto.**

---

## 4. A perda ponderada — e o erro que não vamos repetir

A proposta pede `L = Σ_x w(x)·‖Î(x) − I(x)‖₁` com `w = 1 + λ_o·O(x)`. Duas
correções, uma de espaço e uma de forma.

### 4.1 Ela vive em espaço de token, não de pixel

O treino é `F.mse_loss` sobre velocidade de flow matching em tokens `(B, N, D)`.
Não existe `Î` em pixel no laço, e decodificar o latente a cada step poria o VAE
no caminho do gradiente por 60K steps. A 512²:

```
512² px → VAE 8× → latente 64×64 → _pack_latents 2× → tokens (B, 1024, 64)
1 token = 1 bloco de 16×16 pixels
```

### 4.2 O achado que decide o desenho

A campanha anterior implementou o peso contínuo e **piorou o modelo**. O
diagnóstico está fechado:

```
fração de PIXELS com O > 0,3 : 0,0121
fração de TOKENS com O > 0   : 1,0000
espalhamento do max-pool 16×  : 82,7×
distribuição de O_token: p50=0,019  p75=0,123  p90=0,639  p99=1,000
```

A borda é **1,2% dos pixels e praticamente 100% dos tokens**. Com peso contínuo
e normalização pela média, os tokens de menor oclusão caíram para 0,816: a
supervisão diminuiu ~18% em 74% dos tokens para financiar o topo. O erro na
borda melhorou em termos **relativos** (razão borda/fora de 1,427 → 1,286) e
piorou em **absoluto** em toda parte (E_fora +31%).

A correção é limiarizar em token, `w = 1 + λ·[O_token > θ]`. E o número que
importa **não é a fração de tokens pesados, é o piso** — a normalização conserva
o orçamento de peso, então concentrar em menos tokens só os torna mais pesados e
afunda o piso. O piso sobe quando se reduz `λ` junto.

> **Ressalva.** Essa tabela de θ/λ veio da distribuição ANTIGA de `O`. Ela
> orienta o desenho — limiar, e o piso como critério — e **não** vira default.
> `geocond/loss_weight.py` não tem default de θ nem de λ; eles saem do
> diagnóstico sobre o release novo, na fase 0.

**Max-pool, não média.** Uma borda de oclusão tem 1–2 px. A média a 16× dilui a
amplitude por ~1/16, e um `λ=3` vira um `λ` efetivo de 0,2. A média fica como
variante de ablação, por flag.

**Ponto de redação.** Reponderar a perda de difusão de forma não uniforme
enviesa o estimador do score. É prática padrão (min-SNR e afins), mas o texto
tem de dizer **reponderação perceptual da supervisão**, nunca "perda ponderada
da verossimilhança".

---

## 5. A métrica E_bleed — e por que ela vem antes de tudo

```
E_bleed = (1/|B_θ|) · Σ_{x ∈ B_θ} ‖Î(x) − I(x)‖₁      B_θ = { x : O(x) > θ }
```

As métricas globais diluem o artefato: a região de borda é uma fração pequena
dos pixels, e uma melhora real aparece na terceira casa decimal do LPIPS. O
próprio documento do Wallisson coloca esta métrica como **pré-condição**, e está
certo.

Duas exigências que a campanha anterior aprendeu na prática:

- **Reportar E_fora junto.** Sem a região complementar, uma melhora na borda pode
  vir de degradar o resto e ninguém vê. Foi exatamente o que aconteceu com o A′.
- **Mesma máscara, mesmo τ do condicionamento.** Senão a métrica mede uma região
  diferente da que a perda supervisiona.

---

## 6. As condições

| cond. | o que é | o que testa | custo |
|---|---|---|---|
| **A** | fase 2 retreinada, sem geo | piso de comparação | zero (é o baseline da reprodução) |
| **A′** | A + perda ponderada **limiarizada** | a perda sozinha ajuda? | 1 GPU |
| **B** | A′ + 2 branches geométricos | **a hipótese** | 1 GPU |
| **B′** | B com **ruído de mesma estatística** no lugar de G | informação, ou só capacidade? | 1 GPU |

Os dois controles são os que a proposta pede na §6, e eles são o que separa "a
informação ajudou" de "mais tokens ajudaram". Como o objetivo é publicação, eles
não são opcionais.

**Todas as condições partem do MESMO LoRA da fase 1** e comparam no **mesmo
step**. A fase 2 do paper é 60K, mas 4 condições × 60K é da ordem de um mês de
GPU. Rodamos **15K steps por condição** e comparamos A no step 15.000 — desvio
declarado. Só a vencedora vai a 60K completos.

O batch efetivo continua **32** em todas: com 1 GPU, `grad_accum = 32`. Mesmo
número de micro-batches por step que 4 GPUs com accum 8, então os checkpoints
seguem comparáveis com o §4.1. Muda só o relógio de parede.

**A curvatura não entra nesta rodada.** Três motivos: falta `fx` no release;
segundas derivadas amplificam ruído por `1/h²` sobre profundidade *estimada*; e
como termo de perda em estimação de profundidade, sobre profundidade de verdade
(Spring), a curvatura gaussiana perdeu para o controle berHu em todas as
métricas. Ela fica plumbada por flag, desligada.

---

## 7. O cronograma, nas duas máquinas

As máquinas são diferentes e isso é aproveitado, não sofrido: a **h100n1 é
Docker sem SLURM** e está com o treino base; a **h100n3 é SLURM + Singularity**
e é a bancada.

```
AGORA ───────────────────────────────────── +52 h ──────────────── +5 dias
h100n1  [ fase 1, 4 GPUs, step 13.390/40.000 ]──►[ fase 2 = condição A, 4 GPUs ]
h100n3  [ F0: staging · constantes · E_bleed · smoke ]──►[ A′ e B, 1 GPU cada ]
                                                              └─► B′ na vaga seguinte
```

A fase 1 está em **step 13.390/40.000**, a ~7 s/step: faltam ~52 h.
Isso é folga, não espera: a fase 0 inteira cabe nela.

**Restrições da n3, medidas hoje:**

| | |
|---|---|
| GPUs gerenciadas pelo SLURM | 3 de 8 (`Gres=gpu:h100:3`) — `--gres=gpu:4` fica PENDING para sempre |
| GPUs livres agora | índices 1 e 2 |
| QOS `onejob` | **2 jobs simultâneos**, e 1 já está ocupado |
| GPU4 | com defeito: selecionar **por UUID**, nunca por índice |
| **cota** | **520G de 500G soft, em grace** — ver seção 9 |

---

## 8. Fases, na ordem em que destrancam

### F0 — o portão *(h100n3, enquanto a fase 1 treina)*

| # | o quê | instrumento |
|---|---|---|
| F0.1 | staging do release na n3: `depth/` e `meta/` integrais (~1 GB), subconjunto de `generated/`+`mask/` por cena | `scripts/stage_release_n3.py` |
| F0.2 | calibrar as constantes de normalização sobre o **conjunto de treino**, com a tabela de percentis impressa | `geocond.jobs.calibrar` |
| F0.3 | auditoria dos canais: distribuição de `O`, fração saturada, e o **máximo absoluto entre amostras** | idem |
| F0.4 | **E_bleed no piso e no teto**: linha de identidade e pesos oficiais do paper | `geocond.jobs.ebleed_bench` |
| F0.5 | diagnóstico do peso: escolher `θ` e `λ` pela tabela piso/fração/contraste | `geocond.loss_weight.diagnostico_peso` |

**F0.4 é o portão.** Se a distância entre o piso e o teto for pequena, não há
espaço para melhora e a campanha para aqui, em dois dias em vez de um mês.

**F0.3 tem um teste que discrimina.** Correlação **não** detecta normalização
por imagem, porque é invariante a escala — foi assim que o mapa de defocus
quebrado passou despercebido. O teste que discrimina é comparar o **máximo
absoluto entre amostras**. Aplicado a cada canal novo.

### F1 — bancada *(h100n3)*

- smoke com geo ligado: 3 steps, loss finita, **VRAM medida** com o branch extra
- ordenação dos tokens com o `_pack_latents` **real** do diffusers
- **paridade treino/inferência**: a pilha geométrica gerada pelo caminho do
  dataloader tem que bater numericamente com a do caminho de inferência, para a
  mesma imagem. É a versão geométrica do que `verify_defocus_encoding.py` já faz
  para o defocus. Treinar num range e inferir noutro foi o defeito nº 1 deste
  projeto.

### F2 — a campanha

A, A′, B, B′ a 15K steps. Só depois que a fase 2 base (= condição A) existir.

---

## 9. Riscos

| risco | avaliação e mitigação |
|---|---|
| **Cota estourada** (520G/500G, grace correndo) | O mais urgente e o mais banal. Precedente: disco cheio matou 3 treinos e custou ~18h de 3 GPUs. Mitigação: staging seletivo (não o release inteiro), `keep_last_n_checkpoints` baixo, e **decisão da Julia** sobre o que liberar — nada é apagado sem perguntar. |
| **Redundância**: a rede derivaria os sinais sozinha | Real, e é o que o controle **B′** testa. Argumento a favor: derivar consome capacidade, e o mapa de defocus já passa pelo VAE a 1/8 da resolução — o gradiente que a rede veria é o de um mapa reconstruído por um autoencoder treinado em imagem natural. |
| **Desalinhamento treino/inferência** | Na inferência a BokehNet consome a saída da DeblurNet: a geometria é estimada sobre imagem **gerada**, enquanto no treino é sobre AIF real. Canais a mais amplificam. Mitigação barata: augmentation leve (ruído, desfoque) sobre a pilha. Vale uma linha no texto de qualquer forma — um revisor vai levantar. |
| **Qualidade herdada da profundidade** | A limitação mais séria, e a proposta a nomeia. A tensão a dizer no texto: a maior fraqueza declarada do paper é confiabilidade da profundidade, e esta proposta aposta mais fichas nela. Não invalida; define onde o risco mora — primeira ordem é robusta, segunda não. |
| **VRAM** | 66,3 GB já em uso na fase 1 sem branch extra. Dois branches exigem checkpointing religado. **Medir no smoke, não descobrir no step 3000.** |
| **Deslocamento de domínio** | As estatísticas dos canais dependem da cena. Constantes calibradas num conjunto podem não valer em outro. Mitigação: fixar a partir do treino e **registrar no config**. |
| **Procedência da rota A** | Ressalva permanente: o release grava `source_sample_id` e sha256, então o join é verificável — isto melhorou em relação ao pipeline antigo, que gravava só um UUID. Ainda assim, qualquer publicação declara a composição. |

---

## 10. Protocolo estatístico

Não precisa ser escrito: existe e é exatamente o que a proposta pede.

| peça | onde |
|---|---|
| t pareado, Wilcoxon, IC bootstrap do delta, flag de concordância | `depth-riemannian/riemann/repro.py:compare_paired` |
| driver por cena, com aviso abaixo de 8 cenas | `scripts/evaluate_paired.py` |
| consolidação por seeds, que se recusa a comparar n diferentes | `scripts/consolida_reteste.py` |

- Comparação **pareada por cena**. As diferenças entre condições serão pequenas,
  e comparar médias com intervalos independentes não terá poder para detectá-las.
- **Limiar de relevância prática declarado antes de olhar.** Critério escolhido
  depois de ver o resultado é racionalização.
- Paramétrico e não paramétrico reportados **juntos**.
- Replicar a vencedora em **3 seeds** antes de afirmar qualquer coisa.
- `upload_hf_repo_base` distinto por condição. Dois treinos que compartilharam o
  repo corromperam a curva e os checkpoints ficaram sem procedência.

---

## 11. Decisões pendentes

1. **O `θ` do E_bleed e o limiar de relevância prática.** Têm de ser declarados
   antes da F0.4 produzir número. São o que decide se a campanha produziu algo
   publicável.
2. **A cota.** 520G de 500G, em grace. O que pode sair da n3?
3. **Três canais ou seis?** O plano acima assume seis (2 branches, ~2,5×). Com
   `K~` neutro por falta de `fx`, o sexto canal é constante — então na prática
   são cinco canais úteis em dois branches. Vale considerar `G = [O, ĝx, ĝy]`
   num branch só (~1,65×), que colapsa as condições B/C/D da proposta numa só.
4. **A rota A entra na fase 2?** O §4.1 lido literalmente diz "(ii) 60K steps on
   real data", e a reprodução seguiu isso. Braços com replay sintético já existem
   como variante declarada (`train_bokeh_fase2_replay.yaml`).

---

## 12. Operação

| regra | motivo |
|---|---|
| Nunca excluir job nem apagar nada no cluster sem perguntar. `--time=7-00:00:00`. | Regra do projeto. |
| h100n1 é Docker sem SLURM. Sempre `--user $(id -u):$(id -g)`, container detached. | Sem o `--user` os arquivos nascem do root: há 54 GB de cache que a usuária não consegue apagar. O detached sobrevive a queda de ssh. |
| Na h100n3, GPU por **UUID**, nunca por índice. | GPU4 com defeito contamina o stdout do `nvidia-smi` e desalinha os índices do CUDA. |
| Nunca sobrescrever `PYTHONUSERBASE`. `PEFT_PIN` proibido. | diffusers 0.37.1 exige peft ≥ 0.17; o pin antigo rebaixava transformers e é a origem de quatro jobs falhos. |
| `rsync` sempre com excludes (`outputs/ third_party/ logs/ wandb/ .git/`). | Um `--delete` sem exclude já apagou checkpoints. |
| Download do Hub é **opt-in explícito**. | 552 GB de egress inexplicado medidos neste projeto. |

---

Derivado do paper GenRefocus (arXiv 2512.16923v3), da proposta *Condicionamento
Geométrico para Refocusing Generativo* (Wallisson Policarpo Teodoro), da
auditoria `AUDITORIA_TREINO_BOKEHNET.md` e da leitura direta de `genfocus_train/`
e `release.py`. Contagens do release e estado do cluster medidos em 2026-09-22.
