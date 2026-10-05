# Tabela 2 reproduzida — resultados medidos

Documento novo, de 2026-09-13. **Não substitui** os model cards do HF, que
descrevem cada peso isoladamente. Aqui está a comparação entre todos.

Todas as linhas `medido` saíram do **mesmo pipeline**, no mesmo dia:
resolução nativa (`long_side=0`), `guidance_scale=3.5`, `steps=28`,
`main_adapter=None`, seed 42, zero falhas em 125 imagens por modelo.
`publicado` é a Tabela 2 do arXiv 2512.16923v3.

## DPDD — `akcit-pixel/DDPD:test`, 75 imagens

| modelo | origem | LPIPS ↓ | DISTS ↓ |
|---|---|---|---|
| Input (identidade) | publicado | 0.3485 | 0.1827 |
| Input (identidade) | medido | 0.3440 | 0.1830 |
| GenRefocus (autores) | publicado | **0.1440** | **0.0772** |
| oficial `nycu-cplab` | medido | **0.1596** | **0.0844** |
| nosso final (60k) | medido | **0.1744** | **0.1021** |
| nosso best (15.5k) | medido | 0.1942 | 0.1055 |
| antigo main+cond (60k) | medido | 0.1786 | 0.1132 |
| ab15k rank64 (15k) | medido | **0.1978** | **0.1063** |

## RealDOF — `akcit-pixel/RealDOF:validation`, 50 imagens

| modelo | origem | LPIPS ↓ | DISTS ↓ |
|---|---|---|---|
| Input (identidade) | publicado | 0.5241 | 0.2865 |
| Input (identidade) | medido | 0.5280 | 0.2865 |
| GenRefocus (autores) | publicado | 0.2408 | 0.1126 |
| oficial `nycu-cplab` | medido | 0.2397 | 0.1153 |
| **nosso final (60k)** | medido | **0.2291** | **0.1089** |
| nosso best (15.5k) | medido | 0.2410 | 0.1104 |
| antigo main+cond (60k) | medido | 0.2497 | 0.1151 |
| ab15k rank64 (15k) | pendente | — | — |

---

## O que se sustenta

### 1. O protocolo é válido — provado por duas âncoras independentes

A linha **`Input`** não depende de modelo nenhum, só do dado e da métrica. Ela
reproduz o publicado: LPIPS a 0,7–1,3%, DISTS **idêntico até a 4ª casa** no
RealDOF. Se o protocolo estivesse errado, essa linha erraria.

A linha **`oficial`** é a segunda âncora: no RealDOF ela dá 0.2397 contra
0.2408 publicado — 0,5%. Mesmo modelo, mesma escala.

### 2. Mas há uma anomalia localizada no DPDD

O mesmo peso oficial dá **0.1596** no DPDD contra **0.1440** publicado — 11%.
Como o RealDOF bate, **não é o pipeline nem a métrica**: é algo específico do
DPDD. Candidatos não confirmados: o repo tem 75 imagens e o canônico tem 76; ou
o paper avalia num recorte diferente. 1 imagem em 76 não explica 11%.

**Consequência prática:** comparações no RealDOF podem ser lidas contra o
publicado. No DPDD, só contra a linha `oficial` medida aqui.

### 3. O placar, na mesma régua

| mesa | nosso final vs oficial |
|---|---|
| DPDD | **perdemos** — LPIPS +9,3%, DISTS +21% |
| RealDOF | **ganhamos** — LPIPS −4,4%, DISTS −5,6% |

No RealDOF o nosso final também bate o **publicado** (0.2291 vs 0.2408).

### 4. A `val_loss` escolheu o checkpoint errado

A validação em treino usa a loss de flow matching com σ e ruído fixos:

```
step 15.500   val 0.2703   ← mínimo
step 60.000   val 0.2815   (+4,1%)
```

Parecia overfit. Medido:

```
DPDD    LPIPS  15.5k → 0.1942   60k → 0.1744   (60k 10% melhor)
RealDOF LPIPS  15.5k → 0.2410   60k → 0.2291   (60k  4,9% melhor)
```

**O final ganha nas duas mesas.** A loss de denoise não é proxy de qualidade
perceptual neste regime. Não era overfit — era a métrica de seleção medindo
outra coisa. Quem usar `val_loss` para early stopping em difusão deveria ver
isto.

### 5. Capacidade não explica a diferença no DPDD

Comparação estrutural dos arquivos:

```
oficial   686 tensores | BF16 | rank  64 |  464 MB
nosso     686 tensores | F32  | rank 128 | 1855 MB
chaves em comum: 686    exclusivas: 0
```

Mesmas chaves, mesmos 343 módulos — e **o dobro do rank**.

Atenção ao que isso permite concluir, porque é mais estreito do que parece:

- **Descarta**: "perdemos porque o modelo é pequeno demais". Temos o dobro da
  capacidade e perdemos no DPDD.
- **NÃO descarta** capacidade como fator. Rank alto sobre dados pouco diversos
  *piora* — é overfit. Se nossas 3.000 linhas são ~580 cenas distintas, rank 128
  sobre isso pode estar atrapalhando, enquanto rank 64 sobre 3.000 cenas de
  verdade generalizaria melhor.

As duas hipóteses não competem: **rank alto e dados pouco diversos podem ser a
mesma explicação**. O experimento que separa: treinar com `top_k_mode: scene`
(3.000 cenas distintas) e/ou rank 64, e medir.

Sobre o rank, vale registrar que **o paper contradiz a si mesmo**: o texto
(§4.1) diz *"DeblurNet employs LoRA rank r=128"*, e o peso que os autores
publicaram tem rank **64**. Seguimos o texto. Não é possível estar igual aos
dois.

---

## O que NÃO está aqui, e por quê

**CLIP-IQA, MANIQA e MUSIQ.** A linha `Input` mostrou que nossas variantes
(`clipiqa+`, `maniqa-kadid`) erram 14–39% contra as publicadas. Comparar seria
enganoso. Elas continuam válidas *entre* nossas próprias linhas, e estão nos
JSONs por mesa.

Curiosidade que reforça o item 2: no RealDOF as três do `oficial` ficam
**perto** das publicadas (CLIP-IQA 0.4987 vs 0.4595, MUSIQ 42.47 vs 43.52); no
DPDD, longe. A anomalia é da mesa.

---

## Procedência

Cada linha tem `protocolo_e_resultado.json` e `por_imagem.json` em
`saidas/tab2_*/<mesa>/<linha>/`, registrando resolução da métrica, se o GT
passou por resize, variante de cada métrica, e o sha256 do peso.

**Sem vazamento**, verificado por interseção de `file_name_base`:

```
DDPD train ∩ DDPD test           = 0
DDPD train ∩ DDPD validation     = 0
DDPD validation ∩ DDPD test      = 0
RealBokeh(train) ∩ RealDOF(eval) = 0
DDPD(train) ∩ RealDOF(eval)      = 0
```

| peso | sha256 | onde |
|---|---|---|
| final 60k | `a1ed05e48f7a3469…` | `juliadollis/genrefocus-deblurnet` |
| best 15.5k | `a941d6b5906c78fe…` | `juliadollis/genrefocus-deblurnet-15k` |
| oficial | — | `nycu-cplab/Genfocus-Model` |
| antigo main+cond | — | `juliadollis/genrefocus-deblurnet-paper-4gpu` |


---

## Onde paramos — 2026-09-16

### A/B de 15.000 steps na h100n3 (jobs encadeados)

| braço | muda | estado |
|---|---|---|
| `rank64` (32503) | `deblur_lora_rank: 64` | ✅ 15.000 steps, `val` 0.2747, best no step 14.000 |
| `controle` (32504) | nada (rank 128, `row`) | 🔄 step 8.790/15.000, `val` 0.2741 — fecha ~17/09 |
| `scene` (32505) | `top_k_mode: scene` | ⏳ na fila, dispara sozinho — fecha ~19/09 |

Entre os três muda **exatamente uma linha** de config. Mesmos 15.000 steps,
batch efetivo 32, seed 42, lr 1e-4, warmup 500, cosseno, dados, guidance 3.5,
cond-only, sem GC.

### Sinal preliminar, NÃO conclusivo

No step 8.790 o controle (rank 128) está com `val` **0.2741**, melhor do que o
rank64 alcançou no fim (**0.2747** aos 15.000), com 6.200 steps de sobra.
Sugere que **rank 64 não ajuda** — mas a `val` já errou nesta reprodução (apontou
o checkpoint de 15,5k como melhor e o benchmark deu o de 60k 10% à frente em
LPIPS). **Quem decide é a Tabela 2.**

### Pendências

1. **RealDOF do rank64** — o DPDD fechou (0.1978 / 0.1063), o RealDOF parou em
   4/50 quando a GPU foi liberada. Retomar com `--mesa realdof` (~80 min) para
   não refazer o DPDD.
2. **Avaliar `controle` e `scene`** quando os treinos fecharem.
3. **Subir os dois para `akcit-h100n1`** — os outros já estão lá.

### Cuidado ao comparar

O rank64 (15.000 steps) **não** se compara ao final de 60k (0.1744): schedules de
LR diferentes. A comparação válida é `rank64` vs `controle`, ambos 15.000 steps.

### Modelos publicados

| repo | conteúdo |
|---|---|
| `akcit-h100n1/genrefocus-deblurnet` | final 60k + best 15.5k |
| `akcit-h100n1/genrefocus-deblurnet-ab15k-rank64` | final 15k + best 14k |
| `juliadollis/genrefocus-deblurnet` e `-15k` | os mesmos do 60k |
