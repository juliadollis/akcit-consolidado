# A ablação: qual termo geométrico contribui

Ablação adaptada pelo **Wallisson** e rodada **exatamente como ele entregou**,
trocando só o dataset para o Spring e informando a focal (2585,859, a mediana das
37 sequências).

---

## 1. O resultado

| config | o que é | F-borda | fmax | f_auc | AbsRel | delta1 | RMSE | vs controle |
|---|---|---|---|---|---|---|---|---|
| **B1_berhu+grad** | berHu + grad 0,3 | 0.5314 | 0.6912 | 0.5540 | 0.3341 | 0.5194 | 11.4880 | +0.0302 |
| **B1_berhu+metric** | berHu + metric 0,2 | 0.5287 | 0.6593 | 0.5515 | 0.2639 | 0.6341 | 9.7924 | +0.0276 |
| B1_berhu+geod | berHu + geod 0,1 | 0.5163 | 0.6680 | 0.5472 | 0.2893 | 0.5937 | 9.8114 | +0.0152 |
| B0_berhu | berHu puro, o controle | 0.5011 | 0.6485 | 0.5326 | 0.2645 | 0.6441 | 9.6464 | +0.0000 |
| B7_gaussheavy | gauss com peso alto | 0.4940 | 0.6181 | 0.5087 | 0.2954 | 0.6184 | 9.7177 | -0.0071 |
| ZERO_SHOT_sem_finetune | sem fine-tune | 0.4861 | 0.6201 | 0.5132 | 0.3075 | 0.5990 | 9.7290 | -0.0150 |
| **B1_berhu+normal** | berHu + normal 0,9 | 0.4315 | 0.5579 | 0.4518 | 0.3051 | 0.5989 | 9.6466 | -0.0696 |
| **B1_berhu+gauss** | berHu + gauss 0,45 | 0.4311 | 0.6205 | 0.4669 | 0.2921 | 0.6267 | 10.0792 | -0.0700 |
---

## 2. Como foi rodado

```bash
python scripts/run_ablation.py \
  --train-root /data/spring_split/train \
  --val-root   /data/spring_split/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/ablacao_spring \
  --focal 2585.859
```

Todos os defaults dele preservados: `--filter B0 B1 B7`, `--epochs 30`,
`--batch-size 2 --grad-accum 4 --grad-checkpointing`, `--align-mode detach`,
`--gauss-clamp-metrico 5.0`, `--eval-zero-shot`. O código está intocado em
`ablacao-wallisson/`, incluindo o `metrics.py`.

---

## 3. Por que estes números NÃO se comparam com os nossos

| | ablação | nossa campanha |
|---|---|---|
| split avaliado | **validação** (184 imagens, 6 cenas) | **teste** (485, 13 cenas) |
| máscara de validade no F-borda | **não** (`metrics.py` da branch `fix-geometry`) | **sim** (`origin/main`) |
| `align-mode` | `detach` | `full` |
| épocas | 30 | até 100 com early stop |
| batch | 2 com acumulação 4 | 8 |
| seeds | **1** | 6 |

A prova de que a diferença é de protocolo: o **zero-shot** dá **0,4861** aqui e
**0,5402** na nossa medição. É o mesmo modelo sem fine-tune.

A máscara importa no Spring porque há cenas com muito pixel inválido: a `seq0045`
tem 46% mascarado, a `seq0020` 21%, a `seq0014` 17%. Sem a correção, a fronteira
entre região válida e os zeros vira uma borda falsa no alvo.

---

## 4. O que dá e o que não dá para afirmar

**Dá:** que `gauss` e `normal` são os piores e que `grad`, `metric` e `geod` estão
acima do controle. Essas separações (0,03 e 0,07) sobrevivem ao ruído.

**Não dá:** ordenar `grad`, `metric` e `geod` entre si. Eles estão dentro de
**0,015** um do outro, e o ruído de reexecução medido é **0,0148** no F-borda,
porque o backward do `F.pad(mode="replicate")` na CUDA usa `atomicAdd`. Com n=1
por configuração, essa ordem é sorteio.

---

## 5. O que a confirmação mostrou

Rodamos os três candidatos com o **nosso** protocolo: split de teste, máscara,
`align full`, 100 épocas, **n=6**. Só um sobreviveu.

| termo | na ablação (n=1, validação) | na confirmação (n=6, teste) | veredito |
|---|---|---|---|
| `grad` | +0,0302 | **+0,0137, 6 de 6 seeds** | **confirmado** |
| `metric` | +0,0276 | -0,0050, 1 de 6 | não se sustenta |
| `geod` | +0,0152 | +0,0016, 3 de 5 | dentro do ruído |

O `grad` é também a **única intervenção de toda a campanha que moveu o `fmax`**
(+0,0091), o F-score no melhor limiar de cada modelo. Aquele teto de ~0,77 não
cedeu a seis configurações de perda nem a 768 px de resolução.

A discordância entre as duas colunas era esperada, e é exatamente por isso que a
confirmação existia: n=1 num protocolo diferente é hipótese, não resultado.
