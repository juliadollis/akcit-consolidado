# Fase 2 do BokehNet — registro de decisões

> Objetivo declarado: **máxima fidelidade ao paper GenRefocus (arXiv 2512.16923)**.
> Cada decisão abaixo diz o que o paper manda, o que nós fizemos e por quê.
> Onde divergimos, está marcado como DIVERGÊNCIA com a justificativa e o que
> seria preciso para fechar a lacuna.
>
> Criado em 2026-08-13, ao subir a fase 2 na dgx-H100-01.

---

## 1. Currículo de duas fases

**Paper §4.1, literal:** *"BokehNet is trained in two stages: (i) 40K steps on
synthetic data, and (ii) 60K steps on real data."*

| | Fase 1 | Fase 2 |
|---|---|---|
| Dados | rota a (sintética) | rotas b + c (reais) |
| Steps | 40.000 | 60.000 |
| Estado | **CONCLUÍDA** 2026-08-13 (job 29358, 6d04h31m) | subindo agora |
| Máquina | dgx-H100-02 (SLURM + singularity) | dgx-H100-01 (docker) |

**FIEL.** A rota a NÃO entra na fase 2.

## 2. Optimizer/scheduler entre as fases: RESET

**Decisão:** a fase 2 começa dos pesos LoRA da fase 1 (`--init-lora`), mas com
optimizer e scheduler **novos** (warmup 500 + cosine sobre os 60K).

**Base no paper:** o §3.2(a) chama a fase sintética de *"We **pretrain** with
synthetic data"*. Pré-treino seguido de treino é a convenção em que o segundo
estágio tem schedule próprio. Reforços: o paper nunca usa "curriculum",
"continued training" ou "resume"; e dá orçamentos de step **separados** (40K e
60K), implicando dois schedules, não um cosine único sobre 100K.

**FIEL** (interpretação documentada, já registrada no handoff de 2026-07-19).

## 3. Batch efetivo 32

**Paper §4.1:** *"per-GPU batch size of 1 and gradient accumulation of 8 steps on
4× RTX A6000 GPUs"* = 1 × 8 × 4 = **32**.

**Nosso:** 4 × H100, `batch_size: 1`, `gradient_accumulation_steps: 8` = **32**.

**FIEL, e agora idêntico também no número de GPUs** (a fase 1 rodou em 2 GPUs
com accum 16, que dá o mesmo 32).

## 4. Filtro de qualidade da calibração do K (NOVO nesta fase)

**Paper §3.2(c), literal:** *"The selected K\* is then used as the
pseudo-bokeh-level label for training, **provided that its corresponding SSIM
exceeds a predefined threshold to ensure reliable supervision**."*

**O que motivou:** na inferência de teste do step 27500 (2026-08-13), a amostra
`c_1000` veio com o mapa de defocus **inteiramente zero** enquanto a imagem alvo
tinha bokeh forte. O modelo obedeceu o mapa e devolveu a imagem nítida — o par é
que era contraditório. A auditoria (`scripts/audit_phase2_defocus.py`, 300
amostras) mediu:

| Classe | Rota c | Rota b |
|---|---|---|
| OK | 92,7% | **100%** |
| Saturada | 4,0% | 0% |
| **Contraditória** | **2,7%** | 0% |
| Degenerada consistente | 0,7% | 0% |

E a causa: em 5 das 8 contraditórias o `k` era **exatamente 0**, com
`blur_ratio` de 0,14 a 0,50 (alvo nitidamente mais borrado que a entrada). `k=0`
não é "foto sem bokeh", é o sweep de SSIM (Eq. 5) tendo falhado.

**Implementação:** `min_calibration_ssim` no YAML →
`data.py::_filter_by_calibration_ssim`, aplicado **por fonte**, ANTES do
`select_columns` (a coluna `calibration_ssim` não está em `BOKEH_KEEP_COLUMNS` e
seria descartada). Fontes **sem** a coluna passam inteiras — a rota b deriva o K
da EXIF pela Eq. 3, não por sweep, e no paper o limiar aparece **só** no item (c).

### 4.1 Duas armadilhas medidas antes de treinar

**(i) A rota b TEM a coluna, mas VAZIA.** Nula nas 2.000 linhas checadas. Um
filtro que testasse só a *existência* da coluna descartaria as 11.635 amostras
dela. Foi exatamente o que aconteceu na 1ª tentativa do smoke da fase 2, que
abortou com `ValueError: o filtro zerou o dataset`. O teste correto é se a coluna
está **preenchida**, não se existe. (O erro foi pego pelo smoke, não pelo treino
de 60K — que é para isso que o smoke serve.)

**(ii) O limiar de SSIM NÃO pega os `k = 0`.** Medido na rota c inteira (2.932):

```
k <= 0:  51 amostras (1,7%)
   SSIM delas: min=0,5079  mediana=0,7592  max=0,9908   <- ALTO
   limiar 0,5 pegaria  0/51     limiar 0,7 pegaria 14/51
```

Os pares patológicos têm SSIM **alto**, então escapam do limiar. São dois
defeitos ortogonais e o paper só resolve um explicitamente.

**Por isso descartamos também `k <= 0`, e isso também vem do paper.** A Eq. 5 é
`argmax` sobre o intervalo **ABERTO** `(K_min, K_max)`. Um K gravado exatamente
no limite inferior não é saída válida do sweep; é sentinela de calibração que não
rodou. Fisicamente, `K=0` significa "sem bokeh", o que contradiz um alvo com
`blur_ratio` de 0,14.

### 4.2 O valor do limiar: 0,6 (escolha nossa, documentada)

O paper diz *"predefined threshold"* e **não publica o número**. Distribuição
medida da coluna na rota c:

| | valor |
|---|---|
| mínimo | 0,338 |
| p10 | 0,738 |
| p25 | 0,808 |
| mediana | 0,870 |
| máximo | 0,991 |

| Limiar | Descarta |
|---|---|
| 0,5 | 0,3% |
| **0,6** | **1,2%** |
| 0,7 | 5,5% |

**Escolhemos 0,6.** Fica acima da cauda de falha visível (mínimo 0,338) sem ser
agressivo. O motivo de não subir para 0,7 é específico e importa: nosso
renderizador é o `_render_bokeh_simple` (kernel limitado a 51 px), **não** o
BokehMe do paper. Isso deprime o SSIM por *incompatibilidade de renderizador*, não
por K ruim. Cortar em 0,7 puniria amostras pela limitação da nossa ferramenta em
vez de pela qualidade da calibração.

**Efeito combinado medido no smoke:** 81 de 2.932 amostras da rota c descartadas
(**2,8%**), batendo com os 2,7% de pares contraditórios que a auditoria tinha
achado por um caminho independente (blur_ratio vs mapa zerado).

**FIEL no procedimento; o valor é nosso.** Não aplicar o filtro é que seria a
divergência.

## 5. Mapa de defocus recomposto

**Paper Eq. 2:** `D_def = K · |D − D_focus|`, e a inferência oficial
(`Inference_bokehNet.py:138-140`) fecha com `clip(K·|D−D_foco|/MAX_COC, 0, 1)`,
`MAX_COC = 100`.

**Nosso:** `defocus_source: recompute`, montando o mapa de `depth`+`k`+`s1`.
A coluna `defocus_map` dos dfs é **ignorada** porque foi salva normalizada POR
IMAGEM (`dm/dm.max()`), o que apaga o K.

**FIEL** (fix de 2026-08-05, mantido).

## 6. DIVERGÊNCIA CONHECIDA — o K da rota b não varia

**Paper §3.2(b):** o K do ITW vem da Eq. 3, com `f` e `F` da EXIF e o `D_focus`
da Eq. 4 (mediana da profundidade na máscara do BiRefNet), usando um estimador
**métrico** (referência [7] = Depth Pro, *"Sharp monocular metric depth"*).

**Medido em 2026-08-13, 300 amostras da rota b:**

```
k: min=50.0  mediana=50.0  max=50.0        <- fallback do pipeline, em TODAS
map_max: min=0.250 mediana=0.490 max=0.500 <- teto em 0.5, porque 50*1/100
exif VAZIA: 0   (f_number e focal_length PRESENTES e variando)
```

**Diagnóstico:** a EXIF tem o que a Eq. 3 pede (`f_number` 2,8 a 5,6;
`focal_length` 31 a 95; e `focal_length_35`, que dá o crop factor para o
`pixel_ratio`). O que falta é o `D_focus` em **metros**: a coluna `depth` dos dfs
foi gravada **normalizada em [0,1]**, perdendo a escala métrica que a Eq. 3 exige
dimensionalmente. Sem ela, o pipeline caiu no fallback `k=50`.

**Consequência:** a rota b (11.635 pares, ~80% do dado da fase 2) entra **sem
nenhuma variação de bokeh level**. Ela continua ensinando aparência óptica real,
que é exatamente o que faltou depois da fase 1 (a inferência do step 27500 mostra
o modelo borrando demais em foto real). Mas o **controle de intensidade** virá
quase todo dos ~2.900 pares da rota c.

**Risco:** a controlabilidade (métrica LVCorr do paper, Tab. 3) pode ficar abaixo
do publicado, e agora sabemos exatamente por quê.

**Decisão de agora:** treinar **com a rota b como está**, por escolha explícita
da usuária, enquanto ela conversa com o time de dados. Não é ignorar o problema:
é não bloquear 60K steps por uma correção que depende de terceiros.

**Como fechar a lacuna (sem regerar imagem):** basta o time de dados fornecer a
profundidade métrica do Depth Pro OU o fator de escala usado na normalização.
Com isso, recompomos o K pela Eq. 3 no dataloader, exatamente como já fazemos com
o `defocus_map`.

## 7. DIVERGÊNCIA CONHECIDA — renderizador da rota a (afeta só a fase 1)

Os alvos sintéticos da fase 1 foram renderizados pelo `_render_bokeh_simple`
(kernel limitado a 51 px) porque o `_render_bokehme` levanta `ImportError`
incondicional — o BokehMe nunca foi instalado. Amostras de K alto saturam o blur.
**Não afeta a fase 2**, cujos alvos são fotos reais.

## 8. NÃO IMPLEMENTADO — aperture shape (§3.3)

O paper treina um LoRA **extra** para forma de abertura, com o LoRA base
congelado, sobre o dataset PointLight-1K. É trabalho separado, posterior.

## 9. Hiperparâmetros não especificados pelo paper

O paper não informa optimizer, lr, warmup nem resolução. Nossas escolhas,
idênticas às da fase 1 para manter comparabilidade:

| | valor |
|---|---|
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| scheduler | cosine, warmup 500, min_lr_ratio 0.1 |
| precisão | bf16 |
| resolução | 512² |

Knob futuro (tuning, não fidelidade): se a fase 2 degradar o que a fase 1
aprendeu, testar `lr` menor (ex.: 5e-5), **mantendo** o reset.

## 10. Nomes no HF (nada é sobrescrito)

| Repo | Conteúdo |
|---|---|
| `genrefocus-bokehnet-synth-condlora` | fase 1, snapshot a cada 2500 steps |
| `genrefocus-bokehnet-synth-2gpu` | fase 1, `.safetensors` final |
| **`genrefocus-bokehnet-fase2-smoke`** | **NOVO** — só o smoke de 20 steps |
| **`genrefocus-bokehnet-fase2-real`** | **NOVO** — fase 2, snapshot a cada 2500 |

## 11. Infraestrutura da fase 2 (dgx-H100-01, docker)

Registrado porque a máquina é diferente e **não tem SLURM**:

- Imagem `julia-genrefocus:1.0` com versões **fixas**, idênticas ao ambiente que
  rodou a fase 1 sem falha (`docker/verify_env.py` confere e aborta se divergir).
  O bootstrap antigo instalava `transformers diffusers peft` **sem pin**, que é a
  origem dos 4 jobs falhos de julho.
- `--user $(id -u):$(id -g)` em todo `docker run`. Sem isso os arquivos nascem do
  root: os 54 GB de `hf-cache` da rodada anterior nesta máquina são `root:root` e
  a usuária não consegue apagar.
- Container **detached**, sobrevive a queda de ssh/VPN (aconteceu 2x).
- **NUNCA** `docker system prune`/`rmi`/`rm` de container alheio: há containers
  parados de outras 4 pessoas no host.
- 4 das 8 GPUs, deixando 4 livres para os outros (não há fila protegendo).

## 12. Robustez a arquivo corrompido (herdado da fase 1)

O job 29267 morreu no step 3380 com `OSError: image file is truncated` em UM png
de 204.000. O loader agora pula arquivo corrompido com aviso, limitado a 8
tentativas consecutivas para que falha sistemática ainda quebre o treino.
Confirmado em produção: o mesmo arquivo reapareceu no job 29358, foi pulado, e o
treino seguiu por 36.750 steps.

---

## 13. Variantes treinadas (2026-08-19 em diante) e uma ressalva de leitura

Tres treinos partem do MESMO LoRA da fase 1, com hiperparametros identicos. A
unica diferenca e a composicao/qualidade do dado, entao sao ablacoes limpas.

| Treino | Rota b | output_dir | repo HF |
|---|---|---|---|
| Original | k=50 fixo (fallback) | `bokehnet_fase2_4gpu` | `genrefocus-bokehnet-fase2-real` |
| **kfix** | K pela Eq. 3 (variacao 58,2x) | `bokehnet_fase2_kfix` | `genrefocus-bokehnet-fase2-kfix` |
| **so rota c** | ausente | `bokehnet_fase2_rotac_only` | `genrefocus-bokehnet-fase2-rotac-only` |

### RESSALVA — curva do "so rota c" abaixo do step 14000 nao e confiavel

O mesmo experimento foi submetido DUAS VEZES por engano: um job SLURM na
H100-02 (30394) e um container na H100-01, ambos apontando para o MESMO repo HF
`genrefocus-bokehnet-fase2-rotac-only`. Como os dois sobem um arquivo por step a
cada 1000, os arquivos `bokeh_step1000` ate `bokeh_step13000` sao uma MISTURA
das duas execucoes e nao ha como saber qual step veio de qual.

O job da H100-02 foi cancelado em 2026-08-23 (autorizado pela usuaria) no step
13040; seus checkpoints seguem intactos no disco daquele no, nada foi apagado.
A partir do step 14000 so a execucao da H100-01 escreve, entao dali em diante a
curva e limpa.

**Como ler:** descartar os pontos abaixo de 14000 desta ablacao. Sobram 46
pontos limpos ate o step 60000, suficiente para a comparacao entre treinos.

**Licao:** dois treinos nunca devem compartilhar `upload_hf_repo_base`. Se o
mesmo experimento rodar em duas maquinas, o repo tem de levar o nome do no.
