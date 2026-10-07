# Guia DIODE — avaliação fora do domínio de treino do DepthPro

Documento de implementação. Explica **por que** vamos ao DIODE, **o que muda** no pipeline
(muito pouco) e **como rodar**, passo a passo.

---

## 1. Por que estamos mudando de dataset

Fomos verificar a lista de datasets do paper do DepthPro e encontramos o motivo de todos
os resultados negativos: **o Hypersim está na lista de treino deles**. A tabela de datasets
o marca como "Train, Val", e a tabela de perdas por estágio mostra que ele aparece nos dois
estágios — inclusive no estágio 2, que aplica MAE, MSE, MAGE, **MALE** e MSGE.

O MALE é o erro absoluto de Laplaciano, ou seja, supervisão de **segunda ordem**, e o
estágio 2 existe declaradamente para afiar bordas. Traduzindo: o DepthPro foi treinado no
Hypersim, com supervisão de segunda ordem, especificamente para melhorar bordas naquele
dataset.

Consequências diretas:

- O que vínhamos chamando de "zero-shot" **não é zero-shot**. É o modelo avaliado dentro do
  domínio de treino dele. Esse nome precisa mudar em qualquer material que sair.
- Isso explica tudo de uma vez: por que nenhuma configuração supera o baseline, por que até
  o berHu puro degrada, e por que os termos de segunda ordem não ajudam. Não havia headroom
  a tomar.
- O trabalho de engenharia e o rigor estatístico estão corretos. O que estava errado era a
  escolha do dataset de avaliação, e isso é corrigível.

## 2. Por que o DIODE

Precisávamos de um dataset que o DepthPro **nunca tocou**, nem para treino nem para teste.
O DIODE não aparece em nenhuma linha da tabela de datasets deles.

Além disso ele é quase plug-and-play com o nosso pipeline:

| Característica | Valor |
|---|---|
| RGB | PNG, 1024×768 |
| Profundidade | `*_depth.npy`, float32, **em metros**, mesma resolução |
| Máscara de validade | `*_depth_mask.npy`, binária (1 = retorno válido) |
| Sensor | FARO Focus S350, precisão ±1 mm |
| Densidade de retorno | ~99,6% interno, ~67% externo |
| Organização | hierárquica: cena → scan → recortes |

A profundidade já vem em `.npy` em metros, que é exatamente o que o nosso
`HighQualityDepthDataset` lê. Não há conversão de formato, não há disparidade para
converter, não há HDF5 para extrair.

**Vamos usar o subconjunto interno** (`indoors`), por dois motivos: a densidade de retorno
é muito maior (99,6% contra 67%), o que importa porque medimos borda; e a faixa de
profundidade é compatível com a calibração atual da loss, enquanto o externo vai a 350 m.

## 3. O que mudou no código

Duas coisas, ambas pequenas. Todo o resto do repositório — perdas, treinador, métricas,
modelo, estatística, figuras — funciona sem alteração.

**`riemann/dataset.py`** ganhou duas capacidades opcionais:

- Uma pasta `mask/` agora é reconhecida. Quando existe, a máscara é combinada com o
  critério automático `depth > 0`. Isso aproveita as máscaras do scanner do DIODE em vez de
  inferir validade só pelo valor. Sem a pasta, o comportamento é o de antes.
- Um parâmetro `max_depth` exclui da máscara os pixels acima de um teto. Serve para o
  domínio externo, cuja cauda longa distorceria o alinhamento afim e as métricas.

**`scripts/prepare_diode.py`** é novo. Percorre a hierarquia do DIODE, renomeia e cria
links simbólicos no layout `rgb/`, `depth/`, `mask/`. Não converte nem copia nada por
padrão, então é rápido e não duplica disco.

## 4. O ponto de atenção mais importante: o agrupamento

Toda a nossa estatística pareada agrega **por cena**, e o tamanho efetivo de amostra é o
número de grupos, não de imagens. No DIODE isso exige cuidado, porque a hierarquia tem dois
níveis e eles significam coisas diferentes:

- Um **scan** é uma única aquisição, com o scanner numa posição fixa. Os recortes de um
  mesmo scan vêm todos do mesmo ponto de vista e são altamente correlacionados.
- Uma **cena** é um local, e pode conter vários scans.

O script aceita `--agrupar scan` (padrão) ou `--agrupar scene` (mais conservador). O número
de grupos é impresso no fim da execução, e ele bate exatamente com o `n` que vai aparecer na
comparação pareada — confira esse número antes de prosseguir.

A validação do DIODE tem 10 scans internos, o que daria `n = 10`. É pouco: nosso teste no
Hypersim tinha 29. **Sugestão:** prepare a validação e mais alguns scans do treino, já que
não vamos treinar no DIODE nesta primeira rodada — é avaliação cruzada pura. Com 20 a 30
scans o poder estatístico fica comparável ao que já tínhamos.

## 5. Passo a passo

### 5.1 Baixar

O DIODE está em `diode-dataset.org`. Baixe **val** e, se possível, **train**, do domínio
interno. Descompacte preservando a hierarquia.

### 5.2 Preparar

```bash
# validação interna
python scripts/prepare_diode.py \
  --diode-root /data/diode \
  --split val --dominio indoors \
  --agrupar scan \
  --out-root /data/diode_prep/val_indoor

# mais scans do treino, para aumentar o n (não vamos treinar neles)
python scripts/prepare_diode.py \
  --diode-root /data/diode \
  --split train --dominio indoors \
  --agrupar scan \
  --out-root /data/diode_prep/train_indoor
```

**Confira na saída:** o número de grupos, e a faixa típica de profundidade. Se a faixa for
muito diferente da do Hypersim (~1 a 10 m), avise antes de treinar — os tetos `gauss_clamp`
e `metric_clamp` foram calibrados naquela faixa.

### 5.3 Sanidade rápida

Antes de qualquer coisa cara, confirme que o carregamento está correto:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from riemann.dataset import HighQualityDepthDataset
from riemann.repro import scene_of
ds = HighQualityDepthDataset('/data/diode_prep/val_indoor', size=(512,512))
b = ds[0]
print('amostras :', len(ds))
print('grupos   :', len({scene_of(k) for k in ds.keys}))
print('chave    :', b['key'])
print('cena     :', scene_of(b['key']))
print('depth    : %.2f a %.2f m' % (float(b['depth'][b['mask']>0].min()), float(b['depth'].max())))
print('mascara  : %.3f valido' % float(b['mask'].mean()))
"
```

Esperado: profundidade em metros (não em [0,1]), máscara com fração alta de válidos, e o
número de grupos coerente com o que o prepare reportou.

### 5.4 A pergunta central: existe headroom?

Este é o experimento que decide tudo. Roda a avaliação pareada com os campeões que já
temos, treinados no Hypersim, agora medidos no DIODE.

```bash
python scripts/evaluate_paired.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --seeds-dir  /workspace/runs/champion_heads_final \
  --data-root  /data/diode_prep/val_indoor \
  --out-dir    /workspace/runs/champion_heads_final/aval_diode \
  --variant heads_final --nivel cena
```

E a tabela consolidada:

```bash
python scripts/make_results_table.py \
  --entrada heads_final=/workspace/runs/champion_heads_final/aval_diode \
  --out-dir /workspace/relatorio_diode \
  --titulo "DIODE interno — fora do domínio de treino do DepthPro"
```

**Como ler o resultado.** O número que interessa aqui **não** é se o nosso modelo ganha —
ele foi treinado no Hypersim, então provavelmente vai transferir mal. O que interessa é a
**linha do modelo pré-treinado**: se o AbsRel e o bF-max dele no DIODE forem claramente
piores que no Hypersim, está confirmado que existe headroom fora do domínio de treino, e o
caminho de retreinar num dataset externo se justifica. Se ele for igualmente forte, a
hipótese de headroom cai e precisamos repensar.

### 5.5 Figuras, para inspeção visual

```bash
python scripts/make_riemannian_figure.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads_final/seed_0/best.pt \
  --data-root /data/diode_prep/val_indoor \
  --out-dir /workspace/figuras_diode \
  --variant heads_final --n-imagens 6 --comparar \
  --titulo "DIODE interno"
```

Vale olhar o mapa de área: em cena real com scanner, as bordas devem aparecer mais nítidas
que no Hypersim, e é aí que a diferença entre os modelos fica visível.

## 6. O que enviar de volta

| Item | Onde |
|---|---|
| Saída do prepare (com o número de grupos) | terminal |
| Sanidade do carregamento | terminal |
| Resumo estatístico | `aval_diode/resumo.txt` |
| Tabela consolidada | `relatorio_diode/tabela_geral.*` |
| Painéis comparativos | `figuras_diode/` |

## 7. Próximo passo, depois deste

Se o headroom se confirmar, o passo seguinte é o **Spring**, onde a acurácia de borda
reportada do DepthPro é de apenas 0.079 — de longe a mais baixa entre os datasets testados.
Ele é sintético com ground truth pixel-perfeito, o que torna a métrica de borda confiável.
O custo é maior, porque distribui disparidade estéreo e exige conversão, então deixamos
para depois de saber se o caminho se sustenta.
