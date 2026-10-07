# Datasets — qual usar e por quê

Índice rápido dos três datasets suportados e do papel de cada um na investigação.

| Dataset | No treino do DepthPro? | Papel | Guia |
|---|---|---|---|
| **Hypersim** | **Sim** (Train, Val) | Controle. Demonstra a ausência de headroom em domínio de treino. | `ROTEIRO_IMPLEMENTACAO.md` |
| **DIODE** | Não aparece na tabela | Primeira verificação de headroom fora do domínio. Custo mínimo. | `GUIA_DIODE.md` |
| **Spring** | Apenas teste | Alvo científico principal: menor F1 de borda reportado (0.079). | `GUIA_SPRING.md` |

## A descoberta que motivou tudo

A tabela de datasets do paper do DepthPro lista o **Hypersim como "Train, Val"**, e a
tabela de perdas por estágio o inclui nos dois estágios, inclusive no estágio 2, que aplica
MAE, MSE, MAGE, **MALE** e MSGE. O MALE é erro absoluto de Laplaciano, ou seja, supervisão
de segunda ordem, e o estágio 2 existe declaradamente para afiar bordas.

Traduzindo: o DepthPro foi treinado no Hypersim, com supervisão de segunda ordem,
especificamente para melhorar bordas naquele dataset. Todos os nossos resultados negativos
têm essa explicação.

**Consequência de nomenclatura:** o que chamávamos de "zero-shot" no Hypersim não é
zero-shot. É o modelo avaliado dentro do domínio de treino. Esse nome precisa mudar em
qualquer material que sair.

## Ordem sugerida

1. **DIODE primeiro**, porque custa quase nada — a profundidade já vem em `.npy` em metros,
   então é só reorganizar. Responde rápido se existe headroom fora do domínio de treino.
2. **Spring depois**, se o headroom se confirmar. Exige converter disparidade em
   profundidade, com as armadilhas descritas no guia, mas é onde a hipótese da curvatura
   pode ser testada em condições justas.

## O que é comum aos três

Todos entregam o mesmo layout, e por isso todo o resto do pipeline funciona sem alteração:

```
<raiz>/rgb/<cena>__<id>.png
<raiz>/depth/<cena>__<id>.npy     profundidade métrica crua, em metros
<raiz>/mask/<cena>__<id>.npy      opcional; validade por pixel
```

O prefixo antes de `__` é o que `repro.scene_of` devolve, e é a **unidade de agrupamento**
de toda a estatística pareada. O tamanho efetivo de amostra é o número de grupos, não de
imagens — os scripts de preparação imprimem esse número, e ele foi verificado como idêntico
ao `n` que aparece na comparação pareada.
