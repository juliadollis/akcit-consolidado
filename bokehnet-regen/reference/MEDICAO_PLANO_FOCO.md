# Medição: a máscara do BiRefNet erra o plano de foco

**Data**: 2026-09-10. **Fonte**: piloto da rota C, job SLURM 32224, 162 amostras aceitas
de 204 pares, 29 cenas da RealBokeh (`train` e `validation`), h100n3.

Este documento existe porque o resultado abaixo **mudou o método** da rota C. Ele é a
evidência que justifica `src/qc/focus_region.py`, e é contra ele que qualquer mudança
futura no plano de foco tem que ser comparada.

---

## O que foi comparado

A RealBokeh_3MP publica, por cena, `focus_plane_distance` — a distância do plano de foco
**medida na captura**, com `focus_plane_uncertainty` ao lado. Isso é gabarito.

Comparei, nas 162 amostras aceitas do piloto:

| | |
|---|---|
| **obtida** | `focus_disparity` gravado no metadado = `mediana(1/z_depthpro[M_birefnet])` |
| **gabarito** | `1 / focus_plane_distance_m`, do `metadata/<cena>.json` da origem |

## Resultado `[M]`

| grandeza | valor |
|---|---|
| dentro de ±25% do gabarito | **57 de 162 — 35,2%** |
| razão obtida ÷ gabarito, p05 | 0,165 |
| razão obtida ÷ gabarito, **mediana** | **0,579** |
| razão obtida ÷ gabarito, p95 | 1,505 |
| distância de foco, mediana **medida** | **0,410 m** |
| distância de foco, mediana **implicada pela máscara** | **1,020 m** |
| incerteza mediana publicada pela origem | **±0,010 m** |

A razão mediana de 0,579 significa que a máscara escolhe um plano de foco cerca de
**1,7× mais longe** do que ele realmente estava. E a incerteza de ±1 cm elimina a
hipótese de que o gabarito é que seja ruim.

### As maiores divergências `[M]`

| amostra | medida | implicada pela máscara | razão |
|---|---|---|---|
| `c_realbokeh_train_91_l1..l3` | 10,230 m (±2,06) | 1,649 m | 6,20 |
| `c_realbokeh_validation_70_l1`, `_l10` | 0,405 m (±0,00) | 2,456 m | 0,16 |

## O descarte `[M]`

Além do erro, havia perda: **42 das 204 amostras (20,6%)** foram rejeitadas com
`focus_mask_empty`. A rejeição caiu em **10 cenas inteiras**, com **zero cenas** tendo
aceite e rejeição misturados — coerente com o fato de a máscara sair da AIF, que é a
mesma em todos os níveis de uma cena.

## Por quê `[M]`

Rodei o BiRefNet em 12 cenas guardando o **mapa de probabilidade cru**, antes do limiar
(`scripts/diagnose_empty_masks.py`, job 32231):

```
cena          max   p99.9   média   area@0.05  area@0.20  area@0.50
train_102    0.000   0.000  0.0000    0.00000    0.00000    0.00000
train_1527   0.000   0.000  0.0000    0.00000    0.00000    0.00000
train_1800   0.000   0.000  0.0000    0.00000    0.00000    0.00000
train_1873   0.000   0.000  0.0000    0.00000    0.00000    0.00000
test_102     0.000   0.000  0.0000    0.00000    0.00000    0.00000
test_100     1.000   1.000  0.5220    0.53629    0.53161    0.52573
test_101     1.000   1.000  0.3692    0.37300    0.37115    0.36927
```

Probabilidade **exatamente 0,000** — não é o limiar de 0,5 cortando algo que existe:
baixar para 0,05 não recupera nada. O modelo está **declinando**, não falhando.

E olhando as imagens, a razão fica evidente: a RealBokeh é feita de **cenas**, não de
fotos de objeto. `train_102` é um tronco de árvore num parque; `test_100` é um muro de
pedra. O BiRefNet é um segmentador de objeto **saliente**, e está sendo usado fora do
domínio dele. Quando não declina, o que ele segmenta não é necessariamente o que estava
em foco — daí os 64,8% de desacordo entre as amostras que passaram.

O risco estava, aliás, documentado desde o começo, no cabeçalho de
`src/model_runtime/segmentation.py`: *"o BiRefNet é um segmentador de objeto saliente, e
as duas coisas coincidem só quando o fotógrafo focou o objeto saliente."* O que faltava
era o número.

## O que o paper faz nesta exata situação

`paper.txt:359-368`, §3.2(c), sobre LFDOF e RealBokeh:

> *"due to the increased diversity and complexity of the scenes in these datasets, the
> initial estimate of M is sometimes unreliable. **Rather than simply verifying and
> discarding unreliable cases**, we introduce a manual refinement step. Specifically, we
> re-select a small yet reliable in-focus region to correct M, thereby accurately
> extracting D_focus. This strategy preserves challenging samples rather than excluding
> them."*

Ou seja: o paper **antecipa** este problema, nestes mesmos dois datasets, e **rejeita
explicitamente** a estratégia de descartar — que é exatamente o que nossos 20,6% eram.
Os nossos 35,2% são a quantificação do *"sometimes unreliable"* dele.

## A decisão

Substituir o passo manual por um automático, em `src/qc/focus_region.py`, e **não
descartar**.

O método não usa saliência nem nitidez absoluta. Usa **retenção de detalhe**: temos a
AIF e a bokeh da mesma cena, e no plano de foco a bokeh preservou o detalhe da AIF.

```
retencao(x) = media_local(|laplaciano(bokeh)|) / media_local(|laplaciano(aif)|)
```

O denominador normaliza pela textura da própria cena, que é o que faz medida de nitidez
absoluta falhar: folhagem desfocada tem mais energia de alta frequência que parede lisa
em foco. `tests/test_focus_region.py` monta exatamente essa armadilha e prova, com um
teste dedicado, que a medida absoluta escolheria o lado errado e a razão não.

### Por que NÃO usar a distância medida como rótulo

Ela resolveria a RealBokeh e deixaria o **LFDOF** — que não publica distância nenhuma —
com um rótulo de outra qualidade dentro da mesma rota. Dois níveis de qualidade de
rótulo no mesmo lote é o erro do kfix com outra roupa.

Então: **um método só**, validado contra o gabarito onde ele existe, e a distância medida
gravada no metadado para quem quiser auditar.

### Marcação

Toda amostra carrega `focus_source` (`birefnet` / `birefnet_refined` / `retention_only`)
e `focus_was_refined`, no metadado **e** no manifesto. Isso permite treinar com e sem as
amostras refinadas e medir a diferença — sem isso, "consertamos" seria afirmação sem
teste.

## Aberto `[A]`

- **Viés de escala do Depth Pro.** Parte da divergência pode vir da profundidade e não da
  máscara. As duas causas **não são separáveis por amostra** com o que a RealBokeh publica:
  há uma distância medida por cena, a do plano de foco, e não uma profundidade métrica de
  referência. `scripts/validate_focus_refinement.py` limita a confusão por dois caminhos —
  (1) gabarito fora de `[disparity_min, disparity_max]` da própria amostra, que nenhuma
  máscara alcançaria, logo é da profundidade; (2) razões divididas pela mediana global, que
  remove um fator multiplicativo único e deixa só a dispersão. Limitar não é separar, e o
  relatório diz isso.
- Os parâmetros de `focus_region.py` — janela de 33 px, fração de 5%, piso de acordo de
  0,30 — são todos `[A]`. Nenhum vem do paper, que não publica nada sobre o refinamento.
- Se o refinamento de fato melhora a concordância: a medir com
  `scripts/validate_focus_refinement.py` contra o baseline de **35,2%** registrado aqui.
  O script existe desde a etapa 9 (`REGISTRO.md`) e ainda **não rodou** — precisa de um
  piloto novo, gerado com o refinamento ligado. Ele decide pelo bloco **pareado**
  (a mesma amostra com e sem refino, via `focus_disparity_from_initial_mask`), e não pela
  comparação entre `focus_source`, que é confundida: o grupo `birefnet` é por construção o
  subgrupo em que o segmentador já concordava com a física.
