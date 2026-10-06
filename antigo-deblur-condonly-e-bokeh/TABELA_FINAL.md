# Tabela final de bokeh (formato da Tab. 3 do GenRefocus)

Metricas por imagem, com IC95 por bootstrap percentil (10.000 reamostras).
`LVCorr` esta na convencao do PAPER (quanto maior melhor): e o simetrico do
Pearson cru, porque mais K significa mais desfoque e MENOS variancia do Laplaciano.

> **Ressalva de comparabilidade.** O LF-Bokeh do paper nao foi liberado, entao a
> mesa `LF-Bokeh reproduzido` e uma reconstrucao do protocolo. Os pesos OFICIAIS
> medem LPIPS 0,2047 nela contra 0,0833 publicado. As comparacoes entre modelos
> valem (mesmo dado, mesmo pipeline); os valores absolutos nao sao os do paper.


## LF-Bokeh reproduzido (BLB) (500 imagens de 10 cenas)
  **O IC95 e agrupado por cena: o tamanho efetivo de amostra aqui e 10, nao 500.**

| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | 0.1804 [0.1605, 0.2010] | 0.1233 | 0.9618 | +0.7312 | +0.0567 |
| so rota c (a+c) | 0.1952 [0.1728, 0.2202] | 0.1316 | 0.9582 | +0.7296 | +0.0419 |
| fase 1, so sintetico (a) | 0.2046 [0.1804, 0.2291] | 0.1297 | 0.9582 | +0.9059 | +0.0325 |
| oficial do paper | 0.2047 [0.1724, 0.2431] | 0.1308 | 0.9513 | +0.8868 | +0.0324 |
| nosso original, fase 2 (a+b+c) | 0.2110 [0.1875, 0.2370] | 0.1450 | 0.9551 | +0.4365 | +0.0261 |
| sem filtro de SSIM [step 28000] | 0.2204 [0.2020, 0.2435] | 0.1493 | 0.9522 | +0.5268 | +0.0167 |
| kfix, K da rota b pela Eq. 3 | 0.2252 [0.2054, 0.2481] | 0.1580 | 0.9506 | +0.8288 | +0.0119 |
| LINHA DE IDENTIDADE | 0.2371 [0.2193, 0.2549] | 0.1745 | 0.9477 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.8447 [0.7802, 0.8965] | 0.7156 | 0.6986 | -0.9724 | -0.6076 |

## RealBokeh test v2 (217 imagens de 217 cenas)


| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | 0.1134 [0.1049, 0.1226] | 0.0812 | 0.9825 | +0.8136 | +0.2452 |
| so rota c (a+c) | 0.1140 [0.1053, 0.1234] | 0.0804 | 0.9830 | +0.8245 | +0.2446 |
| nosso original, fase 2 (a+b+c) | 0.1282 [0.1191, 0.1377] | 0.0875 | 0.9805 | +0.8257 | +0.2305 |
| sem filtro de SSIM [step 28000] | 0.1313 [0.1223, 0.1408] | 0.0879 | 0.9800 | +0.8759 | +0.2274 |
| kfix, K da rota b pela Eq. 3 | 0.1433 [0.1324, 0.1545] | 0.0939 | 0.9776 | +0.4832 | +0.2154 |
| fase 1, so sintetico (a) | 0.3235 [0.2999, 0.3465] | 0.1812 | 0.9234 | +0.8609 | +0.0352 |
| oficial do paper | 0.3454 [0.3174, 0.3721] | 0.1885 | 0.9278 | +0.8498 | +0.0133 |
| LINHA DE IDENTIDADE | 0.3587 [0.3395, 0.3777] | 0.2086 | 0.9218 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.7655 [0.7467, 0.7838] | 0.5943 | 0.6950 | -0.9769 | -0.4069 |

## RealDOF (50 imagens de 50 cenas)


| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 45000] | 0.1251 [0.1078, 0.1448] | 0.0907 | 0.9717 | -0.2111 | +0.2028 |
| so rota c (a+c) [step 60000] | 0.1263 [0.1088, 0.1458] | 0.0918 | 0.9717 | -0.1936 | +0.2016 |
| so rota c (a+c) | 0.1271 [0.1097, 0.1466] | 0.0920 | 0.9718 | -0.2061 | +0.2008 |
| so rota c (a+c) [step 30000] | 0.1365 [0.1179, 0.1567] | 0.0964 | 0.9704 | -0.1776 | +0.1914 |
| so rota c (a+c) [step 20000] | 0.1424 [0.1235, 0.1627] | 0.1001 | 0.9703 | +0.0281 | +0.1855 |
| so rota c (a+c) [step 10000] | 0.1522 [0.1300, 0.1759] | 0.1042 | 0.9692 | +0.2509 | +0.1757 |
| nosso original, fase 2 (a+b+c) | 0.2148 [0.1882, 0.2406] | 0.1410 | 0.9599 | +0.1324 | +0.1132 |
| sem filtro de SSIM [step 28000] | 0.2249 [0.1973, 0.2521] | 0.1463 | 0.9594 | +0.5360 | +0.1030 |
| kfix, K da rota b pela Eq. 3 | 0.2524 [0.2280, 0.2764] | 0.1613 | 0.9529 | -0.4599 | +0.0756 |
| oficial do paper | 0.2677 [0.2418, 0.2939] | 0.1410 | 0.9313 | +0.9644 | +0.0603 |
| fase 1, so sintetico (a) | 0.2782 [0.2526, 0.3043] | 0.1533 | 0.9353 | +0.9247 | +0.0497 |
| LINHA DE IDENTIDADE | 0.3279 [0.3036, 0.3512] | 0.2030 | 0.9471 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.8403 [0.8209, 0.8596] | 0.6490 | 0.6804 | -0.9190 | -0.5124 |


## Teste pareado contra `oficial do paper`, cena a cena

Diferenca media por cena, IC95 e p bilateral por bootstrap. Negativo em LPIPS/DISTS = o modelo e melhor.


### LF-Bokeh reproduzido (BLB)

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | DISTS | 500 | 10 | -0.0075 | [-0.0321, +0.0091] | 0.5620 | **o modelo** |
| fase 1, so sintetico (a) | DISTS | 500 | 10 | -0.0012 | [-0.0176, +0.0089] | 0.8798 | **o modelo** |
| so rota c (a+c) | DISTS | 500 | 10 | +0.0008 | [-0.0192, +0.0151] | 0.8430 | a referencia |
| nosso original, fase 2 (a+b+c) | DISTS | 500 | 10 | +0.0141 | [-0.0019, +0.0274] | 0.0812 | a referencia |
| sem filtro de SSIM [step 28000] | DISTS | 500 | 10 | +0.0185 | [-0.0011, +0.0327] | 0.0670 | a referencia |
| kfix, K da rota b pela Eq. 3 | DISTS | 500 | 10 | +0.0271 | [+0.0081, +0.0409] | 0.0076 | a referencia |
| LINHA DE IDENTIDADE | DISTS | 500 | 10 | +0.0436 | [+0.0257, +0.0573] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 500 | 10 | +0.5848 | [+0.5245, +0.6405] | 0.0000 | a referencia |
| so rota c (a+c) [step 60000] | LPIPS | 500 | 10 | -0.0243 | [-0.0704, +0.0091] | 0.2244 | **o modelo** |
| so rota c (a+c) | LPIPS | 500 | 10 | -0.0095 | [-0.0479, +0.0205] | 0.6422 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 500 | 10 | -0.0001 | [-0.0327, +0.0197] | 0.9462 | **o modelo** |
| nosso original, fase 2 (a+b+c) | LPIPS | 500 | 10 | +0.0063 | [-0.0243, +0.0283] | 0.6010 | a referencia |
| sem filtro de SSIM [step 28000] | LPIPS | 500 | 10 | +0.0157 | [-0.0209, +0.0406] | 0.3750 | a referencia |
| kfix, K da rota b pela Eq. 3 | LPIPS | 500 | 10 | +0.0205 | [-0.0159, +0.0463] | 0.2312 | a referencia |
| LINHA DE IDENTIDADE | LPIPS | 500 | 10 | +0.0324 | [-0.0010, +0.0559] | 0.0580 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 500 | 10 | +0.6400 | [+0.5666, +0.7045] | 0.0000 | a referencia |

### RealBokeh test v2

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) | DISTS | 217 | 217 | -0.1080 | [-0.1227, -0.0943] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 60000] | DISTS | 217 | 217 | -0.1073 | [-0.1218, -0.0934] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | DISTS | 217 | 217 | -0.1010 | [-0.1155, -0.0873] | 0.0000 | **o modelo** |
| sem filtro de SSIM [step 28000] | DISTS | 217 | 217 | -0.1006 | [-0.1153, -0.0871] | 0.0000 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | DISTS | 217 | 217 | -0.0946 | [-0.1098, -0.0808] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | DISTS | 217 | 217 | -0.0073 | [-0.0192, +0.0035] | 0.1962 | **o modelo** |
| LINHA DE IDENTIDADE | DISTS | 217 | 217 | +0.0201 | [+0.0050, +0.0339] | 0.0116 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 217 | 217 | +0.4058 | [+0.3841, +0.4270] | 0.0000 | a referencia |
| so rota c (a+c) [step 60000] | LPIPS | 217 | 217 | -0.2319 | [-0.2584, -0.2064] | 0.0000 | **o modelo** |
| so rota c (a+c) | LPIPS | 217 | 217 | -0.2313 | [-0.2581, -0.2053] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | LPIPS | 217 | 217 | -0.2171 | [-0.2436, -0.1922] | 0.0000 | **o modelo** |
| sem filtro de SSIM [step 28000] | LPIPS | 217 | 217 | -0.2141 | [-0.2401, -0.1895] | 0.0000 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | LPIPS | 217 | 217 | -0.2020 | [-0.2297, -0.1753] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 217 | 217 | -0.0219 | [-0.0377, -0.0073] | 0.0036 | **o modelo** |
| LINHA DE IDENTIDADE | LPIPS | 217 | 217 | +0.0133 | [-0.0116, +0.0379] | 0.2844 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 217 | 217 | +0.4202 | [+0.3870, +0.4525] | 0.0000 | a referencia |

### RealDOF

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) [step 45000] | DISTS | 50 | 50 | -0.0503 | [-0.0638, -0.0361] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 60000] | DISTS | 50 | 50 | -0.0492 | [-0.0628, -0.0350] | 0.0000 | **o modelo** |
| so rota c (a+c) | DISTS | 50 | 50 | -0.0490 | [-0.0623, -0.0351] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 30000] | DISTS | 50 | 50 | -0.0446 | [-0.0582, -0.0306] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 20000] | DISTS | 50 | 50 | -0.0409 | [-0.0551, -0.0265] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 10000] | DISTS | 50 | 50 | -0.0368 | [-0.0524, -0.0207] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | DISTS | 50 | 50 | +0.0000 | [-0.0162, +0.0159] | 0.9988 | a referencia |
| sem filtro de SSIM [step 28000] | DISTS | 50 | 50 | +0.0053 | [-0.0116, +0.0221] | 0.5510 | a referencia |
| fase 1, so sintetico (a) | DISTS | 50 | 50 | +0.0123 | [+0.0036, +0.0208] | 0.0072 | a referencia |
| kfix, K da rota b pela Eq. 3 | DISTS | 50 | 50 | +0.0203 | [+0.0056, +0.0347] | 0.0082 | a referencia |
| LINHA DE IDENTIDADE | DISTS | 50 | 50 | +0.0621 | [+0.0476, +0.0762] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 50 | 50 | +0.5080 | [+0.4883, +0.5270] | 0.0000 | a referencia |
| so rota c (a+c) [step 45000] | LPIPS | 50 | 50 | -0.1426 | [-0.1737, -0.1120] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 60000] | LPIPS | 50 | 50 | -0.1413 | [-0.1727, -0.1101] | 0.0000 | **o modelo** |
| so rota c (a+c) | LPIPS | 50 | 50 | -0.1406 | [-0.1715, -0.1099] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 30000] | LPIPS | 50 | 50 | -0.1312 | [-0.1629, -0.0999] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 20000] | LPIPS | 50 | 50 | -0.1253 | [-0.1574, -0.0937] | 0.0000 | **o modelo** |
| so rota c (a+c) [step 10000] | LPIPS | 50 | 50 | -0.1154 | [-0.1502, -0.0803] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | LPIPS | 50 | 50 | -0.0529 | [-0.0868, -0.0197] | 0.0008 | **o modelo** |
| sem filtro de SSIM [step 28000] | LPIPS | 50 | 50 | -0.0427 | [-0.0795, -0.0074] | 0.0138 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | LPIPS | 50 | 50 | -0.0153 | [-0.0454, +0.0145] | 0.3326 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 50 | 50 | +0.0105 | [-0.0084, +0.0291] | 0.2704 | a referencia |
| LINHA DE IDENTIDADE | LPIPS | 50 | 50 | +0.0603 | [+0.0322, +0.0878] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 50 | 50 | +0.5726 | [+0.5403, +0.6049] | 0.0000 | a referencia |
