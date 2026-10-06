# Tabela final de bokeh (formato da Tab. 3 do GenRefocus)

Metricas por imagem, com IC95 por bootstrap percentil (10.000 reamostras).
`LVCorr` esta na convencao do PAPER (quanto maior melhor): e o simetrico do
Pearson cru, porque mais K significa mais desfoque e MENOS variancia do Laplaciano.

> **Ressalva de comparabilidade.** O LF-Bokeh do paper nao foi liberado, entao a
> mesa `LF-Bokeh reproduzido` e uma reconstrucao do protocolo. Os pesos OFICIAIS
> medem LPIPS 0,2047 nela contra 0,0833 publicado. As comparacoes entre modelos
> valem (mesmo dado, mesmo pipeline); os valores absolutos nao sao os do paper.


## EBB400 (foto real, uma cena por linha) (400 imagens de 400 cenas)


| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | 0.1155 [0.1094, 0.1221] | 0.0777 | 0.9791 | +0.6328 | +0.1630 |
| nosso original, fase 2 (a+b+c) | 0.1174 [0.1111, 0.1243] | 0.0794 | 0.9788 | +0.7667 | +0.1611 |
| sem filtro de SSIM [step 28000] | 0.1242 [0.1171, 0.1315] | 0.0830 | 0.9778 | +0.8520 | +0.1544 |
| kfix, K da rota b pela Eq. 3 | 0.1548 [0.1455, 0.1645] | 0.0961 | 0.9726 | +0.0753 | +0.1237 |
| oficial do paper | 0.1687 [0.1581, 0.1802] | 0.1220 | 0.9647 | +0.8477 | +0.1098 |
| fase 1, so sintetico (a) | 0.1735 [0.1634, 0.1842] | 0.1262 | 0.9585 | +0.8512 | +0.1051 |
| LINHA DE IDENTIDADE | 0.2785 [0.2651, 0.2924] | 0.1558 | 0.9528 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.7693 [0.7550, 0.7834] | 0.6003 | 0.6586 | -0.9617 | -0.4908 |

## LF-Bokeh reproduzido (BLB) (500 imagens de 10 cenas)
  **O IC95 e agrupado por cena: o tamanho efetivo de amostra aqui e 10, nao 500.**

| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | 0.1719 [0.1538, 0.1925] | 0.1179 | 0.9638 | +0.7607 | +0.0652 |
| oficial do paper | 0.1812 [0.1574, 0.2032] | 0.1170 | 0.9584 | +0.8934 | +0.0559 |
| fase 1, so sintetico (a) | 0.1968 [0.1735, 0.2174] | 0.1266 | 0.9591 | +0.8808 | +0.0403 |
| nosso original, fase 2 (a+b+c) | 0.2072 [0.1833, 0.2344] | 0.1423 | 0.9559 | +0.5889 | +0.0299 |
| sem filtro de SSIM [step 28000] | 0.2103 [0.1894, 0.2364] | 0.1427 | 0.9559 | +0.6234 | +0.0268 |
| kfix, K da rota b pela Eq. 3 | 0.2261 [0.2073, 0.2484] | 0.1586 | 0.9498 | +0.8625 | +0.0110 |
| LINHA DE IDENTIDADE | 0.2371 [0.2193, 0.2549] | 0.1745 | 0.9477 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.8342 [0.7733, 0.8814] | 0.7016 | 0.7008 | -0.9610 | -0.5971 |

## RealBokeh test v2 (217 imagens de 217 cenas)


| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | 0.1115 [0.1036, 0.1200] | 0.0805 | 0.9829 | +0.8264 | +0.2471 |
| nosso original, fase 2 (a+b+c) | 0.1248 [0.1159, 0.1343] | 0.0858 | 0.9809 | +0.8520 | +0.2339 |
| sem filtro de SSIM [step 28000] | 0.1289 [0.1199, 0.1384] | 0.0865 | 0.9806 | +0.8805 | +0.2298 |
| kfix, K da rota b pela Eq. 3 | 0.1397 [0.1289, 0.1510] | 0.0916 | 0.9787 | +0.5158 | +0.2189 |
| fase 1, so sintetico (a) | 0.2676 [0.2462, 0.2893] | 0.1594 | 0.9421 | +0.8520 | +0.0911 |
| oficial do paper | 0.2867 [0.2596, 0.3137] | 0.1669 | 0.9433 | +0.8400 | +0.0720 |
| LINHA DE IDENTIDADE | 0.3587 [0.3395, 0.3777] | 0.2086 | 0.9218 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.7661 [0.7466, 0.7853] | 0.5892 | 0.6956 | -0.9783 | -0.4075 |

## RealDOF (50 imagens de 50 cenas)


| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |
|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | 0.1178 [0.1013, 0.1370] | 0.0876 | 0.9712 | -0.2962 | +0.2101 |
| oficial do paper | 0.1900 [0.1689, 0.2129] | 0.1167 | 0.9433 | +0.9360 | +0.1379 |
| fase 1, so sintetico (a) | 0.2087 [0.1871, 0.2309] | 0.1288 | 0.9439 | +0.8921 | +0.1193 |
| nosso original, fase 2 (a+b+c) | 0.2157 [0.1897, 0.2408] | 0.1417 | 0.9581 | +0.3746 | +0.1122 |
| sem filtro de SSIM [step 28000] | 0.2243 [0.1974, 0.2516] | 0.1459 | 0.9589 | +0.6383 | +0.1036 |
| kfix, K da rota b pela Eq. 3 | 0.2508 [0.2264, 0.2750] | 0.1612 | 0.9527 | -0.3764 | +0.0772 |
| LINHA DE IDENTIDADE | 0.3279 [0.3036, 0.3512] | 0.2030 | 0.9471 | +0.0000 | +0.0000 |
| sem treino (FLUX.1-dev cru) | 0.8520 [0.8324, 0.8711] | 0.6608 | 0.6800 | -0.8656 | -0.5241 |


## Teste pareado contra `oficial do paper`, cena a cena

Diferenca media por cena, IC95 e p bilateral por bootstrap. Negativo em LPIPS/DISTS = o modelo e melhor.


### EBB400 (foto real, uma cena por linha)

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | DISTS | 400 | 400 | -0.0442 | [-0.0510, -0.0382] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | DISTS | 400 | 400 | -0.0426 | [-0.0493, -0.0365] | 0.0000 | **o modelo** |
| sem filtro de SSIM [step 28000] | DISTS | 400 | 400 | -0.0389 | [-0.0457, -0.0329] | 0.0000 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | DISTS | 400 | 400 | -0.0259 | [-0.0329, -0.0195] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | DISTS | 400 | 400 | +0.0042 | [-0.0023, +0.0102] | 0.1950 | a referencia |
| LINHA DE IDENTIDADE | DISTS | 400 | 400 | +0.0339 | [+0.0271, +0.0404] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 400 | 400 | +0.4783 | [+0.4644, +0.4918] | 0.0000 | a referencia |
| so rota c (a+c) [step 60000] | LPIPS | 400 | 400 | -0.0533 | [-0.0626, -0.0449] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | LPIPS | 400 | 400 | -0.0513 | [-0.0612, -0.0422] | 0.0000 | **o modelo** |
| sem filtro de SSIM [step 28000] | LPIPS | 400 | 400 | -0.0446 | [-0.0543, -0.0353] | 0.0000 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | LPIPS | 400 | 400 | -0.0139 | [-0.0245, -0.0036] | 0.0064 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 400 | 400 | +0.0047 | [-0.0038, +0.0125] | 0.2616 | a referencia |
| LINHA DE IDENTIDADE | LPIPS | 400 | 400 | +0.1098 | [+0.0978, +0.1218] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 400 | 400 | +0.6006 | [+0.5824, +0.6188] | 0.0000 | a referencia |

### LF-Bokeh reproduzido (BLB)

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | DISTS | 500 | 10 | +0.0009 | [-0.0121, +0.0127] | 0.8708 | a referencia |
| fase 1, so sintetico (a) | DISTS | 500 | 10 | +0.0096 | [+0.0038, +0.0166] | 0.0002 | a referencia |
| nosso original, fase 2 (a+b+c) | DISTS | 500 | 10 | +0.0253 | [+0.0072, +0.0473] | 0.0026 | a referencia |
| sem filtro de SSIM [step 28000] | DISTS | 500 | 10 | +0.0258 | [+0.0103, +0.0464] | 0.0002 | a referencia |
| kfix, K da rota b pela Eq. 3 | DISTS | 500 | 10 | +0.0416 | [+0.0283, +0.0590] | 0.0000 | a referencia |
| LINHA DE IDENTIDADE | DISTS | 500 | 10 | +0.0575 | [+0.0450, +0.0726] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 500 | 10 | +0.5846 | [+0.5290, +0.6324] | 0.0000 | a referencia |
| so rota c (a+c) [step 60000] | LPIPS | 500 | 10 | -0.0094 | [-0.0357, +0.0148] | 0.4832 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 500 | 10 | +0.0155 | [+0.0104, +0.0217] | 0.0000 | a referencia |
| nosso original, fase 2 (a+b+c) | LPIPS | 500 | 10 | +0.0260 | [-0.0068, +0.0690] | 0.1456 | a referencia |
| sem filtro de SSIM [step 28000] | LPIPS | 500 | 10 | +0.0291 | [+0.0003, +0.0705] | 0.0454 | a referencia |
| kfix, K da rota b pela Eq. 3 | LPIPS | 500 | 10 | +0.0449 | [+0.0165, +0.0833] | 0.0000 | a referencia |
| LINHA DE IDENTIDADE | LPIPS | 500 | 10 | +0.0559 | [+0.0309, +0.0883] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 500 | 10 | +0.6529 | [+0.5992, +0.6984] | 0.0000 | a referencia |

### RealBokeh test v2

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | DISTS | 217 | 217 | -0.0864 | [-0.1014, -0.0725] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | DISTS | 217 | 217 | -0.0811 | [-0.0965, -0.0672] | 0.0000 | **o modelo** |
| sem filtro de SSIM [step 28000] | DISTS | 217 | 217 | -0.0803 | [-0.0956, -0.0667] | 0.0000 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | DISTS | 217 | 217 | -0.0752 | [-0.0912, -0.0608] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | DISTS | 217 | 217 | -0.0075 | [-0.0212, +0.0051] | 0.2662 | **o modelo** |
| LINHA DE IDENTIDADE | DISTS | 217 | 217 | +0.0417 | [+0.0262, +0.0559] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 217 | 217 | +0.4224 | [+0.3996, +0.4441] | 0.0000 | a referencia |
| so rota c (a+c) [step 60000] | LPIPS | 217 | 217 | -0.1751 | [-0.2011, -0.1511] | 0.0000 | **o modelo** |
| nosso original, fase 2 (a+b+c) | LPIPS | 217 | 217 | -0.1619 | [-0.1874, -0.1383] | 0.0000 | **o modelo** |
| sem filtro de SSIM [step 28000] | LPIPS | 217 | 217 | -0.1578 | [-0.1834, -0.1340] | 0.0000 | **o modelo** |
| kfix, K da rota b pela Eq. 3 | LPIPS | 217 | 217 | -0.1469 | [-0.1743, -0.1213] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 217 | 217 | -0.0191 | [-0.0381, -0.0015] | 0.0328 | **o modelo** |
| LINHA DE IDENTIDADE | LPIPS | 217 | 217 | +0.0720 | [+0.0449, +0.0985] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 217 | 217 | +0.4794 | [+0.4470, +0.5109] | 0.0000 | a referencia |

### RealDOF

| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |
|---|---|---|---|---|---|---|---|
| so rota c (a+c) [step 60000] | DISTS | 50 | 50 | -0.0291 | [-0.0398, -0.0185] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | DISTS | 50 | 50 | +0.0121 | [+0.0078, +0.0169] | 0.0000 | a referencia |
| nosso original, fase 2 (a+b+c) | DISTS | 50 | 50 | +0.0249 | [+0.0092, +0.0396] | 0.0018 | a referencia |
| sem filtro de SSIM [step 28000] | DISTS | 50 | 50 | +0.0292 | [+0.0127, +0.0446] | 0.0012 | a referencia |
| kfix, K da rota b pela Eq. 3 | DISTS | 50 | 50 | +0.0444 | [+0.0315, +0.0566] | 0.0000 | a referencia |
| LINHA DE IDENTIDADE | DISTS | 50 | 50 | +0.0863 | [+0.0745, +0.0977] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | DISTS | 50 | 50 | +0.5441 | [+0.5214, +0.5652] | 0.0000 | a referencia |
| so rota c (a+c) [step 60000] | LPIPS | 50 | 50 | -0.0722 | [-0.0949, -0.0498] | 0.0000 | **o modelo** |
| fase 1, so sintetico (a) | LPIPS | 50 | 50 | +0.0186 | [+0.0082, +0.0290] | 0.0004 | a referencia |
| nosso original, fase 2 (a+b+c) | LPIPS | 50 | 50 | +0.0257 | [-0.0061, +0.0561] | 0.1148 | a referencia |
| sem filtro de SSIM [step 28000] | LPIPS | 50 | 50 | +0.0343 | [+0.0002, +0.0668] | 0.0486 | a referencia |
| kfix, K da rota b pela Eq. 3 | LPIPS | 50 | 50 | +0.0607 | [+0.0330, +0.0874] | 0.0000 | a referencia |
| LINHA DE IDENTIDADE | LPIPS | 50 | 50 | +0.1379 | [+0.1118, +0.1629] | 0.0000 | a referencia |
| sem treino (FLUX.1-dev cru) | LPIPS | 50 | 50 | +0.6620 | [+0.6287, +0.6935] | 0.0000 | a referencia |
