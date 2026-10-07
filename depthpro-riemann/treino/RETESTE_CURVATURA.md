# Reteste da curvatura — passo a passo

Branch: `fix-geometry`. Dois arquivos novos, não alteram nada existente:

- `riemann/geometry.py`
- `scripts/test_geometry_metrica.py`

---

## Por que

`gaussian_curvature` usa `h = 1/max(H,W)` — coordenadas de imagem normalizadas em [0,1] —
com profundidade em metros. Os eixos ficam em unidades diferentes e o resultado não é
curvatura de superfície.

Esfera de raio 2 m a 10 m: K verdadeiro = 1/R² = **0,25**. A fórmula atual devolve **~10**.

Isso explica a cauda de |K| até 38 milhões e os 40% de pixels saturando no teto 50.

O `test_geometry.py` não pegava porque monta a esfera com x, y e z nas mesmas unidades.
Só quebra com dado real.

Correção: retroprojeção real, `S(u,v) = ((u−cx)·D/fx, (v−cy)·D/fy, D)`, com as duas formas
fundamentais. Precisa da focal em pixels.

---

## Passo 1 — Validar a implementação (2 min)

```bash
python scripts/test_geometry_metrica.py
```

Esperado:

| seção | critério |
|---|---|
| [1] esfera | erro < 0,2% nas 8 configurações |
| [2] plano | passa em todas as 5 inclinações |
| [3] cilindro | erro < 0,1% nos 3 raios |
| [4] invariância | dispersão < 1% entre câmeras |
| [5] normais e área | passa |

Se falhar aqui, **para e avisa**. É problema de implementação, não de dado.

A seção [6] imprime o valor antigo e o correto lado a lado. Vale guardar para o report.

---

## Passo 2 — Focal do Spring (1 min)

O `prepare_spring.py` já lê do `intrinsics.txt` e imprime a mediana no log da Fase 0.
Recupera esse valor.

**Atenção:** a focal é para a resolução original (1920 de largura). Ao treinar em
`--size 512`:

```
f_treino = f_original × 512 / 1920
```

Esquecer isso erra a curvatura por um fator (512/1920)² ≈ 0,071. Esse erro **não aparece**
se testar em uma única resolução.

---

## Passo 3 — Medir |K| com a fórmula correta (5 min)

Preencher `F_ORIG` com o valor do passo 2.

```python
import torch, sys
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from riemann.dataset import HighQualityDepthDataset
from riemann.geometry import surface_curvatures

RAIZ   = '/data/spring_prep/test'
SIZE   = 512
F_ORIG = 1234.0        # <-- focal reportada pelo prepare_spring
W_ORIG = 1920

f = F_ORIG * SIZE / W_ORIG
print(f'focal na resolucao de trabalho: {f:.1f} px')

ds = HighQualityDepthDataset(RAIZ, size=(SIZE, SIZE))
dl = DataLoader(ds, batch_size=2, num_workers=0)

Ks = []
for i, b in enumerate(dl):
    if i >= 10:
        break
    r = surface_curvatures(b['depth'].float(), fx=f, smooth_sigma=0.5, clamp_val=None)
    Ks.append(r['K'].flatten())

K = torch.cat(Ks).abs()
K = K[torch.isfinite(K)]

print('percentis de |K|:')
for q in [50, 90, 99, 99.5, 99.9]:
    print(f'  p{q}: {float(torch.quantile(K, q/100)):12.4f}')

print('fracao acima de cada teto:')
for teto in [1, 5, 20, 50]:
    print(f'  > {teto:3d}: {100*float((K > teto).float().mean()):5.2f}%')
```

**O que esperar.** Com a fórmula correta K = 1/R². Um objeto de raio 1 m dá K = 1; de
20 cm dá K = 25. A mediana deve ficar em ordem de unidade, não de milhares.

Se vier gigante, **manda o resultado antes de gastar GPU**.

Com esses percentis a gente escolhe o teto com dado. Expectativa: 5,0 fica folgado, contra
os 40% de saturação anteriores.

---

## Passo 4 — Reexecutar (só depois de 1 a 3 baterem)

Rodar de novo `B0_berhu` (controle) e `B3` no Spring, trocando `gauss_loss` por
`gauss_loss_metrica` e usando o teto escolhido no passo 3.

Mesmo protocolo, mesmo split, para ser comparável com o anterior.

**Não pular direto para cá.** São ~4 h de GPU em cima de um número que ainda não foi
conferido.

---

## Ressalva

O `h` normalizado está no código desde o início e passou por várias revisões sem ser
notado. Os três resultados negativos (Hypersim, DIODE, Spring) foram obtidos com um termo
que não media curvatura.

Isso não valida a hipótese — significa que ela ainda não foi testada. Pode dar o mesmo nulo
com a fórmula correta, e aí a conclusão é a mesma, mas defensável.

---

## Retorno

Manda o resultado dos passos 1 a 3. O passo 4 a gente decide junto.
