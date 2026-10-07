#!/usr/bin/env python3
"""O alinhamento afim esta no espaco errado?

metrics.mde_metrics chama robust_affine_align(pred, target, mask, mode="full")
com pred e target em PROFUNDIDADE (metros), e o comentario do codigo diz que
isso e o "padrao MiDaS/DPT". Mas o MiDaS/DPT alinha em DISPARIDADE (1/Z). A
diferenca importa: minimos quadrados em Z e dominado pelos pixels LONGE, entao
numa cena que vai de 1 m a 47 m o ajuste serve o fundo e destroi o primeiro
plano -- e o AbsRel divide pelo Z verdadeiro, o que explode justamente perto.

Teste controlado, sem modelo: pega o GT de cada cena, cria uma "predicao" com
erro RELATIVO uniforme de 10% (o que um bom modelo faz), e mede o AbsRel depois
de alinhar nos dois espacos. Se o alinhamento em Z for o problema, o AbsRel vai
crescer com a faixa da cena mesmo o erro sendo uniforme por construcao.
"""
import glob, os
import numpy as np

RAIZ = "/data/spring_split/test"
rng = np.random.default_rng(42)

def alinha(p, t, m):
    """escala+deslocamento por minimos quadrados, no espaco em que vier."""
    pm, tm = p[m].mean(), t[m].mean()
    var = ((p[m]-pm)**2).mean()
    cov = ((p[m]-pm)*(t[m]-tm)).mean()
    s = cov/max(var, 1e-12)
    return s*p + (tm - s*pm)

seqs = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{RAIZ}/depth/*.npy")})
print(f"{'cena':<10s} {'p95/p5':>7s} {'AbsRel align-Z':>15s} {'AbsRel align-1/Z':>17s} {'fator':>7s}")
res = []
for s in seqs:
    fs = sorted(glob.glob(f"{RAIZ}/depth/{s}__*.npy"))[:8]
    aZ, aD, faixa = [], [], []
    for f in fs:
        t = np.load(f).astype(np.float64)
        m = np.isfinite(t) & (t > 0)
        if m.sum() < 1000: continue
        # predicao com erro RELATIVO uniforme de 10%, identica nos dois testes
        pred = t * np.exp(rng.normal(0, 0.10, t.shape))
        # 1) alinhar em profundidade (o que o codigo faz)
        pz = alinha(pred, t, m)
        aZ.append(np.abs(pz[m]-t[m]).sum()/t[m].sum() if False else np.mean(np.abs(pz[m]-t[m])/t[m]))
        # 2) alinhar em disparidade (padrao MiDaS/DPT), depois voltar
        dp, dt = 1.0/np.maximum(pred,1e-6), 1.0/np.maximum(t,1e-6)
        dz = alinha(dp, dt, m)
        pd = 1.0/np.maximum(dz, 1e-6)
        aD.append(np.mean(np.abs(pd[m]-t[m])/t[m]))
        v = t[m]; faixa.append(np.percentile(v,95)/max(np.percentile(v,5),1e-6))
    if not aZ: continue
    res.append((s, np.mean(faixa), np.mean(aZ), np.mean(aD)))
for s, fx, z, d in sorted(res, key=lambda x:-x[1]):
    print(f"{s:<10s} {fx:7.1f} {z:15.4f} {d:17.4f} {z/max(d,1e-9):7.2f}x")
z=[r[2] for r in res]; d=[r[3] for r in res]
print(f"\nmedia sobre as 13 cenas: align-Z={np.mean(z):.4f}  align-1/Z={np.mean(d):.4f}")
print(f"o erro verdadeiro, por construcao, e ~0.08 (relativo de 10% em log)")
