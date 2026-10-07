#!/usr/bin/env python3
"""
scripts/diagnostico_clamps.py
=============================
Mede, no dataset real, a magnitude bruta de cada termo da perda e a fração de pixels
saturada pelos tetos atuais. Recomenda tetos calibrados para o dataset em uso.

Por que existe: `gauss_clamp` e `metric_clamp` foram calibrados no Hypersim, que é
interior com profundidade de 1 a 10 metros. A curvatura escala com o inverso do quadrado
do comprimento, então mudar a faixa de profundidade muda a distribuição de K em ordens de
magnitude. Um teto que era generoso num dataset passa a cortar o sinal em outro.

O efeito de um teto apertado demais não é apenas encolher o valor da perda. Pixels
saturados têm **gradiente zero**: eles deixam de contribuir para o aprendizado. Se a maior
parte dos pixels satura, o termo vira uma constante e o experimento fica sem potência
justamente no termo que se quer testar.

Roda em inferência, sem treinar. Custa minutos.

Uso:
    python scripts/diagnostico_clamps.py \\
        --data-root /data/spring_prep/train \\
        --n-lotes 20 --size 512
"""

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.dataset import HighQualityDepthDataset          # noqa: E402
from riemann import losses as L                               # noqa: E402


# ---------------------------------------------------------------------------
# Chamada DEFENSIVA das funcoes de perda.
#
# Este script e usado para diagnosticar versoes diferentes do repositorio, que podem
# nao ter todos os parametros que versoes mais novas introduziram (por exemplo
# `clamp_val` em metric_tensor_loss, ou `mode` em robust_affine_align). Em vez de
# quebrar, detectamos a assinatura em tempo de execucao e passamos apenas o que existe.
# ---------------------------------------------------------------------------
def chamar(fn, *args, **kwargs):
    """Chama fn descartando os kwargs que a assinatura dela nao aceita."""
    if fn is None:
        return None
    try:
        aceitos = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        aceitos = set()
    filtrados = {k: v for k, v in kwargs.items() if k in aceitos}
    return fn(*args, **filtrados)


def curvatura_inline(depth, sigma=0.5, h=None):
    """
    Curvatura Gaussiana calculada AQUI, sem teto, independente da versao do repositorio.

    Serve para medir a distribuicao verdadeira de |K| no ground truth e recomendar o
    teto. Usar a funcao do repositorio nao serviria, porque ela ja aplica um teto e
    esconderia justamente a cauda que queremos ver.
    """
    if sigma and sigma > 0:
        r = max(1, int(3 * sigma))
        x = torch.arange(-r, r + 1, dtype=depth.dtype, device=depth.device)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k = k / k.sum()
        d = torch.nn.functional.conv2d(
            torch.nn.functional.pad(depth, (r, r, 0, 0), mode="replicate"),
            k.view(1, 1, 1, -1))
        depth = torch.nn.functional.conv2d(
            torch.nn.functional.pad(d, (0, 0, r, r), mode="replicate"),
            k.view(1, 1, -1, 1))
    if h is None:
        h = 1.0 / max(depth.shape[-2:])
    dp = torch.nn.functional.pad(depth, (1, 1, 1, 1), mode="replicate")
    zx = (dp[:, :, 1:-1, 2:] - dp[:, :, 1:-1, :-2]) / (2 * h)
    zy = (dp[:, :, 2:, 1:-1] - dp[:, :, :-2, 1:-1]) / (2 * h)
    zxx = (dp[:, :, 1:-1, 2:] - 2 * depth + dp[:, :, 1:-1, :-2]) / (h ** 2)
    zyy = (dp[:, :, 2:, 1:-1] - 2 * depth + dp[:, :, :-2, 1:-1]) / (h ** 2)
    zxy = (dp[:, :, 2:, 2:] - dp[:, :, 2:, :-2]
           - dp[:, :, :-2, 2:] + dp[:, :, :-2, :-2]) / (4 * h ** 2)
    return (zxx * zyy - zxy ** 2) / ((1.0 + zx ** 2 + zy ** 2) ** 2 + 1e-12)


def percentis(t, qs=(50, 90, 99, 99.5, 99.9)):
    v = t[torch.isfinite(t)].abs().flatten()
    if v.numel() == 0:
        return {q: float("nan") for q in qs}
    v = v[torch.randperm(v.numel())[:2_000_000]]  # amostra p/ caber em memoria
    return {q: float(torch.quantile(v.float(), q / 100.0)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-lotes", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gauss-clamp", type=float, default=50.0,
                    help="teto atual, para medir a fracao saturada")
    ap.add_argument("--metric-clamp", type=float, default=100.0)
    ap.add_argument("--gauss-sigma", type=float, default=0.5)
    ap.add_argument("--ruido", type=float, default=0.05,
                    help="perturbacao relativa aplicada ao GT para simular a predicao")
    args = ap.parse_args()

    ds = HighQualityDepthDataset(args.data_root, size=(args.size, args.size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[dados] {len(ds)} imagens em {args.data_root}")

    acc_k, acc_fro, faixas = [], [], []
    termos = {k: [] for k in ["berhu", "grad", "normal", "gauss", "geod", "metric"]}
    sat_g = sat_m = tot = 0

    for i, b in enumerate(dl):
        if i >= args.n_lotes:
            break
        gt = b["depth"].to(args.device).float()
        m = b["mask"].to(args.device)
        # "predicao" simulada: GT perturbado e realinhado, como no treino
        pred = chamar(L.robust_affine_align,
                      gt * (1 + args.ruido * torch.randn_like(gt)), gt, m, mode="full")

        v = gt[m > 0.5]
        if v.numel():
            faixas.append((float(v.min()), float(torch.median(v)), float(v.max())))

        # curvatura e tensor metrico SEM teto, para ver a distribuicao verdadeira
        k = curvatura_inline(gt, args.gauss_sigma)
        acc_k.append(k.detach().cpu())
        sat_g += int((k.abs() > args.gauss_clamp).sum())
        tot += int(k.numel())

        for nome, fn, extra in [
                ("berhu",  getattr(L, "berhu_loss", None), {}),
                ("grad",   getattr(L, "grad_loss", None), {}),
                ("normal", getattr(L, "normal_loss", None), {}),
                ("gauss",  getattr(L, "gauss_loss", None),
                 {"smooth_sigma": args.gauss_sigma, "clamp_val": args.gauss_clamp}),
                ("geod",   getattr(L, "geodesic_loss", None), {}),
                ("metric", getattr(L, "metric_tensor_loss", None),
                 {"clamp_val": args.metric_clamp})]:
            if fn is None:
                continue
            try:
                termos[nome].append(float(chamar(fn, pred, gt, m, **extra)))
            except Exception as e:      # versao antiga com assinatura diferente
                if i == 0:
                    print(f"  [aviso] termo '{nome}' nao pode ser medido nesta versao: {e}")

    if faixas:
        lo = np.median([f[0] for f in faixas]); md = np.median([f[1] for f in faixas])
        hi = np.median([f[2] for f in faixas])
        print(f"[profundidade] min {lo:.2f} / mediana {md:.2f} / max {hi:.2f} m")

    K = torch.cat([a.flatten() for a in acc_k])
    pk = percentis(K)
    print("\n===== DISTRIBUICAO DE |K| NO GROUND TRUTH (sem teto) =====")
    for q, v in pk.items():
        print(f"  p{q:<5}: {v:12.2f}")
    print(f"  fracao de pixels acima do teto atual ({args.gauss_clamp:g}): "
          f"{100*sat_g/max(tot,1):.2f}%")

    print("\n===== MAGNITUDE BRUTA DE CADA TERMO =====")
    print(f"  {'termo':8s} {'media':>10s} {'teto':>10s}  situacao")
    tetos = {"gauss": 2 * args.gauss_clamp, "metric": args.metric_clamp,
             "normal": 2.0}
    for k_, vs in termos.items():
        if not vs:
            continue
        mv = float(np.mean(vs))
        teto = tetos.get(k_)
        if teto is None:
            sit = "sem teto"
        elif mv > 0.9 * teto:
            sit = "SATURADO"
        elif mv > 0.5 * teto:
            sit = "perto do teto"
        else:
            sit = "folgado"
        print(f"  {k_:8s} {mv:10.2f} {('-' if teto is None else f'{teto:.0f}'):>10s}  {sit}")

    rec_g = pk[99.5]
    print("\n===== RECOMENDACAO =====")
    print(f"  gauss_clamp sugerido : {rec_g:.0f}   (percentil 99.5 de |K| no GT)")
    print( "     O teto deve cortar apenas a cauda extrema. Cortar o corpo da")
    print( "     distribuicao zera o gradiente na maioria dos pixels e o termo vira")
    print( "     constante, o que deixa o experimento sem potencia.")
    if args.gauss_clamp < 0.2 * rec_g:
        print(f"  [ALERTA] o teto atual ({args.gauss_clamp:g}) e muito menor que o "
              f"sugerido. O termo de curvatura esta sendo cortado.")
    print("\n  Sobre o termo grad: ele NAO tem teto e escala com a faixa de")
    print("  profundidade, entao domina o train_raw em datasets de alcance longo.")
    print("  Isso nao invalida o treino, porque a normalizacao adaptativa por termo")
    print("  reequilibra as contribuicoes. Mas significa que um train_raw alto NAO e")
    print("  evidencia de que a curvatura esteja saturada: para isso, olhe a fracao")
    print("  de pixels acima do teto, reportada acima.")


if __name__ == "__main__":
    main()
