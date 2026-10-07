#!/usr/bin/env python3
"""
scripts/make_figures.py
=======================
Gera os artefatos visuais do trabalho: mapas de profundidade e mapas de curvatura
Gaussiana, com comparação lado a lado entre o modelo zero-shot e o modelo treinado.

Produz três coisas:
  1. Painel comparativo por imagem (PNG): RGB | GT | zero-shot | treinado | curvatura
     e, opcionalmente, um recorte ampliado numa região de borda.
  2. Mapas individuais em PNG colorizado e em .npy (valores crus), para reuso.
  3. Uma figura resumo com várias cenas empilhadas, pronta para o paper.

Decisões de visualização que importam para a honestidade da figura:
  - A profundidade predita é alinhada ao GT pela MESMA transformação afim usada nas
    métricas. Sem isso, dois modelos com escalas diferentes ficariam com cores
    diferentes por motivo irrelevante.
  - Zero-shot e treinado compartilham a MESMA escala de cor dentro de cada linha, para
    que a comparação seja visual e não artefato de normalização.
  - A curvatura usa colormap divergente e escala SIMÉTRICA em torno de zero, porque o
    sinal da curvatura Gaussiana tem significado geométrico (elíptico vs hiperbólico).
    O limite é um percentil robusto, não o máximo, para um outlier não achatar o mapa.

Uso:
    python scripts/make_figures.py \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --weights /workspace/runs/champion/seed_0/best.pt \\
        --data-root /data/hypersim/test \\
        --out-dir /workspace/figuras \\
        --n-imagens 6 --variant heads

Sem --weights, gera apenas as saídas do modelo base (útil para conferir o pipeline).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.dataset import HighQualityDepthDataset      # noqa: E402
from riemann.losses import gaussian_curvature, robust_affine_align  # noqa: E402
from riemann.model import build_model                    # noqa: E402
from riemann.metrics import all_metrics                  # noqa: E402

CMAP_PROF = "turbo"      # profundidade
CMAP_CURV = "RdBu_r"     # curvatura: divergente, zero no branco


def desnormalizar_rgb(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Desfaz a normalização ImageNet para exibir a imagem."""
    a = t.detach().cpu().numpy().transpose(1, 2, 0)
    a = a * np.array(std) + np.array(mean)
    return np.clip(a, 0, 1)


def limite_robusto(k, pct=98.0):
    """Limite simétrico para a curvatura, por percentil (ignora outliers)."""
    v = np.abs(k[np.isfinite(k)])
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, pct))
    return lim if lim > 1e-8 else 1.0


def salvar_mapa(arr, caminho, cmap, vmin=None, vmax=None, divergente=False):
    """Salva um mapa colorizado sem eixos nem margens."""
    fig, ax = plt.subplots(figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=100)
    if divergente:
        lim = vmax if vmax is not None else limite_robusto(arr)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        ax.imshow(arr, cmap=cmap, norm=norm)
    else:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(caminho, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def prever(model, rgb, tamanho_alvo, device):
    """Forward do modelo, com a saída reinterpolada para o tamanho do GT."""
    p = model(rgb.to(device)).float()
    if p.shape[-2:] != tamanho_alvo:
        p = torch.nn.functional.interpolate(
            p, size=tamanho_alvo, mode="bilinear", align_corners=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="pesos base do DepthPro")
    ap.add_argument("--weights", default=None,
                    help="best.pt da config treinada (omita para so o modelo base)")
    ap.add_argument("--data-root", required=True, help="pasta com rgb/ e depth/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="heads", choices=["heads", "heads_final", "heads_lora"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n-imagens", type=int, default=6,
                    help="quantas imagens gerar (espacadas ao longo do conjunto)")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="indices especificos, em vez do espacamento automatico")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--curv-sigma", type=float, default=2.0,
                    help="suavizacao antes da 2a derivada. 0.5 (antigo) deixa o mapa "
                         "dominado por ruido: a 2a derivada amplifica ruido por 1/h^2. "
                         "Medido: com sigma=0.5 o ruido numa parede PLANA fica ~18000 "
                         "contra sinal ~55000 na borda (SNR 3); com sigma=2.0 o ruido cai "
                         "para ~50 e o SNR sobe para 10.")
    ap.add_argument("--curv-clamp", type=float, default=0.0,
                    help="teto por pixel da curvatura. 0 = SEM clamp (padrao para "
                         "VISUALIZACAO). O clamp de 50 serve a LOSS, para estabilidade de "
                         "gradiente, mas na figura ele destroi o mapa: o sinal real de "
                         "borda chega a ~500, entao tudo satura. Medido: sigma=0.5 com "
                         "clamp=50 deixa 95%% dos pixels saturados; sigma=2.0 sem clamp "
                         "deixa 2%%.")
    ap.add_argument("--zoom", type=int, nargs=4, default=None,
                    metavar=("Y", "X", "H", "W"),
                    help="recorte ampliado de borda, ex.: --zoom 180 220 120 120")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "mapas").mkdir(parents=True, exist_ok=True)
    (out / "paineis").mkdir(parents=True, exist_ok=True)

    ds = HighQualityDepthDataset(args.data_root, size=(args.size, args.size))
    if args.indices:
        idxs = [i for i in args.indices if 0 <= i < len(ds)]
    else:
        n = min(args.n_imagens, len(ds))
        idxs = list(np.linspace(0, len(ds) - 1, n).astype(int))
    print(f"[dados] {len(ds)} imagens; gerando figuras para {len(idxs)}: {idxs}")

    print("[modelo] carregando zero-shot...")
    base = build_model(args.checkpoint, variant=args.variant, device=args.device)
    base.eval()

    treinado = None
    if args.weights:
        print(f"[modelo] carregando treinado: {args.weights}")
        treinado = build_model(args.checkpoint, variant=args.variant, device=args.device)
        # best.pt guarda so os parametros treinaveis -> strict=False
        faltando = treinado.load_state_dict(
            torch.load(args.weights, map_location=args.device), strict=False)
        n_carregado = len(getattr(faltando, "unexpected_keys", [])) if faltando else 0
        print(f"    chaves inesperadas: {n_carregado} (0 e o esperado)")
        treinado.eval()

    linhas_resumo = []

    for idx in idxs:
        amostra = ds[idx]
        chave = amostra["key"]
        rgb_t = amostra["rgb"].unsqueeze(0)
        gt = amostra["depth"].unsqueeze(0).float().to(args.device)
        mask = amostra["mask"].unsqueeze(0).to(args.device)
        hw = gt.shape[-2:]

        rgb_vis = desnormalizar_rgb(amostra["rgb"])
        gt_np = gt[0, 0].cpu().numpy()
        valido = mask[0, 0].cpu().numpy() > 0.5

        # escala de cor comum: definida pelo GT valido
        vmin = float(np.percentile(gt_np[valido], 2)) if valido.any() else None
        vmax = float(np.percentile(gt_np[valido], 98)) if valido.any() else None

        colunas = [("RGB", rgb_vis, "rgb", None),
                   ("Profundidade GT", np.where(valido, gt_np, np.nan), "prof", None)]

        preds = {}
        for nome, modelo in [("zero-shot", base), ("treinado", treinado)]:
            if modelo is None:
                continue
            p = prever(modelo, rgb_t, hw, args.device)
            # alinhar a predicao ao GT com a MESMA transformacao afim das metricas,
            # senao as cores diferem por escala e nao por qualidade
            p_al = robust_affine_align(p, gt, mask, mode="full")
            preds[nome] = p_al
            colunas.append((f"Profundidade ({nome})",
                            np.where(valido, p_al[0, 0].cpu().numpy(), np.nan),
                            "prof", None))

        # curvatura do melhor modelo disponivel
        nome_curv = "treinado" if treinado is not None else "zero-shot"
        K = gaussian_curvature(preds[nome_curv], args.curv_sigma, (args.curv_clamp or 1e12))
        K_np = K[0, 0].cpu().numpy()
        lim_k = limite_robusto(K_np)
        colunas.append((f"Curvatura Gaussiana ({nome_curv})", K_np, "curv", lim_k))

        # ---- salvar mapas individuais -------------------------------------
        np.save(out / "mapas" / f"{chave}_curvatura.npy", K_np.astype(np.float32))
        salvar_mapa(K_np, out / "mapas" / f"{chave}_curvatura.png",
                    CMAP_CURV, vmax=lim_k, divergente=True)
        for nome, p_al in preds.items():
            d = p_al[0, 0].cpu().numpy()
            np.save(out / "mapas" / f"{chave}_profundidade_{nome}.npy", d.astype(np.float32))
            salvar_mapa(d, out / "mapas" / f"{chave}_profundidade_{nome}.png",
                        CMAP_PROF, vmin=vmin, vmax=vmax)

        # ---- painel comparativo -------------------------------------------
        n_col = len(colunas)
        n_lin = 2 if args.zoom else 1
        fig, axes = plt.subplots(n_lin, n_col,
                                 figsize=(3.2 * n_col, 3.6 * n_lin),
                                 layout="constrained")
        axes = np.atleast_2d(axes)

        for j, (titulo, dado, tipo, lim) in enumerate(colunas):
            ax = axes[0, j]
            if tipo == "rgb":
                ax.imshow(dado)
            elif tipo == "curv":
                ax.imshow(dado, cmap=CMAP_CURV,
                          norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
            else:
                ax.imshow(dado, cmap=CMAP_PROF, vmin=vmin, vmax=vmax)
            ax.set_title(titulo, fontsize=9)
            ax.set_axis_off()

            if args.zoom:
                y, x, h, w = args.zoom
                ax2 = axes[1, j]
                rec = dado[y:y + h, x:x + w] if dado.ndim == 2 else dado[y:y + h, x:x + w, :]
                if tipo == "rgb":
                    ax2.imshow(rec)
                elif tipo == "curv":
                    ax2.imshow(rec, cmap=CMAP_CURV,
                               norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
                else:
                    ax2.imshow(rec, cmap=CMAP_PROF, vmin=vmin, vmax=vmax)
                ax2.set_title("detalhe de borda", fontsize=8)
                ax2.set_axis_off()
                # marcar o recorte na imagem cheia
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                           edgecolor="white", linewidth=1.2))

        # metricas no rodape, para a figura ser autoexplicativa
        legenda = f"{chave}"
        for nome, p_al in preds.items():
            m = all_metrics(p_al, gt, mask)
            legenda += (f"   |   {nome}: AbsRel={m['abs_rel']:.4f} "
                        f"bF={m['boundary_fscore']:.4f}")
        fig.suptitle(legenda, fontsize=8)
        destino = out / "paineis" / f"{chave}_painel.png"
        fig.savefig(destino, dpi=150, bbox_inches="tight")
        plt.close(fig)
        linhas_resumo.append((chave, rgb_vis, gt_np, valido, preds, K_np, lim_k,
                              vmin, vmax))
        print(f"  [ok] {chave}")

    # ---- figura resumo (varias cenas empilhadas) --------------------------
    if linhas_resumo:
        n_lin = len(linhas_resumo)
        nomes = ["RGB", "GT"] + list(linhas_resumo[0][4].keys()) + ["Curvatura"]
        fig, axes = plt.subplots(n_lin, len(nomes),
                                 figsize=(2.9 * len(nomes), 2.9 * n_lin),
                                 layout="constrained")
        axes = np.atleast_2d(axes)
        for i, (chave, rgb_vis, gt_np, valido, preds, K_np, lim_k, vmin, vmax) \
                in enumerate(linhas_resumo):
            col = 0
            axes[i, col].imshow(rgb_vis); col += 1
            axes[i, col].imshow(np.where(valido, gt_np, np.nan),
                                cmap=CMAP_PROF, vmin=vmin, vmax=vmax); col += 1
            for nome, p_al in preds.items():
                axes[i, col].imshow(np.where(valido, p_al[0, 0].cpu().numpy(), np.nan),
                                    cmap=CMAP_PROF, vmin=vmin, vmax=vmax); col += 1
            axes[i, col].imshow(K_np, cmap=CMAP_CURV,
                                norm=TwoSlopeNorm(vmin=-lim_k, vcenter=0.0, vmax=lim_k))
            for j in range(len(nomes)):
                axes[i, j].set_axis_off()
                if i == 0:
                    axes[i, j].set_title(nomes[j], fontsize=10)
        fig.savefig(out / "figura_resumo.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "figura_resumo.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\n[saida] figura resumo: {out/'figura_resumo.png'} (e .pdf)")

    print(f"[saida] paineis:  {out/'paineis'}")
    print(f"[saida] mapas:    {out/'mapas'}  (PNG colorizado + .npy cru)")


if __name__ == "__main__":
    main()
