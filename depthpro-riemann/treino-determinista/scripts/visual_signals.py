#!/usr/bin/env python3
"""
scripts/visual_signals.py
=========================
Gera o conjunto completo de visualizações geométricas a partir de um modelo treinado.

Objetivo prático: decidir QUAL SINAL alimentar o DeblurNet / BokehNet. Para isso não
basta o mapa de profundidade; é preciso ver lado a lado os sinais de primeira ordem
(gradiente, tensor métrico, normais) e de segunda ordem (curvatura Gaussiana, curvatura
média, índice de forma), além das sobreposições sobre a imagem original, que é onde se
julga se o sinal está alinhado com as bordas reais do objeto.

Todos os mapas são gravados na MESMA DIMENSÃO da imagem de entrada.

Saídas, por imagem:
    mapas/<chave>/depth.png, K.png, H.png, det_g.png, grad.png, normals.png,
                  shape_index.png, curvedness.png, occlusion.png, k1.png, k2.png
                  (+ .npy com os valores crus de cada um)
    overlays/<chave>/rgb+depth.png, rgb+K.png, rgb+occlusion.png, rgb+normals.png,
                     depth+K.png, depth+occlusion.png
    paineis/<chave>_sinais.png     painel com todos os sinais
    contato/<chave>_contato.png    tira comparativa compacta

Uso:
    python scripts/visual_signals.py \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --weights /workspace/runs/champion/seed_0/best.pt \\
        --data-root /data/hypersim/test \\
        --out-dir /workspace/figuras_sinais \\
        --variant heads --n-imagens 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.dataset import HighQualityDepthDataset   # noqa: E402
from riemann.geometry_maps import VIS_INFO, compute_all  # noqa: E402
from riemann.losses import robust_affine_align        # noqa: E402
from riemann.model import build_model                 # noqa: E402

# ordem de exibição nos painéis
ORDEM = ["depth", "grad", "det_g", "normals", "K", "H",
         "shape_index", "curvedness", "occlusion"]


def desnormalizar_rgb(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    a = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(a * np.array(std) + np.array(mean), 0, 1)


def limite_robusto(a, pct=98.0):
    v = np.abs(a[np.isfinite(a)])
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, pct))
    return lim if lim > 1e-8 else 1.0


def para_rgb(arr, nome):
    """Converte um mapa de sinal em imagem RGB [0,1] usando o colormap adequado."""
    info = VIS_INFO[nome]
    if nome == "normals":
        # normais em (B,3,H,W) -> codificação padrão (n+1)/2
        return np.clip((arr.transpose(1, 2, 0) + 1.0) / 2.0, 0, 1)
    a = arr[0] if arr.ndim == 3 else arr
    if info["divergente"]:
        lim = limite_robusto(a)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    else:
        finito = a[np.isfinite(a)]
        vmin = float(np.percentile(finito, 2)) if finito.size else 0.0
        vmax = float(np.percentile(finito, 98)) if finito.size else 1.0
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        norm = Normalize(vmin=vmin, vmax=vmax)
    return matplotlib.colormaps[info["cmap"]](norm(a))[..., :3]


def sobrepor(base_rgb, sinal_rgb, alfa=0.55, mascara=None):
    """
    Mistura sinal sobre a imagem base. Se `mascara` for dada (valores em [0,1]), a
    opacidade é modulada por ela, o que evita cobrir a foto inteira quando só a região
    de borda interessa.
    """
    if mascara is None:
        a = alfa
    else:
        a = (alfa * np.clip(mascara, 0, 1))[..., None]
    return np.clip(base_rgb * (1 - a) + sinal_rgb * a, 0, 1)


def salvar(img, caminho):
    """Salva uma imagem RGB [0,1] preservando exatamente as dimensões do array."""
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)


def redimensionar(a, hw):
    """Redimensiona um array (H,W) ou (H,W,C) para hw=(H2,W2) com torch."""
    if a.shape[:2] == tuple(hw):
        return a
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    if t.ndim == 2:
        t = t[None, None]
        modo = "bilinear"
    else:
        t = t.permute(2, 0, 1)[None]
        modo = "bilinear"
    r = torch.nn.functional.interpolate(t, size=tuple(hw), mode=modo,
                                        align_corners=False)
    r = r[0]
    return r[0].numpy() if r.shape[0] == 1 else r.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", default=None,
                    help="best.pt treinado; sem isto usa o modelo base")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="heads",
                    choices=["heads", "heads_final", "heads_lora"])
    ap.add_argument("--size", type=int, default=512,
                    help="resolucao de inferencia")
    ap.add_argument("--saida-nativa", action=argparse.BooleanOptionalAction, default=True,
                    help="grava os mapas na dimensao original da imagem de entrada")
    ap.add_argument("--n-imagens", type=int, default=6)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
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
    ap.add_argument("--alfa", type=float, default=0.55,
                    help="opacidade das sobreposicoes")
    args = ap.parse_args()

    out = Path(args.out_dir)
    for sub in ("mapas", "overlays", "paineis", "contato"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ds = HighQualityDepthDataset(args.data_root, size=(args.size, args.size))
    idxs = (args.indices if args.indices
            else list(np.linspace(0, len(ds) - 1,
                                  min(args.n_imagens, len(ds))).astype(int)))
    print(f"[dados] {len(ds)} imagens; gerando {len(idxs)}")

    modelo = build_model(args.checkpoint, variant=args.variant, device=args.device)
    if args.weights:
        modelo.load_state_dict(torch.load(args.weights, map_location=args.device),
                               strict=False)
        print(f"[modelo] pesos treinados: {args.weights}")
    else:
        print("[modelo] usando o modelo base (zero-shot)")
    modelo.eval()

    for idx in idxs:
        am = ds[int(idx)]
        chave = am["key"]
        rgb_t = am["rgb"].unsqueeze(0)
        gt = am["depth"].unsqueeze(0).float().to(args.device)
        mask = am["mask"].unsqueeze(0).to(args.device)

        with torch.no_grad():
            pred = modelo(rgb_t.to(args.device)).float()
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            # alinhar ao GT para os valores terem escala métrica interpretável
            pred = robust_affine_align(pred, gt, mask, mode="full")
            sinais = compute_all(pred, args.curv_sigma, (args.curv_clamp or 1e12))

        # dimensão de saída: a da imagem ORIGINAL em disco, se pedido
        rgb_vis = desnormalizar_rgb(am["rgb"])
        if args.saida_nativa:
            import cv2
            orig = cv2.imread(str(ds.rgb_files[chave]))
            hw = (orig.shape[0], orig.shape[1])
        else:
            hw = rgb_vis.shape[:2]
        rgb_vis = redimensionar(rgb_vis, hw)

        dm = out / "mapas" / chave
        ov = out / "overlays" / chave
        dm.mkdir(parents=True, exist_ok=True)
        ov.mkdir(parents=True, exist_ok=True)

        # ---- mapas individuais ----------------------------------------
        rgbs = {}
        for nome in ORDEM + ["k1", "k2"]:
            arr = sinais[nome][0].cpu().numpy()
            img = para_rgb(arr, nome)
            img = redimensionar(img, hw)
            rgbs[nome] = img
            salvar(img, dm / f"{nome}.png")
            cru = arr if nome == "normals" else arr[0]
            np.save(dm / f"{nome}.npy",
                    (redimensionar(cru.transpose(1, 2, 0) if cru.ndim == 3 else cru, hw)
                     ).astype(np.float32))
        salvar(rgb_vis, dm / "rgb.png")

        # ---- sobreposições -------------------------------------------
        occ = redimensionar(sinais["occlusion"][0, 0].cpu().numpy(), hw)
        combinacoes = [
            ("rgb+depth", rgb_vis, rgbs["depth"], None),
            ("rgb+K", rgb_vis, rgbs["K"], None),
            ("rgb+K_bordas", rgb_vis, rgbs["K"], occ),
            ("rgb+occlusion", rgb_vis, rgbs["occlusion"], occ),
            ("rgb+normals", rgb_vis, rgbs["normals"], None),
            ("rgb+det_g", rgb_vis, rgbs["det_g"], None),
            ("depth+K", rgbs["depth"], rgbs["K"], None),
            ("depth+occlusion", rgbs["depth"], rgbs["occlusion"], occ),
            ("depth+normals", rgbs["depth"], rgbs["normals"], None),
        ]
        for nome, base, sinal, masc in combinacoes:
            salvar(sobrepor(base, sinal, args.alfa, masc), ov / f"{nome}.png")

        # ---- painel com todos os sinais -------------------------------
        itens = [("RGB", rgb_vis)] + [(VIS_INFO[n]["titulo"], rgbs[n]) for n in ORDEM]
        ncol = 5
        nlin = int(np.ceil(len(itens) / ncol))
        fig, axes = plt.subplots(nlin, ncol, figsize=(3.1 * ncol, 3.1 * nlin),
                                 layout="constrained")
        axes = np.atleast_2d(axes)
        for i, (titulo, img) in enumerate(itens):
            ax = axes[i // ncol, i % ncol]
            ax.imshow(img)
            ax.set_title(titulo, fontsize=9)
            ax.set_axis_off()
        for j in range(len(itens), nlin * ncol):
            axes[j // ncol, j % ncol].set_axis_off()
        fig.suptitle(f"{chave} — sinais geométricos", fontsize=11)
        fig.savefig(out / "paineis" / f"{chave}_sinais.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

        # ---- tira de contato (compacta, para escolher sinal rapidamente)
        chave_itens = [("RGB", rgb_vis), ("Profundidade", rgbs["depth"]),
                       ("Curvatura K", rgbs["K"]), ("Oclusão", rgbs["occlusion"]),
                       ("RGB+K nas bordas",
                        sobrepor(rgb_vis, rgbs["K"], args.alfa, occ))]
        fig, axes = plt.subplots(1, len(chave_itens),
                                 figsize=(3.0 * len(chave_itens), 3.2),
                                 layout="constrained")
        for ax, (t, img) in zip(np.atleast_1d(axes), chave_itens):
            ax.imshow(img)
            ax.set_title(t, fontsize=9)
            ax.set_axis_off()
        fig.savefig(out / "contato" / f"{chave}_contato.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

        print(f"  [ok] {chave}  ({hw[1]}x{hw[0]})")

    print(f"\n[saida] mapas     : {out/'mapas'}     (PNG + .npy por sinal)")
    print(f"[saida] overlays  : {out/'overlays'}")
    print(f"[saida] paineis   : {out/'paineis'}")
    print(f"[saida] contato   : {out/'contato'}")
    print("\nPara escolher o sinal do DeblurNet/BokehNet, compare em 'contato':")
    print("  profundidade -> raio do circulo de confusao")
    print("  oclusao      -> onde o kernel de desfoque NAO pode atravessar")
    print("  curvatura K  -> tipo de geometria (domo/sela), util p/ forma do bokeh")
    print("  normais      -> inclinacao da superficie, distorce o bokeh fora do eixo")


if __name__ == "__main__":
    main()
