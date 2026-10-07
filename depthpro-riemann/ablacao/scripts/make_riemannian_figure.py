#!/usr/bin/env python3
"""
scripts/make_riemannian_figure.py
=================================
Painel de apresentação no estilo "métrica riemanniana": tema escuro, 2 linhas x 3 colunas,
com o elemento de área √det(g) em destaque e as sobreposições sobre a imagem original.

Layout:
    linha 1:  RGB original   |  mapa de profundidade  |  √det(g) elemento de área
    linha 2:  RGB ⊕ profund. |  RGB ⊕ métrica         |  vetores normais

Por que o elemento de área e não a curvatura Gaussiana nesta figura: √det(g) = √(1+|∇D|²)
é de PRIMEIRA ordem, então o ruído entra linearmente. A curvatura usa segundas derivadas e
amplifica ruído por 1/h², o que exige suavização pesada e ainda assim produz mapas
granulados. Para comunicar "onde estão as bordas 3D", o elemento de área é o mapa certo.
A curvatura Gaussiana continua disponível em `--modo curvatura`, que troca o painel de
√det(g) por K (intrínseca, distingue região elíptica de hiperbólica).

Uso:
    python scripts/make_riemannian_figure.py \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --weights /workspace/runs/champion_heads_final/seed_0/best.pt \\
        --data-root /data/hypersim/test \\
        --out-dir /workspace/figuras_riemann \\
        --variant heads_final --n-imagens 6 \\
        --titulo "Hypersim — configuração campeã"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.dataset import HighQualityDepthDataset          # noqa: E402
from riemann.geometry_maps import (area_element,             # noqa: E402
                                   principal_curvatures)
from riemann.losses import _surface_normals, robust_affine_align  # noqa: E402
from riemann.metrics import all_metrics                       # noqa: E402
from riemann.model import build_model                        # noqa: E402

FUNDO = "#12141c"
TEXTO = "#E8E8EA"
DESTAQUE = "#F2B705"


def desnormalizar_rgb(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    a = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(a * np.array(std) + np.array(mean), 0, 1)


def normalizar_robusto(a, plo=1.0, phi=99.0):
    """Leva o mapa a [0,1] por percentis, imune a outliers isolados."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, plo), np.percentile(v, phi)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return np.clip((a - lo) / (hi - lo), 0, 1)


def normalizar_area(a, plo=60.0, phi=99.5, gama=1.8):
    """
    Normalização específica do elemento de área.

    Três cuidados que fazem a diferença na figura:
      1. Subtraímos a linha de base 1. Em superfície frontoparalela √det(g) vale exatamente
         1, então (√det(g) − 1) coloca o "plano" em zero, que no colormap `hot` é preto.
      2. O percentil inferior é alto (60 por padrão), porque a maior parte da cena é
         superfície suave: isso joga o piso de ruído para o preto em vez de deixá-lo
         avermelhado.
      3. A gama > 1 comprime os valores baixos e realça os altos, de modo que só as
         quebras de profundidade acendem.
    """
    b = np.maximum(a - 1.0, 0.0)
    v = b[np.isfinite(b)]
    if v.size == 0:
        return np.zeros_like(b)
    lo, hi = np.percentile(v, plo), np.percentile(v, phi)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return np.clip((b - lo) / (hi - lo), 0, 1) ** gama


def colorir(a, cmap, divergente=False):
    if divergente:
        lim = float(np.percentile(np.abs(a[np.isfinite(a)]), 98)) or 1.0
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        return matplotlib.colormaps[cmap](norm(a))[..., :3]
    return matplotlib.colormaps[cmap](Normalize(0, 1)(normalizar_robusto(a)))[..., :3]


def mistura_screen(base, brilho, intensidade=1.0):
    """
    Mistura tipo *screen*: 1 - (1-base)(1-brilho). Ao contrário da mistura linear, ela só
    CLAREIA, então as regiões escuras do mapa deixam a foto intacta e as regiões brilhantes
    (as bordas 3D) acendem sobre ela. É o que dá o efeito de "brilho" da figura.
    """
    b = np.clip(brilho * intensidade, 0, 1)
    return np.clip(1.0 - (1.0 - base) * (1.0 - b), 0, 1)


def mistura_alfa(base, camada, alfa=0.5):
    return np.clip(base * (1 - alfa) + camada * alfa, 0, 1)


def eixo(ax, img, titulo, destaque=False, sub=None):
    ax.imshow(img)
    ax.set_axis_off()
    cor = DESTAQUE if destaque else TEXTO
    txt = titulo if sub is None else f"{titulo}\n{sub}"
    ax.set_title(txt, fontsize=10.5, color=cor,
                 fontweight="bold" if destaque else "normal", pad=8)
    for lado in ("top", "bottom", "left", "right"):
        ax.spines[lado].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="heads",
                    choices=["heads", "heads_final", "heads_lora"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n-imagens", type=int, default=6)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modo", default="area", choices=["area", "curvatura"],
                    help="'area' usa √det(g) (1a ordem, limpo); 'curvatura' usa K "
                         "(2a ordem, intrinseca, mas exige suavizacao)")
    ap.add_argument("--curv-sigma", type=float, default=2.0,
                    help="suavizacao para o modo curvatura (2a derivada amplifica ruido)")
    ap.add_argument("--piso", type=float, default=60.0,
                    help="percentil que vira PRETO no mapa de area. Suba (ex.: 80) se o "
                         "fundo ficar avermelhado; baixe se as bordas sumirem.")
    ap.add_argument("--gama", type=float, default=1.8,
                    help="gama do mapa de area; >1 escurece o piso de ruido e realca "
                         "as quebras de profundidade")
    ap.add_argument("--intensidade", type=float, default=1.0,
                    help="ganho do brilho na sobreposicao RGB + metrica")
    ap.add_argument("--comparar", action="store_true",
                    help="gera painel COMPARATIVO: linha de cima = zero-shot, linha de "
                         "baixo = modelo treinado, mesmas colunas e MESMA escala de cor. "
                         "E a figura que responde 'o treinado melhorou a geometria?'")
    ap.add_argument("--titulo", default=None,
                    help="titulo do painel; o nome da imagem entra automaticamente")
    ap.add_argument("--perdas", default="L_berhu + L_grad + L_normal + L_gauss",
                    help="texto do rodape descrevendo a perda usada no treino")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = HighQualityDepthDataset(args.data_root, size=(args.size, args.size))
    idxs = (args.indices if args.indices
            else list(np.linspace(0, len(ds) - 1,
                                  min(args.n_imagens, len(ds))).astype(int)))
    print(f"[dados] {len(ds)} imagens; gerando {len(idxs)} paineis")

    # No modo comparativo carregamos os DOIS modelos; senao, apenas um.
    modelos = {}
    if args.comparar:
        if not args.weights:
            print("ERRO: --comparar exige --weights (precisa do modelo treinado).")
            sys.exit(1)
        zs = build_model(args.checkpoint, variant=args.variant, device=args.device)
        zs.eval()
        modelos["zero-shot"] = zs
        tr = build_model(args.checkpoint, variant=args.variant, device=args.device)
        tr.load_state_dict(torch.load(args.weights, map_location=args.device),
                           strict=False)
        tr.eval()
        modelos["treinado"] = tr
        print(f"[modelo] comparativo: zero-shot vs {args.weights}")
    else:
        modelo = build_model(args.checkpoint, variant=args.variant, device=args.device)
        if args.weights:
            modelo.load_state_dict(torch.load(args.weights, map_location=args.device),
                                   strict=False)
            print(f"[modelo] pesos treinados: {args.weights}")
        else:
            print("[modelo] zero-shot (sem fine-tune)")
        modelo.eval()
        modelos["modelo"] = modelo

    for idx in idxs:
        am = ds[int(idx)]
        chave = am["key"]
        gt = am["depth"].unsqueeze(0).float().to(args.device)
        mask = am["mask"].unsqueeze(0).to(args.device)
        rgb = desnormalizar_rgb(am["rgb"])
        valido = mask[0, 0].cpu().numpy() > 0.5

        # ---- calcular os campos para cada modelo carregado --------------
        dados = {}
        for nome, mdl in modelos.items():
            with torch.no_grad():
                pred = mdl(am["rgb"].unsqueeze(0).to(args.device)).float()
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = torch.nn.functional.interpolate(
                        pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                pred = robust_affine_align(pred, gt, mask, mode="full")
                if args.modo == "area":
                    campo = area_element(pred)[0, 0].cpu().numpy()
                else:
                    _, _, K, _ = principal_curvatures(pred, args.curv_sigma, None, 1e12)
                    campo = K[0, 0].cpu().numpy()
                normais = _surface_normals(pred)[0].cpu().numpy()
            dados[nome] = {"prof": pred[0, 0].cpu().numpy(), "campo": campo,
                           "normais": normais}
            met = all_metrics(pred, gt, mask)
            dados[nome]["met"] = (f"AbsRel={met['abs_rel']:.4f}  "
                                  f"bF-max={met.get('boundary_fmax', float('nan')):.4f}")

        if args.modo == "area":
            titulo_campo = "√det(g): Área Riemanniana"
            sub_campo = "g_ij = δ_ij + ∂_iD·∂_jD"
            cmap_campo, divergente = "hot", False
        else:
            titulo_campo = "Curvatura Gaussiana K"
            sub_campo = "K = k₁·k₂  (intrínseca)"
            cmap_campo, divergente = "RdBu_r", True

        # ESCALAS DE COR COMPARTILHADAS entre os modelos. Sem isto, a comparação seria
        # inválida: cada painel se auto-normalizaria e a diferença visual viria da
        # normalização, não da geometria.
        todas_prof = np.concatenate([d["prof"][valido].ravel() for d in dados.values()])
        pmin, pmax = np.percentile(todas_prof, [2, 98])
        if args.modo == "area":
            base = np.concatenate([np.maximum(d["campo"] - 1.0, 0.0).ravel()
                                   for d in dados.values()])
            clo, chi = np.percentile(base, [args.piso, 99.5])
            chi = chi if chi - clo > 1e-9 else clo + 1e-9
        else:
            clim = float(np.percentile(
                np.abs(np.concatenate([d["campo"].ravel() for d in dados.values()])), 98)) or 1.0

        def img_campo_de(campo):
            if args.modo == "area":
                v = np.clip((np.maximum(campo - 1.0, 0.0) - clo) / (chi - clo), 0, 1) ** args.gama
                return matplotlib.colormaps[cmap_campo](v)[..., :3]
            return matplotlib.colormaps[cmap_campo](
                TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)(campo))[..., :3]

        def img_prof_de(prof):
            return matplotlib.colormaps["plasma"](Normalize(pmin, pmax)(prof))[..., :3]

        faixa = f"[{pmin:.1f}–{pmax:.1f} m]"

        # ================= modo COMPARATIVO: uma linha por modelo =================
        if args.comparar:
            nomes = ["zero-shot", "treinado"]
            fig, axes = plt.subplots(2, 4, figsize=(21.5, 11.5), layout="constrained")
            fig.patch.set_facecolor(FUNDO)
            for ax in axes.ravel():
                ax.set_facecolor(FUNDO)
            for i, nome in enumerate(nomes):
                d = dados[nome]
                ip = img_prof_de(d["prof"])
                ic = img_campo_de(d["campo"])
                inorm = np.clip((d["normais"].transpose(1, 2, 0) + 1.0) / 2.0, 0, 1)
                ov = (mistura_screen(rgb, ic, args.intensidade) if args.modo == "area"
                      else np.clip(rgb * (1 - 0.75 * normalizar_robusto(np.abs(d["campo"]))[..., None])
                                   + ic * (0.75 * normalizar_robusto(np.abs(d["campo"]))[..., None]), 0, 1))
                dest = (nome == "treinado")
                eixo(axes[i, 0], ip, f"Profundidade — {nome} {faixa}", destaque=dest)
                eixo(axes[i, 1], ic, f"{titulo_campo} — {nome}", destaque=dest,
                     sub=sub_campo if i == 0 else None)
                eixo(axes[i, 2], ov, f"RGB ⊕ métrica — {nome}", destaque=dest)
                eixo(axes[i, 3], inorm, f"Normais — {nome}", destaque=dest,
                     sub=d["met"])
            fig.suptitle(f"{args.titulo or 'Comparação zero-shot vs treinado'} — {chave}",
                         fontsize=15, color=TEXTO, fontweight="bold")
            fig.supxlabel(
                "escalas de cor COMPARTILHADAS entre as linhas   ·   "
                "√det(g) = √(1 + |∇D|²), regiões brilhantes = bordas 3D   ·   "
                f"modelo treinado com {args.perdas}",
                fontsize=10, color="#9AA0AA", style="italic")
            destino = out / f"{chave}_comparativo_{args.modo}.png"

        # ================= modo simples: painel 2x3 de um modelo =================
        else:
            nome = list(dados)[0]
            d = dados[nome]
            img_prof = img_prof_de(d["prof"])
            img_campo = img_campo_de(d["campo"])
            img_norm = np.clip((d["normais"].transpose(1, 2, 0) + 1.0) / 2.0, 0, 1)
            ov_prof = mistura_alfa(rgb, img_prof, 0.45)
            if args.modo == "area":
                ov_campo = mistura_screen(rgb, img_campo, args.intensidade)
            else:
                peso = normalizar_robusto(np.abs(d["campo"]))[..., None]
                ov_campo = np.clip(rgb * (1 - 0.75 * peso) + img_campo * (0.75 * peso), 0, 1)

            fig, axes = plt.subplots(2, 3, figsize=(16.5, 12.0), layout="constrained")
            fig.patch.set_facecolor(FUNDO)
            for ax in axes.ravel():
                ax.set_facecolor(FUNDO)
            eixo(axes[0, 0], rgb, "RGB Original")
            eixo(axes[0, 1], img_prof, f"Mapa de Profundidade {faixa}")
            eixo(axes[0, 2], img_campo, titulo_campo, destaque=True, sub=sub_campo)
            eixo(axes[1, 0], ov_prof, "RGB ⊕ Mapa de Profundidade")
            eixo(axes[1, 1], ov_campo,
                 "RGB ⊕ Métrica Riemanniana" if args.modo == "area" else "RGB ⊕ Curvatura",
                 destaque=True, sub="← CONTRIBUIÇÃO ★ →")
            eixo(axes[1, 2], img_norm, "Vetores Normais (RGB)", sub=d["met"])
            rotulo = "" if nome == "modelo" else f" ({nome})"
            fig.suptitle(f"{args.titulo or 'Métrica Riemanniana'}{rotulo} — {chave}",
                         fontsize=15, color=TEXTO, fontweight="bold")
            if args.modo == "area":
                rodape = ("√det(g) = √(1 + |∇D|²)   ·   regiões brilhantes = bordas 3D   ·   "
                          f"modelo treinado com {args.perdas}")
            else:
                rodape = ("K = curvatura Gaussiana (intrínseca)   ·   vermelho = elíptico, "
                          "azul = hiperbólico, branco = plano   ·   "
                          f"modelo treinado com {args.perdas}")
            fig.supxlabel(rodape, fontsize=10, color="#9AA0AA", style="italic")
            destino = out / f"{chave}_riemann_{args.modo}.png"

        fig.savefig(destino, dpi=150, facecolor=FUNDO, bbox_inches="tight")
        fig.savefig(destino.with_suffix(".pdf"), facecolor=FUNDO, bbox_inches="tight")
        plt.close(fig)

        dm = out / "mapas" / chave
        dm.mkdir(parents=True, exist_ok=True)
        for nome, d in dados.items():
            tag = "" if nome == "modelo" else f"_{nome}"
            np.save(dm / f"{args.modo}{tag}.npy", d["campo"].astype(np.float32))
            np.save(dm / f"profundidade{tag}.npy", d["prof"].astype(np.float32))
            np.save(dm / f"normais{tag}.npy", d["normais"].astype(np.float32))

        print(f"  [ok] {chave}")

    print(f"\n[saida] paineis em {out} (PNG e PDF)")
    print(f"[saida] arrays crus em {out/'mapas'}")


if __name__ == "__main__":
    main()
