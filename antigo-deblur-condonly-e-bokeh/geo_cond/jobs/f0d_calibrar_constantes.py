"""F0d -- calibra as 8 constantes de normalizacao no conjunto de treino.

As constantes de `geo_cond/constants.py` NAO tem default, de proposito: um numero
plausivel silencioso e a classe de defeito que a auditoria encontrou. Elas sao
medidas UMA vez, sobre o dado de treino, e gravadas no config junto do run.

O QUE MEDE, E POR QUE CADA UMA
------------------------------
`tau_occlusion`  divisor do mapa de oclusao, O = min(||grad f||/tau, 1).
    Percentil ALTO da magnitude do gradiente, agregado sobre o conjunto INTEIRO.
    Nao por imagem: sob crop aleatorio o percentil do recorte nao bate com o da
    imagem inteira, e numa cena sem descontinuidade real o percentil vira ruido
    e o mapa satura de quadro cheio.

`u_max`  teto da profundidade inversa. Percentil alto de 1/z.

`s_max`  teto do elemento de area, s = log(sqrt(1+||grad f||^2)) >= 0.

`k0_curvature`  escala da compressao logaritmica da curvatura. Escolhido como a
    MEDIANA de |K| dos pixels com curvatura nao desprezivel, para que o log
    opere na regiao onde o sinal vive, e nao onde ele e ruido.

`kt_max`  teto simetrico do K~ ja comprimido.

`z_percentile_max`, `smooth_sigma`, `min_quant_levels` sao escolhas de metodo,
nao medicoes; ficam com os valores passados por flag e sao gravados junto para
que o run inteiro tenha as 8 num lugar so.

CRITERIO DE ESCOLHA DO PERCENTIL, DECLARADO ANTES
-------------------------------------------------
Percentil alto demais deixa o canal quase todo perto de zero e desperdica a
faixa; baixo demais satura. Reportamos a FRACAO SATURADA em cada candidato e
escolhemos o menor percentil cuja fracao saturada fique abaixo de `--alvo-sat`
(default 1%). O relatorio mostra a tabela inteira, entao a escolha e auditavel.

Uso (container eval, 1 GPU nao e necessaria; roda em CPU):
    python3 f0d_calibrar_constantes.py \\
        --escalares /saida/f0b_rota_c.jsonl --rota c --n 300 \\
        --saida /saida/constantes_rota_c.json
"""

from __future__ import annotations

import argparse, io, json, os, sys

import numpy as np
from PIL import Image

REPOS = {"b": "AKCITPixel3/BKXcuVXCmeRvN", "c": "AKCITPixel3/CMiQdveBBzNii"}


def _pil(x):
    if isinstance(x, Image.Image): return x
    if isinstance(x, dict) and "bytes" in x: return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)): return Image.open(io.BytesIO(x))
    raise TypeError(type(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--escalares", required=True, nargs="+",
                    help="um ou mais .jsonl do F0b (rota b e/ou c)")
    ap.add_argument("--rota", nargs="+", default=["c"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--field", default="inverse", choices=["inverse", "depth"])
    ap.add_argument("--alvo-sat", type=float, default=0.01)
    ap.add_argument("--smooth-sigma", type=float, default=2.0)
    ap.add_argument("--percentil-max", type=float, default=99.5)
    ap.add_argument("--min-quant-levels", type=int, default=256)
    ap.add_argument("--corte-util", type=float, default=20.0)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    sys.path.insert(0, "/proj")
    from datasets import load_dataset
    from geo_cond.signals import (area_element_and_normals, depth01_to_metric,
                                  gaussian_curvature_backprojected,
                                  grad_normalized, niveis_uteis)

    tab = {}
    for cam in a.escalares:
        for ln in open(cam):
            try:
                r = json.loads(ln); tab[r["stem"]] = r
            except Exception: pass
    print(f"[f0d] escalares: {len(tab)}", flush=True)

    amostras_grad, amostras_u, amostras_s, amostras_k = [], [], [], []
    n = 0
    for rota in a.rota:
        ds = load_dataset(REPOS[rota], split="train", streaming=True,
                          token=os.environ.get("HF_TOKEN"))
        passo = max(1, (2932 if rota == "c" else 11635) // max(1, a.n // len(a.rota)))
        for i, r in enumerate(ds):
            if i % passo or n >= a.n:
                if n >= a.n: break
                continue
            e = tab.get(r.get("stem"))
            if not e or not e.get("z_focus_m"):
                continue
            try:
                d = np.asarray(_pil(r["depth"]), dtype=np.float32)
                if d.ndim == 3: d = d[..., 0]
                d01 = d / 65535.0 if d.max() > 1.5 else d
            except Exception:
                continue
            # mesma geometria do dataloader: lado menor -> image_size
            H0, W0 = d01.shape
            esc = a.image_size / min(H0, W0)
            nh, nw = max(a.image_size, round(H0*esc)), max(a.image_size, round(W0*esc))
            d01 = np.asarray(Image.fromarray((d01*65535).astype(np.uint16))
                             .resize((nw, nh), Image.BILINEAR), np.float32)/65535.0

            z = depth01_to_metric(d01, e["z_min_m"], e["z_max_m_bruto"])
            u = 1.0 / np.clip(z, 1e-6, None)
            f = u if a.field == "inverse" else z
            fx_, fy_ = grad_normalized(f, float(min(d01.shape)))
            mag = np.hypot(fx_, fy_)
            s, _, _ = area_element_and_normals(fx_, fy_)

            sub = lambda arr: arr.ravel()[::37]     # subamostra, reduz memoria
            amostras_grad.append(sub(mag)); amostras_u.append(sub(u)); amostras_s.append(sub(s))

            if niveis_uteis(e["z_min_m"], e["z_max_m_bruto"], e["z_focus_m"],
                            corte=a.corte_util) >= a.min_quant_levels:
                fpx = e["focallength_px"] * (min(d01.shape) / min(H0, W0))
                K = gaussian_curvature_backprojected(
                    z, fpx, fpx, (d01.shape[1]-1)/2, (d01.shape[0]-1)/2, a.smooth_sigma)
                amostras_k.append(sub(np.abs(K)))
            n += 1
            if n % 25 == 0: print(f"[f0d] {n}/{a.n}", flush=True)
        if n >= a.n: break

    if n == 0:
        print("ERRO: nenhuma amostra", file=sys.stderr); return 1
    g = np.concatenate(amostras_grad); u = np.concatenate(amostras_u)
    s = np.concatenate(amostras_s)
    k = np.concatenate(amostras_k) if amostras_k else np.zeros(1)
    print(f"\n[f0d] {n} amostras, {g.size} pixels de gradiente", flush=True)

    def escolher(arr, nome):
        print(f"\n  --- {nome} ---")
        print(f"  {'percentil':>10s} {'valor':>12s} {'fracao saturada':>16s}")
        escolhido = None
        for p in (99.0, 99.5, 99.9, 99.95, 99.99):
            v = float(np.percentile(arr, p))
            frac = float((arr >= v).mean())
            marca = ""
            if escolhido is None and frac <= a.alvo_sat:
                escolhido = (p, v); marca = "  <- escolhido"
            print(f"  {p:10.2f} {v:12.6g} {frac:16.5f}{marca}")
        if escolhido is None:
            escolhido = (99.99, float(np.percentile(arr, 99.99)))
            print("  (nenhum atingiu o alvo; usando o mais alto)")
        return escolhido[1]

    tau = escolher(g, f"tau_occlusion  (||grad {a.field}||)")
    umax = escolher(u, "u_max  (1/z, 1/m)")
    smax = escolher(s, "s_max  (log sqrt(det g))")

    kpos = k[k > 0]
    k0 = float(np.median(kpos)) if kpos.size else 1.0
    kt = float(np.percentile(np.log1p(np.abs(k) / max(k0, 1e-12)), 99.5)) if kpos.size else 1.0
    print(f"\n  --- curvatura ---")
    print(f"  k0_curvature = mediana de |K| nao nulo = {k0:.6g} 1/m^2")
    print(f"  kt_max       = p99.5 de log(1+|K|/k0)  = {kt:.6g}")
    print(f"  (amostras com 2a ordem valida: {len(amostras_k)}/{n})")

    consts = {
        "tau_occlusion": tau, "u_max": umax, "s_max": smax,
        "k0_curvature": k0, "kt_max": kt,
        "z_percentile_max": a.percentil_max, "smooth_sigma": a.smooth_sigma,
        "min_quant_levels": a.min_quant_levels,
    }
    meta = {"n_amostras": n, "rotas": a.rota, "field": a.field,
            "image_size": a.image_size, "alvo_saturacao": a.alvo_sat,
            "escalares": a.escalares}
    json.dump({"constantes": consts, "procedencia": meta}, open(a.saida, "w"), indent=1)
    print("\n=== CONSTANTES (cole em geo_constantes no YAML) ===")
    print(json.dumps(consts, indent=2))
    print(f"\ngravado em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
