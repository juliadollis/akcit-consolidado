"""F2 -- diagnostico do peso da perda em espaco de TOKEN.

Pergunta: depois do max-pool 16x, que FRACAO DOS TOKENS recebe peso alto?

Importa porque o peso e normalizado pela media (para desacoplar lambda do passo
do otimizador). Se a borda, que e ~4% dos PIXELS, virar 40% dos TOKENS depois do
max-pool, entao a normalizacao rebaixa os 60% restantes de forma significativa, e
o modelo piora fora da borda. E a hipotese mecanica para o que o E_bleed mediu no
A': erro maior em TODA parte, com a razao borda/fora menor.

Uso: python3 f2_diag_peso.py --escalares /saida/f0b_todas.jsonl --n 40
"""
from __future__ import annotations
import argparse, io, json, os, sys
import numpy as np
from PIL import Image

def _pil(x):
    if isinstance(x, Image.Image): return x
    if isinstance(x, dict) and "bytes" in x: return Image.open(io.BytesIO(x["bytes"]))
    return Image.open(io.BytesIO(x))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--escalares", required=True)
    ap.add_argument("--constantes", required=True)
    ap.add_argument("--rota", default="c")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lado", type=int, default=512)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    sys.path.insert(0, "/proj")
    import torch
    from datasets import load_dataset
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    from geo_cond.constants import GeoConstants
    from geo_cond.loss_weight import occlusion_to_token_weight
    from geo_cond.signals import depth01_to_metric, grad_normalized, occlusion

    consts = GeoConstants(**json.load(open(a.constantes))["constantes"])
    tab = {}
    for ln in open(a.escalares):
        try:
            r = json.loads(ln); tab[r["stem"]] = r
        except Exception: pass
    repo = {"b": "AKCITPixel3/BKXcuVXCmeRvN", "c": "AKCITPixel3/CMiQdveBBzNii"}[a.rota]
    ds = load_dataset(repo, split="train", streaming=True, token=os.environ.get("HF_TOKEN"))

    fr_px, fr_tok, stats = [], [], {L: [] for L in a.lambdas}
    o_tokens = []
    n = 0
    for r in ds:
        if n >= a.n: break
        e = tab.get(r.get("stem"))
        if not e or not e.get("z_focus_m"): continue
        d = np.asarray(_pil(r["depth"]), np.float32)
        if d.ndim == 3: d = d[..., 0]
        d01 = d / 65535.0 if d.max() > 1.5 else d
        H0, W0 = d01.shape
        esc = a.lado / min(H0, W0)
        nh, nw = max(a.lado, round(H0*esc)), max(a.lado, round(W0*esc))
        d01 = np.asarray(Image.fromarray((d01*65535).astype(np.uint16))
                         .resize((nw, nh), Image.BILINEAR), np.float32)/65535.0
        y0 = (nh - a.lado)//2; x0 = (nw - a.lado)//2
        d01 = d01[y0:y0+a.lado, x0:x0+a.lado]
        z = depth01_to_metric(d01, e["z_min_m"], e["z_max_m_bruto"])
        u = 1.0/np.clip(z, 1e-6, None)
        fx, fy = grad_normalized(u, float(a.lado))
        O = occlusion(np.hypot(fx, fy), consts.tau_occlusion)
        fr_px.append(float((O > 0.3).mean()))
        Ot = torch.from_numpy(O)[None, None].float()
        for L in a.lambdas:
            w = occlusion_to_token_weight(Ot, FluxPipeline._pack_latents,
                                          lambda_o=L, normalize=True)
            wf = w.flatten()
            stats[L].append({
                "frac_token_acima_1": float((wf > 1.0).float().mean()),
                "w_min": float(wf.min()), "w_max": float(wf.max()),
                "p10": float(wf.kthvalue(max(1, int(0.10*wf.numel()))).values),
            })
        wn = occlusion_to_token_weight(Ot, FluxPipeline._pack_latents,
                                       lambda_o=1.0, normalize=False)
        fr_tok.append(float((wn.flatten() > 1.0).float().mean()))
        # O_token cru, para escolher o limiar: w = 1 + 1*O_token quando
        # normalize=False, entao O_token = w - 1.
        o_tokens.append((wn.flatten() - 1.0).numpy())
        n += 1
        if n % 10 == 0: print(f"[f2] {n}/{a.n}", flush=True)

    q = lambda v, p: float(np.percentile(v, p))
    print("\n" + "="*62)
    print(f"F2 -- peso em espaco de token, rota {a.rota}, n={n}")
    print("="*62)
    print(f"  fracao de PIXELS com O > 0.3 : mediana={q(fr_px,50):.4f}")
    print(f"  fracao de TOKENS com O > 0    : mediana={q(fr_tok,50):.4f}")
    print(f"  fator de espalhamento do max-pool 16x: {q(fr_tok,50)/max(q(fr_px,50),1e-9):.1f}x")
    print(f"\n  {'lambda':>7s} {'tokens w>1':>11s} {'w_min':>8s} {'w_p10':>8s} {'w_max':>8s}")
    for L in a.lambdas:
        s = stats[L]
        print(f"  {L:7.1f} {np.median([x['frac_token_acima_1'] for x in s]):11.4f} "
              f"{np.median([x['w_min'] for x in s]):8.4f} "
              f"{np.median([x['p10'] for x in s]):8.4f} "
              f"{np.median([x['w_max'] for x in s]):8.4f}")
    print("\n  w_min e o peso dos tokens SEM borda. Quanto menor, mais a")
    print("  normalizacao rebaixa a supervisao fora da regiao de interesse.")

    # --- escolha do limiar theta ---------------------------------------
    ot = np.concatenate(o_tokens)
    print(f"\n  --- distribuicao de O_token (n={ot.size}) ---")
    for p_ in (1, 10, 25, 50, 75, 90, 99):
        print(f"    p{p_:<3d} = {q(ot, p_):.4f}")
    print(f"\n  {'theta':>7s} {'frac tokens':>12s} {'piso (lam=2)':>13s} {'contraste':>10s}")
    escolhido = None
    for th in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        fr = float((ot > th).mean()) if th > 0 else float((ot > 0).mean())
        m = 1.0 + 2.0 * (fr if th > 0 else float(ot.mean()))
        piso = 1.0 / m
        contraste = (1.0 + 2.0 * (1.0 if th > 0 else float(ot.max()))) / 1.0
        marca = ""
        if escolhido is None and th > 0 and 0.08 <= fr <= 0.22:
            escolhido = th; marca = "  <- alvo 10-20%"
        print(f"  {th:7.2f} {fr:12.4f} {piso:13.4f} {contraste:10.2f}{marca}")
    print(f"\n  THETA RECOMENDADO: {escolhido if escolhido else 'nenhum na faixa alvo'}")
    json.dump({"n": n, "frac_px": fr_px, "frac_tok": fr_tok,
               "stats": {str(k): v for k, v in stats.items()}},
              open(a.saida, "w"), indent=1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
