"""F0c -- a faixa metrica do F0b reconstroi a coluna `depth` do df?

PREMISSA QUE PRECISA SER VERIFICADA, NAO ASSUMIDA
-------------------------------------------------
Os canais geometricos fazem

    z(x) = z_min_m + depth01(x) * (z_max_m - z_min_m)

com `depth01` vindo da COLUNA do df e `z_min_m`/`z_max_m` vindo do job F0b, que e
uma execucao NOVA do Depth Pro. Se as duas normalizacoes nao forem a mesma, a
reconstrucao esta errada em toda amostra, e o erro e invisivel no mapa de defocus
(que reescala tudo por `max_coc` de qualquer forma).

E o mesmo tipo de defeito que o `defocus_map` normalizado por imagem ja custou a
este projeto, e o mesmo tipo de teste que resolveu o T1: comparar a grandeza
ABSOLUTA, nao a correlacao.

O QUE MEDE
----------
Para N amostras: roda o Depth Pro (identico ao F0b), pega o `z` metrico, e ajusta
o modelo afim  z ~ a * depth01 + b  contra a coluna armazenada. Se as duas vierem
do mesmo estimador com a mesma normalizacao min-max, entao

    a ~= z_max_f0b - z_min_f0b        e        b ~= z_min_f0b

Reporta R^2 e o erro relativo de `a` e `b` contra o que o F0b gravou.

TAMBEM MEDE A QUANTIZACAO UTIL
------------------------------
O `niveis_quant` do F0b conta niveis na faixa INTEIRA, ceu incluido. Em cena com
ceu no teto de 10.000 m isso da numero alto e enganoso: a cena util vive nos
primeiros milesimos. Aqui contamos os niveis dentro da REGIAO UTIL, definida como
`z <= corte_util * z_focus`, que e onde o circulo de confusao ainda distingue
profundidades.

Uso (container eval, 1 GPU):
    python3 f0c_consistencia_depth.py --rota c --n 40 --saida /saida/f0c_rota_c.json
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
    ap.add_argument("--rota", choices=["b", "c"], default="c")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--f0b", default=None, help="jsonl do F0b, para comparar a e b")
    ap.add_argument("--corte-util", type=float, default=20.0,
                    help="regiao util = z <= corte * z_focus")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    import torch, depth_pro
    from datasets import load_dataset

    tab = {}
    if a.f0b and os.path.exists(a.f0b):
        for ln in open(a.f0b):
            try:
                r = json.loads(ln); tab[r["stem"]] = r
            except Exception: pass
        print(f"[f0c] f0b: {len(tab)} amostras", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, transform = depth_pro.create_model_and_transforms()
    modelo.eval().to(dev)

    ds = load_dataset(REPOS[a.rota], split="train", streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    reg = []
    for r in ds:
        if len(reg) >= a.n: break
        stem = r.get("stem")
        try:
            aif = _pil(r["aif"]).convert("RGB")
            d = np.asarray(_pil(r["depth"]), dtype=np.float32)
            if d.ndim == 3: d = d[..., 0]
            d01 = d / 65535.0 if d.max() > 1.5 else d
            with torch.no_grad():
                z = modelo.infer(transform(aif).to(dev), f_px=None)["depth"]
            z = z.squeeze().float().cpu().numpy()
        except Exception as e:
            print(f"[f0c] {stem}: {e}", flush=True); continue

        if z.shape != d01.shape:
            d01 = np.asarray(Image.fromarray((d01 * 65535).astype(np.uint16))
                             .resize((z.shape[1], z.shape[0]), Image.BILINEAR),
                             dtype=np.float32) / 65535.0
        ok = np.isfinite(z) & (z > 0)
        x = d01[ok].ravel(); y = z[ok].ravel()
        if x.size < 1024: continue
        # subamostra para o ajuste
        idx = np.linspace(0, x.size - 1, min(200000, x.size)).astype(np.int64)
        x, y = x[idx], y[idx]
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        aa, bb = float(coef[0]), float(coef[1])
        res = y - (aa * x + bb)
        r2 = float(1.0 - res.var() / max(y.var(), 1e-12))

        e = {"stem": stem, "a_ajustado": aa, "b_ajustado": bb, "r2": r2,
             "z_min_medido": float(y.min()), "z_max_medido": float(y.max())}
        if stem in tab:
            t = tab[stem]
            faixa_f0b = t["z_max_m_bruto"] - t["z_min_m"]
            e["a_f0b"] = faixa_f0b; e["b_f0b"] = t["z_min_m"]
            e["erro_rel_a"] = abs(aa - faixa_f0b) / max(abs(faixa_f0b), 1e-9)
            e["erro_rel_b"] = abs(bb - t["z_min_m"]) / max(abs(t["z_min_m"]), 1e-9)
            zf = t.get("z_focus_m")
            if zf:
                util = ok & (z <= a.corte_util * zf)
                if util.sum() > 1024:
                    zu = z[util]
                    lo, hi = float(zu.min()), float(zu.max())
                    q = np.rint((zu - lo) / max(hi - lo, 1e-9) * 65535)
                    e["niveis_uteis"] = int(np.unique(q).size)
                    e["fracao_util"] = float(util.sum() / ok.sum())
                    # niveis que a codificacao ATUAL (faixa inteira) daria na regiao util
                    faixa_tot = max(t["z_max_m_bruto"] - t["z_min_m"], 1e-9)
                    qa = np.rint((zu - t["z_min_m"]) / faixa_tot * 65535)
                    e["niveis_uteis_na_codificacao_atual"] = int(np.unique(qa).size)
        reg.append(e)
        if len(reg) % 10 == 0: print(f"[f0c] {len(reg)}/{a.n}", flush=True)

    if not reg: print("nada medido", file=sys.stderr); return 1
    _rel(reg, a)
    json.dump({"n": len(reg), "registros": reg}, open(a.saida, "w"), indent=1)
    return 0


def _q(v, p):
    v = sorted(v); i = (len(v) - 1) * p / 100
    lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (i - lo)) + v[hi] * (i - lo)


def _rel(reg, a):
    print("\n" + "=" * 68)
    print(f"F0c -- consistencia da faixa metrica, rota {a.rota}, n={len(reg)}")
    print("=" * 68)
    r2 = [x["r2"] for x in reg]
    print(f"  R^2 do ajuste afim z ~ a*depth01 + b:  p10={_q(r2,10):.5f}  "
          f"mediana={_q(r2,50):.5f}  min={min(r2):.5f}")
    print(f"    (R^2 ~ 1 significa que a coluna `depth` E este mesmo campo,")
    print(f"     a menos de uma transformacao afim)")
    com = [x for x in reg if "erro_rel_a" in x]
    if com:
        ea = [x["erro_rel_a"] for x in com]; eb = [x["erro_rel_b"] for x in com]
        print(f"\n  erro relativo de a (faixa):  mediana={_q(ea,50):.4f}  p90={_q(ea,90):.4f}")
        print(f"  erro relativo de b (z_min):  mediana={_q(eb,50):.4f}  p90={_q(eb,90):.4f}")
        bons = sum(1 for x in com if x["erro_rel_a"] < 0.05 and x["erro_rel_b"] < 0.05)
        print(f"  amostras com AMBOS abaixo de 5%: {bons}/{len(com)} ({100*bons/len(com):.1f}%)")
        print(f"\n  VEREDITO: {'CONSISTENTE, a reconstrucao vale' if bons > 0.8*len(com) else 'INCONSISTENTE, a reconstrucao NAO vale como esta'}")
    ut = [x for x in reg if "niveis_uteis" in x]
    if ut:
        print(f"\n  --- quantizacao na REGIAO UTIL (z <= {a.corte_util:.0f} * z_focus) ---")
        nu = [x["niveis_uteis"] for x in ut]
        na = [x["niveis_uteis_na_codificacao_atual"] for x in ut]
        fr = [x["fracao_util"] for x in ut]
        print(f"  fracao de pixels na regiao util: p10={_q(fr,10):.3f} med={_q(fr,50):.3f}")
        print(f"  niveis se codificasse SO a util:      p10={_q(nu,10):.0f}  med={_q(nu,50):.0f}")
        print(f"  niveis que a codificacao ATUAL da la: p10={_q(na,10):.0f}  med={_q(na,50):.0f}")
        ruins = sum(1 for v in na if v < 256)
        print(f"  amostras com MENOS de 256 niveis uteis hoje: {ruins}/{len(ut)} ({100*ruins/len(ut):.1f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
