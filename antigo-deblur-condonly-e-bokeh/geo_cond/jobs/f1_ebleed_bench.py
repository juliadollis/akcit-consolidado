"""F1 -- E_bleed sobre um benchmark, para um LoRA do BokehNet.

E o portao da campanha (PLANO secao 6): mede o erro CONCENTRADO na regiao de
borda de profundidade, que e onde vive o vazamento de cor. As metricas globais
diluem o artefato, porque a borda e uma fracao pequena dos pixels.

O mapa de oclusao e o `tau` sao os MESMOS do condicionamento, senao a metrica
mediria uma regiao diferente da que a perda supervisiona.

PROFUNDIDADE NO BENCHMARK
-------------------------
Nenhum dos tres benchmarks tem coluna `depth` (AUDITORIA secao 6). Isso NAO
bloqueia: o proprio pipeline de avaliacao ja roda o Depth Pro em toda imagem
(`bokeh_net.py:93-141`), so nao grava. Aqui rodamos igual, e como e a MESMA
profundidade em todas as condicoes comparadas, o estimador cancela na comparacao
pareada.

O `z_focus` vem da Eq. 4 quando ha mascara, e do proprio Depth Pro sobre a AIF.

Uso (container eval, 1 GPU):
    python3 f1_ebleed_bench.py --bench juliadollis/lf-bokeh-repro-blb \\
        --lora <path.safetensors|none> --constantes /saida/constantes_rota_c.json \\
        --n 50 --saida /saida/ebleed_<modelo>.json
"""

from __future__ import annotations

import argparse, io, json, os, sys

import numpy as np
from PIL import Image


def _pil(x):
    if isinstance(x, Image.Image): return x
    if isinstance(x, dict) and "bytes" in x: return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)): return Image.open(io.BytesIO(x))
    raise TypeError(type(x))


def _q(v, p):
    v = sorted(v); i = (len(v) - 1) * p / 100.0
    lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (i - lo)) + v[hi] * (i - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--constantes", required=True)
    ap.add_argument("--imagens", default=None,
                    help="repo HF com as saidas ja geradas (coluna image_pred). "
                         "Sem isto, mede a LINHA DE IDENTIDADE (entrada = saida).")
    # Esquema dos repos `bokeh-eb-lfrepro-*`, gerados por run_3models.py:
    #   image_best_k     = saida do modelo no K otimo (busca binaria)
    #   image_real_bokeh = alvo
    # A AIF NAO esta neles, entao a profundidade vem do benchmark, casada por
    # `file_name_base`. Usar o alvo borrado para estimar profundidade daria uma
    # borda deslocada, que e exatamente o que a metrica nao pode ter.
    ap.add_argument("--col-pred", default="image_best_k")
    ap.add_argument("--col-alvo", default="image_real_bokeh")
    ap.add_argument("--col-entrada", default="image_blur")
    ap.add_argument("--col-aif-bench", default="image_focus")
    ap.add_argument("--col-chave", default="file_name_base")
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--lado", type=int, default=512)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--rotulo", default="identidade")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    sys.path.insert(0, "/proj")
    import torch, depth_pro
    from datasets import load_dataset
    from geo_cond.constants import GeoConstants
    from geo_cond.ebleed import e_bleed

    consts = GeoConstants(**json.load(open(a.constantes))["constantes"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, transform = depth_pro.create_model_and_transforms()
    modelo.eval().to(dev)

    ds = load_dataset(a.bench, split=a.split, token=os.environ.get("HF_TOKEN"))
    # indice do benchmark por chave, para pegar a AIF (fonte da profundidade)
    idx_bench = {}
    for j in range(len(ds)):
        k = str(ds[j][a.col_chave]) if a.col_chave in ds.column_names else str(j)
        idx_bench.setdefault(k, j)

    if a.imagens:
        preds = load_dataset(a.imagens, split=a.split,
                             token=os.environ.get("HF_TOKEN"))
        fonte = preds
        print(f"[f1] {len(preds)} predicoes de {a.imagens}", flush=True)
    else:
        fonte = ds          # linha de identidade: entrada = saida
        print("[f1] LINHA DE IDENTIDADE (entrada como predicao)", flush=True)

    alvo_n = a.n or len(fonte)
    reg = []
    faltando = 0
    for i in range(min(alvo_n, len(fonte))):
        r = fonte[i]
        try:
            if a.imagens:
                alvo = _pil(r[a.col_alvo]).convert("RGB")
                pred = _pil(r[a.col_pred]).convert("RGB")
                chave = str(r[a.col_chave]) if a.col_chave in fonte.column_names else str(i)
                j = idx_bench.get(chave)
                if j is None:
                    faltando += 1; continue
                aif = _pil(ds[j][a.col_aif_bench]).convert("RGB")
            else:
                alvo = _pil(r[a.col_aif_bench]).convert("RGB")
                pred = _pil(r[a.col_entrada]).convert("RGB")
                aif = alvo
        except Exception as e:
            print(f"[f1] {i}: {e}", flush=True); continue

        # tudo na mesma grade, senao a borda cai no lugar errado
        w, h = alvo.size
        esc = a.lado / min(w, h)
        nw, nh = max(a.lado, round(w*esc)), max(a.lado, round(h*esc))
        red = lambda im: np.asarray(im.resize((nw, nh), Image.BICUBIC), np.float32)/255.0
        A, P = red(alvo), red(pred)

        with torch.no_grad():
            out = modelo.infer(transform(aif).to(dev), f_px=None)
        z = out["depth"].squeeze().float().cpu().numpy()
        z = np.asarray(Image.fromarray(z).resize((nw, nh), Image.BILINEAR), np.float32)
        zf = z[np.isfinite(z) & (z > 0)]
        if zf.size < 1024: continue
        zmin, zmax = float(zf.min()), float(zf.max())
        d01 = np.clip((z - zmin) / max(zmax - zmin, 1e-9), 0.0, 1.0)

        res = e_bleed(P, A, d01, z_min_m=zmin, z_max_m=zmax, consts=consts,
                      theta=a.theta)
        if not np.isfinite(res.e_bleed): continue
        reg.append({"i": i, "e_bleed": res.e_bleed, "e_fora": res.e_fora,
                    "fracao_borda": res.fracao_borda})
        if len(reg) % 10 == 0: print(f"[f1] {len(reg)}", flush=True)

    if faltando:
        print(f"[f1] AVISO: {faltando} predicoes sem par no benchmark (chave "
              f"{a.col_chave!r}). Elas NAO entram na media.", flush=True)
    if not reg:
        print("nada medido", file=sys.stderr); return 1
    eb = [x["e_bleed"] for x in reg]; ef = [x["e_fora"] for x in reg]
    fr = [x["fracao_borda"] for x in reg]
    print("\n" + "="*64)
    print(f"E_bleed -- {a.rotulo} -- {a.bench} -- n={len(reg)}  theta={a.theta}")
    print("="*64)
    print(f"  E_bleed (borda) : media={np.mean(eb):.5f}  mediana={_q(eb,50):.5f}")
    print(f"  E_fora          : media={np.mean(ef):.5f}  mediana={_q(ef,50):.5f}")
    print(f"  razao borda/fora: {np.mean(eb)/max(np.mean(ef),1e-9):.3f}")
    print(f"  fracao de borda : mediana={_q(fr,50):.4f}  (se ~0 ou ~1, theta nao serve)")
    json.dump({"rotulo": a.rotulo, "bench": a.bench, "n": len(reg),
               "theta": a.theta,
               "e_bleed_media": float(np.mean(eb)), "e_bleed_mediana": _q(eb,50),
               "e_fora_media": float(np.mean(ef)), "e_fora_mediana": _q(ef,50),
               "fracao_borda_mediana": _q(fr,50), "registros": reg},
              open(a.saida, "w"), indent=1)
    print(f"\n  gravado em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
