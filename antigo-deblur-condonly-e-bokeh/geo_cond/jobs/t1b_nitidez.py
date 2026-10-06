"""T1b -- confirmação independente do plano de foco, pela NITIDEZ do bokeh real.

O T1 diz qual dos dois candidatos implementa a Eq. 4. Isso não é a mesma coisa
que estar no foco de verdade: na rota b o bokeh é uma FOTO REAL, ninguém escolheu
o plano de foco, a câmera escolheu. A Eq. 4 é uma ESTIMATIVA desse plano, via
máscara do BiRefNet, e `s1` é outra estimativa.

Este job não depende de nenhuma máscara: varre a imagem de bokeh real em janelas,
acha a região mais NITIDA (variância do laplaciano máxima), e lê o `depth01` ali.
A física não mente sobre onde está o foco.

Uso: python3 t1b_nitidez.py --n 120 --kfix-parquet ... --saida t1b.json
"""
from __future__ import annotations
import argparse, io, json, os, subprocess, sys, urllib.parse
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from t1_plano_de_foco import _curl, linhas, _carrega_kfix, ROTA_B


def mapa_nitidez(g: np.ndarray, jan: int = 32) -> np.ndarray:
    """Variância do laplaciano por janela jan x jan, sem sobreposição."""
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    H, W = lap.shape
    nh, nw = H // jan, W // jan
    if nh < 2 or nw < 2:
        return np.zeros((1, 1))
    blocos = lap[:nh * jan, :nw * jan].reshape(nh, jan, nw, jan).transpose(0, 2, 1, 3)
    return blocos.reshape(nh, nw, -1).var(axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--janela", type=int, default=32)
    ap.add_argument("--kfix-parquet", required=True)
    ap.add_argument("--saida", default="t1b.json")
    a = ap.parse_args()
    kfix = _carrega_kfix(a.kfix_parquet)
    passo = max(1, 11635 // max(1, a.n // 20))
    reg = []
    for off in list(range(0, 11615, passo))[: max(1, a.n // 20)]:
        try:
            rows = linhas(ROTA_B, off, 20)
        except Exception as e:
            print(f"[t1b] offset {off}: {e}"); continue
        for r in rows:
            stem = r["stem"]
            if stem not in kfix: continue
            try:
                bok = Image.open(io.BytesIO(_curl(r["bokeh"]["src"], True))).convert("L")
                d = np.asarray(Image.open(io.BytesIO(_curl(r["depth"]["src"], True))), np.float32)
            except Exception as e:
                print(f"[t1b] {stem}: {e}"); continue
            g = np.asarray(bok, np.float32)
            nit = mapa_nitidez(g, a.janela)
            if nit.size < 4: continue
            iy, ix = np.unravel_index(int(np.argmax(nit)), nit.shape)
            d01 = d / 65535.0 if d.max() > 1.5 else d
            # janela correspondente no mapa de profundidade
            sy, sx = d01.shape[0] / g.shape[0], d01.shape[1] / g.shape[1]
            y0 = int(iy * a.janela * sy); y1 = int((iy + 1) * a.janela * sy)
            x0 = int(ix * a.janela * sx); x1 = int((ix + 1) * a.janela * sx)
            bloco = d01[y0:y1, x0:x1]
            if bloco.size < 16: continue
            foco_medido = float(np.median(bloco))
            e = kfix[stem]; faixa = e["z_max_m"] - e["z_min_m"]
            if faixa <= 0: continue
            b = (e["z_focus_m"] - e["z_min_m"]) / faixa
            reg.append({"stem": stem, "foco_por_nitidez": foco_medido,
                        "s1": float(r["s1"]), "z_focus_normalizado": b,
                        "erro_s1": abs(foco_medido - float(r["s1"])),
                        "erro_zfocus": abs(foco_medido - b),
                        "contraste_nitidez": float(nit.max() / max(np.median(nit), 1e-9))})
        print(f"[t1b] offset {off}: acumulado {len(reg)}", flush=True)
        if len(reg) >= a.n: break

    if not reg: print("nenhuma amostra", file=sys.stderr); return 1
    es1 = np.array([x["erro_s1"] for x in reg]); ezf = np.array([x["erro_zfocus"] for x in reg])
    v = int((es1 < ezf).sum())
    print("\n" + "=" * 66)
    print(f"T1b -- foco por NITIDEZ do bokeh real, {len(reg)} amostras, janela {a.janela}px")
    print("=" * 66)
    print(f"  |nitidez - s1       |  mediana={np.median(es1):.5f}  p90={np.percentile(es1,90):.5f}")
    print(f"  |nitidez - z_focus  |  mediana={np.median(ezf):.5f}  p90={np.percentile(ezf,90):.5f}")
    print(f"\n  s1 mais perto em      {v}/{len(reg)} ({100*v/len(reg):.1f}%)")
    print(f"  z_focus mais perto em {len(reg)-v}/{len(reg)} ({100*(len(reg)-v)/len(reg):.1f}%)")
    forte = [x for x in reg if x["contraste_nitidez"] > 5.0]
    if forte:
        vf = sum(1 for x in forte if x["erro_s1"] < x["erro_zfocus"])
        print(f"\n  so com pico de nitidez FORTE (contraste>5, n={len(forte)}): "
              f"s1 vence em {vf} ({100*vf/len(forte):.1f}%)")
    print(f"\n  VEREDITO: {'s1' if v > len(reg)/2 else 'z_focus_m'} mais proximo do foco fisico")
    json.dump({"n": len(reg), "vence_s1": v, "registros": reg}, open(a.saida, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
