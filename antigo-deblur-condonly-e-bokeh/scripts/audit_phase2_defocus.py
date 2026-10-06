#!/usr/bin/env python3
"""Audita a SUPERVISÃO das rotas da FASE 2 (b e c): o mapa de defocus recomposto
bate com o blur que o alvo realmente tem?

Motivo (achado em 2026-08-13): na inferência de teste do step 27500, a amostra
`c_1000` da rota c veio com o mapa de defocus RECOMPOSTO inteiramente ZERO
(nenhum blur pedido), mas a imagem `bokeh` (o ALVO) tinha bokeh real e forte no
fundo. O modelo obedeceu o mapa e devolveu a imagem nítida, o que está CERTO
dado o condicionamento — o par é que é contraditório.

Treinar nesses pares ensina "mapa zero → borre mesmo assim", que é exatamente o
oposto da controlabilidade que o paper mede (LVCorr, §4.1). Como as rotas b e c
são o dado da FASE 2, isso precisa ser quantificado ANTES de gastar 60K steps.

O que o script mede, por amostra (streaming, sem baixar os GB):
  - `map_max`, `map_mean`: o mapa recomposto clip(k*|depth-s1|/max_coc, 0, 1).
  - `frac_zero`: fração de pixels ~0 (nada de blur pedido).
  - `frac_sat`:  fração de pixels em 1.0 (mapa saturado, o teto do max_coc).
  - `lapvar_aif` / `lapvar_bokeh`: variância do Laplaciano (proxy de nitidez).
    `blur_ratio = lapvar_bokeh / lapvar_aif` < 1 significa que o alvo É mais
    borrado que a entrada, ou seja, existe bokeh de verdade para aprender.
  - `k`, `s1`.

Classifica cada amostra:
  CONTRADITORIA  mapa ~zero (map_max < 0.02) MAS alvo nitidamente mais borrado
                 (blur_ratio < 0.7). Supervisão que ensina a ignorar o mapa.
  SATURADA       frac_sat > 0.5. O mapa perdeu a variação espacial no teto.
  DEGENERADA_OK  mapa ~zero e alvo ~igual à entrada (par consistente, sem blur).
  OK             o resto.

Uso (sempre via SLURM sem GPU):
    python3 scripts/audit_phase2_defocus.py --datasets AKCITPixel3/BKXcuVXCmeRvN,AKCITPixel3/CMiQdveBBzNii --n 300

Só leitura. Escreve um JSON com o resumo e os piores casos.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from datasets import load_dataset


def _to_gray(img) -> np.ndarray:
    a = np.asarray(img.convert("L"), dtype=np.float64)
    return a


def _lap_var(img) -> float:
    """Variância do Laplaciano — proxy clássico de nitidez (o mesmo critério que
    o paper usa para filtrar imagens borradas, §B.1)."""
    g = _to_gray(img)
    if g.size == 0:
        return 0.0
    # kernel laplaciano 3x3 via diferenças finitas, sem depender de scipy/cv2
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def _depth01(rec) -> np.ndarray:
    """A coluna `depth` é profundidade normalizada em [0,1], gravada em uint16."""
    d = np.asarray(rec["depth"], dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    return d / 65535.0


def audit_one(rec, max_coc: float) -> dict:
    k = float(rec.get("k") or 0.0)
    s1 = float(rec.get("s1") or 0.0)
    depth = _depth01(rec)
    dmap = np.clip(k * np.abs(depth - s1) / max_coc, 0.0, 1.0)

    lv_aif = _lap_var(rec["aif"])
    lv_bok = _lap_var(rec["bokeh"])
    ratio = (lv_bok / lv_aif) if lv_aif > 1e-9 else float("nan")

    out = {
        "stem": str(rec.get("stem", "")),
        "k": k,
        "s1": s1,
        "map_max": float(dmap.max()),
        "map_mean": float(dmap.mean()),
        "frac_zero": float((dmap < 0.02).mean()),
        "frac_sat": float((dmap >= 0.999).mean()),
        "lapvar_aif": lv_aif,
        "lapvar_bokeh": lv_bok,
        "blur_ratio": ratio,
    }

    if out["map_max"] < 0.02:
        out["classe"] = "CONTRADITORIA" if (ratio == ratio and ratio < 0.7) else "DEGENERADA_OK"
    elif out["frac_sat"] > 0.5:
        out["classe"] = "SATURADA"
    else:
        out["classe"] = "OK"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="repos HF separados por virgula")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=300, help="amostras por dataset")
    ap.add_argument("--max-coc", type=float, default=100.0)
    ap.add_argument("--out", default="outputs/audit_phase2_defocus.json")
    args = ap.parse_args()

    resumo = {}
    for repo in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"\n{'=' * 70}\n[audit] {repo} (n={args.n})\n{'=' * 70}", flush=True)
        try:
            ds = load_dataset(repo, split=args.split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[audit] ERRO abrindo {repo}: {exc}", flush=True)
            continue

        linhas = []
        for i, rec in enumerate(ds):
            if i >= args.n:
                break
            try:
                linhas.append(audit_one(rec, args.max_coc))
            except Exception as exc:  # noqa: BLE001
                print(f"[audit] amostra {i} falhou: {type(exc).__name__}: {exc}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"[audit]   {i + 1}/{args.n}", flush=True)

        if not linhas:
            continue
        classes: dict[str, int] = {}
        for r in linhas:
            classes[r["classe"]] = classes.get(r["classe"], 0) + 1

        n = len(linhas)
        print(f"\n[audit] {repo}: {n} amostras", flush=True)
        for c in sorted(classes, key=lambda x: -classes[x]):
            print(f"[audit]   {c:16s} {classes[c]:5d}  ({100.0 * classes[c] / n:5.1f}%)", flush=True)

        ks = np.array([r["k"] for r in linhas])
        mx = np.array([r["map_max"] for r in linhas])
        print(f"[audit]   k:       min={ks.min():.1f} mediana={np.median(ks):.1f} max={ks.max():.1f}", flush=True)
        print(f"[audit]   map_max: min={mx.min():.3f} mediana={np.median(mx):.3f} max={mx.max():.3f}", flush=True)
        print(f"[audit]   fracao com map_max<0.02: {100.0 * (mx < 0.02).mean():.1f}%", flush=True)

        piores = sorted(
            [r for r in linhas if r["classe"] == "CONTRADITORIA"],
            key=lambda r: r["blur_ratio"],
        )[:10]
        if piores:
            print("[audit]   piores CONTRADITORIAS (mapa zero, alvo borrado):", flush=True)
            for r in piores:
                print(
                    f"[audit]     {r['stem'][:40]:40s} k={r['k']:7.2f} s1={r['s1']:.4f} "
                    f"map_max={r['map_max']:.4f} blur_ratio={r['blur_ratio']:.3f}",
                    flush=True,
                )
        resumo[repo] = {"n": n, "classes": classes, "amostras": linhas}

    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(resumo, fh, indent=2)
        print(f"\n[audit] JSON salvo em {args.out}", flush=True)
    except OSError as exc:
        print(f"[audit] AVISO: nao salvei o JSON: {exc}", flush=True)
    print("[audit] FIM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
