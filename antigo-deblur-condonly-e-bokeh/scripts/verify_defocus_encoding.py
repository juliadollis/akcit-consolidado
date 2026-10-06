"""Verifica a codificacao do `defocus_map` ANTES de escrever o dataloader.

Pergunta central (mesma familia dos bugs de range que ja pegamos):
  Qual transform de `defocus_map` (uint16) reproduz o mapa em [0,1] que a
  inferencia oficial injeta no VAE?

  Inferencia oficial (Inference_bokehNet.py):
      disp        = 1.0 / depth_metric
      defocus_abs = | k * (disp - disp_focus) |        # disp_focus == s1
      cond_map    = (defocus_abs / MAX_COC=100).clamp(0, 1)   # em [0,1]

  Se voces salvaram `defocus_map` = cond_map * 65535 (mesma normalizacao
  ABSOLUTA, /100 + clamp), entao  defocus_map/65535  casa exatamente.
  Se foi min-max POR IMAGEM, NAO casa -> seria bug de condicionamento.

O script:
  1. Dumpa qc + exif COMPLETOS (procuramos MAX_COC / escala de depth / normalizacao).
  2. Checa se defocus_map ~ 0 na regiao de foco (foreground_mask).
  3. Tenta reproduzir defocus_map a partir de depth+s1+k sob varias hipoteses
     de codificacao do `depth`, e reporta qual bate (correlacao + erro).

Nao baixa tudo (streaming). Roda em CPU.
  python3 scripts/verify_defocus_encoding.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from datasets import load_dataset

DATASETS = [
    "AKCITPixel3/AfONERuvNmglv",   # rota "a"
    "AKCITPixel3/BKXcuVXCmeRvN",   # rota "b"
    "AKCITPixel3/CMiQdveBBzNii",   # rota "c"
]
N_ROWS = 4
TOKEN = os.environ.get("HF_TOKEN")
U16 = 65535.0


def as_arr(img):
    return np.asarray(img).astype(np.float64)


def pct(a):
    return {p: round(float(np.percentile(a, p)), 4) for p in (0, 1, 50, 99, 100)}


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def try_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def analyze_row(row, idx):
    print(f"\n  ┌── linha {idx}  stem={row.get('stem')} route={row.get('route')} "
          f"s1={row.get('s1'):.6g} k={row.get('k'):.6g}")

    qc = try_json(row.get("qc") or "")
    ex = try_json(row.get("exif") or "")
    if qc:
        print("  │ qc  :", json.dumps(qc, ensure_ascii=False))
    if ex:
        print("  │ exif:", json.dumps(ex, ensure_ascii=False))

    depth = as_arr(row["depth"])          # uint16 -> float
    dmap = as_arr(row["defocus_map"])     # uint16 -> float
    mask = as_arr(row["foreground_mask"]) > 127

    print(f"  │ depth      : {pct(depth)}")
    print(f"  │ defocus_map: {pct(dmap)}   (÷65535 -> [{dmap.min()/U16:.3f}, {dmap.max()/U16:.3f}])")

    # (2) defocus na regiao de foco deve ser ~0
    if mask.any():
        infoc = dmap[mask] / U16
        outfoc = dmap[~mask] / U16 if (~mask).any() else np.array([0.0])
        print(f"  │ defocus/65535 DENTRO do mask (foco): mediana={np.median(infoc):.4f} "
              f"| FORA: mediana={np.median(outfoc):.4f}")

    # (3) reproduzir defocus_map a partir de depth+s1+k, testando hipoteses de
    #     codificacao do `depth`. Reportamos correlacao com defocus_map e o erro
    #     da versao *65535 clampada.
    s1 = float(row["s1"])
    k = float(row["k"])
    target01 = np.clip(dmap / U16, 0, 1)   # o que achamos ser o cond_map

    hyps = {}
    # H1: depth_uint16 JA e' disparidade (escala 0..1 = depth/65535)
    disp1 = depth / U16
    hyps["depth/65535 = disp"] = np.abs(k * (disp1 - s1))
    # H2: depth_uint16 e' disparidade em unidade crua (sem escala)
    hyps["depth = disp (cru)"] = np.abs(k * (depth - s1))
    # H3: depth_uint16 e' profundidade METRICA normalizada 0..1 -> disp=1/depth
    safe = np.where(depth > 0, depth / U16, np.finfo(np.float32).max)
    disp3 = 1.0 / safe
    hyps["disp=1/(depth/65535)"] = np.abs(k * (disp3 - s1))
    # H4: depth_uint16 metrica crua -> disp = 1/depth
    safe4 = np.where(depth > 0, depth, np.finfo(np.float32).max)
    hyps["disp=1/depth(cru)"] = np.abs(k * (1.0 / safe4 - s1))

    print("  │ --- reproducao do defocus_map (corr com defocus_map/65535) ---")
    best = None
    for name, defocus_abs in hyps.items():
        # tenta com a normalizacao /MAX_COC=100 (a da inferencia) e tambem /max
        cond_100 = np.clip(defocus_abs / 100.0, 0, 1)
        c = corr(cond_100, target01)
        mae = float(np.mean(np.abs(cond_100 - target01)))
        tag = f"{name:<24} corr={c:+.3f}  MAE(/100,clamp)={mae:.4f}"
        print("  │   ", tag)
        if best is None or abs(c) > abs(best[1]):
            best = (name, c)
    print(f"  └── melhor hipotese: {best[0]}  (corr={best[1]:+.3f})")


def main():
    for name in DATASETS:
        print("=" * 80)
        print(name)
        print("=" * 80)
        try:
            ds = load_dataset(name, split="train", streaming=True, token=TOKEN)
        except Exception as exc:
            print("  ERRO:", exc)
            continue
        for i, row in enumerate(ds):
            if i >= N_ROWS:
                break
            analyze_row(row, i)
        print()


if __name__ == "__main__":
    main()
