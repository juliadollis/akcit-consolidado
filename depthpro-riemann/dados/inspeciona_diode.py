"""Confere se o DIODE baixado tem a hierarquia e a fisica esperadas."""
import glob
import os

import numpy as np

RAIZ = "/data/diode"

print("=== TOPO ===")
for p in sorted(glob.glob(os.path.join(RAIZ, "*"))):
    print(" ", p)
for p in sorted(glob.glob(os.path.join(RAIZ, "val", "*"))):
    print("  ", p)

print("\n=== CONTAGEM POR DOMINIO ===")
for dom in ("indoors", "outdoor", "outdoors"):
    base = os.path.join(RAIZ, "val", dom)
    if not os.path.isdir(base):
        continue
    d = glob.glob(os.path.join(base, "scene_*", "scan_*", "*_depth.npy"))
    m = glob.glob(os.path.join(base, "scene_*", "scan_*", "*_depth_mask.npy"))
    r = glob.glob(os.path.join(base, "scene_*", "scan_*", "*.png"))
    cenas = set()
    scans = set()
    for x in d:
        scans.add(os.path.dirname(x))
        cenas.add(os.path.dirname(os.path.dirname(x)))
    print(f"  {dom}: depth={len(d)} mask={len(m)} png={len(r)} "
          f"cenas={len(cenas)} scans={len(scans)}")

print("\n=== AMOSTRA FISICA (indoors) ===")
alvo = os.path.join(RAIZ, "val", "indoors", "scene_*", "scan_*", "*_depth.npy")
amostras = sorted(glob.glob(alvo))[:5]
if not amostras:
    print("  NENHUMA amostra indoors encontrada!")
for p in amostras:
    d = np.load(p)
    stem = p[: -len("_depth.npy")]
    mp = stem + "_depth_mask.npy"
    rgb = stem + ".png"
    m = np.load(mp) if os.path.exists(mp) else None
    dd = d[..., 0] if d.ndim == 3 else d
    if m is not None:
        mm = m[..., 0] if m.ndim == 3 else m
        val = dd[(mm > 0) & np.isfinite(dd) & (dd > 0)]
        uni = np.unique(mm)
        binaria = set(uni.tolist()) <= {0.0, 1.0}
    else:
        val = dd[np.isfinite(dd) & (dd > 0)]
        uni, binaria = None, None
    print(f"\n  {os.path.basename(p)}")
    print(f"    depth dtype={d.dtype} shape={d.shape}")
    print(f"    rgb existe={os.path.exists(rgb)}  mask existe={m is not None}")
    if m is not None:
        print(f"    mask dtype={m.dtype} shape={m.shape} valores={uni[:6]} binaria={binaria}")
        print(f"    cobertura da mask = {100.0*float((mm>0).mean()):.1f}%")
    if val.size:
        print(f"    depth valido: min={val.min():.3f} mediana={np.median(val):.3f} "
              f"max={val.max():.3f} (m)  n={val.size}")
    else:
        print("    depth valido: VAZIO")
