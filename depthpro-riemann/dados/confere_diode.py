import glob
import numpy as np

fs = sorted(glob.glob("/data/diode/val/indoors/*/*/*_depth.npy"))[:6]
print(f"amostra de {len(fs)} arquivos\n")
for f in fs:
    d = np.load(f)
    m = np.load(f.replace("_depth.npy", "_depth_mask.npy"))
    v = d[m.astype(bool)] if m.shape[:2] == d.shape[:2] else d[d > 0]
    print(f"{f.split('/')[-1]}")
    print(f"  depth {d.dtype} {d.shape}  mask {m.dtype} {m.shape} valores={np.unique(m)[:4]}")
    print(f"  validos {m.astype(bool).mean()*100:.1f}%  min={v.min():.2f} p50={np.median(v):.2f} max={v.max():.2f} m")
