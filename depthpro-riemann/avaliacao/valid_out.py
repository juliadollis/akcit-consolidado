import glob

import numpy as np

for nome, raiz in (("indoors", "/data/diode_prep/val"),
                   ("outdoor", "/data/diode_out_prep/val")):
    fs = sorted(glob.glob(raiz + "/mask/*.npy"))[:60]
    fr = [np.load(f).astype(bool).mean() for f in fs]
    ds = [np.load(f.replace("/mask/", "/depth/")) for f in fs[:20]]
    v = np.concatenate([d[np.load(f.replace("/mask/", "/depth/")) > 0].ravel()
                        for d, f in zip(ds, fs[:20])])
    print("%s: amostra=%d  validos med=%.1f%% (min %.1f%%)  "
          "profundidade p50=%.2f p99=%.2f max=%.2f m"
          % (nome, len(fs), 100 * np.median(fr), 100 * min(fr),
             np.median(v), np.percentile(v, 99), v.max()))
