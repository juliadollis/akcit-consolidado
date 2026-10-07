import glob
import json
import os

fs = sorted(glob.glob("/host/runs_*/*/seed_*/summary.json"))[:4]
for f in fs:
    d = json.load(open(f))
    print(f.replace("/host/", ""))
    print("  chaves:", sorted(d)[:14])
    for k in sorted(d):
        if any(t in k.lower() for t in ("temp", "time", "dur", "epoc", "epoch")):
            print("   ", k, "=", d[k])

# fallback: mtime do best.pt contra o do primeiro artefato da pasta
for f in fs:
    p = os.path.dirname(f)
    arts = [os.path.join(p, a) for a in os.listdir(p)]
    ts = sorted(os.path.getmtime(a) for a in arts)
    print(p.replace("/host/", ""), "span=%.1f h" % ((ts[-1] - ts[0]) / 3600))
