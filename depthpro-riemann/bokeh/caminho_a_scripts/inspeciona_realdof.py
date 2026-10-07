"""Inspeciona o espelho RealDOF em disco: schema, nomes, modos e resolucoes."""
import glob, io, json
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image

RAIZ = "/workspace/hf-cache-julia/hub/datasets--akcit-pixel--RealDOF/snapshots"
fs = sorted(glob.glob(RAIZ + "/*/data/*.parquet"))
print("shards:", len(fs))
for f in fs:
    pf = pq.ParquetFile(f)
    print(" ", f.split("/")[-1], "linhas:", pf.metadata.num_rows,
          "row_groups:", pf.metadata.num_row_groups)
print("schema:", pq.ParquetFile(fs[0]).schema_arrow)

nomes, modos, hw, pares_hw = [], Counter(), Counter(), Counter()
for f in fs:
    t = pq.read_table(f).to_pydict()
    for i, nome in enumerate(t["file_name_base"]):
        nomes.append(nome)
        dims = {}
        for col in ("image_focus", "image_blur"):
            cel = t[col][i]
            b = cel["bytes"]
            with Image.open(io.BytesIO(b)) as im:
                modos[(col, im.mode, im.format)] += 1
                hw[(col, im.size)] += 1
                dims[col] = im.size
            if i == 0 and f == fs[0]:
                print("  path da celula:", col, "->", cel.get("path"))
        pares_hw[dims["image_focus"] == dims["image_blur"]] += 1

print("total linhas:", len(nomes))
print("nomes (primeiros 6):", nomes[:6])
print("nomes unicos:", len(set(nomes)))
suf = Counter(n.rsplit("_", 1)[-1] for n in nomes)
print("sufixos:", dict(suf))
print("modos/formatos:", dict(modos))
print("resolucoes distintas:", len(hw), "->", dict(list(hw.items())[:8]))
print("focus e blur com mesma resolucao:", dict(pares_hw))
json.dump(sorted(nomes), open("/workspace/caminho_a_rotac/out/realdof_nomes.json", "w"), indent=1)
