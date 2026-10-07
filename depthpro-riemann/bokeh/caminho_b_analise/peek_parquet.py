import pyarrow.parquet as pq, glob
fs=sorted(glob.glob("/raid/user_juliadollis/julia_docker/hf-cache-julia/hub/datasets--akcit-pixel--RealDOF/snapshots/*/data/*.parquet"))
print(len(fs),"arquivos")
f=pq.ParquetFile(fs[0])
print("schema:", f.schema_arrow)
print("linhas no arq0:", f.metadata.num_rows)
tot=sum(pq.ParquetFile(x).metadata.num_rows for x in fs)
print("total linhas:", tot)
t=f.read_row_group(0).slice(0,1).to_pydict()
for k,v in t.items():
    v0=v[0]
    print(" campo",k,type(v0), (str(v0)[:120] if not isinstance(v0,(bytes,dict)) else (list(v0.keys()) if isinstance(v0,dict) else f"bytes len={len(v0)}")))
