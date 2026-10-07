import pyarrow.parquet as pq, glob, io, json, os
from PIL import Image
from PIL.ExifTags import TAGS
dst="/raid/user_juliadollis/julia_docker/caminho_b_analise/realdof_focus"
fs=sorted(glob.glob("/raid/user_juliadollis/julia_docker/hf-cache-julia/hub/datasets--akcit-pixel--RealDOF/snapshots/*/data/*.parquet"))
n=0; exif_info=[]
for f in fs:
    t=pq.read_table(f).to_pydict()
    for i,nome in enumerate(t["file_name_base"]):
        for campo,suf in (("image_focus","focus"),("image_blur","blur")):
            b=t[campo][i]["bytes"]
            im=Image.open(io.BytesIO(b))
            ex=im.getexif()
            if suf=="focus":
                exif_info.append({"nome":nome,"size":im.size,"format":im.format,
                                  "n_exif":len(ex),
                                  "exif":{TAGS.get(k,k):str(v)[:40] for k,v in ex.items()}})
            im.convert("RGB").save(os.path.join(dst,f"{nome}__{suf}.png"))
        n+=1
print("amostras:",n)
com=[e for e in exif_info if e["n_exif"]>0]
print("com EXIF:",len(com),"de",len(exif_info))
print("exemplo:",json.dumps(exif_info[0],ensure_ascii=False))
json.dump(exif_info,open(dst+"/_exif.json","w"),ensure_ascii=False,indent=1)
