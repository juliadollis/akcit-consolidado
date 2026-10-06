import json, subprocess, sys, statistics as st, struct as _s
S='/private/tmp/claude-501/-Users-juliadollis-Projects-Code-repositorio-ref/69cf681d-d45c-45e4-9c6b-dc9aa899e736/scratchpad'
sys.path.insert(0,S); import png16, pq, cols
def local_cols(path,names):
    data=open(path,'rb').read(); flen=_s.unpack('<I',data[-8:-4])[0]
    r=pq.R(data[len(data)-8-flen:len(data)-8]); mdl=pq.rd_struct(r)
    out={n:[] for n in names}
    for rg in mdl[4]:
        for cc in rg[1]:
            cm=cc[3]; nm='.'.join(x.decode() for x in cm[3])
            if nm in names:
                stt=cm.get(11) or cm[9]; out[nm].extend(cols.read_chunk(data[stt:stt+cm[7]],stt,cm))
    return out
K=local_cols(S+"/audit/kfix.parquet",["stem","k_eq3","z_focus_m","z_min_m","z_max_m","coc_p99_px"])
kmap={(s.decode() if isinstance(s,bytes) else s):i for i,s in enumerate(K["stem"])}
def pct(v,p):
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
eM=[];eD=[];det=[]
for src in sys.argv[1:]:
    for row in json.load(open(src))['rows']:
        r=row['row']; stem=r['stem']
        if stem not in kmap: continue
        i=kmap[stem]; keq=K["k_eq3"][i]; zf=K["z_focus_m"][i]*1000.0
        zmin=K["z_min_m"][i]*1000.0; zmax=K["z_max_m"][i]*1000.0; rec=K["coc_p99_px"][i]
        blob=subprocess.run(["curl","-sL","--max-time","60",r['depth']['src']],capture_output=True).stdout
        try: w,h,bd,ct,px=png16.read_png(blob)
        except Exception as e: print("skip",stem,e,flush=True); continue
        d01=[v/65535.0 for v in px[::11]]
        pM=pct([keq*abs(1.0/max(zmin+v*(zmax-zmin),1e-4)-1.0/zf) for v in d01],99)
        pD=pct([keq*abs((1.0/zmax+v*(1.0/zmin-1.0/zmax))-1.0/zf) for v in d01],99)
        a,b=100*abs(pM-rec)/rec,100*abs(pD-rec)/rec
        eM.append(a); eD.append(b); det.append((stem,rec,pM,a,pD,b,K["z_max_m"][i]))
        print(f"{stem} rec={rec:.4f} M={pM:.4f}({a:.2f}%) D={pD:.4f}({b:.2f}%) zmax={K['z_max_m'][i]:.1f}",flush=True)
n=len(eM)
print(f"\nN={n}")
print(f"HIP. METRICA     erro% mediana={st.median(eM):.3f} max={max(eM):.2f}  <1%:{sum(1 for x in eM if x<1)}/{n}")
print(f"HIP. DISPARIDADE erro% mediana={st.median(eD):.3f} max={max(eD):.2f}  <1%:{sum(1 for x in eD if x<1)}/{n}")
