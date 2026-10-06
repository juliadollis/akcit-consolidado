import os,sys,json,struct,subprocess

TOKEN=os.environ.get("HF_TOKEN","")

def rng(url,start,end):
    r=subprocess.run(["curl","-sL","-H","Authorization: Bearer "+TOKEN,
                      "-H","Range: bytes=%d-%d"%(start,end),url],capture_output=True)
    return r.stdout

def size_of(url):
    r=subprocess.run(["curl","-sIL","-H","Authorization: Bearer "+TOKEN,url],capture_output=True,text=True)
    n=None
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"): n=int(line.split(":")[1].strip())
        if line.lower().startswith("x-linked-size:"): n=int(line.split(":")[1].strip())
    return n

# ---------- thrift compact ----------
class R:
    def __init__(s,b): s.b=b; s.i=0
    def u8(s): v=s.b[s.i]; s.i+=1; return v
    def varint(s):
        sh=0; res=0
        while True:
            b=s.u8(); res|=(b&0x7f)<<sh
            if not b&0x80: return res
            sh+=7
    def zz(s):
        n=s.varint(); return (n>>1)^-(n&1)
    def binary(s):
        n=s.varint(); v=s.b[s.i:s.i+n]; s.i+=n; return v
    def dbl(s):
        v=struct.unpack("<d",s.b[s.i:s.i+8])[0]; s.i+=8; return v

def rd_val(r,t):
    if t==1: return True
    if t==2: return False
    if t==3: return r.u8()
    if t in (4,5,6): return r.zz()
    if t==7: return r.dbl()
    if t==8: return r.binary()
    if t in (9,10):
        h=r.u8(); sz=h>>4; et=h&0x0f
        if sz==15: sz=r.varint()
        return [rd_val(r,et) for _ in range(sz)]
    if t==11:
        sz=r.varint()
        if sz==0: return {}
        kv=r.u8(); kt=kv>>4; vt=kv&0x0f
        return {rd_val(r,kt):rd_val(r,vt) for _ in range(sz)}
    if t==12: return rd_struct(r)
    raise ValueError("type %d"%t)

def rd_struct(r):
    out={}; last=0
    while True:
        h=r.u8()
        if h==0: return out
        t=h&0x0f; d=h>>4
        fid = last+d if d else r.zz()
        last=fid
        out[fid]=rd_val(r,t)

def footer(url):
    n=size_of(url)
    tail=rng(url,n-8,n-1)
    assert tail[4:8]==b"PAR1", tail
    flen=struct.unpack("<I",tail[0:4])[0]
    fb=rng(url,n-8-flen,n-9)
    assert len(fb)==flen,(len(fb),flen)
    return rd_struct(R(fb)),n

PTYPE={0:"BOOLEAN",1:"INT32",2:"INT64",3:"INT96",4:"FLOAT",5:"DOUBLE",6:"BYTE_ARRAY",7:"FBA"}

def analyze(url):
    md,fsize=footer(url)
    nrows=md.get(3)
    schema=md.get(2,[])
    names=[ (e.get(4) or b"").decode() for e in schema ]
    cols={}
    for rg in md.get(4,[]):
        for cc in rg.get(1,[]):
            cm=cc.get(3)
            if not cm: continue
            path=".".join(x.decode() for x in cm.get(3,[]))
            st=cm.get(12) or {}
            d=cols.setdefault(path,{"type":PTYPE.get(cm.get(1),"?"),"num_values":0,
                                    "null_count":0,"has_stats":0,"nrg":0,
                                    "min":None,"max":None,"total_size":0})
            d["nrg"]+=1
            d["num_values"]+=cm.get(5,0)
            d["total_size"]+=cm.get(7,0) or 0
            if 3 in st:
                d["null_count"]+=st[3]; d["has_stats"]+=1
            mn=st.get(6, st.get(2)); mx=st.get(5, st.get(1))
            for key,v in (("min",mn),("max",mx)):
                if v is None: continue
                d[key]=v if d[key] is None else d[key]
    return {"num_rows":nrows,"file_size":fsize,"schema":names,"cols":cols,
            "n_rowgroups":len(md.get(4,[])),"created_by":(md.get(6) or b"").decode()}

def dec(v,t):
    if v is None: return None
    if t=="FLOAT" and len(v)==4: return struct.unpack("<f",v)[0]
    if t=="DOUBLE" and len(v)==8: return struct.unpack("<d",v)[0]
    if t=="INT32" and len(v)==4: return struct.unpack("<i",v)[0]
    if t=="INT64" and len(v)==8: return struct.unpack("<q",v)[0]
    if t=="BYTE_ARRAY":
        try: return v.decode()[:60]
        except: return repr(v[:20])
    return repr(v[:16])
