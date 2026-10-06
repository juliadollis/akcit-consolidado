import sys,struct,json
sys.path.insert(0,'.')
from pq import R, rd_struct, rng, footer

def snappy(data):
    i=0; sh=0; ln=0
    while True:
        b=data[i]; i+=1; ln|=(b&0x7f)<<sh
        if not b&0x80: break
        sh+=7
    out=bytearray()
    n=len(data)
    while i<n:
        tag=data[i]; i+=1; t=tag&3
        if t==0:
            l=tag>>2
            if l<60: c=l+1
            else:
                nb=l-59; c=int.from_bytes(data[i:i+nb],'little')+1; i+=nb
            out+=data[i:i+c]; i+=c
        else:
            if t==1:
                c=4+((tag>>2)&7); off=((tag>>5)<<8)|data[i]; i+=1
            elif t==2:
                c=(tag>>2)+1; off=int.from_bytes(data[i:i+2],'little'); i+=2
            else:
                c=(tag>>2)+1; off=int.from_bytes(data[i:i+4],'little'); i+=4
            s=len(out)-off
            for j in range(c): out.append(out[s+j])
    assert len(out)==ln,(len(out),ln)
    return bytes(out)

def rle_hybrid(buf,bw,count):
    vals=[];i=0
    if bw==0: return [0]*count
    while len(vals)<count and i<len(buf):
        sh=0;hdr=0
        while True:
            b=buf[i];i+=1;hdr|=(b&0x7f)<<sh
            if not b&0x80: break
            sh+=7
        if hdr&1:
            ng=hdr>>1; nv=ng*8; nb=ng*bw
            ch=buf[i:i+nb]; i+=nb; bp=0
            for _ in range(nv):
                v=0
                for k in range(bw):
                    v|=((ch[bp>>3]>>(bp&7))&1)<<k; bp+=1
                vals.append(v)
        else:
            rl=hdr>>1; nb=(bw+7)//8
            v=int.from_bytes(buf[i:i+nb],'little'); i+=nb
            vals.extend([v]*rl)
    return vals[:count]

def bitwidth(m):
    w=0
    while (1<<w)<=m: w+=1
    return w-1 if (1<<(w-1))>m else w

def plain(buf,ptype,n):
    out=[];i=0
    if ptype==4:   # FLOAT
        for _ in range(n): out.append(struct.unpack("<f",buf[i:i+4])[0]); i+=4
    elif ptype==6: # BYTE_ARRAY
        for _ in range(n):
            l=struct.unpack("<I",buf[i:i+4])[0]; i+=4
            out.append(buf[i:i+l]); i+=l
    elif ptype==2:
        for _ in range(n): out.append(struct.unpack("<q",buf[i:i+8])[0]); i+=8
    elif ptype==1:
        for _ in range(n): out.append(struct.unpack("<i",buf[i:i+4])[0]); i+=4
    elif ptype==5:
        for _ in range(n): out.append(struct.unpack("<d",buf[i:i+8])[0]); i+=8
    else: raise ValueError(ptype)
    return out

def read_chunk(blob,base,cm):
    """blob = bytes starting at file offset `base`, covering the whole column chunk."""
    ptype=cm[1]; codec=cm[4]; nvals=cm[5]
    start=cm.get(11) or cm[9]
    pos=start-base
    end=start+cm[7]-base
    dictvals=None; values=[]; defs=[]
    while pos<end and len(values)<nvals:
        r=R(blob); r.i=pos
        ph=rd_struct(r); hdr_end=r.i
        ptypehdr=ph[1]; ucs=ph[2]; cs=ph[3]
        raw=blob[hdr_end:hdr_end+cs]
        data = snappy(raw) if codec==1 else raw
        pos=hdr_end+cs
        if ptypehdr==2:  # DICTIONARY_PAGE
            dh=ph[7]; dictvals=plain(data,ptype,dh[1]); continue
        if ptypehdr==0:  # DATA_PAGE v1
            dh=ph[5]; nv=dh[1]; enc=dh[2]
            j=0
            dl=struct.unpack("<I",data[j:j+4])[0]; j+=4
            dlv=rle_hybrid(data[j:j+dl],1,nv); j+=dl
            npresent=sum(dlv)
            body=data[j:]
            if enc in (2,8):
                bw=body[0]; idx=rle_hybrid(body[1:],bw,npresent)
                vs=[dictvals[x] for x in idx]
            elif enc==0:
                vs=plain(body,ptype,npresent)
            else: raise ValueError("enc %d"%enc)
            it=iter(vs)
            for d in dlv: values.append(next(it) if d else None)
        else: raise ValueError("pagetype %d"%ptypehdr)
    return values

def fetch_column(url,colname,md=None):
    if md is None: md,_=footer(url)
    out=[]
    for rg in md[4]:
        for cc in rg[1]:
            cm=cc[3]
            if '.'.join(x.decode() for x in cm[3])!=colname: continue
            start=cm.get(11) or cm[9]
            blob=rng(url,start,start+cm[7]-1)
            out.extend(read_chunk(blob,start,cm))
    return out
