#!/usr/bin/env python3
"""Extrai membros ESCOLHIDOS de um zip remoto, sem baixar o zip inteiro.

Motivo: o passo 3 do RETESTE_CURVATURA le 10 lotes de 2 = 20 imagens. Baixar 24 GB
para ler 20 imagens custaria ~2 h. O S3 do DaRUS aceita Range (testado: HTTP 206),
e o formato zip guarda o indice no fim do arquivo, entao da para ler o indice por
range, escolher os membros e puxar so as faixas de bytes deles.

Amostramos AO LONGO DAS 37 SEQUENCIAS de proposito: o fx varia 4.7x entre elas, e a
escolha do teto de |K| depende justamente de cobrir essa variacao.
"""
import io, sys, zipfile, urllib.request, json, re

class HTTPFile(io.RawIOBase):
    def __init__(self, url_resolver, tamanho):
        self._resolver = url_resolver; self._url = url_resolver()
        self.tamanho = tamanho; self.pos = 0
    def _get(self, ini, fim):
        for tentativa in range(3):
            try:
                req = urllib.request.Request(self._url, headers={"Range": f"bytes={ini}-{fim}"})
                return urllib.request.urlopen(req, timeout=120).read()
            except Exception:
                if tentativa == 2: raise
                self._url = self._resolver()   # presigned pode expirar
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else (self.pos+off if whence==1 else self.tamanho+off)
        return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def read(self, n=-1):
        if n is None or n < 0: n = self.tamanho - self.pos
        n = min(n, self.tamanho - self.pos)
        if n <= 0: return b""
        d = self._get(self.pos, self.pos+n-1); self.pos += len(d); return d

def resolver(datafile_id):
    def f():
        api = f"https://darus.uni-stuttgart.de/api/access/datafile/{datafile_id}"
        r = urllib.request.urlopen(urllib.request.Request(api, method="GET"), timeout=60)
        return r.geturl()
    return f

def tamanho_de(datafile_id):
    api = "https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/DARUS-3376"
    d = json.load(urllib.request.urlopen(api, timeout=90))
    for x in d["data"]["latestVersion"]["files"]:
        if x["dataFile"]["id"] == datafile_id: return x["dataFile"]["filesize"]
    raise SystemExit(f"id {datafile_id} nao encontrado")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile-id", type=int, required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--por-seq", type=int, default=3, help="quantos quadros por sequencia")
    ap.add_argument("--inclui", default="", help="regex que o nome do membro deve casar")
    ap.add_argument("--so-listar", action="store_true")
    a = ap.parse_args()

    tam = tamanho_de(a.datafile_id)
    print(f"[zip] datafile {a.datafile_id}: {tam/1e9:.2f} GB", flush=True)
    fh = HTTPFile(resolver(a.datafile_id), tam)
    z = zipfile.ZipFile(fh)
    nomes = [n for n in z.namelist() if not n.endswith("/")]
    if a.inclui:
        rx = re.compile(a.inclui); nomes = [n for n in nomes if rx.search(n)]
    print(f"[zip] membros que casam: {len(nomes)}", flush=True)

    # agrupa por sequencia (o diretorio .../train/<seq>/...)
    porseq = {}
    for n in nomes:
        m = re.search(r"/train/([^/]+)/", n)
        porseq.setdefault(m.group(1) if m else "?", []).append(n)

    escolhidos = []
    for seq in sorted(porseq):
        lst = sorted(porseq[seq])
        # espalha ao longo da sequencia em vez de pegar so o comeco
        passo = max(1, len(lst)//max(1, a.por_seq))
        escolhidos += lst[::passo][: a.por_seq]
    print(f"[zip] sequencias: {len(porseq)}   membros escolhidos: {len(escolhidos)}", flush=True)
    if a.so_listar:
        for n in escolhidos[:10]: print("   ", n)
        return 0

    import os
    total = 0
    for i, n in enumerate(escolhidos, 1):
        alvo = os.path.join(a.dest, n)
        os.makedirs(os.path.dirname(alvo), exist_ok=True)
        if os.path.exists(alvo) and os.path.getsize(alvo) > 0:
            continue
        with z.open(n) as src, open(alvo, "wb") as dst:
            d = src.read(); dst.write(d); total += len(d)
        if i % 20 == 0:
            print(f"[zip] {i}/{len(escolhidos)}  {total/1e6:.0f} MB", flush=True)
    print(f"[zip] FIM: {len(escolhidos)} membros, {total/1e6:.0f} MB", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
