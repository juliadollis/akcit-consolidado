"""Round-trip do decodificador PNG de png16.py, nos 5 filtros e 3 formatos.

Roda sem rede e sem GPU:  python3 test_png16.py
O decodificador existe porque esta maquina nao tem PIL nem numpy, e a auditoria
dos dfs precisava ler PNG de 16 bits. NAO use em producao: no cluster ha PIL.
"""
import zlib, struct, random, png16

def _chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)

def encode(vals, w, h, bd, ct, ftype):
    nch = 1 if ct == 0 else 3
    bpp = nch * (bd // 8); stride = w * bpp
    flat = struct.pack('>%dH' % len(vals), *vals) if bd == 16 else bytes(vals)
    raw = b''; prev = bytearray(stride)
    for y in range(h):
        line = bytearray(flat[y*stride:(y+1)*stride]); enc = bytearray(stride)
        for x in range(stride):
            a = line[x-bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x-bpp] if x >= bpp else 0
            if ftype == 0:   p = 0
            elif ftype == 1: p = a
            elif ftype == 2: p = b
            elif ftype == 3: p = (a + b) >> 1
            else:
                pp = a + b - c
                pa, pb, pc = abs(pp-a), abs(pp-b), abs(pp-c)
                p = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            enc[x] = (line[x] - p) & 0xFF
        raw += bytes([ftype]) + bytes(enc); prev = line
    return (b'\x89PNG\r\n\x1a\n'
            + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, bd, ct, 0, 0, 0))
            + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b''))

def main():
    random.seed(7); falhas = 0
    for bd, ct, nch in ((16, 0, 1), (8, 0, 1), (8, 2, 3)):
        w, h = 17, 9
        vals = [random.randrange(0, 65536 if bd == 16 else 256) for _ in range(w*h*nch)]
        for f in range(5):
            W, H, BD, CT, PX = png16.read_png(encode(vals, w, h, bd, ct, f))
            ok = PX == vals and (W, H, BD, CT) == (w, h, bd, ct)
            falhas += (not ok)
            print(f"bd={bd} ct={ct} filtro={f}: {'OK' if ok else 'FALHOU'}")
    print("TODOS OS FILTROS OK" if falhas == 0 else f"{falhas} FALHAS")
    return 1 if falhas else 0

if __name__ == "__main__":
    raise SystemExit(main())
