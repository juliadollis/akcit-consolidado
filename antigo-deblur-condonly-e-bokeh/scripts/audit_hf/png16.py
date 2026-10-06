import zlib, struct

def read_png(blob):
    """PNG -> (w, h, bitdepth, colortype, pixels as flat list of ints, row-major).
    Suporta grayscale 8/16 bits e RGB 8 bits. Sem interlace."""
    assert blob[:8] == b'\x89PNG\r\n\x1a\n', "nao e PNG"
    i = 8; idat = b''; w = h = bd = ct = None; interlace = 0
    while i < len(blob):
        ln = struct.unpack('>I', blob[i:i+4])[0]
        typ = blob[i+4:i+8]
        data = blob[i+8:i+8+ln]
        if typ == b'IHDR':
            w, h, bd, ct, comp, filt, interlace = struct.unpack('>IIBBBBB', data)
        elif typ == b'IDAT':
            idat += data
        elif typ == b'IEND':
            break
        i += 12 + ln
    assert interlace == 0, "interlace nao suportado"
    assert ct in (0, 2), f"colortype {ct} nao suportado"
    nch = 1 if ct == 0 else 3
    bpp = nch * (bd // 8)              # bytes por pixel
    stride = w * bpp
    raw = zlib.decompress(idat)
    assert len(raw) == h * (stride + 1), f"tamanho inesperado {len(raw)} vs {h*(stride+1)}"
    out = bytearray(h * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(h):
        f = raw[pos]; pos += 1
        line = bytearray(raw[pos:pos+stride]); pos += stride
        if f == 1:
            for x in range(bpp, stride): line[x] = (line[x] + line[x-bpp]) & 0xFF
        elif f == 2:
            for x in range(stride): line[x] = (line[x] + prev[x]) & 0xFF
        elif f == 3:
            for x in range(stride):
                a = line[x-bpp] if x >= bpp else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif f == 4:
            for x in range(stride):
                a = line[x-bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x-bpp] if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 0xFF
        elif f != 0:
            raise ValueError(f"filtro {f} invalido")
        out[y*stride:(y+1)*stride] = line
        prev = line
    if bd == 16:
        px = list(struct.unpack('>%dH' % (len(out)//2), bytes(out)))
    else:
        px = list(out)
    return w, h, bd, ct, px
