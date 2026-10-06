"""T2 -- o teto do `k` na rota c é censura de faixa ou saturação do renderizador.

DESVIO REGISTRADO EM RELACAO AO PLANO
-------------------------------------
O `PLANO_CONDICIONAMENTO_GEOMETRICO.md` seção 8 (teste T2) manda refazer o sweep
da Eq. 5 com `K_max = 1000` nas amostras censuradas. Isso exige o renderizador
`R` da Eq. 5, que é o `_render_bokeh_simple` do repositório
`bokehnet-data-pipeline`, do time de dados. Ele NAO está neste repositório:

    grep -rln "_render_bokeh_simple" .   ->  nenhum resultado

Reimplementar o renderizador tornaria o teste inútil: o sweep mediria o NOSSO
renderizador, não o que produziu os rótulos, e a pergunta é justamente sobre o
comportamento daquele.

Substituto executável, que separa as mesmas duas hipóteses sem renderizador
nenhum. As duas fazem previsões diferentes sobre grandezas que já existem:

  A) CENSURA DE FAIXA. O K verdadeiro das censuradas é genuinamente maior que
     300. Previsão: o ALVO delas é mais borrado que o das interiores, e o
     `blur_ratio` acompanha o `k`.

  B) SATURACAO DO RENDERIZADOR. O `_render_bokeh_simple` limita o kernel a 51 px
     (DECISOES_FASE2 seção 7), então acima de um certo CoC o SSIM para de
     melhorar e o argmax corre para a borda. Previsão: existe um JOELHO no SSIM
     de calibração em torno de um CoC de ~51 px, e as censuradas são as que o
     ultrapassam, independentemente de quão borrado o alvo esteja.

Grandezas medidas, todas sem GPU e sem renderizador:
  blur_ratio = var_laplaciano(bokeh) / var_laplaciano(aif)      (a mesma medida
               que `scripts/audit_phase2_defocus.py` usa)
  coc_max_px = k * max|depth01 - s1|                            (o numerador de
               `defocus_from_depth`, antes de dividir por max_coc)

Se o time de dados devolver o `_render_bokeh_simple`, o teste literal da Eq. 5
volta a ser possível e este job vira diagnóstico complementar.

ACESSO AOS DADOS
----------------
A rota c NAO tem viewer no datasets-server (`is-valid` devolve viewer=false,
filter=false, statistics=false), entao o endpoint `/rows` falha em TODOS os
offsets. Medido: 0 de 300 amostras. O unico caminho e a biblioteca `datasets`,
que e exatamente como o dataloader do treino a le (`data.py:_load_hf_split`).

Por isso este job roda NO CLUSTER, dentro do container, e nao no Mac.

Uso (no container, sem GPU):
    python3 t2_teto_do_k.py --n 300 --saida t2_resultado.json
"""

from __future__ import annotations

import argparse, io, json, os, subprocess, sys, urllib.parse

import numpy as np
from PIL import Image

ROTA_C = "AKCITPixel3/CMiQdveBBzNii"
N_ROTA_C = 2932
TETO_SWEEP = 300.0
KERNEL_MAX_PX = 51.0      # limite do _render_bokeh_simple (DECISOES_FASE2 s.7)
DS_ROWS = "https://datasets-server.huggingface.co/rows"


def _curl(url, binario=False):
    tok = os.environ.get("HF_TOKEN", "")
    r = subprocess.run(["curl", "-sL", "--max-time", "90",
                        "-H", f"Authorization: Bearer {tok}", url], capture_output=True)
    return r.stdout if binario else json.loads(r.stdout)


def _pil(x):
    """A coluna vem como PIL (datasets) ou como dict com bytes (parquet cru)."""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict) and "bytes" in x:
        return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    raise TypeError(f"tipo de imagem inesperado: {type(x)}")


def amostras_via_datasets(n: int, passo: int):
    """Le a rota c com a biblioteca `datasets`, em streaming.

    Mesmo caminho que `genfocus_train/data.py` usa. Streaming para nao materializar
    o dataset inteiro nem tocar a cota do disco."""
    from datasets import load_dataset
    tok = os.environ.get("HF_TOKEN")
    ds = load_dataset(ROTA_C, split="train", streaming=True, token=tok)
    vistas = 0
    for i, r in enumerate(ds):
        if i % passo:
            continue
        yield r
        vistas += 1
        if vistas >= n:
            return


def _cinza_pil(im: "Image.Image") -> np.ndarray:
    im = im.convert("L")
    if max(im.size) > 512:
        e = 512 / max(im.size)
        im = im.resize((max(1, int(im.width * e)), max(1, int(im.height * e))), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def var_laplaciano(g: np.ndarray) -> float:
    """Variância do laplaciano 4-vizinhos. Medida de nitidez, menor = mais borrado."""
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--saida", default="t2_resultado.json")
    a = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        print("ERRO: exporte HF_TOKEN", file=sys.stderr); return 2

    passo = max(1, N_ROTA_C // max(1, a.n))
    reg = []
    for i, r in enumerate(amostras_via_datasets(a.n, passo)):
        if True:
            try:
                d = np.asarray(_pil(r["depth"]), dtype=np.float32)
                d01 = d / 65535.0 if d.max() > 1.5 else d
                lv_aif = var_laplaciano(_cinza_pil(_pil(r["aif"])))
                lv_bok = var_laplaciano(_cinza_pil(_pil(r["bokeh"])))
            except Exception as e:
                print(f"[t2] {r.get('stem')}: {e}"); continue
            k = float(r["k"]); s1 = float(r["s1"])
            reg.append({
                "stem": r["stem"], "k": k, "s1": s1,
                "calibration_ssim": (None if r.get("calibration_ssim") is None
                                     else float(r["calibration_ssim"])),
                "coc_max_px": k * float(np.abs(d01 - s1).max()),
                "blur_ratio": (lv_bok / lv_aif) if lv_aif > 0 else None,
                "censurada": k >= TETO_SWEEP,
            })
            if len(reg) % 25 == 0:
                print(f"[t2] acumulado {len(reg)}", flush=True)

    if not reg:
        print("ERRO: nenhuma amostra utilizável", file=sys.stderr); return 1
    _relatorio(reg, a)
    return 0


def _q(v, p):
    v = sorted(v); i = (len(v) - 1) * p / 100.0
    lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (i - lo)) + v[hi] * (i - lo)


def _relatorio(reg, a):
    cen = [x for x in reg if x["censurada"]]
    ins = [x for x in reg if not x["censurada"] and x["k"] > 0]
    print("\n" + "=" * 70)
    print(f"T2 -- teto do k na rota c, {len(reg)} amostras "
          f"({len(cen)} censuradas, {len(ins)} interiores)")
    print("=" * 70)

    for nome, grupo in (("CENSURADAS (k=300)", cen), ("INTERIORES (0<k<300)", ins)):
        if not grupo: continue
        br = [x["blur_ratio"] for x in grupo if x["blur_ratio"] is not None]
        cc = [x["coc_max_px"] for x in grupo]
        ss = [x["calibration_ssim"] for x in grupo if x["calibration_ssim"] is not None]
        print(f"\n  {nome}  n={len(grupo)}")
        if br: print(f"    blur_ratio (menor = alvo mais borrado)  p25={_q(br,25):.4f}  med={_q(br,50):.4f}  p75={_q(br,75):.4f}")
        if cc: print(f"    coc_max_px implicado                    p25={_q(cc,25):.1f}  med={_q(cc,50):.1f}  p75={_q(cc,75):.1f}")
        if ss: print(f"    calibration_ssim                        p25={_q(ss,25):.4f}  med={_q(ss,50):.4f}  p75={_q(ss,75):.4f}")

    print(f"\n  --- hipotese A, censura de faixa ---")
    if cen and ins:
        brc = [x["blur_ratio"] for x in cen if x["blur_ratio"] is not None]
        bri = [x["blur_ratio"] for x in ins if x["blur_ratio"] is not None]
        if brc and bri:
            razao = _q(bri, 50) / max(_q(brc, 50), 1e-9)
            print(f"    blur_ratio mediano interior / censurada = {razao:.2f}")
            print(f"    ({'ALVO DAS CENSURADAS E MAIS BORRADO -> compativel com A' if razao > 1.3 else 'alvos comparaveis -> A nao explica sozinha'})")

    print(f"\n  --- hipotese B, saturacao do kernel de {KERNEL_MAX_PX:.0f} px ---")
    acima = [x for x in reg if x["coc_max_px"] > KERNEL_MAX_PX and x["calibration_ssim"] is not None]
    abaixo = [x for x in reg if x["coc_max_px"] <= KERNEL_MAX_PX and x["calibration_ssim"] is not None]
    if acima and abaixo:
        sa = _q([x["calibration_ssim"] for x in acima], 50)
        sb = _q([x["calibration_ssim"] for x in abaixo], 50)
        print(f"    SSIM mediano com coc <= {KERNEL_MAX_PX:.0f}px: {sb:.4f}  (n={len(abaixo)})")
        print(f"    SSIM mediano com coc >  {KERNEL_MAX_PX:.0f}px: {sa:.4f}  (n={len(acima)})")
        print(f"    queda = {sb - sa:+.4f}  ({'JOELHO PRESENTE -> compativel com B' if sb - sa > 0.02 else 'sem joelho claro -> B nao explica sozinha'})")
    if cen:
        f = sum(1 for x in cen if x["coc_max_px"] > KERNEL_MAX_PX) / len(cen)
        print(f"    fracao das censuradas com coc > {KERNEL_MAX_PX:.0f}px: {100*f:.1f}%")

    json.dump({"n": len(reg), "n_censuradas": len(cen), "registros": reg},
              open(a.saida, "w"), indent=1)
    print(f"\n  gravado em {a.saida}")


if __name__ == "__main__":
    raise SystemExit(main())
