"""T1 -- qual plano de foco implementa a Eq. 4 do paper: `s1` ou `z_focus_m`.

Eq. 4 do GenRefocus, literal:

    D_focus = median( D[M] )

com M a máscara em foco obtida pelo BiRefNet. O df da rota b traz `depth`,
`foreground_mask` e `s1`; a tabela kfix traz `z_focus_m`, `z_min_m`, `z_max_m`.
Se os dois planos de foco fossem o mesmo número em unidades diferentes, valeria

    s1  ==  (z_focus_m - z_min_m) / (z_max_m - z_min_m)

e eles NAO batem: mediana 0,0031, p90 0,079, máximo 0,49, com só 18,4% a 1e-4.
Este job mede a Eq. 4 diretamente nos pixels e diz qual dos dois a implementa.

O teste é legítimo porque a coluna `depth` e o mapa que o job da kfix usou são o
MESMO campo: `z_min_m` e `z_max_m` reconstroem o `coc_p99_px` gravado a partir
dos pixels de `depth` com erro mediano de 0,011% (AUDITORIA seção 3).

Uso:
    python3 t1_plano_de_foco.py --n 200 --saida t1_resultado.json
"""

from __future__ import annotations

import argparse, io, json, os, subprocess, sys, urllib.parse

import numpy as np
from PIL import Image

ROTA_B = "AKCITPixel3/BKXcuVXCmeRvN"
DS_ROWS = "https://datasets-server.huggingface.co/rows"


def _curl(url: str, binario: bool = False):
    tok = os.environ.get("HF_TOKEN", "")
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "90", "-H", f"Authorization: Bearer {tok}", url],
        capture_output=True,
    )
    return r.stdout if binario else json.loads(r.stdout)


def linhas(dataset: str, offset: int, length: int):
    q = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": "train",
         "offset": offset, "length": length}
    )
    d = _curl(f"{DS_ROWS}?{q}")
    if "rows" not in d:
        raise RuntimeError(f"datasets-server devolveu: {str(d)[:300]}")
    return [r["row"] for r in d["rows"]]


def _abre(src: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(_curl(src, binario=True))))


def mediana_na_mascara(depth01: np.ndarray, mask: np.ndarray, limiar: float) -> float | None:
    """Eq. 4: mediana da profundidade DENTRO da máscara em foco."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = mask.astype(np.float32)
    m = m / 255.0 if m.max() > 1.5 else m
    sel = m > limiar
    if sel.sum() < 64:
        return None
    return float(np.median(depth01[sel]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--limiar-mascara", type=float, default=0.5)
    ap.add_argument("--kfix-parquet", default=None,
                    help="parquet local da tabela kfix (colunas por stem)")
    ap.add_argument("--saida", default="t1_resultado.json")
    a = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("ERRO: exporte HF_TOKEN", file=sys.stderr); return 2

    kfix = _carrega_kfix(a.kfix_parquet)
    print(f"[t1] kfix: {len(kfix)} stems")

    # amostragem espalhada, não os N primeiros (que são o topo ordenado)
    passo = max(1, 11635 // max(1, a.n // 20))
    offsets = list(range(0, 11635 - 20, passo))[: max(1, a.n // 20)]

    reg = []
    for off in offsets:
        try:
            rows = linhas(ROTA_B, off, 20)
        except Exception as e:
            print(f"[t1] offset {off}: {e}"); continue
        for r in rows:
            stem = r["stem"]
            if stem not in kfix:
                continue
            try:
                d = _abre(r["depth"]["src"]).astype(np.float32)
                mk = _abre(r["foreground_mask"]["src"])
            except Exception as e:
                print(f"[t1] {stem}: {e}"); continue
            d01 = d / 65535.0 if d.max() > 1.5 else d
            m = mediana_na_mascara(d01, mk, a.limiar_mascara)
            if m is None:
                continue
            e = kfix[stem]
            faixa = e["z_max_m"] - e["z_min_m"]
            if faixa <= 0:
                continue
            b = (e["z_focus_m"] - e["z_min_m"]) / faixa
            reg.append({
                "stem": stem, "eq4_mediana_na_mascara": m,
                "s1": float(r["s1"]), "z_focus_normalizado": b,
                "erro_s1": abs(m - float(r["s1"])), "erro_zfocus": abs(m - b),
                "z_max_saturado": bool(e["z_max_m"] >= 9999.0),
            })
        print(f"[t1] offset {off}: acumulado {len(reg)}", flush=True)
        if len(reg) >= a.n:
            break

    if not reg:
        print("ERRO: nenhuma amostra utilizável", file=sys.stderr); return 1
    _relatorio(reg, a)
    return 0


def _carrega_kfix(caminho: str | None) -> dict:
    if caminho and os.path.exists(caminho):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "audit_hf"))
        import struct as _s
        import pq, cols
        data = open(caminho, "rb").read()
        flen = _s.unpack("<I", data[-8:-4])[0]
        mdl = pq.rd_struct(pq.R(data[len(data) - 8 - flen: len(data) - 8]))
        alvo = {"stem", "z_focus_m", "z_min_m", "z_max_m"}
        out = {k: [] for k in alvo}
        for rg in mdl[4]:
            for cc in rg[1]:
                cm = cc[3]; nm = ".".join(x.decode() for x in cm[3])
                if nm in alvo:
                    st = cm.get(11) or cm[9]
                    out[nm].extend(cols.read_chunk(data[st:st + cm[7]], st, cm))
        dec = lambda x: x.decode() if isinstance(x, bytes) else x
        return {dec(s): {"z_focus_m": f, "z_min_m": mi, "z_max_m": ma}
                for s, f, mi, ma in zip(out["stem"], out["z_focus_m"],
                                        out["z_min_m"], out["z_max_m"])}
    raise SystemExit("passe --kfix-parquet com o parquet local da tabela kfix")


def _relatorio(reg: list[dict], a) -> None:
    es1 = np.array([x["erro_s1"] for x in reg])
    ezf = np.array([x["erro_zfocus"] for x in reg])
    vence_s1 = int((es1 < ezf).sum())
    print("\n" + "=" * 66)
    print(f"T1 -- Eq. 4 medida em {len(reg)} amostras da rota b "
          f"(limiar de máscara {a.limiar_mascara})")
    print("=" * 66)
    for nome, e in (("s1", es1), ("z_focus_m normalizado", ezf)):
        print(f"  |Eq4 - {nome:22s}|  mediana={np.median(e):.5f}  "
              f"p90={np.percentile(e,90):.5f}  media={e.mean():.5f}")
    print(f"\n  s1 mais perto da Eq. 4 em      {vence_s1}/{len(reg)} "
          f"({100*vence_s1/len(reg):.1f}%)")
    print(f"  z_focus_m mais perto da Eq. 4 em {len(reg)-vence_s1}/{len(reg)} "
          f"({100*(len(reg)-vence_s1)/len(reg):.1f}%)")
    nao_sat = [x for x in reg if not x["z_max_saturado"]]
    if nao_sat:
        v = sum(1 for x in nao_sat if x["erro_s1"] < x["erro_zfocus"])
        print(f"\n  so em z_max NAO saturado (n={len(nao_sat)}): "
              f"s1 vence em {v} ({100*v/len(nao_sat):.1f}%)")
    veredito = "s1" if vence_s1 > len(reg) / 2 else "z_focus_m"
    print(f"\n  VEREDITO: {veredito} implementa a Eq. 4")
    json.dump({"n": len(reg), "veredito": veredito,
               "vence_s1": vence_s1, "registros": reg},
              open(a.saida, "w"), indent=1)
    print(f"  gravado em {a.saida}")


if __name__ == "__main__":
    raise SystemExit(main())
