"""F0e -- une os escalares das rotas b e c numa tabela unica para o treino.

Aplica o veredito do teste T3, que foi decidido por regra declarada ANTES:

    razao fx_depthpro / fx_exif, n = 11.635 (rota b):
      mediana do erro relativo = 0,171   p95 = 0,526
      dentro de 5%: 16,4%   dentro de 20%: 57,0%
    regra: mediana < 5% e p95 < 15% -> Depth Pro nas duas.  NAO PASSOU.

Logo:
  ROTA B -> focal da EXIF, `f_mm * pixel_ratio` da tabela kfix. E a camera real.
  ROTA C -> focal do Depth Pro. Nao ha alternativa: a coluna `exif` e 100% NULA.

A heterogeneidade e uma limitacao declarada, e afeta so o canal de curvatura, que
e o de menor retorno esperado dos seis.

Os demais escalares (z_min_m, z_max_m_bruto, z_focus_m) vem do F0b nas DUAS
rotas, e isso e seguro: o `z_focus_m` do F0b reproduz o da tabela kfix com erro
relativo 0,0000 na mediana E no p90, em 11.624 amostras pareadas.

Uso:
    python3 f0e_merge_escalares.py --rota-b f0b_rota_b.jsonl \\
        --rota-c f0b_rota_c.jsonl --kfix kfix.parquet --saida f0b_todas.jsonl
"""

from __future__ import annotations

import argparse, json, os, struct, sys


def carregar_kfix(caminho: str) -> dict[str, float]:
    """fx da EXIF por stem: f_mm * pixel_ratio, ambos da tabela kfix."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                    "scripts", "audit_hf"))
    import cols, pq
    data = open(caminho, "rb").read()
    flen = struct.unpack("<I", data[-8:-4])[0]
    mdl = pq.rd_struct(pq.R(data[len(data) - 8 - flen: len(data) - 8]))
    alvo = {"stem", "f_mm", "pixel_ratio"}
    out = {n: [] for n in alvo}
    for rg in mdl[4]:
        for cc in rg[1]:
            cm = cc[3]
            nm = ".".join(x.decode() for x in cm[3])
            if nm in alvo:
                st = cm.get(11) or cm[9]
                out[nm].extend(cols.read_chunk(data[st:st + cm[7]], st, cm))
    dec = lambda x: x.decode() if isinstance(x, bytes) else x
    return {dec(s): float(f) * float(p)
            for s, f, p in zip(out["stem"], out["f_mm"], out["pixel_ratio"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rota-b", required=True)
    ap.add_argument("--rota-c", required=True)
    ap.add_argument("--kfix", required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    fx_exif = carregar_kfix(a.kfix)
    print(f"[f0e] fx da EXIF: {len(fx_exif)} stems")

    n_b = n_c = sem_exif = 0
    with open(a.saida, "w") as out:
        for caminho, rota in ((a.rota_b, "b"), (a.rota_c, "c")):
            for ln in open(caminho):
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                if r.get("z_focus_m") is None:
                    continue
                if rota == "b":
                    fx = fx_exif.get(r["stem"])
                    if fx is None or fx <= 0:
                        sem_exif += 1
                        continue                      # NUNCA cai em fallback
                    r["focallength_px"] = fx
                    r["fonte_focal"] = "exif"
                    n_b += 1
                else:
                    r["fonte_focal"] = "depth_pro"
                    n_c += 1
                out.write(json.dumps(r) + "\n")

    print(f"[f0e] rota b: {n_b}  (focal da EXIF, {sem_exif} descartadas sem EXIF)")
    print(f"[f0e] rota c: {n_c}  (focal do Depth Pro)")
    print(f"[f0e] total : {n_b + n_c}  ->  {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
