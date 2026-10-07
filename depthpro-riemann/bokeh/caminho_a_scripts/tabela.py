"""Le as condicoes do Caminho A e monta a tabela lado a lado.

Le SSIM e K do manifest.jsonl (superficie de varredura rapida) e PSNR de
meta/<id>.json -> provenance.extra.k_search.calibration_psnr, que e onde ele foi
gravado sem mexer no schema do manifesto.

Tudo que compara bracos e PAREADO por imagem. A mesma foto, tres profundidades: a
variacao entre imagens do RealDOF (SSIM de 0,84 a 0,96) e MUITO maior que a diferenca
entre bracos, entao uma comparacao de medias com desvio entre imagens afogaria o efeito
que se quer medir. O desvio que informa e o do DELTA pareado.
"""
import json, os, sys, math
from pathlib import Path
import numpy as np

RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/raid/user_juliadollis/julia_docker/caminho_a_rotac/out/full50")

ORDEM = ["base_busca", "Bgrad_al_busca", "Bgeod_al_busca",
         "Bgrad_cru_busca", "Bgeod_cru_busca",
         "base_kfixo", "Bgrad_al_kfixo", "Bgeod_al_kfixo"]

ROTULO = {
    "base_busca":      ("DepthPro base",      "busca (Eq. 5)"),
    "Bgrad_al_busca":  ("Bgrad s0 mesclado",  "busca (Eq. 5)"),
    "Bgeod_al_busca":  ("Bgeod s0 mesclado",  "busca (Eq. 5)"),
    "Bgrad_cru_busca": ("Bgrad s0 SEM alinhar", "busca (Eq. 5)"),
    "Bgeod_cru_busca": ("Bgeod s0 SEM alinhar", "busca (Eq. 5)"),
    "base_kfixo":      ("DepthPro base",      "K fixo"),
    "Bgrad_al_kfixo":  ("Bgrad s0 mesclado",  "K fixo"),
    "Bgeod_al_kfixo":  ("Bgeod s0 mesclado",  "K fixo"),
}


def le(cond: Path) -> dict:
    """{sample_id: {ssim, psnr, k, censurado, alinhado}}"""
    man = cond / "manifest.jsonl"
    if not man.is_file():
        return {}
    saida = {}
    for linha in man.open(encoding="utf-8"):
        d = json.loads(linha)
        sid = d["sample_id"]
        reg = {"ssim": float(d["calibration_ssim"]), "k": float(d["k_value"]),
               "censurado": bool(d["is_k_censored"]), "psnr": None,
               "alinhado": None, "k_mode": None}
        meta = cond / "meta" / f"{sid}.json"
        if meta.is_file():
            m = json.loads(meta.read_text(encoding="utf-8"))
            extra = (m.get("provenance") or {}).get("extra") or {}
            ks = extra.get("k_search") or {}
            reg["psnr"] = ks.get("calibration_psnr")
            reg["k_mode"] = ks.get("k_mode")
            al = extra.get("depth_alignment") or {}
            reg["alinhado"] = al.get("aplicado")
        saida[sid] = reg
    return saida


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, float(np.median(v)), v.size


dados = {c: le(RAIZ / c) for c in ORDEM}
dados = {c: d for c, d in dados.items() if d}
if not dados:
    sys.exit(f"nenhuma condicao com manifesto em {RAIZ}")

print(f"# Caminho A — rota C sobre o RealDOF  ({RAIZ})\n")
print("## Tabela lado a lado\n")
cab = ("| fonte de profundidade | calibração | n | SSIM média±dp | SSIM mediana | "
       "PSNR média±dp (dB) | K mediana | K p05–p95 | censuradas | alinh. afim |")
print(cab)
print("|" + "---|" * 10 + "---|")
for c in ORDEM:
    if c not in dados:
        continue
    d = dados[c]
    fonte, calib = ROTULO[c]
    s_m, s_d, s_md, n = ms([v["ssim"] for v in d.values()])
    p_m, p_d, _, np_ = ms([v["psnr"] for v in d.values()])
    ks = np.array([v["k"] for v in d.values()])
    cens = sum(1 for v in d.values() if v["censurado"])
    al = {v["alinhado"] for v in d.values()}
    al_txt = ("aplicado" if al == {True} else
              "identidade (é a referência)" if al == {False} else
              "não solicitado" if al == {None} else f"MISTO {al}")
    if all(v["alinhado"] is False for v in d.values()) and "cru" in c:
        al_txt = "NÃO (controle)"
    print(f"| {fonte} | {calib} | {n} | {s_m:.4f} ± {s_d:.4f} | {s_md:.4f} | "
          f"{p_m:.2f} ± {p_d:.2f} | {np.median(ks):.1f} | "
          f"{np.percentile(ks,5):.1f}–{np.percentile(ks,95):.1f} | {cens} | {al_txt} |")

print("\n## Deltas PAREADOS por imagem (braço − base, mesma foto)\n")
print("| comparação | n | ΔSSIM média±dp | ΔSSIM mediana | vence em | ΔPSNR média±dp (dB) | t pareado |")
print("|---|---|---|---|---|---|---|")


def pareado(a: str, b: str, titulo: str):
    if a not in dados or b not in dados:
        return
    comuns = sorted(set(dados[a]) & set(dados[b]))
    if not comuns:
        return
    ds = np.array([dados[a][s]["ssim"] - dados[b][s]["ssim"] for s in comuns])
    pp = [(dados[a][s]["psnr"], dados[b][s]["psnr"]) for s in comuns]
    dp = np.array([x - y for x, y in pp if x is not None and y is not None
                   and np.isfinite(x) and np.isfinite(y)])
    ganha = int((ds > 0).sum())
    sd = float(ds.std(ddof=1)) if ds.size > 1 else 0.0
    t = (float(ds.mean()) / (sd / math.sqrt(ds.size))) if sd > 0 else float("nan")
    dpm = f"{dp.mean():+.3f} ± {dp.std(ddof=1):.3f}" if dp.size > 1 else "—"
    print(f"| {titulo} | {ds.size} | {ds.mean():+.5f} ± {sd:.5f} | "
          f"{np.median(ds):+.5f} | {ganha}/{ds.size} | {dpm} | {t:+.2f} |")


pareado("Bgrad_al_busca", "base_busca", "Bgrad alinhado − base · busca")
pareado("Bgeod_al_busca", "base_busca", "Bgeod alinhado − base · busca")
pareado("Bgrad_al_kfixo", "base_kfixo", "Bgrad alinhado − base · K fixo")
pareado("Bgeod_al_kfixo", "base_kfixo", "Bgeod alinhado − base · K fixo")
pareado("Bgrad_al_busca", "Bgrad_cru_busca", "Bgrad alinhado − Bgrad CRU · busca")
pareado("Bgeod_al_busca", "Bgeod_cru_busca", "Bgeod alinhado − Bgeod CRU · busca")

print("\n## O que o alinhamento fez com K (razão pareada de K, mesma imagem)\n")
for a, b, titulo in (("Bgrad_al_busca", "Bgrad_cru_busca", "Bgrad"),
                     ("Bgeod_al_busca", "Bgeod_cru_busca", "Bgeod")):
    if a in dados and b in dados:
        comuns = sorted(set(dados[a]) & set(dados[b]))
        r = np.array([dados[a][s]["k"] / dados[b][s]["k"] for s in comuns])
        print(f"- {titulo}: K_alinhado / K_cru = mediana {np.median(r):.4f} "
              f"[{r.min():.4f} , {r.max():.4f}]  (n={r.size})")

for c in ORDEM:
    if c in dados:
        alj = RAIZ / c / "depth_alignment.jsonl"
        if alj.is_file():
            aj = [json.loads(l) for l in alj.open(encoding="utf-8")]
            a = np.array([x["a"] for x in aj]); b = np.array([x["b"] for x in aj])
            r2 = np.array([x["r2"] for x in aj])
            piso = np.array([x["fracao_no_piso"] for x in aj])
            print(f"\n### ajuste afim de {c}  (n={len(aj)}, domínio "
                  f"{aj[0]['dominio']})")
            print(f"- a (escala) : mediana {np.median(a):.4f}  "
                  f"[{a.min():.4f} , {a.max():.4f}]")
            print(f"- b (desloc.): mediana {np.median(b):+.6f} 1/m  "
                  f"[{b.min():+.6f} , {b.max():+.6f}]")
            print(f"- R²         : mediana {np.median(r2):.4f}  "
                  f"[{r2.min():.4f} , {r2.max():.4f}]  "
                  "— quanto a relação entre os braços é mesmo afim")
            print(f"- fração de pixels presos no piso positivo: máx "
                  f"{piso.max():.6f}")

print("\n## K escolhido pela busca, por condição\n")
for c in ORDEM:
    if c in dados and "busca" in c:
        ks = np.array([v["k"] for v in dados[c].values()])
        print(f"- {c:18s} mediana {np.median(ks):8.2f}  média {ks.mean():8.2f}  "
              f"p05 {np.percentile(ks,5):8.2f}  p95 {np.percentile(ks,95):8.2f}  "
              f"min {ks.min():7.2f}  max {ks.max():7.2f}")
