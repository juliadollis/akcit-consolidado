#!/usr/bin/env python3
"""Transforma o piloto em limiares congelados.

    python3 scripts/calibrate_thresholds.py --pilot-dir output/c_pilot \\
        --output-json output/thresholds_v1.json

Esta é a etapa que faltava no pipeline antigo, e a ausência dela custou caro: o
`--k-max 300` foi escolhido a priori e produziu `k == 300` exato em 1.379 de 2.932
amostras — 47% de rótulo censurado que ninguém viu porque nada olhava a distribuição.

## O que a ferramenta faz, e o que ela NÃO faz

Ela **lê** as distribuições que o piloto mediu e **propõe** cortes por percentil,
mostrando quanto cada corte custaria em amostras. Ela **não decide sozinha**: o
relatório é para um humano ler, e o limiar entra no comando do run final.

Para o SSIM da Eq. 5 — o único limiar que o paper afirma existir e não publica — a
proposta por percentil é um ponto de partida ruim sozinha. O certo é revisar um painel
de 300 a 500 casos e calibrar por precisão: `--reviewed-csv` faz isso, e é o caminho
recomendado. O percentil serve para os outros gates e para dar ordem de grandeza.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(int(round(q / 100.0 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def _load_pilot(pilot_dir: Path) -> tuple[list[dict], Counter]:
    """Metadados aceitos + histograma de motivos de rejeição."""
    metas = []
    for path in sorted((pilot_dir / "meta").glob("*.json")):
        metas.append(json.loads(path.read_text(encoding="utf-8")))

    reasons: Counter = Counter()
    rejections = pilot_dir / "rejections.jsonl"
    if rejections.exists():
        for line in rejections.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "rejected":
                reasons[row.get("reason", "?")] += 1
    return metas, reasons


#: Gate -> (chave em `quality`, lado do corte, percentis propostos).
#: `lower` = queremos um piso; `upper` = queremos um teto.
_GATES = [
    ("min_calibration_ssim", "calibration_ssim", "lower", (5, 10, 25)),
    ("min_focus_mask_sharpness_ratio", "focus_mask_sharpness_ratio", "lower", (5, 10, 25)),
    ("min_mask_area_ratio", "mask_area_ratio_min", "lower", (1, 5)),
    ("max_mask_area_ratio", "mask_area_ratio_max", "upper", (95, 99)),
    ("max_mask_border_coverage", "mask_border_coverage", "upper", (95, 99)),
    ("min_aif_laplacian_variance", "aif_laplacian_variance", "lower", (5, 10)),
    ("max_bokeh_over_aif_sharpness", "bokeh_over_aif_sharpness", "upper", (95, 99)),
    ("min_depth_useful_levels", "depth_useful_levels", "lower", (5, 10)),
    ("min_mask_iou", "mask_iou_aif_bokeh", "lower", (5, 10)),
    # O limiar que julga a REGIÃO REFINADA, e o único que a julga com a grandeza certa:
    # razão bokeh/AIF normalizada pela textura da cena. `min_focus_mask_sharpness_ratio`
    # mede nitidez ABSOLUTA e reprovaria uma parede lisa legitimamente em foco.
    ("min_focus_region_retention", "focus_region_retention", "lower", (5, 10, 25)),
]


def _distributions(metas: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for meta in metas:
        for name, gate in (meta.get("quality") or {}).items():
            value = gate.get("value")
            if value is None or value != value:          # descarta NaN
                continue
            out.setdefault(name, []).append(float(value))
    return out


def _propose(dists: dict[str, list[float]], total: int) -> list[dict]:
    proposals = []
    for flag, key, side, percentis in _GATES:
        values = dists.get(key, [])
        if not values:
            proposals.append({"flag": flag, "metric": key, "n": 0,
                              "note": "não medido no piloto"})
            continue
        opcoes = []
        for q in percentis:
            corte = _percentile(values, q if side == "lower" else q)
            perdidos = (sum(1 for v in values if v < corte) if side == "lower"
                        else sum(1 for v in values if v > corte))
            opcoes.append({"percentil": q, "valor": round(corte, 6),
                           "descarta": perdidos,
                           "descarta_pct": round(100 * perdidos / len(values), 2)})
        proposals.append({
            "flag": flag, "metric": key, "side": side, "n": len(values),
            "p05": round(_percentile(values, 5), 6),
            "mediana": round(median(values), 6),
            "p95": round(_percentile(values, 95), 6),
            "opcoes": opcoes,
        })
    return proposals


def _from_reviewed(path: Path, min_precision: float) -> dict:
    """Limiar de SSIM a partir de um painel revisado — o caminho recomendado.

    CSV com `calibration_ssim` e `acceptable_for_control` (1/0, true/false, sim/não).

    Escolhe o MENOR limiar tal que ele **e todos acima dele** atingem a precisão
    pedida. Pegar simplesmente o menor elegível pega ruído: precisão não é monótona
    no limiar, e um corte baixo pode atingir a meta por acaso.
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                ssim = float(row["calibration_ssim"])
            except (KeyError, ValueError):
                continue
            ok = str(row.get("acceptable_for_control", "")).strip().lower()
            rows.append((ssim, ok in {"1", "true", "yes", "y", "sim", "s"}))
    if not rows or not any(ok for _, ok in rows):
        raise ValueError("painel precisa de linhas e de ao menos um aceito")

    positivos = sum(ok for _, ok in rows)
    candidatos = sorted({s for s, _ in rows})
    metricas = []
    for corte in candidatos:
        mantidos = [ok for s, ok in rows if s >= corte]
        if not mantidos:
            continue
        tp = sum(mantidos)
        metricas.append({"threshold": corte, "precision": tp / len(mantidos),
                         "recall": tp / positivos, "retained": len(mantidos)})

    # monotonicidade a partir do topo: o corte só vale se ele e todos acima passam
    escolhido = None
    for i, m in enumerate(metricas):
        if all(x["precision"] >= min_precision for x in metricas[i:]):
            escolhido = m
            break
    if escolhido is None:
        raise ValueError(
            "nenhum limiar atinge a precisão pedida de forma estável. Revise renderer "
            "e máscaras em vez de baixar a régua.")
    return {"method": "menor_limiar_estavel_acima_da_precisao_minima",
            "min_precision": min_precision, "reviewed_rows": len(rows),
            "acceptable_rows": positivos, "chosen": escolhido}


def main() -> int:
    p = argparse.ArgumentParser(description="Congela limiares a partir do piloto.")
    p.add_argument("--pilot-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--reviewed-csv", default="",
                   help="painel humano para o limiar de SSIM — caminho recomendado")
    p.add_argument("--min-precision", type=float, default=0.95)
    args = p.parse_args()

    pilot = Path(args.pilot_dir)
    metas, reasons = _load_pilot(pilot)
    total = len(metas) + sum(reasons.values())

    print(f"\n{'=' * 66}")
    print(f"  piloto: {pilot}")
    print(f"  aceitas    : {len(metas)}")
    print(f"  rejeitadas : {sum(reasons.values())}   de {total} processadas")
    if reasons:
        print("-" * 66)
        print("  motivos de rejeição:")
        largura = max(len(r) for r in reasons)
        for motivo, n in reasons.most_common():
            print(f"    {motivo:<{largura}}  {n:>5}  ({100 * n / total:5.1f}%)")
    print("=" * 66)

    if not metas:
        print("\nNenhuma amostra aceita: não há distribuição para calibrar. "
              "Olhe o histograma acima antes de mexer em limiar.")
        return 1

    dists = _distributions(metas)
    proposals = _propose(dists, len(metas))

    print("\n  distribuições medidas e cortes propostos\n")
    for prop in proposals:
        if not prop.get("n"):
            print(f"  {prop['flag']:<34} {prop['note']}")
            continue
        print(f"  {prop['flag']:<34} p05 {prop['p05']:>10.4f} · "
              f"mediana {prop['mediana']:>10.4f} · p95 {prop['p95']:>10.4f}")
        for o in prop["opcoes"]:
            print(f"      p{o['percentil']:<3} = {o['valor']:>10.4f}   "
                  f"descartaria {o['descarta']:>4} ({o['descarta_pct']:>5.2f}%)")

    k = [m["k_value"] for m in metas]
    censuradas = sum(1 for m in metas if m["is_k_censored"])
    print(f"\n  k_value   p05 {_percentile(k, 5):.2f} · mediana {median(k):.2f} · "
          f"p95 {_percentile(k, 95):.2f}")
    print("            sweep da Eq. 5 — borrão INCREMENTAL sobre a AIF")

    # A âncora analítica (Eq. 3) mede o borrão ABSOLUTO do alvo. Na RealBokeh a "AIF"
    # é f/22, não all-in-focus, então ela já carrega parte do borrão e o sweep só
    # procura o que falta. `k_value < k_analytic` é o ESPERADO, não sintoma de defeito.
    # Imprimir os dois sem essa ressalva já levaria alguém a "consertar" o que está certo.
    analiticos = [m["k_analytic"] for m in metas if m.get("k_analytic") is not None]
    if analiticos:
        razoes = sorted(m["k_value"] / m["k_analytic"] for m in metas
                        if m.get("k_analytic"))
        print(f"  k_analytic p05 {_percentile(analiticos, 5):.2f} · "
              f"mediana {median(analiticos):.2f} · "
              f"p95 {_percentile(analiticos, 95):.2f}")
        print("            Eq. 3 — borrão ABSOLUTO; grandeza DIFERENTE de k_value")
        print(f"  razão k_value/k_analytic  p05 {_percentile(razoes, 5):.3f} · "
              f"mediana {median(razoes):.3f} · p95 {_percentile(razoes, 95):.3f}")
        print("            esperada < 1. Modelos de composição de borrão dão a faixa")
        print("            1 − F/22 (linear) a √(1 − (F/22)²) (quadrática), ambos [A].")
        print("            Razão > 1 em massa é que seria sintoma de defeito real.")
    print(f"  censuradas: {censuradas} de {len(metas)} "
          f"({100 * censuradas / len(metas):.1f}%)")
    print(f"            no dataset antigo eram 47,0% — se este número estiver perto "
          f"disso,\n            o problema é o renderer ou a faixa, não o limiar")

    # De onde saiu a região em foco de cada amostra aceita. Aparece aqui porque um
    # limiar calibrado sem saber a composição do lote é um limiar cego: uma distribuição
    # de `focus_region_retention` medida em 80% de amostras `retention_only` não é a
    # mesma coisa que a de um lote em que o BiRefNet acertou.
    fontes = Counter(m.get("focus_source", "AUSENTE") for m in metas)
    refinadas = sum(n for f, n in fontes.items() if f not in ("birefnet", "AUSENTE"))
    print("\n  região em foco (§3.2(c)):")
    for fonte, n in fontes.most_common():
        print(f"    {fonte:<18} {n:>5}  ({100 * n / len(metas):5.1f}%)")
    print(f"    refinadas          {refinadas:>5}  "
          f"({100 * refinadas / len(metas):5.1f}%)  — treináveis com e sem, por "
          f"`focus_was_refined`")

    report = {
        "pilot_dir": str(pilot), "accepted": len(metas),
        "rejected": sum(reasons.values()), "rejection_reasons": dict(reasons),
        "focus_sources": dict(fontes),
        "k_value": {"p05": _percentile(k, 5), "median": median(k),
                    "p95": _percentile(k, 95), "censored": censuradas,
                    "meaning": "sweep Eq.5 — borrao INCREMENTAL sobre a AIF"},
        "proposals": proposals,
    }
    if args.reviewed_csv:
        report["ssim_from_review"] = _from_reviewed(Path(args.reviewed_csv),
                                                    args.min_precision)
        escolhido = report["ssim_from_review"]["chosen"]
        print(f"\n  limiar de SSIM do painel revisado: {escolhido['threshold']:.4f}  "
              f"(precisão {escolhido['precision']:.3f}, "
              f"recall {escolhido['recall']:.3f})")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  relatório: {out}")
    print("\n  Nenhum limiar foi aplicado. Escolha os valores, passe-os no comando do "
          "run final,\n  e registre a escolha em reference/ACHADOS.md.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
