#!/usr/bin/env python3
"""Consolida os resultados do reteste de curvatura num relatorio unico.

Le os JSONs que o train_single.py grava (summary/test_metrics/test_summary por
seed) e emite a tabela final em markdown, com media, IC95 e o teste pareado
entre B3 e o controle B0.

Existe para o numero do relatorio nunca ser transcrito a mao: roda de novo
quando outra seed fechar e a tabela se atualiza sozinha.

Uso:  python scripts/consolida_reteste.py resultados/reteste_curvatura_<data>
"""
import json, sys
from pathlib import Path
from statistics import mean, stdev

METRICAS = [
    ("boundary_fscore", "F-score de borda", "maior e melhor"),
    ("abs_rel",         "AbsRel",           "menor e melhor"),
    ("d1",              "delta1",           "maior e melhor"),
    ("rmse",            "RMSE",             "menor e melhor"),
]


def le_experimento(dir_exp: Path):
    """Devolve {seed: {metrica: valor}} + o que ainda nao fechou."""
    seeds = {}
    for d in sorted(dir_exp.glob("seed_*")):
        tm = d / "test_metrics.json"
        sm = d / "summary.json"
        if not tm.exists():
            seeds[d.name] = None          # seed comecou mas nao terminou o teste
            continue
        reg = json.loads(tm.read_text())
        if sm.exists():
            s = json.loads(sm.read_text())
            reg["_best_val"] = s.get("best_metric")
            reg["_best_epoch"] = s.get("best_epoch")
            reg["_epocas"] = s.get("history_len")
        seeds[d.name] = reg
    return seeds


def ic95(vals):
    """IC95 pela amplitude observada. Com n=3 nao ha t-student honesto; o
    train_single reporta min/max e reproduzimos isso, deixando claro no rotulo."""
    return (min(vals), max(vals)) if len(vals) > 1 else (vals[0], vals[0])


def main():
    raiz = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    exps = {d.name: le_experimento(d) for d in sorted(raiz.iterdir())
            if d.is_dir() and d.name != "logs"}
    if not exps:
        print(f"nenhum experimento em {raiz}"); return 1

    print(f"# Reteste da curvatura — resultados\n")
    print(f"Fonte: `{raiz}` (JSONs gravados pelo `train_single.py`).\n")

    # estado
    print("## Estado das rodadas\n")
    print("| experimento | seeds concluidas | seeds em andamento |")
    print("|---|---|---|")
    for nome, seeds in exps.items():
        ok = [k for k, v in seeds.items() if v]
        pend = [k for k, v in seeds.items() if not v]
        print(f"| `{nome}` | {len(ok)} | {', '.join(pend) if pend else '—'} |")
    print()

    completos = {n: {k: v for k, v in s.items() if v} for n, s in exps.items()}
    completos = {n: s for n, s in completos.items() if s}

    for chave, rotulo, sentido in METRICAS:
        print(f"## {rotulo} ({sentido})\n")
        print("| experimento | n | media | min | max | por seed |")
        print("|---|---|---|---|---|---|")
        for nome, seeds in completos.items():
            vals = [v[chave] for v in seeds.values() if chave in v]
            if not vals:
                continue
            lo, hi = ic95(vals)
            porseed = ", ".join(f"{x:.4f}" for x in vals)
            print(f"| `{nome}` | {len(vals)} | **{mean(vals):.4f}** | {lo:.4f} | {hi:.4f} | {porseed} |")
        print()

    # comparacao direta, so quando os dois lados tem o mesmo numero de seeds.
    # Existe mais de um braco B3 (um por teto), entao cada um e comparado ao
    # mesmo controle B0: o teto e a unica variavel que muda entre eles.
    b3s = [n for n in completos if n.startswith("B3")]
    b0 = next((n for n in completos if n.startswith("B0")), None)
    for b3 in b3s if b0 else []:
        print(f"## `{b3}` contra o controle `{b0}`\n")
        n3, n0 = len(completos[b3]), len(completos[b0])
        if n3 != n0:
            print(f"> **Comparacao ainda NAO e valida:** {b3} tem {n3} seed(s) e "
                  f"{b0} tem {n0}. Os numeros abaixo sao provisorios.\n")
        print("| metrica | B3 | B0 (controle) | diferenca | quem vence |")
        print("|---|---|---|---|---|")
        for chave, rotulo, sentido in METRICAS:
            v3 = [v[chave] for v in completos[b3].values() if chave in v]
            v0 = [v[chave] for v in completos[b0].values() if chave in v]
            if not v3 or not v0:
                continue
            m3, m0 = mean(v3), mean(v0)
            maior_melhor = sentido.startswith("maior")
            vence = "B3" if ((m3 > m0) == maior_melhor) else "B0"
            print(f"| {rotulo} | {m3:.4f} | {m0:.4f} | {m3-m0:+.4f} | **{vence}** |")
        print()
        if n3 > 1 and n3 == n0:
            v3 = [v["boundary_fscore"] for v in completos[b3].values()]
            v0 = [v["boundary_fscore"] for v in completos[b0].values()]
            d = [a - b for a, b in zip(sorted(v3), sorted(v0))]
            print(f"Diferenca no F-score de borda: media {mean(d):+.4f}, "
                  f"desvio {stdev(d):.4f} (n={len(d)}).\n")
            print("> Com 3 seeds nao ha poder estatistico para um teste pareado "
                  "conclusivo. O que a tabela sustenta e a direcao, nao a significancia.\n")

    # os dois tetos entre si: mesma perda, mesma configuracao, so o clamp muda
    if len(b3s) == 2:
        a, b = sorted(b3s)
        print(f"## Efeito do teto: `{a}` contra `{b}`\n")
        print(f"| metrica | {a} | {b} | diferenca | quem vence |")
        print("|---|---|---|---|---|")
        for chave, rotulo, sentido in METRICAS:
            va = [v[chave] for v in completos[a].values() if chave in v]
            vb = [v[chave] for v in completos[b].values() if chave in v]
            if not va or not vb:
                continue
            ma, mb = mean(va), mean(vb)
            maior_melhor = sentido.startswith("maior")
            vence = a if ((ma > mb) == maior_melhor) else b
            print(f"| {rotulo} | {ma:.4f} | {mb:.4f} | {mb-ma:+.4f} | **{vence}** |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
