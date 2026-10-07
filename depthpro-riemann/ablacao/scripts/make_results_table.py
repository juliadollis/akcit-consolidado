#!/usr/bin/env python3
"""
scripts/make_results_table.py
=============================
Consolida os resultados de todas as variantes numa TABELA GERAL, com o zero-shot como
linha de referência explícita.

Lê os `comparacao_pareada.json` produzidos por `evaluate_paired.py` (um por variante) e
gera três saídas equivalentes:

    tabela_geral.md    para colar no report
    tabela_geral.csv   para planilha
    tabela_geral.txt   para o terminal

A tabela responde diretamente à pergunta "os nossos modelos estão ganhando?", porque
mostra, lado a lado e para cada métrica: o valor do zero-shot, o valor de cada campeã, a
diferença, e o veredito estatístico da comparação emparelhada por cena.

Uso:
    python scripts/make_results_table.py \\
        --entrada heads=/workspace/runs/champion_heads/aval_v2 \\
                  heads_final=/workspace/runs/champion_heads_final/aval_v2 \\
        --out-dir /workspace/relatorio
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Ordem de apresentação e sentido de cada métrica (True = maior é melhor)
METRICAS = [
    ("abs_rel", False, "AbsRel", "erro relativo de profundidade"),
    ("rmse", False, "RMSE", "erro quadrático médio"),
    ("d1", True, "δ1", "fração de pixels dentro de 1.25"),
    ("boundary_fmax", True, "bF-max", "melhor F de borda na varredura de limiar"),
    ("boundary_f_auc", True, "bF-AUC", "média do F ao longo da curva"),
    ("boundary_fscore", True, "bF@0.08", "F de borda no limiar fixo (legado)"),
    ("boundary_precision", True, "bPrec", "precisão de borda"),
    ("boundary_recall", True, "bRecall", "recall de borda"),
]


def carregar(caminho: Path):
    """Aceita a pasta da avaliação ou o JSON diretamente."""
    p = Path(caminho)
    if p.is_dir():
        p = p / "comparacao_pareada.json"
    if not p.exists():
        print(f"  [aviso] nao encontrado: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def veredito(par):
    """Texto curto do resultado estatístico, exigindo os dois testes concordando."""
    if par is None:
        return "—"
    if par.get("significativo"):
        return "SIG melhor" if par["delta_medio"] > 0 else "SIG pior"
    if not par.get("testes_concordam", True):
        return "inconclusivo (testes divergem)"
    return "não significativo"


def fmt(v, casas=4):
    try:
        return f"{float(v):.{casas}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", nargs="+", required=True,
                    metavar="NOME=CAMINHO",
                    help="uma ou mais variantes, ex.: heads=/runs/x/aval "
                         "heads_final=/runs/y/aval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--titulo", default="Resultados no conjunto de teste (held-out)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dados = {}
    for item in args.entrada:
        if "=" not in item:
            print(f"ERRO: formato invalido '{item}'. Use NOME=CAMINHO.")
            sys.exit(1)
        nome, caminho = item.split("=", 1)
        d = carregar(caminho)
        if d:
            dados[nome] = d
            print(f"  [ok] {nome}: {d['n_imagens']} imagens, {d['n_cenas']} cenas, "
                  f"{d['n_sementes']} sementes")
    if not dados:
        print("ERRO: nenhuma entrada valida.")
        sys.exit(1)

    qualquer = next(iter(dados.values()))
    cabecalho_ctx = (f"{qualquer['n_imagens']} imagens em {qualquer['n_cenas']} cenas "
                     f"disjuntas · {qualquer['n_sementes']} sementes por configuração · "
                     f"comparação emparelhada por {qualquer['nivel']}")

    # ---------------- Tabela 1: valores absolutos -------------------------
    linhas_abs = []
    for chave, maior_melhor, rotulo, desc in METRICAS:
        m0 = qualquer["metricas"].get(chave)
        if m0 is None:
            continue
        linha = {"Métrica": rotulo,
                 "Sentido": "maior melhor" if maior_melhor else "menor melhor",
                 "Zero-shot": fmt(m0["pareado"]["media_baseline"])}
        for nome, d in dados.items():
            mm = d["metricas"].get(chave)
            linha[nome] = fmt(mm["pareado"]["media_modelo"]) if mm else "—"
        linhas_abs.append(linha)

    # ---------------- Tabela 2: ganho e veredito --------------------------
    linhas_dif = []
    for chave, maior_melhor, rotulo, desc in METRICAS:
        for nome, d in dados.items():
            mm = d["metricas"].get(chave)
            if mm is None:
                continue
            par = mm["pareado"]
            lo, hi = par["delta_ic95_t"]
            linhas_dif.append({
                "Métrica": rotulo,
                "Config": nome,
                "Zero-shot": fmt(par["media_baseline"]),
                "Modelo": fmt(par["media_modelo"]),
                "Ganho": f"{par['delta_medio']:+.4f}",
                "IC95% do ganho": f"[{lo:+.4f}, {hi:+.4f}]",
                "p (t pareado)": f"{par['p_ttest_pareado']:.2g}",
                "p (Wilcoxon)": f"{par['p_wilcoxon']:.2g}",
                "Vitórias": f"{100*par['taxa_vitorias']:.0f}%",
                "Veredito": veredito(par),
            })

    # ---------------- Escrita ---------------------------------------------
    def tabela_md(linhas, colunas):
        if not linhas:
            return ""
        s = "| " + " | ".join(colunas) + " |\n"
        s += "|" + "|".join(["---"] * len(colunas)) + "|\n"
        for r in linhas:
            s += "| " + " | ".join(str(r.get(c, "")) for c in colunas) + " |\n"
        return s

    cols_abs = ["Métrica", "Sentido", "Zero-shot"] + list(dados)
    cols_dif = ["Métrica", "Config", "Zero-shot", "Modelo", "Ganho",
                "IC95% do ganho", "p (t pareado)", "p (Wilcoxon)", "Vitórias", "Veredito"]

    md = [f"# {args.titulo}", "", cabecalho_ctx, "",
          "## 1. Valores absolutos", "",
          "Cada coluna é a média por cena. O zero-shot é o DepthPro sem nenhum ajuste,",
          "medido no MESMO protocolo.", "",
          tabela_md(linhas_abs, cols_abs),
          "## 2. Ganho contra o zero-shot e significância", "",
          "Comparação emparelhada por cena. O ganho é sempre orientado de modo que",
          "positivo significa **modelo melhor que zero-shot**. O veredito exige que o",
          "teste t pareado e o Wilcoxon concordem.", "",
          tabela_md(linhas_dif, cols_dif),
          "## 3. Como ler", "",
          "- **bF-max** é a métrica de borda a reportar. O bF@0.08 usa um limiar fixo e",
          "  confunde qualidade de borda com agressividade de detecção: um modelo mais",
          "  conservador detecta menos borda e parece pior sem ter piorado.",
          "- Um ganho só conta quando os **dois** testes concordam. Quando divergem, em",
          "  geral indica amostra pequena ou distribuição assimétrica.",
          "- O IC entre sementes (no `resumo.txt` de cada avaliação) mede",
          "  reprodutibilidade do treino, e não o efeito contra o baseline.",
          ]
    (out / "tabela_geral.md").write_text("\n".join(md), encoding="utf-8")

    with open(out / "tabela_geral.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_dif)
        w.writeheader()
        w.writerows(linhas_dif)

    txt = [args.titulo, "=" * len(args.titulo), cabecalho_ctx, "",
           "VALORES ABSOLUTOS (media por cena)", "-" * 60]
    larg = max(len(c) for c in cols_abs) + 2
    txt.append("".join(c.ljust(larg) for c in cols_abs))
    for r in linhas_abs:
        txt.append("".join(str(r.get(c, "")).ljust(larg) for c in cols_abs))
    txt += ["", "GANHO CONTRA O ZERO-SHOT", "-" * 60]
    for r in linhas_dif:
        txt.append(f"{r['Métrica']:>10s} | {r['Config']:<12s} | "
                   f"{r['Modelo']} vs {r['Zero-shot']} | {r['Ganho']} "
                   f"{r['IC95% do ganho']} | {r['Veredito']}")
    texto = "\n".join(txt)
    (out / "tabela_geral.txt").write_text(texto, encoding="utf-8")

    print("\n" + texto)
    print(f"\n[saida] {out/'tabela_geral.md'}")
    print(f"[saida] {out/'tabela_geral.csv'}")
    print(f"[saida] {out/'tabela_geral.txt'}")


if __name__ == "__main__":
    main()
