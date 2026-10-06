#!/usr/bin/env python3
"""
tabela_estilo_paper.py
======================
Monta a tabela final de bokeh no formato da Tab. 3 do GenRefocus, a partir das
metricas POR IMAGEM produzidas por `metricas_por_imagem.py`.

O QUE ESTA TABELA TEM E A ANTIGA NAO TINHA
------------------------------------------
1. **Intervalo de confianca.** A curada so tinha ponto. Sem dispersao nao da
   para dizer se a diferenca entre dois modelos esta fora do ruido.
2. **Comparacao PAREADA.** Os modelos rodam nas MESMAS cenas, entao comparar as
   medias independentes joga fora quase toda a potencia. O pareado por
   `image_id` e o teste certo aqui, e e bem mais sensivel.
3. **LVCorr na convencao do paper.** O nosso Pearson cru e negativo quando o
   modelo OBEDECE o K (mais K -> mais borrado -> menor variancia do Laplaciano).
   O paper reporta "quanto maior melhor" com +0,9368. Sem inverter o sinal, a
   nossa tabela faz o FLUX sem treino (+0,977) parecer o modelo mais controlavel
   de todos, e os nossos bons modelos (-0,82) os piores. Aqui vai invertido, com
   a coluna crua preservada ao lado.

O QUE ELA CONTINUA NAO PODENDO DIZER
------------------------------------
Que os valores absolutos sejam comparaveis aos publicados. O LF-Bokeh deles nao
foi liberado (o roadmap do repo oficial lista "Release Benchmark data" como
tarefa futura), entao a nossa mesa e uma reconstrucao do protocolo, nao a fonte.
Rodando os pesos OFICIAIS na nossa mesa da LPIPS 0,2047 contra 0,0833 publicado.
As comparacoes RELATIVAS entre modelos, todas medidas no mesmo dado e no mesmo
pipeline, seguem validas -- e sao elas que a tabela sustenta.

Uso:
    python3 tabela_estilo_paper.py --entrada /host/por_imagem_gpu*/por_imagem.parquet \
        --saida-md /host/TABELA_FINAL.md --saida-hf juliadollis/genrefocus-tabela-final
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset

METRICAS = [("LPIPS", "menor"), ("DISTS", "menor"), ("CLIP-I", "maior"),
            ("LVCorr_convencao_paper", "maior"), ("SSIM", "maior")]

BENCH = {
    "EBB": "EBB400 (foto real, uma cena por linha)",
    "RB": "RealBokeh test v2",
    "RD": "RealDOF",
    "LFREPRO": "LF-Bokeh reproduzido (BLB)",
}
MODELO = {
    "rotac-only": "so rota c (a+c)",
    "nosso": "nosso original, fase 2 (a+b+c)",
    "kfix": "kfix, K da rota b pela Eq. 3",
    "oficial": "oficial do paper",
    "oficial-paper": "oficial do paper",
    "fase1": "fase 1, so sintetico (a)",
    "nofilter": "sem filtro de SSIM",
    "sem-treino": "sem treino (FLUX.1-dev cru)",
    "LINHA-DE-IDENTIDADE": "LINHA DE IDENTIDADE",
}


def classifica(nome):
    # EBB antes de RD/RB: a busca e por substring e a ordem decide empates
    bench = next((b for b in ("LFREPRO", "EBB", "RD", "RB")
                  if re.search(rf"[-_]{b}\b|{b}", nome)), None)
    chave = next((k for k in sorted(MODELO, key=len, reverse=True) if nome.startswith(k)), None)
    if not chave:
        return bench, None
    rotulo = MODELO[chave]
    m = re.search(r"-(\d+)k-", nome)
    if m:
        rotulo = f"{rotulo} [step {int(m.group(1))*1000}]"
    return bench, rotulo


def cena_de(image_id):
    """Extrai a CENA de um id de imagem.

    Isto nao e detalhe: no `LF-Bokeh reproduzido` as 500 linhas saem de apenas
    **10 cenas** (`blb_277_k00_d00`, `blb_277_k00_d01`, ...), 50 variacoes de k e
    de plano de foco por cena. Tratar as 500 como independentes encolhe o
    intervalo de confianca por um fator de ~7 e inventa precisao que o dado nao
    tem. O tamanho efetivo de amostra ali e 10, nao 500.

    No RealBokeh (`..._test_f_51_level_3_aligned`) e no RealDOF
    (`realdof_10_aligned`) e uma cena por linha, entao o agrupamento nao muda
    nada -- mas a regra tem de valer para os tres do mesmo jeito.
    """
    s = str(image_id)
    s = re.sub(r"_k\d+_d\d+$", "", s)              # blb_277_k00_d00 -> blb_277
    s = re.sub(r"_level_\d+.*$", "", s)             # ..._f_51_level_3_aligned -> ..._f_51
    s = re.sub(r"_(aligned|misaligned|shift_[\d.]+px)$", "", s)
    return s


def _boot_cluster(valores_por_cena, n_boot, rng):
    """Bootstrap AGRUPADO: reamostra CENAS inteiras, nao linhas soltas."""
    cenas = list(valores_por_cena)
    somas = np.array([np.sum(valores_por_cena[c]) for c in cenas], float)
    contas = np.array([len(valores_por_cena[c]) for c in cenas], float)
    idx = rng.integers(0, len(cenas), size=(n_boot, len(cenas)))
    return somas[idx].sum(axis=1) / contas[idx].sum(axis=1)


def ic95(v, cenas=None, n_boot=10000, seed=0):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    v = v[ok]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if cenas is None:
        m = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    else:
        c = np.asarray(cenas)[ok]
        grupos = {}
        for cc, vv in zip(c, v):
            grupos.setdefault(cc, []).append(vv)
        if len(grupos) < 2:
            return float(v.mean()), np.nan, np.nan
        m = _boot_cluster(grupos, n_boot, rng)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pareado(a, b, n_boot=10000, seed=0):
    """a e b: dicts image_id -> valor. Pareia por imagem e reamostra por CENA."""
    comuns = sorted(set(a) & set(b))
    if len(comuns) < 2:
        return None
    difs, cenas = [], []
    for k in comuns:
        d = a[k] - b[k]
        if np.isfinite(d):
            difs.append(d); cenas.append(cena_de(k))
    if len(difs) < 2:
        return None
    grupos = {}
    for c, d in zip(cenas, difs):
        grupos.setdefault(c, []).append(d)
    rng = np.random.default_rng(seed)
    if len(grupos) < 2:
        return None
    md = _boot_cluster(grupos, n_boot, rng)
    p = 2 * min((md <= 0).mean(), (md >= 0).mean())
    return {"n_pares": len(difs), "n_cenas": len(grupos), "delta": float(np.mean(difs)),
            "ic95_lo": float(np.percentile(md, 2.5)),
            "ic95_hi": float(np.percentile(md, 97.5)),
            "p_bootstrap": float(min(1.0, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", nargs="+", required=True)
    ap.add_argument("--saida-md", default="/host/TABELA_FINAL.md")
    ap.add_argument("--saida-hf", default=None)
    ap.add_argument("--referencia", default="oficial do paper",
                    help="modelo usado como referencia do teste pareado")
    ap.add_argument("--incluir-repos", default="bokeh-eq4-",
                    help="substrings dos repos de saida a incluir, separadas por virgula. "
                         "As linhas de IDENTIDADE da campanha antiga podem entrar aqui: "
                         "a identidade devolve a entrada e nao depende do plano de foco, "
                         "entao o numero dela vale para as duas campanhas.")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    arquivos = []
    for padrao in args.entrada:
        arquivos += sorted(glob.glob(padrao))
    if not arquivos:
        print("nenhum parquet encontrado"); return 2
    df = pd.concat([pd.read_parquet(f) for f in arquivos], ignore_index=True)
    print(f"[tabela] {len(df)} linhas por imagem de {len(arquivos)} arquivos", flush=True)

    # repo -> nome do modelo, pela tabela bruta (é ela que guarda o rotulo)
    bruto = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=token).to_pandas()
    rotulo_por_repo = dict(zip(bruto["Dataset"].astype(str), bruto["Model"].astype(str)))

    df["id_interno"] = df["repo"].map(rotulo_por_repo)
    df = df[df["id_interno"].notna()].copy()
    cls = df["id_interno"].map(classifica)
    df["benchmark"] = [BENCH.get(c[0]) for c in cls]
    df["modelo"] = [c[1] for c in cls]
    df = df[df["benchmark"].notna() & df["modelo"].notna()].copy()
    # SELECAO DA CAMPANHA PELO REPO DE SAIDA, nao pelo nome do modelo.
    # A campanha com a Eq. 4 de verdade grava em repos `bokeh-eq4-*`; a antiga,
    # com o plano de foco no pixel central, grava em `bokeh-eval-*`. Separar
    # pelo nome do modelo seria fragil: as duas usam a string "eq4" no rotulo,
    # porque a antiga ja se dizia Eq. 4 sem ser.
    padroes = [x.strip() for x in args.incluir_repos.split(",") if x.strip()]
    df = df[df["repo"].apply(lambda r: any(pt in str(r) for pt in padroes))]
    print(f"[tabela] {df['modelo'].nunique()} modelos x {df['benchmark'].nunique()} benchmarks", flush=True)

    df["cena"] = df["image_id"].map(cena_de)
    linhas = []
    for (bench, modelo), g in df.groupby(["benchmark", "modelo"]):
        reg = {"benchmark": bench, "modelo": modelo, "n": len(g),
               "n_cenas": int(g["cena"].nunique()),
               "repo_imagens": g["repo"].iloc[0], "id_interno": g["id_interno"].iloc[0]}
        for m, _ in METRICAS:
            media, lo, hi = ic95(g[m].values, cenas=g["cena"].values)
            reg[m] = media; reg[f"{m}_ic95_lo"] = lo; reg[f"{m}_ic95_hi"] = hi
        reg["LVCorr_pearson_cru"] = float(g["LVCorr"].mean())
        reg["best_k_mediano"] = float(np.nanmedian(g["best_k"].values)) if g["best_k"].notna().any() else None
        linhas.append(reg)
    tab = pd.DataFrame(linhas)

    # margem de LPIPS sobre a identidade (unica comparacao valida ENTRE mesas)
    piso = {b: s["LPIPS"].iloc[0] for b, s in tab[tab.modelo == "LINHA DE IDENTIDADE"].groupby("benchmark")}
    tab["margem_LPIPS_sobre_identidade"] = [
        (piso.get(b, np.nan) - l) for b, l in zip(tab["benchmark"], tab["LPIPS"])]

    # ---- teste pareado contra a referencia, cena a cena ----
    comparacoes = []
    for bench, g in df.groupby("benchmark"):
        ref = g[g["modelo"] == args.referencia]
        if ref.empty:
            continue
        for m, sentido in METRICAS:
            base = dict(zip(ref["image_id"], ref[m]))
            for modelo, gm in g.groupby("modelo"):
                if modelo == args.referencia:
                    continue
                r = pareado(dict(zip(gm["image_id"], gm[m])), base)
                if r:
                    melhor = ("modelo" if (r["delta"] < 0) == (sentido == "menor") else "referencia")
                    comparacoes.append({"benchmark": bench, "modelo": modelo,
                                        "referencia": args.referencia, "metrica": m,
                                        "vence": melhor, **r})
    comp = pd.DataFrame(comparacoes)

    # ---------------- markdown ----------------
    out = ["# Tabela final de bokeh (formato da Tab. 3 do GenRefocus)\n",
           "Metricas por imagem, com IC95 por bootstrap percentil (10.000 reamostras).",
           "`LVCorr` esta na convencao do PAPER (quanto maior melhor): e o simetrico do",
           "Pearson cru, porque mais K significa mais desfoque e MENOS variancia do Laplaciano.\n",
           "> **Ressalva de comparabilidade.** O LF-Bokeh do paper nao foi liberado, entao a",
           "> mesa `LF-Bokeh reproduzido` e uma reconstrucao do protocolo. Os pesos OFICIAIS",
           "> medem LPIPS 0,2047 nela contra 0,0833 publicado. As comparacoes entre modelos",
           "> valem (mesmo dado, mesmo pipeline); os valores absolutos nao sao os do paper.\n"]

    for bench in sorted(tab["benchmark"].unique()):
        sub = tab[tab.benchmark == bench].sort_values("LPIPS")
        n = int(sub["n"].iloc[0]); nc = int(sub["n_cenas"].iloc[0])
        aviso = ("  **O IC95 e agrupado por cena: o tamanho efetivo de amostra aqui e "
                 f"{nc}, nao {n}.**" if nc < n else "")
        out.append(f"\n## {bench} ({n} imagens de {nc} cenas)\n{aviso}\n")
        out.append("| Metodo | LPIPS ↓ | DISTS ↓ | CLIP-I ↑ | LVCorr ↑ | margem LPIPS s/ identidade |")
        out.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            out.append("| {} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} |".format(
                r["modelo"], r["LPIPS"], r["LPIPS_ic95_lo"], r["LPIPS_ic95_hi"],
                r["DISTS"], r["CLIP-I"], r["LVCorr_convencao_paper"],
                r["margem_LPIPS_sobre_identidade"]))

    if not comp.empty:
        out.append(f"\n\n## Teste pareado contra `{args.referencia}`, cena a cena\n")
        out.append("Diferenca media por cena, IC95 e p bilateral por bootstrap. Negativo em LPIPS/DISTS = o modelo e melhor.\n")
        for bench in sorted(comp["benchmark"].unique()):
            out.append(f"\n### {bench}\n")
            out.append("| Modelo | Metrica | pares | cenas | delta | IC95 | p | vence |")
            out.append("|---|---|---|---|---|---|---|---|")
            s = comp[(comp.benchmark == bench) & (comp.metrica.isin(["LPIPS", "DISTS"]))]
            for _, r in s.sort_values(["metrica", "delta"]).iterrows():
                out.append("| {} | {} | {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} | {} |".format(
                    r["modelo"], r["metrica"], r["n_pares"], r["n_cenas"], r["delta"],
                    r["ic95_lo"], r["ic95_hi"], r["p_bootstrap"],
                    "**o modelo**" if r["vence"] == "modelo" else "a referencia"))

    md = "\n".join(out) + "\n"
    with open(args.saida_md, "w") as f:
        f.write(md)
    print(f"[tabela] markdown em {args.saida_md}", flush=True)
    print(md[:4000])

    if args.saida_hf:
        Dataset.from_pandas(tab.reset_index(drop=True)).push_to_hub(args.saida_hf, token=token, private=True)
        print(f"[tabela] tabela enviada para {args.saida_hf}", flush=True)
        if not comp.empty:
            Dataset.from_pandas(comp.reset_index(drop=True)).push_to_hub(
                args.saida_hf + "-pareado", token=token, private=True)
            print(f"[tabela] pareado enviado para {args.saida_hf}-pareado", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
