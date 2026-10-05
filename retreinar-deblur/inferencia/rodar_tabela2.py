#!/usr/bin/env python3
"""Reproduz a TABELA 2 do paper (benchmark de defocus deblurring).

POR QUE SÓ A TABELA 2
---------------------
O paper tem três tabelas e cada uma usa um dataset diferente:

    Tab. 2  defocus deblurring  -> RealDOF + DPDD   -> DeblurNet   REPRODUZÍVEL
    Tab. 3  bokeh synthesis     -> LF-Bokeh         -> BokehNet    dataset não público
    Tab. 4  refocusing          -> LF-Refocus       -> pipeline    dataset não público

A org `nycu-cplab` não publicou LF-Bokeh nem LF-Refocus. Então a Tabela 2 é o
único número que pode ir lado a lado com o publicado — e mesmo ela com as
ressalvas do PROTOCOLO.md.

AS DUAS MESAS
-------------
    DPDD     -> akcit-pixel/DDPD:test         75 imagens
    RealDOF  -> akcit-pixel/RealDOF:validation 50 imagens  (é o único split que existe)

Atenção ao nome do repo: é `DDPD` (com os D e P trocados em relação ao paper,
que escreve DPDD). Não é typo nosso, é o nome do repo no Hub.

USO
---
    # a nossa linha + a linha Input, nas duas mesas
    python3 inferencia/rodar_tabela2.py --lora pesos/deblur.safetensors --out saidas/tab2

    # só uma mesa, e só 5 imagens, para validar o caminho antes de gastar GPU
    python3 inferencia/rodar_tabela2.py --lora p.safetensors --mesa dpdd -n 5 --out /tmp/t

    # peso main+cond (o modelo de 60k do projeto), com o texto fora do adapter
    python3 inferencia/rodar_tabela2.py --lora nosso.safetensors \
        --main-adapter deblurring --text-adapter none --out saidas/tab2_maincond

    # só remontar a tabela a partir de JSONs já calculados
    python3 inferencia/rodar_tabela2.py --so-tabela --out saidas/tab2

GPU: sim. Tempo: (75 + 50) x ~50 s ~= 1h45 por modelo, mais as métricas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metricas as M
from _bootstrap import imprimir_tabela

MESMO_QUE_MAIN = "__same_as_main__"

# As duas mesas da Tabela 2. Contagens conferidas em HANDOFF_PROJECT_HISTORY.md
# secao 4: DDPD tem validation 73 / test 75 / train 344; RealDOF tem SO
# validation 50.
MESAS = {
    "dpdd": {
        "nome_paper": "DPDD",
        "dataset": "akcit-pixel/DDPD",
        "split": "test",
        "n_esperado": 75,
        "obs": "split de teste, o benchmark canônico do DPDD",
    },
    "realdof": {
        "nome_paper": "RealDOF",
        "dataset": "akcit-pixel/RealDOF",
        "split": "validation",
        "n_esperado": 50,
        "obs": "único split do repo; corresponde ao conjunto de teste de 50 imagens do RealDOF",
    },
}

# Valores PUBLICADOS na Tabela 2 (arXiv 2512.16923v3). Só as linhas `Input` e
# `GenRefocus (Ours)`. O paper traz também DRBNet, Restormer, INIKNet,
# Bokehlicious e DiffCamera; não os reproduzimos aqui, então não os listo para
# não dar a impressão de que foram medidos.
PUBLICADO = {
    "dpdd": {
        "Input": {"LPIPS": 0.3485, "DISTS": 0.1827, "CLIP-IQA": 0.4337,
                  "MANIQA": 0.3325, "MUSIQ": 45.5376},
        "GenRefocus (paper)": {"LPIPS": 0.1440, "DISTS": 0.0772, "CLIP-IQA": 0.4755,
                               "MANIQA": 0.3452, "MUSIQ": 49.4122},
    },
    "realdof": {
        "Input": {"LPIPS": 0.5241, "DISTS": 0.2865, "CLIP-IQA": 0.3562,
                  "MANIQA": 0.2213, "MUSIQ": 28.7087},
        "GenRefocus (paper)": {"LPIPS": 0.2408, "DISTS": 0.1126, "CLIP-IQA": 0.4595,
                               "MANIQA": 0.2884, "MUSIQ": 43.5222},
    },
}


def _norm_adapter(v):
    return None if str(v).lower() in {"none", "null", ""} else v


def fmt(nome_metrica: str, v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}" if nome_metrica != "MUSIQ" else f"{v:.4f}"


def montar_tabela(resultados: dict, chave_mesa: str) -> list[dict]:
    """Uma linha por fonte (publicado e medido), colunas = as 5 métricas."""
    linhas = []
    for rotulo, vals in PUBLICADO[chave_mesa].items():
        linhas.append({"linha": rotulo, "origem": "publicado",
                       **{m: fmt(m, vals.get(m)) for m in M.ORDEM}})
    for rotulo, proto in resultados.items():
        res = proto.get("resultado", {})
        linhas.append({
            "linha": rotulo, "origem": "medido",
            **{m: fmt(m, (res.get(m) or {}).get("media")) for m in M.ORDEM},
        })
    return linhas


def escrever_md(saida: Path, tabelas: dict[str, list[dict]], protocolos: dict) -> Path:
    """Gera TABELA2.md com as duas mesas e o protocolo de cada linha medida."""
    L = ["# Tabela 2 reproduzida — defocus deblurring", "",
         "Valores `publicado` são do arXiv 2512.16923v3, Tabela 2.",
         "Valores `medido` saíram deste pipeline.", "",
         "**Antes de tirar conclusão de um `medido` contra um `publicado`, leia o",
         "`PROTOCOLO.md`**: o paper não informa a resolução de avaliação nem a",
         "variante de cada métrica, e as duas coisas mudam os números.", ""]
    for chave, linhas in tabelas.items():
        mesa = MESAS[chave]
        L += [f"## {mesa['nome_paper']} — `{mesa['dataset']}:{mesa['split']}`", "",
              f"{mesa['obs']}. Esperado: {mesa['n_esperado']} imagens.", "",
              "| linha | origem | " + " | ".join(f"{m} {'↓' if M.SENTIDO[m]=='menor' else '↑'}"
                                                 for m in M.ORDEM) + " |",
              "|---|---|" + "---|" * len(M.ORDEM)]
        for l in linhas:
            L.append("| " + l["linha"] + " | " + l["origem"] + " | "
                     + " | ".join(l[m] for m in M.ORDEM) + " |")
        L.append("")
    L += ["## Protocolo de cada linha medida", ""]
    for rotulo, p in protocolos.items():
        mets = ", ".join(f"{k}={v['variante_pyiqa']}" for k, v in p.get("metricas", {}).items())
        # montado FORA da f-string: aninhar a mesma aspa dentro de f-string só
        # compila em Python 3.12+, e o container do cluster tem 3.10.
        res = [str(r["w"]) + "×" + str(r["h"]) for r in p.get("resolucoes_da_metrica", [])]
        res_txt = ", ".join(res) if res else "—"
        L += [f"### {rotulo}", "",
              f"- dataset: `{p.get('dataset')}:{p.get('split')}` — {p.get('n_amostras')} amostras",
              f"- resoluções da métrica: {res_txt}",
              f"- `long_side`={p.get('long_side')} · GT com resize: {p.get('gt_passou_por_resize')}"
              f" (`{p.get('gt_resize_fn')}`)",
              f"- modo: {p.get('modo')} · adapter: {p.get('variante_adapter')}"
              f" · texto: {p.get('text_adapter')}",
              f"- `guidance_scale`={p.get('guidance_scale')} · `steps`={p.get('steps')}"
              f" · seed={p.get('seed')}",
              f"- variantes de métrica: {mets}", ""]
        peso = p.get("peso")
        if peso:
            L.append(f"- peso: `{peso['nome']}` ({peso['bytes']} bytes, sha256 "
                     f"`{peso['sha256'][:16]}…`)")
            L.append("")
    md = saida / "TABELA2.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md


def main() -> None:
    p = argparse.ArgumentParser(
        description="Roda o benchmark da Tabela 2 (DPDD + RealDOF) e emite a tabela.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lora", default=None, help="peso a avaliar (omita com --so-tabela)")
    p.add_argument("--out", required=True, help="pasta raiz das saídas")
    p.add_argument("--mesa", default="all", choices=["all", *MESAS],
                   help="restringe a uma mesa, para não re-inferir o que já está pronto")
    p.add_argument("-n", type=int, default=None, help="limita imagens por mesa (validação rápida)")
    p.add_argument("--rotulo", default="GenRefocus (nosso)", help="nome da nossa linha")
    p.add_argument("--sem-identidade", action="store_true",
                   help="não calcula a linha Input (útil se já foi calculada)")
    p.add_argument("--so-tabela", action="store_true",
                   help="não roda nada; remonta a tabela dos JSONs já existentes")

    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--long-side", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--main-adapter", default="none")
    p.add_argument("--text-adapter", default=MESMO_QUE_MAIN)
    p.add_argument("--no-tiling", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salvar-imagens", action="store_true")
    M.adicionar_args(p)
    args = p.parse_args()

    if not args.so_tabela and not args.lora:
        raise SystemExit("Informe --lora, ou use --so-tabela para só remontar a tabela.")

    raiz = Path(args.out)
    raiz.mkdir(parents=True, exist_ok=True)
    chaves = list(MESAS) if args.mesa == "all" else [args.mesa]

    ta = args.text_adapter if args.text_adapter == MESMO_QUE_MAIN else _norm_adapter(args.text_adapter)
    sobrescritas = M.variantes_dos_args(args)

    tabelas: dict[str, list[dict]] = {}
    todos_protocolos: dict[str, dict] = {}

    for chave in chaves:
        mesa = MESAS[chave]
        print("\n" + "#" * 74)
        print(f"# MESA {mesa['nome_paper']} — {mesa['dataset']}:{mesa['split']}")
        print("#" * 74)

        resultados: dict[str, dict] = {}

        alvos = []
        if not args.sem_identidade:
            alvos.append(("Input (nosso)", raiz / chave / "input", dict(identidade=True)))
        if args.lora:
            alvos.append((args.rotulo, raiz / chave / "modelo",
                          dict(identidade=False, lora=Path(args.lora))))

        for rotulo, pasta, extra in alvos:
            jp = pasta / "protocolo_e_resultado.json"
            if args.so_tabela:
                if jp.is_file():
                    resultados[rotulo] = json.loads(jp.read_text(encoding="utf-8"))
                    print(f"[tabela] lido de {jp}")
                else:
                    print(f"[tabela] AVISO: {jp} não existe; linha {rotulo!r} fica vazia")
                continue

            from avaliar_deblur import avaliar
            resultados[rotulo] = avaliar(
                dataset=mesa["dataset"], split=mesa["split"], out=pasta,
                n=args.n, steps=args.steps, long_side=args.long_side,
                guidance_scale=args.guidance_scale,
                main_adapter=_norm_adapter(args.main_adapter), text_adapter=ta,
                no_tiling=args.no_tiling, seed=args.seed,
                salvar_imagens=args.salvar_imagens,
                conjunto_variantes=args.variantes, sobrescrever_variantes=sobrescritas,
                rotulo=rotulo, **extra,
            )

        for rotulo, proto in resultados.items():
            todos_protocolos[f"{mesa['nome_paper']} · {rotulo}"] = proto

        tabelas[chave] = montar_tabela(resultados, chave)

        print(f"\n=== {mesa['nome_paper']} ===")
        imprimir_tabela(tabelas[chave], ["linha", "origem", *M.ORDEM])

    md = escrever_md(raiz, tabelas, todos_protocolos)
    (raiz / "tabela2.json").write_text(
        json.dumps({"tabelas": tabelas, "protocolos": todos_protocolos,
                    "publicado": PUBLICADO, "mesas": MESAS},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[tabela] {md}")
    print(f"[tabela] {raiz / 'tabela2.json'}")
    print("\nLEIA o PROTOCOLO.md antes de comparar 'medido' com 'publicado':")
    print("  o paper não informa a resolução de avaliação nem a variante de métrica.")


if __name__ == "__main__":
    main()
