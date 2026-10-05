#!/usr/bin/env python3
"""C0-2 — quantas GTs DISTINTAS o filtro `top_k_sharpest` realmente seleciona.

POR QUE ISTO EXISTE
-------------------
O filtro do paper (§B.1: *"we retain the top 3,000 sharpest images to serve as
additional supervision"*) é implementado ranqueando LINHAS do df. No
`akcit-pixel/RealBokeh` cada linha é (uma entrada borrada de abertura X, o MESMO
`image_focus` da cena), e o score é medido justamente em `image_focus` — logo os
empates dentro de uma cena são EXATOS e o `argsort` puxa cenas inteiras.

FATO A MEDIR (é o que este script faz): quantas GTs distintas existem entre as
linhas selecionadas. Hipótese do plano: ~580 para 3000 linhas.

O QUE ISTO **NÃO** DECIDE: se o paper pretendia 3.000 alvos distintos. A frase é
ambígua num df que é (cena × abertura), e a auditoria independente registrou
três leituras possíveis. Trocar para seleção por cena é ALTERAÇÃO DE PROTOCOLO
(`top_k_mode="scene"`), a ser medida em A/B — não correção de fidelidade.

USO
---
    python3 scripts/c0_2_gts_distintas.py
    python3 scripts/c0_2_gts_distintas.py --k 3000 --recalcular
    python3 scripts/c0_2_gts_distintas.py --regex-cena '^(.*)_f[0-9.]+$'

GPU: não. Rede: sim. Tempo: minutos (decodifica imagens para medir nitidez).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
from collections import Counter
from pathlib import Path

from _comum import adicionar_raiz_ao_path, imprimir_tabela, token_hf

adicionar_raiz_ao_path()


def caminho_cache(cache_id: str, column: str, k: int, n: int, stage_cfg_kwargs: dict) -> str | None:
    """Chama `_filter_cache_path` do data.py passando SÓ o que a assinatura aceita.

    A chave do cache mudou nesta revisão (ganhou versão do seletor e, possivelmente,
    `top_k_mode`/`scene_key`). Hardcodar o nome do arquivo aqui faria este script
    ler um cache obsoleto e reportar o número ERRADO — exatamente o defeito que o
    C5-a descreve. Por isso resolvemos por introspecção da função real.
    """
    from genfocus_train import data as _data

    fn = getattr(_data, "_filter_cache_path", None)
    if fn is None:
        return None

    disponiveis = {
        "cache_id": cache_id, "column": column, "k": k, "n": n,
        **stage_cfg_kwargs,
    }
    sig = inspect.signature(fn)
    kwargs = {nome: disponiveis[nome] for nome in sig.parameters if nome in disponiveis}
    faltando = [
        nome for nome, par in sig.parameters.items()
        if nome not in kwargs and par.default is inspect.Parameter.empty
    ]
    if faltando:
        print(f"  AVISO: _filter_cache_path exige {faltando}, que este script não sabe "
              "preencher. Use --recalcular.")
        return None
    return fn(**kwargs)


def chave_cena(nome: str, regex: re.Pattern | None) -> str:
    """Identidade da cena a partir do `file_name_base`.

    Sem regex, usa a heurística "tira o último campo separado por _ ou -", que
    cobre nomes do tipo `cena0123_f2.8`. A heurística NÃO é confiável para todo
    df — por isso o script reporta também a contagem por hash dos bytes da GT,
    que não depende de convenção de nome nenhuma.
    """
    if regex is not None:
        m = regex.match(nome)
        if m:
            return m.group(1) if m.groups() else m.group(0)
        return nome
    return re.sub(r"[_-][^_-]*$", "", nome)


def md5_imagem(img) -> str:
    """Hash dos bytes decodificados da imagem (independe do container/compressão)."""
    try:
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return hashlib.md5(repr(img).encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(
        description="C0-2: mede a duplicação de ground-truth na seleção do top_k_sharpest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo", default="akcit-pixel/RealBokeh")
    p.add_argument("--split", default="train")
    p.add_argument("--column", default="image_focus", help="coluna usada como medida de foco")
    p.add_argument("--k", type=int, default=3000, help="top_k_sharpest do config")
    p.add_argument("--recalcular", action="store_true",
                   help="ignora o cache e refaz a seleção do zero (lento)")
    p.add_argument("--regex-cena", default=None,
                   help="regex com 1 grupo que extrai a cena do file_name_base")
    p.add_argument("--out", default="outputs/c0_2_gts_distintas.json")
    args = p.parse_args()

    token = token_hf()
    regex = re.compile(args.regex_cena) if args.regex_cena else None

    from datasets import load_dataset

    print("=" * 72)
    print("C0-2 — GTs distintas na seleção do top_k_sharpest")
    print("=" * 72)
    print(f"\nCarregando {args.repo}:{args.split} ...")
    ds = load_dataset(args.repo, split=args.split, token=token)
    n = len(ds)
    print(f"  {n} linhas no split.")

    # ── 1. o split INTEIRO: quantas cenas existem? ──────────────────────────
    tem_nome = "file_name_base" in ds.column_names
    cenas_split = None
    if tem_nome:
        nomes = ds["file_name_base"]
        cenas = [chave_cena(str(x), regex) for x in nomes]
        cont = Counter(cenas)
        cenas_split = len(cont)
        print(f"  {cenas_split} cenas distintas no split inteiro "
              f"(por prefixo de file_name_base; ~{n / max(cenas_split,1):.2f} linhas/cena)")
        print(f"  exemplos de nome: {[str(x) for x in nomes[:3]]}")
        print(f"  exemplos de cena: {cenas[:3]}")
    else:
        print("  AVISO: não há coluna `file_name_base`; contagem por nome indisponível.")

    # ── 2. quais índices o filtro seleciona ─────────────────────────────────
    idx = None
    origem = None
    if not args.recalcular:
        cache = caminho_cache(
            f"{args.repo}_{args.split}", args.column, args.k, n,
            {"top_k_mode": "row", "scene_key": "auto", "mode": "row"},
        )
        if cache:
            print(f"\n  caminho de cache resolvido: {cache}")
            if os.path.isfile(cache):
                idx = json.load(open(cache, encoding="utf-8"))
                origem = f"cache ({cache})"
                print(f"  cache HIT: {len(idx)} índices.")
            else:
                print("  cache MISS — vou recalcular.")

    if idx is None:
        print(f"\n  Medindo nitidez (Laplaciano) de {n} imagens em '{args.column}'...")
        import numpy as np
        from genfocus_train.data import _laplacian_variance

        scores = np.empty(n, dtype=np.float64)
        for i, row in enumerate(ds.select_columns([args.column])):
            scores[i] = _laplacian_variance(row[args.column])
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{n}")
        idx = sorted(int(j) for j in np.argsort(scores)[::-1][: args.k])
        origem = "recalculado"
        # quantos empates exatos? é o mecanismo que causa a duplicação
        empates = n - len(set(float(s) for s in scores))
        print(f"  scores com valor repetido: {empates} de {n} "
              "(empate exato = mesma GT, é o mecanismo do C5-b)")

    print(f"\n  origem da seleção: {origem} | {len(idx)} linhas selecionadas")

    # ── 3. quantas GTs distintas nas linhas selecionadas ────────────────────
    sub = ds.select(idx)

    print("\n  Contando GTs distintas por HASH dos bytes de image_focus...")
    hashes = set()
    for i, row in enumerate(sub.select_columns([args.column])):
        hashes.add(md5_imagem(row[args.column]))
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(idx)}")
    gts_por_hash = len(hashes)

    gts_por_nome = None
    if tem_nome:
        nomes_sel = [str(x) for x in sub["file_name_base"]]
        gts_por_nome = len({chave_cena(x, regex) for x in nomes_sel})

    # ── 4. relatório ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTADO")
    print("=" * 72)
    linhas_tab = [
        {"métrica": "linhas no split", "valor": n},
        {"métrica": "cenas no split (por nome)", "valor": cenas_split if cenas_split else "n/d"},
        {"métrica": "linhas selecionadas (k)", "valor": len(idx)},
        {"métrica": "GTs distintas (hash dos bytes)", "valor": gts_por_hash},
        {"métrica": "cenas distintas (por nome)", "valor": gts_por_nome if gts_por_nome else "n/d"},
        {"métrica": "linhas por GT (média)", "valor": f"{len(idx) / max(gts_por_hash, 1):.2f}"},
    ]
    imprimir_tabela(linhas_tab, ["métrica", "valor"])

    razao = len(idx) / max(gts_por_hash, 1)
    print()
    if razao > 1.5:
        print(f"  → FATO: as {len(idx)} linhas contêm só {gts_por_hash} GTs distintas "
              f"({razao:.1f} linhas por GT).")
        print("    Confirma o mecanismo do C5-b. Se a seleção por cena vale a pena é")
        print("    outra pergunta — é ALTERAÇÃO DE PROTOCOLO e exige o A/B do plano.")
    else:
        print(f"  → As linhas selecionadas já são quase todas GTs distintas ({razao:.2f}/GT).")
        print("    A premissa do C5-b NÃO se confirma neste df; reavaliar o item.")

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({
        "repo": args.repo, "split": args.split, "column": args.column, "k": args.k,
        "origem_selecao": origem, "linhas_split": n, "cenas_split": cenas_split,
        "linhas_selecionadas": len(idx), "gts_distintas_hash": gts_por_hash,
        "cenas_distintas_nome": gts_por_nome, "linhas_por_gt": round(razao, 3),
        "regex_cena": args.regex_cena,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[c0-2] JSON salvo em {destino}")


if __name__ == "__main__":
    main()
