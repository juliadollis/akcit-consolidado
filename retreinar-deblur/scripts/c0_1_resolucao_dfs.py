#!/usr/bin/env python3
"""C0-1 — mede a resolução REALMENTE armazenada nos dfs de treino.

POR QUE ISTO EXISTE
-------------------
Duas decisões do plano dependem deste número, e nenhuma das duas pode ser
tomada por argumento:

* **C11-1 (comparabilidade).** O paper nunca diz a resolução do DPDD; 1680×1120
  é conhecimento externo sobre o dataset original. Se `akcit-pixel/DDPD` estiver
  gravado menor que isso, o df já reduziu o blur em relação aos dados que o
  paper usou, e o LPIPS que calculamos sai numa resolução diferente da
  publicada. Isso NÃO invalida a comparação nosso-vs-oficial dentro do nosso
  pipeline (os dois passam pelo mesmo protocolo); invalida comparar o nosso
  número com o número PUBLICADO. A ação é documentar o protocolo, não regerar
  o df.

* **C4 (escala × cronograma de sigma).** A tabela de regimes do plano foi
  calculada assumindo 1024×688. Se a resolução real for outra, os fatores de
  escala e os `mu` mudam e a tabela precisa ser refeita. Este script já imprime
  a tabela recalculada com o que mediu.

O que ele NÃO faz: não conclui nada sozinho. Ele mede e imprime; a leitura fica
no `docs/REGISTRO_REVISAO.md`.

USO
---
    python3 scripts/c0_1_resolucao_dfs.py                       # 30 linhas/repo
    python3 scripts/c0_1_resolucao_dfs.py -n 100 --out res.json

GPU: não. Rede: sim (streaming, não baixa o df inteiro).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _comum import (
    adicionar_raiz_ao_path, alinhar16, imprimir_tabela, linha_regime,
    seq_len_de, token_hf,
)

adicionar_raiz_ao_path()

# (repo, split, colunas de imagem a medir)
FONTES_PADRAO = [
    ("akcit-pixel/DDPD", "train", ["image_blur", "image_focus"]),
    ("akcit-pixel/RealBokeh", "train", ["image_blur", "image_focus"]),
]


def medir_fonte(repo: str, split: str, colunas: list[str], n: int, token: str) -> dict:
    """Amostra `n` linhas via streaming e coleta os tamanhos de cada coluna."""
    from datasets import load_dataset

    print(f"\n[{repo}:{split}] amostrando {n} linhas (streaming)...")
    stream = load_dataset(repo, split=split, streaming=True, token=token)

    tamanhos: dict[str, list[tuple[int, int]]] = {c: [] for c in colunas}
    nomes: list[str] = []
    vistos = 0
    faltando: set[str] = set()

    for linha in stream:
        if vistos >= n:
            break
        vistos += 1
        nome = linha.get("file_name_base")
        if nome is not None:
            nomes.append(str(nome))
        for col in colunas:
            img = linha.get(col)
            if img is None:
                faltando.add(col)
                continue
            size = getattr(img, "size", None)
            if size is None:          # não decodificou como PIL
                faltando.add(col)
                continue
            tamanhos[col].append((int(size[0]), int(size[1])))

    if faltando:
        print(f"  AVISO: colunas ausentes ou não-imagem: {sorted(faltando)}")

    resultado: dict = {"repo": repo, "split": split, "n_amostras": vistos, "colunas": {}}

    for col in colunas:
        pares = tamanhos[col]
        if not pares:
            resultado["colunas"][col] = {"n": 0, "erro": "sem amostras válidas"}
            continue
        larguras = [w for w, _ in pares]
        alturas = [h for _, h in pares]
        cont = Counter(pares)
        moda_par, moda_freq = cont.most_common(1)[0]
        resultado["colunas"][col] = {
            "n": len(pares),
            "larguras": {"min": min(larguras), "max": max(larguras)},
            "alturas": {"min": min(alturas), "max": max(alturas)},
            "moda": {"w": moda_par[0], "h": moda_par[1],
                     "freq": moda_freq, "fracao": round(moda_freq / len(pares), 4)},
            "distintos": [{"w": w, "h": h, "n": c} for (w, h), c in cont.most_common(8)],
            "homogeneo": len(cont) == 1,
        }
        print(f"  {col}: {len(cont)} resolução(ões) distinta(s); "
              f"moda {moda_par[0]}×{moda_par[1]} ({moda_freq}/{len(pares)})")

    resultado["exemplos_file_name_base"] = nomes[:5]
    return resultado


def tabela_de_regimes(w: int, h: int, image_size: int = 512) -> list[dict]:
    """Recalcula a tabela do C4 para a resolução REAL medida.

    Os três regimes são os que o plano compara:
      treino          — crop image_size², seq fixo
      eval long_side=512 — lado MAIOR vai a 512 (é o que reduz demais)
      eval long_side=0   — resolução nativa; o tile do forward é image_size²,
                           mas o `mu` sai do seq da imagem INTEIRA (flux.py:624)
    """
    linhas = [linha_regime(f"treino {image_size}²", image_size, image_size)]

    # eval com long_side=512: lado maior -> 512, depois trunca a múltiplo de 16
    if w >= h:
        nw, nh = 512, int(round(h * (512 / w)))
    else:
        nh, nw = 512, int(round(w * (512 / h)))
    nw, nh = alinhar16(nw, nh, para_cima=False)
    linhas.append(linha_regime("eval long_side=512", nw, nh))

    # eval long_side=0: nativo, arredonda PARA CIMA a múltiplo de 16
    fw, fh = alinhar16(w, h, para_cima=True)
    l = linha_regime("eval long_side=0", fw, fh)
    l["obs"] = f"forward em tiles de {image_size}² (seq {seq_len_de(image_size, image_size)}), mu do total"
    linhas.append(l)

    # fatores de escala espacial, relativos ao que o treino "short_side" vê
    fator_treino = image_size / min(w, h)
    for l in linhas:
        if l["regime"].startswith("treino"):
            l["escala_vs_treino"] = "1,000× (ref)"
        elif "512" in l["regime"]:
            l["escala_vs_treino"] = f"{(512 / max(w, h)) / fator_treino:.3f}×"
        else:
            l["escala_vs_treino"] = f"{1.0 / fator_treino:.3f}×"
    return linhas


def main() -> None:
    p = argparse.ArgumentParser(
        description="C0-1: mede a resolução armazenada nos dfs e recalcula a tabela de regimes do C4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--n-amostras", type=int, default=30,
                   help="linhas amostradas por repo (streaming)")
    p.add_argument("--repos", nargs="*", default=None,
                   help="repos no formato repo:split (default: DDPD:train e RealBokeh:train)")
    p.add_argument("--image-size", type=int, default=512,
                   help="tamanho do crop de treino, para a tabela de regimes")
    p.add_argument("--out", default="outputs/c0_1_resolucao.json",
                   help="onde salvar o JSON")
    args = p.parse_args()

    token = token_hf()

    if args.repos:
        fontes = []
        for spec in args.repos:
            repo, _, split = spec.partition(":")
            fontes.append((repo, split or "train", ["image_blur", "image_focus"]))
    else:
        fontes = FONTES_PADRAO

    print("=" * 72)
    print("C0-1 — resolução armazenada nos dfs de treino")
    print("=" * 72)

    resultados = []
    for repo, split, colunas in fontes:
        try:
            resultados.append(medir_fonte(repo, split, colunas, args.n_amostras, token))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERRO em {repo}:{split} -> {exc}")
            resultados.append({"repo": repo, "split": split, "erro": str(exc)})

    # ── tabela de regimes recalculada por fonte ─────────────────────────────
    print("\n" + "=" * 72)
    print("Tabela de regimes do C4, RECALCULADA com a resolução medida")
    print("(a do plano assumia 1024×688 — se divergir, o plano precisa ser atualizado)")
    print("=" * 72)

    for r in resultados:
        col = r.get("colunas", {}).get("image_blur") or {}
        moda = col.get("moda")
        if not moda:
            continue
        w, h = moda["w"], moda["h"]
        print(f"\n[{r['repo']}] moda {w}×{h}"
              + ("" if col.get("homogeneo") else "  (ATENÇÃO: resoluções heterogêneas neste repo)"))
        linhas = tabela_de_regimes(w, h, args.image_size)
        imprimir_tabela(linhas, ["regime", "w", "h", "seq", "mu", "exp_mu", "escala_vs_treino"])
        r["tabela_regimes"] = linhas

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps({"fontes": resultados, "image_size": args.image_size},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[c0-1] JSON salvo em {destino}")
    print("[c0-1] Leitura e conclusão: registrar em docs/REGISTRO_REVISAO.md (C11-1 e C4).")


if __name__ == "__main__":
    main()
