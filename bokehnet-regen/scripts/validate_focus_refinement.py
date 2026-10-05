#!/usr/bin/env python3
"""O refinamento automático da região em foco melhorou a concordância? Contra gabarito.

    python3 scripts/validate_focus_refinement.py \\
        --pilot-dir output/c_pilot \\
        --raw-dir   /raid/.../data/RealBokeh_3MP \\
        --output-json output/focus_validation.json

## O gabarito

A RealBokeh_3MP publica, por cena, `focus_plane_distance` — a distância do plano de foco
**medida na captura**, com `focus_plane_uncertainty` ao lado, mediana **±0,010 m**. Isso
é gabarito de verdade, e é contra ele que a máscara é julgada:

    obtida   = `focus_disparity` do metadado = mediana(1/z[região em foco])
    gabarito = 1 / `focus_plane_distance`
    razão    = obtida ÷ gabarito             (= distância MEDIDA ÷ distância IMPLICADA)

Razão 1 é acerto. Razão 6,2 significa que a máscara pôs o plano de foco 6,2× mais perto
do que ele estava; razão 0,16, 6,2× mais longe.

**A linha de base é 35,2% dentro de ±25%** — medido no piloto de 162 amostras / 29 cenas
com a máscara do BiRefNet crua, antes do refinamento (`reference/MEDICAO_PLANO_FOCO.md`).

## Duas comparações, e só uma delas responde a pergunta

1. **Por `focus_source`** (`birefnet` / `birefnet_refined` / `retention_only`). Útil, e
   **confundida**: o grupo `birefnet` é, por construção, o subgrupo em que o BiRefNet já
   concordava com a física. Ele é mais fácil, e ficar melhor nele não prova nada sobre o
   refinamento.
2. **PAREADA, na mesma amostra**: `focus_disparity_from_initial_mask` guarda o que o
   rótulo teria sido sem refinamento. Comparar a amostra com ela mesma elimina a seleção
   de subgrupo, e é este número que decide.

Onde a máscara inicial era vazia não existe linha de base — eram justamente as amostras
**descartadas** antes, 20,6% do piloto. Elas entram numa terceira contagem, "recuperadas",
porque a alternativa a elas não é um número pior: é amostra nenhuma.

## O cuidado metodológico: de quem é a divergência?

O Depth Pro tem viés de escala próprio, então uma razão longe de 1 pode vir da
**profundidade** e não da **máscara**. Com os dados que existem no release, dá para
limitar a confusão de dois jeitos — nenhum deles resolve o problema por amostra, e o
relatório diz isso em voz alta em vez de atribuir tudo à máscara:

* **Alcançabilidade.** O metadado grava `disparity_min` e `disparity_max` da própria
  amostra. Se o gabarito cai **fora** dessa faixa, então **nenhuma máscara** poderia
  tê-lo alcançado, e a divergência é da profundidade (ou do gabarito) — não da máscara.
  Se cai dentro, alguma máscara chegaria lá, e aí a escolha da região é responsável.
* **Escala global.** Um viés multiplicativo global desloca todas as razões pelo mesmo
  fator. Dividindo a razão de cada amostra pela mediana global, o que resta é a
  **dispersão**, que nenhum fator único explica. Se o refinamento ganha também depois
  dessa correção, o ganho não é artefato de escala.

O que **não** dá para fazer com estes dados é separar as duas causas *por amostra*: para
isso seria preciso uma profundidade métrica de referência na cena, e a RealBokeh publica
uma distância só — a do plano de foco.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Concordância medida com a máscara do BiRefNet crua, antes do refinamento. `[M]`,
#: piloto da rota C, job 32224: 57 de 162 amostras dentro de ±25% do gabarito, em 29
#: cenas. Ver `reference/MEDICAO_PLANO_FOCO.md`. É a régua que o refinamento tem que
#: bater; um número novo mais alto que este só vale se vier do mesmo cálculo.
BASELINE_WITHIN_25_PCT = 0.352
BASELINE_N_SAMPLES = 162
BASELINE_N_SCENES = 29

#: Tolerâncias reportadas. ±25% é a do baseline; as outras dão a forma da distribuição.
TOLERANCIAS = (0.10, 0.25, 0.50)

#: Acima disto o próprio gabarito é frouxo e a amostra não serve para julgar máscara.
#: `[A]`: a incerteza mediana publicada é 0,010 m, então isto exclui a cauda, não o corpo.
INCERTEZA_RELATIVA_MAXIMA = 0.10

_FONTES = ("birefnet", "birefnet_refined", "retention_only")


# --------------------------------------------------------------------------------
# Entrada
# --------------------------------------------------------------------------------

def _percentil(valores: list[float], q: float) -> float:
    if not valores:
        return float("nan")
    ordenados = sorted(valores)
    idx = min(int(round(q / 100.0 * (len(ordenados) - 1))), len(ordenados) - 1)
    return ordenados[idx]


def _mediana(valores: list[float]) -> float:
    return _percentil(valores, 50)


def carrega_metadados(pilot_dir: Path) -> list[dict]:
    """Os `meta/<id>.json` aceitos do piloto.

    Lê os metadados, e não o manifesto, porque o diagnóstico pareado
    (`focus_disparity_from_initial_mask`) e a faixa de disparidade da amostra só existem
    lá. O manifesto serve à varredura rápida; esta ferramenta precisa do detalhe.
    """
    metas = []
    for path in sorted((pilot_dir / "meta").glob("*.json")):
        metas.append(json.loads(path.read_text(encoding="utf-8")))
    return metas


def carrega_gabarito(*, raw_dir: Optional[str], gabarito_json: Optional[str]) -> dict:
    """`scene_id` -> `{"focus_plane_distance_m", "focus_plane_uncertainty_m"}`.

    Duas origens, e nenhuma delas inventa valor:

    * `--raw-dir` lê `<split>/metadata/<cena>.json` do `timseizinger/RealBokeh_3MP` pela
      **mesma** função que a enumeração usa (`sources.realbokeh.load_scene_metadata`),
      com a chave qualificada pelo split. Reimplementar a leitura aqui seria uma segunda
      definição da chave de cena, e a numeração REINICIA em cada split: `train_1` e
      `test_1` são cenas físicas diferentes, e trocá-las entregaria a uma cena a
      distância de foco da outra, com JSON válido e completo.
    * `--gabarito-json` aceita um mapa pronto, para auditar um piloto sem ter o
      repositório bruto em disco.
    """
    gabarito: dict[str, dict] = {}
    if raw_dir:
        from sources.realbokeh import load_scene_metadata
        for scene_id, meta in load_scene_metadata(raw_dir).items():
            distancia = meta.get("focus_plane_distance")
            if distancia is None or not (float(distancia) > 0):
                continue                      # sem gabarito é sem gabarito, não é zero
            gabarito[scene_id] = {
                "focus_plane_distance_m": float(distancia),
                "focus_plane_uncertainty_m": _float_ou_nan(
                    meta.get("focus_plane_uncertainty")),
            }
    if gabarito_json:
        cru = json.loads(Path(gabarito_json).read_text(encoding="utf-8"))
        for chave, valor in cru.items():
            if isinstance(valor, (int, float)):
                valor = {"focus_plane_distance_m": float(valor)}
            distancia = float(valor["focus_plane_distance_m"])
            if distancia <= 0:
                continue
            gabarito[chave] = {
                "focus_plane_distance_m": distancia,
                "focus_plane_uncertainty_m": _float_ou_nan(
                    valor.get("focus_plane_uncertainty_m")),
            }
    return gabarito


def _float_ou_nan(valor) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------------
# Avaliação por amostra
# --------------------------------------------------------------------------------

def avalia(metas: list[dict], gabarito: dict) -> tuple[list[dict], Counter]:
    """Uma linha por amostra COM gabarito, e o histograma do que ficou de fora.

    O que fica de fora é contado e nomeado, nunca ignorado: uma taxa de acerto calculada
    sobre um subconjunto silencioso é o jeito mais fácil de publicar um número bonito.
    """
    linhas, fora = [], Counter()
    for meta in metas:
        chave = None
        for candidata in (meta.get("sample_id"), meta.get("scene_id")):
            if candidata in gabarito:
                chave = candidata
                break
        if chave is None:
            fora["sem_gabarito_para_a_cena"] += 1
            continue

        obtida = meta.get("focus_disparity")
        if not obtida or not math.isfinite(obtida) or obtida <= 0:
            fora["focus_disparity_invalido_no_metadado"] += 1
            continue
        if "focus_source" not in meta:
            # Release anterior ao refinamento. Não dá para separar por fonte, e fingir
            # que é `birefnet` inventaria proveniência.
            fora["metadado_sem_focus_source"] += 1
            continue

        distancia = gabarito[chave]["focus_plane_distance_m"]
        incerteza = gabarito[chave]["focus_plane_uncertainty_m"]
        gabarito_disp = 1.0 / distancia

        base = meta.get("focus_disparity_from_initial_mask")
        linhas.append({
            "sample_id": meta.get("sample_id"),
            "scene_id": meta.get("scene_id"),
            "focus_source": meta["focus_source"],
            "focus_was_refined": bool(meta.get("focus_was_refined")),
            "focus_agreement": meta.get("focus_agreement"),
            "focus_retention_in_region": meta.get("focus_retention_in_region"),
            "medida_m": distancia,
            "incerteza_m": incerteza,
            "incerteza_relativa": (incerteza / distancia
                                   if math.isfinite(incerteza) else float("nan")),
            "implicada_m": 1.0 / obtida,
            "razao": obtida / gabarito_disp,
            # A linha de base pareada: o que o rótulo teria sido SEM refinamento.
            "razao_sem_refino": (None if base in (None, 0) or not math.isfinite(base)
                                 else base / gabarito_disp),
            "implicada_sem_refino_m": (None if base in (None, 0)
                                       or not math.isfinite(base) else 1.0 / base),
            # Alcançabilidade: alguma máscara poderia ter chegado ao gabarito?
            "gabarito_alcancavel": _alcancavel(meta, gabarito_disp),
        })
    return linhas, fora


def _alcancavel(meta: dict, gabarito_disp: float) -> Optional[bool]:
    """O gabarito está dentro da faixa de disparidade que a cena tem?

    `None` quando o metadado não grava a faixa. Se `False`, **nenhuma** máscara sobre
    esta profundidade produziria o gabarito: a mediana de um subconjunto está sempre
    dentro do intervalo do conjunto. A divergência é então da profundidade ou do
    gabarito, e atribuí-la à máscara seria erro de causa.
    """
    lo, hi = meta.get("disparity_min"), meta.get("disparity_max")
    if lo is None or hi is None:
        return None
    return bool(float(lo) <= gabarito_disp <= float(hi))


# --------------------------------------------------------------------------------
# Agregação
# --------------------------------------------------------------------------------

def _fracao_dentro(razoes: list[float], tolerancia: float) -> float:
    if not razoes:
        return float("nan")
    return sum(1 for r in razoes if abs(r - 1.0) <= tolerancia) / len(razoes)


def por_cena(linhas: list[dict], campo: str = "razao") -> list[float]:
    """Uma razão por CENA, pela mediana das amostras dela.

    Existe porque a regra do projeto é contar em cenas **e** em amostras: os níveis de
    uma cena compartilham a AIF, logo compartilham profundidade e máscara inicial, e
    contar 2.341 cenas de 5 níveis como 11.705 unidades independentes infla qualquer
    fração de acerto. A `focus_source` PODE variar entre níveis da mesma cena — a
    retenção depende da bokeh, que muda com o nível —, e por isso a agregação por cena
    usa a mediana em vez de assumir um valor único.
    """
    grupos: dict[str, list[float]] = defaultdict(list)
    for linha in linhas:
        valor = linha.get(campo)
        if valor is not None and math.isfinite(valor):
            grupos[linha["scene_id"]].append(float(valor))
    return [_mediana(v) for v in grupos.values() if v]


def estatisticas(linhas: list[dict], campo: str = "razao") -> dict:
    """Concordância e forma da distribuição, em amostras E em cenas."""
    razoes = [l[campo] for l in linhas
              if l.get(campo) is not None and math.isfinite(l[campo])]
    razoes_cena = por_cena(linhas, campo)
    return {
        "n_amostras": len(razoes),
        "n_cenas": len(razoes_cena),
        "dentro": {f"±{int(100 * t)}%": _fracao_dentro(razoes, t) for t in TOLERANCIAS},
        "dentro_por_cena": {f"±{int(100 * t)}%": _fracao_dentro(razoes_cena, t)
                            for t in TOLERANCIAS},
        "razao_p05": _percentil(razoes, 5),
        "razao_mediana": _mediana(razoes),
        "razao_p95": _percentil(razoes, 95),
    }


def comparacao_pareada(linhas: list[dict]) -> dict:
    """O número que decide: a mesma amostra, com e sem refinamento.

    Só entram as amostras que **têm** linha de base. As de máscara vazia não têm, por
    definição — e são contadas separadamente como recuperadas, porque a alternativa a
    elas não é um rótulo pior, é amostra nenhuma.
    """
    com_base = [l for l in linhas if l["razao_sem_refino"] is not None]
    refinadas = [l for l in com_base if l["focus_was_refined"]]
    sem_base = [l for l in linhas if l["razao_sem_refino"] is None]

    def _bloco(grupo: list[dict]) -> dict:
        return {
            "n_amostras": len(grupo),
            "n_cenas": len({l["scene_id"] for l in grupo}),
            "antes": estatisticas(grupo, "razao_sem_refino"),
            "depois": estatisticas(grupo, "razao"),
        }

    saida = {"todas_com_linha_de_base": _bloco(com_base),
             "so_as_refinadas": _bloco(refinadas),
             "recuperadas_sem_linha_de_base": {
                 "n_amostras": len(sem_base),
                 "n_cenas": len({l["scene_id"] for l in sem_base}),
                 "nota": "máscara inicial vazia: antes eram DESCARTADAS "
                         "(focus_mask_empty, 20,6% do piloto)",
                 "concordancia": estatisticas(sem_base, "razao") if sem_base else None,
             }}

    # Quantas amostras melhoraram, pioraram e empataram — a contagem crua, que uma
    # média de frações poderia esconder.
    melhorou = piorou = empatou = 0
    for linha in refinadas:
        antes = abs(linha["razao_sem_refino"] - 1.0)
        depois = abs(linha["razao"] - 1.0)
        if depois < antes - 1e-9:
            melhorou += 1
        elif depois > antes + 1e-9:
            piorou += 1
        else:
            empatou += 1
    saida["por_amostra_refinada"] = {"melhorou": melhorou, "piorou": piorou,
                                     "empatou": empatou}
    return saida


def separacao_de_causas(linhas: list[dict]) -> dict:
    """Limita quanto da divergência pode ser da profundidade, e não da máscara.

    Ver o cabeçalho do módulo. Nenhum dos dois instrumentos separa as causas por
    amostra; os dois juntos dizem **quanto** da divergência não pode ser culpa da
    máscara.
    """
    fora_da_faixa = [l for l in linhas if l["gabarito_alcancavel"] is False]
    sem_faixa = [l for l in linhas if l["gabarito_alcancavel"] is None]
    dentro_da_faixa = [l for l in linhas if l["gabarito_alcancavel"] is True]

    razoes = [l["razao"] for l in linhas if math.isfinite(l["razao"])]
    escala = _mediana(razoes) if razoes else float("nan")

    corrigidas = []
    for linha in linhas:
        if escala and math.isfinite(escala):
            corrigidas.append({**linha, "razao": linha["razao"] / escala,
                               "razao_sem_refino": (
                                   None if linha["razao_sem_refino"] is None
                                   else linha["razao_sem_refino"] / escala)})

    frouxas = [l for l in linhas
               if math.isfinite(l["incerteza_relativa"])
               and l["incerteza_relativa"] > INCERTEZA_RELATIVA_MAXIMA]
    ids_frouxos = {l["sample_id"] for l in frouxas}

    return {
        "gabarito_fora_da_faixa_de_profundidade": {
            "n_amostras": len(fora_da_faixa),
            "n_cenas": len({l["scene_id"] for l in fora_da_faixa}),
            "fracao": len(fora_da_faixa) / len(linhas) if linhas else float("nan"),
            "leitura": "nenhuma máscara sobre esta profundidade produziria o gabarito: "
                       "a divergência é da PROFUNDIDADE ou do gabarito, não da máscara",
        },
        "sem_faixa_no_metadado": len(sem_faixa),
        "restrito_ao_alcancavel": (estatisticas(dentro_da_faixa)
                                   if dentro_da_faixa else None),
        "escala_global_estimada": escala,
        "com_escala_global_corrigida": {
            "nota": "razões divididas pela mediana global. Isto FORÇA a mediana a 1 e "
                    "mede só a DISPERSÃO, que nenhum fator único explica. Serve para "
                    "comparar grupos entre si, nunca como taxa de acerto absoluta.",
            "por_fonte": {fonte: estatisticas([l for l in corrigidas
                                               if l["focus_source"] == fonte])
                          for fonte in _FONTES},
            "pareada": comparacao_pareada(corrigidas) if corrigidas else None,
        },
        "gabarito_frouxo": {
            "n_amostras": len(frouxas),
            "limite_incerteza_relativa": INCERTEZA_RELATIVA_MAXIMA,
            "concordancia_sem_elas": estatisticas(
                [l for l in linhas if l["sample_id"] not in ids_frouxos]
            ) if frouxas else None,
        },
        "o_que_NAO_da_para_separar": (
            "por amostra, viés de escala do Depth Pro e erro de máscara não são "
            "separáveis com o que a RealBokeh publica: há uma distância medida por "
            "cena, a do plano de foco, e não uma profundidade métrica de referência. "
            "As duas medidas acima limitam a confusão; não a eliminam."),
    }


def maiores_divergencias(linhas: list[dict], quantas: int) -> list[dict]:
    """As piores, nomeadas. Ordenadas por |log razão|, que é simétrico: uma razão de 4
    e uma de 1/4 erram o mesmo tanto, em direções opostas."""
    validas = [l for l in linhas if math.isfinite(l["razao"]) and l["razao"] > 0]
    piores = sorted(validas, key=lambda l: abs(math.log(l["razao"])), reverse=True)
    return [{
        "sample_id": l["sample_id"], "scene_id": l["scene_id"],
        "focus_source": l["focus_source"],
        "medida_m": round(l["medida_m"], 3),
        "incerteza_m": (None if not math.isfinite(l["incerteza_m"])
                        else round(l["incerteza_m"], 3)),
        "implicada_m": round(l["implicada_m"], 3),
        "razao": round(l["razao"], 3),
        "razao_sem_refino": (None if l["razao_sem_refino"] is None
                             else round(l["razao_sem_refino"], 3)),
        "gabarito_alcancavel": l["gabarito_alcancavel"],
    } for l in piores[:quantas]]


# --------------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------------

def _pct(valor: float) -> str:
    return "    n/d" if valor != valor else f"{100 * valor:5.1f}%"


def _linha_de_stats(nome: str, st: dict) -> str:
    return (f"  {nome:<22} n={st['n_amostras']:>5} amostras / {st['n_cenas']:>4} cenas"
            f" · ±10% {_pct(st['dentro']['±10%'])}"
            f" · ±25% {_pct(st['dentro']['±25%'])}"
            f" · ±50% {_pct(st['dentro']['±50%'])}")


def _linha_de_razao(nome: str, st: dict) -> str:
    return (f"  {nome:<22} razão obtida÷medida  p05 {st['razao_p05']:6.3f}"
            f" · mediana {st['razao_mediana']:6.3f} · p95 {st['razao_p95']:6.3f}")


def imprime_relatorio(linhas: list[dict], fora: Counter, *, quantas: int) -> dict:
    """Imprime, e devolve o mesmo conteúdo em dict para gravar em JSON."""
    por_fonte = {fonte: [l for l in linhas if l["focus_source"] == fonte]
                 for fonte in _FONTES}
    outras = {l["focus_source"] for l in linhas} - set(_FONTES)

    print("\n" + "=" * 78)
    print("  CONCORDÂNCIA COM A DISTÂNCIA DE FOCO MEDIDA — rota C, §3.2(c)")
    print("=" * 78)
    print(f"  amostras com gabarito : {len(linhas)}"
          f" em {len({l['scene_id'] for l in linhas})} cenas")
    for motivo, n in fora.most_common():
        print(f"  fora da comparação    : {n:>5}  {motivo}")
    if not linhas:
        print("\n  Nenhuma amostra tem gabarito. Sem gabarito não há validação — e "
              "inventar\n  um número aqui seria pior que não ter número.\n")
        return {"n_amostras": 0, "fora": dict(fora)}

    geral = estatisticas(linhas)
    print("-" * 78)
    print("  1. POR FONTE DA REGIÃO EM FOCO")
    print("     (confundida de propósito: `birefnet` é o subgrupo em que o segmentador")
    print("      JÁ concordava com a física, logo é mais fácil por construção)")
    for fonte in _FONTES:
        grupo = por_fonte[fonte]
        if not grupo:
            print(f"  {fonte:<22} n=0")
            continue
        st = estatisticas(grupo)
        print(_linha_de_stats(fonte, st))
        print(_linha_de_razao("", st))
    for fonte in sorted(outras):
        st = estatisticas([l for l in linhas if l["focus_source"] == fonte])
        print(_linha_de_stats(f"{fonte} (FORA DO ENUM)", st))
    print(_linha_de_stats("TODAS", geral))
    print(_linha_de_razao("", geral))
    print(f"  por CENA, ±25%         : {_pct(geral['dentro_por_cena']['±25%'])}"
          f"  ({geral['n_cenas']} cenas — níveis da mesma cena"
          " compartilham AIF, máscara e profundidade)")

    pareada = comparacao_pareada(linhas)
    print("-" * 78)
    print("  2. PAREADA — A MESMA AMOSTRA, COM E SEM REFINAMENTO")
    print("     (é este bloco que responde à pergunta; o de cima não responde)")
    for nome, chave in (("todas com base", "todas_com_linha_de_base"),
                        ("só as refinadas", "so_as_refinadas")):
        bloco = pareada[chave]
        if not bloco["n_amostras"]:
            print(f"  {nome:<22} n=0")
            continue
        print(f"  {nome} — {bloco['n_amostras']} amostras / {bloco['n_cenas']} cenas")
        print(_linha_de_stats("    ANTES (BiRefNet)", bloco["antes"]))
        print(_linha_de_stats("    DEPOIS (refinada)", bloco["depois"]))
    contagem = pareada["por_amostra_refinada"]
    print(f"  amostra a amostra      : melhorou {contagem['melhorou']} · "
          f"piorou {contagem['piorou']} · empatou {contagem['empatou']}")
    rec = pareada["recuperadas_sem_linha_de_base"]
    print(f"  recuperadas            : {rec['n_amostras']} amostras / "
          f"{rec['n_cenas']} cenas sem linha de base — antes eram DESCARTADAS")
    if rec["concordancia"]:
        print(_linha_de_stats("    delas, concordância", rec["concordancia"]))

    causas = separacao_de_causas(linhas)
    print("-" * 78)
    print("  3. DE QUEM É A DIVERGÊNCIA — MÁSCARA OU PROFUNDIDADE?")
    fora_faixa = causas["gabarito_fora_da_faixa_de_profundidade"]
    print(f"  gabarito FORA da faixa de disparidade da cena: "
          f"{fora_faixa['n_amostras']} amostras "
          f"({_pct(fora_faixa['fracao'])}), {fora_faixa['n_cenas']} cenas")
    print("      nenhuma máscara chegaria lá: divergência da PROFUNDIDADE (ou do")
    print("      gabarito), não da escolha de região")
    if causas["restrito_ao_alcancavel"]:
        print(_linha_de_stats("  só o alcançável", causas["restrito_ao_alcancavel"]))
    print(f"  escala global estimada (mediana das razões): "
          f"{causas['escala_global_estimada']:.3f}")
    corr = causas["com_escala_global_corrigida"]
    if corr["pareada"]:
        bloco = corr["pareada"]["so_as_refinadas"]
        if bloco["n_amostras"]:
            print("  com a escala global removida, só as refinadas:")
            print(_linha_de_stats("    ANTES", bloco["antes"]))
            print(_linha_de_stats("    DEPOIS", bloco["depois"]))
    print("  " + causas["o_que_NAO_da_para_separar"].replace(
        ". ", ".\n  ").replace(": ", ":\n      "))

    print("-" * 78)
    print("  4. AS MAIORES DIVERGÊNCIAS")
    piores = maiores_divergencias(linhas, quantas)
    print(f"  {'amostra':<34} {'fonte':<17} {'medida':>9} {'implicada':>10} "
          f"{'razão':>7}  alcançável")
    for p in piores:
        print(f"  {str(p['sample_id']):<34} {p['focus_source']:<17} "
              f"{p['medida_m']:>9.3f} {p['implicada_m']:>10.3f} {p['razao']:>7.3f}  "
              f"{'sim' if p['gabarito_alcancavel'] else 'NÃO'}")

    veredito = _veredito(geral, pareada)
    print("=" * 78)
    for linha in veredito["texto"]:
        print("  " + linha)
    print("=" * 78 + "\n")

    return {
        "baseline": {"dentro_25pct": BASELINE_WITHIN_25_PCT,
                     "n_amostras": BASELINE_N_SAMPLES, "n_cenas": BASELINE_N_SCENES,
                     "fonte": "reference/MEDICAO_PLANO_FOCO.md [M], job 32224"},
        "n_amostras": len(linhas),
        "n_cenas": len({l["scene_id"] for l in linhas}),
        "fora_da_comparacao": dict(fora),
        "geral": geral,
        "por_focus_source": {fonte: (estatisticas(grupo) if grupo else None)
                             for fonte, grupo in por_fonte.items()},
        "pareada": pareada,
        "separacao_de_causas": causas,
        "maiores_divergencias": piores,
        "veredito": veredito,
    }


def _veredito(geral: dict, pareada: dict) -> dict:
    """A resposta, em uma linha, sem maquiagem.

    Regra: quem decide é o **pareado sobre as refinadas**. Se ele piorar, o veredito diz
    PIOROU mesmo que o número geral tenha subido — porque o número geral sobe também
    quando o lote muda de composição, e não é isso que está sendo perguntado.
    """
    bloco = pareada["so_as_refinadas"]
    contagem = pareada["por_amostra_refinada"]
    texto = []

    if not bloco["n_amostras"]:
        return {"decisao": "sem_amostras_refinadas_com_linha_de_base",
                "texto": ["VEREDITO: nenhuma amostra refinada tem linha de base "
                          "pareada — nada a concluir sobre o refinamento."]}

    antes = bloco["antes"]["dentro"]["±25%"]
    depois = bloco["depois"]["dentro"]["±25%"]
    delta = depois - antes
    if delta > 0.01:
        decisao = "melhorou"
    elif delta < -0.01:
        decisao = "PIOROU"
    else:
        decisao = "empatou"

    texto.append(f"VEREDITO ({decisao.upper()}): nas {bloco['n_amostras']} amostras "
                 f"refinadas com linha de base,")
    texto.append(f"  dentro de ±25% do gabarito: {100 * antes:.1f}% ANTES -> "
                 f"{100 * depois:.1f}% DEPOIS  ({100 * delta:+.1f} pp)")
    texto.append(f"  amostra a amostra: melhorou {contagem['melhorou']}, "
                 f"piorou {contagem['piorou']}, empatou {contagem['empatou']}")
    texto.append(f"  linha de base do piloto anterior (BiRefNet cru, "
                 f"{BASELINE_N_SAMPLES} amostras): "
                 f"{100 * BASELINE_WITHIN_25_PCT:.1f}%")
    texto.append(f"  lote inteiro agora: {100 * geral['dentro']['±25%']:.1f}% "
                 f"em {geral['n_amostras']} amostras / {geral['n_cenas']} cenas")
    if decisao == "PIOROU":
        texto.append("  O REFINAMENTO PIOROU A CONCORDÂNCIA. Não congele nada e não "
                     "gere o lote completo")
        texto.append("  com ele; o parâmetro a revisar primeiro é `focus_top_fraction`, "
                     "e depois")
        texto.append("  `focus_agreement_floor`. Ver reference/MEDICAO_PLANO_FOCO.md.")
    if contagem["piorou"] > contagem["melhorou"]:
        texto.append("  ATENÇÃO: mais amostras pioraram do que melhoraram, ainda que a "
                     "fração agregada")
        texto.append("  não mostre isso — a fração pode subir com poucas correções "
                     "grandes.")
    return {"decisao": decisao, "antes_25pct": antes, "depois_25pct": depois,
            "delta_pp": 100 * delta, "por_amostra": contagem, "texto": texto}


def main() -> int:
    p = argparse.ArgumentParser(
        description="O refinamento da região em foco melhorou a concordância com a "
                    "distância de foco MEDIDA?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-dir", required=True,
                   help="diretório de saída de um run da rota C (com meta/*.json)")
    p.add_argument("--raw-dir", default="",
                   help="raiz do timseizinger/RealBokeh_3MP, com "
                        "<split>/metadata/<cena>.json — a origem do gabarito")
    p.add_argument("--gabarito-json", default="",
                   help="alternativa ao --raw-dir: mapa scene_id/sample_id -> "
                        "{focus_plane_distance_m, focus_plane_uncertainty_m}")
    p.add_argument("--output-json", required=True)
    p.add_argument("--top-divergences", type=int, default=10)
    args = p.parse_args()

    if not args.raw_dir and not args.gabarito_json:
        raise SystemExit(
            "sem gabarito não há validação: passe --raw-dir (metadata da RealBokeh_3MP) "
            "ou --gabarito-json.")

    pilot = Path(args.pilot_dir)
    metas = carrega_metadados(pilot)
    if not metas:
        raise SystemExit(f"nenhum meta/*.json em {pilot}. Rode a rota C antes.")

    gabarito = carrega_gabarito(raw_dir=args.raw_dir or None,
                                gabarito_json=args.gabarito_json or None)
    if not gabarito:
        raise SystemExit("o gabarito veio vazio: confira --raw-dir / --gabarito-json.")

    linhas, fora = avalia(metas, gabarito)
    relatorio = imprime_relatorio(linhas, fora, quantas=args.top_divergences)
    relatorio["pilot_dir"] = str(pilot)

    saida = Path(args.output_json)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
    print(f"  relatório: {saida}\n")

    # Código de saída 0 quando há o que concluir. A conclusão em si — melhorou ou não —
    # é do relatório, e não do exit code: um script que devolvesse 1 em "piorou" viraria
    # motivo para alguém "consertar" o exit code em vez de olhar o número.
    return 0 if linhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
