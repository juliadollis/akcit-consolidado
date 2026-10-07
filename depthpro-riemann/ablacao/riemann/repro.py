"""
riemann/repro.py
================
Reprodutibilidade e inferência estatística.

Corrige duas lacunas identificadas nas revisões:

1. **Intervalos de confiança em amostras pequenas.** O bootstrap percentil é
   estruturalmente incapaz de ultrapassar o menor e o maior valor observados. Com 3
   sementes ele entrega cobertura real ~75%, não os 95% anunciados. Para n pequeno use
   `mean_ci` (t de Student), que tem cobertura correta.

2. **Comparação emparelhada.** Confrontar dois intervalos independentes é a análise mais
   fraca possível: a variância "cena fácil vs cena difícil" domina o ruído e mascara o
   efeito. `compare_paired` calcula a diferença item a item (ou cena a cena) contra o
   baseline, o que elimina essa variância e tem muito mais poder.

Regra prática adotada aqui:
  - variação entre SEMENTES (n ~ 3-5)  -> `mean_ci` (t de Student)
  - variação entre ITENS/CENAS (n alto)-> `bootstrap_ci` ou `mean_ci`, ambos válidos
  - comparação contra baseline         -> SEMPRE `compare_paired`
"""

from __future__ import annotations

import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True):
    """Semeia todas as fontes de aleatoriedade. Chame no início de cada run/trial."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Intervalos de confiança
# ---------------------------------------------------------------------------
def mean_ci(values: Sequence[float], conf: float = 0.95
            ) -> Tuple[float, float, float]:
    """
    IC para a média por **t de Student**. Correto para amostras pequenas (sementes).
    Retorna (média, limite_inferior, limite_superior).

    Com n=3 o multiplicador é 4.303 (e não 1.96), o que reflete honestamente a
    incerteza de estimar o desvio a partir de 3 pontos.
    """
    a = np.asarray(list(values), float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(a.mean())
    if n == 1:
        return m, float("nan"), float("nan")
    try:
        from scipy import stats
        t = float(stats.t.ppf(1 - (1 - conf) / 2, df=n - 1))
    except ImportError:
        warnings.warn("[repro] scipy ausente; usando aproximação normal (otimista).")
        t = 1.959963985
    se = float(a.std(ddof=1)) / np.sqrt(n)
    return m, m - t * se, m + t * se


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0
                 ) -> Tuple[float, float, float]:
    """
    IC por bootstrap percentil. Adequado para n grande (itens, cenas).

    AVISO: com n < 10 este método subestima a incerteza, porque os limites não podem
    ultrapassar o menor e o maior valor da amostra. Nesse regime, prefira `mean_ci`.
    """
    a = np.asarray(list(values), float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    if a.size < 10:
        warnings.warn(
            f"[repro] bootstrap com n={a.size} subestima a incerteza "
            f"(cobertura real ~75% para n=3). Use mean_ci() para nível-semente.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = a[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(a.mean()), lo, hi


# ---------------------------------------------------------------------------
# Testes de significância
# ---------------------------------------------------------------------------
def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Teste t pareado. Retorna (t, p). Requer scipy."""
    try:
        from scipy import stats
    except ImportError:
        warnings.warn("[repro] scipy ausente — sem significância.")
        return float("nan"), float("nan")
    t, p = stats.ttest_rel(np.asarray(a, float), np.asarray(b, float))
    return float(t), float(p)


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank pareado (não-paramétrico). Retorna (W, p)."""
    try:
        from scipy import stats
    except ImportError:
        warnings.warn("[repro] scipy ausente — sem significância.")
        return float("nan"), float("nan")
    try:
        w, p = stats.wilcoxon(np.asarray(a, float), np.asarray(b, float))
        return float(w), float(p)
    except ValueError:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Agrupamento por cena
# ---------------------------------------------------------------------------
def scene_of(key: str) -> str:
    """
    Extrai o nome da cena a partir da chave do item.

    O download do Hypersim nomeia como `{cena}__{cam_frame}`, ex.:
    `ai_001_001__cam_00_0000` -> `ai_001_001`. Se não houver o separador, tenta o
    padrão ai_VVV_NNN; caso contrário devolve a chave inteira (1 item = 1 "cena").
    """
    if "__" in key:
        return key.split("__", 1)[0]
    parts = key.split("_")
    if len(parts) >= 3 and parts[0] == "ai":
        return "_".join(parts[:3])
    return key


def group_by_scene(values: Dict[str, float]) -> Dict[str, float]:
    """
    Agrega métricas por item em médias por CENA.

    Isto importa: imagens da mesma cena são fortemente correlacionadas, então tratar
    300 imagens de 4 cenas como 300 amostras independentes produz intervalos
    otimistas demais. O tamanho efetivo de amostra é o número de CENAS.
    """
    acc: Dict[str, List[float]] = defaultdict(list)
    for k, v in values.items():
        if np.isfinite(v):
            acc[scene_of(k)].append(float(v))
    return {s: float(np.mean(vs)) for s, vs in acc.items() if vs}


# ---------------------------------------------------------------------------
# Comparação emparelhada contra um baseline
# ---------------------------------------------------------------------------
def compare_paired(modelo: Dict[str, float], baseline: Dict[str, float],
                   maior_melhor: bool = True, nivel: str = "cena",
                   conf: float = 0.95) -> Dict[str, object]:
    """
    Compara `modelo` contra `baseline` de forma EMPARELHADA, item a item.

    Args:
        modelo, baseline: dicionários {chave_do_item: valor_da_métrica}. As chaves
            precisam ser as mesmas nos dois (o emparelhamento é por chave).
        maior_melhor: True para boundary F-score, False para AbsRel.
        nivel: "cena" agrega por cena antes de comparar (recomendado); "item"
            compara imagem a imagem (intervalos otimistas por correlação intra-cena).

    Retorna um dicionário com a diferença média, o IC da diferença, os p-valores
    pareados (t e Wilcoxon), a taxa de vitórias e o n efetivo.

    Por que emparelhar: a variância entre cenas é muito maior que o efeito que se
    quer medir. Ao subtrair baseline de modelo na MESMA cena, essa variância
    desaparece e o teste ganha poder. Comparar dois ICs independentes é o caminho
    mais fraco e pode declarar "inconclusivo" um efeito real e consistente.
    """
    if nivel == "cena":
        m = group_by_scene(modelo)
        b = group_by_scene(baseline)
    else:
        m, b = dict(modelo), dict(baseline)

    chaves = sorted(set(m) & set(b))
    if not chaves:
        return {"erro": "nenhuma chave em comum entre modelo e baseline"}

    vm = np.array([m[k] for k in chaves], float)
    vb = np.array([b[k] for k in chaves], float)
    # delta > 0 sempre significa "modelo melhor que baseline"
    delta = (vm - vb) if maior_melhor else (vb - vm)

    d_media, d_lo, d_hi = mean_ci(delta, conf=conf)
    t_stat, p_t = paired_ttest(vm, vb)
    w_stat, p_w = wilcoxon(vm, vb)
    vitorias = float(np.mean(delta > 0))

    # IC da diferença também por bootstrap, para comparação entre métodos. Com n de
    # cenas razoável (>=10) os dois devem concordar; divergência é sinal de amostra
    # pequena ou distribuição com cauda pesada, e vale investigar antes de reportar.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, b_lo, b_hi = bootstrap_ci(delta)

    alfa = 1 - conf
    sig_t_ic = bool(np.isfinite(d_lo) and d_lo > 0)          # IC t não contém zero
    sig_t_p = bool(np.isfinite(p_t) and p_t < alfa)           # p do t pareado
    sig_wilcoxon = bool(np.isfinite(p_w) and p_w < alfa)      # p do Wilcoxon
    sig_boot = bool(np.isfinite(b_lo) and b_lo > 0)           # IC bootstrap
    concordam = (sig_t_p == sig_wilcoxon)

    return {
        "nivel": nivel,
        "n": len(chaves),
        "media_modelo": float(vm.mean()),
        "media_baseline": float(vb.mean()),
        "delta_medio": d_media,
        "delta_ic95_t": [d_lo, d_hi],
        "delta_ic95_bootstrap": [b_lo, b_hi],
        "t_estatistica": t_stat,
        "p_ttest_pareado": p_t,
        "wilcoxon_estatistica": w_stat,
        "p_wilcoxon": p_w,
        "taxa_vitorias": vitorias,
        "significativo_t": sig_t_p,
        "significativo_wilcoxon": sig_wilcoxon,
        "significativo_ic_t": sig_t_ic,
        "significativo_ic_bootstrap": sig_boot,
        "testes_concordam": concordam,
        # veredito conservador: exige concordância entre paramétrico e não-paramétrico
        "significativo": bool(sig_t_p and sig_wilcoxon),
        "chaves": chaves,
    }


def formatar_comparacao(res: Dict[str, object], nome_metrica: str = "métrica") -> str:
    """Formata o resultado de `compare_paired`, com os dois testes lado a lado."""
    if "erro" in res:
        return f"[{nome_metrica}] ERRO: {res['erro']}"
    tlo, thi = res["delta_ic95_t"]
    blo, bhi = res["delta_ic95_bootstrap"]

    def marca(ok):
        return "SIM" if ok else "nao"

    aviso = ""
    if not res["testes_concordam"]:
        aviso = ("\n    [ATENCAO] t pareado e Wilcoxon DIVERGEM. Costuma indicar amostra "
                 "pequena\n              ou distribuicao assimetrica. Prefira o Wilcoxon "
                 "e investigue\n              a distribuicao das diferencas antes de "
                 "reportar.")
    return (
        f"[{nome_metrica}] comparacao emparelhada por {res['nivel']} (n={res['n']})\n"
        f"    modelo={res['media_modelo']:.4f}   baseline={res['media_baseline']:.4f}\n"
        f"    ganho medio = {res['delta_medio']:+.4f}   vitorias = "
        f"{100*res['taxa_vitorias']:.0f}%\n"
        f"    IC95% da diferenca:  t={tlo:+.4f}..{thi:+.4f}   "
        f"bootstrap={blo:+.4f}..{bhi:+.4f}\n"
        f"    t pareado (parametrico)     : p={res['p_ttest_pareado']:.4g}  "
        f"significativo={marca(res['significativo_t'])}\n"
        f"    Wilcoxon (nao-parametrico)  : p={res['p_wilcoxon']:.4g}  "
        f"significativo={marca(res['significativo_wilcoxon'])}\n"
        f"    -> VEREDITO: {'SIGNIFICATIVO' if res['significativo'] else 'nao significativo'}"
        f" (exige os dois testes concordando){aviso}")


def resumo_sementes(valores_por_semente: Sequence[float], baseline: Optional[float] = None,
                    maior_melhor: bool = True, nome: str = "métrica") -> str:
    """
    Resume a variação ENTRE SEMENTES com t de Student, e mostra o bootstrap ao lado
    para deixar explícita a diferença entre os dois métodos.
    """
    v = list(valores_por_semente)
    m, lo, hi = mean_ci(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, blo, bhi = bootstrap_ci(v)
    linhas = [f"[{nome}] n={len(v)} sementes: {', '.join(f'{x:.4f}' for x in v)}",
              f"    media={m:.4f}",
              f"    IC95% t de Student = [{lo:.4f}, {hi:.4f}]   <- usar este",
              f"    IC95% bootstrap    = [{blo:.4f}, {bhi:.4f}]   (otimista com n pequeno)"]
    if baseline is not None:
        supera = (lo > baseline) if maior_melhor else (hi < baseline)
        linhas.append(f"    baseline={baseline:.4f} -> "
                      f"{'supera com significancia' if supera else 'NAO estabelecido'} "
                      f"(pelo t)")
    return "\n".join(linhas)
