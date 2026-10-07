"""
riemann/ablation_configs.py
===========================
Gerador do espaço de configurações da ablação EXPANDIDA.

Como o fine-tuning roda numa A100, exploramos um espaço rico de combinações de termos —
muito além do estudo anterior — para mapear com precisão qual geometria importa no regime
DepthPro + alta qualidade.

Blocos gerados:
  B0  Baseline            : berHu puro (piso).
  B1  Marginais isolados  : berHu + cada termo geométrico sozinho (5 configs).
  B2  Pares geométricos   : berHu + todas as combinações de 2 termos (10 configs).
  B3  Trios geométricos   : berHu + todas as combinações de 3 termos (10 configs).
  B4  Quádruplos          : berHu + combinações de 4 termos (5 configs).
  B5  Full stack          : todos os termos.
  B6  Sem berHu           : variantes puramente geométricas (controle) — 3 configs.
  B7  Foco em borda       : configs com peso alto em gauss/normal (hipótese central) — 4.

Total ~ 39 configurações. Cada uma pode rodar nas duas variantes de modelo
(heads / heads_lora) → ~78 experimentos. Ideal para paralelizar/varrer numa A100.
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, List

GEOM_TERMS = ["grad", "normal", "gauss", "geod", "metric"]

# Pesos-base (calibrados a partir do trabalho anterior; a busca Optuna refina)
BASE_W = {
    "berhu": 0.7,
    "grad": 0.3,
    "normal": 0.9,
    "gauss": 0.45,
    "geod": 0.1,
    "metric": 0.2,
}


def _cfg(name: str, active_terms: List[str], berhu: bool = True,
         overrides: Dict[str, float] = None) -> Dict:
    w = {"berhu": BASE_W["berhu"] if berhu else 0.0,
         "grad": 0.0, "normal": 0.0, "gauss": 0.0, "geod": 0.0, "metric": 0.0}
    for t in active_terms:
        w[t] = BASE_W[t]
    if overrides:
        w.update(overrides)
    return {"name": name, "weights": w}


def build_ablation_configs() -> List[Dict]:
    configs: List[Dict] = []

    # B0 — baseline berHu puro
    configs.append(_cfg("B0_berhu", []))

    # B1 — marginais isolados
    for t in GEOM_TERMS:
        configs.append(_cfg(f"B1_berhu+{t}", [t]))

    # B2 — pares geométricos
    for a, b in combinations(GEOM_TERMS, 2):
        configs.append(_cfg(f"B2_berhu+{a}+{b}", [a, b]))

    # B3 — trios geométricos
    for combo in combinations(GEOM_TERMS, 3):
        configs.append(_cfg("B3_berhu+" + "+".join(combo), list(combo)))

    # B4 — quádruplos
    for combo in combinations(GEOM_TERMS, 4):
        configs.append(_cfg("B4_berhu+" + "+".join(combo), list(combo)))

    # B5 — full stack
    configs.append(_cfg("B5_full", GEOM_TERMS))

    # B6 — sem berHu (controle puramente geométrico)
    configs.append(_cfg("B6_nogeo_gauss_normal", ["gauss", "normal"], berhu=False))
    configs.append(_cfg("B6_nogeo_gauss_only", ["gauss"], berhu=False))
    configs.append(_cfg("B6_nogeo_full", GEOM_TERMS, berhu=False))

    # B7 — foco em borda (hipótese central: 2ª ordem domina)
    configs.append(_cfg("B7_gaussheavy", ["gauss", "normal"],
                        overrides={"gauss": 0.9, "normal": 0.9}))
    configs.append(_cfg("B7_gauss_dom", ["gauss", "normal", "grad"],
                        overrides={"gauss": 1.0, "normal": 0.6, "grad": 0.2}))
    configs.append(_cfg("B7_normal_dom", ["gauss", "normal"],
                        overrides={"gauss": 0.4, "normal": 1.0}))
    configs.append(_cfg("B7_champion_prev", ["normal", "gauss"],
                        overrides={"berhu": 0.7, "normal": 0.9, "gauss": 0.45}))

    return configs


if __name__ == "__main__":
    cfgs = build_ablation_configs()
    print(f"Total de configurações: {len(cfgs)}\n")
    for c in cfgs:
        active = [k for k, v in c["weights"].items() if v > 0]
        print(f"  {c['name']:38s} -> {', '.join(active)}")
