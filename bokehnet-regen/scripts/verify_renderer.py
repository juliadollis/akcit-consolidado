#!/usr/bin/env python3
"""Os três testes do `renderer-verifier` contra o BokehMe DE VERDADE.

É a única parte do renderer que não dá para verificar sem GPU, e é ela que autoriza
o BokehMe como renderer de rótulo final. Enquanto não passar, `is_final_label_renderer`
não vale nada — é declaração, não medição.

    python3 scripts/verify_renderer.py --bokehme-dir third_party/BokehMe \
        --output-json output/renderer_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderer.bokehme import BokehMeConfig, BokehMeRenderer          # noqa: E402
from renderer.verification import (                                   # noqa: E402
    DISC_EDGE_RATIO_MAX,
    GAUSSIAN_EDGE_RATIO,
    check_radius_is_linear_in_k,
    edge_width_ratio,
    fit_radius_response,
    point_light_scene,
    radial_profile,
)


def _jsonable(obj):
    """numpy.bool_ e numpy.float64 não são serializáveis por json.dumps."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica o BokehMe contra o contrato.")
    parser.add_argument("--bokehme-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--scene-size", type=int, default=257)
    parser.add_argument("--slope-tolerance", type=float, default=0.05,
                        help="quanto o slope pode se afastar de 1,0. É a escala efetiva de K.")
    args = parser.parse_args()

    report: dict = {"passed": {}, "measurements": {}}

    _banner("carregando o BokehMe (in-process, modelos uma vez)")
    renderer = BokehMeRenderer(args.bokehme_dir, config=BokehMeConfig(), device=args.device)
    provenance = renderer.provenance()
    report["provenance"] = provenance
    print(f"  commit               : {provenance['renderer_commit']}")
    print(f"  demo.pipeline sha256 : {provenance['demo_pipeline_sha256'][:16]}...")
    print(f"  arnet.pth sha256     : {provenance['arnet_sha256'][:16]}...")
    print(f"  iunet.pth sha256     : {provenance['iunet_sha256'][:16]}...")
    print(f"  saída usada          : {provenance['config']['output']}")

    # -- TESTE 1: disco, não gaussiana -----------------------------------------
    _banner("TESTE 1 — disco, não gaussiana")
    image, depth, center = point_light_scene(size=args.scene_size, depth_m=10.0)
    rendered = renderer(image, depth, 1.0 / 2.0, 30.0)
    profile = radial_profile(rendered, center)
    ratio = edge_width_ratio(profile)
    is_disc = ratio < DISC_EDGE_RATIO_MAX
    report["measurements"]["edge_width_ratio"] = float(ratio)
    report["passed"]["disco_nao_gaussiana"] = bool(is_disc)
    print(f"  edge_width_ratio = {ratio:.3f}")
    print(f"    disco    : < {DISC_EDGE_RATIO_MAX}")
    print(f"    gaussiana: ~ {GAUSSIAN_EDGE_RATIO}")
    print(f"  -> {'DISCO (passou)' if is_disc else 'NÃO É DISCO (reprovou)'}")

    # -- TESTE 2: raio == K * |Delta_disp| --------------------------------------
    _banner("TESTE 2 — raio == K * |Delta_disp|  (o que amarra o renderer à Eq. 2)")
    print("  Julgado pelo AJUSTE LINEAR, não por erro relativo ponto a ponto: o medidor")
    print("  tem piso aditivo de ~1 px (fonte pontual discretizada), então em raio pequeno")
    print("  o erro relativo estoura mesmo com resposta perfeitamente linear.")
    print("  `slope` é a ESCALA EFETIVA de K; `residuo` é a LINEARIDADE.\n")

    for saida in ("bokeh_pred", "bokeh_classical", "bokeh_neural"):
        r = BokehMeRenderer(args.bokehme_dir, config=BokehMeConfig(output=saida), device=args.device)
        checks = check_radius_is_linear_in_k(
            r, k_values=(8.0, 16.0, 32.0, 64.0, 96.0),
            scene_depth_m=10.0, focus_depth_m=2.0, size=args.scene_size,
        )
        resp = fit_radius_response(checks)
        print(f"  --- {saida} ---")
        for c in checks:
            print(f"      K={c.k_value:6.1f}  esperado {c.expected_px:7.2f} px  medido {c.measured_px:7.2f} px")
        print(f"      slope    = {resp.slope:.4f}   (contrato: 1,000 +- {args.slope_tolerance})")
        print(f"      offset   = {resp.intercept_px:+.3f} px")
        print(f"      residuo  = {resp.max_residual_px:.4f} px = {100*resp.relative_residual:.2f}% do maior raio  (linear se <= 2%)")
        report["measurements"][f"radius_response_{saida}"] = resp.to_dict()
        if saida == provenance["config"]["output"]:
            report["passed"]["linear_em_k"] = bool(resp.is_linear())
            report["passed"]["escala_bate_com_contrato"] = bool(
                resp.scale_matches_contract(args.slope_tolerance))
            report["measurements"]["k_effective_factor"] = float(resp.slope)
        print()

    # -- TESTE 3: highlight ------------------------------------------------------
    _banner("TESTE 3 — highlight")
    demo_src = (Path(args.bokehme_dir) / "demo.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(demo_src)
    pipeline_src = next(
        ast.get_source_segment(demo_src, n) for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "pipeline"
    )
    usa_highlight = "highlight" in pipeline_src
    report["measurements"]["pipeline_usa_highlight"] = bool(usa_highlight)
    report["passed"]["highlight_decidido"] = True
    if usa_highlight:
        print("  `pipeline` LÊ args.highlight — a decisão importa e está congelada em "
              f"BokehMeConfig.highlight = {provenance['config']['highlight']}")
    else:
        print("  `pipeline` NÃO lê args.highlight: o realce acontece no corpo do demo,")
        print("  ANTES da chamada, alterando a imagem de entrada. No nosso caminho a")
        print("  flag é inerte — o que é uma decisão registrada, não um esquecimento.")

    # -- veredito -----------------------------------------------------------------
    _banner("VEREDITO")
    for nome, ok in report["passed"].items():
        print(f"  {'PASSOU ' if ok else 'FALHOU '}  {nome}")
    apto = all(report["passed"].values())
    report["is_final_label_renderer"] = bool(apto)
    print()
    print("  -> APTO A RÓTULO FINAL" if apto else "  -> NÃO APTO. Não gerar dado com este renderer.")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  relatório: {out}")
    return 0 if apto else 1


if __name__ == "__main__":
    raise SystemExit(main())
