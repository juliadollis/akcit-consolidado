"""Smoke SEM GPU: enumeracao, carregador, ajuste afim e K fixo.

Pega NameError, import quebrado e contrato de assinatura — que compileall nao pega.
"""
import sys, json
sys.path.insert(0, "/workspace/bokehnet-regen/src")
import numpy as np

from qc.rejection import RejectionLog
from sources.realdof import enumerate_pairs, enumeration_summary, scene_source_splits
from sources.realdof_images import RealDOFImageLoader, expected_cell_path
from sources.mirror_images import MirrorIndex, order_pairs_for_sequential_read, sample_pairs_for_pilot
from dataio import split_from_source
from model_runtime.depth_align import ajusta_afim, marca_sem_alinhamento
from qc.metrics import psnr, ssim
from renderer.calibration import evaluate_k_fixed, calibrate_k

MIRROR = "/workspace/caminho_a_rotac/realdof_mirror"

print("== indice ==")
idx = MirrorIndex.build(MIRROR, force=True)
print("linhas no indice:", len(idx))

log = RejectionLog(None)
pares = enumerate_pairs(idx.names(), log=log)
print(enumeration_summary(pares, log))
print("sample_id[0..2]:", [p.sample_id for p in pares[:3]])
print("scene_id unicos:", len({p.scene_id for p in pares}))
print("campos opticos do par 0:", pares[0].f_number, pares[0].focal_length_mm,
      pares[0].focus_plane_distance_m, pares[0].aif_f_number)

sp = split_from_source(scene_source_splits(pares))
print("split:", sp.counts(), "val_fraction:", sp.val_fraction)

print("== piloto determinista ==")
p5a = sample_pairs_for_pilot(pares, limit=5, seed=0)
p5b = sample_pairs_for_pilot(pares, limit=5, seed=0)
assert [p.sample_id for p in p5a] == [p.sample_id for p in p5b], "piloto nao e determinista"
print("piloto seed 0:", [p.sample_id for p in p5a])

print("== path esperado das celulas ==")
print(expected_cell_path("realdof_10_aligned", "image_focus"))
print(expected_cell_path("realdof_14_shift_4.4px", "image_blur"))

print("== carregador: 1 par de verdade ==")
ordenados = order_pairs_for_sequential_read(p5a, idx)
loader = RealDOFImageLoader(MIRROR, idx)
aif, bok = loader(ordenados[0])
print("aif", aif.shape, aif.dtype, "| bokeh", bok.shape, bok.dtype)
assert aif.shape == bok.shape

print("== ajuste afim (sintetico, resposta conhecida) ==")
rng = np.random.default_rng(0)
x = rng.uniform(0.01, 2.0, 50000)
y = 1.7 * x + 0.3
a, b, r2 = ajusta_afim(x, y)
print(f"a={a:.6f} (esperado 1.700000)  b={b:.6f} (esperado 0.300000)  r2={r2:.8f}")
assert abs(a - 1.7) < 1e-9 and abs(b - 0.3) < 1e-9 and r2 > 1 - 1e-12
y2 = 1.7 * x + 0.3 + rng.normal(0, 0.2, x.size)
a2, b2, r22 = ajusta_afim(x, y2)
print(f"com ruido: a={a2:.4f} b={b2:.4f} r2={r22:.4f}")
print("marca sem alinhamento:", json.dumps(marca_sem_alinhamento("teste", solicitado=False)))

print("== K fixo vs busca, com renderizador de mentira ==")
# Renderizador falso: borra por media movel de raio proporcional a K. Nao e BokehMe;
# so serve para provar que as duas funcoes chamam, medem e devolvem a mesma forma.
def render_falso(aif_bgr, depth_m, focus_disp, k):
    r = max(1, int(round(k / 6.0)))
    out = aif_bgr.astype(np.float64)
    ker = np.ones(2 * r + 1) / (2 * r + 1)
    for eixo in (0, 1):
        out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), eixo, out)
    return out.astype(np.uint8)

pequeno_aif = (rng.integers(0, 255, (48, 64, 3))).astype(np.uint8)
pequeno_alvo = render_falso(pequeno_aif, None, 1.0, 18.0)
prof = np.full((48, 64), 2.0, dtype=np.float32); prof[:, 32:] = 5.0

fixo = evaluate_k_fixed(render_falso, aif_bgr=pequeno_aif, target_bgr=pequeno_alvo,
                        depth_m=prof, focus_disparity=0.5, k_value=18.0,
                        work_long_side=None, report_metrics=True)
print("k fixo  ->", fixo.k_value, f"ssim={fixo.calibration_ssim:.6f}",
      "censurado=", fixo.is_censored, "search=", json.dumps(fixo.search))
assert fixo.k_value == 18.0 and fixo.is_censored is False
assert fixo.search["k_mode"] == "k_fixo" and "calibration_psnr" in fixo.search

busca = calibrate_k(render_falso, aif_bgr=pequeno_aif, target_bgr=pequeno_alvo,
                    depth_m=prof, focus_disparity=0.5, k_min=0.5, k_max=120.0,
                    work_long_side=None, report_metrics=True)
print("busca   ->", f"k={busca.k_value:.3f}", f"ssim={busca.calibration_ssim:.6f}",
      "censurado=", busca.is_censored, "k_mode=", busca.search["k_mode"],
      "psnr=", round(busca.search["calibration_psnr"], 3),
      "delta_ssim=", busca.search["ssim_reconferido_delta"])
assert busca.search["k_mode"] == "busca_eq5"
assert abs(busca.search["ssim_reconferido_delta"]) < 1e-12, "renderizador nao determinista"

sem = calibrate_k(render_falso, aif_bgr=pequeno_aif, target_bgr=pequeno_alvo,
                  depth_m=prof, focus_disparity=0.5, k_min=0.5, k_max=120.0,
                  work_long_side=None)
assert sem.k_value == busca.k_value and sem.calibration_ssim == busca.calibration_ssim, \
    "report_metrics mudou o resultado da busca"
assert "calibration_psnr" not in sem.search
print("report_metrics=False nao muda k nem ssim, e nao adiciona psnr: ok")

print("\nSMOKE OK")
