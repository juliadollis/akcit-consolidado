"""Testes do harness de verificação do renderer e da Eq. 5.

O renderer real (BokehMe) precisa de GPU e de dois checkpoints, então não roda aqui.
O que roda — e o que importa testar antes — é o **harness**: se ele não sabe medir um
raio conhecido, ele não vai detectar um renderer errado no cluster.

Por isso os renderers sintéticos abaixo vivem no TESTE, não em `src/`. Ter um segundo
renderer no código de produção é justamente o defeito que se quer evitar: o pipeline
antigo tinha um gaussiano de fallback que virou o renderer de verdade.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderer.calibration import _resize_area, calibrate_k                       # noqa: E402
from renderer.verification import (                                # noqa: E402
    DISC_EDGE_RATIO_MAX,
    GAUSSIAN_EDGE_RATIO,
    check_radius_is_linear_in_k,
    check_radius_matches_contract,
    edge_width_ratio,
    measure_blur_radius_px,
    point_light_scene,
    radial_profile,
)


# ==============================================================================
# Renderers sintéticos — só para provar que o harness mede o que diz medir
# ==============================================================================

def _fft_convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    fh, fw = h + kh - 1, w + kw - 1
    out = np.fft.irfft2(np.fft.rfft2(image, (fh, fw)) * np.fft.rfft2(kernel, (fh, fw)), (fh, fw))
    return out[kh // 2: kh // 2 + h, kw // 2: kw // 2 + w]


def _disc_kernel(radius: float) -> np.ndarray:
    r = max(float(radius), 0.5)
    n = int(np.ceil(r)) * 2 + 1
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    d = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    k = np.clip(r + 0.5 - d, 0.0, 1.0)          # anti-aliasing na borda
    return k / k.sum()


def _gaussian_kernel(sigma: float) -> np.ndarray:
    s = max(float(sigma), 0.5)
    n = int(np.ceil(s * 4)) * 2 + 1
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    k = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * s ** 2))
    return k / k.sum()


def _make_render_fn(kernel_fn, *, radius_cap: float | None = None):
    """Renderer de cena com profundidade CONSTANTE: o CoC é o mesmo em todo pixel.

    Devolve **float**, não uint8. Um ponto de 255 espalhado num disco de raio 12 dá
    0,56 por pixel, que `astype(np.uint8)` trunca para zero — a imagem inteira vira
    preto e o harness mede raio 0. Foi o primeiro erro que estes testes pegaram, e
    vale para o adaptador real: quantizar é gravação, não renderização.

    `radius_cap` reproduz o teto de kernel do renderer antigo (51 px de largura, ou
    seja raio ~25), que é o que quebrava a linearidade em K.
    """
    def render(aif_bgr, depth_m, focus_disparity, k_value):
        coc = np.abs(float(k_value) * (1.0 / np.asarray(depth_m, dtype=np.float64) - float(focus_disparity)))
        radius = float(np.median(coc))
        if radius_cap is not None:
            radius = min(radius, radius_cap)
        source = np.asarray(aif_bgr, dtype=np.float64)
        if radius < 0.5:
            return source.copy()
        kernel = kernel_fn(radius)
        planes = [_fft_convolve_same(source[..., c], kernel) for c in range(3)]
        return np.clip(np.stack(planes, axis=-1), 0, 255)
    return render


_DISC = _make_render_fn(_disc_kernel)
_GAUSSIAN = _make_render_fn(_gaussian_kernel)
_DISC_CAPPED = _make_render_fn(_disc_kernel, radius_cap=25.0)   # o teto de 51 px


# ==============================================================================
# Teste 1 — disco, não gaussiana
# ==============================================================================

class DiscNotGaussian(unittest.TestCase):
    def test_disco_tem_borda_dura(self):
        image, depth, center = point_light_scene()
        out = _DISC(image, depth, 1.0 / 2.0, 20.0)
        ratio = edge_width_ratio(radial_profile(out, center))
        self.assertLess(ratio, DISC_EDGE_RATIO_MAX, f"disco com borda larga demais: {ratio:.3f}")

    def test_gaussiana_e_reprovada(self):
        """O discriminador precisa REPROVAR o renderer antigo, senão não serve."""
        image, depth, center = point_light_scene()
        out = _GAUSSIAN(image, depth, 1.0 / 2.0, 20.0)
        ratio = edge_width_ratio(radial_profile(out, center))
        self.assertGreater(ratio, DISC_EDGE_RATIO_MAX, f"gaussiana passou como disco: {ratio:.3f}")

    def test_razao_da_gaussiana_bate_com_a_teoria(self):
        """exp(-r^2/2s^2) dá (r10-r90)/r50 = 1,43, independente de sigma. Se o
        harness mede outra coisa, ele está errado, não o renderer."""
        image, depth, center = point_light_scene(size=401)
        out = _GAUSSIAN(image, depth, 1.0 / 2.0, 40.0)
        ratio = edge_width_ratio(radial_profile(out, center))
        self.assertAlmostEqual(ratio, GAUSSIAN_EDGE_RATIO, delta=0.35)


# ==============================================================================
# Teste 2 — raio == K * |Delta_disp|
# ==============================================================================

class RadiusMatchesContract(unittest.TestCase):
    def test_raio_bate_com_o_contrato(self):
        check = check_radius_matches_contract(_DISC, k_value=30.0, scene_depth_m=10.0, focus_depth_m=2.0)
        self.assertTrue(check.passed(tolerance=0.10),
                        f"esperado {check.expected_px:.2f} px, medido {check.measured_px:.2f} px "
                        f"(erro {100 * check.relative_error:.1f}%)")

    def test_raio_e_linear_em_k(self):
        checks = check_radius_is_linear_in_k(_DISC, k_values=(10.0, 20.0, 40.0))
        medidos = [c.measured_px for c in checks]
        self.assertAlmostEqual(medidos[1] / medidos[0], 2.0, delta=0.15)
        self.assertAlmostEqual(medidos[2] / medidos[1], 2.0, delta=0.15)

    def test_teto_de_kernel_quebra_a_linearidade_e_o_harness_ve(self):
        """A regressão do D4/D8. Com teto de raio 25 px, K alto para de borrar — e é
        assim que 47% da rota C foi parar em k == 300 exato."""
        checks = check_radius_is_linear_in_k(_DISC_CAPPED, k_values=(10.0, 40.0, 160.0))
        self.assertTrue(checks[0].passed(tolerance=0.10), "o caso baixo deveria passar")
        self.assertFalse(checks[-1].passed(tolerance=0.10), "o harness não viu a saturação")
        self.assertLess(checks[-1].measured_px, checks[-1].expected_px)

    def test_medidor_de_raio_e_calibrado(self):
        """Prova direta de que `measure_blur_radius_px` devolve o raio, e não outra
        estatística proporcional a ele."""
        image, depth, center = point_light_scene(size=401)
        for raio_alvo in (5.0, 12.0, 25.0):
            k = raio_alvo / abs(1.0 / 10.0 - 1.0 / 2.0)
            out = _DISC(image, depth, 1.0 / 2.0, k)
            medido = measure_blur_radius_px(out, center)
            self.assertAlmostEqual(medido, raio_alvo, delta=max(1.0, 0.10 * raio_alvo))


# ==============================================================================
# Eq. 5 — a calibração recupera o K que gerou o alvo
# ==============================================================================

def _textured_scene(size=96, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size // 8, size // 8, 3), dtype=np.uint16)
    image = np.repeat(np.repeat(base, 8, axis=0), 8, axis=1).astype(np.uint8)
    depth = np.full((size, size), 10.0, dtype=np.float32)
    return image, depth


class Eq5Calibration(unittest.TestCase):
    def test_recupera_o_k_que_gerou_o_alvo(self):
        aif, depth = _textured_scene()
        focus_disp, k_verdadeiro = 1.0 / 2.0, 20.0
        alvo = _DISC(aif, depth, focus_disp, k_verdadeiro)

        cal = calibrate_k(_DISC, aif_bgr=aif, target_bgr=alvo, depth_m=depth,
                          focus_disparity=focus_disp, work_long_side=None)

        self.assertAlmostEqual(cal.k_value, k_verdadeiro, delta=2.5)
        self.assertGreater(cal.calibration_ssim, 0.95)
        self.assertFalse(cal.is_censored)

    def test_orcamento_de_avaliacoes_e_baixo(self):
        """O ponto do redesenho: o grid antigo gastava ~40 subprocessos por amostra."""
        aif, depth = _textured_scene()
        alvo = _DISC(aif, depth, 1.0 / 2.0, 20.0)
        cal = calibrate_k(_DISC, aif_bgr=aif, target_bgr=alvo, depth_m=depth,
                          focus_disparity=1.0 / 2.0, work_long_side=None)
        self.assertLessEqual(cal.search["evaluations"], 25)

    def test_marca_censura_quando_o_otimo_encosta_no_teto(self):
        """O alvo pede um K muito acima do teto absoluto: tem que sair censurado, e
        não gravado como se fosse medida exata — foi o `k == 300` da rota C."""
        aif, depth = _textured_scene()
        focus_disp = 1.0 / 2.0
        alvo = _DISC(aif, depth, focus_disp, 400.0)
        cal = calibrate_k(_DISC, aif_bgr=aif, target_bgr=alvo, depth_m=depth,
                          focus_disparity=focus_disp, k_min=0.5, k_max=8.0,
                          k_absolute_max=10.0, work_long_side=None)
        self.assertTrue(cal.is_censored)
        self.assertTrue(cal.search["at_upper_bound"])

    def test_expande_o_teto_antes_de_censurar(self):
        """Teto inicial baixo mas absoluto alto: a busca expande e NÃO censura."""
        aif, depth = _textured_scene()
        focus_disp = 1.0 / 2.0
        alvo = _DISC(aif, depth, focus_disp, 24.0)
        cal = calibrate_k(_DISC, aif_bgr=aif, target_bgr=alvo, depth_m=depth,
                          focus_disparity=focus_disp, k_min=0.5, k_max=10.0,
                          k_absolute_max=200.0, work_long_side=None)
        self.assertGreater(cal.search["expansions"], 0)
        self.assertFalse(cal.is_censored)
        self.assertAlmostEqual(cal.k_value, 24.0, delta=3.0)

    def test_resolucao_de_trabalho_nao_desloca_o_k(self):
        """A calibração roda reduzida por custo. Se o K devolvido não voltar para a
        escala original, é o mesmo defeito do fator do crop de treino."""
        aif, depth = _textured_scene(size=256, seed=3)
        focus_disp, k_verdadeiro = 1.0 / 2.0, 18.0
        alvo = _DISC(aif, depth, focus_disp, k_verdadeiro)

        cheio = calibrate_k(_DISC, aif_bgr=aif, target_bgr=alvo, depth_m=depth,
                            focus_disparity=focus_disp, work_long_side=None)
        reduzido = calibrate_k(_DISC, aif_bgr=aif, target_bgr=alvo, depth_m=depth,
                               focus_disparity=focus_disp, work_long_side=128)

        self.assertAlmostEqual(reduzido.search["work_scale"], 0.5, places=6)
        self.assertAlmostEqual(reduzido.k_value, cheio.k_value, delta=4.0)



class TestReducaoDeArea(unittest.TestCase):
    """`_resize_area` tem que ser média de área DE VERDADE.

    A versão anterior usava `floor`/`ceil` com peso 1 por pixel tocado: em 2000 -> 512,
    **496 de 511** blocos compartilhavam uma linha com o vizinho e a largura efetiva
    oscilava entre 4 e 5 em vez de 3,906. Erro medido: **3,3%** contra a média de área.

    O `argmax` de SSIM(K) **não** se deslocava por causa disso — AIF e alvo passam pela
    mesma redução e o viés cancela (medido: K* idêntico nas duas implementações). O
    conserto vale por estar certo, e este teste existe para que não regrida.
    """

    @staticmethod
    def _referencia(src, dst_hw):
        """Média de área exata, força bruta, com pesos fracionários nas bordas."""
        H, W = src.shape
        dh, dw = dst_hw
        ry = np.linspace(0, H, dh + 1)
        rx = np.linspace(0, W, dw + 1)
        out = np.zeros(dst_hw, dtype=np.float64)
        for i in range(dh):
            for j in range(dw):
                total = area = 0.0
                for y in range(int(np.floor(ry[i])), int(np.ceil(ry[i + 1]))):
                    fy = min(ry[i + 1], y + 1) - max(ry[i], y)
                    if fy <= 0:
                        continue
                    for x in range(int(np.floor(rx[j])), int(np.ceil(rx[j + 1]))):
                        fx = min(rx[j + 1], x + 1) - max(rx[j], x)
                        if fx <= 0:
                            continue
                        total += src[y, x] * fy * fx
                        area += fy * fx
                out[i, j] = total / area
        return out

    def test_bate_com_a_media_de_area_exata(self):
        rng = np.random.default_rng(0)
        src = (rng.random((60, 80)) * 255).astype(np.float64)
        np.testing.assert_allclose(_resize_area(src, (15, 20)),
                                   self._referencia(src, (15, 20)), atol=1e-9)

    def test_escala_nao_inteira(self):
        """3,906 pixels por bloco é o caso real: 2000 -> 512."""
        rng = np.random.default_rng(1)
        src = (rng.random((47, 61)) * 255).astype(np.float64)
        np.testing.assert_allclose(_resize_area(src, (13, 17)),
                                   self._referencia(src, (13, 17)), atol=1e-9)

    def test_preserva_constante(self):
        """Nenhum pixel pode ser contado duas vezes: a média de uma constante é ela."""
        out = _resize_area(np.full((100, 200), 7.0), (25, 50))
        np.testing.assert_allclose(out, 7.0, atol=1e-12)

    def test_preserva_a_media_global(self):
        rng = np.random.default_rng(2)
        src = rng.random((120, 160)).astype(np.float64)
        self.assertAlmostEqual(float(_resize_area(src, (30, 40)).mean()),
                               float(src.mean()), places=6)

    def test_colorida_e_cinza(self):
        rng = np.random.default_rng(3)
        cor = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
        self.assertEqual(_resize_area(cor, (15, 20)).shape, (15, 20, 3))
        self.assertEqual(_resize_area(cor[..., 0], (15, 20)).shape, (15, 20))

    def test_dtype_de_entrada_e_preservado(self):
        cor = np.zeros((60, 80, 3), dtype=np.uint8)
        self.assertEqual(_resize_area(cor, (15, 20)).dtype, np.uint8)

    def test_mesma_resolucao_e_identidade(self):
        rng = np.random.default_rng(4)
        src = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        np.testing.assert_array_equal(_resize_area(src, (32, 32)), src)

    def test_uint8_nao_estoura(self):
        cheio = np.full((40, 40, 3), 255, dtype=np.uint8)
        out = _resize_area(cheio, (10, 10))
        self.assertEqual(int(out.max()), 255)
        self.assertEqual(int(out.min()), 255)


if __name__ == "__main__":
    unittest.main(verbosity=2)