"""Testes de invariante do contrato canônico.

Cada teste aqui existe porque o defeito correspondente aconteceu de verdade e
atravessou revisão humana. Não são testes de cobertura: são regressões.

Rodar:  PYTHONPATH=src python3 -m pytest tests/ -q
   ou:  PYTHONPATH=src python3 tests/test_contract.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import (  # noqa: E402
    MAX_COC,
    MM_PER_M,
    SampleRejected,
    control_metadata,
    decode_defocus_uint16,
    defocus_map,
    encode_defocus_uint16,
    focus_disparity_from_mask,
    k_at_resolution,
    k_eq3_mm,
    k_for_bokehme,
    k_from_exif,
    k_official,
    pixel_ratio,
    sensor_width_mm,
    signed_coc_px,
    validate_metric_depth,
)


def _scene(h=32, w=48, near=1.0, far=20.0):
    """Rampa de profundidade métrica, do perto ao longe."""
    return np.linspace(near, far, h * w, dtype=np.float32).reshape(h, w)


# ==============================================================================
# D1 — o defeito que apagou o K do mapa gravado
# ==============================================================================

class DefocusPreservesK(unittest.TestCase):
    """`dm / dm.max()` cancelava o K algebricamente. `max(defocus) == 65535` em
    TODAS as amostras, com k de 33 a 195."""

    def test_k_diferente_produz_mapa_diferente(self):
        z = _scene()
        focus_disp = 1.0 / 2.0
        maxima = [defocus_map(z, focus_disp, k).max() for k in (5.0, 15.0, 40.0)]
        self.assertEqual(len(set(np.round(maxima, 6))), 3, "K não influencia o mapa")
        self.assertLess(maxima[0], maxima[1])
        self.assertLess(maxima[1], maxima[2])

    def test_mapa_nao_satura_por_construcao(self):
        """A assinatura do defeito era max == 1.0 sempre. Com K pequeno e max_coc
        global, o mapa tem que ficar bem abaixo de 1."""
        z = _scene()
        self.assertLess(defocus_map(z, 1.0 / 2.0, 5.0).max(), 0.2)

    def test_rota_b_ocupa_faixa_baixa_e_isso_e_correto(self):
        """Âncora medida: CoC p99 mediano da rota B é 4,665 px. Com max_coc=100 o
        mapa vive em [0, ~0.05]. Reescalar para 'ocupar [0,1]' é normalização por
        fonte — o mesmo defeito do D1 com granularidade mais grossa."""
        z = _scene(near=2.0, far=70.0)
        k = 16.55  # mediana da rota B, convenção oficial
        dmap = defocus_map(z, 1.0 / 3.11, k)
        self.assertLess(dmap.max(), 0.15)
        self.assertGreater(dmap.max(), 0.01)


# ==============================================================================
# Eq. 4 — mediana da disparidade, não recíproco da mediana da profundidade
# ==============================================================================

class FocusDisparity(unittest.TestCase):
    def test_difere_do_reciproco_da_mediana_da_profundidade(self):
        """`np.median` faz a MÉDIA dos dois centrais em contagem par, e a média de
        dois recíprocos não é o recíproco da média. Contraexemplo explícito."""
        z = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=np.float32)
        mask = np.ones_like(z)

        focus_disp = focus_disparity_from_mask(z, mask)          # média(1/2, 1/4) = 0.375
        reciproco_da_mediana = 1.0 / float(np.median(z))         # 1/3 = 0.3333...

        self.assertAlmostEqual(focus_disp, 0.375, places=6)
        self.assertNotAlmostEqual(focus_disp, reciproco_da_mediana, places=3)

    def test_coincide_em_contagem_impar(self):
        """Com contagem ímpar a mediana é um elemento, e aí as duas formas batem —
        o que explica por que o defeito é fácil de não ver."""
        z = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
        self.assertAlmostEqual(
            focus_disparity_from_mask(z, np.ones_like(z)), 1.0 / float(np.median(z)), places=6
        )

    def test_mascara_vazia_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            focus_disparity_from_mask(_scene(), np.zeros((32, 48), dtype=np.float32))
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")

    def test_foco_no_teto_do_depth_pro_rejeita(self):
        """z_focus = 10.000 m é sentinela do Depth Pro, não plano de foco."""
        z = np.full((8, 8), 10000.0, dtype=np.float32)
        with self.assertRaises(SampleRejected) as ctx:
            focus_disparity_from_mask(z, np.ones_like(z))
        self.assertEqual(ctx.exception.reason, "focus_depth_implausible")


# ==============================================================================
# Eq. 3 — unidades e identidades algébricas
# ==============================================================================

class Eq3Units(unittest.TestCase):
    def test_ancora_numerica_da_rota_b(self):
        """Âncoras independentes: kfix dá k_value 16,6; a EXIF dá 20,1; o default
        oficial é 15,0. Uma cena típica tem que cair nessa faixa."""
        k, diag = k_from_exif(
            focal_length_mm=49.0, f_number=2.8, focal_length_35mm=49.0,
            focus_depth_m=3.11, image_hw=(1024, 1024),
        )
        self.assertGreater(k, 3.0)
        self.assertLess(k, 60.0)
        self.assertAlmostEqual(diag["crop_factor"], 1.0, places=6)

    def test_conversao_mm_para_metro_e_exatamente_1000(self):
        """O defeito de fator 1000: k_eq3 está em px*mm e multiplica disparidade em
        1/mm; a inferência usa 1/m."""
        self.assertAlmostEqual(k_official(16553.9), 16.5539, places=6)
        self.assertEqual(MM_PER_M, 1000.0)

    def test_razao_de_fstops_e_identidade_exata(self):
        """Com f e D_focus fixos, a Eq. 3 diz que k(F1)/k(F2) == F2/F1 EXATAMENTE.
        É o teste que mede em vez de só detectar. Vale a 1% para K analítico; para
        K recuperado por SSIM, reportar a distribuição do desvio."""
        common = dict(focal_length_mm=49.0, focus_depth_m=0.59, px_per_mm=55.6)
        for f1, f2 in [(2.0, 20.0), (2.8, 5.6), (1.4, 11.0)]:
            k1 = k_eq3_mm(f_number=f1, **common)
            k2 = k_eq3_mm(f_number=f2, **common)
            self.assertAlmostEqual(k1 / k2, f2 / f1, places=9)

    def test_k_linear_no_pixel_ratio(self):
        common = dict(focal_length_mm=49.0, f_number=2.8, focus_depth_m=3.11)
        k1 = k_eq3_mm(px_per_mm=40.0, **common)
        k2 = k_eq3_mm(px_per_mm=80.0, **common)
        self.assertAlmostEqual(k2 / k1, 2.0, places=9)

    def test_coc_final_em_pixels_bate_com_a_analise_dimensional(self):
        """mm^2 * px/mm = px*mm; vezes |Delta(1/mm)| = px. Confere numericamente."""
        f_mm, F, z_focus_m, px_mm = 50.0, 1.8, 2.0, 166.7
        k_mm = k_eq3_mm(f_mm, F, z_focus_m, px_mm)
        z_bg_m = 10.0
        coc_mm_convention = k_mm * abs(1.0 / (z_bg_m * MM_PER_M) - 1.0 / (z_focus_m * MM_PER_M))
        coc_m_convention = k_official(k_mm) * abs(1.0 / z_bg_m - 1.0 / z_focus_m)
        self.assertAlmostEqual(coc_mm_convention, coc_m_convention, places=6)
        self.assertGreater(coc_mm_convention, 20.0)   # f/1.8 a 2 m: dezenas de px
        self.assertLess(coc_mm_convention, 200.0)


class PixelRatio(unittest.TestCase):
    def test_usa_o_maior_lado_nao_a_largura(self):
        """Fig. 16: 'largest edge divided by the physical sensor width'. Paisagem e
        retrato transpostos têm que dar o MESMO pixel_ratio."""
        self.assertEqual(pixel_ratio((1024, 1820), 36.0), pixel_ratio((1820, 1024), 36.0))
        self.assertAlmostEqual(pixel_ratio((1024, 1820), 36.0), 1820 / 36.0, places=9)

    def test_identidade_da_tabela_kfix(self):
        """fx_px = f_mm * pixel_ratio = max(W,H) * focal_length_35 / 36,
        verificada em 900/900 linhas."""
        f_mm, f35, hw = 28.0, 42.0, (1024, 683)
        sensor = sensor_width_mm(f_mm, f35)
        fx_via_pixel_ratio = f_mm * pixel_ratio(hw, sensor)
        fx_via_identidade = max(hw) * f35 / 36.0
        self.assertAlmostEqual(fx_via_pixel_ratio, fx_via_identidade, places=6)


# ==============================================================================
# Sem fallback — cada entrada ausente rejeita com slug próprio
# ==============================================================================

class NoFallback(unittest.TestCase):
    def test_sensor_sem_focal_35_rejeita_em_vez_de_assumir_36mm(self):
        """O pipeline antigo assumia 36 mm. Num celular isso subestima K por 5,6x."""
        for bad in (None, 0.0, -1.0, float("nan")):
            with self.assertRaises(SampleRejected) as ctx:
                sensor_width_mm(4.3, bad)
            self.assertEqual(ctx.exception.reason, "sensor_width_unresolvable")

    def test_focal_ausente_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            sensor_width_mm(None, 50.0)
        self.assertEqual(ctx.exception.reason, "exif_focal_length_missing")

    def test_f_number_ausente_rejeita_em_vez_de_virar_k50(self):
        """O fallback `k = 50.0` produziu 11.635/11.635 amostras com K constante."""
        with self.assertRaises(SampleRejected) as ctx:
            k_eq3_mm(50.0, None, 2.0, 100.0)
        self.assertEqual(ctx.exception.reason, "exif_f_number_missing")

    def test_foco_mais_perto_que_a_focal_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            k_eq3_mm(50.0, 2.8, 0.01, 100.0)   # 10 mm < 50 mm
        self.assertEqual(ctx.exception.reason, "focus_distance_below_focal_length")

    def test_k_nao_positivo_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            defocus_map(_scene(), 0.5, 0.0)
        self.assertEqual(ctx.exception.reason, "k_non_positive")

    def test_max_coc_nao_e_parametro(self):
        """Congelado por CONSTRUÇÃO, não por convenção. Como parâmetro com default,
        um override parcial gravaria `max_coc: 100.0` ao lado de um mapa normalizado
        por outro valor — o mecanismo exato do desastre do kfix."""
        import inspect
        self.assertNotIn("max_coc", inspect.signature(defocus_map).parameters)

    def test_backend_de_profundidade_e_obrigatorio_e_validado(self):
        """Afirmar o backend por omissão é mentir. É o campo que prova que o D11 —
        a cascata que trocava Depth Pro por Depth Anything — não disparou."""
        with self.assertRaises(TypeError):
            validate_metric_depth(_scene())                    # sem backend
        with self.assertRaises(SampleRejected) as ctx:
            validate_metric_depth(_scene(), backend="depth_anything")
        self.assertEqual(ctx.exception.reason, "depth_backend_unregistered")

    def test_slug_nao_registrado_levanta_mesmo_com_python_O(self):
        """`assert` é desligado por `python -O`, e aí o conjunto fechado deixaria de
        ser fechado exatamente no run de produção."""
        from control.contract import reject
        with self.assertRaises(KeyError):
            reject("motivo_inventado")


# ==============================================================================
# Profundidade — o que rejeita e, principalmente, o que NÃO rejeita
# ==============================================================================

class DepthValidation(unittest.TestCase):
    def test_teto_de_10000m_no_fundo_NAO_rejeita(self):
        """Correção explícita: céu/infinito pode legitimamente atingir o teto do
        Depth Pro, e em disparidade 1/10000 ~ 0 é inofensivo. Rejeitar todo
        z_max == 10000 descartaria 25,7% da rota B sem motivo."""
        z = _scene(near=1.0, far=20.0)
        z[0, :] = 10000.0
        depth = validate_metric_depth(z, backend="depth_pro")
        self.assertAlmostEqual(depth.max_m, 10000.0, places=3)
        self.assertLess(depth.disparity_min, 1e-3)

    def test_cena_frontoparalela_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            validate_metric_depth(np.full((16, 16), 5.0, dtype=np.float32), backend="depth_pro")
        self.assertEqual(ctx.exception.reason, "depth_range_degenerate")

    def test_profundidade_negativa_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            validate_metric_depth(np.full((16, 16), -1.0, dtype=np.float32), backend="depth_pro")
        self.assertIn(ctx.exception.reason, {"depth_non_positive", "depth_non_finite"})

    def test_backend_e_gravado(self):
        """O fallback silencioso para Depth Anything devolvia DISPARIDADE, gravada
        igual, deixando a amostra espelhada sem rastro."""
        self.assertEqual(validate_metric_depth(_scene(), backend="depth_pro").backend, "depth_pro")


# ==============================================================================
# Resolução — o fator que ninguém aplicava
# ==============================================================================

class Resolution(unittest.TestCase):
    def test_k_escala_com_o_lado_menor(self):
        """Medidos no dataset antigo: 1024x574 -> 0,892 e 624x1024 -> 0,821."""
        self.assertAlmostEqual(k_at_resolution(100.0, (574, 1024), 512), 100.0 * 512 / 574, places=6)
        self.assertAlmostEqual(k_at_resolution(100.0, (1024, 624), 512), 100.0 * 512 / 624, places=6)

    def test_coc_em_pixel_e_consistente_apos_reescala(self):
        """O mapa pós-crop tem que bater com K reescalado. Sem isso, o modelo recebe
        um mapa que não é o que foi gravado."""
        z = _scene(64, 64, 1.0, 20.0)
        focus_disp, k_full = 1.0 / 3.0, 30.0
        coc_full = np.abs(signed_coc_px(z, focus_disp, k_full)).max()

        escala = 0.5
        k_meio = k_at_resolution(k_full, (64, 64), 32)
        coc_meio = np.abs(signed_coc_px(z, focus_disp, k_meio)).max()
        self.assertAlmostEqual(coc_meio / coc_full, escala, places=6)


# ==============================================================================
# Renderer — o contrato do BokehMe
# ==============================================================================

class BokehMeContract(unittest.TestCase):
    def test_k_renderer_preserva_o_coc_canonico(self):
        """O BokehMe recebe disparidade normalizada em [0,1]. Alimentar
        k_renderer = K * (disp_max - disp_min) faz o raio de borrão coincidir com
        K * Delta_disp na convenção métrica."""
        z = _scene(near=1.0, far=20.0)
        disp = 1.0 / z
        d_min, d_max = float(disp.min()), float(disp.max())
        k, focus_disp = 25.0, 1.0 / 3.0

        coc_metrico = abs(k * (d_max - focus_disp))

        k_rend = k_for_bokehme(k, d_min, d_max)
        disp_norm_max = (d_max - d_min) / (d_max - d_min)
        focus_norm = (focus_disp - d_min) / (d_max - d_min)
        coc_renderer = abs(k_rend * (disp_norm_max - focus_norm))

        self.assertAlmostEqual(coc_metrico, coc_renderer, places=5)


# ==============================================================================
# Codificação e metadados
# ==============================================================================

class Encoding(unittest.TestCase):
    def test_roundtrip_uint16(self):
        d = defocus_map(_scene(), 1.0 / 3.0, 20.0)
        back = decode_defocus_uint16(encode_defocus_uint16(d))
        self.assertLess(np.abs(back - d).max(), 2.0 / 65535)

    def test_encode_nao_normaliza_por_imagem(self):
        """Regressão direta do D1: dois K diferentes -> dois máximos diferentes
        DEPOIS de codificar."""
        z = _scene()
        m1 = encode_defocus_uint16(defocus_map(z, 1.0 / 3.0, 5.0)).max()
        m2 = encode_defocus_uint16(defocus_map(z, 1.0 / 3.0, 30.0)).max()
        self.assertNotEqual(int(m1), int(m2))


class Metadata(unittest.TestCase):
    def test_escalares_obrigatorios_presentes(self):
        depth = validate_metric_depth(_scene(), backend="depth_pro")
        meta = control_metadata(
            k_value=16.55, focus_disparity=1.0 / 3.11, depth=depth,
            k_source="eq3_exif", is_k_censored=False,
        )
        for campo in ("control_version", "k_value", "focus_disparity", "focus_depth_m",
                      "z_min_m", "z_max_m", "max_coc", "depth_backend", "is_k_censored",
                      "k_source"):
            self.assertIn(campo, meta)
        self.assertEqual(meta["max_coc"], MAX_COC)

    def test_focus_depth_e_so_leitura_humana(self):
        """O mapa usa focus_disparity. Reconstruir com 1/focus_depth_m reintroduz a
        diferença que o contrato elimina — o roundtrip é exato aqui, mas a regra é
        de uso, e o teste documenta a intenção."""
        depth = validate_metric_depth(_scene(), backend="depth_pro")
        fd = 1.0 / 3.11
        meta = control_metadata(k_value=16.55, focus_disparity=fd, depth=depth,
                                k_source="eq3_exif", is_k_censored=False)
        self.assertAlmostEqual(1.0 / meta["focus_depth_m"], meta["focus_disparity"], places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
