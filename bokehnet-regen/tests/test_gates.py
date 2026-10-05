"""Testes dos gates. O mais importante é o que confere a hipótese física da máscara."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qc.gates import (                                              # noqa: E402
    GateReport, aif_aperture_is_narrow, aif_sharpness, bokeh_is_blurrier_than_aif,
    calibration_ssim_is_reliable, depth_useful_levels,
    focus_depth_plausible, focus_mask_is_sharpest, focus_region_retention,
    mask_area_ratio, mask_iou, pair_shape_matches,
)


def _textura(h, w, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 4, w // 4, 3), dtype=np.uint16)
    return np.repeat(np.repeat(base, 4, axis=0), 4, axis=1).astype(np.uint8)


def _borrar(img, r):
    """Média em janela — o suficiente para reduzir nitidez de forma controlada."""
    out = img.astype(np.float64).copy()
    for _ in range(r):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + out) / 5.0
    return out.astype(np.uint8)


class ModoMedirPrimeiro(unittest.TestCase):
    def test_sem_limiar_nao_bloqueia(self):
        """A instrução é explícita: medir no piloto, congelar limiar depois."""
        r = aif_sharpness(_textura(64, 64))
        self.assertTrue(r.measured_only)
        self.assertTrue(r.passed)
        self.assertGreater(r.value, 0)

    def test_com_limiar_bloqueia(self):
        r = aif_sharpness(_borrar(_textura(64, 64), 6), min_variance=1e6)
        self.assertFalse(r.measured_only)
        self.assertFalse(r.passed)

    def test_relatorio_agrega(self):
        rep = GateReport()
        rep.add(aif_sharpness(_textura(64, 64)))
        rep.add(mask_iou(np.ones((8, 8)), np.ones((8, 8)), min_iou=0.5))
        self.assertTrue(rep.accepted)
        self.assertEqual(rep.blocked_by, [])
        self.assertIn("aif_laplacian_variance", rep.to_dict())


class MascaraEmFoco(unittest.TestCase):
    """O gate que testa a hipótese que o rótulo inteiro assume."""

    def _cena(self, foco_no_objeto: bool):
        h = w = 128
        nitido, borrado = _textura(h, w, 1), _borrar(_textura(h, w, 1), 5)
        mask = np.zeros((h, w), dtype=bool)
        mask[32:96, 32:96] = True                    # o objeto saliente
        bokeh = np.where(mask[..., None], nitido, borrado) if foco_no_objeto \
            else np.where(mask[..., None], borrado, nitido)
        return bokeh.astype(np.uint8), mask

    def test_foco_no_objeto_da_razao_alta(self):
        bokeh, mask = self._cena(foco_no_objeto=True)
        r = focus_mask_is_sharpest(bokeh, mask)
        self.assertGreater(r.value, 3.0)

    def test_foco_no_fundo_e_PEGO(self):
        """O modo de falha dominante, e o que nenhum gate de IoU pega: o fotógrafo
        focou o fundo, o BiRefNet marcou o objeto saliente, e as duas máscaras
        automáticas CONCORDAM no objeto errado."""
        bokeh, mask = self._cena(foco_no_objeto=False)
        r = focus_mask_is_sharpest(bokeh, mask, min_ratio=1.5)
        self.assertLess(r.value, 1.0)
        self.assertFalse(r.passed)

    def test_iou_alta_NAO_pega_esse_caso(self):
        """Prova de que o gate novo é necessário: com foco no fundo, as duas máscaras
        automáticas são idênticas e a IoU vale 1,0."""
        _, mask = self._cena(foco_no_objeto=False)
        r = mask_iou(mask, mask.copy(), min_iou=0.5)
        self.assertEqual(r.value, 1.0)
        self.assertTrue(r.passed)          # passa, e o dado está errado

    def test_area_insuficiente_devolve_nan_sem_quebrar(self):
        r = focus_mask_is_sharpest(_textura(32, 32), np.zeros((32, 32), dtype=bool))
        self.assertTrue(np.isnan(r.value))


class ParEProfundidade(unittest.TestCase):
    def test_shape_diferente_bloqueia_sempre(self):
        r = pair_shape_matches(_textura(64, 64), _textura(64, 96))
        self.assertFalse(r.passed)
        self.assertFalse(r.measured_only)

    def test_bokeh_tem_que_ser_menos_nitida(self):
        aif = _textura(96, 96, 2)
        r_ok = bokeh_is_blurrier_than_aif(aif, _borrar(aif, 5), max_ratio=0.8)
        self.assertLess(r_ok.value, 0.8)
        self.assertTrue(r_ok.passed)

    def test_pega_aif_quase_tao_borrada_quanto_o_alvo(self):
        """O extremo do D6: as 3,0% de cenas cujo maior f-stop em `gt/` é f/2.8 ou mais
        aberto. Ali AIF e alvo têm nitidez parecida e o sweep casa bokeh contra bokeh."""
        base = _textura(96, 96, 3)
        r = bokeh_is_blurrier_than_aif(_borrar(base, 5), _borrar(base, 5), max_ratio=0.8)
        self.assertGreater(r.value, 0.8)
        self.assertFalse(r.passed)

    def test_par_f14_contra_f2_NAO_e_pego_por_este_gate(self):
        """Documenta o limite: com f/14 contra f/2.0 a razão continua baixa, e ainda
        assim a AIF tem borrão residual. Para esse caso o gate é o do f-stop."""
        base = _textura(96, 96, 3)
        r = bokeh_is_blurrier_than_aif(_borrar(base, 4), _borrar(base, 5), max_ratio=0.8)
        self.assertLess(r.value, 0.8)
        self.assertTrue(r.passed)

    def test_gate_do_fstop_pega_o_D6_direto(self):
        """Com limiar EXPLÍCITO — nenhum gate deste módulo bloqueia por default."""
        from qc.gates import aif_aperture_is_narrow
        self.assertFalse(aif_aperture_is_narrow(5.6, min_f_number=16.0).passed)   # 12,7% das cenas
        self.assertFalse(aif_aperture_is_narrow(14.0, min_f_number=16.0).passed)  # mediana de gt/
        self.assertTrue(aif_aperture_is_narrow(22.0, min_f_number=16.0).passed)   # train/in/, correto

    def test_nenhum_gate_bloqueia_por_default(self):
        """A regra do módulo, travada por teste: limiar não medido não bloqueia.
        Foi assim que um default de 16.0 escapou na primeira versão."""
        from qc.gates import aif_aperture_is_narrow, calibration_ssim_is_reliable
        for r in (aif_aperture_is_narrow(2.0), calibration_ssim_is_reliable(0.1),
                  aif_sharpness(_borrar(_textura(64, 64), 8)),
                  bokeh_is_blurrier_than_aif(_textura(64, 64), _textura(64, 64))):
            self.assertTrue(r.measured_only, f"{r.name} tem limiar por default")
            self.assertTrue(r.passed, f"{r.name} bloqueou sem limiar congelado")

    def test_limiar_de_ssim_da_eq5_existe(self):
        """O passo que o paper AFIRMA executar (§3.2(c) e supp. B.2) e que faltava:
        `calibration_ssim` era gravado e nunca filtrado — o defeito P1-C5."""
        from qc.gates import calibration_ssim_is_reliable
        self.assertFalse(calibration_ssim_is_reliable(0.42, min_ssim=0.60).passed)
        self.assertTrue(calibration_ssim_is_reliable(0.87, min_ssim=0.60).passed)
        self.assertFalse(calibration_ssim_is_reliable(None, min_ssim=0.60).passed)

    def test_niveis_uteis_de_profundidade(self):
        colapsada = np.full((64, 64), 100, dtype=np.uint16)
        rica = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        self.assertEqual(depth_useful_levels(colapsada).value, 1.0)
        self.assertGreater(depth_useful_levels(rica).value, 4000)

    def test_plano_de_foco_no_teto_REPROVA(self):
        """O teste anterior só conferia a substring da nota — e consagrava o defeito:
        o gate calculava o veredito e o jogava fora, com `threshold=None`, então
        `z_focus = 10.000 m` (a sentinela do Depth Pro, 25,7% das amostras medidas)
        passava com `passed:true`."""
        no_teto = focus_depth_plausible(1.0 / 10000.0)
        self.assertFalse(all(r.passed for r in no_teto))
        self.assertFalse(any(r.measured_only for r in no_teto))

        normal = focus_depth_plausible(1.0 / 3.0)
        self.assertTrue(all(r.passed for r in normal))
        self.assertAlmostEqual(normal[0].value, 3.0, places=5)

    def test_limites_vem_do_contrato_nao_sao_copia(self):
        """Eram literais copiados de `control.contract`: mudar o contrato não mudava
        o gate. É a forma 'cópias divergem' que o projeto existe para impedir."""
        import inspect
        from control.contract import FOCUS_DEPTH_MAX_M, FOCUS_DEPTH_MIN_M
        params = inspect.signature(focus_depth_plausible).parameters
        self.assertEqual(params["min_m"].default, FOCUS_DEPTH_MIN_M)
        self.assertEqual(params["max_m"].default, FOCUS_DEPTH_MAX_M)

    def test_iou_de_duas_mascaras_vazias_NAO_e_concordancia(self):
        """Dava 1,0 — o pior caso possível com nota máxima, indistinguível de
        concordância real no relatório."""
        vazia = np.zeros((32, 32), dtype=bool)
        r = mask_iou(vazia, vazia.copy(), min_iou=0.5)
        self.assertTrue(np.isnan(r.value))
        self.assertNotEqual(r.value, 1.0)

    def test_iou_de_duas_vazias_e_INEXISTENTE_e_nao_reprovacao(self):
        """A grandeza não existe, e reprovar aqui devolveria o descarte pela porta
        de trás.

        Enquanto a máscara vazia rejeitava a amostra antes dos gates, este ramo era
        inalcançável na rota C. Com o refinamento de `qc.focus_region` a amostra chega
        até aqui — e um NaN que reprova rejeitaria com `gate_mask_iou` exatamente as
        20,6% de amostras que o refinamento acabou de salvar, contra o que o §3.2(c)
        manda fazer. O sinal fica em `focus_initial_mask_was_empty`.
        """
        vazia = np.zeros((32, 32), dtype=bool)
        r = mask_iou(vazia, vazia.copy(), min_iou=0.5)
        self.assertFalse(r.applicable)
        self.assertTrue(r.passed)
        self.assertFalse(r.to_dict()["applicable"], "o metadado tem que dizer que não "
                                                    "foi medido")

    def test_valor_indefinido_nunca_passa_nem_sem_limiar(self):
        """O princípio continua valendo onde a grandeza EXISTE e deu indefinida.

        O caso certo é o SSIM ausente do sweep da Eq. 5: ele tinha que existir para
        toda amostra da rota C, e não existir é falha medida, não inaplicabilidade.
        """
        r = calibration_ssim_is_reliable(None)
        self.assertTrue(r.applicable)
        self.assertFalse(r.passed)

    def test_area_de_mascara(self):
        mask = np.zeros((100, 100), dtype=bool); mask[:2, :2] = True
        lo, hi = mask_area_ratio(mask, min_ratio=0.002, max_ratio=0.95)
        self.assertFalse(lo.passed)
        self.assertTrue(hi.passed)



class TestInaplicavelNaoEReprovado(unittest.TestCase):
    """`applicable=False` significa "esta grandeza não existe para esta fonte".

    É diferente de "existe e deu ruim". Colapsar as duas em NaN reprovaria 100% do
    LFDOF — que não publica f-number — com o slug `gate_aif_aperture_wide`: motivo
    registrado, histograma bem-comportado, conclusão errada.
    """

    def test_f_number_ausente_nao_bloqueia(self):
        r = aif_aperture_is_narrow(None, min_f_number=16.0)
        self.assertFalse(r.applicable)
        self.assertTrue(r.passed)

    def test_f_number_ausente_nao_bloqueia_nem_sem_limiar(self):
        self.assertTrue(aif_aperture_is_narrow(None).passed)

    def test_f_number_lixo_REPROVA(self):
        """A origem afirmou um número e ele é inválido: medição ruim, não ausência."""
        for ruim in (float("nan"), float("inf"), 0.0, -2.8):
            with self.subTest(valor=ruim):
                r = aif_aperture_is_narrow(ruim, min_f_number=16.0)
                self.assertTrue(r.applicable)
                self.assertFalse(r.passed)

    def test_f_number_aberto_continua_reprovando(self):
        """O D6 em si: AIF de f/2.0 não é all-in-focus."""
        self.assertFalse(aif_aperture_is_narrow(2.0, min_f_number=16.0).passed)

    def test_inaplicavel_aparece_no_metadado(self):
        """"Não medimos" nunca pode se disfarçar de "medimos e passou"."""
        d = aif_aperture_is_narrow(None).to_dict()
        self.assertFalse(d["applicable"])
        self.assertTrue(d["passed"])

    def test_gates_normais_seguem_aplicaveis(self):
        d = aif_aperture_is_narrow(22.0).to_dict()
        self.assertTrue(d["applicable"])


class TestRetencaoDaRegiaoEmFoco(unittest.TestCase):
    """O gate que julga a região REFINADA com a grandeza certa.

    `focus_mask_is_sharpest` mede nitidez ABSOLUTA e cai na armadilha que
    `qc.focus_region` existe para evitar: parede lisa em foco tem pouca alta frequência
    e reprovaria, folhagem desfocada tem muita e passaria. Esta razão é normalizada pela
    textura da própria cena.
    """

    def test_mede_sem_bloquear_por_default(self):
        r = focus_region_retention(0.42)
        self.assertAlmostEqual(r.value, 0.42, places=6)
        self.assertIsNone(r.threshold)
        self.assertTrue(r.measured_only)
        self.assertTrue(r.passed)

    def test_bloqueia_quando_o_limiar_esta_congelado(self):
        self.assertFalse(focus_region_retention(0.20, min_retention=0.50).passed)
        self.assertTrue(focus_region_retention(0.80, min_retention=0.50).passed)

    def test_entra_no_relatorio_com_nome_estavel(self):
        rep = GateReport().add(focus_region_retention(0.9, min_retention=0.5))
        self.assertIn("focus_region_retention", rep.to_dict())

    def test_slug_esta_no_vocabulario_fechado(self):
        """Nome de gate não é slug: sem a ponte, o histograma vira vocabulário aberto."""
        from control.contract import REJECTION_REASONS
        from routes.route_c import GATE_TO_REASON
        slug = GATE_TO_REASON["focus_region_retention"]
        self.assertIn(slug, REJECTION_REASONS)


class TestNitidezNaoMedidaNaoReprova(unittest.TestCase):
    """Área insuficiente após erosão é "não medido", não "medido e ruim".

    O ramo passou a ser comum quando a rota C começou a julgar a região refinada: a
    região de retenção é o topo de um quantil, pode sair pontilhada, e três passos de
    erosão a apagam. Antes disto, cada região pontilhada saía rejeitada com
    `gate_focus_mask_not_sharpest` — um slug afirmando que a máscara está no lugar
    errado sem ter medido nitidez nenhuma.
    """

    def test_mascara_minuscula_e_INAPLICAVEL(self):
        bokeh = _textura(64, 64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[30:33, 30:33] = True                 # some com 3 passos de erosão
        r = focus_mask_is_sharpest(bokeh, mask, min_ratio=1.0)
        self.assertFalse(r.applicable)
        self.assertTrue(r.passed)
        self.assertTrue(np.isnan(r.value))

    def test_mascara_normal_segue_sendo_julgada(self):
        bokeh = _textura(64, 64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:56, 8:56] = True
        r = focus_mask_is_sharpest(bokeh, mask, min_ratio=1.0)
        self.assertTrue(r.applicable)
        self.assertTrue(np.isfinite(r.value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
