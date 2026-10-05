"""Testes do refinamento automático da região em foco.

O que precisa ser provado aqui, e por quê:

* **A retenção acha o plano de foco onde a nitidez pura erraria.** É o caso que
  motivou o módulo: folhagem desfocada tem mais energia de alta frequência que parede
  lisa em foco, e um `argmax` de nitidez escolheria a folhagem.
* **Cena lisa devolve `None`, não uma região inventada.** Fallback com outro nome
  continua sendo fallback.
* **Cada ramo da política é exercitado**, e a origem gravada corresponde ao ramo.
* **`was_refined` marca exatamente as amostras que passaram pelo substituto do passo
  manual do paper** — sem isso, "treinar com e sem" não é possível.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                    # noqa: E402

from control.contract import SampleRejected                           # noqa: E402
from qc.focus_region import (                                         # noqa: E402
    DEFAULT_AGREEMENT_FLOOR, DEFAULT_WINDOW_PX, FocusSource, _box_mean, _laplacian_abs, _luma,
    DEFAULT_MIN_DETAIL_PERCENTILE, detail_maps, detail_retention,
    refine_focus_mask, sharpest_region_mask,
)


# --------------------------------------------------------------------------------
# Cenas sintéticas
# --------------------------------------------------------------------------------

def _textura(h: int, w: int, *, escala: int, seed: int, amplitude: float = 1.0):
    """Ruído em blocos — textura com energia de alta frequência controlável."""
    rng = np.random.default_rng(seed)
    base = rng.random((max(-(-h // escala), 1), max(-(-w // escala), 1))).astype(np.float32)
    grande = np.repeat(np.repeat(base, escala, axis=0), escala, axis=1)[:h, :w]
    return (128.0 + amplitude * 100.0 * (grande - 0.5)).astype(np.float32)


def _borra(canal: np.ndarray, raio: int) -> np.ndarray:
    """Borrão de caixa — o que uma lente desfocada faz, em primeira aproximação."""
    if raio <= 0:
        return canal.copy()
    return _box_mean(canal, 2 * raio + 1)


def _bgr(canal: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(canal, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _cena_com_armadilha(h=180, w=240):
    """A cena que quebra qualquer medida de nitidez ABSOLUTA.

    - **Esquerda**: textura fina e forte (folhagem), e ela **borra** — está fora de foco.
    - **Direita**: textura grossa e fraca (parede), e ela **não borra** — está em foco.

    Na bokeh, a esquerda desfocada ainda tem mais detalhe absoluto que a direita nítida.
    Só a razão bokeh/AIF acerta.
    """
    aif = np.zeros((h, w), dtype=np.float32)
    aif[:, :w // 2] = _textura(h, w // 2, escala=4, seed=1, amplitude=1.00)
    aif[:, w // 2:] = _textura(h, w - w // 2, escala=12, seed=2, amplitude=0.08)

    bokeh = aif.copy()
    bokeh[:, :w // 2] = _borra(aif[:, :w // 2], 3)          # esquerda perde detalhe
    return _bgr(aif), _bgr(bokeh)


class TestPrimitivas(unittest.TestCase):

    def test_luma_pondera_os_canais(self):
        px = _luma(np.array([[[0, 0, 255]]], dtype=np.uint8))     # BGR: vermelho puro
        self.assertAlmostEqual(float(px[0, 0]), 0.299 * 255, places=3)

    def test_laplaciano_e_zero_em_area_lisa(self):
        self.assertLess(float(_laplacian_abs(np.full((20, 20), 7.0, np.float32)).max()),
                        1e-5)

    def test_laplaciano_reage_a_borda(self):
        a = np.zeros((20, 20), np.float32)
        a[:, 10:] = 100.0
        self.assertGreater(float(_laplacian_abs(a).max()), 50.0)

    def test_box_mean_de_constante_e_a_constante(self):
        """Inclusive nas bordas: a janela encolhe, não completa com zero."""
        out = _box_mean(np.full((30, 40), 5.0, np.float32), 7)
        np.testing.assert_allclose(out, 5.0, rtol=1e-5)

    def test_box_mean_bate_com_a_media_forca_bruta(self):
        rng = np.random.default_rng(0)
        a = rng.random((23, 29)).astype(np.float32)
        out = _box_mean(a, 5)
        for y, x in ((0, 0), (11, 13), (22, 28), (0, 28)):
            esperado = a[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].mean()
            self.assertAlmostEqual(float(out[y, x]), float(esperado), places=5)

    def test_janela_invalida_e_erro(self):
        with self.assertRaises(ValueError):
            _box_mean(np.zeros((4, 4), np.float32), 0)


class TestRetencao(unittest.TestCase):

    def test_regiao_em_foco_retem_mais_que_a_desfocada(self):
        aif, bokeh = _cena_com_armadilha()
        r = detail_retention(aif, bokeh, window=DEFAULT_WINDOW_PX)
        h, w = r.shape
        margem = 40                                    # ignora a fronteira das metades
        esquerda = np.nanmedian(r[:, margem:w // 2 - margem])
        direita = np.nanmedian(r[:, w // 2 + margem:w - margem])
        self.assertGreater(direita, 0.9, "o lado em foco tinha que reter quase tudo")
        self.assertLess(esquerda, 0.5, "o lado borrado tinha que perder detalhe")

    def test_nitidez_ABSOLUTA_erraria_nesta_cena(self):
        """A justificativa do módulo, como teste.

        Se a energia absoluta do laplaciano na bokeh for MAIOR no lado desfocado, então
        escolher a região por nitidez pura escolheria o lado errado — e a razão é o que
        salva. Se este teste falhar, a cena de teste deixou de ser uma armadilha e os
        outros testes perdem força.
        """
        aif, bokeh = _cena_com_armadilha()
        det = _box_mean(_laplacian_abs(_luma(bokeh)), DEFAULT_WINDOW_PX)
        h, w = det.shape
        margem = 40
        self.assertGreater(float(np.median(det[:, margem:w // 2 - margem])),
                           float(np.median(det[:, w // 2 + margem:w - margem])))

    def test_area_lisa_vira_nan_e_nao_zero(self):
        """"Não dava para saber" é diferente de "borrou"."""
        liso = _bgr(np.full((80, 80), 120.0, np.float32))
        r = detail_retention(liso, liso)
        self.assertTrue(np.isnan(r).all())

    def test_resolucoes_diferentes_rejeitam(self):
        with self.assertRaises(SampleRejected):
            detail_retention(_bgr(np.zeros((10, 10), np.float32)),
                             _bgr(np.zeros((10, 12), np.float32)))

    def test_retencao_fica_em_zero_um(self):
        aif, bokeh = _cena_com_armadilha()
        r = detail_retention(aif, bokeh)
        finitos = r[np.isfinite(r)]
        self.assertGreaterEqual(float(finitos.min()), 0.0)
        self.assertLessEqual(float(finitos.max()), 1.0)


class TestRegiaoMaisNitida(unittest.TestCase):

    def test_seleciona_o_lado_em_foco(self):
        aif, bokeh = _cena_com_armadilha()
        regiao = sharpest_region_mask(*detail_maps(aif, bokeh), top_fraction=0.05)
        self.assertIsNotNone(regiao)
        w = regiao.shape[1]
        na_direita = regiao[:, w // 2:].sum() / max(regiao.sum(), 1)
        self.assertGreater(na_direita, 0.9, "a região tinha que cair no lado em foco")

    def test_cena_lisa_devolve_None(self):
        liso = _bgr(np.full((80, 80), 120.0, np.float32))
        self.assertIsNone(sharpest_region_mask(*detail_maps(liso, liso)))

    def test_fracao_invalida_e_erro(self):
        r = np.full((20, 20), 0.5, np.float32)
        for ruim in (0.0, 1.0, -0.1, 2.0):
            with self.subTest(valor=ruim):
                with self.assertRaises(ValueError):
                    sharpest_region_mask(r, r, top_fraction=ruim)


def _cena_com_ceu(h=200, w=300):
    """O cenário oposto ao da armadilha: **foco na esquerda**, céu liso desfocado.

    É o que quebrou a primeira versão do módulo. Uma superfície lisa que borra continua
    lisa, então `|lap(bokeh)| ≈ |lap(aif)|` e a retenção vale ~1 — o mesmo valor que o
    plano de foco produz. Sem defesa, 42,9% da região ia parar no céu.
    """
    rng = np.random.default_rng(0)
    aif = np.zeros((h, w), dtype=np.float32)
    aif[:, :w // 2] = 128 + 60 * rng.random((h, w // 2)).astype(np.float32)
    aif[:, w // 2:] = np.linspace(100, 140, w - w // 2, dtype=np.float32)[None, :]
    bokeh = aif.copy()
    bokeh[:, w // 2:] = _borra(aif[:, w // 2:], 10)
    return _bgr(aif), _bgr(bokeh)


class TestRegressaoDaSaturacao(unittest.TestCase):
    """Os dois defeitos que uma auditoria adversarial encontrou na primeira versão.

    A retenção satura em 1,0, e a primeira versão selecionava com `>= quantil`. Numa
    cena com céu liso desfocado, **79,7%** dos pixels válidos empatavam no teto — e
    empate faz `>= quantil` selecionar todos eles.
    """

    def test_top_fraction_CONTROLA_a_area(self):
        """Ela era inerte: 0,20 e 0,001 — fator 200 — devolviam a MESMA região."""
        aif, bokeh = _cena_com_ceu()
        r, d = detail_maps(aif, bokeh)
        areas = {}
        for f in (0.20, 0.05, 0.01, 0.002):
            regiao = sharpest_region_mask(r, d, top_fraction=f)
            areas[f] = regiao.mean()
            self.assertAlmostEqual(regiao.mean(), f, delta=0.002,
                                   msg=f"top_fraction={f} não foi respeitada")
        self.assertGreater(areas[0.20], areas[0.002] * 50)

    def test_o_ceu_liso_desfocado_NAO_entra_na_regiao(self):
        """Era 42,9% da região. O céu não tem detalhe a reter: retenção ~1 ali é ruído
        sobre ruído, e não evidência de foco."""
        aif, bokeh = _cena_com_ceu()
        r, d = detail_maps(aif, bokeh)
        regiao = sharpest_region_mask(r, d, top_fraction=0.05)
        w = regiao.shape[1]
        no_ceu = regiao[:, w // 2:].sum() / max(regiao.sum(), 1)
        self.assertLess(no_ceu, 0.02, "a região escorregou para o céu")

    def test_o_desempate_por_detalhe_e_o_que_conserta(self):
        """Medido: mesmo com o piso de textura em 0, o céu é evitado.

        Quem resolve não é o piso — é ordenar por posto e desempatar pelo detalhe da
        AIF. O piso é rede de segurança para quando retenção **e** detalhe empatam.
        """
        aif, bokeh = _cena_com_ceu()
        r, d = detail_maps(aif, bokeh)
        regiao = sharpest_region_mask(r, d, top_fraction=0.05, min_detail_percentile=0.0)
        w = regiao.shape[1]
        self.assertLess(regiao[:, w // 2:].sum() / max(regiao.sum(), 1), 0.02)

    def test_o_piso_alto_QUEBRA_o_caso_legitimo(self):
        """Por que o default é 30 e não 60.

        Na armadilha o lado em foco é o de MENOS textura, e um piso alto o elimina.
        Medido: até 40 os dois cenários acertam 100%; em 50 este cai para 4%, em 60
        para 0%. O default tem que ficar na faixa que serve aos dois.
        """
        aif, bokeh = _cena_com_armadilha()
        r, d = detail_maps(aif, bokeh)
        w = r.shape[1]

        bom = sharpest_region_mask(r, d, top_fraction=0.05,
                                   min_detail_percentile=DEFAULT_MIN_DETAIL_PERCENTILE)
        self.assertGreater(bom[:, w // 2:].sum() / max(bom.sum(), 1), 0.95)

        ruim = sharpest_region_mask(r, d, top_fraction=0.05, min_detail_percentile=60.0)
        self.assertLess(ruim[:, w // 2:].sum() / max(ruim.sum(), 1), 0.10,
                        "se isto passar, a medição que fixou o default mudou")

    def test_o_default_fica_na_faixa_medida_como_segura(self):
        self.assertLessEqual(DEFAULT_MIN_DETAIL_PERCENTILE, 40.0)
        self.assertGreaterEqual(DEFAULT_MIN_DETAIL_PERCENTILE, 0.0)

    def test_percentil_invalido_e_erro(self):
        r = np.full((40, 40), 0.5, np.float32)
        for ruim in (-1.0, 100.0, 150.0):
            with self.subTest(valor=ruim):
                with self.assertRaises(ValueError):
                    sharpest_region_mask(r, r, min_detail_percentile=ruim)

    def test_mapas_de_shapes_diferentes_sao_erro(self):
        with self.assertRaises(ValueError):
            sharpest_region_mask(np.zeros((10, 10), np.float32),
                                 np.zeros((10, 12), np.float32))


class TestPolitica(unittest.TestCase):

    def setUp(self):
        self.aif, self.bokeh = _cena_com_armadilha()
        self.retencao, self.detalhe = detail_maps(self.aif, self.bokeh)
        self.h, self.w = self.retencao.shape

    def _mascara(self, esquerda: bool) -> np.ndarray:
        m = np.zeros((self.h, self.w), dtype=bool)
        if esquerda:
            m[:, : self.w // 2] = True
        else:
            m[:, self.w // 2:] = True
        return m

    def test_mascara_que_concorda_e_MANTIDA(self):
        """O caminho do paper: M confiável não se mexe."""
        out = refine_focus_mask(self._mascara(esquerda=False), self.retencao, self.detalhe)
        self.assertIs(out.source, FocusSource.BIREFNET)
        self.assertFalse(out.was_refined)
        self.assertGreater(out.agreement, 0.9)

    def _retencao_com_pico(self, h=100, w=100, lado=20):
        """Mapa de retenção com um pico NÍTIDO, sem empates.

        A cena sintética de `_cena_com_armadilha` tem retenção exatamente 1,0 em toda a
        metade em foco, então o quantil empata e a região se expande para 43% do quadro
        — correto, mas inútil para exercitar a diferença entre alcance e precisão.
        Aqui o pico ocupa 4% e o resto decai, que é o caso real.
        """
        r = np.linspace(0.0, 0.30, h * w, dtype=np.float32).reshape(h, w)
        r[:lado, :lado] = 0.99
        #: detalhe uniforme: aqui a pergunta é sobre as MÉTRICAS de concordância, não
        #: sobre o piso de textura, que tem testes próprios.
        return r, np.ones_like(r)

    def test_mascara_ENORME_alcanca_tudo_mas_se_denuncia_na_precisao(self):
        """A crítica que motivou `precision` e `iou`.

        Uma máscara cobrindo o quadro inteiro **alcança** toda a região de retenção e
        pontua `agreement = 1,000`, passando pelo ramo "manter" mesmo estando errada em
        quase toda a sua área. `agreement` sozinho não distingue "boa" de "grande".

        Ela continua passando aqui de propósito — quem barra máscara enorme é o gate
        `max_mask_area_ratio`, que mede a coisa certa. O que este teste garante é que o
        caso fica **visível**: `precision` e `iou` desabam.
        """
        r, d = self._retencao_com_pico()
        out = refine_focus_mask(np.ones(r.shape, dtype=bool), r, d)
        self.assertIs(out.source, FocusSource.BIREFNET)
        self.assertAlmostEqual(out.agreement, 1.0, places=6)
        self.assertLess(out.precision, 0.10)
        self.assertLess(out.iou, 0.10)

    def test_mascara_justa_pontua_bem_na_precisao(self):
        """O contraste: máscara do tamanho certo acerta onde importa."""
        r, d = self._retencao_com_pico()
        justa = np.zeros(r.shape, dtype=bool)
        justa[:22, :22] = True                       # o pico, com folga pequena
        out = refine_focus_mask(justa, r, d)
        self.assertGreater(out.agreement, 0.7)
        self.assertGreater(out.precision, 0.7)
        self.assertGreater(out.iou, 0.6)

    def test_a_mascara_ERRADA_pontua_MAIS_ALTO_no_agreement(self):
        """A demonstração mais crua de por que `agreement` sozinho não basta.

        A máscara que cobre o quadro inteiro tem `agreement = 1,000`; a máscara correta,
        justa em volta do pico, tem **0,800**. Ranqueadas por `agreement`, a errada
        ganha — porque alcançar tudo é trivial quando se cobre tudo.

        `precision` inverte o ranking na proporção certa: 0,83 contra 0,05, dezesseis
        vezes. É por isso que as três métricas viajam juntas no metadado.
        """
        r, d = self._retencao_com_pico()
        justa = np.zeros(r.shape, dtype=bool)
        justa[:22, :22] = True
        enorme = refine_focus_mask(np.ones(r.shape, dtype=bool), r, d)
        boa = refine_focus_mask(justa, r, d)

        self.assertGreater(enorme.agreement, boa.agreement,
                           "a errada ganharia no ranking por agreement")
        self.assertGreater(boa.precision, enorme.precision * 10,
                           "e perde de longe na precisão, que é o contrapeso")

    def test_mascara_que_discorda_e_INTERSECTADA(self):
        """M no lado errado: preserva a intenção do BiRefNet, restringe à física."""
        out = refine_focus_mask(self._mascara(esquerda=True), self.retencao, self.detalhe)
        self.assertIn(out.source,
                      (FocusSource.BIREFNET_REFINED, FocusSource.RETENTION_ONLY))
        self.assertTrue(out.was_refined)
        self.assertLess(out.agreement, 0.3)

    def test_mascara_vazia_usa_so_a_retencao_em_vez_de_DESCARTAR(self):
        """Era este o caso que virava `focus_mask_empty` — 20,6% do piloto — e o paper
        diz explicitamente que não descarta."""
        vazia = np.zeros((self.h, self.w), dtype=bool)
        out = refine_focus_mask(vazia, self.retencao, self.detalhe)
        self.assertIs(out.source, FocusSource.RETENTION_ONLY)
        self.assertTrue(out.initial_mask_was_empty)
        self.assertTrue(out.mask.any())

    def test_mascara_None_tambem_funciona(self):
        out = refine_focus_mask(None, self.retencao, self.detalhe)
        self.assertIs(out.source, FocusSource.RETENTION_ONLY)
        self.assertTrue(out.initial_mask_was_empty)

    def test_regiao_refinada_cai_no_lado_em_foco(self):
        """O que importa de verdade: depois do refinamento, a região está em foco."""
        for inicial in (self._mascara(esquerda=True),
                        np.zeros((self.h, self.w), dtype=bool), None):
            with self.subTest(inicial="esquerda" if inicial is not None
                                       and inicial.any() else "vazia"):
                out = refine_focus_mask(inicial, self.retencao, self.detalhe)
                fracao = out.mask[:, self.w // 2:].sum() / max(out.mask.sum(), 1)
                self.assertGreater(fracao, 0.85)

    def test_cena_sem_detalhe_nenhum_REJEITA_com_slug(self):
        """Aqui descartar não é escolha, é consequência: não há plano de foco."""
        liso = _bgr(np.full((80, 80), 120.0, np.float32))
        with self.assertRaises(SampleRejected) as ctx:
            refine_focus_mask(None, *detail_maps(liso, liso))
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")

    def test_was_refined_marca_exatamente_os_dois_ramos_de_refino(self):
        self.assertFalse(refine_focus_mask(self._mascara(False), self.retencao, self.detalhe).was_refined)
        self.assertTrue(refine_focus_mask(None, self.retencao, self.detalhe).was_refined)

    def test_retencao_na_regiao_e_alta_quando_deu_certo(self):
        out = refine_focus_mask(None, self.retencao, self.detalhe)
        self.assertGreater(out.retention_in_region, 0.8)

    def test_area_nao_toma_a_imagem_inteira(self):
        """*small yet reliable*: a região fica contida no plano de foco.

        Nesta cena sintética o lado em foco tem retenção **exatamente 1,0** em todo
        lugar, então o quantil empata e a região se expande para todos os empates. Isso
        é o comportamento certo — pixels igualmente em foco são igualmente elegíveis —,
        e é por isso que a asserção é "não toma o quadro inteiro" e não "top 5%".
        A fração fina é exercitada em `test_fracao_e_respeitada_com_gradiente`.
        """
        out = refine_focus_mask(None, self.retencao, self.detalhe, top_fraction=0.05)
        self.assertGreater(out.area_ratio, 0.0)
        self.assertLess(out.area_ratio, 0.6)

    def test_fracao_e_respeitada_com_gradiente(self):
        """Sem empates, `top_fraction` seleciona de fato a fração pedida."""
        gradiente = np.linspace(0.0, 1.0, 200 * 200, dtype=np.float32).reshape(200, 200)
        regiao = sharpest_region_mask(gradiente, np.ones_like(gradiente), top_fraction=0.05)
        self.assertAlmostEqual(regiao.sum() / gradiente.size, 0.05, delta=0.005)


if __name__ == "__main__":
    unittest.main()
