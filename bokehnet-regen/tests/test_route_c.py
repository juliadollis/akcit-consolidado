"""Rota C de ponta a ponta, com modelos falsos e renderer sintético.

O que este arquivo prova, sem GPU: que o pipeline INTEIRO — profundidade, máscara,
Eq. 4, Eq. 5, gates, escrita, histograma e retomada — produz uma amostra cujo K
sobrevive ao disco e cuja proveniência não mente.

Os dublês vivem aqui, nunca em `src/`. Um segundo renderer ou um segundo backend de
profundidade em produção é exatamente o defeito D4/D11.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from control.contract import MAX_COC, SampleRejected, validate_metric_depth  # noqa: E402
from dataio import (                                                         # noqa: E402
    FOCUS_SOURCE_TO_MASK_SOURCE, MaskSource, build_scene_split, iter_manifest,
    read_metadata,
)
from qc.focus_region import FocusSource                                      # noqa: E402
from routes.route_c import (                                                 # noqa: E402
    RouteCConfig, RouteCStats, process_pair, refine_focus_region, run_route_c,
)
from test_renderer import _disc_kernel, _fft_convolve_same                    # noqa: E402


# ==============================================================================
# Dublês — vivem no teste, nunca em src/
# ==============================================================================

@dataclass
class FakePair:
    scene_id: str
    sample_id: str
    source_dataset: str = "akcit-pixel/RealBokeh"
    source_sample_id: str = "x_level_3"
    source_split: Optional[str] = "train"
    aif_ref: Optional[str] = "image_focus"
    bokeh_ref: Optional[str] = "image_blur"
    f_number: Optional[float] = 2.0
    focal_length_mm: Optional[float] = 49.0
    focus_plane_distance_m: Optional[float] = 3.0
    aif_f_number: Optional[float] = 22.0
    sensor_width_mm: Optional[float] = 36.0


class FakeDepth:
    """Profundidade determinística: gradiente de 1 a 40 m, com um objeto a 3 m."""

    backend = "depth_pro"

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        z = np.linspace(1.0, 40.0, h * w, dtype=np.float32).reshape(h, w)
        z[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 3.0
        return validate_metric_depth(z, backend="depth_pro")


class FakeMask:
    """Máscara sobre o objeto — bate com a região de 3 m da `FakeDepth`."""

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        return mask


class MascaraVazia:
    """O BiRefNet declinando — probabilidade 0 em toda a imagem.

    Medido em 12 cenas da RealBokeh (job 32231): `prob_max == 0,000` exato, e baixar o
    limiar de 0,50 para 0,05 não recupera nada. Era este caso que produzia
    `focus_mask_empty` em 20,6% do piloto.
    """

    def infer(self, image_rgb):
        return np.zeros(image_rgb.shape[:2], dtype=bool)


class MascaraNoPlanoErrado:
    """Máscara sobre o FUNDO, longe do plano de foco — o modo de falha dominante.

    É o caso dos 64,8% de desacordo medidos: o BiRefNet acha um objeto saliente, e o
    fotógrafo focou outra coisa.
    """

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 8, :] = True                 # topo do gradiente de profundidade
        return mask


#: Raios de disco, em pixel, das camadas do renderer sintético abaixo.
_NIVEIS_COC_PX = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _LAYERED(aif_bgr, depth_m, focus_disparity, k_value):
    """Bokeh que VARIA no espaço: cada pixel borra segundo o CoC do contrato.

    Este dublê existe porque o `_DISC` de `test_renderer` **não serve** para a rota C
    depois do refinamento da região em foco. O `_DISC` borra a imagem inteira com um
    raio único, `median(coc)` — é o dublê certo para medir linearidade em K, e é o
    errado aqui: numa cena uniformemente borrada **não existe plano de foco**, a
    retenção de detalhe é igual em todo lugar, e a região que `qc.focus_region` devolve
    é o topo de um empate numérico. Medido: com `_DISC`, `focus_source` saía
    `retention_only` com retenção de 0,21 e a região caía fora da máscara.

    Aqui a região a 3 m fica de fato mais nítida que o resto, que é a hipótese física
    que o refinamento usa — e é a hipótese que o BokehMe real satisfaz.

    Continua sendo dublê de teste: composição de 7 camadas com interpolação linear, não
    o *scatter* do BokehMe. O que ele precisa reproduzir é a monotonicidade do borrão em
    `|K·Δdisp|`, e é isso que ele reproduz.
    """
    disp = 1.0 / np.asarray(depth_m, dtype=np.float64)
    coc = np.abs(float(k_value) * (disp - float(focus_disparity)))
    origem = np.asarray(aif_bgr, dtype=np.float64)

    versoes = [origem]
    for raio in _NIVEIS_COC_PX[1:]:
        versoes.append(np.stack(
            [_fft_convolve_same(origem[..., c], _disc_kernel(raio)) for c in range(3)],
            axis=-1))

    out = versoes[-1].copy()                     # acima do último nível, satura
    for i in range(len(_NIVEIS_COC_PX) - 1):
        lo, hi = _NIVEIS_COC_PX[i], _NIVEIS_COC_PX[i + 1]
        faixa = (coc >= lo) & (coc < hi)
        if not faixa.any():
            continue
        peso = ((coc - lo) / (hi - lo))[..., None]
        out[faixa] = ((1.0 - peso) * versoes[i] + peso * versoes[i + 1])[faixa]
    return np.clip(out, 0.0, 255.0)


def _scene_images(size=96, k_verdadeiro=18.0, seed=0):
    """AIF texturizada e a bokeh renderizada com um K conhecido."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size // 8, size // 8, 3), dtype=np.uint16)
    aif = np.repeat(np.repeat(base, 8, axis=0), 8, axis=1).astype(np.uint8)
    depth = FakeDepth().infer(aif).values_m
    focus_disp = 1.0 / 3.0
    bokeh = _LAYERED(aif, depth, focus_disp, k_verdadeiro).astype(np.uint8)
    return aif, bokeh


def _cena_lisa(size=96):
    """Cinza uniforme nas duas imagens: não há detalhe em lugar nenhum.

    É o único caso em que descartar não é escolha, é consequência — não existe plano de
    foco a extrair.
    """
    liso = np.full((size, size, 3), 120, dtype=np.uint8)
    return liso, liso.copy()


_PROV = {
    "pipeline_commit": "abc1234",
    "depth_model_sha256": "d" * 64,
    "mask_model_sha256": "m" * 64,
    "mask_backend": "birefnet",
    "renderer": {"renderer": "bokehme_public", "renderer_commit": "8b3ed556"},
    "k_effective_factor": 0.9873,
}


def _config(tmp, **kw):
    return RouteCConfig(output_dir=Path(tmp), calibration_long_side=None, **kw)


# ==============================================================================
# Uma amostra, ponta a ponta
# ==============================================================================

class UmaAmostra(unittest.TestCase):
    def test_recupera_o_K_que_gerou_a_bokeh(self):
        """A prova de que a cadeia inteira fecha: Eq. 4 -> Eq. 5 -> rótulo."""
        aif, bokeh = _scene_images(k_verdadeiro=18.0)
        with tempfile.TemporaryDirectory() as tmp:
            sample = process_pair(
                FakePair("cena1", "c_0001"), aif_bgr=aif, bokeh_bgr=bokeh,
                depth_runtime=FakeDepth(), mask_runtime=FakeMask(),
                render_fn=_LAYERED, config=_config(tmp), provenance_base=_PROV)
        self.assertAlmostEqual(sample.control.k_value, 18.0, delta=2.5)
        self.assertGreater(sample.control.calibration_ssim, 0.9)
        self.assertFalse(sample.control.is_k_censored)

    def test_foco_vem_da_mascara_pela_Eq4(self):
        """O objeto está a 3 m e a máscara está sobre ele: focus_disparity ~ 1/3."""
        aif, bokeh = _scene_images()
        with tempfile.TemporaryDirectory() as tmp:
            sample = process_pair(
                FakePair("cena1", "c_0001"), aif_bgr=aif, bokeh_bgr=bokeh,
                depth_runtime=FakeDepth(), mask_runtime=FakeMask(),
                render_fn=_LAYERED, config=_config(tmp), provenance_base=_PROV)
        self.assertAlmostEqual(sample.control.focus_disparity, 1.0 / 3.0, places=5)
        self.assertAlmostEqual(sample.control.focus_depth_m, 3.0, places=4)

    def test_proveniencia_completa_e_verdadeira(self):
        aif, bokeh = _scene_images()
        with tempfile.TemporaryDirectory() as tmp:
            meta = process_pair(
                FakePair("cena1", "c_0001"), aif_bgr=aif, bokeh_bgr=bokeh,
                depth_runtime=FakeDepth(), mask_runtime=FakeMask(),
                render_fn=_LAYERED, config=_config(tmp), provenance_base=_PROV).metadata()
        self.assertEqual(meta["depth_backend"], "depth_pro")
        self.assertEqual(meta["mask_source"], "birefnet")
        self.assertEqual(meta["max_coc"], MAX_COC)
        self.assertEqual(meta["k_source"], "eq5_ssim_sweep")
        self.assertEqual(meta["provenance"]["pipeline_commit"], "abc1234")
        self.assertEqual(meta["provenance"]["image_h"], 96)

    def test_shape_diferente_rejeita_com_slug(self):
        aif, bokeh = _scene_images()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SampleRejected) as ctx:
            process_pair(FakePair("c", "s"), aif_bgr=aif, bokeh_bgr=bokeh[:, :48],
                         depth_runtime=FakeDepth(), mask_runtime=FakeMask(),
                         render_fn=_LAYERED, config=_config(tmp), provenance_base=_PROV)
        self.assertEqual(ctx.exception.reason, "resolution_invalid")


# ==============================================================================
# Gates — bloqueiam só com limiar congelado, e o slug entra no histograma
# ==============================================================================

class Gates(unittest.TestCase):
    def _roda(self, **cfg):
        aif, bokeh = _scene_images()
        with tempfile.TemporaryDirectory() as tmp:
            return process_pair(
                FakePair("cena1", "c_0001"), aif_bgr=aif, bokeh_bgr=bokeh,
                depth_runtime=FakeDepth(), mask_runtime=FakeMask(),
                render_fn=_LAYERED, config=_config(tmp, **cfg), provenance_base=_PROV)

    def test_sem_limiar_nenhum_gate_bloqueia(self):
        self._roda()          # não levanta

    def test_limiar_de_ssim_bloqueia_com_slug_do_conjunto_fechado(self):
        """O passo que o §3.2(c) afirma executar e que faltava inteiro."""
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(min_calibration_ssim=0.999999)
        self.assertEqual(ctx.exception.reason, "gate_calibration_ssim_below_floor")

    def test_gate_do_fstop_da_AIF_pega_o_D6(self):
        aif, bokeh = _scene_images()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SampleRejected) as ctx:
            process_pair(FakePair("c", "s", aif_f_number=5.6),   # 12,7% das cenas
                         aif_bgr=aif, bokeh_bgr=bokeh,
                         depth_runtime=FakeDepth(), mask_runtime=FakeMask(),
                         render_fn=_LAYERED, config=_config(tmp, min_aif_f_number=16.0),
                         provenance_base=_PROV)
        self.assertEqual(ctx.exception.reason, "gate_aif_aperture_wide")

    def test_todo_gate_medido_vai_para_o_metadado(self):
        quality = self._roda().quality
        for nome in ("focus_mask_sharpness_ratio", "calibration_ssim",
                     "depth_useful_levels", "aif_f_number", "mask_iou_aif_bokeh"):
            self.assertIn(nome, quality)
            self.assertIn("value", quality[nome])


# ==============================================================================
# O loop: escrita, histograma, retomada, split
# ==============================================================================

class Loop(unittest.TestCase):
    def _pares(self, n=4):
        return [FakePair(f"cena{i}", f"c_{i:04d}") for i in range(n)]

    def _load(self, pair):
        k = 8.0 + 6.0 * int(pair.scene_id[-1])       # K diferente por cena
        return _scene_images(k_verdadeiro=k, seed=int(pair.scene_id[-1]))

    def test_gera_grava_e_o_K_sobrevive_ao_disco(self):
        pares = self._pares()
        split = build_scene_split([p.scene_id for p in pares], val_fraction=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            stats = run_route_c(pares, load_pair=self._load, depth_runtime=FakeDepth(),
                                mask_runtime=FakeMask(), render_fn=_LAYERED, split=split,
                                config=_config(tmp), provenance_base=_PROV)
            self.assertEqual(stats.written, 4)
            ks = sorted(row["k_value"] for row in iter_manifest(tmp))
            self.assertEqual(len(set(np.round(ks, 3))), 4, "K não sobreviveu — D1")
            self.assertTrue((Path(tmp) / "split.json").exists(), "split não materializado")
            for row in iter_manifest(tmp):
                self.assertIn(row["split"], {"train", "val"})
                self.assertEqual(row["max_coc"], MAX_COC)
                self.assertEqual(row["depth_backend"], "depth_pro")

    def test_histograma_de_rejeicao_sempre_sai(self):
        pares = self._pares(3)
        split = build_scene_split([p.scene_id for p in pares], val_fraction=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            run_route_c(pares, load_pair=self._load, depth_runtime=FakeDepth(),
                        mask_runtime=FakeMask(), render_fn=_LAYERED, split=split,
                        config=_config(tmp, min_calibration_ssim=0.999999),
                        provenance_base=_PROV)
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "rejections.jsonl").read_text().strip().splitlines()]
        self.assertEqual(len(linhas), 3)
        self.assertTrue(all(l["reason"] == "gate_calibration_ssim_below_floor" for l in linhas))

    def test_retomada_pula_o_que_ja_foi_gravado(self):
        pares = self._pares(3)
        split = build_scene_split([p.scene_id for p in pares], val_fraction=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            run_route_c(pares[:2], load_pair=self._load, depth_runtime=FakeDepth(),
                        mask_runtime=FakeMask(), render_fn=_LAYERED, split=split,
                        config=_config(tmp), provenance_base=_PROV)
            stats = run_route_c(pares, load_pair=self._load, depth_runtime=FakeDepth(),
                                mask_runtime=FakeMask(), render_fn=_LAYERED, split=split,
                                config=_config(tmp), provenance_base=_PROV)
        self.assertEqual(stats.skipped_done, 2)
        self.assertEqual(stats.written, 1)

    def test_os_niveis_de_uma_cena_gravam_UM_png_de_profundidade(self):
        """Na rota C a AIF é a MESMA entre os níveis de uma cena — `image_focus` é byte
        a byte idêntico, medido em 90 linhas de 9 cenas do espelho — e o teto do paper
        deixa até 4 níveis por cena. No release publicado
        (`juliadollis/bokehnet-regen-rota-c`) são 15.423 amostras em 4.399 cenas: 3,51
        por cena, e `depth/` cairia de 15.423 para 4.399 arquivos (71,5% a menos).

        Aqui os quatro níveis de `cena0` dividem um PNG só; as quatro máscaras, não.
        """
        pares = [FakePair("cena0", f"c_realbokeh_cena0_l{n}") for n in range(1, 5)]
        split = build_scene_split(["cena0"], val_fraction=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            run_route_c(pares, load_pair=self._load, depth_runtime=FakeDepth(),
                        mask_runtime=FakeMask(), render_fn=_LAYERED, split=split,
                        config=_config(tmp), provenance_base=_PROV)
            raiz = Path(tmp)
            self.assertEqual(sorted(p.name for p in (raiz / "depth").glob("*.png")),
                             ["cena0.png"])
            self.assertEqual(len(list((raiz / "mask").glob("*.png"))), 4)
            linhas = list(iter_manifest(tmp))
            self.assertEqual(len(linhas), 4)
            for linha in linhas:
                with self.subTest(sample_id=linha["sample_id"]):
                    self.assertEqual(linha["depth_ref"], "depth/cena0.png")
                    meta = read_metadata(tmp, linha["sample_id"])
                    self.assertEqual(meta["depth_ref"], "depth/cena0.png")
                    self.assertEqual(meta["depth_layout"], "per_scene")
                    # a ESCALA continua por amostra: o PNG compartilhado só tem a forma
                    self.assertIn("disparity_min", meta)
                    self.assertIn("disparity_max", meta)

    def test_metadado_em_disco_reconstroi_o_mapa(self):
        """O mapa não é gravado; tem que ser derivável só dos escalares."""
        from control.contract import defocus_map
        from dataio.encoding import EncodedDepth, decode_depth_m
        from PIL import Image

        pares = self._pares(1)
        split = build_scene_split(["cena0"], val_fraction=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            run_route_c(pares, load_pair=self._load, depth_runtime=FakeDepth(),
                        mask_runtime=FakeMask(), render_fn=_LAYERED, split=split,
                        config=_config(tmp), provenance_base=_PROV)
            meta = read_metadata(tmp, "c_0000")
            u16 = np.array(Image.open(Path(tmp) / meta["depth_ref"]))

        enc = EncodedDepth(u16, meta["disparity_min"], meta["disparity_max"],
                           (meta["image_h"], meta["image_w"]),
                           (meta["depth_h"], meta["depth_w"]))
        mapa = defocus_map(decode_depth_m(enc), meta["focus_disparity"], meta["k_value"])
        self.assertGreater(float(mapa.max()), 0.0)
        self.assertLessEqual(float(mapa.max()), 1.0)
        self.assertEqual(list((Path(tmp)).glob("**/*defocus*")), [])


# ==============================================================================
# Refinamento da região em foco — o substituto automático do §3.2(c)
#
# O que cada teste aqui precisa provar, e por quê: o piloto mediu que a máscara do
# BiRefNet acerta o plano de foco em 35,2% das amostras e que 20,6% eram DESCARTADAS
# por máscara vazia, sendo que o paper diz explicitamente que não descarta. Então o que
# não pode passar sem teste é (1) a amostra de máscara vazia ATRAVESSAR, (2) sair
# marcada com a origem certa, e (3) a marcação chegar ao disco e ao manifesto.
# ==============================================================================

class RegiaoEmFoco(unittest.TestCase):

    def _processa(self, mask_runtime, *, imagens=None, **cfg):
        aif, bokeh = imagens if imagens is not None else _scene_images()
        with tempfile.TemporaryDirectory() as tmp:
            return process_pair(
                FakePair("cena1", "c_0001"), aif_bgr=aif, bokeh_bgr=bokeh,
                depth_runtime=FakeDepth(), mask_runtime=mask_runtime,
                render_fn=_LAYERED, config=_config(tmp, **cfg),
                provenance_base=_PROV)

    def test_mascara_VAZIA_agora_PASSA_e_sai_marcada_retention_only(self):
        """O teste mais importante desta etapa.

        Antes esta amostra era rejeitada com `focus_mask_empty` — 42 de 204 no piloto,
        20,6%, em 10 cenas inteiras. O §3.2(c) diz textualmente que **não** descarta.
        Agora ela atravessa e sai marcada, para dar para treinar com e sem ela.
        """
        sample = self._processa(MascaraVazia())
        self.assertIs(sample.focus.source, FocusSource.RETENTION_ONLY)
        self.assertTrue(sample.focus.was_refined)
        self.assertTrue(sample.focus.initial_mask_was_empty)
        self.assertEqual(sample.focus.agreement, 0.0)
        self.assertEqual(sample.focus.initial_mask_area_ratio, 0.0)
        self.assertEqual(sample.mask_source, MaskSource.RETENTION_ONLY)
        self.assertTrue(np.asarray(sample.mask).any(), "a região final saiu vazia")
        self.assertGreater(sample.control.focus_disparity, 0.0)

    def test_mascara_vazia_NAO_tem_linha_de_base_pareada(self):
        """`None` é resposta: sem máscara não existe valor anterior a comparar.

        É diferente de zero, e é por isso que o script de validação conta quantas
        amostras ficaram sem linha de base em vez de assumir um número para elas.
        """
        sample = self._processa(MascaraVazia())
        self.assertIsNone(sample.focus.disparity_from_initial_mask)

    def test_mascara_no_plano_ERRADO_e_refinada_e_marcada(self):
        sample = self._processa(MascaraNoPlanoErrado())
        self.assertIn(sample.focus.source,
                      (FocusSource.BIREFNET_REFINED, FocusSource.RETENTION_ONLY))
        self.assertTrue(sample.focus.was_refined)
        self.assertLess(sample.focus.agreement, 0.30)
        self.assertEqual(sample.mask_source,
                         FOCUS_SOURCE_TO_MASK_SOURCE[sample.focus.source])

    def test_mascara_no_plano_errado_MOVE_o_foco_para_perto_da_verdade(self):
        """O refinamento tem que mudar o número, não só a etiqueta.

        A cena tem o objeto a 3 m e a máscara errada cai no gradiente longe dele. Se
        `focus_disparity` continuasse saindo da máscara errada, a marcação seria
        decoração.
        """
        sample = self._processa(MascaraNoPlanoErrado())
        base = sample.focus.disparity_from_initial_mask
        self.assertIsNotNone(base, "sem linha de base não há o que comparar")
        erro_base = abs(base - 1.0 / 3.0)
        erro_refinado = abs(sample.control.focus_disparity - 1.0 / 3.0)
        self.assertLess(erro_refinado, erro_base,
                        f"refinado {erro_refinado:.4f} não é melhor que "
                        f"base {erro_base:.4f}")

    def test_mascara_que_CONCORDA_e_mantida_e_nao_conta_como_refinada(self):
        """O caminho do paper: M confiável não se mexe, e a amostra não fica marcada."""
        sample = self._processa(FakeMask())
        self.assertIs(sample.focus.source, FocusSource.BIREFNET)
        self.assertFalse(sample.focus.was_refined)
        self.assertEqual(sample.mask_source, MaskSource.BIREFNET)
        self.assertAlmostEqual(sample.control.focus_disparity, 1.0 / 3.0, places=5)

    def test_cena_sem_detalhe_nenhum_REJEITA_com_slug(self):
        """Aqui descartar não é escolha, é consequência: não há plano de foco."""
        with self.assertRaises(SampleRejected) as ctx:
            self._processa(MascaraVazia(), imagens=_cena_lisa())
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")

    def test_a_mascara_GRAVADA_e_a_que_produziu_o_foco(self):
        """Proveniência que não mente: o arquivo do disco é a região refinada."""
        sample = self._processa(MascaraNoPlanoErrado())
        from control.contract import focus_disparity_from_mask
        depth = FakeDepth().infer(np.zeros((96, 96, 3), dtype=np.uint8)).values_m
        self.assertAlmostEqual(
            focus_disparity_from_mask(depth, sample.mask),
            sample.control.focus_disparity, places=9,
            msg="a máscara gravada não reproduz o focus_disparity do rótulo")

    def test_resolucao_da_retencao_e_DECLARADA_no_metadado(self):
        """Quantidade em pixel carrega a resolução em que foi medida.

        A janela de 33 px cobre 1,7% do lado longo a 1500x2000 e 6,4% a 512x683: sem a
        resolução ao lado, `focus_retention_window_px` não diz nada.
        """
        meta = self._processa(FakeMask(), focus_retention_long_side=48).metadata()
        self.assertEqual((meta["focus_retention_h"], meta["focus_retention_w"]), (48, 48))
        self.assertEqual(meta["focus_retention_window_px"], 33)
        # A região volta para a resolução da IMAGEM — é ela que casa com a profundidade.
        self.assertEqual((meta["image_h"], meta["image_w"]), (96, 96))

    def test_o_DEFAULT_e_resolucao_cheia_e_isso_foi_medido(self):
        """A resolução de trabalho foi medida e REJEITADA, não esquecida.

        A 1500x2000, `refine_focus_region` custa 433 ms com grade de 512 contra 391 ms
        em resolução cheia: reduzir duas fotos de 3 MP (185 ms cada) custa mais que a
        retenção que a redução economiza (291 ms -> 22 ms). Se alguém trocar este default
        de volta por intuição de custo, este teste pergunta pela medição.
        """
        self.assertIsNone(RouteCConfig(output_dir=Path("/tmp")).focus_retention_long_side)

    def test_resolucao_cheia_quando_long_side_e_None(self):
        meta = self._processa(FakeMask(), focus_retention_long_side=None).metadata()
        self.assertEqual((meta["focus_retention_h"], meta["focus_retention_w"]), (96, 96))

    def test_regiao_refinada_volta_na_resolucao_da_imagem(self):
        """Se voltasse na grade de trabalho, a Eq. 4 rejeitaria por shape — ou pior,
        casaria por acidente com uma profundidade de outra resolução."""
        aif, bokeh = _scene_images()
        cfg = _config("/tmp", focus_retention_long_side=32)
        mask, record = refine_focus_region(
            aif_bgr=aif, bokeh_bgr=bokeh, mask_initial=FakeMask().infer(aif),
            depth_m=FakeDepth().infer(aif).values_m, config=cfg)
        self.assertEqual(mask.shape, aif.shape[:2])
        self.assertEqual(record.retention_hw, (32, 32))

    def test_todo_metadado_de_foco_chega_ao_JSON(self):
        meta = self._processa(FakeMask()).metadata()
        for campo in ("focus_source", "focus_was_refined", "focus_agreement",
                      "focus_retention_in_region", "focus_region_area_ratio",
                      "focus_retention_h", "focus_retention_w",
                      "focus_retention_window_px", "focus_initial_mask_was_empty",
                      "focus_initial_mask_area_ratio",
                      "focus_disparity_from_initial_mask"):
            self.assertIn(campo, meta)

    def test_os_parametros_A_do_refino_ficam_NA_AMOSTRA(self):
        """Dois runs com `focus_top_fraction` diferente produzem rótulos diferentes.

        Se o parâmetro vivesse só no `run_config.json`, um release remontado de dois runs
        não teria como ser separado — e nenhum destes valores vem do paper.
        """
        sample = self._processa(FakeMask(), focus_top_fraction=0.03,
                                focus_agreement_floor=0.4)
        refino = sample.provenance.extra["focus_refinement"]
        self.assertEqual(refino["top_fraction"], 0.03)
        self.assertEqual(refino["agreement_floor"], 0.4)
        self.assertEqual((refino["retention_h"], refino["retention_w"]), (96, 96))
        self.assertIn("[A]", refino["evidence"])

    def test_sem_mascara_inicial_a_area_e_None_e_nao_zero(self):
        """"O segmentador não rodou" é diferente de "rodou e devolveu vazio", e o
        segundo caso é 20,6% do piloto — o número que precisa continuar contável."""
        aif, bokeh = _scene_images()
        _, record = refine_focus_region(
            aif_bgr=aif, bokeh_bgr=bokeh, mask_initial=None,
            depth_m=FakeDepth().infer(aif).values_m, config=_config("/tmp"))
        self.assertIsNone(record.initial_mask_area_ratio)
        self.assertIsNone(record.to_metadata()["focus_initial_mask_area_ratio"])
        self.assertTrue(record.initial_mask_was_empty)

    def test_gate_de_retencao_bloqueia_com_slug_do_conjunto_fechado(self):
        """O limiar que julga a região refinada com a grandeza certa."""
        with self.assertRaises(SampleRejected) as ctx:
            self._processa(FakeMask(), min_focus_region_retention=0.999999)
        self.assertEqual(ctx.exception.reason, "gate_focus_region_retention_low")

    def test_gates_que_comparam_AIF_com_BOKEH_seguem_na_mascara_do_BiRefNet(self):
        """`mask_iou_aif_bokeh` mede divergência do segmentador, não do plano de foco.

        Com as duas máscaras do BiRefNet iguais e cheias, a IoU é 1,0 — e tem que
        continuar sendo, mesmo quando a região final é outra.
        """
        sample = self._processa(FakeMask())
        self.assertAlmostEqual(sample.quality["mask_iou_aif_bokeh"]["value"], 1.0,
                               places=6)

    def test_IoU_indefinida_NAO_desfaz_o_refinamento(self):
        """O gate que reintroduziria o descarte pela porta de trás.

        Com o BiRefNet declinando nas duas imagens, a IoU não existe. Antes disto, NaN
        reprovava, e a amostra que o refinamento acabou de salvar saía rejeitada com
        `gate_mask_iou` — os mesmos 20,6% de volta, com outro slug.
        """
        sample = self._processa(MascaraVazia())
        iou = sample.quality["mask_iou_aif_bokeh"]
        self.assertFalse(iou["applicable"])
        self.assertTrue(iou["passed"], "IoU inexistente reprovando de novo")
        self.assertTrue(np.isnan(iou["value"]))

    def test_todo_focus_source_tem_mask_source_proprio(self):
        """A ponte entre os dois vocabulários é total, e nenhum valor é reusado."""
        self.assertEqual(set(FOCUS_SOURCE_TO_MASK_SOURCE), set(FocusSource))
        valores = [m.value for m in FOCUS_SOURCE_TO_MASK_SOURCE.values()]
        self.assertEqual(len(valores), len(set(valores)),
                         "duas origens de foco compartilhando mask_source")


class MarcacaoNoDisco(unittest.TestCase):
    """A marcação tem que sobreviver ao disco — no JSON e no manifesto."""

    def _roda(self, mask_runtime, tmp, n=2):
        pares = [FakePair(f"cena{i}", f"c_{i:04d}") for i in range(n)]
        split = build_scene_split([p.scene_id for p in pares], val_fraction=0.34)
        stats = run_route_c(
            pares, load_pair=lambda p: _scene_images(seed=int(p.scene_id[-1])),
            depth_runtime=FakeDepth(), mask_runtime=mask_runtime, render_fn=_LAYERED,
            split=split, config=_config(tmp), provenance_base=_PROV)
        return stats

    def test_manifesto_carrega_o_focus_source(self):
        """O manifesto é a superfície de varredura: filtrar por refinamento não pode
        exigir abrir 22.990 JSONs."""
        with tempfile.TemporaryDirectory() as tmp:
            self._roda(MascaraVazia(), tmp)
            linhas = list(iter_manifest(tmp))
        self.assertEqual(len(linhas), 2)
        for linha in linhas:
            self.assertEqual(linha["focus_source"], "retention_only")
            self.assertTrue(linha["focus_was_refined"])
            self.assertIn("focus_agreement", linha)
            self.assertIn("focus_retention_in_region", linha)
            self.assertIn("focus_region_area_ratio", linha)
            # fração medida numa grade: a grade viaja junto
            self.assertEqual((linha["focus_retention_h"], linha["focus_retention_w"]),
                             (96, 96))

    def test_manifesto_e_metadado_nao_divergem(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._roda(FakeMask(), tmp)
            for linha in iter_manifest(tmp):
                meta = read_metadata(tmp, linha["sample_id"])
                for campo in ("focus_source", "focus_was_refined", "focus_agreement",
                              "focus_region_area_ratio"):
                    self.assertEqual(linha[campo], meta[campo], campo)

    def test_stats_contam_por_focus_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(MascaraVazia(), tmp, n=3)
        self.assertEqual(stats.focus_sources["retention_only"], 3)
        self.assertEqual(stats.focus_initial_empty, 3)

    def test_resumo_imprime_contagem_E_percentual(self):
        """Se `retention_only` for alto, tem que saltar aos olhos no log."""
        stats = RouteCStats(written=4, k_values=[10.0, 11.0], ssim_values=[0.9, 0.9])
        stats.focus_sources.update({"birefnet": 1, "retention_only": 3})
        stats.focus_initial_empty = 3
        texto = stats.summary()
        self.assertIn("retention_only", texto)
        self.assertIn("75.0%", texto)
        self.assertIn("25.0%", texto)

    def test_resumo_ALERTA_quando_o_BiRefNet_praticamente_nao_participou(self):
        """Lote majoritariamente `retention_only` não é motivo para parar o run — o
        §3.2(c) manda preservar a amostra — mas é motivo para não publicar sem laudo."""
        stats = RouteCStats(written=10, k_values=[10.0], ssim_values=[0.9])
        stats.focus_sources.update({"birefnet": 2, "retention_only": 8})
        texto = stats.summary()
        self.assertIn("ATENÇÃO", texto)
        self.assertIn("validate_focus_refinement", texto)

    def test_resumo_NAO_alerta_quando_o_lote_e_normal(self):
        stats = RouteCStats(written=10, k_values=[10.0], ssim_values=[0.9])
        stats.focus_sources.update({"birefnet": 8, "retention_only": 2})
        self.assertNotIn("ATENÇÃO", stats.summary())

    def test_resumo_denuncia_fonte_fora_do_enum(self):
        """Vocabulário aberto é defeito, e um resumo que só imprime os três valores
        conhecidos esconderia um quarto."""
        stats = RouteCStats(written=1, k_values=[10.0], ssim_values=[0.9])
        stats.focus_sources.update({"grabcut": 1})
        self.assertIn("FONTE DESCONHECIDA", stats.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
