"""Testes do contrato de dados: codificação, metadados e split por cena."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import MAX_COC, signed_coc_px                    # noqa: E402
from dataio.encoding import (                                          # noqa: E402
    encode_depth, decode_depth_m, decode_disparity,
    quantization_coc_error_px, resize_depth_nearest,
)
from dataio.layout import (                                            # noqa: E402
    DepthLayout, depth_ref_of, require_filename_safe_scene_id,
)
from dataio.sample import (                                            # noqa: E402
    FOCUS_SOURCE_TO_MASK_SOURCE, ControlLabel, FocusRegionRecord, KSource,
    MaskSource, Sample, SampleProvenance, SampleRefs, validate_metadata,
)
from qc.focus_region import FocusSource, RefinedFocusRegion             # noqa: E402
from dataio.split import (                                             # noqa: E402
    build_scene_split, check_no_leak, split_from_source,
)
from dataio.writer import (                                            # noqa: E402
    FileSampleWriter, estimate_disk_budget, iter_manifest, read_depth_u16,
    read_metadata,
)
from control.contract import SampleRejected                            # noqa: E402


def _depth(h=64, w=96, near=1.0, far=60.0):
    return np.linspace(near, far, h * w, dtype=np.float32).reshape(h, w)


_PROV = SampleProvenance(pipeline_commit="abc123", depth_model_sha256="d" * 64,
                         mask_model_sha256="m" * 64, image_hw=(64, 96), seed=7)


def _focus(source=FocusSource.BIREFNET, **kw):
    """A marcação da região em foco, com a origem da máscara batendo com ela.

    Default `BIREFNET`: o caminho do paper, em que M é confiável e não se mexe.
    """
    base = dict(agreement=0.82, precision=0.76, iou=0.64,
                retention_in_region=0.91, area_ratio=0.24,
                retention_hw=(64, 96), retention_window_px=33,
                initial_mask_was_empty=False, initial_mask_area_ratio=0.24,
                disparity_from_initial_mask=1.0 / 3.0)
    base.update(kw)
    return FocusRegionRecord(source=source, **base)


def _sample(sample_id="c_0001", scene="1038", k=18.0, focus=None, **kw):
    focus = _focus() if focus is None else focus
    return Sample(
        sample_id=sample_id, route="c",
        refs=SampleRefs("akcit-pixel/RealBokeh", f"{scene}_level_3", scene,
                        aif_ref="image_focus", bokeh_ref="image_blur", source_split="train"),
        control=ControlLabel(k_value=k, k_source=KSource.EQ5_SSIM_SWEEP,
                             focus_disparity=1.0 / 3.0, is_k_censored=False,
                             depth_backend="depth_pro", calibration_ssim=0.91),
        depth=encode_depth(_depth(), image_hw=(64, 96), long_side=768),
        mask=(np.arange(64 * 96).reshape(64, 96) % 7 == 0),
        # A `mask_source` sai do `focus_source` pela ponte única: o arquivo gravado é a
        # região que produziu `focus_disparity`.
        mask_source=focus.mask_source,
        focus=focus,
        provenance=_PROV, **kw)


def _split_de(*cenas):
    return build_scene_split(list(cenas) or ["1038"], val_fraction=0.1)


# ==============================================================================
# Codificação — quantizar em DISPARIDADE, não em profundidade
# ==============================================================================

class Encoding(unittest.TestCase):
    def test_roundtrip_preserva_o_coc(self):
        """O erro que importa não é o de profundidade, é o de CoC."""
        z = _depth(200, 300, 1.0, 100.0)
        enc = encode_depth(z, image_hw=z.shape[:2], long_side=768)
        z_back = decode_depth_m(enc)
        focus_disp, k = 1.0 / 3.0, 50.0
        coc_orig = np.abs(signed_coc_px(z, focus_disp, k))
        coc_back = np.abs(signed_coc_px(z_back, focus_disp, k))
        self.assertLess(float(np.abs(coc_orig - coc_back).max()), 0.01)

    def test_erro_declarado_bate_com_o_medido(self):
        enc = encode_depth(_depth(120, 160, 1.0, 100.0), image_hw=(120, 160), long_side=768)
        k = 50.0
        declarado = quantization_coc_error_px(enc, k)
        disp_erro = float(np.abs(1.0 / _depth(120, 160, 1.0, 100.0) - decode_disparity(enc)).max())
        self.assertLessEqual(disp_erro * k, declarado * 2.5)
        self.assertLess(declarado, 0.01)

    def test_disparidade_ganha_em_cena_NATURAL(self):
        """A razão de existir do módulo, com a condição explícita.

        Cena natural = conteúdo perto, céu no teto do Depth Pro. É o caso que deixou
        24,7% da rota B com a cena útil em menos de 256 níveis de 65535, mediana de 23
        no subgrupo `z_max >= 1000 m`.

        A vantagem NÃO é universal: numa cena linear em `z` — densa no fundo distante —
        a ordem se inverte. Nossas cenas não são assim, e o teste abaixo trava a
        distinção para ninguém generalizar demais.
        """
        z = np.concatenate([np.linspace(1.0, 50.0, 3600),
                            np.full(496, 10000.0)]).astype(np.float32).reshape(64, 64)
        enc = encode_depth(z, image_hw=z.shape[:2], long_side=768)
        niveis_disp = len(np.unique(enc.disparity_u16))

        z01 = (z - z.min()) / (z.max() - z.min())
        niveis_depth = len(np.unique(np.rint(z01 * 65535).astype(np.uint16)))

        self.assertGreater(niveis_disp, 2500)
        self.assertLess(niveis_depth, 400)
        self.assertGreater(niveis_disp, 5 * niveis_depth)

    def test_em_cena_linear_em_z_a_ordem_se_inverte(self):
        """Documenta o limite da escolha, em vez de fingir que ela é universal."""
        z = np.linspace(1.0, 10000.0, 4096, dtype=np.float32).reshape(64, 64)
        niveis_disp = len(np.unique(encode_depth(z, image_hw=z.shape[:2], long_side=768).disparity_u16))
        z01 = (z - z.min()) / (z.max() - z.min())
        niveis_depth = len(np.unique(np.rint(z01 * 65535).astype(np.uint16)))
        self.assertGreater(niveis_depth, niveis_disp)

    def test_resize_usa_vizinho_mais_proximo(self):
        """Interpolar profundidade atravessa descontinuidade e inventa plano
        intermediário: entre 1 m e 20 m, a média é 10,5 m — superfície fantasma."""
        z = np.full((100, 100), 1.0, dtype=np.float32)
        z[:, 50:] = 20.0
        pequeno = resize_depth_nearest(z, 50)
        self.assertEqual(set(np.unique(pequeno).tolist()), {1.0, 20.0})

    def test_grava_a_resolucao_da_IMAGEM_e_a_do_depth(self):
        """`image_hw` é o shape da IMAGEM, não do array de profundidade. Antes o campo
        se chamava `source_hw` e guardava o shape recebido — nome de uma coisa, valor
        de outra —, e `k_at_resolution` calculava o fator errado em silêncio. Medido:
        Depth Pro a 1152x1536 sobre uma foto 3024x4032 dava erro de 2,63x no K."""
        enc = encode_depth(_depth(384, 512), image_hw=(3024, 4032), long_side=768)
        meta = enc.to_metadata()
        self.assertEqual((meta["image_h"], meta["image_w"]), (3024, 4032))
        self.assertEqual((meta["depth_h"], meta["depth_w"]), (384, 512))

    def test_span_de_disparidade_vem_da_resolucao_CHEIA(self):
        """O vizinho mais próximo descarta o pixel mais perto e encolhe o span — e o
        span alimenta `k_for_bokehme`. Medido: objeto de 3x3 px a 0,30 m some na
        decimação e o K do renderer sai 3,4x menor."""
        z = np.full((512, 512), 20.0, dtype=np.float32)
        z[100:103, 100:103] = 0.30                     # objeto pequeno e muito perto
        enc = encode_depth(z, image_hw=(512, 512), long_side=64)
        self.assertAlmostEqual(enc.disparity_max, 1.0 / 0.30, places=4)
        self.assertAlmostEqual(enc.z_min_m, 0.30, places=5)

    def test_faixa_degenerada_rejeita_com_slug(self):
        """`reject`, não `ValueError`: sem slug a falha escapa do histograma de
        motivos, que é o instrumento que calibra todo limiar."""
        with self.assertRaises(SampleRejected):
            encode_depth(np.full((16, 16), 5.0, dtype=np.float32), image_hw=(16, 16))


# ==============================================================================
# Metadados — ausência é erro, não omissão
# ==============================================================================

class Metadata(unittest.TestCase):
    def test_amostra_completa_valida(self):
        validate_metadata(_sample().metadata())

    def test_campo_faltando_levanta(self):
        meta = _sample().metadata()
        del meta["focus_disparity"]
        with self.assertRaises(ValueError) as ctx:
            validate_metadata(meta)
        self.assertIn("focus_disparity", str(ctx.exception))

    def test_k_invalido_levanta(self):
        meta = _sample().metadata()
        meta["k_value"] = 0.0
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_focus_depth_e_derivado_e_consistente(self):
        meta = _sample().metadata()
        self.assertAlmostEqual(1.0 / meta["focus_depth_m"], meta["focus_disparity"], places=9)

    def test_json_serializa_tipos_numpy(self):
        json.loads(_sample().metadata_json())


# ==============================================================================
# Marcação da região em foco — o requisito de poder treinar COM e SEM as refinadas
# ==============================================================================

class MarcacaoDaRegiaoEmFoco(unittest.TestCase):

    def test_sem_a_marcacao_o_metadado_e_REPROVADO(self):
        """O campo novo é obrigatório, e a ausência é erro, não omissão.

        `focus_disparity` é a base do rótulo inteiro. Sem dizer de onde a região veio,
        ele é um número sem proveniência — e medimos que a máscara do BiRefNet acerta o
        plano de foco em 35,2% das amostras.
        """
        meta = _sample().metadata()
        del meta["focus_source"]
        with self.assertRaises(ValueError) as ctx:
            validate_metadata(meta)
        self.assertIn("focus_source", str(ctx.exception))

    def test_cada_campo_novo_e_exigido_individualmente(self):
        """Um a um: um `frozenset` que perdesse uma entrada passaria em silêncio."""
        for campo in ("focus_source", "focus_was_refined", "focus_agreement",
                      "focus_retention_in_region", "focus_region_area_ratio",
                      "focus_retention_h", "focus_retention_w",
                      "focus_retention_window_px"):
            with self.subTest(campo=campo):
                meta = _sample().metadata()
                del meta[campo]
                with self.assertRaises(ValueError) as ctx:
                    validate_metadata(meta)
                self.assertIn(campo, str(ctx.exception))

    def test_amostra_SEM_objeto_focus_reprova(self):
        """`Sample.focus=None` não grava os campos, e o writer barra antes do disco."""
        amostra = _sample()
        amostra.focus = None
        with self.assertRaises(ValueError):
            validate_metadata(amostra.metadata())

    def test_focus_source_fora_do_enum_reprova(self):
        meta = _sample().metadata()
        meta["focus_source"] = "grabcut"
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_was_refined_que_MENTE_sobre_a_origem_reprova(self):
        """É por este booleano que se monta o treino com e sem as refinadas. Se ele
        puder discordar da origem, o filtro seleciona outro conjunto que não o que diz
        selecionar."""
        meta = _sample(focus=_focus(FocusSource.RETENTION_ONLY)).metadata()
        meta["focus_was_refined"] = False
        with self.assertRaises(SampleRejected) as ctx:
            validate_metadata(meta)
        self.assertEqual(ctx.exception.reason, "focus_provenance_inconsistent")

    def test_mask_source_que_nao_e_a_mascara_do_rotulo_reprova(self):
        """A máscara gravada tem que ser a que produziu `focus_disparity`.

        Dizer `retention_only` no foco e `birefnet` na máscara afirma que o arquivo em
        `mask/<id>.png` é a máscara do segmentador — e quem auditasse o rótulo olharia
        a máscara errada, sem nada denunciar.
        """
        meta = _sample(focus=_focus(FocusSource.RETENTION_ONLY)).metadata()
        meta["mask_source"] = MaskSource.BIREFNET.value
        with self.assertRaises(SampleRejected) as ctx:
            validate_metadata(meta)
        self.assertEqual(ctx.exception.reason, "focus_provenance_inconsistent")

    def test_mask_source_refinada_com_foco_do_birefnet_tambem_reprova(self):
        """A incoerência vale nos dois sentidos."""
        meta = _sample().metadata()
        meta["mask_source"] = MaskSource.BIREFNET_REFINED.value
        with self.assertRaises(SampleRejected):
            validate_metadata(meta)

    def test_area_da_regiao_ZERO_reprova(self):
        """Região vazia não define plano de foco: a Eq. 4 não teria pixel."""
        meta = _sample().metadata()
        meta["focus_region_area_ratio"] = 0.0
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_acordo_fora_de_zero_um_reprova(self):
        for valor in (-0.1, 1.5, float("nan")):
            with self.subTest(valor=valor):
                meta = _sample().metadata()
                meta["focus_agreement"] = valor
                with self.assertRaises(ValueError):
                    validate_metadata(meta)

    def test_resolucao_da_retencao_invalida_reprova(self):
        """Inclusive `None`: a chave existir faz o teste de campo obrigatório passar, e a
        resolução da medida continua não existindo."""
        for campo in ("focus_retention_h", "focus_retention_w",
                      "focus_retention_window_px"):
            for valor in (0, -1, None):
                with self.subTest(campo=campo, valor=valor):
                    meta = _sample().metadata()
                    meta[campo] = valor
                    with self.assertRaises(ValueError):
                        validate_metadata(meta)

    def test_retencao_NaN_na_regiao_e_aceita(self):
        """NaN aqui significa "não havia detalhe medível na região", que é diferente de
        "reteve zero". Colapsar as duas afirmaria borrão onde não houve medida."""
        meta = _sample(focus=_focus(retention_in_region=float("nan"))).metadata()
        validate_metadata(meta)

    def test_toda_fonte_de_foco_tem_fonte_de_mascara(self):
        """A ponte é total. Um `FocusSource` novo sem entrada levanta `KeyError` — não
        escolhe um default, que é como `mask_source="automatic"` nasceu."""
        self.assertEqual(set(FOCUS_SOURCE_TO_MASK_SOURCE), set(FocusSource))
        for source in FocusSource:
            self.assertIsInstance(FOCUS_SOURCE_TO_MASK_SOURCE[source], MaskSource)

    def test_was_refined_bate_com_o_do_modulo_de_refino(self):
        """As duas definições não podem divergir — é a regra "cópias divergem"."""
        for source in FocusSource:
            regiao = RefinedFocusRegion(
                mask=np.ones((4, 4), dtype=bool), source=source, agreement=0.5, precision=0.4, iou=0.3,
                retention_in_region=0.8, area_ratio=0.1, initial_mask_was_empty=False)
            registro = FocusRegionRecord.from_region(
                regiao, retention_hw=(4, 4), retention_window_px=3,
                initial_mask_area_ratio=0.1)
            self.assertEqual(registro.was_refined, regiao.was_refined, source.value)

    def test_from_region_nao_segura_a_mascara(self):
        """Um registro de metadado não deve manter um array de 3 megapixels vivo."""
        regiao = RefinedFocusRegion(
            mask=np.ones((8, 8), dtype=bool), source=FocusSource.RETENTION_ONLY,
            agreement=0.0, precision=0.0, iou=0.0, retention_in_region=0.7, area_ratio=0.05,
            initial_mask_was_empty=True)
        registro = FocusRegionRecord.from_region(
            regiao, retention_hw=(8, 8), retention_window_px=3,
            initial_mask_area_ratio=0.0)
        self.assertNotIn("mask", registro.to_metadata())
        self.assertFalse(any(isinstance(v, np.ndarray)
                             for v in registro.to_metadata().values()))


# ==============================================================================
# Writer — não faz aritmética sobre o sinal
# ==============================================================================

class Writer(unittest.TestCase):
    def test_grava_e_reabre(self):
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample())
            root = Path(tmp)
            # A profundidade é por CENA (`scene_id == "1038"`); a máscara é por amostra.
            self.assertTrue((root / "depth" / "1038.png").exists())
            self.assertTrue((root / "mask" / "c_0001.png").exists())
            meta = json.loads((root / "meta" / "c_0001.json").read_text())
            self.assertEqual(meta["k_value"], 18.0)
            self.assertEqual(meta["scene_id"], "1038")

    def test_nao_grava_defocus_map(self):
        """O mapa é derivado no dataloader pela MESMA função da geração. Gravá-lo
        criaria uma segunda fonte de verdade — que é o defeito D1."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample())
            self.assertEqual(list((Path(tmp)).glob("**/*defocus*")), [])

    def test_k_diferente_sobrevive_a_gravacao(self):
        """Regressão direta do D1: o writer não pode normalizar nada."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample("a", "s1", k=5.0))
                w.write(_sample("b", "s2", k=90.0))
            ks = [json.loads(l)["k_value"] for l in
                  (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()]
            self.assertEqual(sorted(ks), [5.0, 90.0])

    def test_manifesto_carrega_a_resolucao_do_K(self):
        """K é um número EM PIXEL: sem a resolução na mesma linha, o manifesto não diz
        o que o K significa, e comparar K entre rotas vira comparação sem escala."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1")) as w:
                w.write(_sample("a", "s1"))
            linha = json.loads(
                (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()[0])
            for campo in ("image_h", "image_w", "depth_h", "depth_w"):
                with self.subTest(campo=campo):
                    self.assertIn(campo, linha)
            self.assertEqual((linha["image_h"], linha["image_w"]), (64, 96))

    def test_manifesto_carrega_a_marcacao_do_refinamento(self):
        """Sem isto, montar o treino sem as amostras refinadas exigiria abrir 22.990
        JSONs — e o manifesto existe justamente para ser a varredura rápida."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1")) as w:
                w.write(_sample("a", "s1", focus=_focus(FocusSource.RETENTION_ONLY)))
            linha = json.loads(
                (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()[0])
        self.assertEqual(linha["focus_source"], "retention_only")
        self.assertTrue(linha["focus_was_refined"])
        self.assertEqual(linha["mask_source"], "retention_only")
        for campo in ("focus_agreement", "focus_retention_in_region",
                      "focus_region_area_ratio", "focus_retention_h",
                      "focus_retention_w"):
            with self.subTest(campo=campo):
                self.assertIn(campo, linha)

    def test_retomada_le_o_manifesto(self):
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample("c_0001"))
            self.assertEqual(FileSampleWriter(tmp, split=_split_de("1038")).completed_ids(), {"c_0001"})

    def test_metadado_incompleto_nao_chega_ao_disco(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _sample()
            object.__setattr__(s.control, "depth_backend", "depth_anything")
            with self.assertRaises(SampleRejected):
                FileSampleWriter(tmp, split=_split_de("1038")).write(s)
            self.assertEqual(list((Path(tmp) / "meta").glob("*.json")), [])

    def test_orcamento_de_disco(self):
        b = estimate_disk_budget(20554, 768, generates_image=False, image_megapixels=3.0)
        self.assertEqual(b["generated_image_gb"], 0.0)
        self.assertLess(b["total_gb"], 20.0)

    def test_o_orcamento_conta_a_profundidade_POR_CENA(self):
        """O número impresso antes do run tem que descrever o layout em que o run vai
        gravar. Orçando por amostra, a coluna de profundidade da rota A saía **41x**
        maior que a real — e é esse número que sustentou a decisão de não guardar as
        imagens de origem.

        As 69.700 amostras da rota A vêm de 1.700 cenas, então a coluna de profundidade
        encolhe exatamente 41x. (O número MEDIDO no `a_release` é 31,7 GB contra
        0,77 GB, a 444 KB por arquivo; o estimador usa 1,2 B/px, que é um teto e dá
        37,0 GB. São duas grandezas diferentes — este teste afirma o que o ESTIMADOR
        garante, que é a razão, não a medição.)
        """
        por_amostra = estimate_disk_budget(69_700, 768, generates_image=False,
                                           image_megapixels=3.0)
        por_cena = estimate_disk_budget(69_700, 768, generates_image=False,
                                        image_megapixels=3.0, n_scenes=1_700)
        self.assertEqual(por_amostra["n_scenes"], 69_700)
        self.assertEqual(por_cena["n_scenes"], 1_700)
        # a máscara continua por amostra nos dois; só a profundidade encolhe
        self.assertAlmostEqual(por_cena["depth_gb"],
                               por_amostra["depth_gb"] * 1_700 / 69_700, places=2)
        # `delta` e não `places`: os dois valores saem arredondados a 2 casas pelo
        # estimador, e exigir igualdade exata da razão testaria o `round`, não a regra.
        self.assertAlmostEqual(por_amostra["depth_gb"] / por_cena["depth_gb"],
                               69_700 / 1_700, delta=0.5)
        self.assertLess(por_cena["control_gb"], por_amostra["control_gb"] / 10)


# ==============================================================================
# Split por cena — o risco que cresceu de tamanho
# ==============================================================================

class Split(unittest.TestCase):
    def test_deterministico_entre_processos(self):
        """`hash()` do Python é randomizado por PYTHONHASHSEED: usá-lo faria a
        validação de ontem virar treino hoje."""
        cenas = [f"cena_{i}" for i in range(500)]
        a = build_scene_split(cenas, val_fraction=0.1)
        b = build_scene_split(list(reversed(cenas)), val_fraction=0.1)
        self.assertEqual(a.assignment, b.assignment)

    def test_fracao_aproximada(self):
        split = build_scene_split([f"c{i}" for i in range(2000)], val_fraction=0.05)
        frac = split.counts().get("val", 0) / 2000
        self.assertAlmostEqual(frac, 0.05, delta=0.02)

    def test_detecta_vazamento_por_cena(self):
        """21 aberturas da mesma cena: split por imagem colocaria a mesma cena nos
        dois lados. Por cena, não."""
        split = build_scene_split(["cena_A", "cena_B"], val_fraction=0.4)
        rows = ([{"sample_id": f"A_{i}", "scene_id": "cena_A", "split": split.of("cena_A")}
                 for i in range(21)]
                + [{"sample_id": f"B_{i}", "scene_id": "cena_B", "split": split.of("cena_B")}
                   for i in range(5)])
        rep = check_no_leak(rows, split)
        self.assertTrue(rep.clean)
        self.assertEqual(rep.samples_per_scene["cena_A"], 21)
        self.assertIn("21", rep.summary())

    def test_amostra_sem_cena_e_denunciada(self):
        split = build_scene_split(["x"], val_fraction=0.1)
        rep = check_no_leak([{"sample_id": "s1"}], split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.samples_without_scene, ["s1"])

    def test_cena_fora_do_split_e_denunciada(self):
        split = build_scene_split(["x"], val_fraction=0.1)
        rep = check_no_leak([{"sample_id": "s1", "scene_id": "desconhecida", "split": "train"}], split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.scenes_missing_from_split, ["desconhecida"])

    def test_gate_de_vazamento_PODE_reprovar(self):
        """A versão anterior era tautológica: `split.of()` é função pura, então
        `len(v) > 1` era impossível por construção e o ramo de detecção era código
        morto. Agora a linha carrega o split com que foi GRAVADA."""
        split = build_scene_split(["cena_A", "cena_B"], val_fraction=0.4)
        rows = [{"sample_id": "a1", "scene_id": "cena_A", "split": "train"},
                {"sample_id": "a2", "scene_id": "cena_A", "split": "val"}]
        rep = check_no_leak(rows, split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.scenes_in_both, ["cena_A"])
        self.assertIn("VAZAMENTO", rep.summary())

    def test_split_gravado_divergindo_do_materializado_e_pego(self):
        split = build_scene_split(["c1"], val_fraction=0.1)
        errado = "val" if split.of("c1") == "train" else "train"
        rep = check_no_leak([{"sample_id": "s", "scene_id": "c1", "split": errado}], split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.scenes_disagreeing_with_split, ["c1"])

    def test_origem_sem_split_NAO_vai_para_treino_em_silencio(self):
        """`(source or "")` mandava dado ausente para TREINO — o lado que infla a
        métrica — e o gate de vazamento não pegava, porque a cena ESTAVA no split."""
        for ausente in (None, "", "desconhecido"):
            with self.assertRaises(ValueError):
                split_from_source({"c1": "train", "c2": ausente})

    def test_respeita_o_split_da_origem(self):
        """A RealBokeh_3MP já separa train/test/validation por cena — reusar é melhor
        que sortear."""
        split = split_from_source({"a": "train", "b": "test", "c": "validation", "d": "train"})
        self.assertEqual(split.of("a"), "train")
        self.assertEqual(split.of("b"), "val")
        self.assertEqual(split.of("c"), "val")
        self.assertEqual(split.counts(), {"train": 2, "val": 2})

    def test_salva_e_recarrega(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = build_scene_split([f"c{i}" for i in range(50)], val_fraction=0.2)
            path = split.save(Path(tmp) / "split.json")
            from dataio.split import SceneSplit
            self.assertEqual(SceneSplit.load(path).assignment, split.assignment)


# ==============================================================================
# Profundidade por CENA — a dedup, e as três formas de ela dar errado
# ==============================================================================
#
# A premissa, medida: a profundidade é função da CENA. `routes/route_a.PreparedImage`
# calcula `encode_depth` uma vez por imagem e entrega o mesmo `EncodedDepth` às 41
# variantes; sobre o release `a_release` (69.700 amostras, 1.700 cenas), 120 cenas
# sorteadas com semente fixa deram **120/120** com a profundidade byte a byte idêntica
# entre as variantes, 0 divergentes, 0 ausentes. 31,7 GB contra 0,77 GB.
#
# Cada teste aqui prova UMA invariante dessa premissa, e a maioria delas é sobre o que
# acontece quando ela NÃO vale — porque é aí que a dedup rotula amostra com a
# profundidade de outra imagem, e é isso que não pode acontecer em silêncio.

class ProfundidadePorCena(unittest.TestCase):

    def _grava(self, tmp, amostras, **kw):
        cenas = sorted({s.refs.scene_id for s in amostras})
        with FileSampleWriter(tmp, split=_split_de(*cenas), **kw) as w:
            for amostra in amostras:
                w.write(amostra)
            return w.stats

    def test_as_variantes_de_uma_cena_gravam_um_png_so(self):
        """A invariante central: N amostras de uma cena, UM arquivo de profundidade.

        É a economia inteira — 69.700 PNG viram 1.700 —, e é o que a contagem por
        diretório do validador media errado.
        """
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._grava(tmp, [_sample(f"gp_abc_v{i:02d}", "gp_abc")
                                      for i in range(41)])
            root = Path(tmp)
            self.assertEqual(sorted(p.name for p in (root / "depth").glob("*.png")),
                             ["gp_abc.png"])
            self.assertEqual(len(list((root / "mask").glob("*.png"))), 41)
            self.assertEqual(len(list((root / "meta").glob("*.json"))), 41)
            self.assertEqual((stats.depth_files, stats.depth_reused), (1, 40))

    def test_a_mascara_nao_e_deduplicada_junto(self):
        """A profundidade sai da AIF, que as variantes compartilham; a máscara sai da
        região de foco, que é o que varia ENTRE elas. Deduplicar a máscara colapsaria
        exatamente o sinal que distingue uma variante da outra."""
        with tempfile.TemporaryDirectory() as tmp:
            a = _sample("gp_abc_v00", "gp_abc")
            b = _sample("gp_abc_v01", "gp_abc")
            b.mask = (np.arange(64 * 96).reshape(64, 96) % 3 == 0)
            self._grava(tmp, [a, b])
            mascaras = {p.name: p.read_bytes() for p in (Path(tmp) / "mask").glob("*.png")}
            self.assertEqual(len(mascaras), 2)
            self.assertNotEqual(*mascaras.values())

    def test_cada_meta_e_cada_linha_do_manifesto_carregam_a_referencia(self):
        """A referência é o CONTRATO. Sem ela, quem lê teria que adivinhar que
        `gp_abc_v07` mora em `depth/gp_abc.png` — e `scene_id` não é dedutível de
        `sample_id` em rota nenhuma."""
        with tempfile.TemporaryDirectory() as tmp:
            self._grava(tmp, [_sample(f"gp_abc_v{i:02d}", "gp_abc") for i in range(3)])
            root = Path(tmp)
            linhas = list(iter_manifest(root))
            self.assertEqual(len(linhas), 3)
            for linha in linhas:
                with self.subTest(sample_id=linha["sample_id"]):
                    meta = read_metadata(root, linha["sample_id"])
                    self.assertEqual(meta["depth_ref"], "depth/gp_abc.png")
                    self.assertEqual(meta["depth_layout"], "per_scene")
                    self.assertEqual(linha["depth_ref"], meta["depth_ref"])
                    self.assertTrue((root / meta["depth_ref"]).is_file())

    def test_a_escala_continua_por_amostra(self):
        """O PNG compartilhado carrega a FORMA; `disparity_min`/`max` reconstroem a
        ESCALA e continuam por amostra. Se a escala fosse junto com o PNG, a dedup teria
        apagado o normalizador de cada amostra — que é a família do defeito D1."""
        with tempfile.TemporaryDirectory() as tmp:
            self._grava(tmp, [_sample(f"gp_abc_v{i:02d}", "gp_abc") for i in range(2)])
            for sid in ("gp_abc_v00", "gp_abc_v01"):
                meta = read_metadata(tmp, sid)
                for campo in ("disparity_min", "disparity_max", "z_min_m", "z_max_m"):
                    with self.subTest(sample_id=sid, campo=campo):
                        self.assertIn(campo, meta)

    def test_profundidade_divergente_na_mesma_cena_e_erro(self):
        """A premissa da dedup é que a profundidade é função da cena. Quando não é —
        Depth Pro não determinístico, AIF trocada, `image_hw` diferente entre variantes
        — reusar o primeiro PNG rotularia as N−1 seguintes com a profundidade de OUTRA
        imagem. Aqui vira erro, não rótulo errado."""
        with tempfile.TemporaryDirectory() as tmp:
            outra = _sample("gp_abc_v01", "gp_abc")
            outra.depth = encode_depth(_depth(near=2.0, far=90.0),
                                       image_hw=(64, 96), long_side=768)
            with self.assertRaises(ValueError) as ctx:
                self._grava(tmp, [_sample("gp_abc_v00", "gp_abc"), outra])
            self.assertIn("difere da já gravada", str(ctx.exception))

    def test_a_mesma_cena_em_duas_fontes_e_erro(self):
        """`realbokeh.scene_key` e `lfdof.scene_key` são a MESMA função, então
        `train_1275` existe nas duas fontes e são cenas FÍSICAS diferentes. Por amostra
        isso nunca doeu (`c_realbokeh_…` contra `c_lfdof_…`); por cena as duas
        escreveriam `depth/train_1275.png`, e a segunda apagaria a primeira em
        silêncio."""
        with tempfile.TemporaryDirectory() as tmp:
            do_lfdof = _sample("c_lfdof_train_1275_l3", "train_1275")
            object.__setattr__(do_lfdof.refs, "source_dataset", "lfdof")
            with self.assertRaises(ValueError) as ctx:
                self._grava(tmp, [_sample("c_realbokeh_train_1275_l3", "train_1275"),
                                  do_lfdof])
            self.assertIn("duas fontes", str(ctx.exception))

    def test_cenas_que_diferem_so_na_caixa_sao_erro(self):
        """`Train_1` e `train_1` são dois arquivos no Hub e no ext4, e UM só em APFS e
        NTFS: o release perderia uma das duas cenas na primeira cópia para um laptop,
        sem erro nenhum."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self._grava(tmp, [_sample("c_a", "train_1"), _sample("c_b", "Train_1")])
            self.assertIn("caixa", str(ctx.exception))

    def test_scene_id_inseguro_nao_vira_nome_de_arquivo(self):
        """Sanear seria fallback: `a/b` e `a_b` saneariam para o mesmo nome, e a dedup
        fundiria duas cenas físicas num PNG só."""
        for ruim in ("a/b", "a b", "", ".", "..", ".oculto", "cena\n1"):
            with self.subTest(scene_id=ruim):
                with self.assertRaises(ValueError):
                    require_filename_safe_scene_id(ruim)
        for bom in ("gp_a1b2c3d4e5f6", "ebb_000000000000", "train_1275", "9001",
                    "c-1.2"):
            with self.subTest(scene_id=bom):
                self.assertEqual(require_filename_safe_scene_id(bom), bom)

    def test_os_scene_id_reais_das_fontes_sao_seguros(self):
        """A armadilha, conferida contra as funções de id DE VERDADE em vez de contra
        exemplos inventados. Se uma fonte mudar a forma do `scene_id`, este teste é que
        avisa — antes de o release inteiro sair com nomes que o Hub recusa."""
        from sources.genphoto_ebb import SOURCE_SLUGS
        from sources.lfdof import scene_key as lfdof_scene_key
        from sources.realbokeh import scene_key as realbokeh_scene_key

        for slug in SOURCE_SLUGS.values():
            with self.subTest(fonte=slug):
                require_filename_safe_scene_id(f"{slug}_a1b2c3d4e5f6")
        for split in ("train", "test", "validation"):
            for numero in ("1", "1275", "0007"):
                with self.subTest(split=split, numero=numero):
                    require_filename_safe_scene_id(realbokeh_scene_key(split, numero))
                    require_filename_safe_scene_id(lfdof_scene_key(split, numero))
        # rota B: `scene_id` é o `flickr_photo_id` (`scripts/run_route_b.py:164`)
        require_filename_safe_scene_id("20002539892")

    def test_as_duas_fontes_da_rota_c_colidem_no_scene_id(self):
        """Documenta a colisão em vez de assumir que ela não existe: as duas `scene_key`
        devolvem a mesma string. É por isso que o writer precisa da checagem de fonte —
        se um dia as chaves passarem a ser distintas, este teste cai e a checagem pode
        ser reavaliada com a medição na mão."""
        from sources.lfdof import scene_key as lfdof_scene_key
        from sources.realbokeh import scene_key as realbokeh_scene_key
        self.assertEqual(lfdof_scene_key("train", "1275"),
                         realbokeh_scene_key("train", "1275"))

    def test_layout_por_amostra_continua_gravando_por_amostra(self):
        """A dedup não é forçada. Na rota B `scene_id` é o `flickr_photo_id` e
        `sample_id` é `b_<scene_id>`: 13.800 cenas para 13.800 amostras, **0,0% de
        economia**. Ligar a dedup ali renomearia 13.800 arquivos em troca de nada."""
        with tempfile.TemporaryDirectory() as tmp:
            self._grava(tmp, [_sample("b_9001", "9001"), _sample("b_9002", "9002")],
                        depth_layout=DepthLayout.PER_SAMPLE)
            root = Path(tmp)
            self.assertEqual(sorted(p.name for p in (root / "depth").glob("*.png")),
                             ["b_9001.png", "b_9002.png"])
            meta = read_metadata(root, "b_9001")
            self.assertEqual(meta["depth_ref"], "depth/b_9001.png")
            self.assertEqual(meta["depth_layout"], "per_sample")

    def test_a_retomada_reusa_o_png_gravado_por_outro_run(self):
        """O writer de um run novo não tem o hash em memória. Reler o PNG da cena uma
        vez é o preço de não confiar em memória que o processo anterior levou embora —
        e sem isso a retomada regravaria (ou pior, aceitaria em silêncio outra
        profundidade)."""
        with tempfile.TemporaryDirectory() as tmp:
            self._grava(tmp, [_sample("gp_abc_v00", "gp_abc")])
            stats = self._grava(tmp, [_sample("gp_abc_v01", "gp_abc")])
            self.assertEqual((stats.depth_files, stats.depth_reused), (0, 1))
            self.assertEqual(len(list((Path(tmp) / "depth").glob("*.png"))), 1)

    def test_a_retomada_detecta_profundidade_divergente_no_disco(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._grava(tmp, [_sample("gp_abc_v00", "gp_abc")])
            outra = _sample("gp_abc_v01", "gp_abc")
            outra.depth = encode_depth(_depth(near=2.0, far=90.0),
                                       image_hw=(64, 96), long_side=768)
            with self.assertRaises(ValueError) as ctx:
                self._grava(tmp, [outra])
            self.assertIn("difere da já gravada", str(ctx.exception))


# ==============================================================================
# O leitor — resolve os DOIS layouts, e as duas formas de empacotamento
# ==============================================================================
#
# Não há dataloader neste repositório: o treino da BokehNet vive noutro. O que existe é
# `dataio.read_depth_u16`, e ele é o leitor canônico — quem consumir um release chama
# isto ou copia esta resolução, e não inventa a terceira.

class LeituraDosDoisLayouts(unittest.TestCase):

    def test_le_o_release_deduplicado(self):
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("gp_abc")) as w:
                w.write(_sample("gp_abc_v00", "gp_abc"))
            u16 = read_depth_u16(tmp, read_metadata(tmp, "gp_abc_v00"))
            self.assertEqual(u16.dtype, np.uint16)
            self.assertGreater(int(u16.max()), 0)

    def test_le_o_release_ANTIGO_sem_o_campo(self):
        """`bokehnet-regen-rota-c` e o `-rota-a` foram gravados antes do campo existir.
        Quem lê tem que continuar lendo os dois — a compatibilidade é requisito, e um
        leitor que só entende o layout novo transforma dois releases publicados em
        dados ilegíveis."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("gp_abc"),
                                  depth_layout=DepthLayout.PER_SAMPLE) as w:
                w.write(_sample("gp_abc_v00", "gp_abc"))
            meta = read_metadata(tmp, "gp_abc_v00")
            esperado = read_depth_u16(tmp, meta)
            # o `meta/` de um release antigo não tem nenhum dos dois campos
            antigo = {k: v for k, v in meta.items()
                      if k not in ("depth_ref", "depth_layout")}
            np.testing.assert_array_equal(read_depth_u16(tmp, antigo), esperado)

    def test_le_o_release_REAGRUPADO_por_cena_do_Hub(self):
        """O upload insere `depth/<scene_id>/` para caber no limite de 10.000 arquivos
        por pasta do Hub. Quem clona o repositório recebe esse layout, não o do disco."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("gp_abc")) as w:
                w.write(_sample("gp_abc_v00", "gp_abc"))
            meta = read_metadata(tmp, "gp_abc_v00")
            esperado = read_depth_u16(tmp, meta)

            raiz = Path(tmp)
            (raiz / "depth" / "gp_abc").mkdir()
            (raiz / "depth" / "gp_abc.png").rename(
                raiz / "depth" / "gp_abc" / "gp_abc.png")
            np.testing.assert_array_equal(read_depth_u16(tmp, meta), esperado)

    def test_profundidade_ausente_levanta_dizendo_onde_procurou(self):
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("gp_abc")) as w:
                w.write(_sample("gp_abc_v00", "gp_abc"))
            meta = read_metadata(tmp, "gp_abc_v00")
            (Path(tmp) / meta["depth_ref"]).unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                read_depth_u16(tmp, meta)
            self.assertIn("depth/gp_abc.png", str(ctx.exception))

    def test_nao_adivinha_o_caminho_num_release_por_cena(self):
        """`depth_layout=per_scene` sem `depth_ref` é um release que afirma um layout e
        não diz onde o arquivo está. Cair de volta em `depth/<sample_id>.png` daria uma
        resposta ERRADA em vez de nenhuma."""
        with self.assertRaises(ValueError) as ctx:
            depth_ref_of({"sample_id": "gp_abc_v00", "scene_id": "gp_abc",
                          "depth_layout": "per_scene"})
        self.assertIn("não há convenção", str(ctx.exception))


# ==============================================================================
# `validate_metadata` e a referência da profundidade
# ==============================================================================

class ReferenciaDaProfundidadeNoMetadado(unittest.TestCase):

    def _meta(self, **over):
        meta = _sample("gp_abc_v00", "gp_abc").metadata()
        meta.update(over)
        return meta

    def test_meta_sem_os_campos_continua_valido(self):
        """Compatibilidade de leitura: `validate_metadata` é a MESMA função que o
        `publish_release.py` roda sobre os `meta/` dos releases já publicados. Exigir o
        campo aqui reprovaria retroativamente um release correto."""
        validate_metadata(self._meta())

    def test_um_campo_sem_o_outro_e_erro(self):
        with self.assertRaises(ValueError):
            validate_metadata(self._meta(depth_ref="depth/gp_abc.png"))
        with self.assertRaises(ValueError):
            validate_metadata(self._meta(depth_layout="per_scene"))

    def test_referencia_fora_do_layout_declarado_e_erro(self):
        """`depth/<sample_id>.png` num release que diz `per_scene` são 41 arquivos onde
        deveria haver 1 — ou um caminho que não existe."""
        with self.assertRaises(ValueError) as ctx:
            validate_metadata(self._meta(depth_ref="depth/gp_abc_v00.png",
                                         depth_layout="per_scene"))
        self.assertIn("não é o que", str(ctx.exception))

    def test_layout_fora_do_vocabulario_e_erro(self):
        with self.assertRaises(ValueError):
            validate_metadata(self._meta(depth_ref="depth/gp_abc.png",
                                         depth_layout="scene"))

    def test_par_coerente_passa(self):
        validate_metadata(self._meta(depth_ref="depth/gp_abc.png",
                                     depth_layout="per_scene"))
        validate_metadata(self._meta(depth_ref="depth/gp_abc_v00.png",
                                     depth_layout="per_sample"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
