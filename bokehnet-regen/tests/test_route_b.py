"""Rota B de ponta a ponta, com modelos falsos. Sem GPU, sem checkpoint.

O que este arquivo prova:

1. que a cadeia inteira fecha — DeblurNet -> Depth -> BiRefNet -> Eq. 4 -> Eq. 3 ->
   gates -> disco — e que o K que sobrevive ao disco é o K que a Eq. 3 dá **na mão**;
2. que os nove defeitos catalogados da rota B não voltam: cada um tem um teste com o
   número do defeito no nome;
3. que **cada caminho de rejeição dispara**, com slug do conjunto fechado;
4. que as duas decisões desta rota — qual `D_focus` alimenta a Eq. 3, e não aplicar o
   refinamento do §3.2(c) — estão gravadas por amostra e são auditáveis sem reprocessar.

Os dublês vivem aqui, nunca em `src/`. Uma segunda DeblurNet, um segundo backend de
profundidade ou um segundo segmentador em produção é exatamente a família de defeitos
B1/B7/D11.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from control.contract import (                                                # noqa: E402
    MAX_COC, REJECTION_REASONS, SampleRejected, decode_defocus_uint16,
    defocus_map, encode_defocus_uint16, k_from_exif, validate_metric_depth,
)
from dataio import (                                                          # noqa: E402
    MaskSource, build_scene_split, decode_depth_m, iter_manifest, read_metadata,
)
from model_runtime.deblurnet import DeblurredAIF, plan_processing              # noqa: E402
from qc.focus_region import FocusSource                                        # noqa: E402
from routes.route_b import (                                                   # noqa: E402
    AIF_IMAGE_NAME, GATE_TO_REASON, RouteBConfig, RouteBStats,
    deblur_structural_fidelity, exif_crop_factor_suspect, focus_region_record,
    k_in_range, pair_registration, phase_correlation_shift, process_sample,
    run_route_b,
)
from qc.rejection import RejectionLog                                          # noqa: E402
from model_runtime.deblurnet import DeblurVariant                              # noqa: E402
from run_route_b import (                                                      # noqa: E402
    ItwImageLoader, ItwRow, _confere_variante_uniforme, _enumera_itw, build_parser,
)
from test_renderer import _disc_kernel, _fft_convolve_same                     # noqa: E402


# ==============================================================================
# A fonte falsa — a forma do `BokehSource`
# ==============================================================================

@dataclass
class FakeSource:
    """Uma linha do ITW. Os valores da EXIF são os do exemplo aritmético dos testes.

    `focal_length_35mm = 75,0` com `focal_length_mm = 50,0` dá crop factor 1,5 e sensor
    de 24,0 mm — o APS-C que é a mediana medida do dataset (`ACHADOS.md:161-162`).
    """

    scene_id: str = "42"
    sample_id: str = "b_42"
    source_dataset: str = "atfortes/BokehDiffusion"
    source_sample_id: str = "42"
    source_split: Optional[str] = "train"
    bokeh_ref: Optional[str] = "image"
    focal_length_mm: Optional[float] = 50.0
    f_number: Optional[float] = 1.8
    focal_length_35mm: Optional[float] = 75.0
    exif_focus_distance_m: Optional[float] = None
    camera_make: Optional[str] = "Canon"
    camera_model: Optional[str] = "EOS 60D"


# ==============================================================================
# Modelos falsos
# ==============================================================================

class FakeDepth:
    """Gradiente de 1 a 40 m, com um objeto a EXATAMENTE 3 m no quadrante central.

    O objeto a 3,0 m exato é o que torna `focus_disparity = 1/3` independente da
    resolução — sem isso, o teste do defeito B3 (retrato) mudaria duas variáveis ao
    mesmo tempo e não provaria nada sobre `pixel_ratio`.
    """

    backend = "depth_pro"

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        z = np.linspace(1.0, 40.0, h * w, dtype=np.float32).reshape(h, w)
        z[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 3.0
        return validate_metric_depth(z, backend="depth_pro")


class DepthSentinela(FakeDepth):
    """O objeto em foco cai no TETO do Depth Pro: 10.000 m.

    Não é profundidade, é sentinela — e `z_max == 10000` no FUNDO é comportamento
    correto (em disparidade `1/10000 ≈ 0` satura graciosamente). O que é defeito é o
    FOCO no teto, e é isso que este dublê produz.
    """

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        z = np.linspace(1.0, 40.0, h * w, dtype=np.float32).reshape(h, w)
        z[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 10_000.0
        return validate_metric_depth(z, backend="depth_pro")


class FakeMask:
    """Máscara sobre o objeto de 3 m — o caso em que o BiRefNet acerta."""

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        return mask


class MascaraVazia:
    """O BiRefNet declinando — probabilidade 0 em toda a imagem."""

    def infer(self, image_rgb):
        return np.zeros(image_rgb.shape[:2], dtype=bool)


class MascaraIntegral:
    """Máscara cobrindo o quadro inteiro: não distingue foco de fundo."""

    def infer(self, image_rgb):
        return np.ones(image_rgb.shape[:2], dtype=bool)


class MascaraNoPlanoErrado:
    """Máscara sobre o topo do gradiente — o fotógrafo focou outra coisa."""

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 8, :] = True
        return mask


class MascaraBimodal:
    """Metade no objeto a 3 m, metade no fundo distante — em número IGUAL de pixels.

    Existe para tornar a Decisão 1 **testável**. Numa máscara sobre uma região de
    profundidade contínua, `1/median(1/z)` e `median(z)` praticamente coincidem: com
    contagem par, `np.median` faz a média dos dois valores centrais, e num gradiente fino
    esses dois são vizinhos — a média harmônica e a aritmética de dois números vizinhos
    diferem em O((z_a − z_b)²). É por isso que o defeito B5 é *"pequeno e sistemático"*.

    Com os dois valores centrais em **planos diferentes**, a divergência aparece em
    tamanho real, e é ela que a escolha da Decisão 1 resolve — em vez de dispensar.
    """

    def infer(self, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True     # o objeto a 3,0 m
        n_objeto = int(mask.sum())
        # A MESMA quantidade de pixels no fim do gradiente (o fundo). `ravel` percorre na
        # ordem em que `FakeDepth` monta o `linspace`, então o fim é o mais distante.
        plano = mask.ravel()
        fundo = np.zeros(h * w, dtype=bool)
        fundo[-n_objeto:] = True
        plano |= fundo & ~plano
        return plano.reshape(h, w)


class _DeblurBase:
    """Base dos dublês da DeblurNet: devolve um `DeblurredAIF` de verdade.

    Usa o `ProcessingPlan` real (`plan_processing`), que é função pura e não importa
    torch — então a proveniência que o teste vê tem a mesma forma da de produção.
    """

    variante = "official_cond_only"

    def _prov(self, image_hw):
        return {
            "deblur_backend": "genfocus_deblurnet",
            "deblur_variant": self.variante,
            "deblur_lora_sha256": "a" * 64,
            "deblur_repo_id": "nycu-cplab/Genfocus-Model",
            "deblur_weight_filename": "deblurNet.safetensors",
            "main_adapter": None,
            "deblur_lora_mode": "cond_only",
            "prompt": "a sharp photo with everything in focus",
            "num_inference_steps": 28,
            "geometry": plan_processing(image_hw).to_dict(),
        }

    def _pack(self, aif_rgb, image_hw):
        return DeblurredAIF(aif_rgb=np.asarray(aif_rgb, dtype=np.uint8),
                            plan=plan_processing(image_hw),
                            provenance=self._prov(image_hw))


class DeblurPerfeita(_DeblurBase):
    """Devolve a AIF que de fato gerou a bokeh. O caso ideal."""

    def __init__(self, aif_rgb):
        self.aif_rgb = np.asarray(aif_rgb, dtype=np.uint8)

    def infer(self, bokeh_rgb):
        return self._pack(self.aif_rgb, bokeh_rgb.shape[:2])


class DeblurIdentidade(_DeblurBase):
    """Devolve a ENTRADA. É a assinatura do LoRA não ter carregado — o defeito B1."""

    def infer(self, bokeh_rgb):
        return self._pack(bokeh_rgb, bokeh_rgb.shape[:2])


class DeblurLavada(_DeblurBase):
    """Ruído de alta frequência: variância de Laplaciano ALTA, estrutura NENHUMA.

    É o outro lado do defeito B1 — main+cond rodado como cond-only sai lavado — e é
    exatamente o caso que uma razão de nitidez sozinha **não** pega.
    """

    def infer(self, bokeh_rgb):
        rng = np.random.default_rng(7)
        return self._pack(rng.integers(0, 256, size=bokeh_rgb.shape, dtype=np.uint8),
                          bokeh_rgb.shape[:2])


class DeblurDeslocada(_DeblurBase):
    """AIF correta, mas deslocada. O defeito B17 / o `[A]` A13."""

    def __init__(self, aif_rgb, shift_px: int = 5):
        self.aif_rgb = np.asarray(aif_rgb, dtype=np.uint8)
        self.shift_px = int(shift_px)

    def infer(self, bokeh_rgb):
        return self._pack(np.roll(self.aif_rgb, self.shift_px, axis=1),
                          bokeh_rgb.shape[:2])


class DeblurQueMudaShape(_DeblurBase):
    """Devolve outra resolução. K vive na escala de pixel da fonte: tem que rejeitar."""

    def infer(self, bokeh_rgb):
        h, w = bokeh_rgb.shape[:2]
        return self._pack(bokeh_rgb[: h // 2, : w // 2], (h // 2, w // 2))


# ==============================================================================
# Cena sintética
# ==============================================================================

_NIVEIS_COC_PX = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _layered_bokeh(aif_bgr, depth_m, focus_disparity, k_value):
    """Bokeh que VARIA no espaço: cada pixel borra segundo o CoC do contrato.

    Composição de 7 camadas com interpolação linear — dublê, não o *scatter* do BokehMe.
    O que ele precisa reproduzir é que a região no plano de foco fica **mais nítida que o
    resto**, que é a hipótese física de `focus_mask_is_sharpest` e da retenção. Uma
    imagem uniformemente borrada não serve: nela não existe plano de foco, e a retenção
    passa a medir um empate numérico.
    """
    disp = 1.0 / np.asarray(depth_m, dtype=np.float64)
    coc = np.abs(float(k_value) * (disp - float(focus_disparity)))
    origem = np.asarray(aif_bgr, dtype=np.float64)

    versoes = [origem]
    for raio in _NIVEIS_COC_PX[1:]:
        versoes.append(np.stack(
            [_fft_convolve_same(origem[..., c], _disc_kernel(raio)) for c in range(3)],
            axis=-1))

    out = versoes[-1].copy()
    for i in range(len(_NIVEIS_COC_PX) - 1):
        lo, hi = _NIVEIS_COC_PX[i], _NIVEIS_COC_PX[i + 1]
        faixa = (coc >= lo) & (coc < hi)
        if not faixa.any():
            continue
        peso = ((coc - lo) / (hi - lo))[..., None]
        out[faixa] = ((1.0 - peso) * versoes[i] + peso * versoes[i + 1])[faixa]
    return np.clip(out, 0.0, 255.0)


def _cena(h=96, w=96, k_verdadeiro=18.0, seed=0):
    """`(aif_bgr, bokeh_bgr)` — AIF texturizada e a bokeh renderizada com K conhecido."""
    rng = np.random.default_rng(seed)
    bloco = rng.integers(0, 255, size=(max(h // 8, 1), max(w // 8, 1), 3), dtype=np.uint16)
    aif = np.repeat(np.repeat(bloco, 8, axis=0), 8, axis=1).astype(np.uint8)[:h, :w]
    depth = FakeDepth().infer(aif).values_m
    bokeh = _layered_bokeh(aif, depth, 1.0 / 3.0, k_verdadeiro).astype(np.uint8)
    return aif, bokeh


def _cena_lisa(h=96, w=96):
    liso = np.full((h, w, 3), 120, dtype=np.uint8)
    return liso, liso.copy()


_PROV = {
    "pipeline_commit": "abc1234",
    "depth_model_sha256": "d" * 64,
    "mask_model_sha256": "m" * 64,
    "mask_backend": "birefnet",
    "source_license": "flickr, ver o card da origem",
}


def _config(tmp=None, **kw):
    return RouteBConfig(output_dir=Path(tmp or "."), **kw)


def _processa(*, aif=None, bokeh=None, deblur=None, mask=None, config=None, source=None,
              depth_runtime=None):
    """Uma amostra, com os dublês. Devolve o `Sample`."""
    if aif is None or bokeh is None:
        aif, bokeh = _cena()
    aif_rgb = np.ascontiguousarray(aif[..., ::-1])
    return process_sample(
        source or FakeSource(),
        bokeh_bgr=bokeh,
        deblur_runtime=deblur or DeblurPerfeita(aif_rgb),
        depth_runtime=depth_runtime or FakeDepth(),
        mask_runtime=mask or FakeMask(),
        config=config or _config(),
        provenance_base=_PROV,
    )


def _k_esperado(image_hw, source=None):
    """A Eq. 3 na mão, pelo caminho do contrato. O gabarito dos testes de K."""
    src = source or FakeSource()
    k, _ = k_from_exif(focal_length_mm=src.focal_length_mm, f_number=src.f_number,
                       focal_length_35mm=src.focal_length_35mm,
                       focus_depth_m=3.0, image_hw=image_hw)
    return k


# ==============================================================================
# Uma amostra, ponta a ponta
# ==============================================================================

class UmaAmostra(unittest.TestCase):
    def test_K_gravado_e_a_Eq3_calculada_na_mao(self):
        amostra = _processa()
        # `places=7`, não 9: a profundidade é float32, então `1/median(1/z)` não é 3,0
        # exato e a diferença relativa fica em 1e-9. Exigir 9 casas aqui trancaria a
        # precisão do dtype, não a fórmula.
        self.assertAlmostEqual(amostra.control.k_value, _k_esperado((96, 96)), places=7)
        self.assertEqual(amostra.control.k_source.value, "eq3_exif")

    def test_foco_vem_da_mascara_pela_Eq4_na_disparidade(self):
        amostra = _processa()
        # A máscara cobre exatamente o objeto a 3,0 m: `median(1/z[M]) == 1/3`.
        self.assertAlmostEqual(amostra.control.focus_disparity, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(amostra.control.focus_depth_m, 3.0, places=5)

    def test_a_rota_nao_tem_renderizador(self):
        """paper.txt:271,283,292,321 — o alvo da rota B é a própria fotografia.

        O antigo gravava `{"name": "not_applicable_route_b",
        "is_final_label_renderer": True}` (`route_b.py:121`): proveniência que mente na
        direção tranquilizadora (defeito B14).
        """
        amostra = _processa()
        self.assertIsNone(amostra.provenance.renderer)
        self.assertIsNone(amostra.control.k_effective_factor)
        self.assertIsNone(amostra.control.calibration_ssim)
        self.assertFalse(amostra.control.is_k_censored)

    def test_proveniencia_da_DeblurNet_completa(self):
        """Defeito B1: sem variante e sem sha do LoRA não dá para provar se a AIF saiu
        lavada. E a proveniência tem que dizer o `main_adapter` EFETIVO."""
        prov = _processa().provenance.deblurnet
        self.assertEqual(prov["deblur_variant"], "official_cond_only")
        self.assertEqual(len(prov["deblur_lora_sha256"]), 64)
        self.assertEqual(prov["deblur_lora_mode"], "cond_only")
        self.assertIn("main_adapter", prov)
        self.assertIn("geometry", prov)

    def test_hash_dos_TRES_modelos_que_influenciaram_o_rotulo(self):
        """O `CLAUDE.md` exige DeblurNet, DepthPro e **BiRefNet**. O antigo omitia o
        BiRefNet (`control_contract.py:196-200`) — e é ele que define `D_focus`, que
        multiplica direto no K."""
        prov = _processa().provenance
        self.assertEqual(len(prov.depth_model_sha256), 64)
        self.assertEqual(len(prov.mask_model_sha256), 64)
        self.assertEqual(len(prov.deblurnet["deblur_lora_sha256"]), 64)

    def test_a_AIF_e_gravada_e_a_bokeh_e_referencia(self):
        amostra = _processa()
        self.assertIn(AIF_IMAGE_NAME, amostra.generated_images)
        self.assertEqual(amostra.channel_order, "bgr")
        self.assertIsNone(amostra.refs.aif_ref)          # produto, não referência
        self.assertEqual(amostra.refs.bokeh_ref, "image")


# ==============================================================================
# Os defeitos catalogados
# ==============================================================================

class DefeitosQueNaoVoltam(unittest.TestCase):
    def test_B2_o_K_nao_esta_1000x_maior(self):
        """`k_eq3` em px·mm contra disparidade em 1/m dava 1000x. A âncora fecha:
        `k_official(16553,9) = 16,55` e o default oficial é 15,0."""
        k = _processa().control.k_value
        self.assertLess(k, 1000.0, "K na casa dos milhares é a convenção px·mm")
        self.assertGreater(k, 1e-3)
        # O CoC que esse K produz tem que caber em pixels de uma foto, não em milhares.
        depth = FakeDepth().infer(np.zeros((96, 96, 3), np.uint8)).values_m
        mapa = defocus_map(depth, 1.0 / 3.0, k)
        self.assertLess(float(mapa.max()), 1.0,
                        "mapa saturado em 1,0 é o fator 1000x apagando o K")

    def test_B3_pixel_ratio_usa_o_MAIOR_LADO_nao_a_largura(self):
        """paper.txt:1186-1187: *"the image's largest edge length divided by the physical
        sensor width"*. Em retrato, usar a largura subestima K por W/H."""
        aif, bokeh = _cena(h=96, w=64)                    # retrato 2:3
        amostra = _processa(aif=aif, bokeh=bokeh)
        esperado_maior_lado = _k_esperado((96, 64))
        self.assertAlmostEqual(amostra.control.k_value, esperado_maior_lado, places=7)

        diag = amostra.provenance.extra["k_eq3_diagnostics"]
        self.assertEqual(diag["longest_edge_px"], 96)
        self.assertAlmostEqual(diag["pixel_ratio_px_per_mm"], 96.0 / 24.0, places=9)
        # E o valor que sairia da LARGURA é 1,5x menor: é o erro que este teste tranca.
        self.assertAlmostEqual(amostra.control.k_value * (64.0 / 96.0),
                               _k_esperado((64, 64)), places=7)

    def test_B4_max_coc_nao_e_configuravel_em_lugar_nenhum(self):
        self.assertNotIn("max_coc", RouteBConfig.__dataclass_fields__)
        meta = _processa().metadata()
        self.assertEqual(meta["max_coc"], MAX_COC)

    def test_B6_sem_focal_35_rejeita_em_vez_de_assumir_36mm(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(source=FakeSource(focal_length_35mm=None))
        self.assertEqual(ctx.exception.reason, "sensor_width_unresolvable")

    def test_B6_sem_focal_length_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(source=FakeSource(focal_length_mm=None))
        self.assertEqual(ctx.exception.reason, "exif_focal_length_missing")

    def test_B6_sem_f_number_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(source=FakeSource(f_number=None))
        self.assertEqual(ctx.exception.reason, "exif_f_number_missing")

    def test_B7_segmentador_unico_falha_REJEITA_com_slug(self):
        """Não há cascata BiRefNet -> RMBG -> GrabCut, e `mask_source` não pode dizer
        `automatic` — o enum não tem esse valor."""
        with self.assertRaises(SampleRejected) as ctx:
            _processa(mask=MascaraVazia())
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")
        self.assertNotIn("AUTOMATIC", {m.name for m in MaskSource})

    def test_B8_profundidade_gravada_em_DISPARIDADE_nao_minmax_metrica(self):
        amostra = _processa()
        self.assertEqual(amostra.depth.to_metadata()["depth_encoding"],
                         "uint16_linear_in_disparity")

    def test_B11_K_viaja_com_a_resolucao_em_que_foi_medido(self):
        meta = _processa().metadata()
        for campo in ("image_h", "image_w", "depth_h", "depth_w"):
            self.assertIn(campo, meta)
        self.assertEqual((meta["image_h"], meta["image_w"]), (96, 96))

    def test_B12_sample_id_vem_da_CENA_nao_da_posicao_no_dataset(self):
        """`stem = f"b_{index:06d}"` sobre o dataset filtrado renumerava tudo quando o
        filtro mudava, e a retomada por stem passava a pular as amostras erradas."""
        amostra = _processa(source=FakeSource(scene_id="9911", sample_id="b_9911"))
        self.assertEqual(amostra.sample_id, "b_9911")
        self.assertEqual(amostra.refs.scene_id, "9911")

    def test_B16_s1_nao_existe_nem_por_compatibilidade(self):
        meta = _processa().metadata()
        self.assertNotIn("s1", meta)
        self.assertNotIn("s1", json.dumps(meta))

    def test_B17_AIF_em_outra_resolucao_REJEITA(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(deblur=DeblurQueMudaShape())
        self.assertEqual(ctx.exception.reason, "resolution_invalid")

    def test_bokeh_que_nao_e_HxWx3_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(bokeh=np.zeros((16, 16), dtype=np.uint8), aif=_cena()[0])
        self.assertEqual(ctx.exception.reason, "resolution_invalid")


# ==============================================================================
# Decisão 1 — qual `D_focus` alimenta a Eq. 3
# ==============================================================================

class Decisao1FocusDepth(unittest.TestCase):
    def test_a_Eq3_recebe_1_sobre_focus_disparity(self):
        amostra = _processa()
        diag = amostra.provenance.extra["focus_depth_diagnostics"]
        self.assertEqual(diag["focus_depth_convention"], "1/median(1/z[M])")
        self.assertAlmostEqual(diag["focus_depth_m_from_disparity"],
                               1.0 / amostra.control.focus_disparity, places=9)
        # É o valor que entrou na Eq. 3, e o diagnóstico da Eq. 3 confirma.
        self.assertAlmostEqual(
            amostra.provenance.extra["k_eq3_diagnostics"]["focus_depth_m"],
            diag["focus_depth_m_from_disparity"], places=9)

    def test_as_DUAS_leituras_sao_gravadas_lado_a_lado(self):
        """O desvio da literalidade de paper.txt:352 é declarado E medido: `median(z[M])`
        fica ao lado, para quem quiser refazer o rótulo pela leitura literal."""
        diag = _processa().provenance.extra["focus_depth_diagnostics"]
        for campo in ("focus_depth_m_from_disparity", "focus_depth_m_median_z",
                      "focus_depth_ratio"):
            self.assertIn(campo, diag)
            self.assertTrue(np.isfinite(diag[campo]))

    def test_a_divergencia_e_PEQUENA_num_plano_continuo(self):
        """O tamanho real do defeito B5, medido aqui: *"pequeno e sistemático"*.

        Com contagem par, `np.median` faz a média dos dois centrais; num gradiente fino
        esses dois são vizinhos, e média harmônica e aritmética de vizinhos diferem em
        O((z_a − z_b)²). Este teste existe para que ninguém leia a Decisão 1 como se ela
        estivesse escolhendo entre dois números muito diferentes no caso comum.
        """
        diag = _processa(mask=MascaraNoPlanoErrado()) \
            .provenance.extra["focus_depth_diagnostics"]
        self.assertAlmostEqual(diag["focus_depth_ratio"], 1.0, places=5)

    def test_as_duas_leituras_DIVERGEM_quando_a_mascara_cruza_DOIS_planos(self):
        """E aqui a divergência é enorme — o caso que a Decisão 1 de fato resolve.

        Máscara metade no objeto a 3 m, metade no fundo, em número igual de pixels: os
        dois valores centrais da mediana caem em planos diferentes.
        """
        diag = _processa(mask=MascaraBimodal()) \
            .provenance.extra["focus_depth_diagnostics"]
        self.assertLess(diag["focus_depth_ratio"], 0.5)
        self.assertGreater(diag["focus_depth_m_median_z"],
                           2.0 * diag["focus_depth_m_from_disparity"])

    def test_median_z_NUNCA_entra_no_rotulo(self):
        amostra = _processa(mask=MascaraBimodal())
        diag = amostra.provenance.extra["focus_depth_diagnostics"]
        k_pelo_literal = k_from_exif(
            focal_length_mm=50.0, f_number=1.8, focal_length_35mm=75.0,
            focus_depth_m=diag["focus_depth_m_median_z"], image_hw=(96, 96))[0]
        self.assertNotAlmostEqual(amostra.control.k_value, k_pelo_literal, places=6)
        self.assertAlmostEqual(
            amostra.control.k_value,
            k_from_exif(focal_length_mm=50.0, f_number=1.8, focal_length_35mm=75.0,
                        focus_depth_m=diag["focus_depth_m_from_disparity"],
                        image_hw=(96, 96))[0], places=7)


# ==============================================================================
# Decisão 2 — sem refinamento da região em foco
# ==============================================================================

class Decisao2SemRefinamento(unittest.TestCase):
    def test_a_regiao_e_a_mascara_do_BiRefNet_como_veio(self):
        """paper.txt:361-365: o refinamento é introduzido em (c), com *"Similar to (b)"*
        marcando (b) como o caminho sem refino."""
        amostra = _processa()
        self.assertIs(FocusSource(amostra.focus.source), FocusSource.BIREFNET)
        self.assertFalse(amostra.focus.was_refined)
        self.assertIs(MaskSource(amostra.mask_source), MaskSource.BIREFNET)

    def test_mascara_no_plano_ERRADO_NAO_e_corrigida(self):
        """Na rota C esta amostra viraria `birefnet_refined`. Aqui ela segue como veio: o
        rótulo é o que o BiRefNet disse, e o gate `focus_mask_not_sharpest` é quem julga."""
        amostra = _processa(mask=MascaraNoPlanoErrado())
        self.assertIs(FocusSource(amostra.focus.source), FocusSource.BIREFNET)
        self.assertFalse(amostra.focus.was_refined)

    def test_mascara_vazia_NAO_vira_retention_only(self):
        """A diferença mais visível entre as duas rotas: na C isto é aceito e marcado;
        aqui é rejeição contada no histograma, que é o número que decide se a Decisão 2
        se sustenta."""
        with self.assertRaises(SampleRejected) as ctx:
            _processa(mask=MascaraVazia())
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")

    def test_a_decisao_fica_GRAVADA_na_amostra_com_a_linha_do_paper(self):
        decisoes = _processa().provenance.extra["route_b_decisions"]
        self.assertFalse(decisoes["focus_refinement_applied"])
        self.assertIn("paper.txt:361-365", decisoes["focus_refinement_evidence"])
        self.assertEqual(decisoes["d_focus_for_eq3"], "1/focus_disparity")

    def test_a_mascara_GRAVADA_e_a_que_produziu_o_foco(self):
        amostra = _processa()
        esperada = FakeMask().infer(np.zeros((96, 96, 3), np.uint8))
        np.testing.assert_array_equal(np.asarray(amostra.mask) > 0.5, esperada)

    def test_a_retencao_e_MEDIDA_mesmo_sem_refinar(self):
        """`focus_agreement` é o instrumento que decide se a Decisão 2 se sustenta no
        piloto. Ele não pode faltar só porque não houve refinamento."""
        amostra = _processa()
        self.assertGreater(amostra.focus.retention_hw[0], 0)
        self.assertEqual(amostra.focus.retention_hw, (96, 96))
        self.assertTrue(0.0 <= amostra.focus.agreement <= 1.0)

    def test_a_retencao_e_medida_na_resolucao_da_IMAGEM_sempre(self):
        """Sem knob de grade: um knob que muda a grade de um diagnóstico só acrescenta um
        eixo de incomparabilidade entre metades do release."""
        self.assertNotIn("focus_retention_long_side", RouteBConfig.__dataclass_fields__)
        aif, bokeh = _cena(h=96, w=64)
        self.assertEqual(_processa(aif=aif, bokeh=bokeh).focus.retention_hw, (96, 64))

    def test_nao_ha_linha_de_base_pareada_porque_a_mascara_crua_E_o_rotulo(self):
        self.assertIsNone(_processa().focus.disparity_from_initial_mask)

    def test_cena_sem_detalhe_nenhum_ainda_rejeita_com_slug(self):
        liso, liso2 = _cena_lisa()
        with self.assertRaises(SampleRejected) as ctx:
            focus_region_record(aif_bgr=liso, bokeh_bgr=liso2,
                                mask=np.zeros((96, 96), bool), config=_config())
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")


# ==============================================================================
# A AIF é produto de modelo — os dois eixos do gate
# ==============================================================================

class QualidadeDaAIF(unittest.TestCase):
    def test_identidade_tem_SSIM_1_e_razao_de_nitidez_1(self):
        """O defeito B1 pelo lado da identidade: os DOIS eixos apontam para 1,0."""
        aif, bokeh = _cena()
        amostra = _processa(aif=aif, bokeh=bokeh, deblur=DeblurIdentidade())
        q = amostra.quality
        self.assertGreater(q["deblur_structural_ssim_min"]["value"], 0.999)
        self.assertAlmostEqual(q["bokeh_over_aif_sharpness"]["value"], 1.0, places=6)

    def test_identidade_tambem_apaga_a_DISPERSAO_da_retencao(self):
        """O segundo eixo pedido pela auditoria: retenção concentrada em 1,0 significa
        "a DeblurNet não fez nada", e é isso que o refinamento da rota C mascararia."""
        aif, bokeh = _cena()
        diag = _processa(aif=aif, bokeh=bokeh,
                         deblur=DeblurIdentidade()).provenance.extra["deblur_diagnostics"]
        self.assertLess(diag["deblur_retention_spread"], 1e-6)
        self.assertAlmostEqual(diag["deblur_retention_p50"], 1.0, places=6)

    def test_AIF_lavada_tem_SSIM_baixo_e_nitidez_ALTA(self):
        """O caso que a razão de nitidez sozinha NÃO pega: variância de Laplaciano alta
        por artefato, estrutura da entrada perdida."""
        aif, bokeh = _cena()
        q = _processa(aif=aif, bokeh=bokeh, deblur=DeblurLavada()).quality
        self.assertLess(q["deblur_structural_ssim_min"]["value"], 0.2)
        self.assertLess(q["bokeh_over_aif_sharpness"]["value"], 0.2)

    def test_deblur_bem_feita_fica_ENTRE_os_dois_extremos(self):
        """Se os dois limiares fossem inúteis, este caso cairia num dos extremos."""
        q = _processa().quality
        valor = q["deblur_structural_ssim_min"]["value"]
        self.assertGreater(valor, 0.2)
        self.assertLess(valor, 0.999)

    def test_piso_de_SSIM_bloqueia_com_slug_do_conjunto_fechado(self):
        aif, bokeh = _cena()
        with self.assertRaises(SampleRejected) as ctx:
            _processa(aif=aif, bokeh=bokeh, deblur=DeblurLavada(),
                      config=_config(min_deblur_structural_ssim=0.5))
        self.assertEqual(ctx.exception.reason, "gate_aif_sharpness")

    def test_teto_de_SSIM_bloqueia_a_identidade(self):
        aif, bokeh = _cena()
        with self.assertRaises(SampleRejected) as ctx:
            _processa(aif=aif, bokeh=bokeh, deblur=DeblurIdentidade(),
                      config=_config(max_deblur_structural_ssim=0.98))
        self.assertEqual(ctx.exception.reason, "gate_bokeh_not_blurrier")

    def test_razao_de_nitidez_bloqueia_com_slug(self):
        aif, bokeh = _cena()
        with self.assertRaises(SampleRejected) as ctx:
            _processa(aif=aif, bokeh=bokeh, deblur=DeblurIdentidade(),
                      config=_config(max_bokeh_over_aif_sharpness=0.9))
        self.assertEqual(ctx.exception.reason, "gate_bokeh_not_blurrier")

    def test_nitidez_absoluta_da_AIF_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(min_aif_laplacian_variance=1e9))
        self.assertEqual(ctx.exception.reason, "gate_aif_sharpness")

    def test_os_dois_eixos_do_SSIM_saem_do_MESMO_valor_medido(self):
        """Um valor, dois limiares. Medir duas vezes daria dois números para a mesma
        grandeza — a forma de "cópias divergem" dentro de uma função."""
        aif, bokeh = _cena()
        piso, teto = deblur_structural_fidelity(bokeh, aif)
        self.assertEqual(piso.value, teto.value)
        self.assertTrue(piso.higher_is_better)
        self.assertFalse(teto.higher_is_better)
        self.assertIsNone(piso.threshold)
        self.assertIsNone(teto.threshold)

    def test_gate_de_K_mede_e_nao_bloqueia_sem_limiar(self):
        minimo, maximo = k_in_range(16.55)
        self.assertEqual(minimo.value, 16.55)
        self.assertTrue(minimo.passed and maximo.passed)
        self.assertTrue(minimo.measured_only and maximo.measured_only)

    def test_sem_limiar_nenhum_gate_bloqueia(self):
        report_keys = _processa().quality
        self.assertTrue(all(g["passed"] for g in report_keys.values()),
                        [n for n, g in report_keys.items() if not g["passed"]])


# ==============================================================================
# Registro do par — o defeito B17 e o [A] A13
# ==============================================================================

class RegistroDoPar(unittest.TestCase):
    def test_correlacao_de_fase_acha_um_deslocamento_conhecido(self):
        rng = np.random.default_rng(3)
        a = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        b = np.roll(a, 5, axis=1)
        dy, dx, resposta = phase_correlation_shift(a, b)
        self.assertEqual((dy, dx), (0.0, -5.0))
        self.assertGreater(resposta, 5.0)

    def test_par_alinhado_da_deslocamento_zero(self):
        aif, bokeh = _cena()
        resultados = {r.name: r for r in pair_registration(bokeh, aif)}
        self.assertEqual(resultados["pair_registration_shift_px"].value, 0.0)

    def test_AIF_deslocada_e_MEDIDA_em_pixel(self):
        """`ProcessingPlan` mede o que o redimensionamento faz; só a correlação de fase
        mede o que o MODELO faz. O `[A]` A13 nunca foi medido."""
        aif, bokeh = _cena()
        aif_rgb = np.ascontiguousarray(aif[..., ::-1])
        q = _processa(aif=aif, bokeh=bokeh,
                      deblur=DeblurDeslocada(aif_rgb, 5)).quality
        self.assertAlmostEqual(q["pair_registration_shift_px"]["value"], 5.0, places=6)

    def test_deslocamento_acima_do_limiar_bloqueia_com_slug(self):
        aif, bokeh = _cena()
        aif_rgb = np.ascontiguousarray(aif[..., ::-1])
        with self.assertRaises(SampleRejected) as ctx:
            _processa(aif=aif, bokeh=bokeh, deblur=DeblurDeslocada(aif_rgb, 12),
                      config=_config(max_pair_registration_shift_px=6.0))
        self.assertEqual(ctx.exception.reason, "gate_pair_shape_mismatch")

    def test_confianca_minima_da_correlacao_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(min_pair_registration_response=1e9))
        self.assertEqual(ctx.exception.reason, "gate_pair_shape_mismatch")


# ==============================================================================
# Máscara, profundidade e K — os gates herdados
# ==============================================================================

class GatesHerdados(unittest.TestCase):
    def test_area_minima_da_mascara_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(min_mask_area_ratio=0.9))
        self.assertEqual(ctx.exception.reason, "gate_mask_area_ratio")

    def test_area_maxima_da_mascara_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(mask=MascaraIntegral(), config=_config(max_mask_area_ratio=0.9))
        self.assertEqual(ctx.exception.reason, "gate_mask_area_ratio")

    def test_mascara_colada_na_borda_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(mask=MascaraIntegral(),
                      config=_config(max_mask_border_coverage=0.5))
        self.assertEqual(ctx.exception.reason, "gate_mask_border_coverage")

    def test_mascara_fora_da_regiao_mais_nitida_bloqueia_com_slug(self):
        """O gate que confere a HIPÓTESE FÍSICA, e que na rota B é mais forte que na C:
        aqui a bokeh real é o ALVO, então ela está sempre disponível."""
        with self.assertRaises(SampleRejected) as ctx:
            _processa(mask=MascaraNoPlanoErrado(),
                      config=_config(min_focus_mask_sharpness_ratio=1.0))
        self.assertEqual(ctx.exception.reason, "gate_focus_mask_not_sharpest")

    def test_IoU_entre_mascara_da_AIF_e_da_bokeh_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(min_mask_iou=1.1))
        self.assertEqual(ctx.exception.reason, "gate_mask_iou")

    def test_niveis_uteis_de_profundidade_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(min_depth_useful_levels=10 ** 9))
        self.assertEqual(ctx.exception.reason, "gate_depth_useful_levels")

    def test_plano_de_foco_no_TETO_do_Depth_Pro_rejeita_com_slug(self):
        """`z_focus = 10.000 m` não é plano de foco, é sentinela.

        A rejeição vem do **contrato** (`focus_disparity_from_mask`), que roda antes dos
        gates — e o gate `focus_depth_m_max` usa as MESMAS constantes importadas
        (`gates.py:37`), então ele confirma o veredito em vez de duplicá-lo. Exatamente
        como na rota C: quem tranca a faixa é o contrato.
        """
        with self.assertRaises(SampleRejected) as ctx:
            _processa(depth_runtime=DepthSentinela())
        self.assertEqual(ctx.exception.reason, "focus_depth_implausible")

    def test_K_fora_da_faixa_configurada_REJEITA_e_nao_censura(self):
        """Não há varredura, logo não há teto do qual censurar. O `--k-max 300` da rota C
        virou 47,0% de amostras no teto exato (`ACHADOS.md:19`) — aqui isso é impossível
        porque o caminho é rejeição."""
        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(max_k_value=0.001))
        self.assertEqual(ctx.exception.reason, "k_out_of_configured_range")

        with self.assertRaises(SampleRejected) as ctx:
            _processa(config=_config(min_k_value=1e6))
        self.assertEqual(ctx.exception.reason, "k_out_of_configured_range")

    def test_todo_slug_do_mapa_de_gates_esta_no_conjunto_FECHADO(self):
        for gate, slug in GATE_TO_REASON.items():
            self.assertIn(slug, REJECTION_REASONS, f"{gate} -> {slug}")

    def test_todo_gate_medido_vai_para_o_metadado(self):
        q = _processa().metadata()["quality"]
        for esperado in ("deblur_structural_ssim_min", "deblur_structural_ssim_max",
                         "pair_registration_shift_px", "pair_registration_response",
                         "focus_mask_sharpness_ratio", "exif_crop_factor_unity",
                         "k_value_min", "k_value_max", "depth_useful_levels"):
            self.assertIn(esperado, q)
            self.assertIn("value", q[esperado])
            self.assertIn("threshold", q[esperado])


# ==============================================================================
# EXIF — a marcação dos 30,33%
# ==============================================================================

class ExifESensor(unittest.TestCase):
    def test_crop_factor_1_e_MARCADO_e_nunca_bloqueia(self):
        """`ACHADOS.md:169-173`: a tabela `make/model` serve para AUDITAR esses 30%, não
        para preencher lacuna. O gate não recebe limiar — por construção."""
        resultado = exif_crop_factor_suspect(1.0, make="Apple", model="iPhone 12")
        self.assertEqual(resultado.value, 1.0)
        self.assertIsNone(resultado.threshold)
        self.assertTrue(resultado.passed)
        self.assertNotIn("exif_crop_factor_unity", GATE_TO_REASON)

    def test_crop_factor_medido_nao_e_marcado(self):
        self.assertEqual(exif_crop_factor_suspect(1.5).value, 0.0)

    def test_full_frame_de_verdade_e_marcado_e_ACEITO(self):
        amostra = _processa(source=FakeSource(focal_length_mm=50.0,
                                             focal_length_35mm=50.0))
        self.assertEqual(amostra.quality["exif_crop_factor_unity"]["value"], 1.0)
        self.assertAlmostEqual(
            amostra.provenance.extra["k_eq3_diagnostics"]["sensor_width_mm"],
            36.0, places=9)

    def test_a_camera_vai_para_a_proveniencia_de_TODA_amostra(self):
        """Sem o denominador (quantas amostras cada modelo produziu no total) não dá para
        responder "quais modelos ecoam a focal no campo de 35 mm?"."""
        extra = _processa().provenance.extra
        self.assertEqual(extra["camera_make"], "Canon")
        self.assertEqual(extra["camera_model"], "EOS 60D")

    def test_diagnostico_da_Eq3_completo_no_metadado(self):
        diag = _processa().metadata()["provenance"]["extra"]["k_eq3_diagnostics"]
        for campo in ("k_eq3_mm", "k_value", "sensor_width_mm", "crop_factor",
                      "pixel_ratio_px_per_mm", "longest_edge_px", "focal_length_mm",
                      "f_number", "focal_length_35mm", "focus_depth_m"):
            self.assertIn(campo, diag)
        self.assertAlmostEqual(diag["sensor_width_mm"], 24.0, places=9)
        self.assertAlmostEqual(diag["crop_factor"], 1.5, places=9)

    def test_validador_pela_distancia_da_EXIF_e_INDEPENDENTE_do_rotulo(self):
        """paper.txt:344-346 proíbe a EXIF como `D_focus` do rótulo. Como VALIDADOR ela é
        exatamente o que se quer: não passa pelo Depth Pro nem pelo BiRefNet."""
        amostra = _processa(source=FakeSource(exif_focus_distance_m=5.0))
        validador = amostra.provenance.extra["k_validator_exif_focus_distance"]
        self.assertIsNotNone(validador)
        # 5,0 m contra os 3,0 m estimados: números diferentes, e é isso que valida.
        self.assertNotAlmostEqual(validador, amostra.control.k_value, places=6)
        self.assertAlmostEqual(
            validador,
            k_from_exif(focal_length_mm=50.0, f_number=1.8, focal_length_35mm=75.0,
                        focus_depth_m=5.0, image_hw=(96, 96))[0], places=9)

    def test_sem_distancia_na_EXIF_o_validador_e_None_e_nao_rejeita(self):
        amostra = _processa()
        self.assertIsNone(amostra.provenance.extra["k_validator_exif_focus_distance"])
        self.assertIsNone(amostra.control.k_analytic,
                          "k_analytic com o próprio rótulo dentro seria validador falso")


# ==============================================================================
# O laço, o disco e a retomada
# ==============================================================================

def _fontes(n=3):
    return [FakeSource(scene_id=str(i), sample_id=f"b_{i}", source_sample_id=str(i))
            for i in range(n)]


def _roda(tmp, fontes=None, deblur=None, mask=None, **kw):
    fontes = fontes if fontes is not None else _fontes()
    aif, bokeh = _cena()
    aif_rgb = np.ascontiguousarray(aif[..., ::-1])
    split = build_scene_split({f.scene_id for f in fontes}, val_fraction=0.34)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        stats = run_route_b(
            fontes, load_bokeh=lambda _: bokeh,
            deblur_runtime=deblur or DeblurPerfeita(aif_rgb),
            depth_runtime=FakeDepth(), mask_runtime=mask or FakeMask(),
            split=split, config=_config(tmp, **kw), provenance_base=_PROV)
    return stats, buffer.getvalue()


class Laco(unittest.TestCase):
    def test_gera_grava_e_o_K_sobrevive_ao_disco(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats, _ = _roda(tmp)
            self.assertEqual(stats.written, 3)
            meta = read_metadata(tmp, "b_0")
            self.assertAlmostEqual(meta["k_value"], _k_esperado((96, 96)), places=7)
            self.assertEqual(meta["route"], "b")
            self.assertEqual(meta["k_source"], "eq3_exif")

    def test_a_AIF_gerada_chega_ao_disco_com_a_cor_certa(self):
        """`channel_order` DECLARADO: a inversão cega transformava RGB [200,0,0] em
        [0,0,200] sem que nada registrasse a convenção."""
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            aif, bokeh = _cena()
            aif_rgb = np.ascontiguousarray(aif[..., ::-1])
            _roda(tmp, fontes=[FakeSource()], deblur=DeblurPerfeita(aif_rgb))
            caminho = Path(tmp) / "generated" / f"b_42_{AIF_IMAGE_NAME}.jpg"
            self.assertTrue(caminho.is_file())
            lida_rgb = np.asarray(Image.open(caminho).convert("RGB"))
            # JPEG q95 é lossy: compara em média, não pixel a pixel.
            self.assertLess(float(np.abs(lida_rgb.astype(float)
                                         - aif_rgb.astype(float)).mean()), 12.0)

    def test_o_ledger_hasheia_o_JPEG_EM_DISCO_nao_o_array(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp, fontes=[FakeSource()])
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "generated_images.jsonl").read_text().splitlines()]
            self.assertEqual(len(linhas), 1)
            linha = linhas[0]
            self.assertEqual(linha["aif_role"], "generated_by_deblurnet")
            self.assertEqual(linha["bokeh_role"], "reference")
            self.assertEqual(linha["deblur_variant"], "official_cond_only")
            arquivo = Path(tmp) / "generated" / f"b_42_{AIF_IMAGE_NAME}.jpg"
            self.assertEqual(linha["aif_jpeg_sha256"],
                             hashlib.sha256(arquivo.read_bytes()).hexdigest())

    def test_histograma_de_rejeicao_sempre_sai(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, saida = _roda(tmp, mask=MascaraVazia())
            self.assertIn("focus_mask_empty", saida)
            self.assertIn("motivos de rejeição", saida)
            self.assertIn("nenhuma amostra aceita", saida)

    def test_rejeicao_vai_para_o_jsonl_com_slug_e_cena(self):
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp, mask=MascaraVazia())
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "rejections.jsonl").read_text().splitlines()]
            self.assertEqual(len(linhas), 3)
            self.assertTrue(all(l["reason"] == "focus_mask_empty" for l in linhas))
            self.assertTrue(all(l["route"] == "b" for l in linhas))

    def test_retomada_pula_o_que_ja_foi_gravado(self):
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp)
            stats, saida = _roda(tmp)
            self.assertEqual(stats.written, 0)
            self.assertEqual(stats.skipped_done, 3)
            self.assertIn("retomando", saida)

    def test_o_manifesto_carrega_os_campos_que_denunciam_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp)
            linhas = list(iter_manifest(tmp))
            self.assertEqual(len(linhas), 3)
            for linha in linhas:
                self.assertEqual(linha["max_coc"], MAX_COC)
                self.assertEqual(linha["mask_source"], "birefnet")
                self.assertEqual(linha["depth_backend"], "depth_pro")
                self.assertEqual(linha["focus_source"], "birefnet")
                self.assertFalse(linha["focus_was_refined"])
                self.assertIn(linha["split"], ("train", "val"))

    def test_split_por_CENA_materializado(self):
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp)
            split = json.loads((Path(tmp) / "split.json").read_text())
            self.assertEqual(len(split["assignment"]), 3)

    def test_a_dedup_de_profundidade_NAO_e_forcada_nesta_rota(self):
        """A rota B é 1:1 — `scene_id` é o `flickr_photo_id` e `sample_id` é
        `b_<scene_id>` (`scripts/run_route_b.py:164-165`), uma função injetiva, e
        `run_route_b.py:289` ainda rejeita id repetido com `source_duplicate_sample`.

        Medido sobre as 13.800 linhas que passam o filtro: 13.800 cenas, **1,00 amostra
        por cena, 0,0% de economia**. Ligar a dedup aqui renomearia 13.800 arquivos de
        `depth/b_<id>.png` para `depth/<id>.png` sem poupar um byte — mudança de
        contrato de caminho em troca de nada.
        """
        from routes.route_b import RouteBConfig
        from dataio import DepthLayout
        self.assertIs(RouteBConfig.depth_layout, DepthLayout.PER_SAMPLE)

        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp)
            linhas = list(iter_manifest(tmp))
            self.assertEqual(len({l["scene_id"] for l in linhas}), len(linhas),
                             "1 amostra por cena: a dedup não teria o que economizar")
            for linha in linhas:
                with self.subTest(sample_id=linha["sample_id"]):
                    self.assertEqual(linha["depth_ref"],
                                     f"depth/{linha['sample_id']}.png")
                    self.assertTrue((Path(tmp) / linha["depth_ref"]).is_file())
            self.assertEqual(len(list((Path(tmp) / "depth").glob("*.png"))),
                             len(linhas))

    def test_limit_para_o_laco(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats, _ = _roda(tmp, fontes=_fontes(5), limit=2)
            self.assertEqual(stats.written, 2)

    def test_metadado_em_disco_reconstroi_o_mapa_de_defocus(self):
        """O mapa não é gravado — é derivado. O teste prova que o que está no disco basta,
        e que a derivação é a MESMA função da geração."""
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp, fontes=[FakeSource()])
            meta = read_metadata(tmp, "b_42")
            from dataio import EncodedDepth
            from PIL import Image
            u16 = np.asarray(Image.open(Path(tmp) / "depth" / "b_42.png"))
            encoded = EncodedDepth(u16, meta["disparity_min"], meta["disparity_max"],
                                   (meta["image_h"], meta["image_w"]),
                                   (meta["depth_h"], meta["depth_w"]))
            mapa = defocus_map(decode_depth_m(encoded), meta["focus_disparity"],
                               meta["k_value"])
            self.assertTrue(np.isfinite(mapa).all())
            self.assertLessEqual(float(mapa.max()), 1.0)
            # ida e volta pela codificação de disco não muda o mapa
            np.testing.assert_allclose(
                decode_defocus_uint16(encode_defocus_uint16(mapa)), mapa, atol=2e-5)


# ==============================================================================
# O resumo do run
# ==============================================================================

class Resumo(unittest.TestCase):
    def test_o_resumo_imprime_as_ancoras_de_K(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, saida = _roda(tmp)
        for ancora in ("16,6", "20,1", "15,0"):
            self.assertIn(ancora, saida)

    def test_o_resumo_denuncia_a_DeblurNet_que_nao_fez_nada(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, saida = _roda(tmp, deblur=DeblurIdentidade())
        self.assertIn("SSIM(AIF, bokeh)", saida)
        self.assertIn("dispersão da retenção", saida)
        self.assertIn("defeito B1", saida)

    def test_o_resumo_conta_os_crop_factor_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            fontes = [FakeSource(scene_id="1", sample_id="b_1", focal_length_35mm=50.0),
                      FakeSource(scene_id="2", sample_id="b_2")]
            stats, saida = _roda(tmp, fontes=fontes)
        self.assertEqual(stats.crop_factor_unity, 1)
        self.assertIn("crop factor == 1,0 exato", saida)
        self.assertIn("Canon EOS 60D", saida)

    def test_o_resumo_mostra_a_divergencia_da_Decisao_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, saida = _roda(tmp)
        self.assertIn("1/med(1/z) ÷ med(z)", saida)
        self.assertIn("focus_agreement", saida)

    def test_resumo_sem_amostra_aceita_nao_quebra(self):
        self.assertIn("nenhuma amostra aceita", RouteBStats().summary())

    def test_percentis_dizem_quando_a_grandeza_esta_AUSENTE(self):
        """"Ausente" e "medido como zero" são coisas diferentes, e o resumo não pode
        confundi-las — foi o defeito do writer que filtrava `None` do JSON."""
        stats = RouteBStats(k_values=[1.0], written=1)
        self.assertIn("ausente", stats.summary())


# ==============================================================================
# O adaptador do ITW — a única parte que conhece o dataset concreto
# ==============================================================================

def _linha_itw(**kw):
    """Uma linha do `atfortes/BokehDiffusion`, com as colunas medidas em ACHADOS.md."""
    linha = {
        "image": None,
        "focal_length": 50.0,
        "f_number": 1.8,
        "focal_length_35": 75.0,
        "pseudo_aif": False,
        "flickr_photo_id": 314159,
        "flickr_exif": {"make": "NIKON CORPORATION", "model": "NIKON D7000"},
    }
    linha.update(kw)
    return linha


class AdaptadorITW(unittest.TestCase):
    def test_sample_id_vem_do_flickr_photo_id_nao_do_indice(self):
        """Defeito B12: `stem = f"b_{index:06d}"` sobre o dataset FILTRADO renumerava
        tudo quando o filtro mudava, e a retomada passava a pular as amostras erradas."""
        a = ItwRow(_linha_itw(), 0)
        b = ItwRow(_linha_itw(), 9999)
        self.assertEqual(a.sample_id, "b_314159")
        self.assertEqual(a.sample_id, b.sample_id)
        self.assertEqual(a.scene_id, "314159")

    def test_o_sample_id_e_o_MESMO_nas_duas_variantes(self):
        """Regra 5 do §5.3: é o que torna as duas versões comparáveis par a par, e a
        diferença de K entre elas uma medida de quanto a AIF influencia o rótulo."""
        self.assertNotIn("variant", ItwRow(_linha_itw(), 0).sample_id)
        self.assertNotIn("official", ItwRow(_linha_itw(), 0).sample_id)

    def test_sem_flickr_photo_id_rejeita_em_vez_de_usar_o_indice(self):
        with self.assertRaises(SampleRejected) as ctx:
            ItwRow(_linha_itw(flickr_photo_id=None), 7)
        self.assertEqual(ctx.exception.reason, "source_metadata_missing")

    def test_f_number_invalido_rejeita_com_slug(self):
        for valor in (None, 0, -1.0, "abc"):
            with self.assertRaises(SampleRejected) as ctx:
                ItwRow(_linha_itw(f_number=valor), 0)
            self.assertEqual(ctx.exception.reason, "source_metadata_field_invalid")

    def test_focal_length_invalida_rejeita_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            ItwRow(_linha_itw(focal_length=0.0), 0)
        self.assertEqual(ctx.exception.reason, "source_metadata_field_invalid")

    def test_sem_focal_35_o_adaptador_NAO_assume_36mm(self):
        """Defeito B6: o antigo punha `sensor_width_mm = 36.0` quando faltava. Aqui o
        campo vai `None` e quem rejeita é o contrato, com slug no histograma."""
        linha = ItwRow(_linha_itw(focal_length_35=None), 0)
        self.assertIsNone(linha.focal_length_35mm)
        with self.assertRaises(SampleRejected) as ctx:
            _processa(source=linha)
        self.assertEqual(ctx.exception.reason, "sensor_width_unresolvable")

    def test_a_linha_do_ITW_satisfaz_o_protocolo_da_rota(self):
        """Se o adaptador e o `BokehSource` divergirem, é aqui que aparece."""
        amostra = _processa(source=ItwRow(_linha_itw(), 0))
        self.assertEqual(amostra.sample_id, "b_314159")
        self.assertEqual(amostra.refs.source_dataset, "atfortes/BokehDiffusion")
        self.assertEqual(amostra.provenance.extra["camera_make"], "NIKON CORPORATION")

    def test_exif_como_TEXTO_JSON_tambem_e_lido(self):
        linha = ItwRow(_linha_itw(flickr_exif=json.dumps({"Make": "Apple",
                                                          "Model": "iPhone 13"})), 0)
        self.assertEqual(linha.camera_make, "Apple")
        self.assertEqual(linha.camera_model, "iPhone 13")

    def test_exif_ausente_nao_quebra_e_nao_inventa_camera(self):
        linha = ItwRow(_linha_itw(flickr_exif=None), 0)
        self.assertIsNone(linha.camera_make)
        self.assertIsNone(linha.exif_focus_distance_m)

    def test_distancia_de_foco_da_EXIF_e_lida_mas_nunca_e_rotulo(self):
        """paper.txt:344-346 proíbe a EXIF como `D_focus`. Aqui ela existe só como
        validador — e o teste prova que o rótulo NÃO muda quando ela aparece."""
        com = ItwRow(_linha_itw(flickr_exif={"SubjectDistance": "4.5 m"}), 0)
        self.assertAlmostEqual(com.exif_focus_distance_m, 4.5, places=9)
        sem = ItwRow(_linha_itw(flickr_exif={}), 0)
        self.assertIsNone(sem.exif_focus_distance_m)
        self.assertAlmostEqual(_processa(source=com).control.k_value,
                               _processa(source=sem).control.k_value, places=12)

    def test_distancia_de_foco_lixo_vira_None_em_vez_de_numero(self):
        """`"inf"` passa por `float()`, é > 0 e é igual a si mesmo. `math.isfinite` é o
        que pega — e sem ele o validador entraria com um número infinito."""
        for valor in ("inf", "nan", "", "n/a", -1):
            linha = ItwRow(_linha_itw(flickr_exif={"SubjectDistance": valor}), 0)
            self.assertIn(linha.exif_focus_distance_m, (None,),
                          f"{valor!r} virou {linha.exif_focus_distance_m!r}")

    def test_pseudo_aif_e_filtrado_com_slug_no_histograma(self):
        """O filtro é `[A]` A3 — do pipeline antigo, não do paper. Ele descarta, então
        tem que aparecer no histograma; nada é descartado em silêncio."""
        log = RejectionLog(None)
        linhas = _enumera_itw([_linha_itw(pseudo_aif=True),
                               _linha_itw(flickr_photo_id=2)],
                              log=log, incluir_pseudo_aif=False)
        self.assertEqual([l.scene_id for l in linhas], ["2"])
        self.assertEqual(log.reasons["source_metadata_field_invalid"], 1)

    def test_include_pseudo_aif_mantem_a_linha(self):
        log = RejectionLog(None)
        linhas = _enumera_itw([_linha_itw(pseudo_aif=True)], log=log,
                              incluir_pseudo_aif=True)
        self.assertEqual(len(linhas), 1)

    def test_flickr_photo_id_repetido_rejeita_com_slug(self):
        log = RejectionLog(None)
        linhas = _enumera_itw([_linha_itw(), _linha_itw()], log=log,
                              incluir_pseudo_aif=False)
        self.assertEqual(len(linhas), 1)
        self.assertEqual(log.reasons["source_duplicate_sample"], 1)


class LoaderDoITW(unittest.TestCase):
    def _dataset(self, imagem):
        return [{"image": imagem}]

    def test_devolve_BGR_e_grava_o_ledger_da_FONTE(self):
        import hashlib
        from PIL import Image
        aif, _ = _cena()
        buffer = io.BytesIO()
        Image.fromarray(aif[..., ::-1]).save(buffer, format="PNG")
        bytes_png = buffer.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "source_images.jsonl"
            loader = ItwImageLoader(self._dataset({"bytes": bytes_png}),
                                    ledger_path=ledger)
            linha = ItwRow(_linha_itw(), 0)
            bgr = loader(linha)
            self.assertEqual(bgr.shape, (96, 96, 3))
            # BGR: o canal 2 do array é o R do PNG.
            np.testing.assert_array_equal(bgr[..., ::-1], aif[..., ::-1])
            loader.close()
            registro = json.loads(ledger.read_text().splitlines()[0])
            self.assertEqual(registro["bokeh_role"], "reference")
            self.assertEqual(registro["bokeh_sha256"],
                             hashlib.sha256(bytes_png).hexdigest())
            self.assertEqual(registro["sample_id"], "b_314159")

    def test_RGBA_com_alpha_translucido_REJEITA_em_vez_de_compor_sobre_preto(self):
        """`convert("RGB")` comporia sobre preto e INVENTARIA pixel na entrada da
        DeblurNet, do Depth Pro e do BiRefNet."""
        from PIL import Image
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[..., 3] = 128
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        loader = ItwImageLoader(self._dataset({"bytes": buffer.getvalue()}))
        with self.assertRaises(SampleRejected) as ctx:
            loader(ItwRow(_linha_itw(), 0))
        self.assertEqual(ctx.exception.reason, "source_image_alpha_not_opaque")

    def test_celula_de_tipo_desconhecido_REJEITA_com_slug(self):
        loader = ItwImageLoader(self._dataset(12345))
        with self.assertRaises(SampleRejected) as ctx:
            loader(ItwRow(_linha_itw(), 0))
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")


class VarianteNaoSeMistura(unittest.TestCase):
    """Regras 1 e 3 do §5.3: um `--output-dir` por variante, e checagem de uniformidade.

    Sem isto, um `rsync` de duas metades produz um release misto e nada denuncia — a
    mesma família do `max_coc` por rota, que foi o mecanismo do kfix.
    """

    def test_pasta_vazia_passa(self):
        with tempfile.TemporaryDirectory() as tmp:
            _confere_variante_uniforme(Path(tmp), DeblurVariant.OFFICIAL_COND_ONLY)

    def test_mesma_variante_passa(self):
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp, fontes=[FakeSource()])
            _confere_variante_uniforme(Path(tmp), DeblurVariant.OFFICIAL_COND_ONLY)

    def test_variante_DIFERENTE_recusa_com_SystemExit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _roda(tmp, fontes=[FakeSource()])
            with self.assertRaises(SystemExit) as ctx:
                _confere_variante_uniforme(Path(tmp), DeblurVariant.OURS_MAIN_COND)
            self.assertIn("POR VARIANTE", str(ctx.exception))

    def test_o_default_da_CLI_e_a_variante_OFICIAL(self):
        """Decisão declarada: o peso publicado pelos autores é cond-only, e rodá-lo com
        `main_adapter=None` é o correto (`Inference_deblurNet.py:103-111`)."""
        args = build_parser().parse_args(
            ["--output-dir", "x", "--models-dir", "m", "--flux-dir", "f",
             "--genfocus-dir", "g"])
        self.assertEqual(args.deblur_variant, DeblurVariant.OFFICIAL_COND_ONLY.value)
        self.assertIsNone(DeblurVariant(args.deblur_variant).spec.main_adapter)

    def test_a_CLI_oferece_TODAS_as_variantes_do_enum(self):
        """`choices` tem que DERIVAR do enum, não ser uma lista escrita à mão.

        Uma variante nova que não aparecesse aqui seria inatingível pela CLI, e a única
        forma de rodá-la seria contornar o entrypoint — que é onde a tripla é montada.
        Hoje são três, e `ours_cond_only` é a nossa.
        """
        acao = {a.dest: a for a in build_parser()._actions}["deblur_variant"]
        self.assertEqual(sorted(acao.choices),
                         sorted(v.value for v in DeblurVariant))
        self.assertIn("ours_cond_only", acao.choices)

        args = build_parser().parse_args(
            ["--output-dir", "x", "--models-dir", "m", "--flux-dir", "f",
             "--genfocus-dir", "g", "--deblur-variant", "ours_cond_only"])
        spec = DeblurVariant(args.deblur_variant).spec
        self.assertIsNone(spec.main_adapter, "a NOSSA é cond-only: lora_on_main: false")
        self.assertEqual(spec.repo_id, "juliadollis/genrefocus-deblurnet")
        self.assertTrue(spec.pinned_sha256, "sem sha fixo o nome do arquivo decidiria")

    def test_a_CLI_nao_tem_flag_de_main_adapter_nem_de_max_coc(self):
        """As duas juntas são o par de defeitos B1 e B4. `--main-adapter` de texto livre
        com default é o antipadrão exato: um default do lado errado produz o defeito, um
        do lado certo o esconde."""
        acoes = {a.dest for a in build_parser()._actions}
        self.assertNotIn("main_adapter", acoes)
        self.assertNotIn("max_coc", acoes)


if __name__ == "__main__":
    unittest.main()
