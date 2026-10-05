"""Rota A de ponta a ponta, com Depth Pro falso e renderer sintético.

O que este arquivo prova, sem GPU: que o pipeline INTEIRO da rota A — profundidade,
sorteio de `(D_focus, K)`, Eq. 2, render, gates, escrita, histograma e retomada — produz
amostras cujo K sobrevive ao disco, cuja banda em foco é reconstrutível a partir do
metadado, e cuja proveniência não mente.

Os dublês vivem aqui, nunca em `src/`. O renderer sintético é importado de
`test_route_c` de propósito: um segundo dublê de renderer seria uma segunda definição de
"o que é borrão", e o projeto inteiro existe para não ter duas.

## O shim de vocabulário, e por que ele está num teste

A rota A precisa de `FocusSource.SAMPLED_PLANE` e `MaskSource.SAMPLED_PLANE`, que moram
em `src/qc/` e `src/dataio/` — módulos que esta tarefa **não pode editar**. A extensão
está descrita em `reference/ROTA_A_DECISOES.md` e em `routes.route_a._PATCH`.

`com_vocabulario_da_rota_a()` instala os dois valores em tempo de execução, com desmonte,
e afrouxa `_REQUIRED_PROVENANCE` exatamente como o item F6 pede. Ele existe para provar
**duas** coisas: que sem a extensão a rota falha alto em vez de mentir, e que com ela o
pipeline fecha. O shim não é importável de `src/` e o desmonte é obrigatório — um valor de
enum vazando entre testes quebraria
`test_dataio.test_toda_fonte_de_foco_tem_fonte_de_mascara`, que exige que
`FOCUS_SOURCE_TO_MASK_SOURCE` seja total.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from control.contract import (                                               # noqa: E402
    FOCUS_DEPTH_MAX_M, MAX_COC, REJECTION_REASONS, SampleRejected,
    validate_metric_depth,
)
from dataio import (                                                         # noqa: E402
    FOCUS_SOURCE_TO_MASK_SOURCE, MaskSource, build_scene_split, iter_manifest,
    read_metadata,
)
import dataio.sample as dataio_sample                                        # noqa: E402
from qc.focus_region import FocusSource                                      # noqa: E402
from qc.rejection import RejectionLog                                        # noqa: E402
from routes.route_a import (                                                 # noqa: E402
    DEFAULT_SAMPLES_PER_IMAGE, GATE_TO_REASON, RouteAConfig, RouteAStats,
    SAMPLED_FOCUS_SOURCE_VALUE, SAMPLED_MASK_SOURCE_VALUE, VocabularyExtensionRequired,
    build_focus_sampling_pool, build_gate_report, draw_for_variant, draw_plan,
    focus_band_mask, prepare_image, process_variant, resolve_sampled_vocabulary,
    run_route_a, sample_focus_disparity, sample_id_for,
    validate_sampled_focus_disparity, variant_seed,
)
from sources import genphoto_ebb                                             # noqa: E402
from sources.genphoto_ebb import (                                           # noqa: E402
    SOURCE_EBB, SOURCE_GENERATIVE_PHOTOGRAPHY, AifImage, SharpnessMeasurement,
    enumerate_candidates, enumeration_summary, rank_and_cut_per_source,
    validate_source_name,
)
from sources.k_distribution import (                                         # noqa: E402
    DEFAULT_WIDEN_FRACTION, KDistribution, build_distribution, scan_manifest_rows,
)
from test_route_c import _LAYERED                                            # noqa: E402


# ==============================================================================
# O shim de vocabulário — a extensão DESCRITA, aplicada só em teste
# ==============================================================================

def _instala_membro(enum_cls, nome: str, valor: str):
    """Acrescenta um membro a um `str, Enum` em tempo de execução.

    Não é jeito de escrever código de produção — é jeito de provar que o patch descrito
    em `reference/ROTA_A_DECISOES.md` é suficiente. O `str.__new__` é obrigatório porque
    os dois enums herdam de `str`.
    """
    ja = enum_cls._member_map_.get(nome)
    if ja is not None:
        # O patch definitivo entrou: o membro é nativo. O shim vira no-op em vez de
        # instalar uma duplicata — que corromperia `_member_names_` e faria a iteração
        # do enum explodir depois do desmonte.
        return ja
    membro = str.__new__(enum_cls, valor)
    membro._name_ = nome
    membro._value_ = valor
    membro.__objclass__ = enum_cls
    enum_cls._member_map_[nome] = membro
    enum_cls._value2member_map_[valor] = membro
    enum_cls._member_names_.append(nome)
    return membro


def _remove_membro(enum_cls, nome: str, valor: str) -> None:
    if enum_cls._member_names_.count(nome) <= 1 and nome in enum_cls.__dict__:
        return          # nativo: não desmonta o que não foi o shim que montou
    enum_cls._member_map_.pop(nome, None)
    enum_cls._value2member_map_.pop(valor, None)
    if nome in enum_cls._member_names_:
        enum_cls._member_names_.remove(nome)


@contextlib.contextmanager
def sem_vocabulario_da_rota_a():
    """Remove temporariamente o membro NATIVO, para a guarda continuar testada.

    O patch de vocabulário já entrou em `qc/focus_region.py` e `dataio/sample.py`, então
    o caminho de falha de `resolve_sampled_vocabulary` não é mais alcançável por acidente.
    Ele continua valendo — se alguém remover o membro, a rota tem que parar antes de
    gravar, e não gravar com um valor emprestado. Este contexto é o que prova isso.
    """
    foco = FocusSource._member_map_.pop("SAMPLED_PLANE", None)
    mascara = MaskSource._member_map_.pop("SAMPLED_PLANE", None)
    if foco is not None:
        FocusSource._value2member_map_.pop(foco.value, None)
        FocusSource._member_names_.remove("SAMPLED_PLANE")
        FOCUS_SOURCE_TO_MASK_SOURCE.pop(foco, None)
    if mascara is not None:
        MaskSource._value2member_map_.pop(mascara.value, None)
        MaskSource._member_names_.remove("SAMPLED_PLANE")
    try:
        yield
    finally:
        if foco is not None:
            FocusSource._member_map_["SAMPLED_PLANE"] = foco
            FocusSource._value2member_map_[foco.value] = foco
            FocusSource._member_names_.append("SAMPLED_PLANE")
        if mascara is not None:
            MaskSource._member_map_["SAMPLED_PLANE"] = mascara
            MaskSource._value2member_map_[mascara.value] = mascara
            MaskSource._member_names_.append("SAMPLED_PLANE")
        if foco is not None and mascara is not None:
            FOCUS_SOURCE_TO_MASK_SOURCE[foco] = mascara


@contextlib.contextmanager
def com_vocabulario_da_rota_a():
    """Aplica as TRÊS mudanças descritas em `routes.route_a._PATCH`, e desfaz no fim."""
    nativo = "SAMPLED_PLANE" in FocusSource._member_map_
    foco = _instala_membro(FocusSource, "SAMPLED_PLANE", SAMPLED_FOCUS_SOURCE_VALUE)
    mascara = _instala_membro(MaskSource, "SAMPLED_PLANE", SAMPLED_MASK_SOURCE_VALUE)
    FOCUS_SOURCE_TO_MASK_SOURCE[foco] = mascara
    # F6: `mask_model_sha256` não pode ser obrigatório numa rota sem segmentador. A rota
    # A grava `mask_rule` no lugar, e é isso que o patch definitivo tem que exigir.
    sem_mascara = tuple(c for c in dataio_sample._REQUIRED_PROVENANCE
                        if c != "mask_model_sha256")
    patch = mock.patch.object(dataio_sample, "_REQUIRED_PROVENANCE", sem_mascara)
    patch.start()
    try:
        yield foco, mascara
    finally:
        patch.stop()
        # Só desmonta o que o shim montou. Com o patch definitivo aplicado, `foco` e
        # `mascara` são os membros NATIVOS e o mapeamento veio do `sample.py` — removê-lo
        # aqui deixaria o enum quebrado para os testes seguintes.
        if not nativo:
            FOCUS_SOURCE_TO_MASK_SOURCE.pop(foco, None)
        _remove_membro(FocusSource, "SAMPLED_PLANE", SAMPLED_FOCUS_SOURCE_VALUE)
        _remove_membro(MaskSource, "SAMPLED_PLANE", SAMPLED_MASK_SOURCE_VALUE)


# ==============================================================================
# Dublês
# ==============================================================================

@dataclass
class FakeAif:
    """Uma imagem AIF, na forma do `AifSource`."""

    scene_id: str
    source_dataset: str = SOURCE_GENERATIVE_PHOTOGRAPHY
    source_sample_id: str = "cena/imagem.png"
    source_split: Optional[str] = None
    aif_ref: Optional[str] = "cena/imagem.png"
    source_revision: Optional[str] = "rev-abc"
    laplacian_variance: Optional[float] = 1234.5
    sharpness_hw: Optional[tuple] = (300, 400)


class FakeDepth:
    """Profundidade determinística: gradiente de 1 a 40 m com um objeto a 3 m.

    Conta as chamadas. É o instrumento do teste de retomada: a rota A tem que pular a
    imagem inteira quando as N variantes já estão no manifesto, e "pular" significa **não
    rodar o Depth Pro** — num run de 70 mil amostras retomado no meio, é a diferença entre
    horas e minutos.
    """

    backend = "depth_pro"

    def __init__(self):
        self.calls = 0

    def infer(self, image_rgb):
        self.calls += 1
        h, w = image_rgb.shape[:2]
        z = np.linspace(1.0, 40.0, h * w, dtype=np.float32).reshape(h, w)
        z[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 3.0
        return validate_metric_depth(z, backend="depth_pro")


class DepthGradiente(FakeDepth):
    """Gradiente puro de 1 a 40 m, sem platô. Cada quantil é um valor distinto."""

    def infer(self, image_rgb):
        self.calls += 1
        h, w = image_rgb.shape[:2]
        z = np.linspace(1.0, 40.0, h * w, dtype=np.float32).reshape(h, w)
        return validate_metric_depth(z, backend="depth_pro")


class DepthComCeu(FakeDepth):
    """Cena natural: 10% de céu no teto de 10.000 m do Depth Pro (`ACHADOS.md:54`)."""

    def infer(self, image_rgb):
        depth = super().infer(image_rgb)
        z = np.array(depth.values_m)
        z[: max(1, z.shape[0] // 10), :] = 10000.0
        return validate_metric_depth(z, backend="depth_pro")


class DepthTodaCeu(FakeDepth):
    """Cena inteira no teto: não existe plano de foco candidato."""

    def infer(self, image_rgb):
        z = np.full(image_rgb.shape[:2], 5000.0, dtype=np.float32)
        z[0, 0] = 9000.0                     # faixa não degenerada, mas tudo implausível
        return validate_metric_depth(z, backend="depth_pro")


class DepthEmOutraGrade(FakeDepth):
    """Profundidade numa grade diferente da imagem — a família do defeito D12."""

    def infer(self, image_rgb):
        depth = super().infer(image_rgb)
        return validate_metric_depth(np.array(depth.values_m)[::2, ::2],
                                     backend="depth_pro")


def _aif(size=96, seed=0):
    """AIF texturizada: blocos de 8 px, para o laplaciano ter do que viver."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size // 8, size // 8, 3), dtype=np.uint16)
    return np.repeat(np.repeat(base, 8, axis=0), 8, axis=1).astype(np.uint8)


def _render_nulo(aif_bgr, depth_m, focus_disparity, k_value):
    """O renderer que NÃO fez nada — devolve a AIF. É o defeito A3 pelo silêncio."""
    return np.asarray(aif_bgr, dtype=np.float64)


def _render_grade_errada(aif_bgr, depth_m, focus_disparity, k_value):
    return np.asarray(aif_bgr, dtype=np.float64)[:, : aif_bgr.shape[1] // 2]


def _distribuicao(p_lo=0.004, p_hi=0.040, n=4000, seed=0) -> KDistribution:
    """Distribuição sintética de `k_per_long_side`, no molde do que B e C produzirão.

    Lognormal porque K é razão de grandezas ópticas positivas. A faixa foi escolhida para
    que, com lado longo 96, `k_value` caia na ordem das âncoras medidas (`CONTRATO.md:77`:
    16,6 / 20,1 / 15,0).
    """
    rng = np.random.default_rng(seed)
    centro = np.sqrt(p_lo * p_hi)
    valores = rng.lognormal(mean=np.log(centro), sigma=0.45, size=n)
    linhas = [{"route": "c", "k_value": float(v) * 1500.0, "image_h": 1000,
               "image_w": 1500, "is_valid_for_control": True, "is_k_censored": False}
              for v in valores]
    return build_distribution([scan_manifest_rows(linhas, release_dir="sintetico")],
                              control_version="metric_disparity_official_v1")


_PROV = {
    "pipeline_commit": "abc1234",
    "depth_model_sha256": "d" * 64,
    "mask_model_sha256": "",          # a rota A não roda segmentador
    "mask_backend": None,
    "renderer": {"renderer": "bokehme_public", "renderer_commit": "8b3ed556",
                 "is_final_label_renderer": True},
    "k_effective_factor": 0.9873,
}


def _config(tmp, **kw):
    kw.setdefault("k_distribution", _distribuicao())
    kw.setdefault("samples_per_image", 4)
    return RouteAConfig(output_dir=Path(tmp), **kw)


def _prepara(tmp, depth=None, size=96, **kw):
    fonte = FakeAif("gp_aaaaaaaaaaaa")
    return fonte, prepare_image(fonte, aif_bgr=_aif(size),
                                depth_runtime=depth or FakeDepth(),
                                config=_config(tmp, **kw))


# ==============================================================================
# O vocabulário — extensão, nunca reuso
# ==============================================================================

class Vocabulario(unittest.TestCase):
    def test_a_extensao_JA_ESTA_no_vocabulario(self):
        """O patch entrou: os dois enums têm `SAMPLED_PLANE` nativo e a ponte é total."""
        foco, mascara = resolve_sampled_vocabulary()
        self.assertEqual(foco.value, SAMPLED_FOCUS_SOURCE_VALUE)
        self.assertEqual(mascara.value, SAMPLED_MASK_SOURCE_VALUE)
        self.assertIs(FOCUS_SOURCE_TO_MASK_SOURCE[foco], mascara)

    def test_se_alguem_REMOVER_a_extensao_a_rota_falha_alto(self):
        with sem_vocabulario_da_rota_a(), self.assertRaises(
                VocabularyExtensionRequired) as ctx:
            resolve_sampled_vocabulary()
        texto = str(ctx.exception)
        self.assertIn("FocusSource", texto)
        self.assertIn("MaskSource", texto)
        self.assertIn(SAMPLED_FOCUS_SOURCE_VALUE, texto)
        self.assertIn("mask_model_sha256", texto)

    def test_nao_e_SampleRejected_logo_nao_entra_no_histograma(self):
        """Vocabulário faltando é release que não pode existir, não amostra ruim."""
        self.assertFalse(issubclass(VocabularyExtensionRequired, SampleRejected))

    def test_os_valores_novos_nao_COLIDEM_com_nenhum_existente(self):
        """Cada valor aparece uma vez só em cada enum — nada foi reusado nem duplicado.

        Era isto que o teste anterior media pelo avesso, quando o membro ainda não
        existia. Agora que ele existe, a afirmação que importa é a unicidade.
        """
        for enum_cls in (FocusSource, MaskSource):
            valores = [m.value for m in enum_cls]
            with self.subTest(enum=enum_cls.__name__):
                self.assertEqual(len(valores), len(set(valores)))
        self.assertIn(SAMPLED_FOCUS_SOURCE_VALUE, {m.value for m in FocusSource})
        self.assertIn(SAMPLED_MASK_SOURCE_VALUE, {m.value for m in MaskSource})

    def test_nao_reusa_DEPTH_BAND_que_era_da_rota_A_ANTIGA(self):
        """`depth_band` descrevia a banda no quantil 0,12 da profundidade (defeito A7)."""
        self.assertNotEqual(SAMPLED_MASK_SOURCE_VALUE, MaskSource.DEPTH_BAND.value)

    def test_com_a_extensao_resolve(self):
        with com_vocabulario_da_rota_a() as (foco, mascara):
            self.assertEqual(resolve_sampled_vocabulary(), (foco, mascara))

    def test_o_shim_virou_no_op_e_nao_corrompe_o_enum(self):
        """Com o membro nativo, o shim não pode instalar duplicata.

        Se instalasse, `_member_names_` ficaria com duas entradas e o desmonte removeria
        o mapa da nativa — a iteração do enum explodiria depois, longe daqui.
        """
        antes_foco = list(FocusSource._member_names_)
        antes_mascara = list(MaskSource._member_names_)
        with com_vocabulario_da_rota_a():
            self.assertEqual(FocusSource._member_names_.count("SAMPLED_PLANE"), 1)
            self.assertEqual(MaskSource._member_names_.count("SAMPLED_PLANE"), 1)
        # e depois do desmonte o enum continua inteiro
        self.assertEqual(FocusSource._member_names_, antes_foco)
        self.assertEqual(MaskSource._member_names_, antes_mascara)
        self.assertEqual(len(FOCUS_SOURCE_TO_MASK_SOURCE), len(list(FocusSource)))
        resolve_sampled_vocabulary()          # continua resolvendo


# ==============================================================================
# O sorteio — hipercubo latino, determinístico
# ==============================================================================

class Sorteio(unittest.TestCase):
    def test_cobre_todos_os_estratos_nas_duas_margens(self):
        """É o ponto do hipercubo latino: 41 i.i.d. deixariam imagens sem K alto."""
        plano = draw_plan("cena", n=41, seed=0)
        self.assertEqual(sorted(d.stratum_k for d in plano), list(range(41)))
        self.assertEqual(sorted(d.stratum_focus for d in plano), list(range(41)))
        for d in plano:
            self.assertTrue(d.stratum_k / 41 <= d.u_k < (d.stratum_k + 1) / 41)
            self.assertTrue(d.stratum_focus / 41 <= d.u_focus < (d.stratum_focus + 1) / 41)

    def test_deterministico_por_cena_e_semente(self):
        a = draw_plan("cena_x", n=8, seed=3)
        b = draw_plan("cena_x", n=8, seed=3)
        self.assertEqual([(d.u_k, d.u_focus) for d in a],
                         [(d.u_k, d.u_focus) for d in b])
        outra = draw_plan("cena_y", n=8, seed=3)
        self.assertNotEqual([d.u_k for d in a], [d.u_k for d in outra])
        outra_semente = draw_plan("cena_x", n=8, seed=4)
        self.assertNotEqual([d.u_k for d in a], [d.u_k for d in outra_semente])

    def test_uma_variante_sozinha_bate_com_o_plano(self):
        """Refazer UMA amostra a partir do `sample_id` é o que auditoria exige."""
        plano = draw_plan("cena_x", n=41, seed=7)
        for i in (0, 17, 40):
            self.assertEqual(draw_for_variant("cena_x", i, n=41, seed=7), plano[i])

    def test_semente_nao_depende_do_PYTHONHASHSEED(self):
        """`hash()` é randomizado por processo — o release deixaria de ser reproduzível."""
        import hashlib
        esperado = int.from_bytes(
            hashlib.sha256(b"0:cena_x").digest()[:8], "big")
        self.assertEqual(variant_seed("cena_x", 0), esperado)

    def test_sample_id_e_funcao_da_cena_e_da_variante(self):
        """O antigo era o ÍNDICE NA LISTA: reordenar o manifesto renomeava tudo (A9)."""
        self.assertEqual(sample_id_for("gp_abc", 7), "gp_abc_v07")
        self.assertEqual(sample_id_for("gp_abc", 7, width=3), "gp_abc_v007")

    def test_largura_do_id_cresce_com_N(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_config(tmp, samples_per_image=41).variant_id_width, 2)
            self.assertEqual(_config(tmp, samples_per_image=200).variant_id_width, 3)

    def test_zero_variantes_por_imagem_e_erro_de_config(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _config(tmp, samples_per_image=0)


# ==============================================================================
# K — a grandeza livre de resolução
# ==============================================================================

class KLivreDeResolucao(unittest.TestCase):
    def test_mesma_imagem_em_duas_resolucoes_da_o_MESMO_CoC_relativo(self):
        """O teste que a auditoria pede para travar `[A]` A7.

        `k_per_long_side` é a grandeza sorteada; `k_value` é ela vezes `max(H,W)`. Logo o
        CoC em pixel cresce com a resolução — e o CoC **relativo ao lado longo** não muda.
        É isso que torna o K transportável entre a rota B e a rota A (defeito A5).
        """
        dist = _distribuicao()
        sorteado = dist.sample(0.5)
        k_pequeno = sorteado.at_long_side(500)
        k_grande = sorteado.at_long_side(4000)
        self.assertAlmostEqual(k_pequeno / 500, k_grande / 4000, places=12)
        self.assertAlmostEqual(k_grande / k_pequeno, 8.0, places=9)

    def test_alargamento_preserva_a_mediana_e_abre_os_dois_lados(self):
        dist = _distribuicao()
        self.assertAlmostEqual(dist.sample(0.5).per_long_side, dist.p50, places=12)
        lo, hi = dist.widened_support(0.25)
        self.assertAlmostEqual(lo, dist.p50 - 1.25 * (dist.p50 - dist.p01), places=12)
        self.assertAlmostEqual(hi, dist.p50 + 1.25 * (dist.p99 - dist.p50), places=12)
        self.assertLess(lo, dist.p01)
        self.assertGreater(hi, dist.p99)

    def test_alargamento_nunca_produz_K_nao_positivo(self):
        """`signed_coc_px` rejeita K <= 0; o piso é derivado da própria distribuição."""
        dist = _distribuicao(p_lo=0.02, p_hi=0.021)     # faixa estreita e alta
        for u in np.linspace(0.0, 1.0, 51):
            amostra = dist.sample(float(u), widen_fraction=0.9)
            self.assertGreater(amostra.per_long_side, 0.0)
        self.assertGreater(dist.positive_floor(0.25), 0.0)

    def test_o_piso_e_contado_por_amostra(self):
        dist = _distribuicao()
        presa = dist.sample(0.0, widen_fraction=0.95)
        self.assertTrue(presa.clamped_to_floor)
        self.assertTrue(presa.to_metadata()["k_clamped_to_positive_floor"])
        self.assertFalse(dist.sample(0.5).clamped_to_floor)

    def test_o_valor_observado_viaja_ao_lado_do_alargado(self):
        """Sem ele não dá para responder 'este K existia nos dados?'."""
        meta = _distribuicao().sample(0.9).to_metadata()
        self.assertIn("k_per_long_side", meta)
        self.assertIn("k_per_long_side_observed", meta)
        self.assertNotEqual(meta["k_per_long_side"], meta["k_per_long_side_observed"])


class DistribuicaoDeK(unittest.TestCase):
    def test_a_rota_B_publicada_e_recusada(self):
        """Medido: `k = 50,0` em 11.635/11.635 (`ACHADOS.md:14`)."""
        linhas = [{"route": "b", "k_value": 50.0, "image_h": 1000, "image_w": 1500,
                   "is_valid_for_control": True, "is_k_censored": False}
                  for _ in range(2000)]
        dist = build_distribution(
            [scan_manifest_rows(linhas, release_dir="b")], control_version="v")
        motivo = dist.degenerate_reason()
        self.assertIsNotNone(motivo)
        self.assertIn("distintos", motivo)

    def test_a_rota_C_publicada_e_recusada_pela_censura(self):
        """Medido: 47,0% no teto exato de 300 (`ACHADOS.md:19`)."""
        rng = np.random.default_rng(0)
        linhas = []
        for i in range(2000):
            censurada = i < 940                      # 47,0%
            linhas.append({"route": "c",
                           "k_value": 300.0 if censurada else float(rng.uniform(10, 290)),
                           "image_h": 1500, "image_w": 2000,
                           "is_valid_for_control": True, "is_k_censored": censurada})
        dist = build_distribution(
            [scan_manifest_rows(linhas, release_dir="c")], control_version="v")
        motivo = dist.degenerate_reason()
        self.assertIsNotNone(motivo)
        self.assertIn("censuradas", motivo)
        self.assertAlmostEqual(dist.per_route["c"]["censored_share"], 0.47, places=3)

    def test_distribuicao_saudavel_passa(self):
        self.assertIsNone(_distribuicao().degenerate_reason())

    def test_exclui_censurada_sem_resolucao_e_invalida_contando_cada_uma(self):
        linhas = [
            {"route": "c", "k_value": 10.0, "image_h": 100, "image_w": 200,
             "is_valid_for_control": True, "is_k_censored": False},
            {"route": "c", "k_value": 300.0, "image_h": 100, "image_w": 200,
             "is_valid_for_control": True, "is_k_censored": True},
            {"route": "c", "k_value": 10.0, "image_h": None, "image_w": None,
             "is_valid_for_control": True, "is_k_censored": False},
            {"route": "c", "k_value": 0.0, "image_h": 100, "image_w": 200,
             "is_valid_for_control": True, "is_k_censored": False},
            {"route": "c", "k_value": 10.0, "image_h": 100, "image_w": 200,
             "is_valid_for_control": False, "is_k_censored": False},
        ]
        scan = scan_manifest_rows(linhas, release_dir="x")
        self.assertEqual(scan.lines_total, 5)
        self.assertEqual(scan.lines_used, 1)
        self.assertEqual(scan.excluded_censored, 1)
        self.assertEqual(scan.excluded_missing_resolution, 1)
        self.assertEqual(scan.excluded_k_invalid, 1)
        self.assertEqual(scan.excluded_not_valid_for_control, 1)
        self.assertEqual(scan.values, [10.0 / 200])

    def test_a_censura_e_medida_sobre_as_linhas_VISTAS(self):
        """Com o denominador errado esta fração é 0,0 sempre, e o guarda nunca dispara."""
        linhas = [{"route": "c", "k_value": 10.0, "image_h": 10, "image_w": 20,
                   "is_valid_for_control": True, "is_k_censored": i < 5}
                  for i in range(10)]
        scan = scan_manifest_rows(linhas, release_dir="x")
        self.assertEqual(scan.routes["c"]["seen"], 10)
        self.assertEqual(scan.routes["c"]["censored"], 5)
        self.assertEqual(scan.routes["c"]["total"], 5)

    def test_grandeza_errada_no_JSON_e_recusada(self):
        """K cru, sem a resolução em que foi medido, é o defeito A5."""
        dados = _distribuicao().to_dict()
        dados["quantity"] = "k_value"
        with self.assertRaises(ValueError):
            KDistribution.from_dict(dados)

    def test_schema_desconhecido_e_recusado(self):
        dados = _distribuicao().to_dict()
        dados["schema"] = "outra_coisa_v9"
        with self.assertRaises(ValueError):
            KDistribution.from_dict(dados)

    def test_json_ida_e_volta_preserva_o_sorteio(self):
        dist = _distribuicao()
        volta = KDistribution.from_dict(json.loads(dist.to_json()))
        for u in (0.0, 0.3, 0.5, 0.97, 1.0):
            self.assertAlmostEqual(dist.sample(u).per_long_side,
                                   volta.sample(u).per_long_side, places=12)

    def test_distribuicao_vazia_e_erro_de_entrada(self):
        with self.assertRaises(ValueError):
            build_distribution([scan_manifest_rows([], release_dir="x")],
                               control_version="v")


# ==============================================================================
# O plano de foco — dentro da faixa da imagem, com massa de cena
# ==============================================================================

class PlanoDeFoco(unittest.TestCase):
    def _pool(self, z):
        return build_focus_sampling_pool(z)

    def test_cai_sempre_dentro_da_faixa_de_disparidade_da_imagem(self):
        """A pergunta central: `D_focus` fora da faixa degenera o mapa."""
        for z in (np.linspace(1.0, 40.0, 400).reshape(20, 20),
                  np.linspace(0.5, 3.0, 400).reshape(20, 20),
                  np.concatenate([np.full(200, 1.2), np.full(200, 25.0)]).reshape(20, 20)):
            pool = self._pool(z)
            for u in np.linspace(0.0, 1.0, 21):
                valor, faixa = sample_focus_disparity(
                    pool, float(u), quantile_low=0.05, quantile_high=0.95)
                self.assertGreaterEqual(valor, faixa.disparity_min)
                self.assertLessEqual(valor, faixa.disparity_max)

    def test_o_ceu_no_teto_sai_da_populacao_de_sorteio(self):
        """Medido: sem a restrição, 4,9% das variantes morrem em focus_depth_implausible
        numa cena com 10% de céu; com ela, 0,0%."""
        z = np.linspace(1.0, 40.0, 1000).reshape(25, 40)
        z[:2, :] = 10000.0                                   # 8% de céu
        pool = self._pool(z)
        self.assertLess(pool.pool_fraction, 1.0)
        self.assertGreater(pool.pool_fraction, 0.9)
        for u in np.linspace(0.0, 1.0, 41):
            valor, _ = sample_focus_disparity(pool, float(u), quantile_low=0.0,
                                              quantile_high=1.0)
            # Nenhum sorteio cai na sentinela: 1/10000 não está na população.
            validate_sampled_focus_disparity(valor)
            self.assertLessEqual(1.0 / valor, FOCUS_DEPTH_MAX_M)

    def test_populacao_vazia_rejeita_a_IMAGEM_com_slug(self):
        z = np.full((8, 8), 5000.0)
        z[0, 0] = 9000.0
        with self.assertRaises(SampleRejected) as ctx:
            self._pool(z)
        self.assertEqual(ctx.exception.reason, "focus_depth_implausible")

    def test_plano_implausivel_rejeita_com_slug_registrado(self):
        with self.assertRaises(SampleRejected) as ctx:
            validate_sampled_focus_disparity(1.0 / 5000.0)
        self.assertEqual(ctx.exception.reason, "focus_depth_implausible")
        with self.assertRaises(SampleRejected) as ctx:
            validate_sampled_focus_disparity(0.0)
        self.assertEqual(ctx.exception.reason, "focus_disparity_invalid")
        with self.assertRaises(SampleRejected) as ctx:
            validate_sampled_focus_disparity(float("nan"))
        self.assertEqual(ctx.exception.reason, "focus_disparity_invalid")

    def test_o_quantil_e_invariante_a_transformacao_monotona(self):
        """A correção medida ao item A6: `1/quantil_z(1−q) == quantil_disp(q)`.

        Se isto vale, sortear no quantil da profundidade métrica e no quantil da
        disparidade produz **o mesmo plano de foco**. O que muda o resultado é quantil
        contra uniforme, não `z` contra `1/z`. O que continua valendo de A6 é a unidade
        PRIMÁRIA, e essa é `focus_disparity` — travada em `test_metadado_grava_disparidade`.

        A igualdade é **exata** em estatística de ordem (quantil que cai num dado) e
        **aproximada** quando `np.quantile` interpola linearmente entre dois vizinhos:
        interpolar 1/z não é o inverso de interpolar z. O erro medido é de 1e-9 a 1e-7
        conforme a densidade da amostra — nove ordens de grandeza abaixo de qualquer
        diferença que mudasse um rótulo, e é por isso que a conclusão se sustenta.
        """
        z = np.linspace(1.0, 60.0, 5000)
        disp = 1.0 / z
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            a = 1.0 / float(np.quantile(z, 1 - q))
            b = float(np.quantile(disp, q))
            self.assertLess(abs(a / b - 1.0), 1e-5)

    def test_uniforme_na_faixa_esvazia_a_banda_e_o_quantil_nao(self):
        """A medição que justifica a regra, reproduzida em teste.

        Cena de retrato: sujeito a 1,2 m, fundo de 8 a 30 m. Uniforme na faixa de
        disparidade põe o plano de foco no vazio entre os dois grupos, e a banda em foco
        sai VAZIA na maioria dos sorteios — cada uma dessas variantes morreria em
        `focus_mask_empty`. Por quantil, nenhuma.
        """
        from control.contract import signed_coc_px
        z = np.linspace(8.0, 30.0, 40 * 40).reshape(40, 40)
        z[13:27, 13:27] = 1.2
        pool = build_focus_sampling_pool(z)
        k = 15.0
        vazias_quantil = vazias_uniforme = 0
        lo, hi = pool.disparity_full_min, pool.disparity_full_max
        for u in (np.arange(41) + 0.5) / 41:
            por_quantil, _ = sample_focus_disparity(pool, float(u), quantile_low=0.05,
                                                    quantile_high=0.95)
            uniforme = lo + float(u) * (hi - lo)
            for valor, contador in ((por_quantil, "q"), (uniforme, "u")):
                coc = np.abs(signed_coc_px(z, valor, k))
                vazia = not (coc <= 0.5).any()
                if contador == "q":
                    vazias_quantil += int(vazia)
                else:
                    vazias_uniforme += int(vazia)
        self.assertEqual(vazias_quantil, 0)
        self.assertGreater(vazias_uniforme, 20, "a medição dizia 73,2% com K=15")

    def test_banda_vazia_rejeita_com_slug(self):
        coc = np.full((8, 8), 5.0)
        with self.assertRaises(SampleRejected) as ctx:
            focus_band_mask(coc, band_px=0.5)
        self.assertEqual(ctx.exception.reason, "focus_mask_empty")

    def test_banda_e_exatamente_a_regra_declarada(self):
        coc = np.array([[0.0, 0.4, 0.6], [1.0, -0.5, -2.0]])
        banda = focus_band_mask(coc, band_px=0.5)
        np.testing.assert_array_equal(banda, np.abs(coc) <= 0.5)


# ==============================================================================
# Gates — bloqueiam só com limiar congelado, e o slug entra no histograma
# ==============================================================================

def _distribuicao_forte():
    """K grande o bastante para saturar o mapa numa imagem de 96 px."""
    return _distribuicao(p_lo=4.0, p_hi=6.0)


class Gates(unittest.TestCase):
    def _roda(self, *, render=_LAYERED, depth=None, **cfg):
        with tempfile.TemporaryDirectory() as tmp:
            fonte, preparada = _prepara(tmp, depth=depth, **cfg)
            plano = draw_plan(fonte.scene_id, n=4, seed=0)
            with com_vocabulario_da_rota_a():
                return process_variant(preparada, plano[0], render_fn=render,
                                       config=_config(tmp, **cfg),
                                       provenance_base=_PROV)

    def test_sem_limiar_nenhum_gate_bloqueia(self):
        self._roda()                      # não levanta

    def test_todo_gate_medido_vai_para_o_metadado(self):
        quality = self._roda().quality
        for nome in ("aif_laplacian_variance", "bokeh_over_aif_sharpness",
                     "rendered_coc_p99_px", "defocus_saturation_ratio",
                     "mask_area_ratio_min", "depth_useful_levels", "focus_depth_m_min"):
            self.assertIn(nome, quality)
            self.assertIn("value", quality[nome])

    def test_todo_gate_do_relatorio_tem_slug_no_GATE_TO_REASON(self):
        """Sem a ponte total, um gate reprovado levantaria `KeyError` no laço."""
        for nome in self._roda().quality:
            self.assertIn(nome, GATE_TO_REASON, f"gate {nome} sem slug")

    def test_todo_slug_do_GATE_TO_REASON_esta_no_vocabulario_fechado(self):
        self.assertTrue(set(GATE_TO_REASON.values()) <= REJECTION_REASONS)

    def test_nitidez_da_AIF_bloqueia_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(min_aif_laplacian_variance=1e12)
        self.assertEqual(ctx.exception.reason, "gate_aif_sharpness")

    def test_render_que_nao_fez_nada_bloqueia(self):
        """O defeito A3 pelo lado do silêncio: renderer devolvendo a entrada."""
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(render=_render_nulo, max_bokeh_over_aif_sharpness=0.5)
        self.assertEqual(ctx.exception.reason, "gate_bokeh_not_blurrier")

    def test_coc_p99_pequeno_bloqueia(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(min_rendered_coc_p99_px=1e6)
        self.assertEqual(ctx.exception.reason, "gate_bokeh_not_blurrier")

    def test_saturacao_do_mapa_bloqueia(self):
        """Com K sorteado grande, `|CoC| >= MAX_COC` numa fração grande da imagem."""
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(k_distribution=_distribuicao_forte(),
                       max_defocus_saturation_ratio=0.0)
        self.assertEqual(ctx.exception.reason, "k_out_of_configured_range")

    def test_a_saturacao_medida_com_MAX_COC_100_e_maior_que_zero_com_K_grande(self):
        amostra = self._roda(k_distribution=_distribuicao_forte())
        self.assertGreater(amostra.quality["defocus_saturation_ratio"]["value"], 0.0)
        self.assertEqual(amostra.quality["defocus_saturation_ratio"]["threshold"], None)

    def test_area_da_banda_bloqueia_pelos_dois_lados(self):
        """Com K grande a banda é estreita; com K pequeno ela cobre quase tudo."""
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(k_distribution=_distribuicao_forte(), min_mask_area_ratio=0.9)
        self.assertEqual(ctx.exception.reason, "gate_mask_area_ratio")
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(max_mask_area_ratio=0.01)
        self.assertEqual(ctx.exception.reason, "gate_mask_area_ratio")

    def test_niveis_de_profundidade_bloqueiam(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(min_depth_useful_levels=10**9)
        self.assertEqual(ctx.exception.reason, "gate_depth_useful_levels")

    def test_plano_implausivel_bloqueia_pelo_gate_tambem(self):
        """Arame de tropeço: o pool restrito já impede, mas o gate tem que acusar."""
        from routes.route_a import enforce_gates
        with tempfile.TemporaryDirectory() as tmp:
            fonte, preparada = _prepara(tmp)
            relatorio = build_gate_report(
                aif_bgr=preparada.aif_bgr, bokeh_bgr=preparada.aif_bgr,
                mask=np.ones(preparada.image_hw, dtype=bool),
                disparity_u16=preparada.encoded.disparity_u16,
                coc_px=np.zeros(preparada.image_hw),
                defocus=np.zeros(preparada.image_hw),
                focus_disparity=1.0 / 5000.0, source=fonte,
                config=_config(tmp))
            with self.assertRaises(SampleRejected) as ctx:
                enforce_gates(relatorio)
        self.assertEqual(ctx.exception.reason, "gate_focus_depth_implausible")

    def test_render_em_outra_grade_rejeita_antes_dos_gates(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._roda(render=_render_grade_errada)
        self.assertEqual(ctx.exception.reason, "resolution_invalid")

    def test_profundidade_em_outra_grade_rejeita_a_imagem(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SampleRejected) as ctx:
            _prepara(tmp, depth=DepthEmOutraGrade())
        self.assertEqual(ctx.exception.reason, "resolution_invalid")

    def test_AIF_sem_tres_canais_rejeita(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SampleRejected) as ctx:
            prepare_image(FakeAif("x"), aif_bgr=np.zeros((8, 8), dtype=np.uint8),
                          depth_runtime=FakeDepth(), config=_config(tmp))
        self.assertEqual(ctx.exception.reason, "resolution_invalid")


# ==============================================================================
# Uma amostra — rótulo, proveniência, e o que NÃO existe nesta rota
# ==============================================================================

class UmaAmostra(unittest.TestCase):
    def _amostra(self, **cfg):
        with tempfile.TemporaryDirectory() as tmp:
            fonte, preparada = _prepara(tmp, **cfg)
            plano = draw_plan(fonte.scene_id, n=4, seed=0)
            with com_vocabulario_da_rota_a():
                return preparada, process_variant(
                    preparada, plano[0], render_fn=_LAYERED, config=_config(tmp, **cfg),
                    provenance_base=_PROV)

    def test_k_value_e_k_per_long_side_vezes_o_lado_longo(self):
        preparada, amostra = self._amostra()
        regra = amostra.provenance.extra["mask_rule"]
        self.assertAlmostEqual(
            amostra.control.k_value,
            regra["k_per_long_side"] * max(preparada.image_hw), places=9)

    def test_metadado_grava_disparidade_como_campo_primario(self):
        _, amostra = self._amostra()
        meta = amostra.metadata()
        self.assertIn("focus_disparity", meta)
        self.assertAlmostEqual(meta["focus_depth_m"], 1.0 / meta["focus_disparity"],
                               places=9)
        self.assertEqual(meta["max_coc"], MAX_COC)

    def test_o_que_a_rota_A_nao_tem(self):
        """Eq. 5, censura, validador analítico e segmentador: nenhum existe aqui."""
        _, amostra = self._amostra()
        self.assertEqual(amostra.control.k_source.value, "sampled_from_bc")
        self.assertIsNone(amostra.control.calibration_ssim)
        self.assertFalse(amostra.control.is_k_censored)
        self.assertIsNone(amostra.control.k_analytic)
        self.assertEqual(amostra.provenance.mask_model_sha256, "")
        self.assertIsNone(amostra.provenance.extra["route_a_decisions"]["birefnet"])
        self.assertIsNone(amostra.provenance.extra["route_a_decisions"]["eq5"])

    def test_o_renderer_E_o_produtor_do_rotulo(self):
        """Única rota em que isso vale (paper.txt:331, 292)."""
        _, amostra = self._amostra()
        self.assertTrue(amostra.provenance.renderer["is_final_label_renderer"])
        self.assertEqual(amostra.control.k_effective_factor, 0.9873)
        self.assertEqual(list(amostra.generated_images), ["bokeh"])
        self.assertEqual(amostra.channel_order, "bgr")

    def test_a_regra_da_mascara_reconstroi_a_banda(self):
        """`mask_rule` é o análogo do hash de modelo: o que permite refazer a máscara."""
        from control.contract import signed_coc_px
        preparada, amostra = self._amostra()
        regra = amostra.provenance.extra["mask_rule"]
        refeita = np.abs(signed_coc_px(preparada.depth_m, regra["focus_disparity"],
                                       regra["k_value"])) <= regra["focus_band_coc_px"]
        np.testing.assert_array_equal(np.asarray(amostra.mask) > 0.5, refeita)
        self.assertTrue(regra["is_derived_from_label"])
        self.assertFalse(regra["is_measured_on_scene"])

    def test_a_proveniencia_declara_que_agreement_e_por_construcao(self):
        _, amostra = self._amostra()
        self.assertEqual(amostra.focus.agreement, 1.0)
        self.assertTrue(np.isnan(amostra.focus.retention_in_region))
        self.assertIsNone(amostra.focus.initial_mask_area_ratio)
        self.assertIn("POR CONSTRUÇÃO",
                      amostra.provenance.extra["focus_region_semantics"])

    def test_focus_was_refined_sai_True_e_isso_e_o_defeito_DESCRITO(self):
        """Pino de comportamento atual, não aprovação.

        `FocusRegionRecord.was_refined` é `FocusSource(...) is not BIREFNET`
        (`dataio/sample.py:223`), então qualquer fonte nova sai como "refinada". A rota A
        não refina nada. A correção — testar pertinência a
        `{BIREFNET_REFINED, RETENTION_ONLY}` — está descrita no relatório e em
        `reference/ROTA_A_DECISOES.md`, e **não** foi aplicada aqui. Quando for, este
        teste passa a esperar `False` e é por ele que alguém vai lembrar de mudar.
        """
        _, amostra = self._amostra()
        self.assertTrue(amostra.focus.was_refined)

    def test_o_sorteio_vai_inteiro_para_a_proveniencia(self):
        """O histórico não gravou quais (K, D_focus) geraram cada variante."""
        _, amostra = self._amostra()
        sorteio = amostra.provenance.extra["sampling"]
        for chave in ("stratum_k", "stratum_focus", "n_strata", "variant_seed",
                      "samples_per_image", "focus_sampling_pool_fraction",
                      "focus_sampling_disparity_min", "k_distribution_n"):
            self.assertIn(chave, sorteio)

    def test_variantes_da_mesma_imagem_tem_K_e_plano_diferentes(self):
        """O sinal que o pré-treino existe para ensinar (paper.txt:332-333).

        Cena de gradiente puro, sem platô: cada estrato de quantil cai num valor de
        disparidade diferente. Com platô o comportamento é outro e está no teste abaixo.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fonte, preparada = _prepara(tmp, depth=DepthGradiente(), samples_per_image=8)
            config = _config(tmp, samples_per_image=8)
            with com_vocabulario_da_rota_a():
                amostras = [process_variant(preparada, d, render_fn=_LAYERED,
                                            config=config, provenance_base=_PROV)
                            for d in draw_plan(fonte.scene_id, n=8, seed=0)]
        ks = {round(a.control.k_value, 6) for a in amostras}
        focos = {round(a.control.focus_disparity, 9) for a in amostras}
        self.assertEqual(len(ks), 8)
        self.assertEqual(len(focos), 8)
        self.assertEqual(len({a.refs.scene_id for a in amostras}), 1)

    def test_um_plato_na_cena_colapsa_planos_de_foco_e_isso_e_declarado(self):
        """Consequência direta de amostrar o CONTEÚDO, e não a geometria.

        A `FakeDepth` tem 25% do quadro exatamente a 3 m. A função quantil é constante
        nessa faixa, então estratos diferentes caem no MESMO plano de foco. Não é defeito:
        é o sorteio seguindo a massa da cena, que é o que impede o plano de cair no vazio
        (ver `test_uniforme_na_faixa_esvazia_a_banda_e_o_quantil_nao`). O K continua
        diferente em todas as variantes, então nenhuma amostra é duplicata de outra.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fonte, preparada = _prepara(tmp, samples_per_image=8)
            config = _config(tmp, samples_per_image=8)
            with com_vocabulario_da_rota_a():
                amostras = [process_variant(preparada, d, render_fn=_LAYERED,
                                            config=config, provenance_base=_PROV)
                            for d in draw_plan(fonte.scene_id, n=8, seed=0)]
        focos = {round(a.control.focus_disparity, 9) for a in amostras}
        self.assertLess(len(focos), 8)
        self.assertEqual(len({round(a.control.k_value, 9) for a in amostras}), 8)


# ==============================================================================
# O laço: escrita, histograma, retomada, split, e o custo de uma imagem ruim
# ==============================================================================

class Loop(unittest.TestCase):
    def _fontes(self, n=2):
        return [FakeAif(f"gp_{i:012d}", source_dataset=(
            SOURCE_GENERATIVE_PHOTOGRAPHY if i % 2 == 0 else SOURCE_EBB))
            for i in range(n)]

    def _load(self, fonte):
        return _aif(seed=int(fonte.scene_id.split("_")[-1]))

    def _roda(self, tmp, fontes=None, depth=None, load=None, **cfg):
        fontes = fontes or self._fontes()
        split = build_scene_split([f.scene_id for f in fontes], val_fraction=0.3)
        with com_vocabulario_da_rota_a():
            return run_route_a(fontes, load_aif=load or self._load,
                               depth_runtime=depth or FakeDepth(), render_fn=_LAYERED,
                               split=split, config=_config(tmp, **cfg),
                               provenance_base=_PROV)

    def test_gera_N_variantes_por_imagem_e_o_K_sobrevive_ao_disco(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp)
            self.assertEqual(stats.written, 8)              # 2 imagens x 4 variantes
            linhas = list(iter_manifest(tmp))
            self.assertEqual(len(linhas), 8)
            self.assertEqual(len({l["scene_id"] for l in linhas}), 2)
            ks = [l["k_value"] for l in linhas]
            self.assertEqual(len(set(np.round(ks, 6))), 8, "K não sobreviveu — D1")
            for l in linhas:
                self.assertEqual(l["max_coc"], MAX_COC)
                self.assertEqual(l["depth_backend"], "depth_pro")
                self.assertEqual(l["k_source"], "sampled_from_bc")
                self.assertEqual(l["mask_source"], SAMPLED_MASK_SOURCE_VALUE)
                self.assertEqual(l["focus_source"], SAMPLED_FOCUS_SOURCE_VALUE)
                self.assertIn(l["split"], {"train", "val"})
                self.assertTrue(l["is_valid_for_control"])
            self.assertTrue((Path(tmp) / "split.json").exists())

    def test_as_variantes_de_uma_imagem_gravam_UM_png_de_profundidade(self):
        """Ponta a ponta, na rota em que a economia vale mais.

        `PreparedImage` já calculava `encode_depth` uma vez por imagem e entregava o
        mesmo `EncodedDepth` às N variantes — o writer é que gravava N cópias idênticas.
        Medido no `a_release`: 120 de 120 cenas com a profundidade byte a byte idêntica
        entre as 41 variantes; 69.700 arquivos e 31,7 GB contra 1.700 e 0,77 GB.

        As máscaras continuam sendo N: elas são o que VARIA entre as variantes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self._roda(tmp, fontes=self._fontes(2))
            raiz = Path(tmp)
            self.assertEqual(
                sorted(p.name for p in (raiz / "depth").glob("*.png")),
                ["gp_000000000000.png", "gp_000000000001.png"])
            self.assertEqual(len(list((raiz / "mask").glob("*.png"))), 8)
            for linha in iter_manifest(tmp):
                with self.subTest(sample_id=linha["sample_id"]):
                    self.assertEqual(linha["depth_ref"],
                                     f"depth/{linha['scene_id']}.png")
                    self.assertTrue((raiz / linha["depth_ref"]).is_file())

    def test_o_run_config_declara_o_layout_da_profundidade(self):
        """Quem valida e quem lê não inferem o layout contando arquivos: um release por
        cena tem menos PNG que amostras de propósito, e contar confundiria isso com um
        release incompleto."""
        from routes.route_a import RouteAConfig
        from dataio import DepthLayout
        self.assertIs(RouteAConfig.depth_layout, DepthLayout.PER_SCENE)

    def test_as_variantes_de_uma_imagem_ficam_do_MESMO_lado_do_split(self):
        """Sem isso, 41 variantes da mesma imagem vazam entre treino e validação (A9)."""
        with tempfile.TemporaryDirectory() as tmp:
            self._roda(tmp, fontes=self._fontes(6))
            por_cena = {}
            for l in iter_manifest(tmp):
                por_cena.setdefault(l["scene_id"], set()).add(l["split"])
            self.assertTrue(all(len(v) == 1 for v in por_cena.values()))

    def test_sample_id_e_cena_mais_variante(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._roda(tmp, fontes=self._fontes(1))
            ids = sorted(l["sample_id"] for l in iter_manifest(tmp))
        self.assertEqual(ids, [f"gp_000000000000_v{i:02d}" for i in range(4)])

    def test_a_bokeh_renderizada_vai_para_o_disco_com_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._roda(tmp, fontes=self._fontes(1))
            jpegs = sorted((Path(tmp) / "generated").glob("*_bokeh.jpg"))
            self.assertEqual(len(jpegs), 4)
            ledger = [json.loads(l) for l in
                      (Path(tmp) / "generated_images.jsonl").read_text().splitlines()]
            self.assertEqual(len(ledger), 4)
            for linha in ledger:
                self.assertEqual(linha["bokeh_role"], "generated_by_bokehme")
                self.assertEqual(linha["aif_role"], "reference")
                self.assertEqual(len(linha["bokeh_jpeg_sha256"]), 64)
                self.assertEqual(linha["source_revision"], "rev-abc")

    def test_o_mapa_de_defocus_nao_vai_para_o_disco_e_e_derivavel(self):
        from control.contract import defocus_map
        from dataio.encoding import EncodedDepth, decode_depth_m
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            self._roda(tmp, fontes=self._fontes(1))
            meta = read_metadata(tmp, "gp_000000000000_v00")
            # O caminho vem do `depth_ref` gravado no metadado — que é o contrato de
            # leitura depois da dedup: a profundidade é por CENA e o nome dela NÃO é
            # dedutível do `sample_id`.
            u16 = np.array(Image.open(Path(tmp) / meta["depth_ref"]))
            self.assertEqual(list(Path(tmp).glob("**/*defocus*")), [])
        enc = EncodedDepth(u16, meta["disparity_min"], meta["disparity_max"],
                           (meta["image_h"], meta["image_w"]),
                           (meta["depth_h"], meta["depth_w"]))
        mapa = defocus_map(decode_depth_m(enc), meta["focus_disparity"], meta["k_value"])
        self.assertGreater(float(mapa.max()), 0.0)
        self.assertLessEqual(float(mapa.max()), 1.0)

    def test_a_mascara_no_disco_e_a_banda_da_regra_gravada(self):
        """Reconstrução byte a byte a partir do metadado — o que fecha o item F6."""
        from control.contract import signed_coc_px
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            self._roda(tmp, fontes=self._fontes(1))
            meta = read_metadata(tmp, "gp_000000000000_v02")
            mascara = np.array(Image.open(Path(tmp) / "mask" /
                                          "gp_000000000000_v02.png")) > 127
        regra = meta["provenance"]["extra"]["mask_rule"]
        depth = FakeDepth().infer(_aif(seed=0)).values_m
        refeita = np.abs(signed_coc_px(depth, regra["focus_disparity"],
                                       regra["k_value"])) <= regra["focus_band_coc_px"]
        np.testing.assert_array_equal(mascara, refeita)

    def test_retomada_pula_a_imagem_INTEIRA_sem_rodar_o_Depth_Pro(self):
        """Retomar 70 mil amostras não pode custar 1,7 mil inferências de profundidade."""
        with tempfile.TemporaryDirectory() as tmp:
            primeira = FakeDepth()
            self._roda(tmp, depth=primeira)
            self.assertEqual(primeira.calls, 2)
            segunda = FakeDepth()
            stats = self._roda(tmp, depth=segunda)
        self.assertEqual(stats.skipped_done, 8)
        self.assertEqual(stats.written, 0)
        self.assertEqual(segunda.calls, 0, "o Depth Pro rodou de novo numa retomada")

    def test_uma_imagem_ruim_custa_N_variantes_e_UMA_linha_no_histograma(self):
        def load_que_falha(fonte):
            if fonte.scene_id.endswith("0"):
                from control.contract import reject
                reject("source_image_unreadable", "arquivo truncado")
            return self._load(fonte)

        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp, load=load_que_falha)
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "rejections.jsonl").read_text().splitlines()
                      if json.loads(l)["status"] == "rejected"]
        self.assertEqual(stats.written, 4)
        self.assertEqual(stats.images_lost, 1)
        self.assertEqual(stats.variants_lost_to_image, 4)
        self.assertEqual(len(linhas), 1, "a imagem ruim entrou N vezes no histograma")
        self.assertEqual(linhas[0]["reason"], "source_image_unreadable")
        self.assertEqual(linhas[0]["variants_lost"], 4)
        self.assertEqual(linhas[0]["stage"], "image")

    def test_rejeicao_por_variante_entra_uma_vez_por_variante(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp, min_rendered_coc_p99_px=1e9)
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "rejections.jsonl").read_text().splitlines()]
        self.assertEqual(stats.written, 0)
        rejeitadas = [l for l in linhas if l["status"] == "rejected"]
        self.assertEqual(len(rejeitadas), 8)
        self.assertTrue(all(l["reason"] == "gate_bokeh_not_blurrier" for l in rejeitadas))
        self.assertTrue(all(l["stage"] == "variant" for l in rejeitadas))
        self.assertEqual({l["variant_index"] for l in rejeitadas}, {0, 1, 2, 3})

    def test_limite_de_amostras_aceitas(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp, limit=3)
        self.assertEqual(stats.written, 3)

    def test_se_o_vocabulario_SUMIR_o_laco_nao_toca_em_nada(self):
        """A guarda roda ANTES de qualquer inferência: zero chamadas de profundidade,
        nenhum manifesto criado."""
        fontes = self._fontes(1)
        split = build_scene_split([f.scene_id for f in fontes], val_fraction=0.3)
        depth = FakeDepth()
        with tempfile.TemporaryDirectory() as tmp, sem_vocabulario_da_rota_a():
            with self.assertRaises(VocabularyExtensionRequired):
                run_route_a(fontes, load_aif=self._load, depth_runtime=depth,
                            render_fn=_LAYERED, split=split, config=_config(tmp),
                            provenance_base=_PROV)
            self.assertEqual(depth.calls, 0)
            self.assertFalse((Path(tmp) / "manifest.jsonl").exists())

    def test_o_resumo_conta_em_cenas_E_em_amostras(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp)
        resumo = stats.summary()
        self.assertIn("cenas (imagens)  : 2", resumo)
        self.assertIn("amostras         : 8", resumo)
        self.assertIn("4.0 por imagem", resumo)
        self.assertIn("k_per_long_side", resumo)
        self.assertIn(SOURCE_EBB, resumo)

    def test_o_resumo_grita_quando_a_razao_cai_para_1(self):
        """Uma variante por imagem é o defeito A2, e ele tem que doer no log."""
        stats = RouteAStats()
        stats.per_scene.update({"a": 1, "b": 1, "c": 1})
        self.assertIn("ATENÇÃO", "\n".join(stats.coverage_summary()))

    def test_resumo_vazio_nao_explode(self):
        self.assertIn("nenhuma amostra aceita", RouteAStats().summary())

    def test_cena_inteira_no_teto_do_DepthPro_rejeita_a_imagem(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp, depth=DepthTodaCeu(), fontes=self._fontes(1))
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "rejections.jsonl").read_text().splitlines()]
        self.assertEqual(stats.written, 0)
        self.assertEqual(linhas[0]["reason"], "focus_depth_implausible")

    def test_cena_com_ceu_nao_perde_variante_nenhuma(self):
        """Com a população restrita, os 10% de céu não custam nenhuma variante."""
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._roda(tmp, depth=DepthComCeu(), fontes=self._fontes(1))
            metas = [read_metadata(tmp, l["sample_id"]) for l in iter_manifest(tmp)]
        self.assertEqual(stats.written, 4)
        for meta in metas:
            sorteio = meta["provenance"]["extra"]["sampling"]
            self.assertLess(sorteio["focus_sampling_pool_fraction"], 1.0)
            self.assertLessEqual(meta["focus_depth_m"], FOCUS_DEPTH_MAX_M)


# ==============================================================================
# A fonte — Generative Photography [80] + EBB! [27]
# ==============================================================================

def _medida(nome: str, *, lapvar: float, hw=(600, 800), sha: Optional[str] = None):
    digest = sha or ("%064x" % (abs(hash(nome)) % (1 << 256)))
    return SharpnessMeasurement(image_hw=hw, sharpness_hw=hw,
                                laplacian_variance=lapvar, sha256=digest)


class Fonte(unittest.TestCase):
    def test_DiffCamera_nao_e_fonte_e_o_erro_diz_por_que(self):
        """A divergência nº 1 da auditoria: o pipeline antigo aceitava DiffCamera."""
        with self.assertRaises(ValueError) as ctx:
            validate_source_name("DiffCamera")
        self.assertIn("[69]", str(ctx.exception))
        self.assertIn("baseline", str(ctx.exception))

    def test_fonte_desconhecida_e_erro_de_configuracao_nao_rejeicao(self):
        with self.assertRaises(ValueError):
            validate_source_name("LFDOF")
        self.assertEqual(validate_source_name(SOURCE_EBB), SOURCE_EBB)

    def _arvore(self, tmp, nomes):
        raiz = Path(tmp)
        for nome in nomes:
            caminho = raiz / nome
            caminho.parent.mkdir(parents=True, exist_ok=True)
            caminho.write_bytes(b"x")
        return raiz

    def test_enumera_medindo_nitidez_sha256_e_resolucao(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = self._arvore(tmp, ["a.png", "sub/b.jpg", "c.txt"])
            log = RejectionLog()
            achadas = enumerate_candidates(
                raiz, SOURCE_GENERATIVE_PHOTOGRAPHY, log=log,
                revision="v1", note="lado nítido",
                measure=lambda p: _medida(p.name, lapvar=100.0 + len(p.name),
                                          sha=f"{p.name:>0}".encode().hex().ljust(64, "0")))
        self.assertEqual(len(achadas), 2, "arquivo não-imagem entrou")
        for imagem in achadas:
            self.assertEqual(imagem.source_dataset, SOURCE_GENERATIVE_PHOTOGRAPHY)
            self.assertEqual(imagem.source_revision, "v1")
            self.assertEqual(imagem.source_note, "lado nítido")
            self.assertTrue(imagem.scene_id.startswith("gp_"))
            self.assertEqual(imagem.sharpness_hw, (600, 800))

    def test_scene_id_nao_depende_da_ordem_de_leitura(self):
        """O antigo era o índice na lista: reordenar renomeava todas as amostras (A9)."""
        medidas = {"a.png": "1" * 64, "b.png": "2" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            raiz = self._arvore(tmp, list(medidas))
            achadas = enumerate_candidates(
                raiz, SOURCE_EBB, log=RejectionLog(),
                measure=lambda p: _medida(p.name, lapvar=1.0, sha=medidas[p.name]))
        self.assertEqual({im.scene_id for im in achadas},
                         {"ebb_" + "1" * 12, "ebb_" + "2" * 12})

    def test_arquivo_duplicado_por_sha256_rejeita_com_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = self._arvore(tmp, ["a.png", "copia.png"])
            log = RejectionLog()
            achadas = enumerate_candidates(
                raiz, SOURCE_EBB, log=log,
                measure=lambda p: _medida(p.name, lapvar=1.0, sha="f" * 64))
        self.assertEqual(len(achadas), 1)
        self.assertEqual(log.reasons["source_duplicate_sample"], 1)

    def test_imagem_ilegivel_entra_no_histograma_e_nao_mata_a_enumeracao(self):
        def mede(caminho):
            if caminho.name == "ruim.png":
                raise OSError("truncado")
            return _medida(caminho.name, lapvar=5.0, sha=caminho.name.ljust(64, "0"))

        with tempfile.TemporaryDirectory() as tmp:
            raiz = self._arvore(tmp, ["boa.png", "ruim.png", "outra.png"])
            log = RejectionLog()
            achadas = enumerate_candidates(raiz, SOURCE_EBB, log=log, measure=mede)
        self.assertEqual(len(achadas), 2)
        self.assertEqual(log.reasons["source_image_unreadable"], 1)

    def test_resolucao_invalida_entra_no_histograma(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = self._arvore(tmp, ["a.png"])
            log = RejectionLog()
            achadas = enumerate_candidates(
                raiz, SOURCE_EBB, log=log,
                measure=lambda p: _medida(p.name, lapvar=1.0, hw=(0, 0),
                                          sha="a" * 64))
        self.assertEqual(achadas, [])
        self.assertEqual(log.reasons["source_metadata_field_invalid"], 1)

    # -- o corte POR FONTE, que é o defeito A8/D13 ----------------------------

    def _candidatas(self, fonte, variancias, slug):
        return [AifImage(scene_id=f"{slug}_{i:012d}", source_dataset=fonte,
                         source_sample_id=f"{i}.png", aif_ref=f"{i}.png",
                         path=Path(f"/tmp/{i}.png"), sha256=f"{i:064d}",
                         image_hw=(600, 800), laplacian_variance=v,
                         sharpness_hw=(600, 800))
                for i, v in enumerate(variancias)]

    def test_o_ranking_e_o_corte_acontecem_DENTRO_de_cada_fonte(self):
        """Ranking global selecionaria pela fonte de maior resolução, não pela mais nítida.

        Aqui `[80]` tem variância 10x maior que a EBB! — que é exatamente o que resolução
        e compressão diferentes produzem. Com ranking único e corte em 4, as 4 vagas iriam
        todas para `[80]` e a EBB! sairia inteira do pré-treino.
        """
        altas = self._candidatas(SOURCE_GENERATIVE_PHOTOGRAPHY,
                                 [1000.0, 900.0, 800.0, 700.0], "gp")
        baixas = self._candidatas(SOURCE_EBB, [100.0, 90.0, 80.0, 70.0], "ebb")
        mantidas, cortes = rank_and_cut_per_source(
            altas + baixas,
            quota={SOURCE_GENERATIVE_PHOTOGRAPHY: 2, SOURCE_EBB: 2})
        por_fonte = {}
        for im in mantidas:
            por_fonte.setdefault(im.source_dataset, []).append(im.laplacian_variance)
        self.assertEqual(por_fonte[SOURCE_GENERATIVE_PHOTOGRAPHY], [1000.0, 900.0])
        self.assertEqual(por_fonte[SOURCE_EBB], [100.0, 90.0])
        cortes_por_fonte = {c.source_dataset: c for c in cortes}
        self.assertEqual(cortes_por_fonte[SOURCE_EBB].cut_variance, 90.0)
        self.assertEqual(cortes_por_fonte[SOURCE_EBB].n_candidates, 4)
        self.assertEqual(cortes_por_fonte[SOURCE_EBB].to_dict()["ranking_scope"],
                         "per_source")

    def test_sem_quota_nada_e_cortado(self):
        """Quota é declarada, nunca default: a divisão do pool de 1,7K não é publicada."""
        candidatas = self._candidatas(SOURCE_EBB, [3.0, 2.0, 1.0], "ebb")
        mantidas, _ = rank_and_cut_per_source(candidatas)
        self.assertEqual(len(mantidas), 3)

    def test_piso_de_variancia_por_fonte(self):
        candidatas = self._candidatas(SOURCE_EBB, [3.0, 2.0, 1.0], "ebb")
        mantidas, cortes = rank_and_cut_per_source(
            candidatas, min_variance={SOURCE_EBB: 2.0})
        self.assertEqual([im.laplacian_variance for im in mantidas], [3.0, 2.0])
        self.assertEqual(cortes[0].min_variance, 2.0)

    def test_quota_para_fonte_que_nao_e_do_paper_e_erro(self):
        with self.assertRaises(ValueError):
            rank_and_cut_per_source(self._candidatas(SOURCE_EBB, [1.0], "ebb"),
                                    quota={"DiffCamera": 10})

    def test_empate_de_variancia_e_desfeito_pelo_sha256(self):
        candidatas = self._candidatas(SOURCE_EBB, [5.0, 5.0, 5.0], "ebb")
        uma, _ = rank_and_cut_per_source(candidatas, quota={SOURCE_EBB: 2})
        outra, _ = rank_and_cut_per_source(list(reversed(candidatas)),
                                           quota={SOURCE_EBB: 2})
        self.assertEqual([im.sha256 for im in uma], [im.sha256 for im in outra])

    def test_o_resumo_cita_a_ancora_de_1700_do_paper(self):
        candidatas = self._candidatas(SOURCE_EBB, [5.0, 4.0], "ebb")
        mantidas, cortes = rank_and_cut_per_source(candidatas)
        resumo = enumeration_summary(mantidas, RejectionLog(), cortes)
        self.assertIn("1700", resumo)
        self.assertIn("POR FONTE", resumo)

    def test_o_resumo_avisa_quando_a_fonte_tem_varias_resolucoes_de_medicao(self):
        a, b = self._candidatas(SOURCE_EBB, [5.0, 4.0], "ebb")
        b = AifImage(**{**b.__dict__, "sharpness_hw": (300, 400)})
        _, cortes = rank_and_cut_per_source([a, b])
        self.assertIn("resoluções de medição", enumeration_summary([a, b],
                                                                   RejectionLog(), cortes))


class PixelsDaFonte(unittest.TestCase):
    """Leitura de arquivo de verdade — com Pillow, sem cv2."""

    def _png(self, caminho, array, modo="RGB"):
        from PIL import Image
        Image.fromarray(array, mode=modo).save(caminho)

    def test_le_em_BGR_e_mede_nitidez_na_grade_declarada(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "a.png"
            self._png(caminho, _aif(64))
            medida = genphoto_ebb.measure_with_pillow(caminho)
            self.assertEqual(medida.image_hw, (64, 64))
            self.assertEqual(medida.sharpness_hw, (64, 64))
            self.assertGreater(medida.laplacian_variance, 0.0)
            self.assertEqual(len(medida.sha256), 64)
            reduzida = genphoto_ebb.measure_with_pillow(caminho, long_side=32)
            self.assertEqual(reduzida.sharpness_hw, (32, 32))
            self.assertNotEqual(reduzida.laplacian_variance, medida.laplacian_variance)

    def test_canal_alpha_nao_opaco_rejeita_com_slug(self):
        """`convert("RGB")` comporia sobre preto e INVENTARIA pixel na entrada do modelo."""
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "a.png"
            rgba = np.dstack([_aif(32), np.full((32, 32), 128, dtype=np.uint8)])
            self._png(caminho, rgba, modo="RGBA")
            with self.assertRaises(SampleRejected) as ctx:
                genphoto_ebb.load_bgr(caminho)
        self.assertEqual(ctx.exception.reason, "source_image_alpha_not_opaque")

    def test_resolucao_diferente_da_declarada_rejeita(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "a.png"
            self._png(caminho, _aif(32))
            with self.assertRaises(SampleRejected) as ctx:
                genphoto_ebb.load_bgr(caminho, expected_hw=(64, 64))
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_arquivo_que_nao_e_imagem_rejeita(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "a.png"
            caminho.write_bytes(b"nao sou png")
            with self.assertRaises(SampleRejected) as ctx:
                genphoto_ebb.load_bgr(caminho)
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_o_loader_grava_o_ledger_de_bytes_de_origem(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "a.png"
            self._png(caminho, _aif(32))
            imagem = AifImage(
                scene_id="ebb_000000000000", source_dataset=SOURCE_EBB,
                source_sample_id="a.png", aif_ref="a.png", path=caminho,
                sha256="a" * 64, image_hw=(32, 32), laplacian_variance=9.0,
                sharpness_hw=(32, 32), source_revision="rev-1")
            loader = genphoto_ebb.AifImageLoader(
                ledger_path=Path(tmp) / "source_images.jsonl")
            bgr = loader(imagem)
            loader(imagem)                          # não duplica a linha
            loader.close()
            linhas = [json.loads(l) for l in
                      (Path(tmp) / "source_images.jsonl").read_text().splitlines()]
        self.assertEqual(bgr.shape, (32, 32, 3))
        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0]["role"], "aif_reference")
        self.assertEqual(linhas[0]["aif_sha256"], "a" * 64)
        self.assertEqual(linhas[0]["source_revision"], "rev-1")


class FonteRealAlimentandoARota(unittest.TestCase):
    """Adaptador de fonte + rota, juntos, com arquivos de verdade no disco.

    O `Protocol` não é verificado em tempo de execução: `AifImage` satisfazer `AifSource`
    é afirmação que só um teste sustenta. Este é o teste — se alguém renomear um campo do
    adaptador, é aqui que aparece, e não na amostra 40.000 de um run de GPU.
    """

    def test_enumera_ranqueia_e_gera(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            for nome, fonte in (("gp", SOURCE_GENERATIVE_PHOTOGRAPHY), ("ebb", SOURCE_EBB)):
                (raiz / nome).mkdir()
                for i in range(2):
                    Image.fromarray(_aif(64, seed=i + (0 if nome == "gp" else 10)),
                                    mode="RGB").save(raiz / nome / f"{i}.png")

            log = RejectionLog()
            candidatas = []
            for nome, fonte in (("gp", SOURCE_GENERATIVE_PHOTOGRAPHY),
                                ("ebb", SOURCE_EBB)):
                candidatas.extend(enumerate_candidates(raiz / nome, fonte, log=log,
                                                       revision="rev-teste"))
            self.assertEqual(len(candidatas), 4)
            mantidas, cortes = rank_and_cut_per_source(
                candidatas, quota={SOURCE_GENERATIVE_PHOTOGRAPHY: 1, SOURCE_EBB: 1})
            self.assertEqual(len(mantidas), 2)

            saida = raiz / "out"
            loader = genphoto_ebb.AifImageLoader(
                ledger_path=saida / "source_images.jsonl")
            split = build_scene_split([im.scene_id for im in mantidas],
                                      val_fraction=0.3)
            with com_vocabulario_da_rota_a():
                stats = run_route_a(
                    mantidas, load_aif=loader, depth_runtime=FakeDepth(),
                    render_fn=_LAYERED, split=split,
                    config=_config(saida, samples_per_image=3),
                    provenance_base=_PROV)
            loader.close()
            linhas = list(iter_manifest(saida))
            metas = [read_metadata(saida, l["sample_id"]) for l in linhas]
            ledger = [json.loads(l) for l in
                      (saida / "source_images.jsonl").read_text().splitlines()]

        self.assertEqual(stats.written, 6)                     # 2 imagens x 3 variantes
        self.assertEqual({l["source_dataset"] for l in linhas},
                         {SOURCE_GENERATIVE_PHOTOGRAPHY, SOURCE_EBB})
        self.assertEqual(len(ledger), 2, "o ledger é por IMAGEM, não por variante")
        for meta in metas:
            extra = meta["provenance"]["extra"]
            self.assertEqual(extra["source_revision"], "rev-teste")
            self.assertIsNotNone(extra["aif_laplacian_variance"])
            self.assertEqual(extra["aif_sharpness_hw"], [64, 64])
            self.assertEqual(meta["image_h"], 64)


if __name__ == "__main__":
    unittest.main()
