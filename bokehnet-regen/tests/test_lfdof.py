"""Testes do adaptador de fonte do LFDOF e do seu carregador de pixels.

Todos rodam **sem rede**, com fixtures em memória: o CI não pode depender de um repo
privado no HF estar de pé. Os números citados nas asserções vêm da medição registrada
no cabeçalho de `sources/lfdof.py` (83 shards, 11.972 linhas, 840 cenas), lida por
projeção de coluna do parquet — fixture em memória prova que o parser funciona no que
eu escrevi, não no que o dataset tem, e por isso a medição está no docstring e não
aqui.

O que cada bloco defende, em uma linha:

- **o `scene_id` agrupa todas as variantes de uma AIF**, e o split não pode separá-las:
  é a única coisa que impede vazamento numa fonte com 15 desfocadas por cena;
- nome fora do padrão levanta, nunca devolve `None` — `None` vira `scene_id` ausente,
  que é amostra fora do split, que é vazamento com outro nome;
- **os quatro campos ópticos são `None` e NÃO podem ser injetados** — e `None` aqui faz
  o gate ficar inaplicável, não reprovado (o defeito C2);
- `scene_level_count` é a contagem OBSERVADA, nunca `max(level)`: 3 cenas têm buraco;
- **todo caminho de rejeição tem um teste que o faz disparar.** Gate que não pode
  reprovar é pior que gate nenhum — este projeto já teve um teste chamado "detecta
  vazamento" que afirmava `clean is True`.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                      # noqa: E402
from PIL import Image                                                   # noqa: E402

from control.contract import REJECTION_REASONS, SampleRejected          # noqa: E402
from dataio.split import split_from_source                              # noqa: E402
from qc.gates import aif_aperture_is_narrow                             # noqa: E402
from qc.rejection import RejectionLog                                   # noqa: E402
from sources.lfdof import (                                             # noqa: E402
    ALIGNMENT_ALIGNED, ALIGNMENT_MISALIGNED, ALIGNMENT_SHIFT,
    ALL_REJECTION_REASONS, LFDOF_AIF_COLUMN, LFDOF_BOKEH_COLUMN, LFDOF_DATASET,
    LFDOF_IMAGE_HW, LFDOF_UNUSED_COLUMN, NEW_REJECTION_REASONS,
    SOURCE_REJECTION_REASONS, SOURCE_SPLITS, LFDOFPair, ParsedName, enumerate_pairs,
    enumeration_summary, pair_from_name, parse_file_name_base, parse_full_name,
    reject_source, scene_key, scene_numbers_shared_between_splits,
    scene_source_splits, scenes_with_level_gaps,
)
from sources.lfdof_images import (                                      # noqa: E402
    COLUMN_ROLE, INDEX_FILENAME, LFDOFImageLoader, MirrorIndex, MirrorLocation,
    expected_cell_path, order_pairs_for_sequential_read, sample_pairs_for_pilot,
)

try:
    import pyarrow  # noqa: F401
    TEM_PYARROW = True
except ImportError:
    TEM_PYARROW = False


# ==============================================================================
# Fixtures
# ==============================================================================

def _name(scene="1275", level=1, split="train", suffix="aligned") -> str:
    """Um `file_name_base` do espelho. `1275` é uma cena real, de 15 níveis."""
    return f"lfdof_{split}_data_{scene}_level_{level}_{suffix}"


def _cena(scene="1275", niveis=15, split="train", suffix="aligned") -> list[str]:
    return [_name(scene, n, split, suffix) for n in range(1, niveis + 1)]


def _log() -> RejectionLog:
    return RejectionLog()          # sem `path`: histograma em memória, zero I/O


def _par(scene="1275", level=1, split="train", suffix="aligned",
         count=15) -> LFDOFPair:
    return pair_from_name(_name(scene, level, split, suffix), scene_level_count=count)


def _png(rgb: tuple[int, int, int], hw: tuple[int, int] = (4, 6),
         alpha: int | None = None) -> bytes:
    """PNG de cor sólida. Com `alpha`, sai RGBA — que é o modo REAL do LFDOF."""
    if alpha is None:
        arr = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
        arr[:, :] = rgb
        img = Image.fromarray(arr, mode="RGB")
    else:
        arr = np.zeros((hw[0], hw[1], 4), dtype=np.uint8)
        arr[:, :, :3] = rgb
        arr[:, :, 3] = alpha
        img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _index(pares: dict[str, tuple[str, int]], shards: list[str]) -> MirrorIndex:
    return MirrorIndex({n: MirrorLocation(s, r) for n, (s, r) in pares.items()},
                       shards=shards)


class _FakeColumn:
    def __init__(self, valores: list):
        self._v = valores

    def __getitem__(self, i):
        class _Cell:
            def __init__(self, v):
                self._v = v

            def as_py(self):
                return self._v
        return _Cell(self._v[i])


class _FakeTable:
    """O mínimo de `pyarrow.Table` que `LFDOFImageLoader` toca.

    Existe para que a suíte cubra `__call__`, o ledger, a checagem de papel e o memo de
    AIF **sem** pyarrow: o python desta máquina não o tem, e um caminho testado só no
    cluster é um caminho não testado.
    """

    def __init__(self, colunas: dict[str, list]):
        self._c = {k: _FakeColumn(v) for k, v in colunas.items()}

    def column(self, name):
        return self._c[name]


class _LoaderFake(LFDOFImageLoader):
    """Loader com `_table` trocada por uma tabela em memória."""

    def __init__(self, tabelas: dict[str, _FakeTable], index: MirrorIndex, **kw):
        super().__init__("/nao/usado", index, **kw)
        self._tabelas = tabelas

    def _table(self, shard: str):
        return self._tabelas[shard]


def _cell(nome: str, papel: str, rgb=(255, 0, 0), hw=(4, 6), alpha=255,
          path: str | None = ...):
    coluna = {"focus": LFDOF_AIF_COLUMN, "blur": LFDOF_BOKEH_COLUMN}[papel]
    if path is ...:
        path = expected_cell_path(nome, coluna)
    return {"bytes": _png(rgb, hw, alpha), "path": path}


# ==============================================================================
# O nome do espelho
# ==============================================================================

class TestParseNome(unittest.TestCase):

    def test_extrai_cena_e_nivel(self):
        self.assertEqual(parse_file_name_base(_name("1275", 3)), ("train_1275", 3))

    def test_scene_id_e_qualificado_pelo_split(self):
        """`train_1275` e `test_1275` são cenas DIFERENTES.

        Hoje a interseção medida de números entre splits é 0, então a qualificação não
        está salvando nada — ela está mantendo a chave correta se a origem mudar, e
        igual à da RealBokeh, onde a numeração de fato reinicia.
        """
        self.assertEqual(scene_key("train", "1275"), "train_1275")
        self.assertNotEqual(parse_full_name(_name("1275", 1, "train")).scene_id,
                            parse_full_name(_name("1275", 1, "test")).scene_id)

    def test_campos_completos(self):
        p = parse_full_name(_name("2464", 7, "test"))
        self.assertIsInstance(p, ParsedName)
        self.assertEqual((p.scene_id, p.scene_number, p.level, p.source_split),
                         ("test_2464", "2464", 7, "test"))
        self.assertEqual(p.alignment, ALIGNMENT_ALIGNED)
        self.assertIsNone(p.alignment_shift_px_at_source_hw)
        self.assertEqual(p.raw, _name("2464", 7, "test"))

    def test_misaligned_nao_inventa_deslocamento_zero(self):
        """`misaligned` é "não fechou e não digo quanto". 0,0 seria o valor de
        "perfeitamente alinhado" — o fallback mais perigoso possível aqui."""
        p = parse_full_name(_name(suffix="misaligned"))
        self.assertEqual(p.alignment, ALIGNMENT_MISALIGNED)
        self.assertIsNone(p.alignment_shift_px_at_source_hw)
        self.assertFalse(_par(suffix="misaligned").is_aligned)

    def test_shift_vira_numero_em_pixel(self):
        p = parse_full_name(_name(suffix="shift_4.4px"))
        self.assertEqual(p.alignment, ALIGNMENT_SHIFT)
        self.assertAlmostEqual(p.alignment_shift_px_at_source_hw, 4.4)
        self.assertFalse(_par(suffix="shift_4.4px").is_aligned)

    def test_shift_inteiro_tambem_casa(self):
        self.assertAlmostEqual(
            parse_full_name(_name(suffix="shift_3px")).alignment_shift_px_at_source_hw,
            3.0)

    def test_aligned_e_o_unico_is_aligned(self):
        self.assertTrue(_par().is_aligned)

    def test_nivel_alto_casa(self):
        """A cena de 17 níveis existe: 1 de 840, medida."""
        self.assertEqual(parse_file_name_base(_name(level=17))[1], 17)


class TestParseNomeRejeita(unittest.TestCase):
    """Cada caminho de `source_name_unparseable` e `source_level_out_of_range`."""

    def _rejeita(self, nome, motivo="source_name_unparseable"):
        with self.assertRaises(SampleRejected) as ctx:
            parse_file_name_base(nome)
        self.assertEqual(ctx.exception.reason, motivo)
        return ctx.exception

    def test_nome_vazio(self):
        self._rejeita("")

    def test_nome_nao_string(self):
        self._rejeita(None)
        self._rejeita(1275)

    def test_prefixo_errado(self):
        """O nome da RealBokeh NÃO pode ser aceito pelo parser do LFDOF."""
        self._rejeita("timseizinger_realbokeh_3mp_train_f_1000_level_1_aligned")

    def test_sem_o_literal_data(self):
        self._rejeita("lfdof_train_1275_level_1_aligned")

    def test_cena_nao_numerica(self):
        self._rejeita("lfdof_train_data_abc_level_1_aligned")

    def test_split_que_a_origem_nao_publica(self):
        """`validation` não existe no LFDOF (medido: só train e test).

        Aparecer como `source_name_unparseable` no histograma é o comportamento
        desejado: alto e claro, em vez de virar um terceiro `scene_id` em silêncio.
        """
        exc = self._rejeita(_name(split="validation"))
        self.assertIn("validation", str(exc))
        self.assertNotIn("validation", SOURCE_SPLITS)

    def test_sufixo_de_alinhamento_desconhecido(self):
        self._rejeita(_name(suffix="quasealinhado"))
        self._rejeita(_name(suffix="shift_4.4"))        # sem o "px"
        self._rejeita(_name(suffix="shift_px"))         # sem o número

    def test_nivel_ausente(self):
        self._rejeita("lfdof_train_data_1275_level__aligned")

    def test_sobra_no_fim_nao_passa(self):
        self._rejeita(_name() + "_extra")

    def test_nivel_zero_rejeita_com_slug_proprio(self):
        """Nível 0 é `source_level_out_of_range`, não "nome feio": o nível é 1-based
        (menor nível medido 1, zero cenas com nível 0, em 840 cenas)."""
        exc = self._rejeita(_name(level=0), "source_level_out_of_range")
        self.assertIn("1-based", str(exc))

    def test_nunca_devolve_none(self):
        """A alternativa a levantar seria devolver `None`, e `scene_id=None` é amostra
        fora do split — vazamento com outro nome."""
        for nome in ("", "lixo", _name(level=0)):
            with self.subTest(nome=nome):
                with self.assertRaises(SampleRejected):
                    parse_file_name_base(nome)


# ==============================================================================
# O par
# ==============================================================================

class TestPar(unittest.TestCase):

    def test_campos_do_protocolo(self):
        p = _par("1275", 3)
        self.assertEqual(p.scene_id, "train_1275")
        self.assertEqual(p.sample_id, "c_lfdof_train_1275_l3")
        self.assertEqual(p.source_dataset, LFDOF_DATASET)
        self.assertEqual(p.source_sample_id, _name("1275", 3))
        self.assertEqual(p.source_split, "train")
        self.assertEqual((p.aif_ref, p.bokeh_ref),
                         (LFDOF_AIF_COLUMN, LFDOF_BOKEH_COLUMN))

    def test_sample_id_tem_prefixo_proprio_da_fonte(self):
        """`c_lfdof_` vs `c_realbokeh_`: as duas fontes escrevem no MESMO release, e id
        colidindo entre fontes é a mesma classe de defeito que dentro de uma fonte."""
        self.assertTrue(_par().sample_id.startswith("c_lfdof_"))

    def test_sample_id_nao_depende_da_anotacao_de_alinhamento(self):
        """Re-anotar um par não pode mudar o `sample_id`: a retomada por
        `completed_ids()` reprocessaria a amostra como se fosse nova."""
        self.assertEqual(_par(suffix="aligned").sample_id,
                         _par(suffix="shift_2.5px").sample_id)
        self.assertNotEqual(_par(suffix="aligned").source_sample_id,
                            _par(suffix="shift_2.5px").source_sample_id)

    def test_nao_referencia_a_coluna_de_saida_de_modelo(self):
        """`image_pre_deblur` é pré-foco da DRB-Net — saída de MODELO, não origem.

        Usá-la como AIF poria uma imagem desborrada por uma rede no lugar da
        all-in-focus real; é o D6 com outro nome.
        """
        refs = {_par().aif_ref, _par().bokeh_ref}
        self.assertNotIn(LFDOF_UNUSED_COLUMN, refs)
        self.assertEqual(LFDOF_UNUSED_COLUMN, "image_pre_deblur")

    def test_scene_level_count_e_obrigatorio(self):
        """Sem default: a contagem é informação de CORPUS, não de linha. Um default
        (`1`, ou `max(level)`) seria número inventado sobre a diversidade do dataset."""
        with self.assertRaises(TypeError):
            pair_from_name(_name())            # type: ignore[call-arg]

    def test_scene_level_count_invalido_e_erro_de_programa(self):
        """`ValueError`, não rejeição: contagem errada é bug de quem contou, não dado
        faltando da origem — e rejeitar a amostra esconderia o bug no histograma."""
        with self.assertRaises(ValueError):
            pair_from_name(_name(), scene_level_count=0)
        with self.assertRaises(ValueError):
            pair_from_name(_name(), scene_level_count=-3)

    def test_par_e_imutavel(self):
        with self.assertRaises(FrozenInstanceError):
            _par().level = 9                    # type: ignore[misc]


class TestCamposOpticosAusentes(unittest.TestCase):
    """O ponto que separa "a grandeza não existe" de "faltou o dado"."""

    def test_os_quatro_sao_none(self):
        p = _par()
        self.assertIsNone(p.f_number)
        self.assertIsNone(p.aif_f_number)
        self.assertIsNone(p.focal_length_mm)
        self.assertIsNone(p.focus_plane_distance_m)

    def test_ninguem_pode_injetar_um_numero(self):
        """`init=False`: aceitar `--f-number 2.8` "só para o validador rodar" é fallback
        numérico com outro nome."""
        for campo in ("f_number", "aif_f_number", "focal_length_mm",
                      "focus_plane_distance_m"):
            with self.subTest(campo=campo):
                with self.assertRaises(TypeError):
                    LFDOFPair(
                        scene_id="train_1275", level=1,
                        sample_id="x", source_dataset=LFDOF_DATASET,
                        source_sample_id=_name(), source_split="train",
                        aif_ref=LFDOF_AIF_COLUMN, bokeh_ref=LFDOF_BOKEH_COLUMN,
                        alignment=ALIGNMENT_ALIGNED,
                        alignment_shift_px_at_source_hw=None,
                        scene_level_count=15,
                        **{campo: 2.8},         # type: ignore[arg-type]
                    )

    def test_declarados_como_campos_do_dataclass(self):
        """Precisam existir como campos, não só como atributo de classe por acidente:
        é assim que o `PairSource` os encontra e que um repack os enxerga."""
        nomes = {f.name for f in fields(LFDOFPair)}
        self.assertLessEqual({"f_number", "aif_f_number", "focal_length_mm",
                              "focus_plane_distance_m"}, nomes)

    def test_o_gate_de_abertura_fica_INAPLICAVEL_e_nao_reprova(self):
        """O defeito C2: com `NaN` sem `applicable`, 100% do LFDOF era rejeitado com o
        slug `gate_aif_aperture_wide` — "abertura larga", que é falso sobre o dado."""
        r = aif_aperture_is_narrow(_par().aif_f_number)
        self.assertFalse(r.applicable)
        self.assertTrue(r.passed)
        self.assertIn("não publica", r.note)

    def test_o_gate_reprova_quando_a_origem_AFIRMA_lixo(self):
        """Contraprova: `applicable=False` não é um passe livre. Origem que afirma um
        f-number inválido continua reprovando — senão o gate não pode reprovar nada."""
        r = aif_aperture_is_narrow(-1.0)
        self.assertTrue(r.applicable)
        self.assertFalse(r.passed)

    def test_o_gate_continua_bloqueando_com_limiar_congelado(self):
        r = aif_aperture_is_narrow(2.8, min_f_number=16.0)
        self.assertFalse(r.passed)

    def test_sem_validador_analitico(self):
        """`_analytic_k` devolve `None` para todo par do LFDOF. É aceitável e declarado:
        o validador audita o rótulo, não o produz."""
        from routes.route_c import _analytic_k
        self.assertIsNone(_analytic_k(_par(), LFDOF_IMAGE_HW))
        self.assertFalse(_par().has_analytic_validator)

    def test_validador_ausente_nao_e_rejeicao(self):
        """Nenhum slug do vocabulário fala de validador analítico ausente — e é isso
        que prova que a ausência não reprova amostra."""
        for slug in ALL_REJECTION_REASONS:
            self.assertNotIn("analytic", slug)


# ==============================================================================
# Enumeração
# ==============================================================================

class TestEnumeracao(unittest.TestCase):

    def test_enumera_uma_cena_inteira(self):
        log = _log()
        pares = enumerate_pairs(_cena("1275", 15), log=log)
        self.assertEqual(len(pares), 15)
        self.assertEqual({p.scene_id for p in pares}, {"train_1275"})
        self.assertEqual(sorted(p.level for p in pares), list(range(1, 16)))
        self.assertEqual(log.accepted, 15)
        self.assertEqual(log.rejected, 0)

    def test_scene_level_count_e_a_contagem_OBSERVADA(self):
        """As 3 cenas com buraco no meio: `max(level)` daria 15, a cena tem 14 pares.
        Um `scene_level_count` derivado do máximo mentiria sobre a diversidade."""
        nomes = [n for n in _cena("2022", 15) if "_level_10_" not in n]
        pares = enumerate_pairs(nomes, log=_log())
        self.assertEqual(len(pares), 14)
        self.assertEqual({p.scene_level_count for p in pares}, {14})
        self.assertEqual(max(p.level for p in pares), 15)

    def test_buraco_no_meio_NAO_e_rejeicao(self):
        """No LFDOF o nível não indexa nada (não há `target_avs`): o par que existe é
        válido e o que falta simplesmente não existe no espelho."""
        nomes = [n for n in _cena("2022", 15) if "_level_10_" not in n]
        log = _log()
        enumerate_pairs(nomes, log=log)
        self.assertEqual(log.rejected, 0)

    def test_buraco_e_CONTADO_e_reportado(self):
        nomes = [n for n in _cena("2022", 15) if "_level_10_" not in n]
        pares = enumerate_pairs(nomes, log=_log())
        self.assertEqual(scenes_with_level_gaps(pares), {"train_2022": [10]})
        self.assertIn("buraco", enumeration_summary(pares, _log()))

    def test_cena_sem_buraco_nao_aparece_no_diagnostico(self):
        self.assertEqual(scenes_with_level_gaps(enumerate_pairs(_cena(), log=_log())),
                         {})

    def test_nome_invalido_entra_no_histograma_e_nao_some(self):
        log = _log()
        pares = enumerate_pairs(_cena("1275", 2) + ["lixo", ""], log=log)
        self.assertEqual(len(pares), 2)
        self.assertEqual(log.reasons["source_name_unparseable"], 2)
        self.assertIn("source_name_unparseable", log.summary())

    def test_nome_invalido_nao_infla_a_contagem_de_nenhuma_cena(self):
        """Uma linha que não se sabe de que cena é não pode inflar a diversidade de
        cena nenhuma — é o motivo de a passagem 1 contar só o que parseou."""
        pares = enumerate_pairs(_cena("1275", 3) + ["lixo"], log=_log())
        self.assertEqual({p.scene_level_count for p in pares}, {3})

    def test_duplicata_dispara_o_slug(self):
        """Mesma (cena, nível) com anotação de alinhamento diferente -> mesmo
        `sample_id`. Sem este gate a segunda vira a primeira no writer."""
        log = _log()
        pares = enumerate_pairs([_name(level=1, suffix="aligned"),
                                 _name(level=1, suffix="misaligned")], log=log)
        self.assertEqual(len(pares), 1)
        self.assertEqual(log.reasons["source_duplicate_sample"], 1)

    def test_duplicata_preserva_a_PRIMEIRA(self):
        pares = enumerate_pairs([_name(level=1, suffix="aligned"),
                                 _name(level=1, suffix="misaligned")], log=_log())
        self.assertEqual(pares[0].alignment, ALIGNMENT_ALIGNED)

    def test_log_e_obrigatorio(self):
        """Um default `None` faria da chamada curta um descarte silencioso."""
        with self.assertRaises(TypeError):
            enumerate_pairs(_cena())            # type: ignore[call-arg]

    def test_ordem_de_entrada_e_preservada(self):
        """`order_pairs_for_sequential_read` depende disso para o shard aberto por vez."""
        nomes = _cena("1275", 5) + _cena("1279", 5)
        pares = enumerate_pairs(nomes, log=_log())
        self.assertEqual([p.source_sample_id for p in pares], nomes)

    def test_lista_vazia_nao_explode(self):
        self.assertEqual(enumerate_pairs([], log=_log()), [])

    def test_alinhamento_sobrevive_a_enumeracao(self):
        """3,71% das linhas são não-`aligned` (medido). Descartar isso ao parsear jogaria
        fora o único sinal de qualidade de par que a origem publica."""
        pares = enumerate_pairs([_name(level=1, suffix="aligned"),
                                 _name(level=2, suffix="misaligned"),
                                 _name(level=3, suffix="shift_2.5px")], log=_log())
        self.assertEqual([p.alignment for p in pares],
                         [ALIGNMENT_ALIGNED, ALIGNMENT_MISALIGNED, ALIGNMENT_SHIFT])
        self.assertEqual([p.is_aligned for p in pares], [True, False, False])


# ==============================================================================
# Split por cena — o que impede o vazamento
# ==============================================================================

class TestSplitPorCena(unittest.TestCase):

    def test_as_15_variantes_de_uma_AIF_caem_TODAS_no_MESMO_lado(self):
        """O teste que o `scene_id` existe para passar.

        O LFDOF entrega 15 desfocadas por AIF em 682 das 840 cenas. Se o split
        separasse duas variantes da mesma cena, a MESMA all-in-focus estaria nos dois
        lados — e `image_focus` é byte a byte idêntico entre as variantes (medido), ou
        seja seria literalmente a mesma imagem em treino e em validação.
        """
        pares = enumerate_pairs(_cena("1275", 15, "train")
                                + _cena("2464", 15, "test"), log=_log())
        split = split_from_source(scene_source_splits(pares))
        for cena in ("train_1275", "test_2464"):
            lados = {split.of(p.scene_id) for p in pares if p.scene_id == cena}
            with self.subTest(cena=cena):
                self.assertEqual(len(lados), 1,
                                 f"cena {cena} ficou nos dois lados: {lados}")

    def test_e_os_dois_lados_existem_de_verdade(self):
        """Contraprova do teste acima: ele passaria trivialmente se tudo caísse num
        lado só. Aqui os dois lados têm cena — senão "sem vazamento" seria vazio."""
        pares = enumerate_pairs(_cena("1275", 15, "train")
                                + _cena("2464", 15, "test"), log=_log())
        contagens = split_from_source(scene_source_splits(pares)).counts()
        self.assertGreater(contagens.get("train", 0), 0)
        self.assertGreater(contagens.get("val", 0), 0)

    def test_o_scene_id_nao_depende_do_nivel_nem_do_alinhamento(self):
        """É o que faz o agrupamento funcionar: só cena e split entram na chave."""
        ids = {parse_full_name(_name("1275", n, "train", s)).scene_id
               for n in (1, 7, 15) for s in ("aligned", "misaligned", "shift_3.0px")}
        self.assertEqual(ids, {"train_1275"})

    def test_cena_nos_dois_splits_e_ValueError(self):
        """Cena nos dois lados é vazamento, e "o último ganha" o esconderia."""
        pares = [_par("1275", 1, "train"), _par("1275", 2, "train")]
        pares[1] = LFDOFPair(
            scene_id="train_1275", level=2, sample_id="c_lfdof_train_1275_l2",
            source_dataset=LFDOF_DATASET, source_sample_id=_name("1275", 2),
            source_split="test",                       # a contradição
            aif_ref=LFDOF_AIF_COLUMN, bokeh_ref=LFDOF_BOKEH_COLUMN,
            alignment=ALIGNMENT_ALIGNED, alignment_shift_px_at_source_hw=None,
            scene_level_count=15)
        with self.assertRaises(ValueError) as ctx:
            scene_source_splits(pares)
        self.assertIn("vazamento", str(ctx.exception))

    def test_split_de_origem_desconhecido_e_ValueError(self):
        par = LFDOFPair(
            scene_id="quixote_1", level=1, sample_id="x",
            source_dataset=LFDOF_DATASET, source_sample_id="y",
            source_split="quixote",
            aif_ref=LFDOF_AIF_COLUMN, bokeh_ref=LFDOF_BOKEH_COLUMN,
            alignment=ALIGNMENT_ALIGNED, alignment_shift_px_at_source_hw=None,
            scene_level_count=1)
        with self.assertRaises(ValueError):
            scene_source_splits([par])

    def test_sem_par_nao_ha_split_a_montar(self):
        with self.assertRaises(ValueError):
            scene_source_splits([])

    def test_herda_os_dois_splits_da_origem(self):
        pares = enumerate_pairs(_cena("1275", 3, "train") + _cena("2464", 3, "test"),
                                log=_log())
        self.assertEqual(scene_source_splits(pares),
                         {"train_1275": "train", "test_2464": "test"})


class TestInvarianteDeNumeroDeCena(unittest.TestCase):
    """A única forma de vazamento que o `scene_id` qualificado NÃO pega — medida."""

    def test_hoje_nenhum_numero_e_compartilhado(self):
        pares = enumerate_pairs(_cena("1275", 3, "train") + _cena("2464", 3, "test"),
                                log=_log())
        self.assertEqual(scene_numbers_shared_between_splits(pares), {})

    def test_e_DETECTA_quando_passa_a_ser(self):
        """Se um dia a origem reusar números entre splits — como a RealBokeh faz —,
        `train_1275` e `test_1275` viram duas cenas em silêncio. Este é o detector, e
        ele tem que poder disparar."""
        pares = enumerate_pairs(_cena("1275", 2, "train") + _cena("1275", 2, "test"),
                                log=_log())
        self.assertEqual(scene_numbers_shared_between_splits(pares),
                         {"1275": {"train", "test"}})

    def test_o_summary_imprime_a_contagem(self):
        pares = enumerate_pairs(_cena("1275", 2, "train") + _cena("1275", 2, "test"),
                                log=_log())
        self.assertIn("números de cena em mais de um split: 1",
                      enumeration_summary(pares, _log()))


# ==============================================================================
# O relatório
# ==============================================================================

class TestSumario(unittest.TestCase):

    def test_conta_cenas_E_amostras(self):
        pares = enumerate_pairs(_cena("1275", 15) + _cena("1279", 15), log=_log())
        texto = enumeration_summary(pares, _log())
        self.assertIn("pares     : 30", texto)
        self.assertIn("cenas     : 2", texto)

    def test_termina_com_o_histograma_de_rejeicao(self):
        log = _log()
        pares = enumerate_pairs(_cena("1275", 2) + ["lixo"], log=log)
        texto = enumeration_summary(pares, log)
        self.assertIn("motivos de rejeição", texto)
        self.assertIn("source_name_unparseable", texto)

    def test_declara_a_ausencia_do_validador_analitico(self):
        texto = enumeration_summary(enumerate_pairs(_cena(), log=_log()), _log())
        self.assertIn("AUSENTE em 100%", texto)

    def test_reporta_a_distribuicao_de_alinhamento(self):
        pares = enumerate_pairs([_name(level=1), _name(level=2, suffix="misaligned")],
                                log=_log())
        texto = enumeration_summary(pares, _log())
        self.assertIn("alinhamento", texto)
        self.assertIn("não-aligned: 1", texto)

    def test_sem_par_nao_explode(self):
        self.assertIsInstance(enumeration_summary([], _log()), str)


# ==============================================================================
# Vocabulário de rejeição
# ==============================================================================

class TestSlugs(unittest.TestCase):

    def test_os_reusados_ja_estao_no_contrato(self):
        self.assertLessEqual(SOURCE_REJECTION_REASONS, REJECTION_REASONS)

    def test_os_novos_estao_no_vocabulario_do_contrato(self):
        """Os dois slugs desta fonte foram registrados em
        `control.contract.SOURCE_REJECTION_REASONS`.

        Este teste era, antes, o lembrete de que ainda faltava registrá-los — falhava
        de propósito. Agora ele guarda o outro lado: um slug novo declarado aqui e
        esquecido lá sumiria do histograma agregado entre rotas, que é justamente o
        instrumento que denuncia fallback novo.
        """
        from control.contract import SOURCE_REJECTION_REASONS as CONTRATO
        self.assertTrue(NEW_REJECTION_REASONS)
        self.assertLessEqual(NEW_REJECTION_REASONS, CONTRATO)
        self.assertLessEqual(CONTRATO, REJECTION_REASONS)
        for slug in NEW_REJECTION_REASONS:
            with self.subTest(slug=slug):
                self.assertTrue(slug.startswith("source_"))

    def test_reject_source_funciona_para_os_dois_conjuntos(self):
        for slug in ALL_REJECTION_REASONS:
            with self.subTest(slug=slug):
                with self.assertRaises(SampleRejected) as ctx:
                    reject_source(slug, "detalhe")
                self.assertEqual(ctx.exception.reason, slug)
                self.assertEqual(ctx.exception.detail, "detalhe")

    def test_slug_inventado_e_KeyError(self):
        """É o que mantém o vocabulário fechado — sem isso, cada caminho novo de falha
        criaria um bucket novo e o histograma pararia de agregar entre runs."""
        with self.assertRaises(KeyError):
            reject_source("source_ficou_estranho", "x")

    def test_o_conjunto_completo_e_a_uniao(self):
        self.assertEqual(ALL_REJECTION_REASONS,
                         SOURCE_REJECTION_REASONS | NEW_REJECTION_REASONS)

    def test_nao_precisa_dos_slugs_de_metadata(self):
        """O LFDOF não tem `metadata/<cena>.json`, então não pode faltar — e um slug
        que nunca dispara é ruído no histograma."""
        for slug in ("source_metadata_missing", "source_metadata_field_invalid",
                     "source_f_number_invalid"):
            self.assertNotIn(slug, ALL_REJECTION_REASONS)


# ==============================================================================
# Carregador de pixels — o `path` que nomeia o papel
# ==============================================================================

class TestPathEsperado(unittest.TestCase):

    def test_monta_o_nome_do_arquivo_original(self):
        self.assertEqual(expected_cell_path(_name("1275", 3), LFDOF_AIF_COLUMN),
                         "LFDOF_train_data_1275_focus_level_3_aligned.png")
        self.assertEqual(expected_cell_path(_name("1275", 3), LFDOF_BOKEH_COLUMN),
                         "LFDOF_train_data_1275_blur_level_3_aligned.png")

    def test_cobre_test_e_alinhamento_anotado(self):
        self.assertEqual(
            expected_cell_path(_name("2464", 12, "test", "shift_3.6px"),
                               LFDOF_AIF_COLUMN),
            "LFDOF_test_data_2464_focus_level_12_shift_3.6px.png")

    def test_coluna_fora_do_schema_nao_tem_papel_a_esperar(self):
        self.assertIsNone(expected_cell_path(_name(), "image_qualquer"))

    def test_nome_fora_do_padrao_nao_tem_path_a_esperar(self):
        self.assertIsNone(expected_cell_path("lixo", LFDOF_AIF_COLUMN))

    def test_a_coluna_nao_usada_tem_papel_declarado(self):
        self.assertEqual(COLUMN_ROLE[LFDOF_UNUSED_COLUMN], "pre-deblur")


class TestDecodificacao(unittest.TestCase):

    def _loader(self, **kw) -> LFDOFImageLoader:
        kw.setdefault("expected_hw", None)
        return LFDOFImageLoader("/nao/usado", _index({}, []), **kw)

    def test_devolve_BGR_e_nao_RGB(self):
        """Vermelho puro tem que sair como `[0, 0, 255]`. Trocar RGB por BGR não quebra
        nada visivelmente e atravessa o pipeline até virar bokeh de cor invertida."""
        out = self._loader()._decode(_png((255, 0, 0)), sample_id="s",
                                     column=LFDOF_AIF_COLUMN)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (4, 6, 3))
        np.testing.assert_array_equal(out[0, 0], [0, 0, 255])

    def test_azul_confirma_o_outro_extremo(self):
        out = self._loader()._decode(_png((0, 0, 255)), sample_id="s",
                                     column=LFDOF_BOKEH_COLUMN)
        np.testing.assert_array_equal(out[0, 0], [255, 0, 0])

    def test_RGBA_opaco_perde_so_o_plano_constante(self):
        """O modo REAL do LFDOF é RGBA com alpha ≡ 255 (medido em 90/90 linhas)."""
        out = self._loader()._decode(_png((255, 0, 0), alpha=255), sample_id="s",
                                     column=LFDOF_AIF_COLUMN)
        self.assertEqual(out.shape, (4, 6, 3))
        np.testing.assert_array_equal(out[0, 0], [0, 0, 255])

    def test_saida_e_contigua(self):
        out = self._loader()._decode(_png((10, 20, 30), alpha=255), sample_id="s",
                                     column=LFDOF_AIF_COLUMN)
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_alpha_nao_opaco_REJEITA_em_vez_de_compor_sobre_preto(self):
        """`convert('RGB')` comporia sobre preto — inventar pixel onde a origem
        declarou transparência, na entrada do Depth Pro, do BiRefNet e do SSIM."""
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._decode(_png((255, 0, 0), alpha=254), sample_id="s",
                                   column=LFDOF_AIF_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_alpha_not_opaque")
        self.assertIn("254", str(ctx.exception))

    def test_alpha_parcialmente_transparente_tambem_rejeita(self):
        arr = np.zeros((4, 6, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        arr[0, 0, 3] = 0                       # um pixel só
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, "PNG")
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._decode(buf.getvalue(), sample_id="s",
                                   column=LFDOF_AIF_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_alpha_not_opaque")

    def test_bytes_corrompidos_rejeitam_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._decode(b"nao sou png", sample_id="s",
                                   column=LFDOF_AIF_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_cinza_vira_tres_canais(self):
        buf = io.BytesIO()
        Image.fromarray(np.full((4, 6), 128, dtype=np.uint8), mode="L").save(buf, "PNG")
        out = self._loader()._decode(buf.getvalue(), sample_id="s",
                                     column=LFDOF_AIF_COLUMN)
        self.assertEqual(out.shape, (4, 6, 3))

    def test_resolucao_diferente_da_declarada_REJEITA(self):
        """K vive em pixel: outra resolução é outro rótulo, sem mudar nada no JSON."""
        loader = self._loader(expected_hw=LFDOF_IMAGE_HW)
        with self.assertRaises(SampleRejected) as ctx:
            loader._decode(_png((1, 2, 3), hw=(4, 6), alpha=255), sample_id="s",
                           column=LFDOF_AIF_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")
        self.assertIn("CONTRATO", str(ctx.exception))

    def test_a_resolucao_declarada_e_a_medida(self):
        self.assertEqual(LFDOF_IMAGE_HW, (688, 1008))

    def test_expected_hw_e_o_default_do_loader(self):
        """Não redimensionar é o comportamento certo; não CONFERIR seria o defeito, e um
        default `None` faria de não conferir o caminho fácil."""
        loader = LFDOFImageLoader("/nao/usado", _index({}, []))
        self.assertEqual(loader._expected_hw, LFDOF_IMAGE_HW)


class TestCelulaInvalida(unittest.TestCase):

    def _loader(self) -> LFDOFImageLoader:
        return LFDOFImageLoader("/nao/usado", _index({}, []), expected_hw=None)

    def test_celula_que_nao_e_dict_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._raw_bytes(b"cru", sample_id="s",
                                      column=LFDOF_AIF_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_sem_bytes_e_sem_path_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._raw_bytes({"bytes": None, "path": None},
                                      sample_id="s", column=LFDOF_BOKEH_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_zero_bytes_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._raw_bytes({"bytes": b"", "path": "x.png"},
                                      sample_id="s", column=LFDOF_AIF_COLUMN)
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_devolve_bytes_e_path(self):
        data, path = self._loader()._raw_bytes(
            {"bytes": b"abc", "path": "p.png"}, sample_id="s",
            column=LFDOF_AIF_COLUMN)
        self.assertEqual((data, path), (b"abc", "p.png"))


class TestChecagemDePapel(unittest.TestCase):
    """A checagem cruzada por linha. É o que a RealBokeh faz contra `target_avs`."""

    def _loader(self) -> LFDOFImageLoader:
        return LFDOFImageLoader("/nao/usado", _index({}, []), expected_hw=None)

    def test_path_correto_passa(self):
        self._loader()._check_role(
            sample_id="s", source_sample_id=_name("1275", 3),
            column=LFDOF_AIF_COLUMN,
            path="LFDOF_train_data_1275_focus_level_3_aligned.png")

    def test_coluna_da_AIF_trazendo_a_desfocada_REJEITA(self):
        """Colunas trocadas produzem um dataset inteiro plausível com AIF e alvo
        invertidos, e o sweep da Eq. 5 encontraria um `K*` sem denunciar nada."""
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._check_role(
                sample_id="s", source_sample_id=_name("1275", 3),
                column=LFDOF_AIF_COLUMN,
                path="LFDOF_train_data_1275_blur_level_3_aligned.png")
        self.assertEqual(ctx.exception.reason, "source_image_role_mismatch")

    def test_pre_deblur_na_coluna_da_AIF_REJEITA(self):
        """O pior caso: saída de MODELO entrando como all-in-focus."""
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._check_role(
                sample_id="s", source_sample_id=_name("1275", 3),
                column=LFDOF_AIF_COLUMN,
                path="LFDOF_train_data_1275_pre-deblur_level_3_aligned.png")
        self.assertEqual(ctx.exception.reason, "source_image_role_mismatch")

    def test_linha_errada_do_mesmo_papel_REJEITA(self):
        """`path` do nível 4 na linha do nível 3: shard remontado, índice velho."""
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._check_role(
                sample_id="s", source_sample_id=_name("1275", 3),
                column=LFDOF_BOKEH_COLUMN,
                path="LFDOF_train_data_1275_blur_level_4_aligned.png")
        self.assertEqual(ctx.exception.reason, "source_image_role_mismatch")

    def test_diretorio_no_path_nao_atrapalha(self):
        self._loader()._check_role(
            sample_id="s", source_sample_id=_name("1275", 3),
            column=LFDOF_AIF_COLUMN,
            path="lfdof/train/LFDOF_train_data_1275_focus_level_3_aligned.png")

    def test_path_ausente_NAO_rejeita(self):
        """A checagem é bônus que a origem oferece, não a fonte primária do papel —
        rejeitar por ausência inviabilizaria qualquer mock."""
        self._loader()._check_role(sample_id="s", source_sample_id=_name(),
                                   column=LFDOF_AIF_COLUMN, path=None)


# ==============================================================================
# O carregador de ponta a ponta, sem pyarrow
# ==============================================================================

class TestCarregamento(unittest.TestCase):

    def _monta(self, **kw):
        nomes = _cena("1275", 3)
        tabela = _FakeTable({
            "file_name_base": nomes,
            LFDOF_AIF_COLUMN: [_cell(n, "focus", rgb=(255, 0, 0)) for n in nomes],
            LFDOF_BOKEH_COLUMN: [_cell(n, "blur", rgb=(0, 255, 0)) for n in nomes],
        })
        idx = _index({n: ("s0.parquet", i) for i, n in enumerate(nomes)},
                     ["s0.parquet"])
        pares = enumerate_pairs(nomes, log=_log())
        kw.setdefault("expected_hw", (4, 6))
        return _LoaderFake({"s0.parquet": tabela}, idx, **kw), pares

    def test_le_o_par_na_ordem_AIF_bokeh(self):
        loader, pares = self._monta()
        aif, bokeh = loader(pares[0])
        np.testing.assert_array_equal(aif[0, 0], [0, 0, 255])      # vermelho -> BGR
        np.testing.assert_array_equal(bokeh[0, 0], [0, 255, 0])    # verde

    def test_le_todos_os_niveis(self):
        loader, pares = self._monta()
        for par in pares:
            with self.subTest(level=par.level):
                self.assertEqual(loader(par)[0].shape, (4, 6, 3))

    def test_o_memo_da_AIF_decodifica_UMA_vez_por_cena(self):
        """A AIF é byte a byte idêntica nos N níveis (medido). Decodificá-la 15 vezes
        por cena é 15× trabalho jogado fora."""
        loader, pares = self._monta()
        for par in pares:
            loader(par)
        self.assertEqual(loader.aif_decodes, 1)
        self.assertEqual(loader.aif_cache_hits, 2)
        self.assertIn("reusos do memo", loader.cache_summary())

    def test_o_memo_NAO_pode_servir_pixel_errado(self):
        """Se a AIF de um nível diferir da do anterior, o sha difere e ele decodifica.
        Um memo por `scene_id` ASSUMIRIA a igualdade em vez de conferi-la."""
        nomes = _cena("1275", 2)
        tabela = _FakeTable({
            "file_name_base": nomes,
            LFDOF_AIF_COLUMN: [_cell(nomes[0], "focus", rgb=(255, 0, 0)),
                               _cell(nomes[1], "focus", rgb=(0, 0, 255))],  # DIFERENTE
            LFDOF_BOKEH_COLUMN: [_cell(n, "blur") for n in nomes],
        })
        idx = _index({n: ("s0.parquet", i) for i, n in enumerate(nomes)},
                     ["s0.parquet"])
        loader = _LoaderFake({"s0.parquet": tabela}, idx, expected_hw=(4, 6))
        pares = enumerate_pairs(nomes, log=_log())
        a = loader(pares[0])[0]
        b = loader(pares[1])[0]
        self.assertEqual(loader.aif_decodes, 2)
        self.assertEqual(loader.aif_cache_hits, 0)
        np.testing.assert_array_equal(a[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(b[0, 0], [255, 0, 0])

    def test_cache_desligado_decodifica_sempre(self):
        loader, pares = self._monta(cache_aif=False)
        for par in pares:
            loader(par)
        self.assertEqual(loader.aif_decodes, 3)
        self.assertEqual(loader.aif_cache_hits, 0)

    def test_indice_apontando_para_a_linha_errada_REJEITA(self):
        """Índice cacheado de um snapshot antigo sobre um shard remontado: TODOS os
        pares sairiam trocados e o histograma ficaria limpo."""
        loader, pares = self._monta()
        loader._index = _index({pares[0].source_sample_id: ("s0.parquet", 2)},
                               ["s0.parquet"])
        with self.assertRaises(SampleRejected) as ctx:
            loader(pares[0])
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")
        self.assertIn("force=True", str(ctx.exception))

    def test_ledger_grava_sha_shard_linha_e_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "source_images.jsonl"
            loader, pares = self._monta(ledger_path=ledger)
            loader(pares[0])
            loader.close()
            linha = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(linha["sample_id"], "c_lfdof_train_1275_l1")
        self.assertEqual(linha["scene_id"], "train_1275")
        self.assertEqual((linha["shard"], linha["row"]), ("s0.parquet", 0))
        self.assertEqual(len(linha["aif_sha256"]), 64)
        self.assertNotEqual(linha["aif_sha256"], linha["bokeh_sha256"])
        self.assertEqual(linha["aif_column"], LFDOF_AIF_COLUMN)
        self.assertIn("focus", linha["aif_path"])
        self.assertFalse(linha["aif_from_cache"])

    def test_ledger_marca_o_reuso_do_memo(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "l.jsonl"
            loader, pares = self._monta(ledger_path=ledger)
            loader(pares[0])
            loader(pares[1])
            loader.close()
            linhas = [json.loads(x) for x in
                      ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([l["aif_from_cache"] for l in linhas], [False, True])
        # o sha vai em TODA linha, não só na primeira: é ele que torna o reuso auditável
        self.assertEqual(linhas[0]["aif_sha256"], linhas[1]["aif_sha256"])

    def test_store_grava_a_AIF_uma_vez_por_cena_e_a_bokeh_por_amostra(self):
        """Gravar a AIF por `sample_id` escreveria a mesma imagem 15 vezes em 682 das
        840 cenas: ~13 GB contra ~1 GB."""
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "source"
            loader, pares = self._monta(store_dir=store)
            for par in pares:
                loader(par)
            nomes = sorted(p.name for p in store.iterdir())
        self.assertEqual(nomes, ["c_lfdof_train_1275_l1_bokeh.png",
                                 "c_lfdof_train_1275_l2_bokeh.png",
                                 "c_lfdof_train_1275_l3_bokeh.png",
                                 "train_1275_aif.png"])

    def test_store_grava_os_bytes_ORIGINAIS(self):
        """Recomprimir falsificaria a evidência: o sha256 do ledger deixaria de bater
        com o arquivo ao lado dele."""
        with tempfile.TemporaryDirectory() as tmp:
            store, ledger = Path(tmp) / "source", Path(tmp) / "l.jsonl"
            loader, pares = self._monta(store_dir=store, ledger_path=ledger)
            loader(pares[0])
            loader.close()
            linha = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
            import hashlib
            no_disco = hashlib.sha256(
                (store / linha["aif_file"]).read_bytes()).hexdigest()
        self.assertEqual(no_disco, linha["aif_sha256"])

    def test_papel_trocado_no_shard_REJEITA_no_caminho_real(self):
        """A checagem de papel tem que disparar dentro de `__call__`, não só isolada."""
        nomes = _cena("1275", 1)
        tabela = _FakeTable({
            "file_name_base": nomes,
            # a coluna da AIF trazendo o `path` da desfocada
            LFDOF_AIF_COLUMN: [_cell(nomes[0], "blur")],
            LFDOF_BOKEH_COLUMN: [_cell(nomes[0], "blur")],
        })
        idx = _index({nomes[0]: ("s0.parquet", 0)}, ["s0.parquet"])
        loader = _LoaderFake({"s0.parquet": tabela}, idx, expected_hw=(4, 6))
        with self.assertRaises(SampleRejected) as ctx:
            loader(enumerate_pairs(nomes, log=_log())[0])
        self.assertEqual(ctx.exception.reason, "source_image_role_mismatch")

    def test_alpha_nao_opaco_REJEITA_no_caminho_real(self):
        nomes = _cena("1275", 1)
        tabela = _FakeTable({
            "file_name_base": nomes,
            LFDOF_AIF_COLUMN: [_cell(nomes[0], "focus", alpha=100)],
            LFDOF_BOKEH_COLUMN: [_cell(nomes[0], "blur")],
        })
        idx = _index({nomes[0]: ("s0.parquet", 0)}, ["s0.parquet"])
        loader = _LoaderFake({"s0.parquet": tabela}, idx, expected_hw=(4, 6))
        with self.assertRaises(SampleRejected) as ctx:
            loader(enumerate_pairs(nomes, log=_log())[0])
        self.assertEqual(ctx.exception.reason, "source_image_alpha_not_opaque")

    def test_a_rejeicao_do_carregador_entra_no_histograma(self):
        """É a integração que importa: o slug tem que chegar ao `RejectionLog` com o
        `sample_id` certo, senão a linha some do JSONL."""
        nomes = _cena("1275", 1)
        tabela = _FakeTable({
            "file_name_base": nomes,
            LFDOF_AIF_COLUMN: [{"bytes": b"nao sou png", "path":
                                expected_cell_path(nomes[0], LFDOF_AIF_COLUMN)}],
            LFDOF_BOKEH_COLUMN: [_cell(nomes[0], "blur")],
        })
        idx = _index({nomes[0]: ("s0.parquet", 0)}, ["s0.parquet"])
        loader = _LoaderFake({"s0.parquet": tabela}, idx, expected_hw=(4, 6))
        par = enumerate_pairs(nomes, log=_log())[0]
        log = _log()
        try:
            loader(par)
        except SampleRejected as exc:
            log.reject_from(par.sample_id, exc, {"scene_id": par.scene_id})
        self.assertEqual(log.reasons["source_image_unreadable"], 1)
        self.assertIn("source_image_unreadable", log.summary())

    def test_le_so_as_colunas_que_usa(self):
        """`image_pre_deblur` fica de fora da leitura do parquet: é saída de modelo, e
        carregá-la gastaria um terço da banda e da RAM para nada."""
        fonte = Path(__file__).resolve().parents[1] / "src/sources/lfdof_images.py"
        texto = fonte.read_text(encoding="utf-8")
        trecho = texto.split("table = pq.read_table(")[1].split(")")[0]
        self.assertIn("LFDOF_AIF_COLUMN", trecho)
        self.assertIn("LFDOF_BOKEH_COLUMN", trecho)
        self.assertNotIn("LFDOF_UNUSED_COLUMN", trecho)


# ==============================================================================
# Reuso do índice e da amostragem de `mirror_images`
# ==============================================================================

class TestReusoDoIndice(unittest.TestCase):
    """Importados, não copiados: quatro cópias de um cálculo foi como este projeto
    chegou a quatro interpretações de K."""

    def test_sao_os_MESMOS_objetos_de_mirror_images(self):
        from sources import mirror_images
        self.assertIs(MirrorIndex, mirror_images.MirrorIndex)
        self.assertIs(order_pairs_for_sequential_read,
                      mirror_images.order_pairs_for_sequential_read)
        self.assertIs(sample_pairs_for_pilot, mirror_images.sample_pairs_for_pilot)

    def test_ordena_por_shard_e_linha(self):
        nomes = _cena("1275", 3)
        idx = _index({nomes[0]: ("s07.parquet", 12), nomes[1]: ("s00.parquet", 40_000),
                      nomes[2]: ("s00.parquet", 3)}, ["s00.parquet", "s07.parquet"])
        pares = enumerate_pairs(nomes, log=_log())
        self.assertEqual(
            [p.level for p in order_pairs_for_sequential_read(pares, idx)], [3, 2, 1])

    def test_piloto_traz_cena_inteira_ou_nenhuma(self):
        nomes = [n for c in range(40) for n in _cena(f"{1000 + c}", 5)]
        pares = enumerate_pairs(nomes, log=_log())
        escolhidos = sample_pairs_for_pilot(pares, limit=50, seed=0)
        por_cena: dict[str, int] = {}
        for p in escolhidos:
            por_cena[p.scene_id] = por_cena.get(p.scene_id, 0) + 1
        for cena, n in por_cena.items():
            with self.subTest(cena=cena):
                self.assertEqual(n, 5, "cena entrou pela metade — não é a unidade certa")
        self.assertGreaterEqual(len(por_cena), 10)


@unittest.skipUnless(TEM_PYARROW, "pyarrow não instalado nesta máquina")
class TestSobreParquetDeVerdade(unittest.TestCase):
    """Roda onde pyarrow existe. Monta um espelho falso de 2 shards e fecha o ciclo."""

    def _espelho(self, root: Path) -> list[str]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        todos: list[str] = []
        for shard, cenas in (("data/train-00000.parquet", ["1275", "1279"]),
                             ("data/test-00000.parquet", ["2464"])):
            split = "train" if "train" in shard else "test"
            nomes, focus, blur, pre = [], [], [], []
            for i, cena in enumerate(cenas):
                # A MESMA AIF em todos os níveis da cena, e DIFERENTE entre cenas —
                # que é o que o espelho tem (medido) e o que faz o memo por sha256
                # decodificar uma vez por cena em vez de uma vez no run inteiro.
                aif = _png((255, 10 * i + int(split == "test"), 0), alpha=255)
                for nivel in (1, 2):
                    nome = _name(cena, nivel, split)
                    nomes.append(nome)
                    focus.append({"bytes": aif,
                                  "path": expected_cell_path(nome, LFDOF_AIF_COLUMN)})
                    blur.append({"bytes": _png((0, 255, 0), alpha=255),
                                 "path": expected_cell_path(nome, LFDOF_BOKEH_COLUMN)})
                    pre.append({"bytes": _png((0, 0, 255)), "path": "irrelevante.png"})
            (root / shard).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({LFDOF_UNUSED_COLUMN: pre,
                                     LFDOF_BOKEH_COLUMN: blur,
                                     LFDOF_AIF_COLUMN: focus,
                                     "file_name_base": nomes}),
                           root / shard)
            todos.extend(nomes)
        return todos

    def test_indice_enumeracao_e_carga_fecham_o_ciclo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._espelho(root)
            idx = MirrorIndex.build(root)
            self.assertEqual(len(idx), 6)
            self.assertTrue((root / INDEX_FILENAME).is_file())
            self.assertEqual(len(MirrorIndex.build(root)), 6)    # segunda vez, do cache

            log = _log()
            pares = enumerate_pairs(idx.names(), log=log)
            self.assertEqual(len(pares), 6)
            self.assertEqual(log.rejected, 0)
            self.assertEqual({p.scene_id for p in pares},
                             {"train_1275", "train_1279", "test_2464"})
            self.assertEqual({p.scene_level_count for p in pares}, {2})

            split = split_from_source(scene_source_splits(pares))
            for cena in ("train_1275", "train_1279", "test_2464"):
                lados = {split.of(p.scene_id) for p in pares if p.scene_id == cena}
                self.assertEqual(len(lados), 1)

            loader = LFDOFImageLoader(root, idx, expected_hw=(4, 6))
            aif_por_cena: dict[str, bytes] = {}
            for par in order_pairs_for_sequential_read(pares, idx):
                aif, bokeh = loader(par)
                self.assertEqual(int(aif[0, 0, 2]), 255)             # R -> canal 2
                self.assertEqual(int(aif[0, 0, 0]), 0)               # B
                np.testing.assert_array_equal(bokeh[0, 0], [0, 255, 0])
                antes = aif_por_cena.setdefault(par.scene_id, aif.tobytes())
                self.assertEqual(antes, aif.tobytes(),
                                 "a AIF mudou entre níveis da MESMA cena")
            # AIF distinta por cena: 3 cenas
            self.assertEqual(len({v for v in aif_por_cena.values()}), 3)
            # 3 cenas, 2 níveis cada: 3 decodificações de AIF, 3 reusos
            self.assertEqual((loader.aif_decodes, loader.aif_cache_hits), (3, 3))
            loader.close()


if __name__ == "__main__":
    unittest.main()
