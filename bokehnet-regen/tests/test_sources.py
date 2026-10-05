"""Testes do adaptador de fonte da RealBokeh.

Todos rodam **sem rede**, com fixtures em memória: o CI não pode depender de um repo
privado no HF nem de um endpoint público estar de pé. O único teste que toca o HF é o
`TestHuggingFaceOpcional`, pulado por default e ligado por
`BOKEHNET_HF_TESTS=1` — ele existe porque fixture em memória prova que o parser
funciona no que eu escrevi, não no que o dataset tem.

O que cada bloco defende, em uma linha:

- o índice de `target_avs` é `level - 1`, e índice fora da lista REJEITA em vez de fazer
  clamp — clamp gravaria o f-number errado sem denunciar;
- nome fora do padrão levanta, nunca devolve `None` — `None` vira `scene_id` ausente,
  que é amostra fora do split, que é vazamento com outro nome;
- par sem metadata da cena entra no histograma com slug, nunca some;
- nenhum campo físico tem default: a ausência rejeita, não vira constante.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import REJECTION_REASONS, SampleRejected          # noqa: E402
from dataio.split import split_from_source                              # noqa: E402
from qc.rejection import RejectionLog                                   # noqa: E402
from sources.realbokeh import (                                         # noqa: E402
    ALIGNMENT_ALIGNED, ALIGNMENT_MISALIGNED, ALIGNMENT_SHIFT,
    MIRROR_AIF_COLUMN, MIRROR_BOKEH_COLUMN, MIRROR_DATASET, MIRROR_IMAGE_HW,
    PENDING_REJECTION_REASONS, RealBokehPair, enumerate_pairs, enumeration_summary,
    f_number_for_level, level_count, load_scene_metadata, load_split_metadata,
    pair_from_name, parse_file_name_base, parse_full_name, scene_key,
    scene_source_splits,
)


# ==============================================================================
# Fixtures — cópias fiéis de dois JSONs reais do `timseizinger/RealBokeh_3MP`
# ==============================================================================

#: `train/metadata/1000.json`, verbatim (lido do repo público em 2026-09-10).
META_1000 = {
    "id": 1000,
    "source_image": "in/1000_f22.JPG",
    "source_av": 22.0,
    "target_images": [
        "gt/1000/1000_f2.0.JPG", "gt/1000/1000_f3.2.JPG", "gt/1000/1000_f4.0.JPG",
        "gt/1000/1000_f16.JPG", "gt/1000/1000_f18.JPG",
    ],
    "target_avs": [2.0, 3.2, 4.0, 16.0, 18.0],
    "focal_length": 60,
    "ISO": 100,
    "EV": 11.375,
    "focus_plane_distance": 1.92,
    "focus_plane_uncertainty": 0.12,
}

#: `train/metadata/1038.json`, verbatim. É a cena cujo `image_focus` foi conferido
#: byte a byte contra `train/in/1038_f22.JPG` (`reference/ACHADOS.md`).
META_1038 = {
    "id": 1038,
    "source_image": "in/1038_f22.JPG",
    "source_av": 22.0,
    "target_images": [
        "gt/1038/1038_f2.0.JPG", "gt/1038/1038_f2.2.JPG", "gt/1038/1038_f5.0.JPG",
        "gt/1038/1038_f7.1.JPG", "gt/1038/1038_f18.JPG",
    ],
    "target_avs": [2.0, 2.2, 5.0, 7.1, 18.0],
    "focal_length": 70,
    "ISO": 200,
    "EV": 8.625,
    "focus_plane_distance": 1.895,
    "focus_plane_uncertainty": 0.11499999999999988,
}

#: Chaveado por `scene_key(split, numero)` — a numeração de cena REINICIA em cada
#: split no espelho (train_1, test_1 e validation_1 são cenas físicas diferentes), e
#: chavear pelo número cru fundiria as três.
META = {"train_1000": META_1000, "train_1038": META_1038}


def _name(scene="1000", level=1, split="train", suffix="aligned"):
    return f"timseizinger_realbokeh_3mp_{split}_f_{scene}_level_{level}_{suffix}"


def _meta(**overrides):
    out = dict(META_1000)
    for key, value in overrides.items():
        if value is _ABSENT:
            out.pop(key, None)
        else:
            out[key] = value
    return out


class _Absent:
    pass


_ABSENT = _Absent()


def _log():
    return RejectionLog()          # sem `path`: histograma em memória, zero I/O


# ==============================================================================
# O nome do espelho
# ==============================================================================

class TestParseFileNameBase(unittest.TestCase):

    def test_extrai_cena_e_nivel(self):
        self.assertEqual(parse_file_name_base(_name("1038", 3)), ("train_1038", 3))

    def test_cena_e_nivel_de_varios_digitos(self):
        self.assertEqual(parse_file_name_base(_name("42", 21)), ("train_42", 21))

    def test_mesmo_numero_em_splits_diferentes_sao_cenas_DIFERENTES(self):
        """Medido no espelho: a numeração de cena REINICIA em cada split. `train` tem
        3.959 cenas numeradas de 1 em diante, e `test` e `validation` têm 220 cada,
        também começando em 1 — os mesmos números, cenas físicas outras.

        Um `scene_id` cru colidiria em dois lugares ao mesmo tempo: o split ficaria
        furado (a cena 1 estaria nos três lados) e o `sample_id` se repetiria, fazendo
        o gate de duplicata descartar 2.495 amostras boas em silêncio.
        """
        ids = {parse_file_name_base(_name("1", 1, split=s))[0]
               for s in ("train", "test", "validation")}
        self.assertEqual(ids, {"train_1", "test_1", "validation_1"})

    def test_scene_number_preserva_o_numero_cru(self):
        """É por ele que se acha `<split>/metadata/<numero>.json`."""
        parsed = parse_full_name(_name("1038", 3, split="test"))
        self.assertEqual(parsed.scene_number, "1038")
        self.assertEqual(parsed.scene_id, "test_1038")

    def test_split_e_alinhamento_vem_do_nome(self):
        parsed = parse_full_name(_name("1000", 5))
        self.assertEqual(parsed.source_split, "train")
        self.assertEqual(parsed.alignment, ALIGNMENT_ALIGNED)
        self.assertIsNone(parsed.alignment_shift_px_at_mirror_hw)

    def test_sufixo_misaligned_e_reconhecido(self):
        """183 das 20.495 linhas do espelho dizem `misaligned`. Rejeitá-las no parser
        jogaria fora o único sinal de qualidade de par que a origem publica."""
        parsed = parse_full_name(_name("1029", 1, suffix="misaligned"))
        self.assertEqual(parsed.alignment, ALIGNMENT_MISALIGNED)

    def test_misaligned_nao_vira_deslocamento_zero(self):
        """A origem diz que não fechou e NÃO diz quanto. Gravar 0,0 px aqui seria
        afirmar alinhamento perfeito — fallback numérico com sinal trocado."""
        parsed = parse_full_name(_name("1029", 1, suffix="misaligned"))
        self.assertIsNone(parsed.alignment_shift_px_at_mirror_hw)

    def test_sufixo_shift_traz_o_deslocamento_em_pixels(self):
        parsed = parse_full_name(_name("101", 2, suffix="shift_2.1px"))
        self.assertEqual(parsed.alignment, ALIGNMENT_SHIFT)
        self.assertAlmostEqual(parsed.alignment_shift_px_at_mirror_hw, 2.1)

    def test_deslocamento_carrega_a_resolucao_no_nome_do_campo(self):
        """Regra 3 do contrato: toda quantidade em pixel carrega a resolução em que foi
        medida. O campo é `..._at_mirror_hw`, e `MIRROR_IMAGE_HW` diz qual é."""
        campos = {f.name for f in fields(RealBokehPair)}
        self.assertIn("alignment_shift_px_at_mirror_hw", campos)
        self.assertNotIn("alignment_shift_px", campos)
        self.assertEqual(MIRROR_IMAGE_HW, (1500, 2000))

    # -- rejeições -------------------------------------------------------------

    def _rejeita(self, name, reason="source_name_unparseable"):
        with self.assertRaises(SampleRejected) as ctx:
            parse_file_name_base(name)
        self.assertEqual(ctx.exception.reason, reason)

    def test_rejeita_prefixo_errado(self):
        self._rejeita("outro_dataset_train_f_1000_level_1_aligned")

    def test_rejeita_sem_nivel(self):
        self._rejeita("timseizinger_realbokeh_3mp_train_f_1000_aligned")

    def test_rejeita_sufixo_desconhecido(self):
        self._rejeita(_name("1000", 1, suffix="whatever"))

    def test_rejeita_split_desconhecido(self):
        self._rejeita(_name("1000", 1, split="trainval"))

    def test_rejeita_string_vazia(self):
        self._rejeita("")

    def test_rejeita_none(self):
        self._rejeita(None)

    def test_rejeita_nivel_zero(self):
        """O nível do espelho é 1-based: 3.959/3.959 cenas com níveis contíguos 1..n e
        ZERO cenas com nível 0. Um `level_0` é nome corrompido, não nível válido."""
        self._rejeita(_name("1000", 0), reason="source_level_out_of_range")

    def test_nao_devolve_none_em_nenhum_caso(self):
        """O ponto do requisito: falhar alto, nunca devolver `None` em silêncio. Um
        `None` viraria `scene_id` ausente no manifesto, e amostra sem cena é amostra
        fora do split — vazamento com outro nome."""
        for ruim in ["", "x", None, 17, _name("1000", 1, suffix="misalign")]:
            with self.subTest(nome=ruim):
                with self.assertRaises(SampleRejected):
                    parse_file_name_base(ruim)


# ==============================================================================
# O join com `target_avs` — 1-based, e sem clamp
# ==============================================================================

class TestFNumberForLevel(unittest.TestCase):

    def test_nivel_1_e_o_primeiro_elemento(self):
        self.assertEqual(f_number_for_level(META_1000, 1), 2.0)

    def test_nivel_n_e_o_ultimo_elemento(self):
        self.assertEqual(f_number_for_level(META_1000, 5), 18.0)

    def test_indice_e_level_menos_um_em_toda_a_lista(self):
        for level, esperado in enumerate(META_1038["target_avs"], start=1):
            with self.subTest(level=level):
                self.assertEqual(f_number_for_level(META_1038, level), esperado)

    def test_um_indice_a_mais_rejeita_em_vez_de_fazer_clamp(self):
        """O teste que separa 1-based correto de clamp: com clamp, o nível 6 numa cena
        de 5 níveis devolveria 18,0 (o último) e a amostra iria para o disco com a
        abertura errada, sem nada denunciar."""
        with self.assertRaises(SampleRejected) as ctx:
            f_number_for_level(META_1000, 6)
        self.assertEqual(ctx.exception.reason, "source_level_out_of_range")
        self.assertIn("SEM clamp", str(ctx.exception))

    def test_nivel_zero_rejeita(self):
        """Se o índice fosse 0-based, `level 0` seria válido e devolveria 2,0. Ele
        rejeita — é a afirmação do 1-based codificada."""
        with self.assertRaises(SampleRejected) as ctx:
            f_number_for_level(META_1000, 0)
        self.assertEqual(ctx.exception.reason, "source_level_out_of_range")

    def test_nivel_negativo_rejeita(self):
        with self.assertRaises(SampleRejected):
            f_number_for_level(META_1000, -1)

    def test_confere_ordem_contra_o_nome_do_arquivo(self):
        """`target_avs[i]` tem que bater com o f-number do nome de `target_images[i]`.
        É a checagem que valida a ORDEM da lista — o join por nível assume que ela é a
        mesma dos arquivos."""
        embaralhado = _meta(target_avs=[3.2, 2.0, 4.0, 16.0, 18.0])
        with self.assertRaises(SampleRejected) as ctx:
            f_number_for_level(embaralhado, 1)
        self.assertEqual(ctx.exception.reason, "source_f_number_invalid")

    def test_sem_target_images_o_f_number_ainda_sai_de_target_avs(self):
        """`target_avs` é a fonte primária; a checagem cruzada é bônus."""
        self.assertEqual(f_number_for_level(_meta(target_images=_ABSENT), 2), 3.2)

    def test_target_avs_ausente_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            f_number_for_level(_meta(target_avs=_ABSENT), 1)
        self.assertEqual(ctx.exception.reason, "source_metadata_field_invalid")

    def test_target_avs_vazio_rejeita(self):
        with self.assertRaises(SampleRejected):
            f_number_for_level(_meta(target_avs=[], target_images=[]), 1)

    def test_f_number_nao_numerico_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            f_number_for_level(_meta(target_avs=["2.0", 3.2, 4.0, 16.0, 18.0],
                                     target_images=_ABSENT), 1)
        self.assertEqual(ctx.exception.reason, "source_f_number_invalid")

    def test_f_number_nao_positivo_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            f_number_for_level(_meta(target_avs=[0.0, 3.2, 4.0, 16.0, 18.0],
                                     target_images=_ABSENT), 1)
        self.assertEqual(ctx.exception.reason, "source_f_number_invalid")

    def test_f_number_nan_rejeita(self):
        with self.assertRaises(SampleRejected):
            f_number_for_level(_meta(target_avs=[float("nan")] * 5,
                                     target_images=_ABSENT), 1)

    def test_level_count(self):
        self.assertEqual(level_count(META_1000), 5)


# ==============================================================================
# Leitura dos JSONs de cena
# ==============================================================================

class TestLoadSceneMetadata(unittest.TestCase):

    def _dir(self, conteudo: dict):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        for nome, meta in conteudo.items():
            (base / nome).write_text(json.dumps(meta), encoding="utf-8")
        return base

    def test_le_e_indexa_pelo_stem(self):
        base = self._dir({"1000.json": META_1000, "1038.json": META_1038})
        carregado = load_split_metadata(base, "train")
        self.assertEqual(sorted(carregado), ["train_1000", "train_1038"])
        self.assertEqual(carregado["train_1038"]["focal_length"], 70)

    def test_ignora_arquivo_que_nao_e_json(self):
        base = self._dir({"1000.json": META_1000})
        (base / "README.md").write_text("nada", encoding="utf-8")
        self.assertEqual(sorted(load_split_metadata(base, "train")), ["train_1000"])

    def test_split_invalido_e_erro(self):
        base = self._dir({"1000.json": META_1000})
        with self.assertRaises(ValueError):
            load_split_metadata(base, "treino")

    def test_raiz_le_os_tres_splits_sem_fundir_cenas(self):
        """O mesmo número de cena em dois splits tem que virar duas entradas.

        Fundir daria à cena de `test` a distância de foco da cena de `train` — sem
        nada denunciar, porque o JSON tem todos os campos e é perfeitamente válido.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raiz = Path(tmp.name)
        for split, meta in (("train", META_1000), ("test", META_1038)):
            d = raiz / split / "metadata"
            d.mkdir(parents=True)
            (d / "1.json").write_text(json.dumps({**meta, "id": 1}), encoding="utf-8")

        carregado = load_scene_metadata(raiz)
        self.assertEqual(sorted(carregado), ["test_1", "train_1"])
        self.assertEqual(carregado["train_1"]["focal_length"], 60)
        self.assertEqual(carregado["test_1"]["focal_length"], 70)

    def test_raiz_sem_layout_de_split_e_erro_que_ensina(self):
        base = self._dir({"1000.json": META_1000})
        with self.assertRaises(ValueError) as ctx:
            load_scene_metadata(base)
        self.assertIn("train/metadata", str(ctx.exception))

    def test_id_divergente_do_nome_e_erro_duro(self):
        """Não é rejeição de amostra: rejeição é para dado faltando. Arquivo cujo nome
        contradiz o conteúdo é fonte corrompida, e o join da rota C é por esse nome."""
        base = self._dir({"1000.json": {**META_1000, "id": 999}})
        with self.assertRaises(ValueError) as ctx:
            load_split_metadata(base, "train")
        self.assertIn("contradiz", str(ctx.exception))

    def test_diretorio_vazio_e_erro(self):
        """Devolver `{}` faria as 20.495 linhas caírem em `source_metadata_missing`, e
        o histograma diria 'o repo bruto perdeu todas as cenas' quando o que houve foi
        caminho errado."""
        with self.assertRaises(ValueError):
            load_scene_metadata(self._dir({}))

    def test_caminho_inexistente_e_erro(self):
        with self.assertRaises(ValueError):
            load_scene_metadata("/nao/existe/em/lugar/nenhum")


# ==============================================================================
# O par
# ==============================================================================

class TestPairFromName(unittest.TestCase):

    def test_par_completo(self):
        par = pair_from_name(_name("1038", 4), META)
        self.assertEqual(par.scene_id, "train_1038")
        self.assertEqual(par.level, 4)
        self.assertEqual(par.sample_id, "c_realbokeh_train_1038_l4")
        self.assertEqual(par.source_sample_id, _name("1038", 4))
        self.assertEqual(par.source_dataset, MIRROR_DATASET)
        self.assertEqual(par.source_split, "train")
        self.assertEqual(par.f_number, 7.1)
        self.assertEqual(par.aif_f_number, 22.0)
        self.assertEqual(par.focal_length_mm, 70.0)
        self.assertEqual(par.focus_plane_distance_m, 1.895)
        self.assertAlmostEqual(par.focus_plane_uncertainty_m, 0.115)
        self.assertEqual(par.aif_ref, MIRROR_AIF_COLUMN)
        self.assertEqual(par.bokeh_ref, MIRROR_BOKEH_COLUMN)
        self.assertEqual(par.scene_level_count, 5)
        self.assertTrue(par.is_aligned)

    def test_caminhos_brutos(self):
        """`raw_aif_path` é o arquivo cujo sha256 já bateu com o `image_focus` do
        espelho (cena 1038, `59d8e910ca69…`). É o que torna a AIF auditável."""
        par = pair_from_name(_name("1038", 4), META)
        self.assertEqual(par.raw_aif_path, "train/in/1038_f22.JPG")
        self.assertEqual(par.raw_bokeh_path, "train/gt/1038/1038_f7.1.JPG")

    def test_aif_f_number_vem_do_metadata_nao_da_constante_22(self):
        """f/22 é o valor real em 3.960 de 3.960 cenas, e é exatamente por isso que
        cravá-lo passaria despercebido. Ele é lido."""
        par = pair_from_name(_name("1000", 1), {"train_1000": _meta(source_av=16.0)})
        self.assertEqual(par.aif_f_number, 16.0)

    def test_sem_metadata_da_cena_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            pair_from_name(_name("9999", 1), META)
        self.assertEqual(ctx.exception.reason, "source_metadata_missing")
        self.assertIn("9999", str(ctx.exception))

    def test_campos_da_eq3_ausentes_rejeitam(self):
        for campo in ("focal_length", "focus_plane_distance", "focus_plane_uncertainty",
                      "source_av", "source_image"):
            with self.subTest(campo=campo):
                with self.assertRaises(SampleRejected) as ctx:
                    pair_from_name(_name("1000", 1), {"train_1000": _meta(**{campo: _ABSENT})})
                self.assertEqual(ctx.exception.reason, "source_metadata_field_invalid")

    def test_incerteza_zero_e_afirmacao_legitima(self):
        """Incerteza 0 é "medi e não sobrou dúvida"; distância focal 0 é impossível.
        Por isso os dois campos têm validadores diferentes."""
        par = pair_from_name(_name("1000", 1), {"train_1000": _meta(focus_plane_uncertainty=0.0)})
        self.assertEqual(par.focus_plane_uncertainty_m, 0.0)

    def test_distancia_de_foco_zero_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            pair_from_name(_name("1000", 1), {"train_1000": _meta(focus_plane_distance=0.0)})
        self.assertEqual(ctx.exception.reason, "source_metadata_field_invalid")

    def test_distancia_de_foco_negativa_rejeita(self):
        with self.assertRaises(SampleRejected):
            pair_from_name(_name("1000", 1), {"train_1000": _meta(focus_plane_distance=-1.0)})

    def test_focal_booleana_rejeita(self):
        """`True` é `int` em Python e passaria por um teste de tipo ingênuo, virando
        focal de 1 mm."""
        with self.assertRaises(SampleRejected):
            pair_from_name(_name("1000", 1), {"train_1000": _meta(focal_length=True)})

    def test_nivel_alem_do_que_a_cena_tem_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            pair_from_name(_name("1000", 9), META)
        self.assertEqual(ctx.exception.reason, "source_level_out_of_range")

    def test_nenhum_campo_fisico_tem_default(self):
        """Regra do projeto: nada de default de assinatura carregando grandeza física.
        Um `f_number: float = 22.0` transformaria a ausência de metadata em constante."""
        for campo in fields(RealBokehPair):
            with self.subTest(campo=campo.name):
                import dataclasses
                self.assertIs(campo.default, dataclasses.MISSING)
                self.assertIs(campo.default_factory, dataclasses.MISSING)

    def test_par_e_imutavel(self):
        par = pair_from_name(_name("1000", 1), META)
        with self.assertRaises(Exception):
            par.f_number = 1.4          # frozen dataclass


class TestConformidadeComPairSource(unittest.TestCase):
    """`RealBokehPair` tem que satisfazer o `PairSource` de `routes/route_c.py`.

    A lista está aqui em vez de importada de `routes.route_c` de propósito: importar a
    rota puxaria BokehMe, gates e writer para dentro de um teste de adaptador de fonte,
    e um teste que quebra por causa de dependência de terceiro não diz mais nada sobre
    o adaptador. Se a rota mudar o protocolo, este teste é o que tem que ser atualizado
    junto — e falha alto.
    """

    ATRIBUTOS = (
        "scene_id", "sample_id", "source_dataset", "source_sample_id", "source_split",
        "aif_ref", "bokeh_ref", "f_number", "focal_length_mm",
        "focus_plane_distance_m", "aif_f_number",
    )

    def test_tem_todos_os_atributos_do_protocolo(self):
        par = pair_from_name(_name("1038", 2), META)
        for atributo in self.ATRIBUTOS:
            with self.subTest(atributo=atributo):
                self.assertTrue(hasattr(par, atributo))
                self.assertIsNotNone(getattr(par, atributo))


# ==============================================================================
# Enumeração — nada some em silêncio
# ==============================================================================

class TestEnumeratePairs(unittest.TestCase):

    def test_enumera_o_que_existe(self):
        nomes = [_name("1000", n) for n in range(1, 6)] + [_name("1038", n) for n in range(1, 6)]
        log = _log()
        pares = enumerate_pairs(nomes, META, log=log)
        self.assertEqual(len(pares), 10)
        self.assertEqual(log.accepted, 10)
        self.assertEqual(log.rejected, 0)
        self.assertEqual({p.scene_id for p in pares}, {"train_1000", "train_1038"})
        self.assertEqual([p.f_number for p in pares[:5]], [2.0, 3.2, 4.0, 16.0, 18.0])

    def test_cada_rejeicao_tem_slug_e_entra_no_histograma(self):
        nomes = [
            _name("1000", 1),                    # ok
            _name("1000", 6),                    # nível fora de target_avs
            _name("9999", 1),                    # cena sem metadata
            "lixo_que_nao_parseia",              # nome fora do padrão
            _name("1038", 2),                    # ok
        ]
        log = _log()
        pares = enumerate_pairs(nomes, META, log=log)
        self.assertEqual(len(pares), 2)
        self.assertEqual(log.accepted, 2)
        self.assertEqual(log.rejected, 3)
        self.assertEqual(dict(log.reasons), {
            "source_level_out_of_range": 1,
            "source_metadata_missing": 1,
            "source_name_unparseable": 1,
        })

    def test_nenhuma_linha_desaparece(self):
        """Invariante do módulo: aceitas + rejeitadas == linhas de entrada."""
        nomes = [_name("1000", n) for n in range(1, 9)] + ["ruim", _name("77", 1)]
        log = _log()
        pares = enumerate_pairs(nomes, META, log=log)
        self.assertEqual(log.total, len(nomes))
        self.assertEqual(log.accepted, len(pares))

    def test_duplicata_de_cena_e_nivel_rejeita(self):
        """Duas linhas com o mesmo (cena, nível) virariam dois `sample_id` iguais, e a
        retomada por `completed_ids()` trataria a segunda como a primeira."""
        nomes = [_name("1000", 1), _name("1000", 1, suffix="misaligned")]
        log = _log()
        pares = enumerate_pairs(nomes, META, log=log)
        self.assertEqual(len(pares), 1)
        self.assertEqual(dict(log.reasons), {"source_duplicate_sample": 1})

    def test_sample_id_nao_muda_com_a_anotacao_de_alinhamento(self):
        """Se o `sample_id` embutisse o sufixo, re-anotar um par o faria parecer novo e
        a retomada reprocessaria a amostra."""
        a = pair_from_name(_name("1000", 1), META)
        b = pair_from_name(_name("1000", 1, suffix="shift_3.0px"), META)
        self.assertEqual(a.sample_id, b.sample_id)
        self.assertNotEqual(a.source_sample_id, b.source_sample_id)
        self.assertNotEqual(a.alignment, b.alignment)

    def test_cena_incompleta_nao_e_rejeicao(self):
        """Medido: 11 cenas do espelho têm MENOS linhas do que níveis declarados — 59
        imagens ausentes, sempre na cauda (níveis mais fechados). O par que existe é
        válido; o que falta simplesmente não existe. `scene_level_count` guarda o que a
        cena DECLARA, e é ele que denuncia a diferença.
        """
        log = _log()
        pares = enumerate_pairs([_name("1000", 1), _name("1000", 2)], META, log=log)
        self.assertEqual(len(pares), 2)
        self.assertEqual(log.rejected, 0)
        self.assertEqual({p.scene_level_count for p in pares}, {5})

    def test_entrada_vazia_devolve_lista_vazia(self):
        log = _log()
        self.assertEqual(enumerate_pairs([], META, log=log), [])
        self.assertEqual(log.total, 0)

    def test_log_e_obrigatorio(self):
        """Sem default: um `log=None` transformaria a chamada curta — a que todo mundo
        escreve — em descarte silencioso."""
        with self.assertRaises(TypeError):
            enumerate_pairs([_name("1000", 1)], META)      # type: ignore[call-arg]

    def test_resumo_traz_cenas_e_amostras(self):
        nomes = [_name("1000", n) for n in range(1, 6)] + [_name("1038", n) for n in range(1, 3)]
        log = _log()
        texto = enumeration_summary(enumerate_pairs(nomes, META, log=log), log)
        self.assertIn("pares     : 7", texto)
        self.assertIn("cenas     : 2", texto)
        self.assertIn("2 -> 1", texto)     # uma cena com 2 níveis
        self.assertIn("5 -> 1", texto)     # uma cena com 5 níveis


# ==============================================================================
# Split
# ==============================================================================

class TestSceneSourceSplits(unittest.TestCase):

    def test_mapa_cena_para_split(self):
        log = _log()
        pares = enumerate_pairs([_name("1000", 1), _name("1000", 2), _name("1038", 1)],
                                META, log=log)
        self.assertEqual(scene_source_splits(pares),
                         {"train_1000": "train", "train_1038": "train"})

    def test_alimenta_split_from_source(self):
        log = _log()
        meta = {**META, "validation_1038": META_1038}
        pares = enumerate_pairs(
            [_name("1000", 1), _name("1038", 1, split="validation")], meta, log=log)
        split = split_from_source(scene_source_splits(pares))
        self.assertEqual(split.of("train_1000"), "train")
        self.assertEqual(split.of("validation_1038"), "val")
        self.assertEqual(split.counts(), {"train": 1, "val": 1})

    def test_mesmo_numero_em_dois_splits_vira_duas_cenas(self):
        """O `scene_id` qualificado torna o vazamento estruturalmente impossível.

        Antes, os dois pares viravam a cena `1000` nos dois lados e `scene_source_splits`
        levantava. Agora eles são cenas distintas, cada uma num lado — que é a verdade
        física: são fotos diferentes.
        """
        log = _log()
        meta = {**META, "test_1000": META_1000}
        pares = enumerate_pairs([_name("1000", 1), _name("1000", 2, split="test")],
                                meta, log=log)
        self.assertEqual(scene_source_splits(pares),
                         {"train_1000": "train", "test_1000": "test"})
        self.assertEqual(len({p.sample_id for p in pares}), 2)

    def test_guarda_contra_cena_em_dois_splits_continua_viva(self):
        """A checagem virou defesa em profundidade — mantida porque um adaptador futuro
        pode montar `scene_id` de outro jeito."""
        class _Falso:
            scene_id = "x"
            def __init__(self, split): self.source_split = split
        with self.assertRaises(ValueError) as ctx:
            scene_source_splits([_Falso("train"), _Falso("test")])
        self.assertIn("vazamento", str(ctx.exception))

    def test_sem_pares_e_erro(self):
        with self.assertRaises(ValueError):
            scene_source_splits([])


# ==============================================================================
# Vocabulário de rejeição
# ==============================================================================

class TestVocabularioDeRejeicao(unittest.TestCase):

    def test_slugs_pendentes_estao_declarados(self):
        """Enquanto `contract.REJECTION_REASONS` não registrar estes slugs, eles vivem
        em `PENDING_REJECTION_REASONS` — declarados, não inventados no meio do código.
        Quando forem registrados, este teste continua passando e o conjunto volta a ser
        fechado num lugar só."""
        self.assertTrue(PENDING_REJECTION_REASONS)
        for slug in PENDING_REJECTION_REASONS:
            with self.subTest(slug=slug):
                self.assertTrue(slug.startswith("source_"))

    def test_todo_slug_emitido_e_conhecido(self):
        conhecidos = REJECTION_REASONS | PENDING_REJECTION_REASONS
        entradas = [
            "lixo", _name("1000", 0), _name("9999", 1), _name("1000", 6),
            _name("1000", 1), _name("1000", 1),
        ]
        log = _log()
        enumerate_pairs(entradas, {**META, "train_1000": _meta()}, log=log)
        self.assertTrue(log.reasons)
        for slug in log.reasons:
            with self.subTest(slug=slug):
                self.assertIn(slug, conhecidos)

    def test_slugs_da_fonte_estao_no_vocabulario_do_contrato(self):
        """Os seis slugs `source_*` agora vivem em `contract.SOURCE_REJECTION_REASONS`.

        Se alguém acrescentar um slug aqui e esquecer de registrar lá, o histograma
        agregado entre rotas perde a categoria em silêncio — que é exatamente o modo
        de falha que o vocabulário fechado existe para impedir.
        """
        from control.contract import SOURCE_REJECTION_REASONS
        self.assertLessEqual(PENDING_REJECTION_REASONS, SOURCE_REJECTION_REASONS)
        self.assertLessEqual(SOURCE_REJECTION_REASONS, REJECTION_REASONS)

    def test_slug_desconhecido_nao_passa(self):
        from sources.realbokeh import _reject
        with self.assertRaises(KeyError):
            _reject("motivo_inventado_agora", "detalhe")


# ==============================================================================
# Opcional — toca o HF de verdade
# ==============================================================================

@unittest.skipUnless(os.environ.get("BOKEHNET_HF_TESTS") == "1",
                     "toca a rede; ligue com BOKEHNET_HF_TESTS=1")
class TestHuggingFaceOpcional(unittest.TestCase):
    """Confere o adaptador contra o dataset REAL.

    Só o `timseizinger/RealBokeh_3MP` é tocado: ele é público, e é dele que vem o
    `metadata/` do join. O espelho `akcit-pixel/RealBokeh` é privado e precisa do token
    em `~/.cache/huggingface/token`; o teste que depende dele é pulado quando o token
    não existe, em vez de falhar — CI sem credencial não é defeito de código.
    """

    RAW = "https://huggingface.co/datasets/timseizinger/RealBokeh_3MP/resolve/main/"

    def _json(self, path):
        import urllib.request
        with urllib.request.urlopen(self.RAW + path, timeout=90) as resposta:
            return json.loads(resposta.read())

    def test_fixture_bate_com_o_json_real(self):
        """A fixture em memória é cópia verbatim. Se o repo mudar, este teste avisa —
        e é o único jeito de a fixture não virar ficção com o tempo."""
        self.assertEqual(self._json("train/metadata/1038.json"), META_1038)

    def test_join_real_de_uma_cena(self):
        meta = self._json("train/metadata/1000.json")
        par = pair_from_name(_name("1000", 3), {"train_1000": meta})
        self.assertEqual(par.f_number, 4.0)
        self.assertEqual(par.raw_bokeh_path, "train/gt/1000/1000_f4.0.JPG")

    def test_espelho_privado_quando_ha_token(self):
        """Lê só a coluna `file_name_base` por HTTP Range — ~68 KB de um shard de
        470 MB — e confere que os nomes casam com o parser e que os níveis são 1-based.
        """
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if not token_path.exists():
            self.skipTest("sem token do HF; o espelho é privado")
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow ausente")

        import io
        import urllib.request
        token = token_path.read_text().strip()

        class _HttpFile(io.RawIOBase):
            def __init__(self, url):
                self.url, self.pos = url, 0
                pedido = urllib.request.Request(
                    url, method="HEAD", headers={"Authorization": f"Bearer {token}"})
                with urllib.request.urlopen(pedido, timeout=60) as resposta:
                    self.size = int(resposta.headers["Content-Length"])

            def readable(self):
                return True

            def seekable(self):
                return True

            def tell(self):
                return self.pos

            def seek(self, deslocamento, de_onde=0):
                self.pos = (deslocamento if de_onde == 0
                            else self.pos + deslocamento if de_onde == 1
                            else self.size + deslocamento)
                return self.pos

            def read(self, n=-1):
                if n is None or n < 0:
                    n = self.size - self.pos
                if n == 0:
                    return b""
                fim = min(self.pos + n, self.size) - 1
                pedido = urllib.request.Request(self.url, headers={
                    "Authorization": f"Bearer {token}", "Range": f"bytes={self.pos}-{fim}"})
                with urllib.request.urlopen(pedido, timeout=120) as resposta:
                    dados = resposta.read()
                self.pos += len(dados)
                return dados

        url = ("https://huggingface.co/datasets/akcit-pixel/RealBokeh/resolve/main/"
               "data/train-00000-of-00085.parquet")
        tabela = pq.ParquetFile(_HttpFile(url)).read(columns=["file_name_base"])
        nomes = tabela.column("file_name_base").to_pylist()
        self.assertGreater(len(nomes), 0)

        niveis_por_cena: dict[str, list[int]] = {}
        for nome in nomes:
            cena, nivel = parse_file_name_base(nome)     # levanta se o padrão mudou
            niveis_por_cena.setdefault(cena, []).append(nivel)
        # 1-based: nenhum nível 0, e os níveis de uma cena completa são contíguos de 1.
        self.assertEqual(min(min(v) for v in niveis_por_cena.values()), 1)


if __name__ == "__main__":
    unittest.main()
