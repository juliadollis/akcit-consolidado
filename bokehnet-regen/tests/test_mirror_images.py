"""Testes de `sources/mirror_images.py`.

O que aqui se prova, e por quê:

* **ordem dos canais** — trocar RGB por BGR não quebra nada visivelmente, e o erro
  atravessa todo o pipeline até virar bokeh com cor invertida no dataset final. É a
  classe de defeito que só um teste com pixel de cor conhecida pega.
* **amostragem do piloto por CENA** — o piloto é o que congela os dez limiares. Se ele
  sair de uma dúzia de cenas, os limiares saem calibrados para uma dúzia de cenas.
* **resolução declarada** — K vive em pixel. Uma imagem que chega em outra resolução
  muda o significado do rótulo sem mudar nada no JSON.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                    # noqa: E402
from PIL import Image                                                 # noqa: E402

from control.contract import SampleRejected                           # noqa: E402
from sources.mirror_images import (                                   # noqa: E402
    INDEX_FILENAME, MirrorImageLoader, MirrorIndex, MirrorLocation,
    order_pairs_for_sequential_read, sample_pairs_for_pilot,
)

try:
    import pyarrow  # noqa: F401
    TEM_PYARROW = True
except ImportError:
    TEM_PYARROW = False


class _Par:
    """O mínimo do `PairSource` que este módulo toca."""

    def __init__(self, scene_id: str, level: int, name: str):
        self.scene_id = scene_id
        self.level = level
        self.sample_id = f"c_realbokeh_{scene_id}_l{level}"
        self.source_sample_id = name
        self.aif_ref = "image_focus"
        self.bokeh_ref = "image_blur"


def _png(rgb: tuple[int, int, int], hw: tuple[int, int] = (4, 6)) -> bytes:
    arr = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
    arr[:, :] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _index(pares: dict[str, tuple[str, int]], shards: list[str]) -> MirrorIndex:
    return MirrorIndex({n: MirrorLocation(s, r) for n, (s, r) in pares.items()},
                       shards=shards)


class TestIndice(unittest.TestCase):

    def test_le_do_cache_sem_tocar_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / INDEX_FILENAME).write_text(json.dumps({
                "shards": ["a.parquet", "b.parquet"],
                "locations": {"n0": ["a.parquet", 0], "n1": ["b.parquet", 7]},
            }), encoding="utf-8")
            idx = MirrorIndex.build(root)
        self.assertEqual(len(idx), 2)
        self.assertEqual(idx.locate("n1"), MirrorLocation("b.parquet", 7))

    def test_nome_ausente_diz_o_que_fazer(self):
        idx = _index({"n0": ("a.parquet", 0)}, ["a.parquet"])
        with self.assertRaises(KeyError) as ctx:
            idx.locate("nao_existe")
        self.assertIn("force=True", str(ctx.exception))

    def test_sem_parquet_e_sem_cache_ensina_o_download(self):
        if not TEM_PYARROW:
            self.skipTest("build sem cache importa pyarrow")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                MirrorIndex.build(tmp)
        self.assertIn("huggingface-cli download", str(ctx.exception))


class TestOrdemDeLeitura(unittest.TestCase):

    def test_ordena_por_shard_e_linha(self):
        idx = _index({"z": ("s00.parquet", 40_000), "a": ("s07.parquet", 12),
                      "m": ("s00.parquet", 3)},
                     ["s00.parquet", "s07.parquet"])
        pares = [_Par("1", 1, "a"), _Par("2", 1, "z"), _Par("3", 1, "m")]
        nomes = [p.source_sample_id for p in order_pairs_for_sequential_read(pares, idx)]
        self.assertEqual(nomes, ["m", "z", "a"])

    def test_ordem_e_estavel_entre_chamadas(self):
        idx = _index({f"n{i}": ("s0.parquet", i) for i in range(20)}, ["s0.parquet"])
        pares = [_Par(str(i), 1, f"n{i}") for i in range(20)]
        a = [p.source_sample_id for p in order_pairs_for_sequential_read(pares, idx)]
        b = [p.source_sample_id for p in order_pairs_for_sequential_read(pares[::-1], idx)]
        self.assertEqual(a, b)


class TestAmostragemDoPiloto(unittest.TestCase):

    def _universo(self, cenas: int = 40, niveis: int = 5) -> list[_Par]:
        return [_Par(f"{c:04d}", n, f"c{c}_l{n}")
                for c in range(cenas) for n in range(1, niveis + 1)]

    def test_traz_cena_inteira_ou_nenhuma(self):
        universo = self._universo()
        escolhidos = sample_pairs_for_pilot(universo, limit=50, seed=0)
        por_cena: dict[str, int] = {}
        for p in escolhidos:
            por_cena[p.scene_id] = por_cena.get(p.scene_id, 0) + 1
        for cena, n in por_cena.items():
            with self.subTest(cena=cena):
                self.assertEqual(n, 5, "cena entrou pela metade — não é a unidade certa")

    def test_espalha_por_muitas_cenas_e_nao_pega_prefixo(self):
        universo = self._universo()
        escolhidos = sample_pairs_for_pilot(universo, limit=50, seed=0)
        cenas = {p.scene_id for p in escolhidos}
        self.assertGreaterEqual(len(cenas), 10)
        # o prefixo da ordem de disco seriam as cenas 0000..0009
        prefixo = {f"{c:04d}" for c in range(10)}
        self.assertNotEqual(cenas, prefixo)

    def test_deterministico_na_seed_e_sensivel_a_ela(self):
        universo = self._universo()
        a = [p.sample_id for p in sample_pairs_for_pilot(universo, limit=50, seed=0)]
        b = [p.sample_id for p in sample_pairs_for_pilot(universo, limit=50, seed=0)]
        c = [p.sample_id for p in sample_pairs_for_pilot(universo, limit=50, seed=1)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_limite_maior_que_o_universo_devolve_tudo(self):
        universo = self._universo(cenas=3, niveis=2)
        self.assertEqual(len(sample_pairs_for_pilot(universo, limit=999)), 6)

    def test_limite_nao_positivo_e_erro(self):
        with self.assertRaises(ValueError):
            sample_pairs_for_pilot(self._universo(), limit=0)


class TestDecodificacao(unittest.TestCase):

    def _loader(self, **kw) -> MirrorImageLoader:
        return MirrorImageLoader("/nao/usado", _index({}, []), **kw)

    def test_devolve_bgr_e_nao_rgb(self):
        """Vermelho puro tem que sair como `[0, 0, 255]`."""
        loader = self._loader()
        out = loader._decode({"bytes": _png((255, 0, 0))},
                             sample_id="s", column="image_focus")
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (4, 6, 3))
        np.testing.assert_array_equal(out[0, 0], [0, 0, 255])

    def test_azul_confirma_o_outro_extremo(self):
        out = self._loader()._decode({"bytes": _png((0, 0, 255))},
                                     sample_id="s", column="image_blur")
        np.testing.assert_array_equal(out[0, 0], [255, 0, 0])

    def test_saida_e_contigua(self):
        out = self._loader()._decode({"bytes": _png((10, 20, 30))},
                                     sample_id="s", column="image_focus")
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_cinza_vira_tres_canais(self):
        buf = io.BytesIO()
        Image.fromarray(np.full((4, 6), 128, dtype=np.uint8), mode="L").save(buf, "PNG")
        out = self._loader()._decode({"bytes": buf.getvalue()},
                                     sample_id="s", column="image_focus")
        self.assertEqual(out.shape, (4, 6, 3))

    def test_bytes_corrompidos_rejeitam_com_slug(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._decode({"bytes": b"nao sou png"},
                                   sample_id="s", column="image_focus")
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_celula_vazia_rejeita(self):
        with self.assertRaises(SampleRejected) as ctx:
            self._loader()._decode({"bytes": None, "path": None},
                                   sample_id="s", column="image_blur")
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")

    def test_resolucao_diferente_da_declarada_rejeita(self):
        """K vive em pixel: outra resolução é outro rótulo."""
        loader = self._loader(expected_hw=(1500, 2000))
        with self.assertRaises(SampleRejected) as ctx:
            loader._decode({"bytes": _png((1, 2, 3), hw=(4, 6))},
                           sample_id="s", column="image_focus")
        self.assertEqual(ctx.exception.reason, "source_image_unreadable")
        self.assertIn("CONTRATO", str(ctx.exception))

    def test_sem_expected_hw_qualquer_resolucao_passa(self):
        out = self._loader()._decode({"bytes": _png((1, 2, 3), hw=(8, 8))},
                                     sample_id="s", column="image_focus")
        self.assertEqual(out.shape, (8, 8, 3))


@unittest.skipUnless(TEM_PYARROW, "pyarrow não instalado nesta máquina")
class TestSobreParquetDeVerdade(unittest.TestCase):
    """Roda no cluster, onde pyarrow existe. Constrói um espelho falso de 2 shards."""

    def _espelho(self, root: Path) -> list[_Par]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pares = []
        for shard, cenas in (("s00.parquet", ["0001", "0002"]),
                             ("s01.parquet", ["0003"])):
            nomes, focus, blur = [], [], []
            for cena in cenas:
                for nivel in (1, 2):
                    nome = f"tim_{cena}_level_{nivel}_aligned"
                    nomes.append(nome)
                    focus.append({"bytes": _png((255, 0, 0)), "path": nome + "_f.png"})
                    blur.append({"bytes": _png((0, 255, 0)), "path": nome + "_b.png"})
                    pares.append(_Par(cena, nivel, nome))
            pq.write_table(pa.table({"file_name_base": nomes,
                                     "image_focus": focus, "image_blur": blur}),
                           root / shard)
        return pares

    def test_indice_e_carga_fecham_o_ciclo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pares = self._espelho(root)
            idx = MirrorIndex.build(root)
            self.assertEqual(len(idx), 6)
            self.assertTrue((root / INDEX_FILENAME).is_file())

            # segunda chamada vem do cache e dá o mesmo resultado
            self.assertEqual(len(MirrorIndex.build(root)), 6)

            loader = MirrorImageLoader(root, idx, expected_hw=(4, 6))
            ordenados = order_pairs_for_sequential_read(pares, idx)
            for par in ordenados:
                aif, bokeh = loader(par)
                np.testing.assert_array_equal(aif[0, 0], [0, 0, 255])
                np.testing.assert_array_equal(bokeh[0, 0], [0, 255, 0])

    def test_nome_repetido_no_espelho_e_erro(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = {"bytes": _png((1, 1, 1)), "path": "x.png"}
            pq.write_table(pa.table({"file_name_base": ["dup", "dup"],
                                     "image_focus": [cell, cell],
                                     "image_blur": [cell, cell]}),
                           root / "s.parquet")
            with self.assertRaises(ValueError) as ctx:
                MirrorIndex.build(root)
        self.assertIn("repetidos", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
