"""Testes do runtime de modelos — o que dá para verificar sem GPU.

O que importa aqui não é a inferência (precisa de GPU e checkpoint), é o CONTRATO:
falha alto e cedo, proveniência completa, e nenhum caminho de fallback.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import DEPTH_BACKENDS                        # noqa: E402
from model_runtime import DEPTH_BACKEND, MASK_BACKEND               # noqa: E402
from model_runtime.depth import DepthProRuntime, _resize_bilinear   # noqa: E402
from model_runtime.segmentation import BiRefNetRuntime, _sha256_dir  # noqa: E402


class SemFallback(unittest.TestCase):
    def test_depth_sem_checkpoint_falha_alto_e_cedo(self):
        """Antes havia três `except Exception: pass` em cascata. O Depth Anything
        devolve DISPARIDADE, gravada igual, e a amostra saía espelhada."""
        with self.assertRaises(FileNotFoundError) as ctx:
            DepthProRuntime("/caminho/que/nao/existe/depth_pro.pt")
        self.assertIn("NÃO existe backend alternativo", str(ctx.exception))

    def test_birefnet_sem_snapshot_falha_alto_e_cedo(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            BiRefNetRuntime("/caminho/que/nao/existe")
        self.assertIn("mask_source", str(ctx.exception))

    def test_backend_registrado_no_contrato(self):
        """O nome que vai no disco tem que ser aceito por `validate_metadata`."""
        self.assertIn(DEPTH_BACKEND, DEPTH_BACKENDS)
        self.assertEqual(MASK_BACKEND, "birefnet")

    def test_importa_sem_torch(self):
        """torch é importado dentro dos métodos: o pacote tem que carregar sem GPU."""
        import model_runtime
        self.assertTrue(hasattr(model_runtime, "DepthProRuntime"))


class Proveniencia(unittest.TestCase):
    def test_depth_hasheia_o_checkpoint_antes_de_carregar(self):
        """O hash é barato e obrigatório; o modelo é caro e preguiçoso."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "depth_pro.pt"
            ckpt.write_bytes(b"conteudo-de-teste")
            rt = DepthProRuntime(ckpt, device="cpu")
            prov = rt.provenance()
            self.assertEqual(prov["depth_backend"], "depth_pro")
            self.assertEqual(len(prov["depth_model_sha256"]), 64)
            self.assertIsNone(rt._model)          # não carregou

    def test_hash_de_snapshot_e_estavel_e_sensivel(self):
        """Independente da ordem do filesystem, e muda se o conteúdo mudar."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birefnet"; root.mkdir()
            (root / "config.json").write_bytes(b"{}")
            (root / "model.bin").write_bytes(b"pesos")
            a = _sha256_dir(root)
            self.assertEqual(a, _sha256_dir(root))
            (root / "model.bin").write_bytes(b"pesos-v2")
            self.assertNotEqual(a, _sha256_dir(root))

    def test_birefnet_grava_o_limiar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"; root.mkdir()
            (root / "config.json").write_bytes(b"{}")
            prov = BiRefNetRuntime(root, device="cpu", threshold=0.4).provenance()
            self.assertEqual(prov["mask_threshold"], 0.4)
            self.assertEqual(len(prov["mask_model_sha256"]), 64)


class Reamostragem(unittest.TestCase):
    def test_upsample_bilinear_preserva_extremos(self):
        a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        out = _resize_bilinear(a, (4, 4))
        self.assertEqual(out.shape, (4, 4))
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[-1, -1]), 3.0, places=5)

    def test_mesma_resolucao_e_noop(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertIs(_resize_bilinear(a, (3, 4)), a)

    def test_bilinear_e_monotono_em_rampa(self):
        """Upsample de campo suave não pode inventar oscilação."""
        a = np.linspace(1.0, 10.0, 16, dtype=np.float32).reshape(4, 4)
        out = _resize_bilinear(a, (8, 8))
        self.assertTrue(np.all(np.diff(out[0]) >= -1e-5))



class TestHashDoSnapshot(unittest.TestCase):
    """O `mask_model_sha256` tem que mudar quando o MODELO muda, e só então."""

    def _snapshot(self, arquivos: dict) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raiz = Path(tmp.name)
        for nome, conteudo in arquivos.items():
            destino = raiz / nome
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(conteudo if isinstance(conteudo, bytes)
                                else conteudo.encode("utf-8"))
        return raiz

    def test_cache_do_hf_nao_entra_no_hash(self):
        """`snapshot_download` deixa `.cache/huggingface/download/*.metadata` dentro do
        diretório do modelo, com etag e horário do download. Medido no BiRefNet baixado
        no cluster: 15 arquivos.

        Se entrassem, dois snapshots byte a byte iguais teriam hashes diferentes, e a
        proveniência diria "modelo diferente" sem diferença nenhuma.
        """
        limpo = self._snapshot({"config.json": '{"a":1}', "model.safetensors": b"pesos"})
        sujo = self._snapshot({
            "config.json": '{"a":1}', "model.safetensors": b"pesos",
            ".cache/huggingface/download/model.safetensors.metadata": "etag-2026-09-10",
            ".cache/huggingface/download/config.json.lock": "",
        })
        self.assertEqual(_sha256_dir(limpo), _sha256_dir(sujo))

    def test_conteudo_diferente_muda_o_hash(self):
        a = self._snapshot({"config.json": '{"a":1}', "model.safetensors": b"pesos"})
        b = self._snapshot({"config.json": '{"a":1}', "model.safetensors": b"outros"})
        self.assertNotEqual(_sha256_dir(a), _sha256_dir(b))

    def test_arquivo_a_mais_muda_o_hash(self):
        a = self._snapshot({"config.json": "{}"})
        b = self._snapshot({"config.json": "{}", "extra.py": "print(1)"})
        self.assertNotEqual(_sha256_dir(a), _sha256_dir(b))

    def test_mesmo_conteudo_em_pasta_diferente_muda_o_hash(self):
        """O hash usa o caminho relativo, não só o nome: mover um arquivo de lugar é
        uma mudança no snapshot."""
        a = self._snapshot({"sub/x.bin": b"igual"})
        b = self._snapshot({"outra/x.bin": b"igual"})
        self.assertNotEqual(_sha256_dir(a), _sha256_dir(b))

    def test_hash_e_deterministico(self):
        raiz = self._snapshot({"config.json": "{}", "a/b.bin": b"x"})
        self.assertEqual(_sha256_dir(raiz), _sha256_dir(raiz))


if __name__ == "__main__":
    unittest.main(verbosity=2)