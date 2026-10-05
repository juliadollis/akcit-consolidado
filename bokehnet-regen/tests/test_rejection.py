"""Testes do registro de rejeição — a infraestrutura que substitui o fallback."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import SampleRejected            # noqa: E402
from qc.rejection import RejectionLog                  # noqa: E402


class Rejection(unittest.TestCase):
    def test_histograma_agrega_por_motivo(self):
        log = RejectionLog()
        log.accept("a")
        log.reject("b", "sensor_width_unresolvable")
        log.reject("c", "sensor_width_unresolvable")
        log.reject("d", "focus_mask_empty")
        self.assertEqual(log.accepted, 1)
        self.assertEqual(log.rejected, 3)
        self.assertEqual(log.reasons["sensor_width_unresolvable"], 2)
        self.assertIn("sensor_width_unresolvable", log.summary())
        self.assertIn("focus_mask_empty", log.summary())

    def test_captura_a_excecao_do_contrato(self):
        log = RejectionLog()
        try:
            raise SampleRejected("k_non_positive", "K=0.0")
        except SampleRejected as exc:
            log.reject_from("x", exc)
        self.assertEqual(log.reasons["k_non_positive"], 1)

    def test_retomada_ignora_apenas_as_aceitas(self):
        """Rejeitada tem que ser reprocessada: um gate recalibrado pode aceitá-la."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            log = RejectionLog(path)
            log.accept("ok-1")
            log.reject("rej-1", "focus_mask_empty")
            log.close()
            self.assertEqual(RejectionLog(path).completed_ids(), {"ok-1"})

    def test_jsonl_e_valido_e_tem_o_motivo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            log = RejectionLog(path)
            log.reject("s", "depth_range_degenerate", "z_max/z_min=1.001", {"scene_id": "42"})
            log.close()
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["reason"], "depth_range_degenerate")
            self.assertEqual(row["scene_id"], "42")
            self.assertEqual(row["status"], "rejected")

    def test_summary_sem_amostra_nao_quebra(self):
        self.assertIn("nenhuma amostra", RejectionLog().summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
