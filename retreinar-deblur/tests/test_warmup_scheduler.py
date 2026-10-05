"""C3 — o LR do primeiro update.

DEFEITO ORIGINAL (árvores antigas, `genfocus_train/trainer.py`):
`optimizer.step()` era chamado ANTES de `scheduler.step(global_step)`, e o
`param_groups[0]["lr"]` nasce com o lr base do config. Resultado com
lr=1e-4 e warmup=500:

    update #1: lr = 1.000e-04   <-- 500x acima do primeiro ponto do warmup
    update #2: lr = 4.000e-07
    update #3: lr = 6.000e-07

ACHADO DESTA REVISÃO, que corrige o próprio plano: o defeito é SÓ a ordem.
A fórmula `base_lr * (step + 1) / warmup` já está certa — dela saem 2e-7,
4e-7, 6e-7 para os updates #1, #2, #3 quando o scheduler é aplicado ANTES do
update. O plano propunha também trocar `(step + 1)` por `max(step, 1)`; isso
estaria ERRADO, porque produziria 2e-7, 2e-7, 4e-7 — um degrau achatado no
começo e todo o warmup deslocado um passo para baixo. Este teste fixa o alvo
correto para que ninguém aplique aquele "fix".

CORREÇÃO: chamar `scheduler.step(state.global_step)` uma vez antes do laço
(com global_step=0, ou no ponto certo da curva quando há resume), mantendo a
fórmula intacta.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

from _helpers import OtimizadorFalso, extrair_classe

LR_BASE = 1.0e-4
WARMUP = 500
TOTAL = 60000
MIN_RATIO = 0.1


def _scheduler(otimizador):
    cls = extrair_classe(
        "trainer.py", "WarmupCosineScheduler", {"math": math, "Any": Any}
    )
    return cls(
        optimizer=otimizador,
        total_steps=TOTAL,
        warmup_steps=WARMUP,
        min_lr_ratio=MIN_RATIO,
    )


class TestWarmup(unittest.TestCase):
    def test_curva_do_warmup_e_linear_de_2e7_em_2e7(self):
        """A fórmula, isolada da ordem de chamada."""
        sched = _scheduler(OtimizadorFalso(LR_BASE))
        esperado = [2.0e-7, 4.0e-7, 6.0e-7, 8.0e-7]
        obtido = [sched._lr_at(LR_BASE, k) for k in range(4)]
        for i, (e, o) in enumerate(zip(esperado, obtido)):
            self.assertAlmostEqual(
                o, e, places=12,
                msg=(
                    f"_lr_at(step={i}) = {o:.3e}, esperado {e:.3e}. "
                    "Se deu 2e-7 duas vezes seguidas, alguém trocou (step+1) "
                    "por max(step,1) — ver o docstring deste arquivo."
                ),
            )

    def test_ordem_correta_o_primeiro_update_nao_ve_o_lr_base(self):
        """O que o C3 de fato corrige: aplicar o scheduler ANTES do update."""
        otim = OtimizadorFalso(LR_BASE)
        sched = _scheduler(otim)

        vistos = []
        global_step = 0
        sched.step(global_step)             # <-- a linha que o C3 acrescenta
        for _ in range(4):
            vistos.append(otim.lr)          # o lr que o optimizer.step() usaria
            global_step += 1
            sched.step(global_step)

        self.assertAlmostEqual(vistos[0], 2.0e-7, places=12,
                               msg=f"primeiro update viu {vistos[0]:.3e}; "
                                   "sem o scheduler.step inicial isso é 1e-4")
        for i, e in enumerate([2.0e-7, 4.0e-7, 6.0e-7, 8.0e-7]):
            self.assertAlmostEqual(vistos[i], e, places=12)

    def test_ordem_antiga_reproduz_o_defeito(self):
        """Documenta o bug: sem o step inicial, o update #1 vê o lr base."""
        otim = OtimizadorFalso(LR_BASE)
        sched = _scheduler(otim)

        vistos = []
        global_step = 0
        for _ in range(3):                  # SEM o sched.step(0) inicial
            vistos.append(otim.lr)
            global_step += 1
            sched.step(global_step)

        self.assertAlmostEqual(vistos[0], LR_BASE, places=12)
        self.assertAlmostEqual(vistos[0] / 2.0e-7, 500.0, places=6)

    def test_fim_do_warmup_e_o_lr_base(self):
        sched = _scheduler(OtimizadorFalso(LR_BASE))
        self.assertAlmostEqual(sched._lr_at(LR_BASE, WARMUP - 1), LR_BASE, places=12)

    def test_cosseno_termina_no_min_lr_ratio(self):
        sched = _scheduler(OtimizadorFalso(LR_BASE))
        self.assertAlmostEqual(
            sched._lr_at(LR_BASE, TOTAL), LR_BASE * MIN_RATIO, places=12
        )


if __name__ == "__main__":
    unittest.main()
