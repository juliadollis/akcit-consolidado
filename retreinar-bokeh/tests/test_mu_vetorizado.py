"""C4 — a reta do `mu` do shift do FLUX, fixada em números.

O `mu` sai de `calculate_shift` do diffusers, que é uma reta em `seq_len`
definida pela config do scheduler do FLUX-dev:

    m  = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    mu = seq_len * m + (base_shift - base_image_seq_len * m)

Estes quatro valores sustentam a tabela do C4 no plano e a escolha dos braços
do experimento. Ficam fixados aqui para que uma mudança em `sample_sigma` que
altere a distribuição de sigma seja pega por um teste, e não por um treino de
três dias que sai diferente sem explicação.

seq 672  = 512x336  -> avaliação com long_side=512 numa imagem 3:2
seq 1024 = 512x512  -> o crop de treino
seq 2752 = 1024x688 -> imagem inteira armazenada (o que a inferência usa para
                       calcular o mu, ANTES do tiling)
seq 7350 = 1680x1120 -> DPDD em resolução original
"""

from __future__ import annotations

import math
import unittest

# Config do scheduler do FLUX.1-dev.
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


def calculate_shift(
    seq_len: float,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - base_seq_len * m
    return seq_len * m + b


ESPERADO = {
    672:  (0.5704, 1.769),
    1024: (0.6300, 1.878),
    2752: (0.9225, 2.516),
    7350: (1.7008, 5.478),
}


def tokens(largura: int, altura: int) -> int:
    """Tokens FLUX: VAE 8x, _pack_latents 2x -> 16 px por token em cada eixo."""
    return (largura // 16) * (altura // 16)


class TestMu(unittest.TestCase):
    def test_valores_do_plano(self):
        for seq, (mu_esp, exp_esp) in ESPERADO.items():
            with self.subTest(seq=seq):
                mu = calculate_shift(seq)
                self.assertAlmostEqual(mu, mu_esp, places=4)
                self.assertAlmostEqual(math.exp(mu), exp_esp, places=3)

    def test_dimensoes_batem_com_os_seq_len(self):
        self.assertEqual(tokens(512, 336), 672)
        self.assertEqual(tokens(512, 512), 1024)
        self.assertEqual(tokens(1024, 688), 2752)
        self.assertEqual(tokens(1680, 1120), 7350)

    def test_mu_e_monotono_crescente_em_seq(self):
        seqs = sorted(ESPERADO)
        mus = [calculate_shift(s) for s in seqs]
        self.assertEqual(mus, sorted(mus))

    def test_a_reta_passa_pelos_pontos_de_ancoragem(self):
        self.assertAlmostEqual(calculate_shift(BASE_IMAGE_SEQ_LEN), BASE_SHIFT, places=12)
        self.assertAlmostEqual(calculate_shift(MAX_IMAGE_SEQ_LEN), MAX_SHIFT, places=12)

    def test_o_crop_e_a_imagem_inteira_dao_mu_diferente(self):
        """O fato que motiva o eixo `sigma_mu_source` do C4."""
        mu_crop = calculate_shift(tokens(512, 512))
        mu_inteira = calculate_shift(tokens(1024, 688))
        self.assertGreater(math.exp(mu_inteira) / math.exp(mu_crop), 1.3)

    def test_shift_aplicado_ao_sigma_preserva_o_intervalo(self):
        """sigma = exp(mu)*u / (1 + (exp(mu)-1)*u) leva (0,1) em (0,1)."""
        for seq in ESPERADO:
            e = math.exp(calculate_shift(seq))
            for u in (1e-6, 0.25, 0.5, 0.75, 1 - 1e-6):
                s = (e * u) / (1.0 + (e - 1.0) * u)
                self.assertGreater(s, 0.0)
                self.assertLess(s, 1.0000001)
            self.assertAlmostEqual((e * 1.0) / (1.0 + (e - 1.0) * 1.0), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
