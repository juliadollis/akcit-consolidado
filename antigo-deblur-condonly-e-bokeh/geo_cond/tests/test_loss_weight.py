"""Testes do peso da perda em espaço de token. Sem GPU.

O teste central é o de ORDENACAO: constrói um mapa com um único bloco de 16x16
pixels aceso e confere que exatamente um token acende, e que o índice dele é o
esperado. Um erro de ordenação aqui não levanta exceção nenhuma e custaria 60K
steps de treino aprendendo a ponderar o lugar errado.

Usa o `_pack_latents` REAL do diffusers instalado, não uma reimplementação.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from geo_cond.loss_weight import (
    VAE_STRIDE,
    occlusion_to_token_weight,
    weighted_flow_matching_loss,
)

PACK = FluxPipeline._pack_latents
TOKEN_PX = VAE_STRIDE * 2          # 16 pixels de lado por token


def _zeros(B=1, H=64, W=64):
    return torch.zeros(B, 1, H, W)


# ------------------------------------------------------------- ordenação

@pytest.mark.parametrize("bloco_y,bloco_x", [(0, 0), (0, 3), (2, 1), (3, 3)])
def test_um_bloco_aceso_acende_exatamente_um_token_no_indice_certo(bloco_y, bloco_x):
    H = W = 64
    o = _zeros(H=H, W=W)
    y0, x0 = bloco_y * TOKEN_PX, bloco_x * TOKEN_PX
    o[0, 0, y0:y0 + TOKEN_PX, x0:x0 + TOKEN_PX] = 1.0

    w = occlusion_to_token_weight(o, PACK, lambda_o=1.0, normalize=False)
    acesos = (w[0, :, 0] > 1.0).nonzero().flatten().tolist()

    blocos_por_linha = W // TOKEN_PX
    esperado = bloco_y * blocos_por_linha + bloco_x
    assert acesos == [esperado], (
        f"bloco ({bloco_y},{bloco_x}) devia acender o token {esperado}, acendeu {acesos}"
    )
    assert w.shape == (1, (H // TOKEN_PX) * (W // TOKEN_PX), 1)


def test_um_pixel_aceso_acende_o_token_que_o_contem():
    """Resolução fina: um único pixel, não um bloco inteiro."""
    o = _zeros(H=64, W=64)
    o[0, 0, 37, 5] = 1.0                       # bloco (2, 0)
    w = occlusion_to_token_weight(o, PACK, lambda_o=1.0, normalize=False)
    acesos = (w[0, :, 0] > 1.0).nonzero().flatten().tolist()
    assert acesos == [2 * (64 // TOKEN_PX) + 0]


def test_mapa_uniforme_da_peso_uniforme():
    o = torch.full((2, 1, 64, 64), 0.5)
    w = occlusion_to_token_weight(o, PACK, lambda_o=2.0, normalize=False)
    assert torch.allclose(w, torch.full_like(w, 1.0 + 2.0 * 0.5))


# ------------------------------------------------------- max contra média

def test_max_preserva_a_amplitude_de_borda_fina_e_a_media_dilui():
    """A decisão 1 do módulo, em número: uma borda de 1 px de largura dentro de
    um bloco de 16x16 é 1/16 da área, então a média a reduz a ~1/16."""
    o = _zeros(H=64, W=64)
    o[0, 0, :, 16] = 1.0                        # uma coluna de 1 px
    w_max = occlusion_to_token_weight(o, PACK, lambda_o=3.0, pool="max", normalize=False)
    w_avg = occlusion_to_token_weight(o, PACK, lambda_o=3.0, pool="avg", normalize=False)
    pico_max = float(w_max.max()) - 1.0
    pico_avg = float(w_avg.max()) - 1.0
    assert abs(pico_max - 3.0) < 1e-6, f"max devia dar lambda cheio, deu {pico_max:.4f}"
    assert abs(pico_avg - 3.0 / TOKEN_PX) < 1e-6, f"media devia diluir por 1/16, deu {pico_avg:.4f}"
    assert pico_max / pico_avg > 15.0


# ------------------------------------------------------------ normalização

def test_normalizacao_mantem_a_media_do_peso_em_um():
    """Sem isso, lambda_o vira multiplicador do passo do otimizador e o controle
    'perda ponderada sobre a linha de base' deixa de isolar o que deveria."""
    torch.manual_seed(0)
    o = torch.rand(3, 1, 64, 64)
    for lam in (0.5, 1.0, 3.0, 10.0):
        w = occlusion_to_token_weight(o, PACK, lambda_o=lam, normalize=True)
        assert abs(float(w.mean()) - 1.0) < 1e-6


def test_sem_normalizacao_a_media_cresce_com_lambda():
    torch.manual_seed(0)
    o = torch.rand(1, 1, 64, 64)
    m1 = float(occlusion_to_token_weight(o, PACK, lambda_o=1.0, normalize=False).mean())
    m3 = float(occlusion_to_token_weight(o, PACK, lambda_o=3.0, normalize=False).mean())
    assert m3 > m1 > 1.0


# ------------------------------------------------------------------ perda

def test_peso_none_reproduz_a_perda_atual_bit_a_bit():
    torch.manual_seed(1)
    pred, tgt = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    assert torch.equal(
        weighted_flow_matching_loss(pred, tgt), F.mse_loss(pred.float(), tgt.float())
    )


def test_lambda_zero_com_normalizacao_reproduz_a_perda_atual():
    """Continuidade da ablação: a condição com lambda_o = 0 tem de ser
    NUMERICAMENTE a linha de base, senão A' não isola nada."""
    torch.manual_seed(2)
    pred, tgt = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    o = torch.rand(2, 1, 64, 64)
    w = occlusion_to_token_weight(o, PACK, lambda_o=0.0, normalize=True)
    a = weighted_flow_matching_loss(pred, tgt, w)
    b = F.mse_loss(pred.float(), tgt.float())
    assert torch.allclose(a, b, atol=1e-6), f"{float(a)} vs {float(b)}"


def test_o_peso_concentra_a_perda_onde_a_oclusao_esta():
    """O efeito pretendido: erro na região marcada pesa mais que o mesmo erro
    fora dela."""
    N = (64 // TOKEN_PX) ** 2
    o = _zeros(H=64, W=64)
    o[0, 0, 0:TOKEN_PX, 0:TOKEN_PX] = 1.0                 # token 0
    w = occlusion_to_token_weight(o, PACK, lambda_o=3.0, normalize=True)

    pred, tgt = torch.zeros(1, N, 8), torch.zeros(1, N, 8)
    dentro = pred.clone(); dentro[0, 0] = 1.0
    fora = pred.clone(); fora[0, N - 1] = 1.0
    l_dentro = weighted_flow_matching_loss(dentro, tgt, w)
    l_fora = weighted_flow_matching_loss(fora, tgt, w)
    assert float(l_dentro) > float(l_fora) * 3.0


def test_dtype_bf16_entra_e_a_perda_sai_fp32():
    """O treino roda em bf16; a perda tem de subir para fp32 como a atual faz."""
    pred = torch.randn(1, 16, 64, dtype=torch.bfloat16)
    tgt = torch.randn(1, 16, 64, dtype=torch.bfloat16)
    w = torch.ones(1, 16, 1, dtype=torch.bfloat16)
    assert weighted_flow_matching_loss(pred, tgt, w).dtype == torch.float32


# ------------------------------------------------------------- validações

def test_shape_errado_levanta():
    with pytest.raises(ValueError):
        occlusion_to_token_weight(torch.zeros(1, 3, 64, 64), PACK, lambda_o=1.0)
    with pytest.raises(ValueError):
        occlusion_to_token_weight(torch.zeros(1, 1, 60, 64), PACK, lambda_o=1.0)
    with pytest.raises(ValueError):
        occlusion_to_token_weight(torch.zeros(1, 1, 64, 64), PACK, lambda_o=1.0, pool="min")
    with pytest.raises(ValueError):
        weighted_flow_matching_loss(torch.zeros(1, 4, 8), torch.zeros(1, 4, 8),
                                    torch.ones(1, 9, 1))


def test_512_da_1024_tokens_como_o_treino_real():
    """A aritmética que o plano usa: 512 px -> N = 1024 tokens, 1 token = 16x16."""
    w = occlusion_to_token_weight(_zeros(H=512, W=512), PACK, lambda_o=1.0)
    assert w.shape == (1, 1024, 1)


# ------------------------------------------------------- limiar em token

def test_theta_torna_o_peso_binario_e_esparso():
    """A correção que o diagnóstico F2 exigiu: sem limiar, 100% dos tokens
    recebem peso > 1 (a borda é 1,2% dos pixels mas 82,7x mais espalhada em
    token), e a normalização rebaixa quem não é borda."""
    torch.manual_seed(3)
    o = torch.rand(1, 1, 64, 64) * 0.5          # quase tudo abaixo de 0,5
    o[0, 0, 20:24, :] = 0.9                     # uma faixa alta
    cont = occlusion_to_token_weight(o, PACK, lambda_o=2.0, normalize=False)
    bin_ = occlusion_to_token_weight(o, PACK, lambda_o=2.0, normalize=False, theta=0.6)
    assert float((cont[..., 0] > 1.0).float().mean()) > 0.9, "sem limiar, quase todo token pesa"
    fr = float((bin_[..., 0] > 1.0).float().mean())
    assert 0.0 < fr < 0.5, f"com limiar o peso tem de ficar esparso, deu {fr:.3f}"
    vals = set(round(float(v), 6) for v in bin_.flatten())
    assert vals == {1.0, 3.0}, f"o peso tem de ser binario 1 ou 1+lambda, veio {vals}"


def test_theta_concentra_o_peso_em_vez_de_espalhar():
    """O que o limiar faz, e o que ele NAO faz.

    Escrevi este teste primeiro afirmando que o limiar ELEVA o piso. Errado: o
    "orcamento" de peso e o mesmo, porque a normalizacao pela media o conserva.
    Com theta=0,6 o piso ate CAI (0,667 contra 0,904), porque o peso alto fica
    concentrado em menos tokens e portanto e mais alto.

    O que o limiar entrega e CONTRASTE: a razao entre quem recebe supervisao
    extra e quem nao recebe. Com peso continuo essa razao e 1,41; com limiar e
    3,00. E o contraste que determina se a reponderacao distingue borda de
    nao-borda, e era ele que estava perdido quando 100% dos tokens recebiam peso
    parecido."""
    torch.manual_seed(4)
    o = torch.rand(1, 1, 64, 64) * 0.5
    o[0, 0, 20:24, :] = 0.9
    cont = occlusion_to_token_weight(o, PACK, lambda_o=2.0)
    binr = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.6)
    contraste_cont = float(cont.max() / cont.min())
    contraste_bin = float(binr.max() / binr.min())
    assert contraste_bin > contraste_cont * 1.5, (
        f"o limiar tem de aumentar o contraste: continuo={contraste_cont:.2f} "
        f"binario={contraste_bin:.2f}"
    )
    # e o peso alto tem de cair sobre a faixa que realmente e borda
    alta = binr[0, :, 0] > binr.min() + 1e-6
    assert 0.05 < float(alta.float().mean()) < 0.5


def test_theta_zero_reproduz_o_comportamento_do_Alinha():
    """Continuidade: a condição A' já treinada usou theta=0. O default tem de
    reproduzi-la exatamente, senão a comparação com ela se perde."""
    torch.manual_seed(5)
    o = torch.rand(2, 1, 64, 64)
    assert torch.equal(occlusion_to_token_weight(o, PACK, lambda_o=2.0),
                       occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.0))
