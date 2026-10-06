"""Peso da perda por oclusão, no espaço em que o treino realmente roda.

A seção 5.3 do documento de proposta escreve

    L = soma_x w(x) ||I_chapeu(x) - I(x)||_1 ,   w(x) = 1 + lambda_o O(x)

mas o treino não tem `I_chapeu` nem `I` em pixel em ponto nenhum. Ele é

    F.mse_loss(prediction, target)          genfocus_train/models.py:176-181

sobre a VELOCIDADE de flow matching em tokens latentes empacotados, de forma
(B, N, D), chamada em `trainer.py:639` e `trainer.py:931`. Decodificar o latente
a cada step para aplicar um peso em pixel colocaria o decoder do VAE no caminho
do gradiente por 60K steps.

Então o peso vive no espaço de token. A 512 pixels, N = (512/16)^2 = 1024 e
D = 64, e **um token corresponde a um bloco de 16x16 pixels** (o VAE reduz 8x e o
`_pack_latents` dobra mais 2x em cada eixo).

TRES DECISOES QUE DECIDEM SE ISSO FUNCIONA
------------------------------------------
1. Redução por MAX, não por média. Uma borda de oclusão tem 1 a 2 px de largura;
   a média a 16x dilui a amplitude por volta de 1/16, e um lambda_o = 3 vira um
   lambda_o efetivo de 0,2. O max preserva a amplitude e estende o peso ao bloco
   inteiro, que é a unidade que o modelo de fato prevê.
2. Normalizar `w` pela própria média. Sem isso lambda_o também multiplica o passo
   efetivo do otimizador, e o controle "perda ponderada sobre a linha de base"
   deixa de separar supervisão concentrada de learning rate maior, que é
   exatamente o que ele existe para separar.
3. A ordenação vem do `_pack_latents` do próprio pipeline, nunca de um reshape
   escrito à mão. Um erro de ordenação aqui é silencioso e custaria 60K steps.

Nota para o texto do paper: reponderar a perda de difusão de forma não uniforme
enviesa o estimador do score. É prática padrão (min-SNR weighting e afins), mas o
texto tem de dizer "reponderação perceptual da supervisão", não "perda ponderada
da verossimilhança".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["occlusion_to_token_weight", "weighted_flow_matching_loss", "VAE_STRIDE"]

#: Redução espacial do VAE do FLUX. O `_pack_latents` dobra mais 2x por eixo,
#: então um token cobre VAE_STRIDE * 2 = 16 pixels de lado.
VAE_STRIDE = 8


def occlusion_to_token_weight(
    occlusion: torch.Tensor,
    pack_latents,
    *,
    lambda_o: float,
    latent_channels: int = 16,
    pool: str = "max",
    normalize: bool = True,
    theta: float = 0.0,
) -> torch.Tensor:
    """Mapa de oclusão em pixel -> peso por token, na ordem do `_pack_latents`.

    Parameters
    ----------
    occlusion
        (B, 1, H, W) em [0,1]. H e W múltiplos de 16.
    pack_latents
        A função `_pack_latents` DO PIPELINE, passada de fora
        (`self._pipe._pack_latents`). Não reimplementamos a ordenação.
    lambda_o
        w = 1 + lambda_o * O_token. O documento sugere entre 1 e 3.
    pool
        "max" (default, ver decisão 1) ou "avg" (variante de ablação).
    normalize
        Divide `w` pela média do batch, para desacoplar lambda_o do passo
        efetivo do otimizador (decisão 2).
    theta
        Limiar sobre `O` JÁ REDUZIDO A TOKEN. Com theta > 0 o peso vira
        `w = 1 + lambda * [O_token > theta]`, binário e esparso.

        Existe por causa de uma medição, não por gosto: a região de borda é
        1,2% dos PIXELS mas **100% dos TOKENS**, porque o max-pool 16x espalha
        por um fator de 82,7x. Com peso contínuo e normalização pela média, isso
        rebaixou os tokens de menor oclusão a 0,816 com lambda=2,0, ou seja
        cortou ~18% da supervisão em 74% dos tokens. Foi o que fez a condição A'
        piorar em toda parte (E_fora +31%), e não só falhar na borda.

        theta = 0 (default) mantém o comportamento contínuo anterior, para que a
        comparação com o A' já treinado continue possível.

    Returns
    -------
    (B, N, 1), pronto para broadcast sobre D.
    """
    if occlusion.dim() != 4 or occlusion.shape[1] != 1:
        raise ValueError(f"occlusion deve ser (B, 1, H, W); recebido {tuple(occlusion.shape)}")
    B, _, H, W = occlusion.shape
    if H % (VAE_STRIDE * 2) or W % (VAE_STRIDE * 2):
        raise ValueError(f"H e W devem ser múltiplos de {VAE_STRIDE*2}; recebido {H}x{W}")
    if lambda_o < 0:
        raise ValueError(f"lambda_o deve ser >= 0; recebido {lambda_o}")

    o = occlusion.float()
    if pool == "max":
        red = F.max_pool2d(o, kernel_size=VAE_STRIDE, stride=VAE_STRIDE)
    elif pool == "avg":
        red = F.avg_pool2d(o, kernel_size=VAE_STRIDE, stride=VAE_STRIDE)
    else:
        raise ValueError(f"pool deve ser 'max' ou 'avg'; recebido {pool!r}")

    h, w_ = red.shape[-2:]
    # Replica para o número de canais do latente e empacota com a MESMA função do
    # pipeline. Depois do pack, os D = C*4 valores de um token são o bloco 2x2
    # repetido em C canais, então reduzir no último eixo devolve exatamente a
    # redução 16x, na ordem certa, sem nenhum reshape nosso.
    rep = red.expand(B, latent_channels, h, w_).contiguous()
    tok = pack_latents(rep, B, latent_channels, h, w_)          # (B, N, C*4)
    o_tok = tok.max(dim=-1).values if pool == "max" else tok.mean(dim=-1)

    if theta > 0.0:
        o_tok = (o_tok > theta).to(o_tok.dtype)
    weight = 1.0 + lambda_o * o_tok.unsqueeze(-1)               # (B, N, 1)
    if normalize:
        weight = weight / weight.mean().clamp_min(1e-8)
    return weight


def weighted_flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE de flow matching, opcionalmente ponderada por token.

    Com `weight=None` é EXATAMENTE `F.mse_loss(prediction.float(), target.float())`,
    a perda atual de `models.py:176-181`. O fp32 é o mesmo, pelo mesmo motivo de
    estabilidade numérica.
    """
    pred = prediction.float()
    tgt = target.float()
    if weight is None:
        return F.mse_loss(pred, tgt)
    if weight.dim() != 3 or weight.shape[0] != pred.shape[0] or weight.shape[1] != pred.shape[1]:
        raise ValueError(
            f"weight deve ser (B, N, 1) casando com pred {tuple(pred.shape)}; "
            f"recebido {tuple(weight.shape)}"
        )
    return (weight.float() * (pred - tgt) ** 2).mean()
