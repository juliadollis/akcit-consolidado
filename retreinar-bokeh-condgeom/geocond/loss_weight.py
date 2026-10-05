"""Peso da perda por oclusão — no espaço de TOKEN, e LIMIARIZADO.

Por que existe
--------------
A §5.3 do documento de proposta escreve a perda ponderada em PIXEL,

    L = Σ_x w(x)·‖Î(x) − I(x)‖₁ ,      w(x) = 1 + λ_o·O(x)

mas o treino não tem `Î` nem `I` em pixel em lugar nenhum. Ele é

    F.mse_loss(prediction.float(), target.float())      genfocus_train/models.py

sobre a VELOCIDADE de flow matching, em tokens latentes empacotados de forma
(B, N, D). Decodificar o latente a cada step só para aplicar um peso em pixel
colocaria o decoder do VAE no caminho do gradiente por 60K steps. Então o peso
vive no espaço de token, e é aí que mora todo o problema deste módulo.

A 512², N = (512/16)² = 1024 e **um token cobre um bloco de 16×16 pixels**: o
VAE reduz 8× e o `_pack_latents` do FLUX dobra mais 2× em cada eixo.

O ACHADO QUE DEFINE O DESENHO (a campanha anterior PIOROU o modelo)
-------------------------------------------------------------------
A v1 implementou `w = 1 + λ_o·O(x)` contínuo, com redução por max e normalização
pela média do batch. Mediu-se, depois do fato:

    fração de PIXELS com O > 0,3 : 0,0121      (a borda é 1,2% da imagem)
    fração de TOKENS com O > 0   : 1,0000      (a borda é ~100% dos tokens)
    espalhamento do max-pool 16× : 82,7×
    distribuição de O_token      : p50=0,019  p75=0,123  p90=0,639  p99=1,000

Ou seja: o max-pool 16× transforma uma borda de 1,2% dos pixels em praticamente
TODOS os tokens. Com peso contínuo e normalização pela média, os tokens de menor
oclusão caíram para 0,816 — a supervisão DIMINUIU ~18% em 74% dos tokens para
financiar o topo. O resultado medido foi exatamente isso: o erro na borda
melhorou em termos RELATIVOS (razão borda/fora caiu de 1,427 para 1,286) e
piorou em termos ABSOLUTOS em toda parte (E_fora +31%).

A CORREÇÃO, E O NÚMERO QUE IMPORTA
-----------------------------------
Limiarizar o peso EM TOKEN:   w = 1 + λ·[O_token > θ].

E o número que decide θ e λ **não é a fração de tokens pesados: é o PISO**.
A normalização pela média conserva o orçamento de peso, então concentrar em
menos tokens só os deixa mais pesados e AFUNDA o piso. Com peso binário o piso
tem forma fechada — `mean(w) = 1 + λ·f` e o token não-borda vale `1/mean(w)`:

    piso = 1 / (1 + λ·f)          contraste = 1 + λ        (f = fração pesada)

que é a tabela do diagnóstico, verificada termo a termo:

    θ=0,0  λ=2,0 → f 0,992  piso 0,806  contraste 3,00   (contínuo: foi o que piorou)
    θ=0,3  λ=2,0 → f 0,167  piso 0,750  contraste 3,00   (limiar só, piso CAI)
    θ=0,9  λ=1,0 → f 0,075  piso 0,930  contraste 2,00   (λ menor junto, piso SOBE)

Repare na segunda linha: limiarizar SOZINHO não salva o piso — ele cai de 0,806
para 0,750. O piso só sobe quando λ desce junto. Por isso `diagnostico_peso()`
existe: é para rodar ANTES de treinar, sobre O do release novo, e escolher (θ,λ)
olhando o piso. Sem ela a escolha é chute, e um chute custa 60K steps.

SEM DEFAULT DE θ OU λ, DE PROPÓSITO
------------------------------------
Os três pares acima vieram da distribuição ANTIGA de O (profundidade linear
normalizada por imagem — a que a `referencia_antiga/LEIA.md` invalidou). Eles
orientam o DESENHO, não são constantes. `lambda_o` e `theta` são keyword-only
SEM default: quem chama passa, depois de recalibrar sobre o release novo.

Nota para o texto do paper: reponderar a perda de difusão de forma não uniforme
enviesa o estimador do score. É prática padrão (min-SNR weighting e afins), mas
o texto tem de dizer "reponderação perceptual da supervisão", não "perda
ponderada da verossimilhança".
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

__all__ = [
    "VAE_STRIDE",
    "TOKEN_PX",
    "LATENT_CHANNELS",
    "occlusion_to_token_weight",
    "diagnostico_peso",
    "weighted_flow_matching_loss",
]

#: Redução espacial do VAE do FLUX.
VAE_STRIDE = 8

#: Lado, em PIXELS, do bloco que um token cobre. VAE 8× + `_pack_latents` 2×.
#: É este 16 que produz o espalhamento de 82,7× medido: a borda tem 1-2 px de
#: largura e o bloco tem 256 px de área.
TOKEN_PX = VAE_STRIDE * 2

#: Canais do latente do VAE do FLUX. Só entra aqui para que o tensor replicado
#: tenha a forma que o `_pack_latents` real espera; o valor do peso não depende
#: dele (todos os canais são cópias do mesmo mapa).
LATENT_CHANNELS = 16

_POOLS = {"max", "mean", "avg"}


def _valida_mapa(O_pixel: torch.Tensor) -> tuple[int, int, int]:
    if O_pixel.dim() != 4 or O_pixel.shape[1] != 1:
        raise ValueError(
            f"O_pixel deve ser (B, 1, H, W) — o canal O da pilha; "
            f"recebido {tuple(O_pixel.shape)}"
        )
    B, _, H, W = O_pixel.shape
    if H % TOKEN_PX or W % TOKEN_PX:
        raise ValueError(
            f"H e W devem ser múltiplos de {TOKEN_PX} (VAE 8× + _pack_latents 2×); "
            f"recebido {H}x{W}. Com resto, o último bloco de token não fecha e o "
            f"peso passaria a cair num token deslocado — silenciosamente."
        )
    return B, H, W


def _o_por_token(
    O_pixel: torch.Tensor,
    pack_latents_fn: Callable[..., torch.Tensor],
    *,
    pool: str,
    latent_channels: int,
) -> torch.Tensor:
    """(B,1,H,W) em pixel → (B,N) em token, NA ORDEM DO `_pack_latents` REAL.

    A redução é feita em dois tempos, e isso não é firula:

    1. pool 8× em pixel → resolução do LATENTE (H/8, W/8);
    2. `pack_latents_fn` agrupa o 2×2 restante e escreve os 4 valores no eixo
       final (D = C·4), na ordem exata do pipeline; reduzir esse eixo fecha o
       16×.

    O passo 2 usa a função RECEBIDA (o `_pack_latents` real do pipe), nunca um
    reshape nosso. Um erro de ordenação aqui NÃO levanta exceção: o treino
    aprenderia a ponderar o lugar errado da imagem por 60K steps e o único
    sintoma seria uma métrica pior no fim.
    """
    B, H, W = _valida_mapa(O_pixel)
    o = O_pixel.float()

    if pool == "max":
        red = F.max_pool2d(o, kernel_size=VAE_STRIDE, stride=VAE_STRIDE)
    else:  # "mean" / "avg"
        red = F.avg_pool2d(o, kernel_size=VAE_STRIDE, stride=VAE_STRIDE)

    h, w = red.shape[-2:]
    # Replica o mesmo mapa nos C canais do latente: o `_pack_latents` espera
    # (B, C, h, w) e devolve (B, N, C·4). Como os C canais são idênticos,
    # reduzir o eixo final é reduzir exatamente o bloco 2×2 — e a posição de
    # cada token na sequência veio do pipeline, não de nós.
    rep = red.expand(B, latent_channels, h, w).contiguous()
    tok = pack_latents_fn(rep, B, latent_channels, h, w)      # (B, N, C·4)
    if tok.dim() != 3 or tok.shape[0] != B:
        raise ValueError(
            f"pack_latents_fn devolveu {tuple(tok.shape)}; esperado (B, N, D). "
            "Passe o `_pack_latents` do pipe (self.backbone._pipe._pack_latents)."
        )
    return tok.max(dim=-1).values if pool == "max" else tok.mean(dim=-1)


def occlusion_to_token_weight(
    O_pixel: torch.Tensor,
    pack_latents_fn: Callable[..., torch.Tensor],
    *,
    lambda_o: float,
    theta: float,
    pool: str = "max",
    normalizar: bool = True,
    latent_channels: int = LATENT_CHANNELS,
) -> torch.Tensor:
    """Mapa de oclusão em pixel → peso por token (B, N, 1).

    Parameters
    ----------
    O_pixel
        (B, 1, H, W) em [0,1] — o canal `O` da pilha (`CANAIS[1]`), na MESMA
        resolução em que as imagens entram no VAE. H e W múltiplos de 16.
    pack_latents_fn
        O `_pack_latents` REAL do pipe (`backbone._pipe._pack_latents`). A
        ordenação dos tokens vem dele; ver `_o_por_token`.
    lambda_o
        Amplitude do peso extra: `w = 1 + λ·[O_token > θ]`. Sem default —
        recalibrar sobre o release novo olhando o `piso` de `diagnostico_peso`.
        λ=0 reproduz a linha de base exatamente (w ≡ 1), que é a condição A'.
    theta
        Limiar sobre `O` JÁ REDUZIDO A TOKEN. **Duas semânticas, e a diferença
        é o assunto deste módulo:**

        * `theta > 0` → peso BINÁRIO, `w = 1 + λ·[O_token > θ]`. É o modo
          normal. Valores de w: exatamente {1, 1+λ}.
        * `theta == 0` → peso CONTÍNUO, `w = 1 + λ·O_token`. É o modo da
          campanha v1, que PIOROU o modelo (E_fora +31%). Fica disponível
          apenas como BRAÇO DE ABLAÇÃO, para que a comparação com o A' já
          treinado continue possível. Não é para ser o modo de produção.
    pool
        "max" (padrão) ou "mean" ("avg" é aceito como sinônimo, para o modo de
        ablação).

        O max é OBRIGATÓRIO pelo motivo original: uma borda de oclusão tem 1 a
        2 px de largura e o bloco do token tem 16×16 = 256 px. A média dilui a
        amplitude por ~1/16, então um λ=3 nominal vira um λ EFETIVO de 0,2 —
        mede-se isso no teste `test_max_preserva...`. O max preserva a amplitude
        e estende o peso ao bloco inteiro, que é a unidade que o modelo prevê.
    normalizar
        Divide `w` pela MÉDIA DO BATCH, de modo que `mean(w) == 1`.

        Isto é o que impede λ de também multiplicar o passo efetivo do
        otimizador. Sem isso, o controle "perda ponderada contra a linha de
        base" deixaria de separar supervisão CONCENTRADA de learning rate MAIOR
        — que é exatamente o que ele existe para separar. O preço é o piso: com
        orçamento conservado, quem não é borda cai abaixo de 1. Ver
        `diagnostico_peso`.

        A média é do LOTE INTEIRO (B·N), não por amostra. É a escolha exata:
        a perda é `mean` sobre (B, N, D), então só a média global garante que a
        ESCALA da perda ponderada é idêntica à da não ponderada. Normalizar por
        amostra preservaria a escala só em média e faria uma amostra com muita
        borda ser rebaixada em relação a uma amostra lisa — trocaria a
        reponderação ENTRE TOKENS por uma reponderação entre AMOSTRAS, que não é
        o que a §5.3 pede.

    Returns
    -------
    torch.Tensor
        (B, N, 1) em float32, pronto para broadcast sobre D em (B, N, D).
        O chamador faz o `.to(pred.dtype)`.
    """
    if not (lambda_o >= 0.0):
        raise ValueError(f"lambda_o deve ser >= 0; recebido {lambda_o!r}")
    if not (0.0 <= theta < 1.0):
        raise ValueError(
            f"theta deve estar em [0,1); recebido {theta!r}. O ∈ [0,1] e o teste é "
            "ESTRITO (O > θ), então θ >= 1 esvaziaria a região de borda sem avisar."
        )
    if pool not in _POOLS:
        raise ValueError(f"pool deve ser 'max' ou 'mean'; recebido {pool!r}")

    o_tok = _o_por_token(
        O_pixel, pack_latents_fn, pool=pool, latent_channels=latent_channels
    )                                                        # (B, N)

    if theta > 0.0:
        ativo = (o_tok > theta).to(o_tok.dtype)              # binário
    else:
        ativo = o_tok                                        # contínuo (ablação)

    weight = 1.0 + float(lambda_o) * ativo.unsqueeze(-1)     # (B, N, 1)
    if normalizar:
        # clamp_min só protege contra batch degenerado; com λ>=0 a média é >= 1.
        weight = weight / weight.mean().clamp_min(1e-8)
    return weight


def diagnostico_peso(
    O_pixel: torch.Tensor,
    pack_latents_fn: Callable[..., torch.Tensor],
    *,
    lambda_o: float,
    theta: float,
    pool: str = "max",
    latent_channels: int = LATENT_CHANNELS,
) -> dict:
    """Os números para escolher (θ, λ) ANTES de treinar. Sem isso a escolha é chute.

    Rode sobre um lote representativo do release e olhe o **piso**. A tabela do
    diagnóstico (distribuição antiga de O, reproduzida aqui como referência de
    forma, não como alvo):

        θ=0,0 λ=2,0 → fracao 0,992  piso 0,806  contraste 3,00
        θ=0,3 λ=2,0 → fracao 0,167  piso 0,750  contraste 3,00
        θ=0,9 λ=1,0 → fracao 0,075  piso 0,930  contraste 2,00

    Returns
    -------
    dict
        fracao_tokens_pesados
            Fração de tokens com `O_token > θ` — os que recebem supervisão
            extra. NÃO é o número que decide: ver `piso`.
        piso
            MENOR peso normalizado do lote. É o que decide. Mede quanto a
            supervisão encolheu para o token que NÃO é borda. A campanha que
            piorou tinha piso 0,806, isto é, −18% de supervisão em 74% dos
            tokens. Forma fechada no modo binário: `1/(1+λ·fracao)`.
        contraste
            max(w)/min(w). No modo binário é exatamente `1+λ`. É a separação
            entre borda e não-borda — era ela que estava perdida quando ~100%
            dos tokens recebiam peso parecido.
        media_peso_sem_normalizar
            `1 + λ·fracao` no modo binário. O "orçamento" que a normalização
            conserva; explica por que concentrar afunda o piso.
        p50, p75, p90, p99
            Percentis de `O_token`. É onde se lê um θ candidato: a v1 tinha
            p50=0,019 / p75=0,123 / p90=0,639 / p99=1,000.
        fracao_pixels_acesos
            Fração de PIXELS com `O > θ` (o mesmo θ, para a razão abaixo ser
            comparável). Na v1, com θ=0,3, era 0,0121.
        fracao_tokens_O_positivo
            Fração de tokens com `O_token > 0`, INDEPENDENTE de θ. Na v1 era
            1,0000 — é ela, contra os 1,2% de pixels, que dá o 82,7× histórico.
        espalhamento
            fracao_tokens_pesados / fracao_pixels_acesos, com o MESMO θ dos dois
            lados. Quanto o max-pool 16× infla a região. `inf` se nenhum pixel
            passa do limiar mas algum token passa.
        espalhamento_historico
            fracao_tokens_O_positivo / fracao_pixels_acesos — a razão exata do
            diagnóstico antigo (82,7× com θ=0,3 em pixel e 0 em token). Só para
            comparar com o número registrado; a de cima é a honesta.
        theta, lambda_o, pool
            Ecoados, para o dict poder ser serializado como linha de tabela.
    """
    o_tok = _o_por_token(
        O_pixel, pack_latents_fn, pool=pool, latent_channels=latent_channels
    )
    w = occlusion_to_token_weight(
        O_pixel, pack_latents_fn, lambda_o=lambda_o, theta=theta, pool=pool,
        normalizar=True, latent_channels=latent_channels,
    )

    plano = o_tok.reshape(-1)
    quantis = torch.quantile(
        plano.double(), torch.tensor([0.50, 0.75, 0.90, 0.99], dtype=torch.float64)
    )
    frac_tok = float((o_tok > theta).double().mean())
    frac_pix = float((O_pixel.float() > theta).double().mean())
    frac_tok_pos = float((o_tok > 0.0).double().mean())

    w_min, w_max = float(w.min()), float(w.max())
    return {
        "fracao_tokens_pesados": frac_tok,
        "piso": w_min,
        "contraste": (w_max / w_min) if w_min > 0 else float("inf"),
        "media_peso_sem_normalizar": 1.0 + float(lambda_o) * (
            frac_tok if theta > 0.0 else float(o_tok.double().mean())
        ),
        "p50": float(quantis[0]),
        "p75": float(quantis[1]),
        "p90": float(quantis[2]),
        "p99": float(quantis[3]),
        "fracao_pixels_acesos": frac_pix,
        "fracao_tokens_O_positivo": frac_tok_pos,
        "espalhamento": (frac_tok / frac_pix) if frac_pix > 0 else (
            float("inf") if frac_tok > 0 else float("nan")
        ),
        "espalhamento_historico": (frac_tok_pos / frac_pix) if frac_pix > 0 else (
            float("inf") if frac_tok_pos > 0 else float("nan")
        ),
        "theta": float(theta),
        "lambda_o": float(lambda_o),
        "pool": pool,
    }


def weighted_flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE de flow matching, opcionalmente ponderada por token.

    Com `weight=None` isto é **BIT A BIT** `F.mse_loss(prediction.float(),
    target.float())` — a perda atual de `genfocus_train/models.py`. Não é
    elegância: é o que torna a condição de controle A' GRATUITA (mesmo código,
    mesmo caminho numérico) e o que permite afirmar, sem ressalva, que λ=0
    reproduz a linha de base. Há teste com `torch.equal`, não `allclose`.

    O fp32 é o mesmo da perda atual, pelo mesmo motivo: o treino roda em bf16 e
    a soma dos quadrados em bf16 perde dígitos.
    """
    pred = prediction.float()
    tgt = target.float()
    if weight is None:
        return F.mse_loss(pred, tgt)
    if weight.dim() != 3 or weight.shape[0] != pred.shape[0] or weight.shape[1] != pred.shape[1]:
        raise ValueError(
            f"weight deve ser (B, N, 1) casando com prediction {tuple(pred.shape)}; "
            f"recebido {tuple(weight.shape)}"
        )
    # mean() sobre (B,N,D) com w broadcast em D: é a média ponderada com o mesmo
    # denominador da mse_loss (B·N·D). Como mean(w)=1 quando normalizado, a
    # ESCALA da perda é preservada — só a distribuição entre tokens muda.
    return (weight.float() * (pred - tgt) ** 2).mean()
