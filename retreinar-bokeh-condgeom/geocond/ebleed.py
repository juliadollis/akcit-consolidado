"""E_bleed — o erro CONCENTRADO na região de borda de profundidade (§5.4).

    E_bleed = (1/|B|) · Σ_{x∈B} ‖Î(x) − I(x)‖₁ ,      B = { x : O(x) > θ }

Por que não basta PSNR/SSIM/LPIPS
---------------------------------
A região de borda é ~1,2% dos pixels (medido). Qualquer métrica global dilui o
artefato de vazamento nessa proporção: uma melhora real aparece na terceira casa
decimal e some no ruído entre seeds. O documento de proposta coloca o E_bleed
como PRÉ-CONDIÇÃO da campanha justamente por isso.

E POR QUE ELE SOZINHO TAMBÉM NÃO BASTA
--------------------------------------
Porque um modelo que borra tudo baixa o E_bleed por acidente. Isto não é
hipótese: foi o que aconteceu com a condição A' da campanha v1. A razão
borda/fora MELHOROU (1,427 → 1,286) enquanto o erro ABSOLUTO piorou em toda
parte (E_fora +31%). Se a tabela só tivesse a coluna do E_bleed, a campanha
teria sido declarada um sucesso parcial. Por isso esta função devolve SEMPRE o
trio (e_bleed, e_fora, razao), e nunca só o primeiro.

TRÊS INVARIANTES DE MEDIÇÃO
---------------------------
1. **O mapa `O` é RECEBIDO, não recalculado.** Tem de ser o MESMO `O` e o MESMO
   `tau` do condicionamento. Se a métrica usasse outro limiar ou outro tau, ela
   mediria uma região DIFERENTE da que a perda ponderada supervisiona, e a
   comparação entre condições deixaria de ser sobre a mesma coisa. Por isso a
   assinatura pede `O_pixel` e não profundidade — não há como errar o tau aqui
   sem errá-lo também no treino.
2. **O `theta` é o MESMO da perda.** Mesmo motivo. Sem default: quem chama
   passa o θ que usou em `occlusion_to_token_weight`.
3. **Máscara de validade EROSIONADA**, no espírito do `boundary_fscore` de
   depth-riemannian: a fronteira de uma região inválida (sem profundidade,
   fora do campo, buraco de reprojeção) produz um degrau artificial em `O` e
   entraria como "borda". Erodir a validade antes de cortar B garante que a
   borda medida é borda de CENA, não borda de buraco.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["e_bleed", "erodir"]


def _para_numpy(x: Any, nome: str) -> np.ndarray:
    """Aceita np.ndarray ou tensor torch (CPU ou não). Sem importar torch."""
    if hasattr(x, "detach"):          # torch.Tensor, sem depender do import
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{nome} tem NaN/Inf; a média sairia NaN sem aviso")
    return a


def erodir(mascara: np.ndarray, raio: int) -> np.ndarray:
    """Erosão binária com elemento 3×3 (8-vizinhança), aplicada `raio` vezes.

    Em numpy puro, de propósito: `scipy` não é dependência deste pacote e uma
    erosão 3×3 é um min sobre os 8 deslocamentos.

    A borda da IMAGEM é tratada como VÁLIDA (padding por réplica), para que uma
    máscara toda-True continue toda-True. O que se quer erodir é a fronteira da
    região INVÁLIDA, não o quadro.
    """
    if raio <= 0:
        return mascara.astype(bool)
    m = mascara.astype(bool)
    for _ in range(int(raio)):
        p = np.pad(m, 1, mode="edge")
        acc = p[1:-1, 1:-1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                acc &= p[1 + dy: 1 + dy + m.shape[0], 1 + dx: 1 + dx + m.shape[1]]
        m = acc
    return m


def e_bleed(
    pred: Any,
    alvo: Any,
    O_pixel: Any,
    *,
    theta: float,
    valido: Any | None = None,
    raio_erosao: int = 1,
) -> dict:
    """E_bleed, E_fora e a razão entre eles.

    Parameters
    ----------
    pred, alvo
        (H, W, 3) em [0,1] — ou (H, W) para monocromático. O L1 é a média sobre
        os canais, como em ‖·‖₁ por pixel normalizada pelo nº de canais (assim o
        número é comparável entre mono e RGB).
    O_pixel
        (H, W) em [0,1] — o canal `O` da pilha, **o mesmo mapa e o mesmo tau do
        condicionamento**, já na resolução das imagens. Reamostrar aqui dentro
        deslocaria a borda; por isso uma resolução diferente é erro, não
        conveniência.
    theta
        B = {x : O(x) > θ}. Sem default: passe o MESMO θ da perda ponderada.
    valido
        (H, W) booleano, opcional. Pixels com profundidade válida. Quando None,
        tudo é válido (e a erosão então não remove nada, por causa do padding
        por réplica).
    raio_erosao
        Quantas vezes erodir a máscara de validade. 1 (padrão) remove a casca de
        1 px em volta de cada buraco — que é exatamente onde `O` inventa um
        degrau. Pixels removidos saem das DUAS regiões, não migram de uma para a
        outra.

    Returns
    -------
    dict
        e_bleed
            L1 médio DENTRO de B. Menor é melhor. `nan` se B é vazio — melhor
            um nan explícito do que um número que parece medida.
        e_fora
            L1 médio FORA de B (e dentro da validade). É a coluna que expõe a
            degradação global: sem ela, o A' teria passado por sucesso.
        razao
            e_bleed / e_fora. É o número RELATIVO (1,427 → 1,286 na v1); só
            significa alguma coisa lido JUNTO com e_fora.
        n_pixels_borda
            |B|. Se for 0 ou o total, o θ não serve para esta cena.
        frac_borda
            n_pixels_borda / n_pixels_validos. Na v1, com θ=0,3, era 0,0121.
        n_pixels_fora, n_pixels_validos
            Contagens para auditar quanto a erosão removeu.
    """
    p = _para_numpy(pred, "pred")
    a = _para_numpy(alvo, "alvo")
    O = _para_numpy(O_pixel, "O_pixel")

    if p.shape != a.shape:
        raise ValueError(f"pred {p.shape} e alvo {a.shape} têm shapes diferentes")
    if p.ndim not in (2, 3):
        raise ValueError(f"pred deve ser (H,W,3) ou (H,W); recebido {p.shape}")
    hw = p.shape[:2]

    if O.ndim == 3 and O.shape[-1] == 1:
        O = O[..., 0]
    if O.ndim == 3 and O.shape[0] == 1:          # (1,H,W) vindo de um tensor
        O = O[0]
    if O.shape != hw:
        raise ValueError(
            f"O_pixel {O.shape} não casa com a imagem {hw}; reamostre ANTES, "
            "para que a borda caia exatamente no mesmo lugar. Reamostrar aqui "
            "produziria um número plausível medindo outra região."
        )
    if not (0.0 <= theta < 1.0):
        raise ValueError(
            f"theta deve estar em [0,1); recebido {theta!r}. O teste é ESTRITO "
            "(O > θ), então θ >= 1 esvaziaria B sem avisar."
        )

    # L1 por pixel, média sobre canais.
    l1 = np.abs(p.astype(np.float64) - a.astype(np.float64))
    if l1.ndim == 3:
        l1 = l1.mean(axis=2)

    if valido is None:
        val = np.ones(hw, dtype=bool)
    else:
        val = _para_numpy(valido, "valido").astype(bool)
        if val.shape != hw:
            raise ValueError(f"valido {val.shape} não casa com a imagem {hw}")
    val = erodir(val, raio_erosao)

    borda = (O > theta) & val
    fora = (~(O > theta)) & val

    n_b = int(borda.sum())
    n_f = int(fora.sum())
    n_v = int(val.sum())

    dentro = float(l1[borda].mean()) if n_b else float("nan")
    afora = float(l1[fora].mean()) if n_f else float("nan")
    razao = (dentro / afora) if (n_b and n_f and afora > 0.0) else float("nan")

    return {
        "e_bleed": dentro,
        "e_fora": afora,
        "razao": razao,
        "n_pixels_borda": n_b,
        "frac_borda": (n_b / n_v) if n_v else float("nan"),
        "n_pixels_fora": n_f,
        "n_pixels_validos": n_v,
    }
