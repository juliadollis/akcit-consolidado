"""
Wrappers de treino para DeblurNet e BokehNet.

Este arquivo é fino e direto. A lógica pesada (forward via transformer_forward,
LoRA, sigma sampling) está em backbone.py. Aqui apenas:
  - Convertemos batches do dataset em tokens FLUX
  - Chamamos forward_train_step do backbone
  - Retornamos (prediction, target) prontos para a loss

DeblurNet (Stage 1, paper §3.1):
    Input:  blurry image
    Output: AIF image
    Conditioning: 1 condition (blurry encoded como tokens)
    Sequence: S_t = [X_t ; E(I_in)]

BokehNet (Stage 2, paper §3.2):
    Input:  AIF image + defocus map (D_def)
    Output: bokeh image
    Conditioning: 2 conditions (AIF tokens + D_def tokens)
    Sequence: S_t = [X_t ; E(I_aif) ; tokens(D_def)]

CONTRATO DO BOKEHNET (conferido linha a linha no Inference_bokehNet.py e no
Genfocus/pipeline/flux.py oficiais — o treino TEM que casar com isso):
  - 1 adapter, nome "bokeh"; `main_adapter` não é passado → None → LoRA ativo
    SÓ nos branches de condição. Bate com backbone.adapters = [None, None] + [A]*n_cond.
  - LoRA rank 64 (paper §4.1).
  - 2 condições, ambas adapter "bokeh":
      cond 1 = imagem AIF   → VAE em [-1, 1]  (No_preprocess=False)
      cond 2 = mapa defocus → VAE em [ 0, 1]  (No_preprocess=True — pula o *2-1)
  - defocus_map: usamos o PRÉ-COMPUTADO dos dfs (uint16 ÷ 65535 → [0,1]).
    A inferência tem que gerar o mapa com a MESMA normalização dos dados (ver
    nota de consistência no dataloader) — senão treina num range e infere noutro.
  - prompt: "an excellent photo with a large aperture"

NÃO IMPLEMENTADO: aperture-shape control (paper §3.3) — é um branch de condição
EXTRA, treinado depois com o LoRA base congelado. Fica para um segundo momento.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FluxBackbone, TextEmbeddings


# =============================================================================
# Tipos auxiliares
# =============================================================================

@dataclass
class TrainBatchOutputs:
    """Output de make_train_batch — pronto para a loss."""
    prediction: torch.Tensor    # (B, N, D) — velocidade predita
    target: torch.Tensor        # (B, N, D) — velocidade alvo (ε - x_0)
    weight: torch.Tensor | None = None
    """(B, N, 1) — peso por token da perda ponderada por oclusão, ou None.

    Viaja junto de prediction/target para que a perda não precise recalcular a
    redução 16× nem reempacotar. Ver geo_cond/loss_weight.py."""


# =============================================================================
# DeblurNet
# =============================================================================

class DeblurNet(nn.Module):
    """
    Stage 1 do GenRefocus.

    Forward de treino: dado (blurry, sharp), encoda ambos para tokens, monta
    o sequence [main, cond_blurry] e chama o backbone.
    """

    def __init__(self, backbone: FluxBackbone, text_embeddings: TextEmbeddings):
        super().__init__()
        self.backbone = backbone
        # Não registra como buffer — TextEmbeddings é dataclass, não nn.Module.
        # São tensores .detach()ed, sem gradiente.
        self.text_embeddings = text_embeddings

    def make_train_batch(
        self,
        blurry_image: torch.Tensor,   # (B, 3, H, W) em [-1, 1]
        sharp_image: torch.Tensor,    # (B, 3, H, W) em [-1, 1]
    ) -> TrainBatchOutputs:
        """Encoda imagens, faz forward, retorna pred + target."""
        # Encode (sem gradiente, dentro do backbone)
        sharp_tokens, sharp_ids = self.backbone.encode_image_to_tokens(sharp_image)
        blurry_tokens, blurry_ids = self.backbone.encode_image_to_tokens(blurry_image)

        # Forward de treino: 1 condição (blurry)
        pred, target = self.backbone.forward_train_step(
            clean_tokens=sharp_tokens,
            clean_ids=sharp_ids,
            condition_tokens_list=[blurry_tokens],
            condition_ids_list=[blurry_ids],
            text_embeddings=self.text_embeddings,
        )
        return TrainBatchOutputs(prediction=pred, target=target)


# =============================================================================
# BokehNet (esqueleto, não usado no sprint atual)
# =============================================================================

class BokehNet(nn.Module):
    """
    Stage 2 do GenRefocus: (AIF, D_def) → bokeh.

    Sequence: S_t = [X_t ; E(I_aif) ; E(D_def)]  — 2 condições.
    Ver o CONTRATO no topo do módulo: o range do D_def ([0,1], não [-1,1]) é
    load-bearing e casa com o No_preprocess=True da inferência oficial.
    """

    def __init__(
        self,
        backbone: FluxBackbone,
        text_embeddings: TextEmbeddings,
        max_coc: float = 100.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.text_embeddings = text_embeddings
        # max_coc NÃO é usado no treino (o defocus_map já vem pré-computado e
        # normalizado nos dfs). Guardamos só como referência: é a constante de
        # normalização que a INFERÊNCIA precisa usar para gerar um mapa no mesmo
        # range [0,1] que o modelo viu. Ver nota de consistência no dataloader.
        self.max_coc = max_coc

    def make_train_batch(
        self,
        aif_image: torch.Tensor,        # (B, 3, H, W) em [-1, 1]
        target_image: torch.Tensor,     # (B, 3, H, W) em [-1, 1] — bokeh GT
        defocus_map: torch.Tensor,      # (B, 3, H, W) em [ 0, 1] — PRÉ-COMPUTADO
        geo_map: torch.Tensor | None = None,   # (B, 6, H, W) em [0, 1]
        geo_branches: bool = False,
        occlusion_lambda: float = 0.0,
        occlusion_pool: str = "max",
        occlusion_theta: float = 0.0,
    ) -> TrainBatchOutputs:
        """
        Forward com 2 condições: AIF (em [-1,1]) + defocus map (em [0,1]).

        Usamos o `defocus_map` PRÉ-COMPUTADO dos dfs (coluna `defocus_map`, uint16,
        já dividido por 65535 no dataloader). Verificado que é um mapa de defocus
        genuíno: corr ~1.0 com |k·(disp − s1)|, disp = depth/65535, e ~0 no plano
        de foco. NÃO recalculamos nada aqui — usar o mapa exato que a geração de
        dados produziu é o mais fiel.

        RANGE (verificado no código oficial, load-bearing): a inferência monta

            cond_img = Condition(clean_input, "bokeh")                    # No_preprocess=False → VAE em [-1,1]
            cond_dmf = Condition(cond_map, "bokeh", [0,0], 1.0, No_preprocess=True)  # VAE em [0,1] CRU

        e `encode_images(..., No_preprocess=True)` pula o `image_processor.preprocess`
        (o passo [0,1]→[-1,1]). Logo o mapa entra no VAE em [0,1] e a AIF em [-1,1].
        Assimétrico de propósito. O treino replica isso: aif em [-1,1], defocus em [0,1].
        """
        # AIF: em [-1, 1] (equivale ao image_processor.preprocess do oficial).
        aif_tokens, aif_ids = self.backbone.encode_image_to_tokens(aif_image)

        # Target (bokeh GT) — para extrair o "x_0 limpo" do flow.
        target_tokens, target_ids = self.backbone.encode_image_to_tokens(target_image)

        # Defocus map: já em [0, 1], 3 canais → direto pro VAE, SEM reescalar.
        defocus_tokens, defocus_ids = self.backbone.encode_image_to_tokens(defocus_map)

        cond_tokens = [aif_tokens, defocus_tokens]
        cond_ids = [aif_ids, defocus_ids]

        if geo_map is not None and geo_branches:
            # CONDICIONAMENTO GEOMÉTRICO (geo_cond).
            #
            # Os 6 canais viram DOIS branches de condição de 3 canais cada, e não
            # um só, porque o VAE recebe imagens de 3 canais. O agrupamento não é
            # arbitrário: o `group_mask` (backbone.py:431-434) faz cada condição
            # atender só a si mesma, ao texto e ao branch principal, então os dois
            # branches geométricos são MUTUAMENTE CEGOS. O que precisa ser lido
            # junto no mesmo pixel tem de ficar no mesmo branch:
            #
            #   G1 = [ s, n_x, n_y ]   o termo de PRIMEIRA ORDEM completo
            #                          (magnitude E direção da inclinação)
            #   G2 = [ u, O, K~ ]      escala, visibilidade, segunda ordem
            #
            # A combinação entre G1 e G2 acontece no branch principal, que vê tudo.
            # A ordem dos canais vem de geo_cond.signals.CANAIS.
            if geo_map.shape[1] != 6:
                raise ValueError(
                    f"geo_map deve ter 6 canais na ordem de geo_cond.signals.CANAIS; "
                    f"recebido {tuple(geo_map.shape)}"
                )
            g1 = geo_map[:, [2, 3, 4]]   # s, n_x, n_y
            g2 = geo_map[:, [0, 1, 5]]   # u, O, K~
            for g in (g1, g2):
                # [0,1] CRU, como o mapa de defocus (No_preprocess=True).
                tok, ids = self.backbone.encode_image_to_tokens(g.contiguous())
                cond_tokens.append(tok)
                cond_ids.append(ids)

        pred, target = self.backbone.forward_train_step(
            clean_tokens=target_tokens,
            clean_ids=target_ids,
            condition_tokens_list=cond_tokens,
            condition_ids_list=cond_ids,
            text_embeddings=self.text_embeddings,
        )

        weight = None
        if occlusion_lambda > 0.0:
            if geo_map is None:
                raise ValueError(
                    "occlusion_lambda > 0 exige geo_map (o canal O vem dele). "
                    "Para a condição A' do plano, que aplica só a perda ponderada "
                    "sem canais novos, passe geo_map e deixe o branch desligado."
                )
            from geo_cond.loss_weight import occlusion_to_token_weight
            # canal 1 = O, ver geo_cond.signals.CANAIS
            weight = occlusion_to_token_weight(
                geo_map[:, 1:2], self.backbone._pipe._pack_latents,
                lambda_o=occlusion_lambda, pool=occlusion_pool,
                theta=occlusion_theta,
            ).to(pred.dtype)

        return TrainBatchOutputs(prediction=pred, target=target, weight=weight)


# =============================================================================
# Loss
# =============================================================================

def flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Loss MSE em fp32 (estabilidade numérica) entre velocidade predita e alvo.
    Esta é a loss padrão de rectified flow / flow matching.

    Com `weight` (B, N, 1), aplica a reponderação por oclusão do plano §5.3, no
    espaço de TOKEN. Com weight=None é bit a bit a perda anterior — há teste.

    Nota para o texto do paper: reponderar a perda de difusão de forma não
    uniforme enviesa o estimador do score. É prática padrão (min-SNR weighting e
    afins), mas o texto tem de dizer "reponderação perceptual da supervisão", não
    "perda ponderada da verossimilhança".
    """
    if weight is None:
        return F.mse_loss(prediction.float(), target.float())
    from geo_cond.loss_weight import weighted_flow_matching_loss
    return weighted_flow_matching_loss(prediction, target, weight)


# =============================================================================
# Utilidades
# =============================================================================

def freeze_module(module: nn.Module) -> None:
    """Congela todos os parâmetros do módulo."""
    for p in module.parameters():
        p.requires_grad = False
