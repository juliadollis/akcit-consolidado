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

NOTA SOBRE O BOKEH NESTE COMMIT:
  O BokehNet abaixo está IMPLEMENTADO mas NÃO TESTADO ainda nesta refatoração.
  Foco do Sprint atual = DeblurNet. Quando começar BokehNet, validar:
    1. Como o paper preprocessa D_def antes do VAE (range, channels)
    2. Onde entra a `aperture_shape` para shape-aware variant
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FluxBackbone, TextEmbeddings
from .math_utils import compute_defocus_map, normalize_defocus_condition


# =============================================================================
# Tipos auxiliares
# =============================================================================

@dataclass
class TrainBatchOutputs:
    """Output de make_train_batch — pronto para a loss."""
    prediction: torch.Tensor    # (B, N, D) — velocidade predita
    target: torch.Tensor        # (B, N, D) — velocidade alvo (ε - x_0)


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
    Stage 2 do GenRefocus.

    NOTA: implementado mas NÃO TESTADO neste commit. Validar antes de usar:
      - Range/normalização do defocus map antes do VAE encode
      - Como a `aperture_shape` é apendada ao sequence (paper §3.3)
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
        self.max_coc = max_coc

    def make_train_batch(
        self,
        aif_image: torch.Tensor,        # (B, 3, H, W) em [-1, 1]
        target_image: torch.Tensor,     # (B, 3, H, W) em [-1, 1] — bokeh GT
        depth_map: torch.Tensor,        # (B, H, W) — disparity (1/depth)
        focus_plane: torch.Tensor,      # (B,) — disparity do plano de foco
        k: torch.Tensor,                # (B,) — bokeh level K
    ) -> TrainBatchOutputs:
        """
        Forward com 2 condições: AIF + defocus map.

        D_def = K * |D - D_focus|  (paper Eq. 2)
        Normalizado para [0, 1] dividindo por max_coc, expandido para 3 canais
        (mesma convenção do Inference_bokehNet.py linha 140).
        """
        # AIF como tokens (mesma normalização do Deblur)
        aif_tokens, aif_ids = self.backbone.encode_image_to_tokens(aif_image)

        # Target (bokeh GT) — para extrair o "x_0 limpo" do flow
        target_tokens, target_ids = self.backbone.encode_image_to_tokens(target_image)

        # Defocus map: B,H,W → B,3,H,W em [0, 1] → reescala para [-1, 1] antes do VAE
        # (VAE do FLUX foi treinado com input em [-1, 1])
        defocus = compute_defocus_map(depth_map, focus_plane, k)              # (B, H, W)
        defocus_3ch = normalize_defocus_condition(defocus, self.max_coc)      # (B, 3, H, W) em [0, 1]
        defocus_input = defocus_3ch * 2.0 - 1.0                               # (B, 3, H, W) em [-1, 1]
        defocus_tokens, defocus_ids = self.backbone.encode_image_to_tokens(defocus_input)

        pred, target = self.backbone.forward_train_step(
            clean_tokens=target_tokens,
            clean_ids=target_ids,
            condition_tokens_list=[aif_tokens, defocus_tokens],
            condition_ids_list=[aif_ids, defocus_ids],
            text_embeddings=self.text_embeddings,
        )
        return TrainBatchOutputs(prediction=pred, target=target)


# =============================================================================
# Loss
# =============================================================================

def flow_matching_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss MSE em fp32 (estabilidade numérica) entre velocidade predita e alvo.
    Esta é a loss padrão de rectified flow / flow matching.
    """
    return F.mse_loss(prediction.float(), target.float())


# =============================================================================
# Utilidades
# =============================================================================

def freeze_module(module: nn.Module) -> None:
    """Congela todos os parâmetros do módulo."""
    for p in module.parameters():
        p.requires_grad = False
