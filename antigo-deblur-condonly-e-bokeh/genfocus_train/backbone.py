"""
Backbone FLUX para treino do GenRefocus.

Diferenças críticas vs versão anterior:
  - USA Genfocus.pipeline.flux.transformer_forward (o conditioning OminiControl-
    style implementado pelos autores do paper). NÃO inventamos um forward próprio.
  - USA FlowMatchEulerDiscreteScheduler do FLUX. NÃO usamos noise schedule DDPM.
  - LoRA injetado via add_adapter (preserva tipo FluxTransformer2DModel,
    permite enable_gradient_checkpointing nativo).
  - Sigma sampling logit-normal com shift dependente de seq_len (mesma fórmula
    da inferência, garantindo consistência treino↔inferência).
  - Sem fallbacks. Se algo não funciona, o erro é explícito.

Referências:
  - Paper §3.1: S_t = [X_t ; E(I_in)] — token concatenation
  - Paper §4.1: backbone FLUX-1-dev, LoRA rank 128 (Deblur) / 64 (Bokeh)
  - Genfocus/pipeline/flux.py: transformer_forward (linha 360), Condition (99),
    encode_images (38), specify_lora (145).
  - diffusers/examples/flux-control/train_control_lora_flux.py: padrão oficial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from peft import LoraConfig

# Importa a maquinaria do paper. Sem isso, não há reprodução.
# O repo oficial dos autores (Genfocus) NÃO faz parte deste pacote — precisa
# estar no PYTHONPATH. Veja scripts/setup_genfocus.sh e o README.
try:
    from Genfocus.pipeline.flux import transformer_forward
except ModuleNotFoundError as exc:  # pragma: no cover - depende de repo externo
    raise ModuleNotFoundError(
        "Não foi possível importar `Genfocus.pipeline.flux.transformer_forward`. "
        "Esse módulo vem do repositório oficial dos autores e precisa estar no "
        "PYTHONPATH. Rode `bash scripts/setup_genfocus.sh` (clona o repo) e/ou "
        "exporte `PYTHONPATH=$PWD/third_party/Genfocus:$PYTHONPATH`. "
        "Detalhes no README.md, seção 'Pré-requisitos'."
    ) from exc

from .config import ModelConfig


# =============================================================================
# Constantes
# =============================================================================

# Lista de módulos onde LoRA é injetado.
# DEVE BATER 1:1 com os módulos que `Genfocus.pipeline.flux` envolve em
# `specify_lora()`. Verifique:
#   flux.py linhas 209-212  → to_q, to_k, to_v
#           linha 264       → to_out[0]
#           linha 288       → norm1.linear
#           linha 318       → ff.net[2]
#           linha 341       → norm.linear (single block)
#           linha 343       → proj_mlp (single block)
#           linha 352       → proj_out (single block)
#           linha 385       → x_embedder
#
# IMPORTANTE: confirme essa lista inspecionando o .safetensors do paper antes
# do treino real. Comando sugerido:
#   from safetensors.torch import load_file
#   sd = load_file("deblurNet.safetensors")
#   modules = sorted({k.split(".lora_")[0].split(".")[-1] for k in sd})
#   print(modules)
LORA_TARGET_MODULES = [
    # Atenção (image branch)
    "to_q", "to_k", "to_v", "to_out.0",
    # Norm + MLP (dual blocks)
    "norm1.linear",
    "ff.net.2",
    # Norm + MLP (single blocks)
    "norm.linear",
    "proj_mlp",
    "proj_out",
    # Image embedder
    "x_embedder",
]

# Nome do adapter PEFT. Convenção do diffusers/peft.
ADAPTER_NAME = "default"


@dataclass
class TextEmbeddings:
    """Embeddings pré-computados de um prompt fixo. Imutáveis durante o treino."""

    prompt_embeds: torch.Tensor          # (1, seq, dim_text)
    pooled_prompt_embeds: torch.Tensor   # (1, 768)
    text_ids: torch.Tensor               # (seq, 3)


# =============================================================================
# FLUX Backbone
# =============================================================================

class FluxBackbone(nn.Module):
    """
    Backbone FLUX-1-dev + LoRA para treino do GenRefocus.

    Encapsula:
      - VAE (frozen)
      - Transformer DiT (frozen, com LoRA treinável)
      - Scheduler do FLUX (FlowMatchEulerDiscreteScheduler)
      - Encoder de prompt (text encoders são liberados após pré-computar)

    O método principal é `compute_loss(blurry_or_target_latents, condition_latents,
    text_embeddings)` que faz: amostra σ → adiciona ruído → forward → loss.
    """

    def __init__(self, model_config: ModelConfig, lora_rank: int):
        super().__init__()
        self.model_config = model_config
        self.lora_rank = lora_rank

        # ── Carrega pipeline COMPLETO temporariamente para encodar prompt ────
        # Estratégia: carrega FluxPipeline normalmente; encoda o prompt; depois
        # libera os text encoders e tokenizers (economia ~10GB VRAM).
        # Quem quiser passar pre-computed embeddings via TextEmbeddings pode
        # passar `text_embeddings` direto e pular essa fase em precompute_text().
        self._pipe: FluxPipeline | None = None
        self._scheduler_config = None
        self.vae: nn.Module | None = None
        self.transformer: nn.Module | None = None

        # Estado de inicialização (chamar .load() antes de usar)
        self._loaded = False

    # -------------------------------------------------------------------------
    # Carregamento
    # -------------------------------------------------------------------------

    def load(self, dtype: torch.dtype, device: torch.device):
        """
        Carrega FluxPipeline, congela VAE+text encoders, injeta LoRA no transformer.
        Idempotente — chamadas subsequentes são no-op.
        """
        if self._loaded:
            return

        # ── Desliga o backend cuDNN-SDPA (fused attention) ───────────────────
        # O attn_forward do Genfocus chama F.scaled_dot_product_attention com
        # query e key/value de SEQ-LENS DIFERENTES (concatena texto+main+cond),
        # e o FLUX usa head_dim=128. Esse formato faz o backend cuDNN escolher
        # um kernel que QUEBRA NO BACKWARD em H100/bf16, com:
        #   RuntimeError: Expected mha_graph->execute(...).is_good() ... got false
        # Desligar o cuDNN-SDPA força Flash/mem-efficient/math, que lidam com
        # esse formato sem erro. (Os outros backends seguem ligados.)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

        pipe = FluxPipeline.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)

        self._pipe = pipe
        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self._scheduler_config = pipe.scheduler.config

        # Congela VAE + text encoders (LoRA será adicionado depois ao transformer)
        self.vae.requires_grad_(False)
        self.vae.eval()
        if pipe.text_encoder is not None:
            pipe.text_encoder.requires_grad_(False)
            pipe.text_encoder.eval()
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.requires_grad_(False)
            pipe.text_encoder_2.eval()

        # Congela transformer ANTES do add_adapter
        self.transformer.requires_grad_(False)

        # Gradient checkpointing: ANTES do add_adapter, no objeto base.
        # `Genfocus.pipeline.flux.transformer_forward` checa
        # `self.training and self.gradient_checkpointing` (linhas 426 e 445)
        # para usar `torch.utils.checkpoint.checkpoint`.
        if self.model_config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # Injeta LoRA via add_adapter (NÃO get_peft_model — preserva o tipo)
        self._inject_lora()

        self._loaded = True

    def _inject_lora(self):
        """Adiciona LoRA ao transformer mantendo o tipo FluxTransformer2DModel."""
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        self.transformer.add_adapter(lora_config, adapter_name=ADAPTER_NAME)

        # Cast LoRA params para fp32 (estabilidade do AdamW).
        # Pesos do base ficam em bf16; só LoRA fica em fp32.
        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

        # Validação: se target_modules não bate com nada, treinaria 0 params.
        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        if n_trainable == 0:
            raise RuntimeError(
                "Nenhum parâmetro treinável após add_adapter. "
                "LORA_TARGET_MODULES não bate com os nomes reais do FluxTransformer2DModel. "
                "Inspecione com: print([n for n,_ in self.transformer.named_modules()][:50])"
            )

    # -------------------------------------------------------------------------
    # Pré-computação de prompt (chamado uma vez antes do loop de treino)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def precompute_text(self, prompt: str, device: torch.device, dtype: torch.dtype) -> TextEmbeddings:
        """
        Encoda um prompt fixo. Após esta chamada, é seguro chamar
        `release_text_encoders()` para liberar VRAM.
        """
        self._require_loaded()
        prompt_embeds, pooled_prompt_embeds, text_ids = self._pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        return TextEmbeddings(
            prompt_embeds=prompt_embeds.detach().clone().to(device=device, dtype=dtype),
            pooled_prompt_embeds=pooled_prompt_embeds.detach().clone().to(device=device, dtype=dtype),
            text_ids=text_ids.detach().clone().to(device=device, dtype=dtype),
        )

    def release_text_encoders(self):
        """Libera ~10GB de VRAM removendo text encoders e tokenizers."""
        self._require_loaded()
        if self._pipe is None:
            return
        self._pipe.text_encoder = None
        self._pipe.text_encoder_2 = None
        self._pipe.tokenizer = None
        self._pipe.tokenizer_2 = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Encoding de imagens → tokens FLUX
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def encode_image_to_tokens(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pipeline completo: imagem RGB em [-1, 1] → tokens FLUX + IDs de posição.

        Args:
            image: (B, 3, H, W) em [-1, 1]
        Returns:
            tokens: (B, N, D)  onde N = (H/16) * (W/16) (latente do VAE é H/8,
                    e _pack_latents reduz mais 2× em cada dim)
            ids:    (N, 3)
        """
        self._require_loaded()
        # VAE encode.
        # O VAE é carregado em bf16, mas o dataloader entrega imagens em float32.
        # No treino real (accelerate) o VAE roda FORA do autocast (só o
        # transformer é prepared), então sem este cast o conv_in estoura com:
        #   "Input type (float) and bias type (c10::BFloat16) should be the same".
        # Convertemos a imagem para o dtype/device dos pesos do VAE.
        vae_param = next(self.vae.parameters())
        image = image.to(device=vae_param.device, dtype=vae_param.dtype)
        latents = self.vae.encode(image).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # latents: (B, C_lat, H/8, W/8)

        # Pack para formato FLUX (B, N, D)
        B, C, H, W = latents.shape
        tokens = self._pipe._pack_latents(latents, B, C, H, W)  # (B, N, D)
        ids = self._pipe._prepare_latent_image_ids(
            B, H // 2, W // 2, latents.device, latents.dtype
        )  # (N, 3)
        return tokens, ids

    # -------------------------------------------------------------------------
    # Sigma sampling (rectified flow do FLUX)
    # -------------------------------------------------------------------------

    def sample_sigma(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Amostra σ ∈ (0, 1) seguindo a receita do FLUX:
          1. u ~ sigmoid(N(0, 1))  [logit-normal padrão]
          2. σ = (exp(μ) * u) / (1 + (exp(μ) - 1) * u)  [shift dep. seq_len]

        μ é o mesmo `calculate_shift` usado na inferência (Genfocus/pipeline/flux.py
        linha 625), garantindo distribuição idêntica entre treino e inferência.
        """
        u = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))

        cfg = self._scheduler_config
        mu = calculate_shift(
            seq_len,
            cfg.base_image_seq_len,
            cfg.max_image_seq_len,
            cfg.base_shift,
            cfg.max_shift,
        )
        exp_mu = math.exp(mu)
        sigma = (exp_mu * u) / (1.0 + (exp_mu - 1.0) * u)
        return sigma.to(dtype=dtype)

    # -------------------------------------------------------------------------
    # Forward de treino (uma única "denoising step")
    # -------------------------------------------------------------------------

    def forward_train_step(
        self,
        clean_tokens: torch.Tensor,
        clean_ids: torch.Tensor,
        condition_tokens_list: Sequence[torch.Tensor],
        condition_ids_list: Sequence[torch.Tensor],
        text_embeddings: TextEmbeddings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executa um step de treino:
          1. Amostra σ ~ logit-normal-shifted
          2. Constrói x_t = (1-σ) * x_0 + σ * ε
          3. target_v = ε - x_0
          4. Forward via Genfocus.pipeline.flux.transformer_forward com:
              branches = [text, main, *conditions]
              adapters = [None, None, ADAPTER_NAME, ...]   # LoRA SÓ nas condições
             (casa com a inferência oficial, que não passa main_adapter → None)
          5. Retorna (predição, target)

        Args:
            clean_tokens: (B, N, D) — tokens da imagem alvo (limpa)
            clean_ids: (N, 3) — IDs de posição da imagem alvo
            condition_tokens_list: lista de (B, N_i, D) — uma por condição
            condition_ids_list: lista de (N_i, 3) — IDs por condição
            text_embeddings: prompt pré-computado (compartilhado entre samples)

        Returns:
            (noise_pred, target_velocity), ambos (B, N, D)
        """
        self._require_loaded()
        device = clean_tokens.device
        dtype = clean_tokens.dtype
        B, N, D = clean_tokens.shape

        # ── Sigma sampling (inside no_grad — não há gradiente sobre σ) ────────
        with torch.no_grad():
            sigma_u = self.sample_sigma(B, N, device, dtype)   # (B,)
            sigma = sigma_u.view(B, 1, 1)                       # (B, 1, 1)

            # Rectified flow: x_t = (1-σ)·x_0 + σ·ε ; v* = ε - x_0
            noise = torch.randn_like(clean_tokens)
            noisy_tokens = (1.0 - sigma) * clean_tokens + sigma * noise
            target_velocity = noise - clean_tokens  # (B, N, D)

            # Timesteps por branch:
            #   - texto e main: σ (timestep atual de denoising)
            #   - cada condição: 0.0 (condições são "limpas")
            t_main = sigma_u
            t_cond = torch.zeros(B, device=device, dtype=dtype)

            # Guidance:
            # FLUX.1-dev é guidance-distilled (espera guidance scalar como input
            # do modelo). Convenção em training scripts oficiais (diffusers
            # train_control_lora_flux.py, flux-qlora): usar guidance=1.0.
            # Usar 3.5 (valor de inferência) atrapalha convergência.
            guidance_main = torch.ones(B, device=device, dtype=dtype)
            guidance_cond = torch.ones(B, device=device, dtype=dtype)

            # Expansão para o batch
            n_cond = len(condition_tokens_list)
            pe = text_embeddings.prompt_embeds.expand(B, -1, -1).contiguous()
            ppe = text_embeddings.pooled_prompt_embeds.expand(B, -1).contiguous()

        # ── Branches ──────────────────────────────────────────────────────────
        # ORDEM IMPORTA — flux.py espera [text_features..., image_features...]:
        #   text_features = [pe]
        #   image_features = [noisy_tokens, *condition_tokens_list]
        # ═══════════════════════════════════════════════════════════════════
        # VARIANTE "PAPER" (cond-only): LoRA SÓ na condição, igual ao design do
        # deblurNet.safetensors OFICIAL. adapters = [texto=None, main=None, cond=LoRA].
        #   - texto: None (base FLUX)
        #   - main:  None (base FLUX)   ← LoRA NÃO age no branch principal
        #   - cada cond: ADAPTER_NAME (LoRA ativa)
        #
        # ✅ Casa com a INFERÊNCIA OFICIAL: o Inference_deblurNet.py chama
        # generate() com main_adapter=None (LoRA só na cond). Logo, o modelo
        # treinado assim roda DIRETO na pipeline oficial — sem o main_adapter
        # custom que a variante main+cond (projeto original) exigia.
        # ═══════════════════════════════════════════════════════════════════
        n_branches = 1 + 1 + n_cond  # texto + main + n_conds
        adapters = [None, None] + [ADAPTER_NAME] * n_cond
        timesteps = [t_main, t_main] + [t_cond] * n_cond
        guidances = [guidance_main, guidance_main] + [guidance_cond] * n_cond
        pooled = [ppe] * n_branches
        img_ids = [clean_ids] + list(condition_ids_list)
        txt_ids = [text_embeddings.text_ids]
        image_features = [noisy_tokens] + list(condition_tokens_list)
        text_features = [pe]

        # group_mask: cada branch atende a quem?
        # CASA com Genfocus.pipeline.flux.generate() (linhas 655-658 do oficial):
        #   group_mask = ones(branch_n, branch_n)
        #   group_mask[2:, 2:] = diag([1]*n_conditions)
        # Isto é INCONDICIONAL na inferência oficial (não depende de kv_cache —
        # o comentário antigo aqui estava errado: a linha condicionada a
        # kv_cache é OUTRA, `group_mask[2:, :2] = False`, que fica comentada/
        # não usada). O efeito real: cada condição só atende a si mesma (+
        # texto+main), NUNCA a outra condição. Para o DeblurNet (1 condição)
        # isso é um no-op (diag de 1 elemento == ones). Para o BokehNet (2
        # condições: AIF e defocus map) isso PASSA A IMPORTAR — sem isto, o
        # treino deixava AIF e defocus se cross-atenderem, o que a inferência
        # oficial nunca faz. Verificado empiricamente rodando o attn_forward
        # oficial (não reimplementado) com as duas variantes de group_mask.
        group_mask = torch.ones(n_branches, n_branches, dtype=torch.bool, device=device)
        if n_cond > 1:
            cond_diag = torch.eye(n_cond, dtype=torch.bool, device=device)
            group_mask[2:, 2:] = cond_diag

        # ── Forward via transformer_forward CUSTOMIZADO do paper ──────────────
        # Retorna SOMENTE a predição do branch main (primeiros N tokens).
        # O gradiente flui pelo main + cross-attention para conds (LoRA recebe
        # gradiente em ambos).
        output = transformer_forward(
            self.transformer,
            image_features=image_features,
            text_features=text_features,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=timesteps,
            pooled_projections=pooled,
            guidances=guidances,
            adapters=adapters,
            group_mask=group_mask,
        )
        noise_pred = output[0]  # (B, N, D)

        if noise_pred.shape != target_velocity.shape:
            raise RuntimeError(
                f"Shape mismatch: pred={tuple(noise_pred.shape)} "
                f"vs target={tuple(target_velocity.shape)}. "
                "Algo está errado em transformer_forward."
            )

        return noise_pred, target_velocity

    # -------------------------------------------------------------------------
    # Utilidades
    # -------------------------------------------------------------------------

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Retorna lista de parâmetros treináveis (LoRA only)."""
        self._require_loaded()
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "FluxBackbone não foi carregado. Chame .load(dtype, device) primeiro."
            )

    @property
    def device(self) -> torch.device:
        self._require_loaded()
        return next(self.transformer.parameters()).device


# =============================================================================
# Factory
# =============================================================================

def create_backbone(model_config: ModelConfig, stage: str) -> FluxBackbone:
    """
    Cria o backbone correto para o estágio.

    stage in {"deblur", "bokeh"} — escolhe o LoRA rank do paper.
    """
    if stage == "deblur":
        rank = model_config.deblur_lora_rank
    elif stage == "bokeh":
        rank = model_config.bokeh_lora_rank
    else:
        raise ValueError(f"stage inválido: {stage} (esperado 'deblur' ou 'bokeh')")

    return FluxBackbone(model_config=model_config, lora_rank=rank)
