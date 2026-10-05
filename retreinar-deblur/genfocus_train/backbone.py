"""
Backbone FLUX para treino do GenRefocus.

ÁRVORE NOVA (retreinar-deblur/). O treino antigo segue intacto em
`genrefocus_deblurnet_paper/` e `genrefocus_deblurnet/`.

Decisões estruturais (inalteradas nesta revisão, todas conferidas contra o
código oficial dos autores):
  - USA `Genfocus.pipeline.flux.transformer_forward` (o conditioning
    OminiControl-style implementado pelos autores). NÃO reimplementamos forward.
  - Rectified flow do FLUX: x_t = (1-σ)·x₀ + σ·ε, alvo v* = ε - x₀.
  - LoRA injetado via `add_adapter` (preserva o tipo FluxTransformer2DModel e
    permite `enable_gradient_checkpointing` nativo).
  - Sem fallback silencioso. Se algo não bate, o erro é explícito.

MUDANÇAS DESTA REVISÃO (ver PLANO_CORRECOES_DEBLURNET.md e MUDANCAS_CODIGO.md):
  C1  `LORA_TARGET_MODULES` virou regex ancorado + trava de contagem (343).
      A lista de strings capturava a projeção final `transformer.proj_out`.
  C2  `guidance` do treino deixou de ser 1.0 hardcodado; vem da config, por
      estágio.
  C4  `sample_sigma` passou a receber `seq_len` POR AMOSTRA, e `forward_train_step`
      escolhe de onde ele vem (`crop` vs `full_image`) via `sigma_mu_source`.
  C8  `lora_on_main` configurável + `lora_info()` para o metadata do checkpoint.

Referências:
  - Paper §3.1: S_t = [X_t ; E(I_in)] — token concatenation.
  - Paper §4.1: backbone FLUX-1-dev, LoRA rank 128 (Deblur) / 64 (Bokeh).
  - Genfocus/pipeline/flux.py: transformer_forward (360), specify_lora (146),
    encode_images (38), Condition (99).
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
# O repo oficial dos autores vive em `third_party/Genfocus/` e precisa estar no
# PYTHONPATH (os .slurm exportam). NUNCA modifique aquela cópia: ela é a
# referência que permite comparar o nosso treino com a inferência oficial.
try:
    from Genfocus.pipeline.flux import transformer_forward
except ModuleNotFoundError as exc:  # pragma: no cover - depende de repo externo
    raise ModuleNotFoundError(
        "Não foi possível importar `Genfocus.pipeline.flux.transformer_forward`. "
        "Esse módulo vem do repositório oficial dos autores. Exporte "
        "`PYTHONPATH=$PWD/third_party/Genfocus:$PYTHONPATH` (os scripts em "
        "slurm/ já fazem isso). Detalhes no README.md."
    ) from exc

from .config import (
    ModelConfig,
    TrainConfig,
    lora_rank_for,
    stage_config,
    train_guidance_for,
)


# =============================================================================
# Alvos do LoRA  (C1)
# =============================================================================

# ── COMO ERA ATÉ ESTA REVISÃO (mantido só como documentação do defeito) ──────
# Uma LISTA de strings. PEFT, quando `target_modules` é uma lista, casa por
#     key == target  or  key.endswith("." + target)
# A projeção final do FluxTransformer2DModel chama-se literalmente `proj_out`
# no TOPO do módulo, então `key == "proj_out"` casava e o PEFT injetava LoRA
# também em `transformer.proj_out` — além dos 38 `single_transformer_blocks.N.proj_out`
# que de fato queríamos.
#
# Por que isso importa: em `Genfocus/pipeline/flux.py` a projeção final roda
#     452:  image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
#     453:  output = self.proj_out(image_hidden_states)
# FORA de qualquer `specify_lora`. Como o `specify_lora` é o único mecanismo que
# zera o `scaling` do adapter por branch, essa LoRA ficava ATIVA
# incondicionalmente — inclusive com `main_adapter=None`, que é o modo da
# inferência oficial. Ou seja: a variante "cond-only" não era cond-only, tinha
# um módulo treinável agindo direto na saída do branch principal.
#
# Conta (FluxTransformer2DModel do FLUX.1-dev: 19 blocos duplos + 38 single):
#     19 duplos × (to_q, to_k, to_v, to_out.0, norm1.linear, ff.net.2) = 114
#     38 single × (to_q, to_k, to_v, norm.linear, proj_mlp, proj_out)  = 228
#     x_embedder                                                       =   1
#                                                                total = 343  ← oficial
#     + transformer.proj_out (topo)                                    =   1
#                                                                total = 344  ← nosso, errado
# O 343 bate exato com o header do `bokehNet.safetensors` oficial (686 tensores).
_LORA_TARGET_MODULES_LEGADO = [
    "to_q", "to_k", "to_v", "to_out.0",
    "norm1.linear",
    "ff.net.2",
    "norm.linear",
    "proj_mlp",
    "proj_out",     # ← o culpado: casava também com a projeção final de topo
    "x_embedder",
]

# ── COMO É AGORA ─────────────────────────────────────────────────────────────
# Regex ANCORADO. Quando `target_modules` é uma `str`, PEFT usa
# `re.fullmatch(target_modules, key)`, então `^...$` garante que só os módulos
# DENTRO dos blocos casem. `x_embedder` continua sendo de topo de propósito:
# `flux.py:385` o envolve em `specify_lora`, então ele É controlado por branch.
#
# Os alvos correspondem 1:1 aos 7 pontos de `specify_lora` em flux.py:
#   209  to_q, to_k, to_v      (attn, duplos e single)
#   264  to_out[0]             (attn dos duplos; single é pre_only, não tem)
#   288  norm1.linear          (duplos)
#   318  ff.net[2]             (duplos)
#   341  norm.linear, proj_mlp (single)
#   352  proj_out              (single)
#   385  x_embedder            (topo, controlado)
# NÃO casam, e é intencional: `norm_out.linear`, `norm1_context.linear`,
# `ff_context.net.2`, `add_q_proj/add_k_proj/add_v_proj`, `to_add_out`
# (módulos do branch de TEXTO, que o oficial também não treina) e a projeção
# final `proj_out` de topo.
# Os dois tipos de bloco são descritos SEPARADAMENTE de propósito: eles não têm
# o mesmo conjunto de módulos, e um padrão único para os dois aceitaria nomes
# que não existem no FLUX.1-dev.
#   - duplo  (19×): to_q/to_k/to_v, to_out.0, norm1.linear, ff.net.2
#   - single (38×): to_q/to_k/to_v, norm.linear, proj_mlp, proj_out
#                   (NÃO tem to_out: o Attention do bloco single é pre_only=True)
# 19×6 + 38×6 + x_embedder = 343, exatamente os módulos do checkpoint oficial.
LORA_TARGET_MODULES = (
    r"^transformer_blocks\.\d+\."
    r"(attn\.(to_q|to_k|to_v|to_out\.0)|norm1\.linear|ff\.net\.2)$"
    r"|^single_transformer_blocks\.\d+\."
    r"(attn\.(to_q|to_k|to_v)|norm\.linear|proj_mlp|proj_out)$"
    r"|^x_embedder$"
)

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

    Encapsula VAE (congelado), transformer DiT (congelado, com LoRA treinável),
    a config do scheduler do FLUX e o encoder de prompt (liberado após
    pré-computar os embeddings).

    Os parâmetros que MUDAM O MODELO chegam explicitamente pelo construtor, não
    por leitura global de config: é o que permite rodar o fatorial do plano e o
    que o `lora_info()` grava no metadata do checkpoint.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        lora_rank: int,
        *,
        train_guidance: float,
        cond_guidance: float,
        lora_on_main: bool,
        sigma_mu_source: str,
    ):
        super().__init__()
        self.model_config = model_config
        self.lora_rank = lora_rank

        # ── C2 ──────────────────────────────────────────────────────────────
        self.train_guidance = float(train_guidance)
        self.cond_guidance = float(cond_guidance)
        # ── C8 ──────────────────────────────────────────────────────────────
        self.lora_on_main = bool(lora_on_main)
        # ── C4 ──────────────────────────────────────────────────────────────
        self.sigma_mu_source = str(sigma_mu_source)

        # `lora_on_text=True` não é suportado e nunca foi treinado assim. Se
        # alguém puser isso num YAML, tem que explodir e não ser ignorado em
        # silêncio (é a armadilha que o docstring de config.py descreve).
        if getattr(model_config, "lora_on_text", False):
            raise NotImplementedError(
                "model.lora_on_text=True não é suportado. O treino nunca põe LoRA "
                "no branch de texto, e a inferência oficial (`generate`) não "
                "consegue separar texto de main: ela monta "
                "`adapters = [main_adapter]*2 + c_adapters`, então ligar o texto "
                "exigiria o patch `text_adapter` do C6. Mantenha False."
            )

        self._pipe: FluxPipeline | None = None
        self._scheduler_config = None
        self.vae: nn.Module | None = None
        self.transformer: nn.Module | None = None
        self.n_lora_modules: int = 0

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

        # C4 — a nossa fórmula vetorizada do mu TEM que ser a mesma reta do
        # `calculate_shift` do diffusers. Se uma versão futura do diffusers
        # mudar a fórmula, isto falha alto em vez de divergir em silêncio.
        self._verificar_mu_bate_com_diffusers()

        self._loaded = True

    def _inject_lora(self):
        """Adiciona LoRA ao transformer mantendo o tipo FluxTransformer2DModel."""
        lora_config = LoraConfig(
            r=self.lora_rank,
            # alpha = r  →  scaling = alpha/r = 1.0, que é exatamente o valor
            # que `specify_lora` FORÇA na inferência oficial
            # (`module.scaling[adapter] = 1`). Mudar isto descasa treino de
            # inferência sem nenhum aviso.
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        self.transformer.add_adapter(lora_config, adapter_name=ADAPTER_NAME)

        # Cast LoRA params para fp32 (estabilidade do AdamW).
        # Pesos do base ficam em bf16; só LoRA fica em fp32. O cálculo continua
        # em bf16 dentro do autocast; isto é o padrão "master weights fp32".
        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

        # ── C1: trava de contagem ────────────────────────────────────────────
        from peft.tuners.lora import LoraLayer

        modulos = [
            nome for nome, mod in self.transformer.named_modules()
            if isinstance(mod, LoraLayer)
        ]
        self.n_lora_modules = len(modulos)
        esperado = int(getattr(self.model_config, "expected_lora_modules", 0) or 0)

        if esperado and self.n_lora_modules != esperado:
            de_topo = [n for n in modulos if "." not in n]
            raise RuntimeError(
                f"LoRA injetado em {self.n_lora_modules} módulos; esperado {esperado} "
                f"(o checkpoint oficial tem 343: 19×6 duplos + 38×6 single + x_embedder). "
                f"Módulos de TOPO encontrados (suspeitos — só `x_embedder` deveria "
                f"aparecer aqui): {de_topo}. "
                f"Se a diferença for intencional, ajuste `model.expected_lora_modules` "
                f"no YAML e registre o porquê em MUDANCAS_CODIGO.md."
            )

        # Se target_modules não bate com nada, treinaria 0 params.
        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        if n_trainable == 0:
            raise RuntimeError(
                "Nenhum parâmetro treinável após add_adapter. LORA_TARGET_MODULES "
                "(regex) não casou com nenhum módulo do FluxTransformer2DModel. "
                "Inspecione com: print([n for n,_ in self.transformer.named_modules()][:80])"
            )

        print(
            f"[lora] {self.n_lora_modules} módulos | rank={self.lora_rank} "
            f"alpha={self.lora_rank} | {n_trainable/1e6:.1f}M params treináveis | "
            f"variante={'main+cond' if self.lora_on_main else 'cond-only'}"
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
    # Sigma sampling (rectified flow do FLUX)   (C4)
    # -------------------------------------------------------------------------

    def _mu_de_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        """mu do shift dinâmico, vetorizado — a MESMA reta do `calculate_shift`.

        O `calculate_shift` do diffusers é escalar; aqui precisamos de um mu por
        amostra, porque com `sigma_mu_source="full_image"` cada imagem de origem
        tem um número de tokens diferente.
        """
        cfg = self._scheduler_config
        m = (cfg.max_shift - cfg.base_shift) / (
            cfg.max_image_seq_len - cfg.base_image_seq_len
        )
        b = cfg.base_shift - cfg.base_image_seq_len * m
        return seq_len.to(torch.float32) * m + b

    def _verificar_mu_bate_com_diffusers(self) -> None:
        """Confere `_mu_de_seq_len` contra o `calculate_shift` escalar do diffusers.

        Se uma versão do diffusers mudar a fórmula do shift, isto falha no load
        em vez de treinar com um cronograma de sigma silenciosamente diferente
        do da inferência.
        """
        cfg = self._scheduler_config
        for seq in (256, 672, 1024, 2752, 4096):
            nosso = float(self._mu_de_seq_len(torch.tensor([seq])).item())
            deles = float(
                calculate_shift(
                    seq,
                    cfg.base_image_seq_len,
                    cfg.max_image_seq_len,
                    cfg.base_shift,
                    cfg.max_shift,
                )
            )
            if not math.isclose(nosso, deles, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError(
                    "A fórmula vetorizada do mu divergiu do `calculate_shift` do "
                    f"diffusers em seq_len={seq}: nosso={nosso!r} diffusers={deles!r}. "
                    "O diffusers provavelmente mudou o shift; reveja "
                    "`_mu_de_seq_len` antes de treinar."
                )

    def sample_sigma(
        self,
        seq_len: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Amostra σ ∈ (0, 1) por amostra, seguindo a receita do FLUX:
          1. u ~ sigmoid(N(0, 1))                       [logit-normal padrão]
          2. σ = (exp(μ)·u) / (1 + (exp(μ) - 1)·u)      [shift dependente de seq_len]

        Args:
            seq_len: (B,) — nº de tokens que define o μ de CADA amostra.
                     Quem escolhe a semântica é `forward_train_step`, conforme
                     `self.sigma_mu_source`.
        Returns:
            (B,) em `dtype`.

        SOBRE A CORRESPONDÊNCIA COM A INFERÊNCIA — leia antes de mudar:
        o docstring anterior afirmava que esta fórmula "garante distribuição
        idêntica entre treino e inferência". Isso é FALSO, por dois motivos:

        (a) A inferência calcula o μ a partir de `image_seq_len =
            latents.shape[1]` da imagem INTEIRA (`flux.py:624`), ANTES do
            tiling — cada tile de 32×32 tokens herda o cronograma da imagem
            completa. O treino, antes desta revisão, usava sempre o seq_len do
            CROP. Para uma imagem 1024×688: μ da imagem = 0,9225 (exp 2,516)
            contra μ do crop 512² = 0,6300 (exp 1,878). `sigma_mu_source`
            existe justamente para tornar isso um EIXO DE EXPERIMENTO.

        (b) Mesmo casando o μ, treino e inferência não ficam "idênticos": o
            treino AMOSTRA de uma densidade sobre (0,1) e a inferência PERCORRE
            uma trajetória discreta de 28 passos. Como o modelo é condicionado
            em σ e o treino cobre (0,1) inteiro, um μ diferente é
            desbalanceamento de DENSIDADE, não fora-de-domínio. Que casar
            melhore é HIPÓTESE não medida — ver C4 no plano.
        """
        if seq_len.ndim != 1:
            raise ValueError(f"seq_len deve ser (B,), recebido {tuple(seq_len.shape)}")

        B = seq_len.shape[0]
        u = torch.sigmoid(torch.randn(B, device=device, dtype=torch.float32))

        exp_mu = torch.exp(self._mu_de_seq_len(seq_len.to(device)))
        sigma = (exp_mu * u) / (1.0 + (exp_mu - 1.0) * u)
        return sigma.to(dtype=dtype)

    def _seq_len_para_sigma(
        self,
        B: int,
        n_tokens_crop: int,
        full_seq_len: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Escolhe o seq_len que alimenta o μ, conforme `sigma_mu_source` (C4)."""
        if self.sigma_mu_source == "crop":
            # Comportamento anterior a esta revisão: μ do próprio crop.
            return torch.full((B,), n_tokens_crop, dtype=torch.long, device=device)

        if self.sigma_mu_source == "full_image":
            if full_seq_len is None:
                raise ValueError(
                    "sigma_mu_source='full_image' exige `full_seq_len` no batch, "
                    "mas veio None. O dataloader precisa emitir a chave "
                    "'full_seq_len' (nº de tokens da imagem de origem inteira, "
                    "alinhada a 16). Sem fallback: um μ silenciosamente errado "
                    "invalidaria o braço do fatorial."
                )
            sl = full_seq_len.reshape(-1).to(device=device, dtype=torch.long)
            if sl.shape[0] != B:
                raise ValueError(
                    f"full_seq_len tem batch {sl.shape[0]}, esperado {B}."
                )
            if bool((sl <= 0).any()):
                raise ValueError(f"full_seq_len tem valor não-positivo: {sl.tolist()}")
            return sl

        raise ValueError(
            f"sigma_mu_source={self.sigma_mu_source!r} inválido "
            "(esperado 'crop' ou 'full_image')."
        )

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
        full_seq_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executa um step de treino:
          1. Amostra σ ~ logit-normal deslocada (μ conforme `sigma_mu_source`)
          2. Constrói x_t = (1-σ)·x₀ + σ·ε
          3. target_v = ε - x₀
          4. Forward via `Genfocus.pipeline.flux.transformer_forward`
          5. Retorna (predição do branch main, target)

        Args:
            clean_tokens: (B, N, D) — tokens da imagem alvo (limpa)
            clean_ids: (N, 3) — IDs de posição da imagem alvo
            condition_tokens_list: lista de (B, N_i, D) — uma por condição
            condition_ids_list: lista de (N_i, 3) — IDs por condição
            text_embeddings: prompt pré-computado (compartilhado entre samples)
            full_seq_len: (B,) long — nº de tokens da imagem de ORIGEM inteira.
                Obrigatório quando `sigma_mu_source="full_image"`; ignorado com
                `"crop"`.

        Returns:
            (noise_pred, target_velocity), ambos (B, N, D)
        """
        self._require_loaded()
        device = clean_tokens.device
        dtype = clean_tokens.dtype
        B, N, D = clean_tokens.shape

        # ── Sigma sampling (inside no_grad — não há gradiente sobre σ) ────────
        with torch.no_grad():
            seq_len_mu = self._seq_len_para_sigma(B, N, full_seq_len, device)
            sigma_u = self.sample_sigma(seq_len_mu, device, dtype)   # (B,)
            sigma = sigma_u.view(B, 1, 1)                            # (B, 1, 1)

            # Rectified flow: x_t = (1-σ)·x_0 + σ·ε ; v* = ε - x_0
            noise = torch.randn_like(clean_tokens)
            noisy_tokens = (1.0 - sigma) * clean_tokens + sigma * noise
            target_velocity = noise - clean_tokens  # (B, N, D)

            # Timesteps por branch:
            #   - texto e main: σ (timestep atual de denoising)
            #   - cada condição: 0.0 (condições são "limpas")
            # Bate com a inferência: `flux.py:614` faz
            # `c_timesteps.append(torch.zeros([1], device=device))`.
            t_main = sigma_u
            t_cond = torch.zeros(B, device=device, dtype=dtype)

            # ── Guidance (C2) ────────────────────────────────────────────────
            # O FLUX.1-dev é guidance-distilled: este escalar é ENTRADA do
            # modelo (vai para `time_text_embed(timestep, guidance, pooled)` e
            # produz o vetor de modulação `temb` daquele branch). NÃO é CFG.
            #
            # FATO conferido na inferência oficial — o valor difere por estágio:
            #   deblur: `Inference_deblurNet.py` e `demo.py` chamam `generate()`
            #           SEM `guidance_scale`, e o default é 3.5 (`flux.py:467`).
            #   bokeh:  `Inference_bokehNet.py` e `demo.py` passam
            #           `guidance_scale=1.0` explicitamente.
            #   condição: 1.0 nos dois casos (`flux.py:616`, `torch.ones`).
            #
            # O comentário que estava aqui afirmava que "usar 3.5 atrapalha
            # convergência". Isso NUNCA foi medido neste projeto — foi removido.
            #
            # HIPÓTESE não medida: que treinar com o mesmo valor da inferência é
            # melhor. O valor sai do experimento 2×2 (treino ∈ {1.0, 3.5} ×
            # inferência ∈ {1.0, 3.5}) descrito no C2 do plano. Por isso vem da
            # config e não é mais hardcodado.
            guidance_main = torch.full(
                (B,), self.train_guidance, device=device, dtype=dtype
            )
            guidance_cond = torch.full(
                (B,), self.cond_guidance, device=device, dtype=dtype
            )

            # Expansão para o batch
            n_cond = len(condition_tokens_list)
            pe = text_embeddings.prompt_embeds.expand(B, -1, -1).contiguous()
            ppe = text_embeddings.pooled_prompt_embeds.expand(B, -1).contiguous()

        # ── Branches ──────────────────────────────────────────────────────────
        # ORDEM IMPORTA — flux.py espera [text_features..., image_features...]:
        #   text_features  = [pe]
        #   image_features = [noisy_tokens, *condition_tokens_list]
        # e `adapters` é indexado na ordem [texto, main, *conds].
        #
        # ── C8: em qual branch a LoRA age ────────────────────────────────────
        #   texto: SEMPRE None. O treino nunca pôs LoRA no texto.
        #          ATENÇÃO: a inferência oficial NÃO consegue reproduzir isso
        #          quando se liga o main, porque `generate` monta
        #          `adapters = [main_adapter]*2 + c_adapters` — o índice 0 é o
        #          texto. Com `main_adapter="deblurring"` o texto ganha LoRA em
        #          38 single blocks, coisa que o treino nunca fez. É o C6 do
        #          plano; a correção é o parâmetro `text_adapter` na NOSSA cópia
        #          do `generate`, não aqui.
        #   main:  ADAPTER_NAME só se `lora_on_main`. O default False reproduz a
        #          inferência oficial, que usa `main_adapter=None` (cond-only).
        #   conds: SEMPRE ADAPTER_NAME.
        main_adapter = ADAPTER_NAME if self.lora_on_main else None
        n_branches = 1 + 1 + n_cond  # texto + main + n_conds
        adapters = [None, main_adapter] + [ADAPTER_NAME] * n_cond
        timesteps = [t_main, t_main] + [t_cond] * n_cond
        guidances = [guidance_main, guidance_main] + [guidance_cond] * n_cond
        pooled = [ppe] * n_branches
        img_ids = [clean_ids] + list(condition_ids_list)
        txt_ids = [text_embeddings.text_ids]
        image_features = [noisy_tokens] + list(condition_tokens_list)
        text_features = [pe]

        # group_mask: cada branch atende a quem?
        # CASA com `Genfocus.pipeline.flux.generate()`:
        #   group_mask = ones(branch_n, branch_n)
        #   group_mask[2:, 2:] = diag([1]*n_conditions)
        # Isto é INCONDICIONAL na inferência oficial (a linha condicionada a
        # kv_cache é OUTRA, `group_mask[2:, :2] = False`, que fica comentada).
        # Efeito: cada condição só atende a si mesma (+ texto + main), NUNCA a
        # outra condição. Para o DeblurNet (1 condição) é no-op. Para o BokehNet
        # (2 condições: AIF e defocus map) importa: sem isto o treino deixaria
        # AIF e defocus se cross-atenderem, o que a inferência nunca faz.
        group_mask = torch.ones(n_branches, n_branches, dtype=torch.bool, device=device)
        if n_cond > 1:
            cond_diag = torch.eye(n_cond, dtype=torch.bool, device=device)
            group_mask[2:, 2:] = cond_diag

        # ── Forward via transformer_forward CUSTOMIZADO do paper ──────────────
        # Retorna SOMENTE a predição do branch main. O gradiente flui pelo main
        # e pela cross-attention para as condições (é por ali que a LoRA
        # cond-only recebe gradiente).
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

    def lora_info(self) -> dict:
        """Tudo que MUDA O MODELO e precisa ser reproduzido na inferência (C8).

        O trainer grava isto no metadata do checkpoint e num `.json` ao lado do
        `.safetensors` exportado. Sem isso não dá para saber, olhando um
        artefato, se ele precisa de `main_adapter="deblurring"` na inferência —
        foi exatamente essa ambiguidade que produziu o bug da saída lavada.
        """
        return {
            "n_lora_modules": int(self.n_lora_modules),
            "target_pattern": LORA_TARGET_MODULES,
            "rank": int(self.lora_rank),
            "alpha": int(self.lora_rank),   # alpha = r  →  scaling 1.0
            "adapter_variant": "main+cond" if self.lora_on_main else "cond-only",
            "text_adapter": None,           # o treino NUNCA põe LoRA no texto
            "train_guidance": float(self.train_guidance),
            "cond_guidance": float(self.cond_guidance),
            "sigma_mu_source": str(self.sigma_mu_source),
        }

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

def create_backbone(config: TrainConfig, stage: str) -> FluxBackbone:
    """
    Cria o backbone do estágio, com TODO parâmetro que muda o modelo vindo da
    config — nada hardcodado. É o que permite rodar o fatorial do plano e o que
    o `lora_info()` grava no checkpoint.

    stage in {"deblur", "bokeh"}.
    """
    stage_cfg = stage_config(config, stage)
    return FluxBackbone(
        model_config=config.model,
        lora_rank=lora_rank_for(config, stage),
        train_guidance=train_guidance_for(config, stage),
        cond_guidance=config.model.cond_train_guidance,
        lora_on_main=config.model.lora_on_main,
        sigma_mu_source=stage_cfg.sigma_mu_source,
    )
