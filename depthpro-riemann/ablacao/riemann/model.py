"""
riemann/model.py
================
Wrapper do DepthPro para fine-tuning com duas variantes de congelamento:

  Variante A — "heads": congela o encoder ViT; treina só as cabeças de decodificação.
  Variante B — "heads_lora": além das cabeças, insere LoRA (rank baixo) nas camadas de
               atenção do backbone.

A ideia: a geometria fina das bordas (onde a curvatura Gaussiana atua) é produzida nas
cabeças de decodificação. Congelar o encoder reduz esquecimento catastrófico do
conhecimento métrico do DepthPro e cabe folgado numa A100.

Requer o pacote oficial `depth_pro` (github.com/apple/ml-depth-pro) e os pesos
`depth_pro.pt`. Se ausente, este módulo lança erro instrutivo.
"""

from __future__ import annotations
from dataclasses import replace
from typing import List
import torch
import torch.nn as nn


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


class LoRALinear(nn.Module):
    """
    Adaptador LoRA para uma camada Linear: W' = W + (alpha/r) · B·A.
    W permanece congelado; só A e B treinam.
    """
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        _freeze(self.base)
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A), self.B) * self.scale


def _inject_lora(model: nn.Module, r: int = 16, alpha: int = 32,
                 target_substr=("qkv", "proj", "attn")) -> int:
    """
    Substitui camadas Linear cujo nome contenha algum de target_substr por LoRALinear.
    Retorna o número de camadas adaptadas.
    """
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and any(s in name.lower() for s in target_substr):
            setattr(model, name, LoRALinear(child, r, alpha))
            count += 1
        else:
            count += _inject_lora(child, r, alpha, target_substr)
    return count


def _inverse_to_depth(inv: torch.Tensor) -> torch.Tensor:
    # Recíproco em float32 (1/x estoura o fp16 do autocast).
    #
    # PISO SOFTPLUS em vez de clamp duro (fix e62c4a6, validado na B200):
    # o clamp(min=1e-4) criava um ATRATOR DEGENERADO — pixels saturados viravam
    # profundidade ~1e4 m com GRADIENTE ZERO. Combinado com o alinhamento afim, uma
    # predição constante vira "prever a média do GT", um mínimo local absorvente (sem
    # gradiente para escapar), e o treino colapsava. O softplus dá um piso suave com
    # gradiente VIVO: teto ~100 m, desvio < 1e-3 para profundidades <= 50 m.
    inv = inv.float()
    inv_floor = 1e-2 + torch.nn.functional.softplus(inv - 1e-2, beta=200.0)
    return 1.0 / inv_floor


class DepthProRiemann(nn.Module):
    """
    Wrapper de fine-tuning do DepthPro.

    Args:
        checkpoint: caminho para depth_pro.pt.
        variant: "heads" (só cabeças) ou "heads_lora" (cabeças + LoRA no backbone).
        lora_rank, lora_alpha: hiperparâmetros do LoRA (variante B).
        device: "cuda"/"cpu".

    forward(image) -> depth normalizado [0,1], shape (B,1,H,W).
    """
    def __init__(self, checkpoint: str, variant: str = "heads",
                 lora_rank: int = 16, lora_alpha: int = 32,
                 device: str = "cuda", grad_checkpointing: bool = False):
        super().__init__()
        self.variant = variant
        self.device = device
        self.grad_checkpointing = grad_checkpointing
        self._build(checkpoint, lora_rank, lora_alpha)
        if grad_checkpointing:
            self._enable_grad_checkpointing()

    def _build(self, checkpoint, lora_rank, lora_alpha):
        try:
            import depth_pro
        except ImportError:
            raise ImportError(
                "Pacote 'depth_pro' não encontrado. Instale:\n"
                "  git clone https://github.com/apple/ml-depth-pro\n"
                "  pip install -e ml-depth-pro"
            )

        # Carregar arquitetura + pesos. O config default do depth_pro embute
        # checkpoint_uri="./checkpoints/depth_pro.pt" (relativo ao CWD) — apontamos
        # para o checkpoint recebido, senão o load falha fora da raiz do repo.
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        config = DEFAULT_MONODEPTH_CONFIG_DICT
        if checkpoint:
            config = replace(config, checkpoint_uri=checkpoint)
        model, _ = depth_pro.create_model_and_transforms(config=config,
                                                         device=self.device)

        self.core = model

        # --- Congelamento conforme a variante ---
        _freeze(self.core)  # congela tudo primeiro

        # Escopo do descongelamento, por variante:
        #   heads       -> decoder DPT completo + cabeça de FOV (~342M params)
        #   heads_final -> apenas a cabeça de profundidade (~491K params)
        #   heads_lora  -> como heads, mais adaptadores LoRA no backbone
        #
        # Por que heads_final existe: a cabeça de FOV sozinha responde por ~304M dos
        # 342M da variante heads (89%) e é irrelevante para a borda. Além disso, o paper
        # do DepthPro treina a cabeça de FOV SEPARADAMENTE, sobre features congeladas da
        # rede de profundidade, e afirma que separar é melhor que treinar junto. Liberar
        # o FOV junto com a profundidade contraria esse desenho e degrada o AbsRel.
        if self.variant == "heads_final":
            head_substr = ("head",)
            excluir = ("fov",)
        else:
            head_substr = ("head", "decoder", "fov", "upsample", "final")
            excluir = ()

        for name, module in self.core.named_modules():
            n = name.lower()
            if any(s in n for s in head_substr) and not any(e in n for e in excluir):
                _unfreeze(module)
        # garantir que nada excluído ficou treinável (submódulo pode ter sido liberado
        # por um pai cujo nome casou antes)
        for name, p in self.core.named_parameters():
            if any(e in name.lower() for e in excluir):
                p.requires_grad_(False)

        n_head_params = sum(p.numel() for p in self.core.parameters() if p.requires_grad)
        if n_head_params == 0:
            raise RuntimeError(
                f"Nenhuma cabeça de decodificação encontrada pelas substrings {head_substr}. "
                "Os nomes de módulo desta versão do DepthPro mudaram — inspecione "
                "model.named_modules() e ajuste head_substr.")
        self._n_head_params = n_head_params

        # Variante B: injetar LoRA no backbone
        self._n_lora = 0
        if self.variant == "heads_lora":
            self._n_lora = _inject_lora(self.core, lora_rank, lora_alpha)

        self.to(self.device)

    def _enable_grad_checkpointing(self):
        """
        Ativa gradient checkpointing quando disponível. Troca ~30% de tempo por uma
        grande economia de VRAM (recomputa ativações no backward). Essencial na 4090.

        Tenta a API padrão do módulo; se o backbone expuser .gradient_checkpointing_enable()
        (comum em ViTs estilo HF/timm) usa-a. Caso contrário, marca uma flag para o
        forward usar torch.utils.checkpoint nos blocos do encoder, se acessíveis.
        """
        enabled = False
        for _, module in self.core.named_modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable()
                    enabled = True
                except Exception:
                    pass
            # timm ViT expõe .grad_checkpointing = True
            if hasattr(module, "grad_checkpointing"):
                try:
                    module.grad_checkpointing = True
                    enabled = True
                except Exception:
                    pass
        if enabled:
            print("[Modelo] gradient checkpointing ATIVADO no backbone.")
        else:
            print("[Modelo] aviso: não foi possível ativar gradient checkpointing "
                  "automaticamente nesta versão do DepthPro. Reduza batch/resolução se "
                  "faltar VRAM.")

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B,3,H,W) já normalizada conforme o transform do DepthPro.
        Retorna depth normalizado [0,1] (B,1,H,W).
        """
        # A API do DepthPro expõe .infer para inferência métrica; para treino usamos
        # o forward do modelo diretamente para manter o grafo de gradiente.
        # O forward oficial exige entrada quadrada de img_size (1536) — replicamos
        # o redimensionamento que o infer() faz, ida e volta, sem sair do grafo.
        H_in, W_in = image.shape[-2:]
        tgt = self.core.img_size
        if (H_in, W_in) != (tgt, tgt):
            image = nn.functional.interpolate(
                image, size=(tgt, tgt), mode="bilinear", align_corners=False)
        out = self.core(image)
        # out varia com a versão do DepthPro. A API oficial retorna profundidade
        # INVERSA canônica — é preciso inverter antes de comparar com o GT de
        # profundidade, senão normais/curvatura treinam contra a superfície errada.
        if isinstance(out, dict):
            if "depth" in out:
                depth = out["depth"]
            elif "canonical_inverse_depth" in out:
                depth = _inverse_to_depth(out["canonical_inverse_depth"])
            else:
                raise RuntimeError(
                    f"Saída do DepthPro sem chave conhecida: {list(out.keys())}")
        elif isinstance(out, (tuple, list)):
            # forward oficial: (canonical_inverse_depth, fov_deg)
            depth = _inverse_to_depth(out[0])
        else:
            depth = out
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[-2:] != (H_in, W_in):
            depth = nn.functional.interpolate(
                depth, size=(H_in, W_in), mode="bilinear", align_corners=False)
        # NÃO normalizamos por min-max aqui. Min-max reescala o eixo z com fatores
        # diferentes entre pred e GT, e a curvatura Gaussiana NÃO é invariante a essa
        # reescala — isso deformava a gauss_loss (e fazia o termo explodir ~400x vs berHu).
        # Devolvemos profundidade em escala crua; o alinhamento pred<->GT é afim (escala+
        # shift por mínimos quadrados), feito uma única vez dentro da loss e das métricas.
        return depth


def build_model(checkpoint: str, variant: str = "heads",
                lora_rank: int = 16, lora_alpha: int = 32,
                device: str = "cuda", grad_checkpointing: bool = False) -> DepthProRiemann:
    m = DepthProRiemann(checkpoint, variant, lora_rank, lora_alpha, device,
                        grad_checkpointing)
    print(f"[Modelo] variante={variant} | params treináveis={m.count_trainable():,} "
          f"| grad_ckpt={grad_checkpointing}")
    return m
