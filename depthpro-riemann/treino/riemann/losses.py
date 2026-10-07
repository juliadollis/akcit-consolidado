"""
riemann/losses.py
=================
Riemannian-Aware Loss para fine-tuning do DepthPro.

Adaptada do trabalho anterior (FCRN + NYU) para o regime DepthPro + imagens de alta
qualidade. Combina até cinco termos ponderados:

    L_total = λ_berhu·L_berHu + λ_grad·L_grad + λ_normal·L_normal
              + λ_gauss·L_gauss + λ_geod·L_geod

Fundamentos geométricos:
  - O mapa de profundidade z = D(x,y) define uma superfície S = (x, y, D(x,y)) em R³.
  - Essa superfície induz um tensor métrico (primeira forma fundamental) g.
  - A curvatura Gaussiana K é um invariante INTRÍNSECO de 2ª ordem (Theorema Egregium):
    depende só de g, não do mergulho. Marca dobras/oclusões da superfície de profundidade.
  - Normais de superfície capturam orientação local (1ª–2ª ordem).

NOTA NUMÉRICA CRÍTICA: L_gauss é 2ª ordem → amplifica ruído. Por isso:
  (1) treinar sobre GT limpo (Hypersim), (2) suavização opcional antes das 2ª derivadas,
  (3) clamp dos valores extremos de curvatura.

Todas as perdas assumem tensores (B, 1, H, W) e usam uma máscara de validade opcional.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import surface_curvatures


# ---------------------------------------------------------------------------
# Utilitários de derivadas espaciais (Sobel/gradiente) com padding replicado
# ---------------------------------------------------------------------------

# Espaçamento físico entre pixels no domínio da imagem.
# As derivadas por diferença finita dão dz/d(pixel); a geometria (curvatura, normais)
# precisa de dz/d(coordenada física). Sem corrigir por h, a curvatura fica com escala
# errada (validado: esfera dá K=0 com h=1, e K=1/R² correto com h físico).
# Padrão: coordenadas normalizadas onde a MAIOR dimensão da imagem mapeia para [0,1],
# i.e. h = 1/max(H,W). Isso é escala-invariante e estável entre resoluções.

def _default_spacing(t: torch.Tensor) -> float:
    _, _, H, W = t.shape
    return 1.0 / max(H, W)


def _grad_xy(t: torch.Tensor, h: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradiente espacial (∂/∂x, ∂/∂y) por diferenças centrais, padding replicado.
    h = espaçamento físico por pixel. t: (B,1,H,W) -> (gx, gy).
    """
    if h is None:
        h = _default_spacing(t)
    tp = F.pad(t, (1, 1, 1, 1), mode="replicate")
    gx = (tp[:, :, 1:-1, 2:] - tp[:, :, 1:-1, :-2]) * (0.5 / h)
    gy = (tp[:, :, 2:, 1:-1] - tp[:, :, :-2, 1:-1]) * (0.5 / h)
    return gx, gy


def _second_derivs(t: torch.Tensor, h: Optional[float] = None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Segundas derivadas (z_xx, z_yy, z_xy) por diferenças finitas, escaladas por 1/h².
    """
    if h is None:
        h = _default_spacing(t)
    h2 = h * h
    tp = F.pad(t, (1, 1, 1, 1), mode="replicate")
    z_xx = (tp[:, :, 1:-1, 2:] - 2 * t + tp[:, :, 1:-1, :-2]) / h2
    z_yy = (tp[:, :, 2:, 1:-1] - 2 * t + tp[:, :, :-2, 1:-1]) / h2
    tpp = F.pad(t, (1, 1, 1, 1), mode="replicate")
    z_xy = (tpp[:, :, 2:, 2:] - tpp[:, :, 2:, :-2]
            - tpp[:, :, :-2, 2:] + tpp[:, :, :-2, :-2]) * (0.25 / h2)
    return z_xx, z_yy, z_xy


def _smooth(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Suavização Gaussiana leve (kernel separável 5x5) p/ estabilizar 2ª ordem."""
    if sigma <= 0:
        return t
    radius = 2
    xs = torch.arange(-radius, radius + 1, device=t.device, dtype=t.dtype)
    k1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    tp = F.pad(t, (radius, radius, radius, radius), mode="replicate")
    tp = F.conv2d(tp, kx)
    tp = F.conv2d(tp, ky)
    return tp


# ---------------------------------------------------------------------------
# Termos individuais da loss
# ---------------------------------------------------------------------------

def berhu_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None, c_frac: float = 0.2) -> torch.Tensor:
    """
    berHu (reverse Huber): L2 para erros grandes, L1 para pequenos.
    c = c_frac * max|pred-target| (por batch). Baseline Euclidiano.
    """
    diff = pred - target
    if mask is not None:
        diff = diff * mask
    absd = diff.abs()
    c = c_frac * absd.max().clamp(min=1e-6)
    l1 = absd
    l2 = (diff ** 2 + c ** 2) / (2 * c)
    loss = torch.where(absd <= c, l1, l2)
    if mask is not None:
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def grad_loss(pred: torch.Tensor, target: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Consistência de gradiente de 1ª ordem (nitidez de borda)."""
    pgx, pgy = _grad_xy(pred)
    tgx, tgy = _grad_xy(target)
    d = (pgx - tgx).abs() + (pgy - tgy).abs()
    if mask is not None:
        return (d * mask).sum() / mask.sum().clamp(min=1.0)
    return d.mean()


def _surface_normals(depth: torch.Tensor, h: Optional[float] = None) -> torch.Tensor:
    """
    Normais de superfície de S=(x,y,z). n = (-z_x, -z_y, 1) / ||.||.
    Retorna (B,3,H,W).
    """
    zx, zy = _grad_xy(depth, h)
    ones = torch.ones_like(depth)
    n = torch.cat([-zx, -zy, ones], dim=1)
    n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    return n


def normal_loss(pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                h: Optional[float] = None) -> torch.Tensor:
    """Consistência de normais: 1 - cos(θ) entre normais preditas e do GT."""
    np_ = _surface_normals(pred, h)
    nt = _surface_normals(target, h)
    cos = (np_ * nt).sum(dim=1, keepdim=True).clamp(-1, 1)
    d = 1.0 - cos
    if mask is not None:
        return (d * mask).sum() / mask.sum().clamp(min=1.0)
    return d.mean()


def gaussian_curvature(depth: torch.Tensor, smooth_sigma: float = 0.0,
                       clamp_val: float = 50.0, h: Optional[float] = None) -> torch.Tensor:
    """
    Curvatura Gaussiana de S=(x,y,D(x,y)):

        K = (z_xx·z_yy − z_xy²) / (1 + z_x² + z_y²)²

    Invariante intrínseco de 2ª ordem (Theorema Egregium). clamp para estabilidade.
    h = espaçamento físico por pixel (default: 1/max(H,W), coords normalizadas).
    Validado: esfera de raio R -> K = 1/R²; plano e cilindro -> K = 0.
    """
    z = _smooth(depth, smooth_sigma) if smooth_sigma > 0 else depth
    zx, zy = _grad_xy(z, h)
    zxx, zyy, zxy = _second_derivs(z, h)
    num = zxx * zyy - zxy ** 2
    den = (1.0 + zx ** 2 + zy ** 2) ** 2
    K = num / (den + 1e-6)
    return K.clamp(-clamp_val, clamp_val)


def gauss_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               smooth_sigma: float = 0.5, clamp_val: float = 50.0,
               h: Optional[float] = None) -> torch.Tensor:
    """Consistência de curvatura Gaussiana (o termo diferencial do método)."""
    kp = gaussian_curvature(pred, smooth_sigma, clamp_val, h)
    kt = gaussian_curvature(target, smooth_sigma, clamp_val, h)
    d = (kp - kt).abs()
    if mask is not None:
        return (d * mask).sum() / mask.sum().clamp(min=1.0)
    return d.mean()


def gauss_loss_metrica(pred: torch.Tensor, target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       fx: float = 0.0, fy: Optional[float] = None,
                       smooth_sigma: float = 0.5,
                       clamp_val: Optional[float] = None) -> torch.Tensor:
    """Consistência de curvatura Gaussiana MÉTRICA (retroprojeção 3D real).

    Substitui gauss_loss: usa a K física da superfície S=((u-cx)D/fx,(v-cy)D/fy,D)
    (ver riemann/geometry.py) em vez do gráfico em coordenadas de imagem. Requer a
    focal em pixels na resolução de trabalho. clamp_val em 1/m²."""
    if not fx or fx <= 0:
        raise ValueError("gauss_loss_metrica requer fx>0 (focal em px da resolução de trabalho)")
    kp = surface_curvatures(pred, fx=fx, fy=fy, smooth_sigma=smooth_sigma, clamp_val=clamp_val)["K"]
    kt = surface_curvatures(target, fx=fx, fy=fy, smooth_sigma=smooth_sigma, clamp_val=clamp_val)["K"]
    d = (kp - kt).abs()
    if mask is not None:
        return (d * mask).sum() / mask.sum().clamp(min=1.0)
    return d.mean()


def geodesic_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  h: Optional[float] = None) -> torch.Tensor:
    """
    Distância geodésica aproximada: comprimento de arco local sobre a superfície,
    ds = sqrt(1 + z_x² + z_y²). Penaliza diferença do elemento de arco pred vs GT.
    Aproximação barata (não resolve geodésicas globais). Termo caro/ruidoso — usar só
    na ablação.
    """
    pgx, pgy = _grad_xy(pred, h)
    tgx, tgy = _grad_xy(target, h)
    ds_p = torch.sqrt(1.0 + pgx ** 2 + pgy ** 2 + 1e-6)
    ds_t = torch.sqrt(1.0 + tgx ** 2 + tgy ** 2 + 1e-6)
    d = (ds_p - ds_t).abs()
    if mask is not None:
        return (d * mask).sum() / mask.sum().clamp(min=1.0)
    return d.mean()


def metric_tensor_loss(pred: torch.Tensor, target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       h: Optional[float] = None,
                       clamp_val: float = 100.0) -> torch.Tensor:
    """
    Consistência do tensor métrico (1ª forma fundamental) via norma de Frobenius.
    g = [[1+z_x², z_x z_y], [z_x z_y, 1+z_y²]]. ||g_pred - g_target||_F.
    (Termo 'L_metric' do trabalho anterior — incluído para a ablação completa.)
    """
    pgx, pgy = _grad_xy(pred, h)
    tgx, tgy = _grad_xy(target, h)
    # componentes de g
    def comps(gx, gy):
        return (1 + gx ** 2, gx * gy, 1 + gy ** 2)
    p11, p12, p22 = comps(pgx, pgy)
    t11, t12, t22 = comps(tgx, tgy)
    fro = ((p11 - t11) ** 2 + 2 * (p12 - t12) ** 2 + (p22 - t22) ** 2 + 1e-8).sqrt()
    # CLAMP (novo): o tensor métrico envolve QUADRADOS de gradientes, então é o termo de
    # maior magnitude bruta (~10³) e o mais propenso a divergir — foi o que estourou para
    # NaN na B200 (época 15+, config com `metric`). O gauss já tinha clamp; este não tinha.
    # Limitar por pixel mantém o gradiente vivo abaixo do teto e impede runaway.
    if clamp_val and clamp_val > 0:
        fro = fro.clamp(max=clamp_val)
    if mask is not None:
        return (fro * mask).sum() / mask.sum().clamp(min=1.0)
    return fro.mean()


# ---------------------------------------------------------------------------
# Loss composta configurável
# ---------------------------------------------------------------------------

@dataclass
class RiemannWeights:
    berhu: float = 1.0
    grad: float = 0.0
    normal: float = 0.0
    gauss: float = 0.0
    geod: float = 0.0
    metric: float = 0.0
    # hiperparâmetros de estabilidade da curvatura
    gauss_smooth_sigma: float = 0.5
    gauss_clamp: float = 50.0
    # Curvatura MÉTRICA: usa gauss_loss_metrica (retroprojeção 3D real) no lugar de
    # gauss_loss. Requer fx/fy (focal em pixels da resolução de trabalho); ver
    # riemann/geometry.py. gauss_clamp continua sendo o teto, agora em 1/m².
    gauss_metrica: bool = False
    fx: float = 0.0
    fy: float = 0.0
    # Normalização adaptativa por termo: divide cada termo pela sua magnitude típica
    # (EMA) antes de aplicar o peso. Sem isto, termos em unidades diferentes (berHu em
    # metros, metric ~10³, curvatura ~10¹) tornam os pesos reféns das unidades — foi o que
    # fez a gauss/metric "explodir" vs berHu no report. Com isto, os pesos controlam a
    # IMPORTÂNCIA RELATIVA de forma estável. Ligado por padrão.
    normalize_terms: bool = True
    ema_decay: float = 0.99
    # Congela as escalas de normalização após N passos. Depois disso o total volta a ser um
    # sinal de treino legítimo (desce quando o modelo melhora) e comparável entre épocas/runs
    # — sem isso, a EMA persegue o termo e o total fica ancorado ~1.0, mascarando o progresso.
    # 0 = nunca congelar (comportamento antigo, NÃO recomendado para diagnóstico).
    norm_freeze_steps: int = 200
    # Válvula de segurança do congelamento: se um termo ultrapassar este fator vezes a
    # escala congelada, a normalização volta a adaptar para ele (evita runaway/NaN).
    runaway_factor: float = 20.0
    # Teto por pixel do termo do tensor métrico (o de maior magnitude bruta e o que
    # divergiu na B200). 0 desliga.
    metric_clamp: float = 100.0
    # Termo de ESCALA ABSOLUTA (peso). Ver scale_loss(): mede o erro de escala global SEM
    # alinhamento afim, devolvendo ao modelo o sinal que os termos alinhados descartam
    # (hipótese 4 do report). É a correção correta para a cegueira à escala — não o
    # `align_mode` (ver nota abaixo). Ative com um peso pequeno (ex.: 0.1–0.3) junto do berHu.
    scale: float = 0.0
    # Modo do alinhamento afim usado pelos termos de FORMA (berhu/grad/normal/gauss/geod/
    # metric). "full" é o padrão correto: a curvatura exige escala coerente entre pred e GT.
    # NOTA: mexer aqui NÃO resolve a cegueira à escala (o alinhamento casa a escala no
    # forward de qualquer modo); para recuperar o sinal de escala, use o termo `scale`.
    # "detach"/"scale" ficam disponíveis para diagnóstico, mas não são a solução da hip. 4.
    align_mode: str = "full"


def robust_affine_align(pred: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        mode: str = "detach"
                        ) -> torch.Tensor:
    """
    Alinha `pred` a `target` por transformação AFIM (escala s + deslocamento t) estimada
    por mínimos quadrados na região válida, POR AMOSTRA:  pred_aligned = s·pred + t.

    Traz pred e GT à MESMA escala métrica — condição para a curvatura Gaussiana ser
    comparável (a curvatura NÃO é invariante a reescala do eixo z) e para os termos
    geométricos não explodirem vs berHu.

    O parâmetro `mode` controla COMO o gradiente atravessa o alinhamento — esta é a
    variável do experimento sobre a hipótese 4 do report (o afim descartava o sinal de
    escala global):

      - "full"   : comportamento anterior. s e t0 fazem parte do grafo; o gradiente flui
                   por eles e ACABA REMOVENDO qualquer melhoria de escala/offset que o
                   modelo faça — a loss fica cega a escala global (suspeito de estagnar o
                   treino já na época 1).
      - "detach" : (PADRÃO/experimento) s e t0 são calculados e DESTACADOS do grafo. O
                   alinhamento acontece (curvatura coerente), mas no backward eles são
                   constantes — então o erro de escala global CHEGA ao modelo como
                   gradiente, em vez de ser absorvido. Deve destravar a loss de treino.
      - "scale"  : só escala global destacada (sem shift), variante mais conservadora.

    Se a hipótese 4 estiver certa, "detach" faz a loss de treino descer na 1ª época
    (protocolo controlado: berHu puro, 1 época, subset fixo) onde "full" estagnava.
    """
    B = pred.shape[0]
    if mask is None:
        mask = torch.ones_like(pred)
    m = (mask > 0.5).float()

    p = pred.view(B, -1)
    t = target.view(B, -1)
    mm = m.view(B, -1)

    w_sum = mm.sum(dim=1).clamp(min=1.0)
    p_mean = (p * mm).sum(dim=1) / w_sum
    t_mean = (t * mm).sum(dim=1) / w_sum
    p_var = (mm * (p - p_mean[:, None]) ** 2).sum(dim=1) / w_sum
    pt_cov = (mm * (p - p_mean[:, None]) * (t - t_mean[:, None])).sum(dim=1) / w_sum

    s = pt_cov / p_var.clamp(min=1e-8)
    t0 = t_mean - s * p_mean
    s = s.view(B, 1, 1, 1)
    t0 = t0.view(B, 1, 1, 1)

    if mode == "detach":
        # Coeficientes como constantes: alinha sem propagar gradiente pela remoção de
        # escala/shift. O modelo passa a "sentir" o erro de escala global.
        return s.detach() * pred + t0.detach()
    elif mode == "scale":
        # Só escala (destacada), sem deslocamento.
        return s.detach() * pred
    else:  # "full"
        return s * pred + t0


def scale_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Erro de ESCALA ABSOLUTA, SEM alinhamento afim. Este é o sinal que os termos alinhados
    (berhu/grad/normal/gauss/geod/metric após robust_affine_align) descartam por construção
    — a crítica da hipótese 4 do report: "o alinhamento afim torna a loss cega a melhorias
    globais de escala".

    Combina dois sinais: (a) a razão média pred/GT deve tender a 1 (corrige escala
    multiplicativa); (b) um erro absoluto leve (corrige offset). Assim o modelo recebe
    gradiente para acertar a escala métrica real — algo que os termos de forma não fornecem
    porque operam depois do alinhamento.

    Use com peso pequeno (0.1–0.3) junto do berHu. Os termos de forma continuam alinhados
    (a curvatura precisa de escala coerente); este termo, em paralelo, ancora a escala.
    """
    if mask is None:
        mask = torch.ones_like(pred)
    v = (mask > 0.5) & (target > 1e-6)
    p = pred[v].clamp(min=1e-6)
    t = target[v].clamp(min=1e-6)
    if p.numel() == 0:
        return pred.sum() * 0.0
    ratio_err = (p / t).mean().sub(1.0).abs()      # escala multiplicativa -> 1
    abs_err = (p - t).abs().mean()                 # offset/escala absoluta
    return ratio_err + 0.1 * abs_err


class RiemannianAwareLoss(nn.Module):
    """
    Loss composta. Ative termos definindo pesos > 0. Retorna (loss_total, dict_por_termo).

    Exemplo:
        w = RiemannWeights(berhu=0.7, normal=0.9, gauss=0.45)
        crit = RiemannianAwareLoss(w)
        loss, parts = crit(pred, target, mask)
    """
    def __init__(self, weights: RiemannWeights):
        super().__init__()
        self.w = weights
        # Escala típica (EMA) de cada termo, para normalização adaptativa. Começa em None
        # e é preenchida no 1º passo com a magnitude observada. Não é parâmetro treinável.
        self._ema_scale: Dict[str, float] = {}
        # Valores CRUS do último forward (observabilidade — ver _term).
        self._last_raw: Dict[str, float] = {}
        self._frozen: bool = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Dict[str, float]]:
        # Loss sempre em float32, fora do autocast: as 2ª derivadas escalam por
        # 1/h² (~2.6e5 em 512px) e seus produtos estouram o fp16 (inf-inf=NaN),
        # zerando o treino via GradScaler. O custo fp32 aqui é desprezível.
        with torch.autocast(pred.device.type, enabled=False):
            return self._forward_fp32(pred.float(), target.float(),
                                      mask.float() if mask is not None else None)

    def _term(self, name, raw):
        """Normaliza um termo bruto pela sua escala típica (EMA) e devolve o normalizado.
        A EMA é atualizada sem gradiente; a divisão preserva o gradiente do termo.

        IMPORTANTE (observabilidade): o valor CRU de cada termo é sempre guardado em
        self._last_raw, porque o valor normalizado fica ancorado perto de 1.0 por
        construção — ele comprime o progresso e NÃO é comparável entre runs (cada run
        normaliza pela própria EMA). Para julgar "a loss desceu?", use SEMPRE os valores
        crus (colunas raw_* nos logs/CSV).
        """
        self._last_raw[name] = float(raw.detach().abs().cpu())
        if not self.w.normalize_terms:
            return raw
        mag = self._last_raw[name] + 1e-8
        if name not in self._ema_scale:
            self._ema_scale[name] = mag
        elif not self._frozen:
            d = self.w.ema_decay
            self._ema_scale[name] = d * self._ema_scale[name] + (1 - d) * mag
        elif mag > self.w.runaway_factor * self._ema_scale[name]:
            # VÁLVULA DE SEGURANÇA: congelar a escala tirou a supressão automática que a
            # EMA fazia quando um termo disparava (foi provavelmente o que deixou o termo
            # `metric` divergir para NaN na B200). Se a magnitude crescer muito acima da
            # escala congelada, voltamos a adaptar para esse termo — preserva a
            # observabilidade no regime normal e evita runaway no regime patológico.
            self._ema_scale[name] = mag / self.w.runaway_factor
        return raw / self._ema_scale[name]

    def freeze_normalization(self):
        """Congela as escalas de normalização no estado atual.

        Depois de congelar, os pesos seguem balanceados MAS o total volta a ser um sinal
        de treino legítimo (desce quando o modelo melhora) e comparável entre épocas.
        Chamado automaticamente pelo Trainer após `norm_freeze_steps` passos."""
        self._frozen = True

    def _forward_fp32(self, pred, target, mask):
        w = self.w
        parts: Dict[str, torch.Tensor] = {}

        # Termo de ESCALA na predição CRUA (antes do alinhamento) — recupera o sinal de
        # escala global que o afim descarta (hipótese 4). Fica FORA da normalização por EMA
        # porque seu papel é justamente ancorar a escala absoluta.
        if w.scale > 0:
            parts["scale"] = w.scale * scale_loss(pred, target, mask)

        # Termos de FORMA operam na predição ALINHADA (a curvatura exige escala coerente
        # entre pred e GT). O alinhamento traz ambos à mesma escala métrica.
        pred_aligned = robust_affine_align(pred, target, mask, mode=self.w.align_mode)

        # Cada termo é normalizado pela própria escala (EMA) e só então ponderado, para
        # que os pesos controlem importância relativa, não unidades. Ver RiemannWeights.
        if w.berhu > 0:
            parts["berhu"] = w.berhu * self._term("berhu", berhu_loss(pred_aligned, target, mask))
        if w.grad > 0:
            parts["grad"] = w.grad * self._term("grad", grad_loss(pred_aligned, target, mask))
        if w.normal > 0:
            parts["normal"] = w.normal * self._term("normal", normal_loss(pred_aligned, target, mask))
        if w.gauss > 0:
            if w.gauss_metrica:
                parts["gauss"] = w.gauss * self._term("gauss", gauss_loss_metrica(
                    pred_aligned, target, mask, fx=w.fx, fy=(w.fy or None),
                    smooth_sigma=w.gauss_smooth_sigma, clamp_val=w.gauss_clamp))
            else:
                parts["gauss"] = w.gauss * self._term("gauss", gauss_loss(
                    pred_aligned, target, mask, w.gauss_smooth_sigma, w.gauss_clamp))
        if w.geod > 0:
            parts["geod"] = w.geod * self._term("geod", geodesic_loss(pred_aligned, target, mask))
        if w.metric > 0:
            parts["metric"] = w.metric * self._term("metric", metric_tensor_loss(
                pred_aligned, target, mask, clamp_val=w.metric_clamp))

        if not parts:
            raise ValueError("Nenhum termo de loss ativo (todos os pesos são 0).")

        total = sum(parts.values())
        parts_f = {k: float(v.detach().cpu()) for k, v in parts.items()}
        parts_f["total"] = float(total.detach().cpu())
        # Valores CRUS (não normalizados) — use ESTES para julgar se o treino progride.
        # O total/termos normalizados ficam ancorados ~1.0 e comprimem o progresso.
        for k, v in self._last_raw.items():
            parts_f[f"raw_{k}"] = v
        parts_f["raw_total"] = sum(self._last_raw.values())
        return total, parts_f
