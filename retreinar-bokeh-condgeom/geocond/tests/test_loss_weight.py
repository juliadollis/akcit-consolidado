"""Testes do peso da perda em espaço de token. Sem GPU, sem rede.

O teste central é o de ORDENAÇÃO: um único bloco de 16×16 pixels aceso tem de
acender exatamente UM token, no índice esperado. Um erro de ordenação aqui não
levanta exceção nenhuma — o treino simplesmente aprenderia a ponderar o lugar
errado da imagem por 60K steps, e o único sintoma seria uma métrica pior no fim.

`_pack_latents`: usamos o REAL do diffusers quando ele está instalado. Quando
não está (ambiente local sem o stack de difusão), cai num STUB que replica a
ordenação linha a linha — e o teste `test_stub_bate_com_o_pack_latents_real`
fica marcado com skip EXPLÍCITO, dizendo que essa verificação precisa rodar no
cluster. Enquanto ela não rodar lá, a ordenação está verificada apenas contra a
nossa leitura do código do diffusers, não contra o diffusers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# A raiz do repositório no sys.path: `geocond` é um pacote namespace e este
# arquivo pode ser coletado de qualquer cwd.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

torch = pytest.importorskip("torch", reason="torch é obrigatório para este módulo")
import torch.nn.functional as F  # noqa: E402

from geocond.loss_weight import (  # noqa: E402
    LATENT_CHANNELS,
    TOKEN_PX,
    VAE_STRIDE,
    diagnostico_peso,
    occlusion_to_token_weight,
    weighted_flow_matching_loss,
)


# ---------------------------------------------------------------- _pack_latents

def _pack_latents_stub(latents, batch_size, num_channels_latents, height, width):
    """Cópia da ordenação do `FluxPipeline._pack_latents` do diffusers.

    Transcrito do código (diffusers/pipelines/flux/pipeline_flux.py). Existe só
    para os testes rodarem sem o stack de difusão instalado; o módulo em si
    NUNCA reimplementa isto — recebe a função do pipe.
    """
    x = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


try:
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    PACK = FluxPipeline._pack_latents
    PACK_EH_REAL = True
except Exception:                                   # pragma: no cover
    PACK = _pack_latents_stub
    PACK_EH_REAL = False


@pytest.mark.skipif(
    not PACK_EH_REAL,
    reason=(
        "diffusers ausente: a ordenação dos tokens está sendo verificada contra o "
        "STUB, não contra o `_pack_latents` real. ESTE TESTE PRECISA RODAR NO "
        "CLUSTER, onde o diffusers do treino está instalado — um desvio de "
        "ordenação é silencioso e custaria 60K steps."
    ),
)
def test_stub_bate_com_o_pack_latents_real():
    torch.manual_seed(0)
    x = torch.randn(2, LATENT_CHANNELS, 8, 6)
    real = FluxPipeline._pack_latents(x, 2, LATENT_CHANNELS, 8, 6)
    stub = _pack_latents_stub(x, 2, LATENT_CHANNELS, 8, 6)
    assert torch.equal(real, stub), "o stub dos testes divergiu do diffusers"


# ------------------------------------------------------------------ utilidades

def _zeros(B=1, H=64, W=64):
    return torch.zeros(B, 1, H, W)


def _de_tokens(o_tok_2d: torch.Tensor) -> torch.Tensor:
    """(h, w) em token → (1, 1, h*16, w*16) em pixel, blocos constantes.

    Com blocos constantes o max-pool 16× devolve o valor do bloco exatamente,
    então dá para prescrever a distribuição de `O_token` e conferir o piso
    contra a forma fechada.
    """
    px = o_tok_2d.repeat_interleave(TOKEN_PX, 0).repeat_interleave(TOKEN_PX, 1)
    return px[None, None].float()


# ------------------------------------------------------------------ ordenação

@pytest.mark.parametrize("bloco_y,bloco_x", [(0, 0), (0, 3), (2, 1), (3, 3)])
def test_um_bloco_aceso_acende_exatamente_um_token_no_indice_certo(bloco_y, bloco_x):
    H = W = 64
    o = _zeros(H=H, W=W)
    y0, x0 = bloco_y * TOKEN_PX, bloco_x * TOKEN_PX
    o[0, 0, y0:y0 + TOKEN_PX, x0:x0 + TOKEN_PX] = 1.0

    w = occlusion_to_token_weight(o, PACK, lambda_o=1.0, theta=0.5, normalizar=False)
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
    o[0, 0, 37, 5] = 1.0                        # linha 37 → bloco 2; coluna 5 → bloco 0
    w = occlusion_to_token_weight(o, PACK, lambda_o=1.0, theta=0.5, normalizar=False)
    acesos = (w[0, :, 0] > 1.0).nonzero().flatten().tolist()
    assert acesos == [2 * (64 // TOKEN_PX) + 0]


def test_512_da_1024_tokens_como_o_treino_real():
    """A aritmética do plano: 512 px → N = 1024 tokens, 1 token = 16×16 px."""
    w = occlusion_to_token_weight(
        _zeros(H=512, W=512), PACK, lambda_o=1.0, theta=0.3
    )
    assert w.shape == (1, 1024, 1)


def test_mapa_uniforme_da_peso_uniforme():
    o = torch.full((2, 1, 64, 64), 0.5)
    w = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.3, normalizar=False)
    assert torch.allclose(w, torch.full_like(w, 3.0))       # todos acima de θ


# -------------------------------------------------------------- max × média

def test_max_preserva_a_amplitude_de_borda_fina_e_a_media_dilui_por_16():
    """A razão de o `max` ser obrigatório, em número.

    Uma borda de oclusão tem 1-2 px de largura. Dentro de um bloco de 16×16 uma
    coluna de 1 px é 1/16 da área, então a MÉDIA a reduz a 1/16: um λ=3 nominal
    vira um λ EFETIVO de 3/16 = 0,1875 ≈ 0,2. O max mantém os 3,0.
    """
    o = _zeros(H=64, W=64)
    o[0, 0, :, 16] = 1.0                        # uma coluna de 1 px de largura
    w_max = occlusion_to_token_weight(
        o, PACK, lambda_o=3.0, theta=0.0, pool="max", normalizar=False)
    w_mean = occlusion_to_token_weight(
        o, PACK, lambda_o=3.0, theta=0.0, pool="mean", normalizar=False)

    pico_max = float(w_max.max()) - 1.0
    pico_mean = float(w_mean.max()) - 1.0
    assert abs(pico_max - 3.0) < 1e-6, f"max devia dar λ cheio, deu {pico_max:.4f}"
    assert abs(pico_mean - 3.0 / TOKEN_PX) < 1e-6, (
        f"média devia diluir por 1/16 → 0,1875; deu {pico_mean:.4f}"
    )
    assert pico_max / pico_mean == pytest.approx(16.0, abs=1e-4)


def test_avg_e_sinonimo_de_mean():
    o = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(7))
    a = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.0, pool="mean")
    b = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.0, pool="avg")
    assert torch.equal(a, b)


# ------------------------------------------------------- espalhamento 16×

def test_espalhamento_de_uma_linha_de_1px_e_exatamente_16x():
    """A geometria que produziu o 82,7× medido, isolada.

    Uma coluna de 1 px acesa num mapa 512² é 1/512 dos pixels; depois do
    max-pool 16× ela acende a coluna INTEIRA de tokens, 32/1024 = 1/32. A razão
    é 16 — o lado do bloco. Na imagem real, com bordas em várias orientações e
    já espessas, esse fator empilhou até 82,7×.
    """
    o = _zeros(H=512, W=512)
    o[0, 0, :, 100] = 1.0
    d = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.3)
    assert d["fracao_pixels_acesos"] == pytest.approx(1 / 512)
    assert d["fracao_tokens_pesados"] == pytest.approx(1 / 32)
    assert d["espalhamento"] == pytest.approx(16.0)


# ------------------------------------------------- normalização e o PISO

def test_normalizacao_mantem_a_media_do_peso_em_um():
    """Sem isto, λ vira multiplicador do passo efetivo do otimizador e o
    controle "perda ponderada contra a linha de base" deixa de separar
    supervisão concentrada de learning rate maior."""
    torch.manual_seed(0)
    o = torch.rand(3, 1, 64, 64)
    for lam in (0.5, 1.0, 3.0, 10.0):
        for th in (0.0, 0.3, 0.9):
            w = occlusion_to_token_weight(o, PACK, lambda_o=lam, theta=th)
            assert abs(float(w.mean()) - 1.0) < 1e-6, f"λ={lam} θ={th}"


def test_sem_normalizar_a_media_cresce_com_lambda():
    torch.manual_seed(0)
    o = torch.rand(1, 1, 64, 64)
    m1 = float(occlusion_to_token_weight(
        o, PACK, lambda_o=1.0, theta=0.3, normalizar=False).mean())
    m3 = float(occlusion_to_token_weight(
        o, PACK, lambda_o=3.0, theta=0.3, normalizar=False).mean())
    assert m3 > m1 > 1.0


def _mapa_v1() -> torch.Tensor:
    """Um mapa sintético 512² cuja distribuição de `O_token` REPRODUZ a medida.

    1024 tokens, quatro grupos, escolhidos para casar com o diagnóstico da v1:

        8   tokens a 0,00  → fração com O>0    = 1016/1024 = 0,992
        845 tokens a 0,01586
        94  tokens a 0,35  → fração com O>0,3  =  171/1024 = 0,167
        77  tokens a 1,00  → fração com O>0,9  =   77/1024 = 0,075

    e média de O_token = 0,1204, que é o que produz o piso 0,806 do modo
    contínuo com λ=2. Com isso as TRÊS linhas da tabela do diagnóstico saem do
    mesmo mapa.
    """
    vals = torch.cat([
        torch.zeros(8),
        torch.full((845,), 0.01586),
        torch.full((94,), 0.35),
        torch.full((77,), 1.0),
    ])
    # embaralha determinístico, para o resultado não depender da ordem
    perm = torch.randperm(1024, generator=torch.Generator().manual_seed(11))
    return _de_tokens(vals[perm].reshape(32, 32))


def test_a_tabela_do_diagnostico_sai_do_mapa_sintetico():
    """As três linhas que decidiram o desenho, conferidas de ponta a ponta:

        θ=0,0 λ=2,0 → fração 0,992  piso 0,806  contraste 3,00  (contínuo: PIOROU)
        θ=0,3 λ=2,0 → fração 0,167  piso 0,750  contraste 3,00  (limiar só)
        θ=0,9 λ=1,0 → fração 0,075  piso 0,930  contraste 2,00  (λ menor junto)
    """
    o = _mapa_v1()
    d0 = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.0)
    d3 = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.3)
    d9 = diagnostico_peso(o, PACK, lambda_o=1.0, theta=0.9)

    assert d0["fracao_tokens_pesados"] == pytest.approx(0.992, abs=5e-4)
    assert d0["piso"] == pytest.approx(0.806, abs=2e-3)
    assert d0["contraste"] == pytest.approx(3.00, abs=1e-3)

    assert d3["fracao_tokens_pesados"] == pytest.approx(0.167, abs=5e-4)
    assert d3["piso"] == pytest.approx(0.750, abs=2e-3)
    assert d3["contraste"] == pytest.approx(3.00, abs=1e-3)

    assert d9["fracao_tokens_pesados"] == pytest.approx(0.075, abs=5e-4)
    assert d9["piso"] == pytest.approx(0.930, abs=2e-3)
    assert d9["contraste"] == pytest.approx(2.00, abs=1e-3)


def test_limiarizar_sozinho_ABAIXA_o_piso_e_reduzir_lambda_junto_o_levanta():
    """O achado que o módulo existe para não deixar esquecer.

    A normalização conserva o ORÇAMENTO de peso, então concentrar o peso em
    menos tokens só os torna mais pesados — e afunda quem não é borda. Trocar
    θ=0 por θ=0,3 com o MESMO λ=2 derruba o piso de 0,806 para 0,750. O piso só
    sobe quando λ desce junto (θ=0,9, λ=1 → 0,930).
    """
    o = _mapa_v1()
    piso_cont = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.0)["piso"]
    piso_lim = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.3)["piso"]
    piso_bom = diagnostico_peso(o, PACK, lambda_o=1.0, theta=0.9)["piso"]
    assert piso_lim < piso_cont, (
        f"limiarizar sozinho tinha de ABAIXAR o piso: {piso_lim:.3f} vs {piso_cont:.3f}"
    )
    assert piso_bom > piso_cont, (
        f"λ menor junto do limiar tinha de LEVANTAR o piso: {piso_bom:.3f}"
    )


@pytest.mark.parametrize("lam,frac_alvo", [(2.0, 0.167), (1.0, 0.075), (3.0, 0.992)])
def test_piso_binario_segue_a_forma_fechada(lam, frac_alvo):
    """No modo binário, piso = 1/(1+λ·f) e contraste = 1+λ, exatamente."""
    o = _mapa_v1()
    theta = {0.167: 0.3, 0.075: 0.9, 0.992: 0.0001}[frac_alvo]
    d = diagnostico_peso(o, PACK, lambda_o=lam, theta=theta)
    f = d["fracao_tokens_pesados"]
    assert d["piso"] == pytest.approx(1.0 / (1.0 + lam * f), rel=1e-5)
    assert d["contraste"] == pytest.approx(1.0 + lam, rel=1e-5)


def test_percentis_de_O_token_saem_no_diagnostico():
    o = _mapa_v1()
    d = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.3)
    assert d["p50"] == pytest.approx(0.01586, abs=1e-4)
    assert d["p75"] == pytest.approx(0.01586, abs=1e-4)      # 84,5% está no grupo baixo
    assert d["p90"] == pytest.approx(0.35, abs=1e-3)
    assert d["p99"] == pytest.approx(1.0, abs=1e-3)
    assert d["theta"] == 0.3 and d["lambda_o"] == 2.0 and d["pool"] == "max"


def test_espalhamento_historico_usa_O_positivo_em_token():
    """O 82,7× registrado comparava fração de tokens com O>0 contra fração de
    PIXELS com O>0,3 — limiares diferentes dos dois lados. O dict devolve as
    duas razões para o número histórico continuar reproduzível sem que a razão
    honesta (mesmo θ) seja contaminada."""
    o = _mapa_v1()
    d = diagnostico_peso(o, PACK, lambda_o=2.0, theta=0.3)
    assert d["fracao_tokens_O_positivo"] == pytest.approx(0.992, abs=5e-4)
    assert d["espalhamento_historico"] > d["espalhamento"]


# ------------------------------------------------------------ limiar × contínuo

def test_theta_positivo_torna_o_peso_binario_e_esparso():
    torch.manual_seed(3)
    o = torch.rand(1, 1, 64, 64) * 0.5          # quase tudo abaixo de 0,5
    o[0, 0, 20:24, :] = 0.9                     # uma faixa alta
    cont = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.0, normalizar=False)
    binr = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.6, normalizar=False)

    assert float((cont[..., 0] > 1.0).float().mean()) > 0.9, (
        "sem limiar, quase todo token pesa — é o 82,7× de espalhamento"
    )
    fr = float((binr[..., 0] > 1.0).float().mean())
    assert 0.0 < fr < 0.5, f"com limiar o peso tem de ficar esparso; deu {fr:.3f}"
    vals = {round(float(v), 6) for v in binr.flatten()}
    assert vals == {1.0, 3.0}, f"o peso tem de ser 1 ou 1+λ; veio {vals}"


def test_theta_aumenta_o_contraste():
    """O que o limiar entrega: CONTRASTE entre borda e não-borda."""
    torch.manual_seed(4)
    o = torch.rand(1, 1, 64, 64) * 0.5
    o[0, 0, 20:24, :] = 0.9
    c_cont = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.0)
    c_bin = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.6)
    contraste_cont = float(c_cont.max() / c_cont.min())
    contraste_bin = float(c_bin.max() / c_bin.min())
    assert contraste_bin > contraste_cont, (
        f"contínuo={contraste_cont:.2f} binário={contraste_bin:.2f}"
    )
    assert contraste_bin == pytest.approx(3.0, rel=1e-5)


def test_theta_zero_e_o_modo_continuo_da_v1():
    """θ=0 NÃO é "limiar em zero": é o modo CONTÍNUO, o braço de ablação.
    Tem de reproduzir `w = 1 + λ·O_token`, que é o que a condição A' treinou."""
    torch.manual_seed(5)
    o = torch.rand(1, 1, 64, 64)
    w = occlusion_to_token_weight(o, PACK, lambda_o=2.0, theta=0.0, normalizar=False)
    esperado = 1.0 + 2.0 * F.max_pool2d(o, TOKEN_PX, TOKEN_PX).reshape(1, -1, 1)
    assert torch.allclose(w, esperado, atol=1e-6)
    # e tem de ter mais de dois valores distintos — não é binário
    assert len({round(float(v), 5) for v in w.flatten()}) > 2


# ----------------------------------------------------------------------- perda

def test_peso_none_reproduz_a_perda_atual_BIT_A_BIT():
    """A condição de controle A' sai de graça: mesmo código, mesmo caminho
    numérico. `torch.equal`, não `allclose` — a afirmação é de bit."""
    torch.manual_seed(1)
    pred, tgt = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    assert torch.equal(
        weighted_flow_matching_loss(pred, tgt),
        F.mse_loss(pred.float(), tgt.float()),
    )


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.9])
def test_lambda_zero_reproduz_a_linha_de_base(theta):
    """λ=0 tem de ser NUMERICAMENTE a linha de base, para qualquer θ."""
    torch.manual_seed(2)
    pred, tgt = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    o = torch.rand(2, 1, 64, 64)
    w = occlusion_to_token_weight(o, PACK, lambda_o=0.0, theta=theta)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-7)
    a = weighted_flow_matching_loss(pred, tgt, w)
    b = F.mse_loss(pred.float(), tgt.float())
    assert torch.allclose(a, b, atol=1e-6), f"{float(a)} vs {float(b)}"


def test_o_peso_concentra_a_perda_onde_a_oclusao_esta():
    N = (64 // TOKEN_PX) ** 2
    o = _zeros(H=64, W=64)
    o[0, 0, 0:TOKEN_PX, 0:TOKEN_PX] = 1.0                 # token 0
    w = occlusion_to_token_weight(o, PACK, lambda_o=3.0, theta=0.5)

    tgt = torch.zeros(1, N, 8)
    dentro = tgt.clone(); dentro[0, 0] = 1.0
    fora = tgt.clone(); fora[0, N - 1] = 1.0
    l_dentro = weighted_flow_matching_loss(dentro, tgt, w)
    l_fora = weighted_flow_matching_loss(fora, tgt, w)
    assert float(l_dentro) > float(l_fora) * 3.0


def test_dtype_bf16_entra_e_a_perda_sai_fp32():
    """O treino roda em bf16; a perda tem de subir para fp32 como a atual faz."""
    pred = torch.randn(1, 16, 64, dtype=torch.bfloat16)
    tgt = torch.randn(1, 16, 64, dtype=torch.bfloat16)
    w = torch.ones(1, 16, 1, dtype=torch.bfloat16)
    assert weighted_flow_matching_loss(pred, tgt, w).dtype == torch.float32


def test_gradiente_flui_pela_perda_ponderada():
    """O peso é constante; o gradiente tem de passar pela predição."""
    pred = torch.randn(1, 16, 8, requires_grad=True)
    tgt = torch.randn(1, 16, 8)
    w = torch.ones(1, 16, 1)
    weighted_flow_matching_loss(pred, tgt, w).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ------------------------------------------------------------------ validações

def test_shape_e_parametros_errados_levantam():
    with pytest.raises(ValueError):                       # canais != 1
        occlusion_to_token_weight(
            torch.zeros(1, 3, 64, 64), PACK, lambda_o=1.0, theta=0.3)
    with pytest.raises(ValueError):                       # H não múltiplo de 16
        occlusion_to_token_weight(
            torch.zeros(1, 1, 60, 64), PACK, lambda_o=1.0, theta=0.3)
    with pytest.raises(ValueError):                       # pool inválido
        occlusion_to_token_weight(
            torch.zeros(1, 1, 64, 64), PACK, lambda_o=1.0, theta=0.3, pool="min")
    with pytest.raises(ValueError):                       # λ negativo
        occlusion_to_token_weight(
            torch.zeros(1, 1, 64, 64), PACK, lambda_o=-1.0, theta=0.3)
    with pytest.raises(ValueError):                       # θ >= 1 esvaziaria B
        occlusion_to_token_weight(
            torch.zeros(1, 1, 64, 64), PACK, lambda_o=1.0, theta=1.0)
    with pytest.raises(ValueError):                       # weight com N errado
        weighted_flow_matching_loss(
            torch.zeros(1, 4, 8), torch.zeros(1, 4, 8), torch.ones(1, 9, 1))


def test_lambda_e_theta_nao_tem_default():
    """Os (θ,λ) da tabela vieram da distribuição ANTIGA de O. Não viram default:
    serão recalibrados sobre o release novo. Quem chama passa."""
    with pytest.raises(TypeError):
        occlusion_to_token_weight(_zeros(), PACK)          # type: ignore[call-arg]
    with pytest.raises(TypeError):
        occlusion_to_token_weight(_zeros(), PACK, lambda_o=2.0)  # type: ignore[call-arg]


def test_VAE_STRIDE_e_TOKEN_PX_sao_8_e_16():
    assert (VAE_STRIDE, TOKEN_PX) == (8, 16)
