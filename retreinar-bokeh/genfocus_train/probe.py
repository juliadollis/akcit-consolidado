"""Probe de CONTROLABILIDADE durante o treino — LVCorr, §4.1(ii).

Por que este arquivo existe, em números medidos:

    LVCorr                      LF-Bokeh   RealBokeh   RealDOF
    fase 1, só sintético         +0,9059    +0,8609     +0,9247
    pesos oficiais do paper      +0,8868    +0,8498     +0,9644
    nosso, fase 2, a+b+c         +0,4365    +0,8257     +0,1324

E a degradação é MONÓTONA no tempo de treino: na variante só-rota-c, no RealDOF,
+0,2509 em 10K steps, +0,0281 em 20K, −0,1776 em 30K, −0,2061 em 60K.

**A loss de flow matching caiu o tempo todo.** O único instrumento do treino era
cego para o modo de falha que efetivamente ocorreu, e o preço foi 60K steps × 4
GPUs para descobrir depois.

O probe mede exatamente a grandeza que quebrou: gera a MESMA cena com K
crescente e confere que a nitidez cai monotonicamente.

Definição, do paper §4.1(ii): *"for images generated from the same all-in-focus
input with a fixed focus plane across varying bokeh levels K, we compute the
Pearson correlation coefficient between these K values and the Laplacian
variance trend"*. Como nitidez CAI quando K cresce, a correlação crua é
negativa; o número que o paper reporta é positivo, então o sinal é invertido —
está explícito em `lvcorr` e travado por teste.

Custo estimado [A]: 8 imagens × 4 valores de K = 32 gerações a 512², ~10 s cada
≈ 5 min. A cada 5.000 steps num run de 60K = 12 probes ≈ 1 h ≈ **~1% do
orçamento**. Ainda não medido em GPU.

COMPARABILIDADE COM O HARNESS DE AVALIAÇÃO — o que bate e o que não bate.
Este probe existe para vigiar, DURANTE o treino, a mesma grandeza que o
`vision-pipeline` mede depois. Conferido linha a linha:

  bate:      a grade de K ({1,5,10,15}); o sinal invertido do Pearson
             (`LVCorr_convencao_paper = -lv` em `metricas_por_imagem.py:155`);
             a luminância BT.601 (o `cv2.COLOR_BGR2GRAY` do harness usa os
             mesmos 0,299/0,587/0,114); o kernel de 4 vizinhos.
  NÃO bate:  o harness usa `cv2.Laplacian`, que preenche a borda por reflexão e
             inclui os pixels de borda na variância; aqui a borda é descartada.
             O efeito é uma constante multiplicativa por imagem, e o Pearson é
             invariante a escala positiva — mas não é bit a bit, e está dito.
  NÃO bate:  o harness roda na resolução do benchmark; o probe roda em 512².

Logo o número do probe é comparável em TENDÊNCIA e em ORDEM DE GRANDEZA com a
tabela, e não é um substituto da avaliação. Quem publicar tabela usa o harness.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# A grade de K.
#
# A Fig. 12 do paper mostra K ∈ {0, 5, 10, 15}, mas **o K=0 dela é rotulado
# "K=0(Input)"** — é a imagem de entrada exibida para comparação, não uma
# geração com mapa nulo.
#
# O que vale aqui não é a figura: é o HARNESS DE AVALIAÇÃO DESTE PROJETO, que
# usa `K_VALUES = [1, 5, 10, 15]`
# (`../genrefocus_deblurnet_paper/vision-pipeline/evaluation/eval_bokeh_synthesis.py:35`)
# e foi com ele que se mediram os números contra os quais este probe existe para
# comparar (+0,9059 na fase 1, +0,4365 na fase 2).
#
# Usar {0,5,10,15} produziria um número SISTEMATICAMENTE MAIOR e não comparável:
# com K=0 o mapa é todo zero, a saída tende à AIF, e esse ponto de nitidez
# máxima ancora a correlação. Um probe que não é comparável com a tabela que ele
# deve vigiar não serve para nada.
K_GRID_PAPER = (1.0, 5.0, 10.0, 15.0)


# =============================================================================
# A métrica — numpy puro, sem GPU, sem rede, testável
# =============================================================================

def laplacian_variance(img_rgb: np.ndarray) -> float:
    """Variância do Laplaciano da luminância. Maior = mais nítida.

    Kernel de 4 vizinhos, o mesmo de `data._laplacian_variance` — mas SEM o
    redimensionamento para 256² que aquele faz. Aqui a imagem já está na
    resolução de geração e reamostrar mudaria a escala do gradiente, que é
    exatamente a grandeza medida.
    """
    a = np.asarray(img_rgb, dtype=np.float32)
    if a.ndim == 3:
        # Luminância BT.601. Média simples dos canais daria peso igual ao azul,
        # que carrega menos detalhe e mais ruído em sensor real.
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = (
        4.0 * a[1:-1, 1:-1]
        - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    )
    return float(lap.var())


def lvcorr(k_values: Sequence[float], sharpness: Sequence[float]) -> float:
    """Pearson entre K e a nitidez, com o sinal INVERTIDO.

    Nitidez cai quando K cresce, então a correlação crua é negativa e o número
    do paper é positivo. `+1,0` = controle perfeito e monotônico; `0` = o modelo
    ignora o K; negativo = o modelo faz o OPOSTO do pedido.

    Devolve `nan` quando alguma das séries é constante — que já é um diagnóstico:
    nitidez constante significa que o K não mudou nada na saída.
    """
    k = np.asarray(k_values, dtype=np.float64)
    s = np.asarray(sharpness, dtype=np.float64)
    if k.size != s.size or k.size < 2:
        raise ValueError(f"séries incompatíveis: {k.size} valores de K, {s.size} de nitidez")
    if k.std() == 0 or s.std() == 0:
        return float("nan")
    return float(-np.corrcoef(k, s)[0, 1])


def lvcorr_agregado(por_imagem: dict[str, float]) -> float:
    """Média dos LVCorr por imagem, ignorando os `nan`.

    Média por IMAGEM e não sobre a nuvem de pontos toda: a variância do
    Laplaciano tem escala muito diferente entre cenas (textura fina versus céu
    liso), e juntar tudo numa correlação só deixaria a cena mais texturizada
    dominar o número.
    """
    validos = [v for v in por_imagem.values() if np.isfinite(v)]
    if not validos:
        return float("nan")
    return float(np.mean(validos))


# =============================================================================
# A geração — precisa de GPU. Isolada aqui para o resto do módulo ser testável.
# =============================================================================

def _mapa_de_defocus_para_probe(
    disparidade: np.ndarray, focus_disparity: float, k_value: float
):
    """Mapa (1, 3, H, W) em [0,1], como a inferência oficial o monta."""
    import torch

    from . import control

    mapa = control.defocus_map(disparidade, focus_disparity, k_value)
    t = torch.from_numpy(mapa)[None]           # (1, H, W)
    return t.repeat(3, 1, 1).unsqueeze(0)      # (1, 3, H, W)


def carregar_conjunto_do_probe(diretorio: str | Path) -> list[dict[str, Any]]:
    """Lê o conjunto FIXO do probe de um diretório.

    Formato, um `.npz` por imagem, gerado por `scripts/build_probe_set.py`:

        aif             uint8  (H, W, 3)     — a entrada, all-in-focus
        disparity       float32 (H, W)       — 1/z em 1/m, do Depth Pro
        focus_disparity float32 escalar      — mediana da disparidade na máscara

    Deliberadamente NÃO é um repo HF nem um split do treino: o conjunto do probe
    tem que ser o MESMO em todos os runs e em todos os braços, e tem que ter
    `scene_id` disjunto do treino. Um probe que muda entre runs mede a diferença
    entre conjuntos, não entre modelos.
    """
    diretorio = Path(os.path.expanduser(str(diretorio)))
    if not diretorio.is_dir():
        raise FileNotFoundError(
            f"conjunto do probe não encontrado: {diretorio}. "
            "Gere com scripts/build_probe_set.py."
        )
    arquivos = sorted(diretorio.glob("*.npz"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum .npz em {diretorio}.")
    itens = []
    for caminho in arquivos:
        dados = np.load(caminho)
        faltando = [c for c in ("aif", "disparity", "focus_disparity") if c not in dados]
        if faltando:
            raise ValueError(f"{caminho.name} não tem {faltando}.")
        itens.append({
            "id": caminho.stem,
            "aif": dados["aif"],
            "disparity": dados["disparity"].astype(np.float32),
            "focus_disparity": float(dados["focus_disparity"]),
        })
    return itens


def run_lvcorr_probe(
    *,
    backbone,
    text_embeddings,
    conjunto: list[dict[str, Any]],
    k_grid: Sequence[float] = K_GRID_PAPER,
    num_inference_steps: int = 28,
    seed: int = 1234,
) -> dict[str, float]:
    """Gera `len(conjunto) × len(k_grid)` imagens e devolve o LVCorr.

    A configuração é a da INFERÊNCIA OFICIAL, sem exceção — um probe rodado com
    outros parâmetros mede outro sistema:
      `guidance_scale=1.0`, 28 passos, `NO_TILED_DENOISE=True` (o conjunto é
      512², um tile só), `kv_cache=False`, prompt e adapter da BokehNet.

    NÃO derruba o treino: qualquer exceção vira aviso e `{}`. É a mesma política
    da validação — um instrumento que mata o experimento que está medindo é pior
    que instrumento nenhum.
    """
    import torch
    from PIL import Image

    from Genfocus.pipeline.flux import Condition, generate, seed_everything

    from .backbone import ADAPTER_NAME

    resultados: dict[str, float] = {}
    backbone.transformer.eval()
    # DEFEITO CORRIGIDO (revisão externa): salvava só o RNG do Torch na CPU.
    # Mas `seed_everything` mexe em torch E numpy, e a geração consome o RNG da
    # GPU. Resultado: MEDIR a controlabilidade alterava a sequência aleatória
    # do treino dali em diante — o instrumento perturbava o experimento, e só
    # no rank principal, que é onde ele roda.
    estado_cpu = torch.get_rng_state()
    estado_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    estado_np = np.random.get_state()
    try:
        for item in conjunto:
            aif_pil = Image.fromarray(np.asarray(item["aif"], dtype=np.uint8)).convert("RGB")
            w, h = aif_pil.size
            nitidez = []
            for k in k_grid:
                mapa = _mapa_de_defocus_para_probe(
                    item["disparity"], item["focus_disparity"], float(k)
                )
                cond_img = Condition(aif_pil, ADAPTER_NAME)
                cond_map = Condition(
                    mapa, ADAPTER_NAME, [0, 0], 1.0, No_preprocess=True
                )
                seed_everything(seed)
                gerador = torch.Generator(device=backbone.device).manual_seed(seed)
                # A9 — o probe tem que medir O MODELO TREINADO. Com
                # `lora_on_main=true` o treino põe LoRA no branch principal, e a
                # inferência EXIGE `main_adapter` (é o que o `train check` avisa).
                # Sem repassar, o probe mediria um modelo diferente do que está
                # sendo treinado e o número não significaria nada.
                main_adapter = ADAPTER_NAME if backbone.lora_on_main else None
                with torch.no_grad():
                    saida = generate(
                        backbone._pipe,
                        height=h,
                        width=w,
                        main_adapter=main_adapter,
                        prompt=None,
                        prompt_embeds=text_embeddings.prompt_embeds,
                        pooled_prompt_embeds=text_embeddings.pooled_prompt_embeds,
                        num_inference_steps=int(num_inference_steps),
                        conditions=[cond_img, cond_map],
                        guidance_scale=1.0,
                        kv_cache=False,
                        generator=gerador,
                        NO_TILED_DENOISE=True,
                    ).images[0]
                nitidez.append(laplacian_variance(np.asarray(saida)))
            resultados[str(item["id"])] = lvcorr(list(k_grid), nitidez)
    except Exception as exc:  # noqa: BLE001 — o probe NUNCA derruba o treino
        print(f"[probe] AVISO: LVCorr falhou ({exc}); seguindo sem o número.", flush=True)
        return {}
    finally:
        backbone.transformer.train()
        torch.set_rng_state(estado_cpu)
        if estado_cuda is not None:
            torch.cuda.set_rng_state_all(estado_cuda)
        np.random.set_state(estado_np)

    return resultados
