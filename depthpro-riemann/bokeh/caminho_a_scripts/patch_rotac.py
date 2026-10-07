#!/usr/bin/env python3
"""Aplica o Caminho A na rota C: fonte realdof, alinhamento afim, K fixo, PSNR.

Substituicao EXATA, com backup .antes_caminho_a e verificacao de ocorrencia unica.
Idempotente: se o alvo ja nao existe e o texto novo ja esta la, pula e avisa.
"""
import sys, shutil
from pathlib import Path

RAIZ = Path("/raid/user_juliadollis/julia_docker/bokehnet-regen")

EDICOES = []   # (arquivo, alvo, novo)


def ed(arquivo, alvo, novo):
    EDICOES.append((arquivo, alvo, novo))


# ---------------------------------------------------------------- qc/metrics.py
ed("src/qc/metrics.py",
'''def laplacian_variance(image: np.ndarray) -> float:''',
'''def psnr(image_a: np.ndarray, image_b: np.ndarray, *,
         data_range: float = 255.0) -> float:
    """PSNR em dB entre duas imagens do mesmo shape, em [0, 255].

    **Não é a função objetivo de nada.** A Eq. 5 maximiza SSIM, e trocar o critério
    mudaria o rótulo. PSNR entra como segunda leitura no relatório: SSIM e PSNR discordam
    com frequência em bokeh (SSIM pesa estrutura, PSNR pesa erro absoluto), e uma
    comparação entre fontes de profundidade que só reporta a métrica que a busca
    OTIMIZOU é uma comparação que se autoconfirma.

    Sobre os três canais, não sobre o cinza: a conversão para luma de `to_gray` descarta
    diferença de cor, e o renderizador mexe nos três canais. `inf` quando as imagens são
    idênticas — o caso é registrado como está, sem teto inventado.
    """
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes diferentes: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((float(data_range) ** 2) / mse))


def laplacian_variance(image: np.ndarray) -> float:''')


# ------------------------------------------------------ src/renderer/calibration.py
ed("src/renderer/calibration.py",
'''from control.contract import SampleRejected
from qc.metrics import ssim''',
'''from control.contract import SampleRejected
from qc.metrics import psnr, ssim''')

# Extrai a grade de trabalho para funcao propria, sem mudar o que ela faz.
ed("src/renderer/calibration.py",
'''    full_h, full_w = aif_bgr.shape[:2]
    scale = 1.0
    aif_w, target_w, depth_w = aif_bgr, target_bgr, depth_m
    if work_long_side and max(full_h, full_w) > work_long_side:
        scale = work_long_side / float(max(full_h, full_w))
        size = (max(1, round(full_h * scale)), max(1, round(full_w * scale)))
        aif_w = _resize_area(aif_bgr, size)         # foto: média por bloco
        target_w = _resize_area(target_bgr, size)   # foto: o alvo do SSIM, idem
        depth_w = _resize_nearest(depth_m, size)    # profundidade: vizinho

    cache: dict[float, float] = {}''',
'''    aif_w, target_w, depth_w, scale = _grade_de_trabalho(
        aif_bgr, target_bgr, depth_m, work_long_side)

    cache: dict[float, float] = {}''')

ed("src/renderer/calibration.py",
'''def calibrate_k(
    render_fn: RenderFn,
    *,
    aif_bgr: np.ndarray,
    target_bgr: np.ndarray,
    depth_m: np.ndarray,
    focus_disparity: float,
    k_min: float = K_MIN_DEFAULT,
    k_max: float = K_MAX_DEFAULT,
    k_absolute_max: float = K_ABSOLUTE_MAX_DEFAULT,
    coarse_points: int = 7,
    tolerance: float = 0.25,
    max_evaluations: int = 40,
    work_long_side: Optional[int] = 512,
) -> KCalibration:''',
'''def _grade_de_trabalho(
    aif_bgr: np.ndarray,
    target_bgr: np.ndarray,
    depth_m: np.ndarray,
    work_long_side: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A redução de custo da Eq. 5, e a escala que ela impõe a K.

    Extraída de `calibrate_k` para que `evaluate_k_fixed` use **a mesma** — duas cópias
    da redução divergiriam, e a divergência entraria como viés sistemático entre a
    condição "com busca" e a condição "K fixo" sem nada denunciar. É a mesma família do
    "cópias divergem" que produziu quatro interpretações de K neste projeto.

    Devolve `(aif, alvo, profundidade, escala)`. `escala` é o fator que converte K da
    resolução ORIGINAL para a de trabalho: CoC em pixel escala com a resolução.
    """
    full_h, full_w = aif_bgr.shape[:2]
    scale = 1.0
    aif_w, target_w, depth_w = aif_bgr, target_bgr, depth_m
    if work_long_side and max(full_h, full_w) > work_long_side:
        scale = work_long_side / float(max(full_h, full_w))
        size = (max(1, round(full_h * scale)), max(1, round(full_w * scale)))
        aif_w = _resize_area(aif_bgr, size)         # foto: média por bloco
        target_w = _resize_area(target_bgr, size)   # foto: o alvo do SSIM, idem
        depth_w = _resize_nearest(depth_m, size)    # profundidade: vizinho
    return aif_w, target_w, depth_w, scale


def evaluate_k_fixed(
    render_fn: RenderFn,
    *,
    aif_bgr: np.ndarray,
    target_bgr: np.ndarray,
    depth_m: np.ndarray,
    focus_disparity: float,
    k_value: float,
    work_long_side: Optional[int] = 512,
    report_metrics: bool = False,
) -> KCalibration:
    """Renderiza UMA vez, com K dado, e mede. **Não busca nada.**

    Existe para separar duas contribuições que o `calibrate_k` mistura. A busca maximiza
    SSIM contra a própria referência, então parte da qualidade medida vem do ajuste de K
    e não da profundidade: um mapa de profundidade pior pode alcançar SSIM parecido se a
    busca puder compensar com outro K. Com K fixo e igual nos três braços, o que sobra na
    diferença de SSIM é a profundidade.

    `is_censored` é sempre `False`, e é afirmação e não descuido: censura é um fato sobre
    a BORDA DA BUSCA (`k == k_max` porque o ótimo ainda crescia), e aqui não houve busca
    nem borda. Marcar `True` ou `False` por proximidade de algum limite inventaria um
    conceito que não se aplica.

    O `k_source` da amostra continua sendo `eq5_ssim_sweep` — `dataio.KSource` é
    vocabulário fechado mantido por outra pessoa e não é editado aqui. Quem lê tem que
    olhar `k_search.k_mode`, que diz `k_fixo` e traz de onde o valor veio. Esta é uma
    folga declarada: o modo K fixo é instrumento de EXPERIMENTO e o diretório de saída
    dele não é release publicável enquanto `KSource` não tiver o termo próprio.
    """
    if aif_bgr.shape[:2] != target_bgr.shape[:2]:
        raise SampleRejected("resolution_invalid",
                             f"aif {aif_bgr.shape[:2]} != alvo {target_bgr.shape[:2]}")
    if not np.isfinite(k_value) or k_value <= 0:
        raise SampleRejected("k_out_of_configured_range",
                             f"K fixo tem que ser finito e positivo: {k_value!r}")

    aif_w, target_w, depth_w, scale = _grade_de_trabalho(
        aif_bgr, target_bgr, depth_m, work_long_side)

    rendered = render_fn(aif_w, depth_w, float(focus_disparity),
                         float(k_value) * scale)
    valor_ssim = float(ssim(rendered, target_w))
    busca = {
        "k_mode": "k_fixo",
        "k_fixo": float(k_value),
        "evaluations": 1,
        "work_long_side": int(work_long_side) if work_long_side else None,
        "work_scale": float(scale),
        "at_upper_bound": False,
        "at_lower_bound": False,
    }
    if report_metrics:
        busca["calibration_psnr"] = float(psnr(rendered, target_w))
    return KCalibration(k_value=float(k_value), calibration_ssim=valor_ssim,
                        is_censored=False, search=busca)


def calibrate_k(
    render_fn: RenderFn,
    *,
    aif_bgr: np.ndarray,
    target_bgr: np.ndarray,
    depth_m: np.ndarray,
    focus_disparity: float,
    k_min: float = K_MIN_DEFAULT,
    k_max: float = K_MAX_DEFAULT,
    k_absolute_max: float = K_ABSOLUTE_MAX_DEFAULT,
    coarse_points: int = 7,
    tolerance: float = 0.25,
    max_evaluations: int = 40,
    work_long_side: Optional[int] = 512,
    report_metrics: bool = False,
) -> KCalibration:''')

ed("src/renderer/calibration.py",
'''    return KCalibration(
        k_value=float(k_star),
        calibration_ssim=float(best_ssim),
        is_censored=censored,
        search={
            "k_min": float(k_min),''',
'''    busca_extra: dict = {}
    if report_metrics:
        # Uma renderização A MAIS, FORA do orçamento de `max_evaluations`, só para medir.
        # O orçamento existe para limitar a BUSCA; contar esta aqui faria o relatório
        # mudar o resultado que ele relata. O SSIM recalculado tem que bater com o da
        # busca (o renderizador é determinístico) e a diferença é gravada em vez de
        # assumida — se algum dia der diferente, é descoberta, não ruído.
        rendered = render_fn(aif_w, depth_w, float(focus_disparity),
                             float(k_star) * scale)
        busca_extra["calibration_psnr"] = float(psnr(rendered, target_w))
        busca_extra["ssim_reconferido_delta"] = float(
            ssim(rendered, target_w) - best_ssim)

    return KCalibration(
        k_value=float(k_star),
        calibration_ssim=float(best_ssim),
        is_censored=censored,
        search={
            "k_mode": "busca_eq5",
            **busca_extra,
            "k_min": float(k_min),''')


# ------------------------------------------------------------ src/routes/route_c.py
ed("src/routes/route_c.py",
'''from renderer.calibration import (
    K_ABSOLUTE_MAX_DEFAULT, K_MAX_DEFAULT, K_MIN_DEFAULT, calibrate_k,
    resize_area_for_photo, resize_nearest,
)''',
'''from renderer.calibration import (
    K_ABSOLUTE_MAX_DEFAULT, K_MAX_DEFAULT, K_MIN_DEFAULT, calibrate_k,
    evaluate_k_fixed, resize_area_for_photo, resize_nearest,
)''')

ed("src/routes/route_c.py",
'''    limit: Optional[int] = None
    seed: int = 0''',
'''    # --- modo de calibração ---
    #: `None` = Eq. 5 com busca, que é o rótulo do paper e o default INTOCADO.
    #: Um valor = renderiza uma vez com ESTE K e mede, sem buscar. Ver
    #: `renderer.calibration.evaluate_k_fixed` para o porquê de existir: a busca
    #: maximiza SSIM contra a própria referência, então parte da qualidade medida vem
    #: do ajuste e não da profundidade. Rodar os dois modos separa as contribuições.
    k_fixed: Optional[float] = None
    #: Mede PSNR além do SSIM, ao custo de UMA renderização a mais por amostra, fora do
    #: orçamento da busca. `False` por default: é instrumentação de experimento, e o
    #: caminho que gera release não deve pagar por ela.
    report_render_metrics: bool = False

    limit: Optional[int] = None
    seed: int = 0''')

ed("src/routes/route_c.py",
'''    calibration = calibrate_k(
        render_fn, aif_bgr=aif_bgr, target_bgr=bokeh_bgr, depth_m=depth.values_m,
        focus_disparity=focus_disparity, k_min=config.k_min, k_max=config.k_max,
        k_absolute_max=config.k_absolute_max,
        work_long_side=config.calibration_long_side,
    )''',
'''    if config.k_fixed is None:
        calibration = calibrate_k(
            render_fn, aif_bgr=aif_bgr, target_bgr=bokeh_bgr, depth_m=depth.values_m,
            focus_disparity=focus_disparity, k_min=config.k_min, k_max=config.k_max,
            k_absolute_max=config.k_absolute_max,
            work_long_side=config.calibration_long_side,
            report_metrics=config.report_render_metrics,
        )
    else:
        # Sem busca: K vem de fora, igual em todos os braços da comparação. O que sobra
        # na diferença de SSIM é a profundidade, e não o ajuste.
        calibration = evaluate_k_fixed(
            render_fn, aif_bgr=aif_bgr, target_bgr=bokeh_bgr, depth_m=depth.values_m,
            focus_disparity=focus_disparity, k_value=config.k_fixed,
            work_long_side=config.calibration_long_side,
            report_metrics=config.report_render_metrics,
        )''')

ed("src/routes/route_c.py",
'''            extra={"k_search": calibration.search,
                   "mask_backend": provenance_base.get("mask_backend"),''',
'''            extra={"k_search": calibration.search,
                   "mask_backend": provenance_base.get("mask_backend"),
                   # Se a profundidade desta amostra foi alinhada por afim contra o
                   # Depth Pro base, e em que domínio. Fica NA AMOSTRA e não só no
                   # `run_config.json`: um artefato sem esta marca é ilegível depois,
                   # porque "alinhado" e "não alinhado" produzem números da mesma
                   # ordem de grandeza e nada os separa a olho.
                   "depth_alignment": provenance_base.get("depth_alignment"),''')

# ----------------------------------------------------------- scripts/run_route_c.py
ed("scripts/run_route_c.py",
'''    p.add_argument("--source", required=True, choices=["realbokeh", "lfdof"])''',
'''    p.add_argument("--source", required=True,
                   choices=["realbokeh", "lfdof", "realdof"])''')

ed("scripts/run_route_c.py",
'''def _carrega_fonte(args):
    if args.source == "realbokeh":
        return _carrega_realbokeh(args)
    if args.source == "lfdof":
        return _carrega_lfdof(args)''',
'''def _carrega_realdof(args):
    """RealDOF: 50 pares fotográficos reais, um por cena, JÁ EM DISCO.

    Mesmo formato de espelho do LFDOF (`image_focus` = AIF, `image_blur` = alvo,
    `file_name_base` = identificador), e as mesmas ausências: sem `--raw-dir`, sem
    `--sensor-width-mm`, sem EXIF em nenhuma das 50 imagens (medido). `_analytic_k`
    devolve `None` e o gate `aif_aperture_is_narrow` sai `applicable=False` — isso é o
    comportamento CORRETO, não uma lacuna a preencher.

    Duas diferenças em relação ao LFDOF, e as duas mudam uma linha cada:

    1. **Sem teto de níveis.** Um par por cena, então `--max-levels-per-scene` não teria
       o que cortar. `_aplica_teto` é chamado do mesmo jeito e vira no-op — chamá-lo
       mantém o log do run com a mesma forma nas três fontes.
    2. **Sem `expected_hw`.** São cinco resoluções nas 50 linhas. Ver o cabeçalho de
       `sources/realdof.py` para a consequência declarada sobre K em pixel.
    """
    from sources.realdof import (REALDOF_DATASET, REALDOF_IMAGE_HW, enumerate_pairs,
                                 enumeration_summary, scene_source_splits)
    from sources.realdof_images import RealDOFImageLoader
    from sources.mirror_images import (MirrorIndex, order_pairs_for_sequential_read,
                                       sample_pairs_for_pilot)

    if not args.mirror_dir:
        raise SystemExit(
            "--source realdof exige --mirror-dir (o snapshot local de "
            "akcit-pixel/RealDOF, com os 4 shards parquet em data/). O snapshot JÁ "
            "ESTÁ em disco: não há download a fazer.")

    print("[fonte] indexando o espelho do RealDOF (só a coluna de nome)…")
    index = MirrorIndex.build(args.mirror_dir, force=args.rebuild_index)
    print(f"[fonte] índice: {len(index)} linhas.")

    log = RejectionLog(Path(args.output_dir) / "source_rejections.jsonl")
    pares = enumerate_pairs(index.names(), log=log, source_dataset=REALDOF_DATASET)
    print(enumeration_summary(pares, log))

    pares = _aplica_teto(pares, args)
    pares = _aplica_fatia(pares, args)

    if args.pilot:
        pares = sample_pairs_for_pilot(pares, limit=args.pilot, seed=args.seed)
        print(f"[fonte] piloto: {len(pares)} pares de "
              f"{len({p.scene_id for p in pares})} cenas (seed {args.seed}).")

    pares = order_pairs_for_sequential_read(pares, index)

    saida = Path(args.output_dir)
    loader = RealDOFImageLoader(
        args.mirror_dir, index, expected_hw=REALDOF_IMAGE_HW,
        ledger_path=saida / "source_images.jsonl",
        store_dir=(saida / "source") if args.store_source_images else None)
    extra = {"source_dataset": REALDOF_DATASET,
             "source_image_hw": None,
             "source_image_hw_evidence": "[M] cinco resoluções nas 50 linhas; o "
                                         "carregador não fixa expected_hw e o HW de "
                                         "cada leitura vai para source_images.jsonl",
             "sensor_width_mm": None,
             "sensor_width_mm_evidence": "nenhuma das 50 imagens do RealDOF tem EXIF "
                                         "(medido); sem Eq. 3 a fechar"}
    return pares, loader, scene_source_splits(pares), extra


def _carrega_fonte(args):
    if args.source == "realbokeh":
        return _carrega_realbokeh(args)
    if args.source == "lfdof":
        return _carrega_lfdof(args)
    if args.source == "realdof":
        return _carrega_realdof(args)''')

ed("scripts/run_route_c.py",
'''    busca = p.add_argument_group("busca de K (Eq. 5)")
    busca.add_argument("--k-min", type=float, default=None)
    busca.add_argument("--k-max", type=float, default=None)
    busca.add_argument("--k-absolute-max", type=float, default=None)
    return p''',
'''    busca = p.add_argument_group("busca de K (Eq. 5)")
    busca.add_argument("--k-min", type=float, default=None)
    busca.add_argument("--k-max", type=float, default=None)
    busca.add_argument("--k-absolute-max", type=float, default=None)
    busca.add_argument("--k-fixo", type=float, default=None,
                       help="DESLIGA a busca e renderiza uma vez com este K. A busca "
                            "maximiza SSIM contra a própria referência, então parte da "
                            "qualidade medida vem do ajuste de K e não da profundidade; "
                            "com K fixo e igual em todos os braços, o que sobra na "
                            "diferença é a profundidade. `is_k_censored` sai sempre "
                            "False (não houve busca, logo não houve borda) e "
                            "`k_search.k_mode` sai `k_fixo`.")
    busca.add_argument("--registrar-psnr", action="store_true",
                       help="mede PSNR além do SSIM, ao custo de UMA renderização a "
                            "mais por amostra, fora do orçamento da busca. Reportar só "
                            "a métrica que a busca OTIMIZOU é uma comparação que se "
                            "autoconfirma.")

    prof = p.add_argument_group(
        "profundidade",
        "Qual Depth Pro produz a profundidade, e se ela é alinhada à do base.")
    prof.add_argument("--depth-checkpoint", default=None,
                      help="checkpoint do Depth Pro a MEDIR. Default: "
                           "<models-dir>/checkpoints/depth_pro.pt. Tem que carregar com "
                           "`strict=True` — um best.pt cru do nosso treino aborta em "
                           "`create_model_and_transforms`; use os mesclados.")
    prof.add_argument("--depth-checkpoint-base", default=None,
                      help="checkpoint que serve de REFERÊNCIA DE ESCALA para "
                           "--alinhar-afim-com-base. Default: o mesmo caminho do base "
                           "em <models-dir>/checkpoints/depth_pro.pt.")
    prof.add_argument("--alinhar-afim-com-base", action="store_true",
                      help="antes de a profundidade entrar no renderizador, alinha a "
                           "predição deste braço à do Depth Pro base NA MESMA IMAGEM, "
                           "por mínimos quadrados (escala e deslocamento). Os nossos "
                           "pesos não são métricos — o treino alinhava por afim dentro "
                           "da loss, então a escala absoluta nunca entrou no gradiente, "
                           "e a profundidade sai em ~57-61%% da base com fator variável "
                           "por imagem (0,27 a 0,84). Sem alinhar, a diferença medida no "
                           "bokeh seria deriva de escala (D_focus muda ~1,7x) e não "
                           "borda melhor, e nada no pipeline reclamaria porque os "
                           "valores continuam fisicamente plausíveis.")
    prof.add_argument("--dominio-alinhamento", default="disparidade",
                      choices=["disparidade", "profundidade"],
                      help="em que grandeza a afim é resolvida. Default `disparidade`, "
                           "que é o domínio em que o resto do pipeline vive "
                           "(`focus_disparity` é mediana de 1/z, e CoC = K*(1/z - "
                           "D_focus)). `profundidade` existe para medir a diferença.")
    prof.add_argument("--alinhamento-jsonl", default=None,
                      help="onde gravar (a, b, R², fração no piso) de cada amostra. "
                           "Default: <output-dir>/depth_alignment.jsonl quando o "
                           "alinhamento está ligado.")
    return p''')

ed("scripts/run_route_c.py",
'''    modelos = Path(args.models_dir)
    depth = DepthProRuntime(modelos / "checkpoints" / "depth_pro.pt", device=args.device)
    mask = BiRefNetRuntime(modelos / "BiRefNet", device=args.device)''',
'''    modelos = Path(args.models_dir)
    ckpt_base = Path(args.depth_checkpoint_base or
                     (modelos / "checkpoints" / "depth_pro.pt"))
    ckpt_alvo = Path(args.depth_checkpoint or
                     (modelos / "checkpoints" / "depth_pro.pt"))
    braco = DepthProRuntime(ckpt_alvo, device=args.device)
    depth, marca_alinhamento, alinhado = _monta_profundidade(args, braco, ckpt_base)
    mask = BiRefNetRuntime(modelos / "BiRefNet", device=args.device)''')

ed("scripts/run_route_c.py",
'''    provenance_base = {
        "pipeline_commit": _git_commit(raiz) or f"sha256tree:{_source_sha256(raiz)}",
        "pipeline_source_sha256": _source_sha256(raiz),
        **depth.provenance(),''',
'''    provenance_base = {
        "pipeline_commit": _git_commit(raiz) or f"sha256tree:{_source_sha256(raiz)}",
        "pipeline_source_sha256": _source_sha256(raiz),
        "depth_alignment": marca_alinhamento,
        **depth.provenance(),''')

ed("scripts/run_route_c.py",
'''    stats = run_route_c(pares, load_pair=load_pair, depth_runtime=depth,
                        mask_runtime=mask, render_fn=renderer, split=split,
                        config=config, provenance_base=provenance_base)
    return 0 if stats.written else 1''',
'''    try:
        stats = run_route_c(pares, load_pair=load_pair, depth_runtime=depth,
                            mask_runtime=mask, render_fn=renderer, split=split,
                            config=config, provenance_base=provenance_base)
    finally:
        # Os ajustes afins saem para disco mesmo se o run abortar no meio: eles são a
        # evidência de QUANTO o alinhamento mexeu, e um run que morreu na amostra 40
        # ainda tem 39 ajustes que respondem isso.
        _grava_alinhamentos(args, alinhado)
    return 0 if stats.written else 1''')

ed("scripts/run_route_c.py",
'''def main() -> int:
    args = build_parser().parse_args()''',
'''def _monta_profundidade(args, braco, ckpt_base: Path):
    """`(runtime, marcação, wrapper ou None)`.

    Três situações, e elas são DIFERENTES — por isso a marcação tem dois booleanos e não
    um. Um artefato em que "sem alinhamento" é um campo só não diz se o experimento
    testou a hipótese ou se alguém esqueceu a flag:

    | flag | braço | `solicitado` | `aplicado` |
    |---|---|---|---|
    | ausente | qualquer | `False` | `False` |
    | presente | É o próprio base | `True` | `False` (identidade) |
    | presente | um afinado | `True` | `True` |

    O caso do meio não roda o ajuste: ajustar a base contra ela mesma dá `a=1, b=0` por
    construção, e pagar uma segunda inferência de Depth Pro por linha para redescobrir
    isso seria dobrar o custo do braço de controle sem medir nada. A identificação é por
    **sha256 do checkpoint**, não por caminho: dois caminhos podem apontar para os mesmos
    bytes, e é dos bytes que a profundidade vem.
    """
    from model_runtime.depth import DepthProRuntime
    from model_runtime.depth_align import DepthProAlinhado, marca_sem_alinhamento

    if not args.alinhar_afim_com_base:
        return braco, marca_sem_alinhamento(
            "--alinhar-afim-com-base não foi passado: a profundidade entra no "
            "renderizador na escala CRUA do checkpoint. Para um checkpoint não métrico "
            "isso mistura deriva de escala com diferença de forma.",
            solicitado=False), None

    referencia = DepthProRuntime(ckpt_base, device=args.device)
    sha_braco = braco.provenance()["depth_model_sha256"]
    sha_base = referencia.provenance()["depth_model_sha256"]
    if sha_braco == sha_base:
        print(f"[profundidade] o braço É a referência (sha {sha_braco[:16]}…): "
              "alinhamento seria a identidade, e não roda.")
        return braco, marca_sem_alinhamento(
            f"o braço e a referência são o mesmo checkpoint (sha256 {sha_braco}); "
            "o ajuste afim seria a identidade (a=1, b=0) por construção.",
            solicitado=True), None

    print(f"[profundidade] alinhando {sha_braco[:16]}… contra a base {sha_base[:16]}… "
          f"no domínio {args.dominio_alinhamento!r}, por amostra.")
    wrapper = DepthProAlinhado(alvo=braco, base=referencia,
                               dominio=args.dominio_alinhamento)
    return wrapper, wrapper.provenance()["depth_alignment"], wrapper


def _grava_alinhamentos(args, alinhado) -> None:
    """Um JSONL com o ajuste de cada amostra, na ordem em que foram inferidas.

    Não é o mesmo que a marcação por amostra em `meta/<id>.json`: lá fica *que* houve
    alinhamento, aqui fica *quanto* ele mexeu. Os dois números que respondem "o
    alinhamento mudou o resultado o quanto se esperava?" são `r2` (a relação entre os
    braços é mesmo afim?) e `razao_mediana_disparidade` (de quanto foi a correção de
    escala nesta imagem?).
    """
    if alinhado is None or not alinhado.ajustes:
        return
    destino = Path(args.alinhamento_jsonl or
                   (Path(args.output_dir) / "depth_alignment.jsonl"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for i, ajuste in enumerate(alinhado.ajustes):
            fh.write(json.dumps({"ordem": i, **ajuste.como_dict()},
                                ensure_ascii=False) + "\\n")
    print(f"[profundidade] {len(alinhado.ajustes)} ajustes afins -> {destino}")


def main() -> int:
    args = build_parser().parse_args()''')

ed("scripts/run_route_c.py",
'''    config = RouteCConfig(
        output_dir=Path(args.output_dir), seed=args.seed, limit=args.limit,''',
'''    config = RouteCConfig(
        output_dir=Path(args.output_dir), seed=args.seed, limit=args.limit,
        k_fixed=args.k_fixo, report_render_metrics=args.registrar_psnr,''')


def main() -> int:
    falhas = 0
    por_arquivo: dict[str, list] = {}
    for arq, alvo, novo in EDICOES:
        por_arquivo.setdefault(arq, []).append((alvo, novo))

    for arq, edicoes in por_arquivo.items():
        caminho = RAIZ / arq
        texto = caminho.read_text(encoding="utf-8")
        original = texto
        for alvo, novo in edicoes:
            n = texto.count(alvo)
            if n == 1:
                texto = texto.replace(alvo, novo, 1)
                print(f"  ok   {arq}: trecho de {len(alvo)} chars substituido")
            elif n == 0 and novo in texto:
                print(f"  pula {arq}: ja aplicado")
            else:
                print(f"  ERRO {arq}: alvo aparece {n}x (esperado 1). Trecho:\\n"
                      f"       {alvo[:90]!r}")
                falhas += 1
        if texto != original:
            backup = caminho.with_suffix(caminho.suffix + ".antes_caminho_a")
            if not backup.exists():
                shutil.copy2(caminho, backup)
                print(f"  bkp  {backup}")
            caminho.write_text(texto, encoding="utf-8")
    if falhas:
        print(f"\\n{falhas} edicao(oes) NAO aplicada(s). Nada mais foi tocado.")
        return 1
    print("\\ntodas as edicoes aplicadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
