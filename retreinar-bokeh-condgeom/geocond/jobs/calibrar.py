"""Calibra as constantes de normalização dos canais geométricos.

    python3 -m geocond.jobs.calibrar --release <dir> --amostras 300 --out constantes.json

POR QUE ESTE JOB EXISTE, E POR QUE ELE NÃO TEM DEFAULT PARA COPIAR
------------------------------------------------------------------
`GeoConstants` não tem default nenhum, de propósito. Um número plausível
silencioso reintroduz a classe de defeito que a auditoria deste projeto
encontrou: o valor pedido vira outro valor, sem erro e sem aviso. As constantes
são MEDIDAS, uma vez, sobre o conjunto de TREINO, e gravadas no config junto
com as demais.

E elas têm de ser **fixas entre imagens**. Não é estética: sob crop aleatório um
percentil calculado no recorte não bate com o da imagem inteira, e treino e
inferência veriam normalizações diferentes. Pior, numa cena sem descontinuidade
real o percentil é ruído e o mapa de oclusão vira ruído saturado de quadro
cheio. Por isso o percentil é tirado do POOL de todas as amostras, nunca por
amostra.

AS CONSTANTES DA CAMPANHA ANTIGA NÃO SERVEM
--------------------------------------------
`tau_occlusion=88,170`, `u_max=2,723`, `s_max=4,479` e as demais foram
calibradas sobre profundidade métrica normalizada por imagem, com a escala
vindo de uma tabela externa. Aqui o PNG é disparidade linear entre
`disparity_min`/`disparity_max` do `meta/`, então as grandezas mudaram de
unidade e de distribuição. Ver `referencia_antiga/LEIA.md`.

O QUE ELE IMPRIME, E POR QUÊ
-----------------------------
A tabela de percentis inteira, não só o valor escolhido. Uma constante que sai
de um job sem a distribuição ao lado é um número que ninguém consegue auditar
depois — e a escolha do percentil é uma decisão, não um fato.

SEM GPU E SEM REDE. Lê o release do disco.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..constants import GeoConstants
from ..contrato import CONTRATO_DE_CONTROLE, AmostraGeo
from ..signals import (
    comprimento_de_normalizacao,
    disparidade_metrica,
    elemento_de_area_e_normais,
    gradiente_normalizado,
    niveis_quantizacao,
    suavizar,
)

#: Percentil que vira `tau_occlusion` e `s_max`. DECISÃO, e o motivo de ser 99,5
#: e não 100: o máximo cru é um único pixel e é ruído de quantização ou artefato
#: de borda do Depth Pro. Com 99,5 a saturação do canal fica em 0,5% dos pixels,
#: que é a fração que se quer — o canal `O` deve saturar NA borda, não em lugar
#: nenhum (tau alto demais) nem em toda parte (tau baixo demais).
PERCENTIL_TAU = 99.5
PERCENTIL_U = 99.9   # `u` é disparidade; o teto tem de cobrir o objeto mais próximo
PERCENTIL_S = 99.5

#: `smooth_sigma` é ESCOLHA, não medição: é o compromisso entre suprimir o ruído
#: de quantização da profundidade estimada e não borrar a descontinuidade que o
#: canal existe para marcar. 2,0 px é o valor herdado, e o kernel agora tem raio
#: round(3σ)=6, não o raio fixo 2 que a v1 usava (com σ=2 aquilo cobria só ±1σ,
#: ou seja era caixa truncada, não gaussiana).
SMOOTH_SIGMA = 2.0

#: Abaixo disto a derivada segunda é ruído de quantização, não geometria.
MIN_NIVEIS = 256


def _le_amostra(raiz: Path, sid: str, meta: dict) -> tuple[np.ndarray, AmostraGeo]:
    """PNG de disparidade + escalares do `meta/`, no layout PLANO do release local.

    O layout AGRUPADO por cena (`depth/<cena>/<id>.png`) é o que o push para o
    Hub produz, porque o Hub recusa diretório com mais de 10.000 arquivos. Os
    dois existem de verdade; aqui a detecção é por tentativa.
    """
    for cand in (raiz / "depth" / f"{sid}.png",
                 raiz / "depth" / str(meta.get("scene_id", "")) / f"{sid}.png",
                 raiz / "depth" / f"{meta.get('scene_id', '')}.png"):
        if cand.exists():
            caminho = cand
            break
    else:
        raise FileNotFoundError(f"profundidade de {sid!r} não encontrada em {raiz}")

    im = Image.open(caminho)
    arr = np.asarray(im)
    if arr.dtype != np.uint16:
        raise ValueError(
            f"{caminho.name}: dtype {arr.dtype}, esperado uint16. O contrato é "
            "`uint16_linear_in_disparity` e decidir pelo MODO do PIL em vez do "
            "dtype é como a convenção errada entrou antes."
        )
    disp01 = arr.astype(np.float64) / 65535.0
    altura, largura = disp01.shape
    amostra = AmostraGeo(
        disparity_min=float(meta["disparity_min"]),
        disparity_max=float(meta["disparity_max"]),
        largura_px=largura,
        altura_px=altura,
        # O `meta/` NÃO traz focal — conferido campo a campo no release da rota A.
        # Sem ela a curvatura retroprojetada não existe e o canal sai neutro.
        focallength_px=None,
    )
    return disp01, amostra


def _quantidades_cruas(disp01: np.ndarray, amostra: AmostraGeo, field: str):
    """As grandezas ANTES de qualquer normalização — é delas que saem as constantes."""
    disp = disparidade_metrica(disp01, amostra)
    campo = disp if field == "inverse" else 1.0 / np.maximum(disp, 1e-6)
    suave = suavizar(campo, SMOOTH_SIGMA)
    L = comprimento_de_normalizacao(amostra)
    fx, fy = gradiente_normalizado(suave, L)
    grad_mag = np.hypot(fx, fy)
    s, _, _ = elemento_de_area_e_normais(fx, fy)
    return disp, grad_mag, s


def _tabela(nome: str, pool: np.ndarray, percentis) -> dict:
    vals = {f"p{p}": float(np.percentile(pool, p)) for p in percentis}
    vals["max"] = float(pool.max())
    vals["mean"] = float(pool.mean())
    print(f"\n  {nome}  (n = {pool.size:,} pixels amostrados)")
    for k, v in vals.items():
        print(f"    {k:>7s} = {v:.6g}")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--amostras", type=int, default=300,
                    help="quantas amostras do split de TREINO entram no pool")
    ap.add_argument("--particao", default="train",
                    help="só o treino calibra: calibrar na validação a contamina")
    ap.add_argument("--field", default="inverse", choices=("inverse", "depth"))
    ap.add_argument("--semente", type=int, default=1234)
    ap.add_argument("--pixels-por-amostra", type=int, default=20000,
                    help="subamostragem por imagem, para o pool caber em memória")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    raiz = args.release
    manifesto = raiz / "manifest.jsonl"
    if not manifesto.exists():
        raise SystemExit(f"[erro] sem manifest.jsonl em {raiz}")

    # ── seleção: só o TREINO, e determinística ───────────────────────────────
    linhas = []
    with manifesto.open(encoding="utf-8") as fh:
        for linha in fh:
            d = json.loads(linha)
            if str(d.get("split", "")) == args.particao:
                linhas.append(d)
    if not linhas:
        raise SystemExit(f"[erro] nenhuma amostra na partição {args.particao!r}")

    rng = random.Random(args.semente)
    rng.shuffle(linhas)
    escolhidas = linhas[: args.amostras]
    print(f"[calibrar] release   : {raiz}")
    print(f"[calibrar] partição  : {args.particao}  ({len(linhas):,} amostras)")
    print(f"[calibrar] amostradas: {len(escolhidas)}  (semente {args.semente})")
    print(f"[calibrar] field     : {args.field}")

    pool_u, pool_g, pool_s = [], [], []
    niveis, contratos, sem_focal = [], set(), 0
    rng_px = np.random.default_rng(args.semente)

    for i, linha in enumerate(escolhidas):
        sid = linha["sample_id"]
        cam_meta = raiz / "meta" / f"{sid}.json"
        if not cam_meta.exists():
            cam_meta = raiz / "meta" / str(linha.get("scene_id", "")) / f"{sid}.json"
        meta = json.loads(cam_meta.read_text(encoding="utf-8"))

        contratos.add(str(meta.get("control_version", "?")))
        if not meta.get("focallength_px"):
            sem_focal += 1

        disp01, amostra = _le_amostra(raiz, sid, meta)
        niveis.append(niveis_quantizacao(disp01))
        disp, grad_mag, s = _quantidades_cruas(disp01, amostra, args.field)

        # subamostra pixels: o pool inteiro de 300 imagens a 768×494 seriam
        # 114M de valores por grandeza, e o percentil não precisa disso.
        n = min(args.pixels_por_amostra, disp.size)
        idx = rng_px.choice(disp.size, size=n, replace=False)
        pool_u.append(disp.ravel()[idx])
        pool_g.append(grad_mag.ravel()[idx])
        pool_s.append(s.ravel()[idx])

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(escolhidas)}")

    # ── o contrato tem de ser o esperado, senão as constantes descrevem outra coisa
    if contratos != {CONTRATO_DE_CONTROLE}:
        raise SystemExit(
            f"[erro] control_version do release = {sorted(contratos)}, esperado "
            f"{CONTRATO_DE_CONTROLE!r}. Calibrar sobre outra convenção produziria "
            "constantes plausíveis e erradas — exatamente o que este job existe "
            "para impedir."
        )

    pool_u = np.concatenate(pool_u)
    pool_g = np.concatenate(pool_g)
    pool_s = np.concatenate(pool_s)
    niveis = np.asarray(niveis)

    print("\n" + "=" * 72)
    print("DISTRIBUIÇÕES — é isto que torna a escolha do percentil auditável")
    print("=" * 72)
    percentis = (50, 75, 90, 95, 99, 99.5, 99.9)
    t_u = _tabela("u  — disparidade métrica (1/m)", pool_u, percentis)
    t_g = _tabela(f"‖∇f‖ — gradiente normalizado, field={args.field}", pool_g, percentis)
    t_s = _tabela("s  — log sqrt(det g)", pool_s, percentis)

    print(f"\n  níveis de quantização por amostra: "
          f"mín {niveis.min():,} · mediana {int(np.median(niveis)):,} · máx {niveis.max():,}")
    abaixo = int((niveis < MIN_NIVEIS).sum())
    print(f"  amostras abaixo de {MIN_NIVEIS} níveis: {abaixo}/{len(niveis)} "
          f"({100*abaixo/len(niveis):.2f}%) — nelas a 2ª ordem sairia neutra")
    print(f"  amostras sem focallength_px: {sem_focal}/{len(escolhidas)} "
          f"({100*sem_focal/len(escolhidas):.1f}%)")

    # ── as constantes ────────────────────────────────────────────────────────
    tau = float(np.percentile(pool_g, PERCENTIL_TAU))
    u_max = float(np.percentile(pool_u, PERCENTIL_U))
    s_max = float(np.percentile(pool_s, PERCENTIL_S))

    # Curvatura: o release não traz `fx`, então `K~` sai NEUTRO em 100% das
    # amostras e estas duas constantes não são exercidas. Ficam com valores
    # DECLARADAMENTE não calibrados, e o JSON registra isso — em vez de um
    # número inventado que pareceria medido.
    k0, kt = 1.0, 8.0
    curvatura_calibrada = sem_focal < len(escolhidas)

    consts = GeoConstants(
        tau_occlusion=tau,
        u_max=u_max,
        s_max=s_max,
        k0_curvature=k0,
        kt_max=kt,
        smooth_sigma=SMOOTH_SIGMA,
        min_niveis_para_segunda_ordem=MIN_NIVEIS,
    )

    # fração de pixels que saturam cada canal com a escolha feita
    sat_O = float((pool_g >= tau).mean())
    sat_u = float((pool_u >= u_max).mean())
    sat_s = float((pool_s >= s_max).mean())

    print("\n" + "=" * 72)
    print("CONSTANTES ESCOLHIDAS")
    print("=" * 72)
    print(f"  tau_occlusion = {tau:.6g}   (p{PERCENTIL_TAU} de ‖∇f‖; satura {100*sat_O:.2f}% dos px)")
    print(f"  u_max         = {u_max:.6g}   (p{PERCENTIL_U} de disp;  satura {100*sat_u:.2f}%)")
    print(f"  s_max         = {s_max:.6g}   (p{PERCENTIL_S} de s;     satura {100*sat_s:.2f}%)")
    print(f"  k0_curvature  = {k0:.6g}   NÃO CALIBRADO — sem focal, K~ é neutro")
    print(f"  kt_max        = {kt:.6g}   NÃO CALIBRADO — idem")
    print(f"  smooth_sigma  = {SMOOTH_SIGMA}   ESCOLHA declarada, não medição")
    print(f"  min_niveis    = {MIN_NIVEIS}     ESCOLHA declarada")

    saida = {
        "geo_constantes": consts.to_dict(),
        "procedencia": {
            "release": str(raiz),
            "control_version": CONTRATO_DE_CONTROLE,
            "particao": args.particao,
            "amostras": len(escolhidas),
            "semente": args.semente,
            "field": args.field,
            "pixels_por_amostra": args.pixels_por_amostra,
            "percentil_tau": PERCENTIL_TAU,
            "percentil_u": PERCENTIL_U,
            "percentil_s": PERCENTIL_S,
            "curvatura_calibrada": curvatura_calibrada,
            "amostras_sem_focal": sem_focal,
        },
        "distribuicoes": {"u": t_u, "grad": t_g, "s": t_s},
        "saturacao": {"O": sat_O, "u": sat_u, "s": sat_s},
        "niveis_quantizacao": {
            "min": int(niveis.min()), "mediana": float(np.median(niveis)),
            "max": int(niveis.max()), "abaixo_do_minimo": abaixo,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[calibrar] gravado em {args.out}")
    print("[calibrar] cole o bloco `geo_constantes` no YAML da campanha.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
