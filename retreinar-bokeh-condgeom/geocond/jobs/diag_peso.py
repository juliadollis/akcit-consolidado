"""Escolhe `theta` e `lambda_o` da perda ponderada, por medição.

    python3 -m geocond.jobs.diag_peso --release <dir> --amostras 64 --out diag.json

POR QUE ESTE JOB EXISTE
-----------------------
A perda ponderada por oclusão da proposta (§5.3) já **piorou o modelo** uma vez.
O diagnóstico ficou fechado: a borda é ~1,2% dos PIXELS e praticamente 100% dos
TOKENS, porque a 512² um token do FLUX cobre 16×16 px. Com peso contínuo e
normalização pela média, os tokens de menor oclusão caíram para 0,816 — a
supervisão diminuiu ~18% em 74% dos tokens para financiar o topo, e o erro fora
da borda subiu 31%.

A correção é limiarizar em token, `w = 1 + λ·[O_token > θ]`. E o número que
governa a escolha **não é a fração de tokens pesados, é o PISO**: a normalização
conserva o orçamento de peso, então concentrar em menos tokens só os torna mais
pesados e afunda o piso. O piso sobe quando se reduz `λ` junto.

Por isso `geocond/loss_weight.py` não tem default de θ nem de λ, e por isso
estes números saem daqui — medidos na distribuição do release NOVO, não na
tabela da campanha antiga, que foi calculada noutra unidade e noutra convenção.

CRITÉRIO DECLARADO ANTES DE OLHAR O RESULTADO
--------------------------------------------
Escolhemos o par (θ, λ) que maximiza o CONTRASTE sujeito a **piso ≥ 0,95**.

O 0,95 não é arbitrário: o fracasso medido teve piso 0,806, ou seja rebaixou a
supervisão em ~19% na maior parte dos tokens. Exigir piso ≥ 0,95 limita esse
rebaixamento a ≤5%, que é a ordem da variação natural da própria loss entre
steps. Fixado aqui, no código, antes de rodar — critério escolhido depois de ver
o resultado é racionalização.

SEM GPU. Usa o `_pack_latents` REAL do diffusers, porque a ordenação dos tokens
é parte do que está sendo medido.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..constants import GeoConstants
from ..contrato import CANAIS, CONTRATO_DE_CONTROLE, AmostraGeo
from ..signals import pilha_geometrica

#: Piso mínimo aceitável do peso normalizado. Ver o cabeçalho.
PISO_MINIMO = 0.95

#: A grade varrida. `theta = 0` é o modo CONTÍNUO (o que falhou), mantido na
#: tabela justamente para o contraste com as opções limiarizadas ficar visível.
GRADE_THETA = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95)
GRADE_LAMBDA = (0.5, 1.0, 2.0, 3.0)

IDX_O = CANAIS.index("O")


def _le(raiz: Path, sid: str, meta: dict) -> tuple[np.ndarray, AmostraGeo]:
    for cand in (raiz / "depth" / f"{sid}.png",
                 raiz / "depth" / str(meta.get("scene_id", "")) / f"{sid}.png",
                 raiz / "depth" / f"{meta.get('scene_id','')}.png"):
        if cand.exists():
            caminho = cand
            break
    else:
        raise FileNotFoundError(sid)
    arr = np.asarray(Image.open(caminho))
    if arr.dtype != np.uint16:
        raise ValueError(f"{caminho.name}: dtype {arr.dtype}, esperado uint16")
    d01 = arr.astype(np.float64) / 65535.0
    h, w = d01.shape
    return d01, AmostraGeo(
        disparity_min=float(meta["disparity_min"]),
        disparity_max=float(meta["disparity_max"]),
        largura_px=w, altura_px=h, focallength_px=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--constantes", required=True, type=Path,
                    help="JSON do geocond.jobs.calibrar (as constantes MEDIDAS)")
    ap.add_argument("--amostras", type=int, default=64)
    ap.add_argument("--particao", default="train")
    ap.add_argument("--lado", type=int, default=512, help="o crop do treino")
    ap.add_argument("--semente", type=int, default=1234)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from diffusers import FluxPipeline
    from ..loss_weight import diagnostico_peso

    consts = GeoConstants.from_dict(
        json.loads(args.constantes.read_text(encoding="utf-8"))["geo_constantes"]
    )

    linhas = [json.loads(l) for l in (args.release / "manifest.jsonl").open(encoding="utf-8")]
    linhas = [d for d in linhas if str(d.get("split", "")) == args.particao]
    rng = random.Random(args.semente)
    rng.shuffle(linhas)
    escolhidas = linhas[: args.amostras]
    print(f"[diag] release={args.release}")
    print(f"[diag] {len(escolhidas)} amostras da partição {args.particao!r} "
          f"(de {len(linhas):,}), semente {args.semente}")

    # ── acumula o canal O de todas as amostras, no CROP do treino ────────────
    lado = args.lado
    canais_O = []
    for i, linha in enumerate(escolhidas):
        sid = linha["sample_id"]
        cm = args.release / "meta" / f"{sid}.json"
        if not cm.exists():
            cm = args.release / "meta" / str(linha.get("scene_id", "")) / f"{sid}.json"
        meta = json.loads(cm.read_text(encoding="utf-8"))
        if str(meta.get("control_version")) != CONTRATO_DE_CONTROLE:
            raise SystemExit(f"[erro] control_version inesperado em {sid}")
        d01, amostra = _le(args.release, sid, meta)
        pilha = pilha_geometrica(d01, amostra, consts)
        O = np.asarray(pilha.canais)[IDX_O]
        # reamostra para o lado do treino: o que importa é a ESTATÍSTICA de O na
        # grade em que o token vive, e o dataloader entrega 512².
        im = Image.fromarray((O * 65535).astype(np.uint16), mode="I;16").resize(
            (lado, lado), Image.NEAREST)
        canais_O.append(np.asarray(im).astype(np.float32) / 65535.0)
        if (i + 1) % 16 == 0:
            print(f"  ... {i+1}/{len(escolhidas)}")

    O_t = torch.from_numpy(np.stack(canais_O)).unsqueeze(1)  # (B,1,S,S)
    pack = FluxPipeline._pack_latents
    frac_px = float((O_t > 0.3).float().mean().item())
    print(f"\n[diag] fração de PIXELS com O > 0,3: {frac_px:.4f}")

    # ── a grade ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'theta':>6} {'lambda':>7} {'fração':>8} {'piso':>8} {'contraste':>10}   veredito")
    print("=" * 78)
    tabela, viaveis = [], []
    for th in GRADE_THETA:
        for lam in GRADE_LAMBDA:
            d = diagnostico_peso(O_t, pack, lambda_o=lam, theta=th, pool="max")
            ok = d["piso"] >= PISO_MINIMO
            linha = {"theta": th, "lambda_o": lam,
                     "fracao": d["fracao_tokens_pesados"], "piso": d["piso"],
                     "contraste": d["contraste"], "viavel": ok,
                     "p50": d["p50"], "p90": d["p90"], "p99": d["p99"],
                     "frac_tokens_O_positivo": d["fracao_tokens_O_positivo"]}
            tabela.append(linha)
            if ok:
                viaveis.append(linha)
            marca = "ok" if ok else f"piso < {PISO_MINIMO}"
            extra = "  <- o modo CONTINUO, que FALHOU" if th == 0.0 else ""
            print(f"{th:6.2f} {lam:7.2f} {d['fracao_tokens_pesados']:8.3f} "
                  f"{d['piso']:8.3f} {d['contraste']:10.2f}   {marca}{extra}")

    d0 = tabela[0]
    print(f"\n[diag] tokens com O > 0: {d0['frac_tokens_O_positivo']:.3f}   "
          f"<- o achado que motiva o limiar (borda esparsa em pixel, densa em token)")
    print(f"[diag] distribuição de O_token: p50={d0['p50']:.4f} "
          f"p90={d0['p90']:.4f} p99={d0['p99']:.4f}")

    if not viaveis:
        print(f"\n[diag] NENHUM par atinge piso >= {PISO_MINIMO}. A perda ponderada "
              "NAO entra: o criterio foi declarado antes e nao se negocia depois.")
        escolhido = None
    else:
        escolhido = max(viaveis, key=lambda r: (r["contraste"], r["piso"]))
        print(f"\n[diag] ESCOLHIDO  theta={escolhido['theta']} "
              f"lambda_o={escolhido['lambda_o']}  "
              f"(piso {escolhido['piso']:.3f}, contraste {escolhido['contraste']:.2f}, "
              f"{escolhido['fracao']*100:.1f}% dos tokens)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "escolhido": escolhido,
        "criterio": {"piso_minimo": PISO_MINIMO,
                     "regra": "max contraste sujeito a piso >= piso_minimo",
                     "declarado_antes_de_olhar": True},
        "grade": tabela,
        "procedencia": {"release": str(args.release), "amostras": len(escolhidas),
                        "particao": args.particao, "semente": args.semente,
                        "lado": lado, "frac_pixels_O_maior_0.3": frac_px},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[diag] gravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
