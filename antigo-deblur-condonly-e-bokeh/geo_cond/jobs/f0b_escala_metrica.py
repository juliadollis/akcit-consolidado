"""F0b -- escala metrica e focal por amostra, com Depth Pro.

Produz, por `stem`, o que os 4 canais geometricos dependentes de escala exigem e
que hoje nao existe para a rota c:

    z_min_m, z_max_m      faixa metrica, para  z = z_min + depth01*(z_max-z_min)
    z_focus_m             plano de foco pela Eq. 4 do paper (mediana na mascara)
    focallength_px        focal em pixels, para a curvatura retroprojetada
    niveis_quant          gate de quantizacao (canais de 2a ordem)
    coc_p99_px            diagnostico, comparavel com a coluna da tabela kfix

DECISOES QUE SEGUEM O PAPER, LITERAIS
-------------------------------------
Eq. 4:  D_focus = median( D[M] ),  M = mascara em foco.
        Usamos a coluna `foreground_mask`, que existe e esta 100% preenchida nas
        duas rotas (AUDITORIA secao 1). NAO rodamos BiRefNet de novo: a mascara
        do df e a que gerou os rotulos, e regerar introduziria uma segunda fonte.

        O teste T1 (REGISTRO 2026-09-04) mediu que `z_focus_m` da tabela kfix E
        essa mediana, a 1e-5, em 178/180 amostras. Este job reproduz o MESMO
        procedimento na rota c.

`focallength_px` vem do proprio Depth Pro, com `f_px=None`. O modelo estima a
focal e a devolve junto da profundidade metrica, no mesmo forward. Hoje esse
valor e calculado e DESCARTADO nas tres copias do pipeline
(`grep -rn focallength_px` -> zero usos).

SENTINELA DO z_max (correcao, nao fidelidade)
---------------------------------------------
`z_max_m == 10000.0` exato, o teto do Depth Pro, em 2.988 das 11.635 amostras da
rota b (25,7%). Usar o max cru espalha a faixa util por 4 ordens de grandeza e
destroi a quantizacao. Gravamos os DOIS: `z_max_m_bruto` e `z_max_m` (percentil
`--percentil-max`, default 99,5). Quem consome escolhe, e a decisao fica no
config em vez de embutida aqui.

Uso (no container julia-genrefocus-eval:3.0, 1 GPU):
    python3 f0b_escala_metrica.py --rota c --n 0 --saida /saida/f0b_rota_c.jsonl
    python3 f0b_escala_metrica.py --rota b --n 400 --saida /saida/f0b_rota_b.jsonl
"""

from __future__ import annotations

import argparse, io, json, os, sys, time

import numpy as np
from PIL import Image

REPOS = {"b": "AKCITPixel3/BKXcuVXCmeRvN", "c": "AKCITPixel3/CMiQdveBBzNii"}
N_TOTAL = {"b": 11635, "c": 2932}


def _pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict) and "bytes" in x:
        return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    raise TypeError(f"tipo de imagem inesperado: {type(x)}")


def _mascara01(m) -> np.ndarray:
    a = np.asarray(_pil(m).convert("L"), dtype=np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rota", choices=["b", "c"], required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = tudo")
    ap.add_argument("--limiar-mascara", type=float, default=0.5)
    ap.add_argument("--percentil-max", type=float, default=99.5)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--a-cada", type=int, default=25)
    a = ap.parse_args()

    import torch, depth_pro
    from datasets import load_dataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[f0b] dispositivo: {dev}", flush=True)
    modelo, transform = depth_pro.create_model_and_transforms()
    modelo.eval().to(dev)

    ds = load_dataset(REPOS[a.rota], split="train", streaming=True,
                      token=os.environ.get("HF_TOKEN"))

    # append-only: retomar de onde parou, sem refazer nada
    feitos = set()
    if os.path.exists(a.saida):
        with open(a.saida) as f:
            for ln in f:
                try: feitos.add(json.loads(ln)["stem"])
                except Exception: pass
        print(f"[f0b] retomando: {len(feitos)} ja feitos", flush=True)

    alvo = a.n or N_TOTAL[a.rota]
    t0 = time.time(); n = 0; pulados = 0
    with open(a.saida, "a") as out:
        for r in ds:
            if n >= alvo:
                break
            stem = r.get("stem")
            if not stem or stem in feitos:
                continue
            try:
                aif = _pil(r["aif"]).convert("RGB")
                with torch.no_grad():
                    pred = modelo.infer(transform(aif).to(dev), f_px=None)
                z = pred["depth"].squeeze().float().cpu().numpy()      # metros
                f_px = float(pred["focallength_px"].item())
            except Exception as e:
                print(f"[f0b] {stem}: FALHOU {e}", flush=True); pulados += 1; continue

            # Eq. 4: mediana da profundidade DENTRO da mascara em foco
            try:
                m = _mascara01(r["foreground_mask"])
                if m.shape != z.shape:
                    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                                   .resize((z.shape[1], z.shape[0]), Image.BILINEAR),
                                   dtype=np.float32) / 255.0
                sel = m > a.limiar_mascara
                z_focus = float(np.median(z[sel])) if sel.sum() >= 64 else None
            except Exception as e:
                print(f"[f0b] {stem}: mascara {e}", flush=True); z_focus = None

            zf = z[np.isfinite(z) & (z > 0)]
            if zf.size < 1024:
                pulados += 1; continue
            z_min = float(zf.min())
            z_max_bruto = float(zf.max())
            z_max_pct = float(np.percentile(zf, a.percentil_max))

            # quantizacao: quantos niveis uint16 a cena util ocuparia
            faixa = max(z_max_pct - z_min, 1e-9)
            d01 = np.clip((zf - z_min) / faixa, 0.0, 1.0)
            niveis = int(np.unique(np.rint(d01 * 65535)).size)

            reg = {
                "stem": stem, "rota": a.rota,
                "z_min_m": z_min,
                "z_max_m": z_max_pct,
                "z_max_m_bruto": z_max_bruto,
                "z_max_saturado": bool(z_max_bruto >= 9999.0),
                "z_focus_m": z_focus,
                "focallength_px": f_px,
                "largura_px": int(aif.size[0]), "altura_px": int(aif.size[1]),
                "niveis_quant": niveis,
                "percentil_max_usado": a.percentil_max,
            }
            if z_focus is not None and z_focus > 0:
                u = 1.0 / np.clip(zf, 1e-6, None)
                reg["coc_rel_p99"] = float(np.percentile(np.abs(u - 1.0 / z_focus), 99))
            out.write(json.dumps(reg) + "\n"); out.flush()
            n += 1
            if n % a.a_cada == 0:
                dt = time.time() - t0
                print(f"[f0b] {n}/{alvo}  {dt/n:.2f}s/img  "
                      f"restam ~{(alvo-n)*dt/n/60:.1f}min  pulados={pulados}", flush=True)

    print(f"[f0b] FIM rota {a.rota}: {n} gravados, {pulados} pulados, "
          f"{(time.time()-t0)/60:.1f} min -> {a.saida}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
