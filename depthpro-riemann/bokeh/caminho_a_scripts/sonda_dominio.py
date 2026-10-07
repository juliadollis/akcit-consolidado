"""Em QUAL dominio a relacao entre os bracos e afim: disparidade ou profundidade?

A escolha nao e estilistica. Se a relacao for afim em disparidade e alinharmos em
metros (ou vice-versa), o residuo do ajuste entra como distorcao justamente na grandeza
que o renderizador consome (CoC = K*(1/z - D_focus)), e a comparacao entre bracos passa
a medir o erro do alinhamento junto com a diferenca de forma.

Mede R^2 do ajuste de 1 grau nos dois dominios, nas MESMAS imagens, para os dois bracos.
5 imagens bastam: e um fato sobre a familia de transformacao, nao uma media a estimar.
"""
import sys, glob, json, os
sys.path.insert(0, "/workspace/bokehnet-regen/src")
import numpy as np, torch
from PIL import Image

from model_runtime import DepthProRuntime
from model_runtime.depth_align import ajusta_afim
from sources.mirror_images import MirrorIndex, order_pairs_for_sequential_read, sample_pairs_for_pilot
from sources.realdof import enumerate_pairs
from sources.realdof_images import RealDOFImageLoader
from qc.rejection import RejectionLog

MIRROR = "/workspace/caminho_a_rotac/realdof_mirror"
CKPTS = {
    "base":     "/workspace/models/checkpoints/depth_pro.pt",
    "Bgrad_s0": "/workspace/caminho_b_analise/ckpt_merged/depth_pro_Bgrad_s0.pt",
    "Bgeod_s0": "/workspace/caminho_b_analise/ckpt_merged/depth_pro_Bgeod_s0.pt",
}
N = int(os.environ.get("N_IMAGENS", "5"))

idx = MirrorIndex.build(MIRROR)
pares = enumerate_pairs(idx.names(), log=RejectionLog(None))
alvo = order_pairs_for_sequential_read(sample_pairs_for_pilot(pares, limit=N, seed=0), idx)
print("imagens:", [p.sample_id for p in alvo], flush=True)

loader = RealDOFImageLoader(MIRROR, idx)
aifs = {}
for p in alvo:
    aif_bgr, _ = loader(p)
    aifs[p.sample_id] = np.ascontiguousarray(aif_bgr[..., ::-1])   # BGR -> RGB
print("AIFs decodificadas.", flush=True)

prof = {}
for nome, caminho in CKPTS.items():
    rt = DepthProRuntime(caminho, device="cuda")
    rt.load()
    print(f"[{nome}] strict=True ok, sha {rt.provenance()['depth_model_sha256'][:16]}", flush=True)
    prof[nome] = {sid: rt.infer(img).values_m.astype(np.float64)
                  for sid, img in aifs.items()}
    del rt
    torch.cuda.empty_cache()

linhas = []
print("\n%-12s %-26s %10s %10s %10s %10s" %
      ("braco", "imagem", "r2_disp", "r2_prof", "a_disp", "razao_z"))
for nome in ("Bgrad_s0", "Bgeod_s0"):
    for sid in aifs:
        zb, za = prof["base"][sid], prof[nome][sid]
        ok = np.isfinite(zb) & (zb > 0) & np.isfinite(za) & (za > 0)
        ad, bd, r2d = ajusta_afim((1.0 / za)[ok], (1.0 / zb)[ok])
        ap, bp, r2p = ajusta_afim(za[ok], zb[ok])
        razao_z = float(np.median(za[ok]) / np.median(zb[ok]))
        linhas.append({"braco": nome, "imagem": sid, "r2_disparidade": r2d,
                       "r2_profundidade": r2p, "a_disparidade": ad,
                       "b_disparidade": bd, "a_profundidade": ap,
                       "b_profundidade": bp, "razao_z_mediana": razao_z})
        print("%-12s %-26s %10.5f %10.5f %10.5f %10.4f" % (nome, sid, r2d, r2p, ad, razao_z))

print("\n== mediana por braco ==")
for nome in ("Bgrad_s0", "Bgeod_s0"):
    sub = [l for l in linhas if l["braco"] == nome]
    print("%-12s r2_disparidade=%.5f   r2_profundidade=%.5f" %
          (nome, float(np.median([l["r2_disparidade"] for l in sub])),
           float(np.median([l["r2_profundidade"] for l in sub]))))

json.dump(linhas, open("/workspace/caminho_a_rotac/out/sonda_dominio.json", "w"), indent=1)
print("\ngravado em caminho_a_rotac/out/sonda_dominio.json")
