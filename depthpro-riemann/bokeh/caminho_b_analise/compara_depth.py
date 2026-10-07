"""Carrega cada checkpoint pelo caminho EXATO do DepthProRuntime (strict=True,
create_model_and_transforms sem alteracao) e mede focal e escala da profundidade
nas 50 imagens do RealDOF."""
import sys, glob, json, os
sys.path.insert(0,"/workspace/src")
import numpy as np, torch
from PIL import Image

CKPTS = {
 "base":     "/raid/user_juliadollis/julia_docker/models/checkpoints/depth_pro.pt",
 "Bgrad_s0": "/raid/user_juliadollis/julia_docker/caminho_b_analise/ckpt_merged/depth_pro_Bgrad_s0.pt",
 "Bgeod_s0": "/raid/user_juliadollis/julia_docker/caminho_b_analise/ckpt_merged/depth_pro_Bgeod_s0.pt",
}
imgs = sorted(glob.glob("/raid/user_juliadollis/julia_docker/caminho_b_analise/realdof_focus/*__focus.png"))
print("imagens:", len(imgs), flush=True)

from model_runtime import DepthProRuntime
res = {}
for nome, p in CKPTS.items():
    print("==== ", nome, flush=True)
    rt = DepthProRuntime(p, device="cuda")
    rt.load()          # create_model_and_transforms, strict=True, INTOCADO
    print("   carregou strict=True OK; sha:", rt.provenance()["depth_model_sha256"][:16], flush=True)
    linhas=[]
    for q in imgs:
        im = np.asarray(Image.open(q).convert("RGB"))
        with torch.no_grad():
            t = rt._transform(Image.fromarray(im))
            out = rt._model.infer(t.unsqueeze(0).to("cuda"))
            d = out["depth"].squeeze().float().cpu().numpy()
            fpx = float(out["focallength_px"].squeeze().cpu())
        linhas.append({"img": os.path.basename(q), "f_px": fpx,
                       "p10": float(np.percentile(d,10)), "med": float(np.median(d)),
                       "p90": float(np.percentile(d,90)), "min": float(d.min()),
                       "max": float(d.max()),
                       "disp_med": float(np.median(1.0/d))})
    res[nome]=linhas
    del rt; torch.cuda.empty_cache()

json.dump(res, open("/raid/user_juliadollis/julia_docker/caminho_b_analise/saida/depth_stats.json","w"), indent=1)

print("\n===== RESUMO (mediana sobre as 50 imagens) =====")
for nome, L in res.items():
    f=np.array([x["f_px"] for x in L]); m=np.array([x["med"] for x in L])
    print(f"{nome:10s} f_px med={np.median(f):9.2f}  z_med med={np.median(m):8.3f} m  "
          f"z_med p10={np.percentile(m,10):7.3f}  p90={np.percentile(m,90):8.3f}  "
          f"z_max med={np.median([x[max] for x in L]):8.2f}")
b={x["img"]:x for x in res["base"]}
print("\n===== RAZAO PAREADA vs base =====")
for nome in ("Bgrad_s0","Bgeod_s0"):
    rz=np.array([x["med"]/b[x["img"]]["med"] for x in res[nome]])
    rf=np.array([x["f_px"]/b[x["img"]]["f_px"] for x in res[nome]])
    rd=np.array([x["disp_med"]/b[x["img"]]["disp_med"] for x in res[nome]])
    print(f"{nome}: razao z_med  med={np.median(rz):.4f} min={rz.min():.4f} max={rz.max():.4f}")
    pad = " " * len(nome)
    print(pad + f"  razao f_px   med={np.median(rf):.6f} (1.0 = focal intocada)"); print(pad + f"  razao D_focus(disp) med={np.median(rd):.4f}  -> K da Eq.3 escala com isto")
