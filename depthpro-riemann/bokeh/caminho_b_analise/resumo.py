import json, numpy as np
res=json.load(open("/raid/user_juliadollis/julia_docker/caminho_b_analise/saida/depth_stats.json"))
print("===== mediana sobre as 50 imagens do RealDOF =====")
for nome,L in res.items():
    f=np.array([x["f_px"] for x in L]); m=np.array([x["med"] for x in L])
    mx=np.array([x["max"] for x in L]); mn=np.array([x["min"] for x in L])
    print("%-9s f_px=%9.2f  z_med=%8.3f m  z_min=%7.4f  z_max=%9.2f" % (nome, np.median(f), np.median(m), np.median(mn), np.median(mx)))
b={x["img"]:x for x in res["base"]}
print()
print("===== razao pareada vs base (mediana / min / max sobre as 50) =====")
for nome in ("Bgrad_s0","Bgeod_s0"):
    rz=np.array([x["med"]/b[x["img"]]["med"] for x in res[nome]])
    rf=np.array([x["f_px"]/b[x["img"]]["f_px"] for x in res[nome]])
    rd=np.array([x["disp_med"]/b[x["img"]]["disp_med"] for x in res[nome]])
    print("%s" % nome)
    print("   z_med      : %.4f  [%.4f , %.4f]" % (np.median(rz), rz.min(), rz.max()))
    print("   f_px       : %.6f  [%.6f , %.6f]   (1.0 = cabeca de FOV intocada)" % (np.median(rf), rf.min(), rf.max()))
    print("   D_focus    : %.4f  [%.4f , %.4f]   (K da Eq.3 e o mapa de desfoque escalam com isto)" % (np.median(rd), rd.min(), rd.max()))
