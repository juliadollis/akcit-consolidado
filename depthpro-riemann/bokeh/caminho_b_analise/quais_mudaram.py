import torch, collections, sys
B="/raid/user_juliadollis/julia_docker"
base = torch.load(B+"/models/checkpoints/depth_pro.pt", map_location="cpu", weights_only=False)
for alvo in ["runs_confirma/Bgeod/seed_0/best.pt","runs_confirma/Bgrad/seed_0/best.pt"]:
    nosso = torch.load(B+"/"+alvo, map_location="cpu", weights_only=False)
    strip = {k[5:] if k.startswith("core.") else k: v for k,v in nosso.items()}
    print("=====", alvo, "n=",len(strip))
    mud=[]
    for k in strip:
        if k not in base: print("  UNEXPECTED", k); continue
        if not torch.equal(base[k].float(), strip[k].float()): mud.append(k)
    print("  mudaram:", len(mud))
    print("  por prefixo:", dict(sorted(collections.Counter(".".join(k.split(".")[:2]) for k in mud).items())))
    print("  lista:", sorted(mud))
    falt=[k for k in base if k not in strip]
    print("  faltando:", len(falt), "prefixos:", dict(sorted(collections.Counter(".".join(k.split(".")[:2]) for k in falt).items())))
