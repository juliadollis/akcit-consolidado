import torch, collections
B="/raid/user_juliadollis/julia_docker"
base = torch.load(B+"/models/checkpoints/depth_pro.pt", map_location="cpu", weights_only=False)
nosso = torch.load(B+"/runs_confirma/Bgeod/seed_0/best.pt", map_location="cpu", weights_only=False)
print("base: n chaves", len(base))
print("base prefixos nivel1:", dict(collections.Counter(k.split(".")[0] for k in base)))
print("base prefixos nivel2:", dict(sorted(collections.Counter(".".join(k.split(".")[:2]) for k in base).items())))
print()
strip = {k[len("core."):] if k.startswith("core.") else k: v for k,v in nosso.items()}
bset, nset = set(base), set(strip)
print("nosso(sem core.): n", len(nset))
print("== depois de tirar o prefixo core. ==")
print("chaves do nosso que NAO existem na base (unexpected):", len(nset-bset))
for k in sorted(nset-bset)[:20]: print("   +", k)
falt = bset-nset
print("chaves da base AUSENTES no nosso (missing):", len(falt))
print("  prefixos das ausentes:", dict(sorted(collections.Counter(".".join(k.split(".")[:3]) for k in falt).items())[:40]))
# shape mismatch
mm=[k for k in (bset&nset) if tuple(base[k].shape)!=tuple(strip[k].shape)]
print("shape mismatch:", len(mm), mm[:10])
# quantos pesos realmente mudaram vs base
import torch as T
iguais=0; difs=0; maxdelta=0.0
for k in (bset&nset):
    a,b2=base[k].float(), strip[k].float()
    if T.equal(a,b2): iguais+=1
    else:
        difs+=1; maxdelta=max(maxdelta, (a-b2).abs().max().item())
print(f"tensores identicos a base: {iguais}; diferentes: {difs}; max|delta|={maxdelta:.6g}")
