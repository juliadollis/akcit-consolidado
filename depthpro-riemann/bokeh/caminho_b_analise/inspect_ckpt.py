import sys, torch, collections
p = sys.argv[1]
obj = torch.load(p, map_location="cpu", weights_only=False)
print("tipo topo:", type(obj))
if isinstance(obj, dict):
    ks = list(obj.keys())
    print("n chaves topo:", len(ks))
    print("primeiras 20 chaves topo:", ks[:20])
    # heuristica: state_dict puro?
    sd = obj
    for cand in ("state_dict","model","model_state_dict","net"):
        if cand in obj and isinstance(obj[cand], dict):
            print("  -> wrapper detectado em:", cand)
            sd = obj[cand]
            break
    tensores = {k:v for k,v in sd.items() if hasattr(v,"shape")}
    print("n tensores:", len(tensores))
    pref = collections.Counter(k.split(".")[0] for k in tensores)
    print("prefixos nivel 1:", dict(pref))
    pref2 = collections.Counter(".".join(k.split(".")[:2]) for k in tensores)
    print("prefixos nivel 2:", dict(sorted(pref2.items())))
    print("--- amostra de 15 chaves de tensor ---")
    for k in list(tensores)[:15]:
        print("   ", k, tuple(tensores[k].shape))
    nao_tensor = {k:type(v).__name__ for k,v in sd.items() if not hasattr(v,"shape")}
    if nao_tensor: print("chaves nao-tensor:", nao_tensor)
