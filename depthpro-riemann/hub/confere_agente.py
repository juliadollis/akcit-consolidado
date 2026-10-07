import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
for rid, tipo in (("akcit-dephpro/depthpro-riemann-modelos", "model"),
                  ("akcit-dephpro/depthpro-riemann-avaliacoes", "dataset")):
    try:
        info = api.repo_info(rid, repo_type=tipo, files_metadata=True)
        fs = info.siblings
        tot = sum(s.size or 0 for s in fs)
        print(f"{rid}")
        print(f"   privado={info.private}  arquivos={len(fs)}  {tot/1e9:.1f} GB")
        print(f"   best.pt: {sum(1 for s in fs if s.rfilename.endswith('best.pt'))}")
        pref = {}
        for s in fs:
            pref[s.rfilename.split("/")[0]] = pref.get(s.rfilename.split("/")[0], 0) + 1
        print(f"   por pasta: {dict(sorted(pref.items(), key=lambda x: -x[1])[:6])}")
    except Exception as e:
        print(f"{rid} -> ERRO {str(e)[:120]}")
    print()

print("o repo de teste ainda existe?")
try:
    api.repo_info("akcit-dephpro/_teste_de_escrita", repo_type="model")
    print("   SIM, ainda esta la")
except Exception as e:
    print(f"   nao ({str(e)[:60]})")
