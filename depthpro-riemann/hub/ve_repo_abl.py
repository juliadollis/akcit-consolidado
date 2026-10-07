import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
for rid in ("akcit-h100n1/depthpro-ablacao-seeds",):
    print("=== " + rid + " ===")
    try:
        info = api.repo_info(rid, repo_type="model", files_metadata=True)
        print("privado:", info.private, " ultima alteracao:", info.lastModified)
        for s in sorted(info.siblings, key=lambda x: x.rfilename):
            print("  %-60s %.2f GB" % (s.rfilename, (s.size or 0) / 1e9))
    except Exception as e:
        print("  erro:", str(e)[:160])
