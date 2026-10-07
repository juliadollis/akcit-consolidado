from huggingface_hub import HfApi
a = HfApi()
print("whoami:", a.whoami()["name"])
try:
    print("models:", [m.id for m in a.list_models(author="akcit-dephpro")])
except Exception as e:
    print("models err:", e)
try:
    print("datasets:", [d.id for d in a.list_datasets(author="akcit-dephpro")])
except Exception as e:
    print("datasets err:", e)
try:
    import huggingface_hub as h
    print("hub:", h.__version__)
except Exception as e:
    print(e)
