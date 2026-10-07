#!/usr/bin/env python3
"""Adiciona reivindicacao atomica ao avalia_modelos.py.

Os dois fiscais varrem a mesma lista de checkpoints na mesma ordem. O filtro de
pendentes so olhava se o test_metrics.json ja existia, e olhava uma vez, no
inicio da varredura: os dois comecavam do mesmo estado e avaliavam exatamente o
mesmo par ao mesmo tempo. Uma das placas nao rendia nada.

mkdir e atomico no POSIX: ou cria o diretorio, ou levanta FileExistsError. Nao
existe janela entre checar e criar, que e o que um `if not exists` tem.
"""
import re
import shutil
import sys

ALVO = "/raid/user_juliadollis/julia_docker/avalia_modelos.py"
shutil.copy2(ALVO, ALVO + ".antes_do_claim")
src = open(ALVO).read()

if "_reivindica" in src:
    print("ja aplicado, nada a fazer")
    sys.exit(0)

for mod in ("import os", "import time"):
    if not re.search(rf"^{mod}$", src, re.M):
        src = src.replace("import json", f"{mod}\nimport json", 1)

FUNC = '''

def _reivindica(dest, maximo_h=6):
    """Reivindica o par (checkpoint, mesa) de forma atomica.

    Dois fiscais varrem a mesma lista na mesma ordem. Sem isto os dois avaliam o
    mesmo par ao mesmo tempo e uma das placas nao rende nada -- foi o que
    aconteceu em 17/09. `mkdir` ou cria, ou levanta FileExistsError; nao ha
    janela entre checar e criar.

    Uma claim orfa (fiscal morto por queda de rede ou OOM) devolve o par para a
    fila depois de `maximo_h` sem resultado, senao um par ficaria travado para
    sempre.
    """
    dest.mkdir(parents=True, exist_ok=True)
    marca = dest / ".claim"
    try:
        marca.mkdir()
        return True
    except FileExistsError:
        if (dest / "test_metrics.json").exists():
            return False
        idade = time.time() - marca.stat().st_mtime
        if idade > maximo_h * 3600:
            os.utime(marca, None)
            print(f"[avalia] claim orfa de {idade/3600:.1f} h em {dest}; retomo",
                  flush=True)
            return True
        return False

'''

ANTIGO_FILTRO = """        pendentes = [n for n in loaders
                     if not (saida / rotulo / n / "test_metrics.json").exists()]"""
NOVO_FILTRO = """        pendentes = [n for n in loaders
                     if not (saida / rotulo / n / "test_metrics.json").exists()
                     and _reivindica(saida / rotulo / n)]"""
assert ANTIGO_FILTRO in src, "filtro de pendentes nao encontrado"
src = src.replace(ANTIGO_FILTRO, NOVO_FILTRO, 1)

ANTIGO_MKDIR = """            dest = saida / rotulo / nome
            dest.mkdir(parents=True, exist_ok=True)"""
NOVO_MKDIR = """            dest = saida / rotulo / nome   # ja criado pela reivindicacao"""
assert ANTIGO_MKDIR in src, "mkdir do destino nao encontrado"
src = src.replace(ANTIGO_MKDIR, NOVO_MKDIR, 1)

m = re.search(r"^def main\(", src, re.M)
assert m, "main() nao encontrada"
src = src[:m.start()] + FUNC.lstrip("\n") + "\n" + src[m.start():]

open(ALVO, "w").write(src)
print("aplicado; backup em", ALVO + ".antes_do_claim")
