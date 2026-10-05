#!/usr/bin/env python3
"""Teste do desvio C6 (`--text-adapter`) do `infer_deblur.py`. Roda sem GPU.

Monta um `Genfocus.pipeline.flux` FALSO e confere que o context manager
`adapter_de_texto` reescreve só o branch de texto, nas três chamadas a
`transformer_forward` que o `generate` oficial faz.

FIDELIDADE DO STUB (importa): o `generate` falso é definido DENTRO do módulo
falso via `exec(..., flux.__dict__)`, para que `generate.__globals__ is
flux.__dict__` — igual ao `flux.py` real. Um stub cujo `generate` fosse definido
neste arquivo teria `__globals__` diferente e o monkeypatch não o alcançaria; o
teste passaria a validar a coisa errada (foi o primeiro erro cometido aqui).

    python3 scripts/test_c6_monkeypatch.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

AQUI = Path(__file__).resolve().parent

FONTE_STUB = '''
registro = []

def transformer_forward(transformer, **kwargs):
    registro.append(list(kwargs["adapters"]))
    return (None,)

def generate(main_adapter=None, c_adapters=("deblurring",), n_text=1, ramos=3):
    """Replica a montagem das 3 chamadas do generate oficial (tiled, normal,
    image_guidance_scale != 1.0): adapters = [main_adapter]*2 + c_adapters."""
    for _ in range(ramos):
        transformer_forward(
            None,
            text_features=["prompt_embeds"] * n_text,
            adapters=[main_adapter] * (1 + n_text) + list(c_adapters),
        )
'''


def montar_stub():
    pkg = types.ModuleType("Genfocus"); pkg.__path__ = []
    sub = types.ModuleType("Genfocus.pipeline"); sub.__path__ = []
    flux = types.ModuleType("Genfocus.pipeline.flux")
    exec(compile(FONTE_STUB, "<flux-stub>", "exec"), flux.__dict__)
    pkg.pipeline = sub
    sub.flux = flux
    sys.modules.update({
        "Genfocus": pkg, "Genfocus.pipeline": sub, "Genfocus.pipeline.flux": flux,
    })
    assert flux.generate.__globals__ is flux.__dict__, "stub infiel"
    return flux


def carregar_infer():
    spec = importlib.util.spec_from_file_location("_infer", AQUI / "infer_deblur.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    flux = montar_stub()
    mod = carregar_infer()
    MESMO = mod.MESMO_QUE_MAIN
    reg = flux.registro

    def run(txt, main, **kw):
        reg.clear()
        with mod.adapter_de_texto(txt) as ap:
            flux.generate(main_adapter=main, **kw)
        return ap, list(reg)

    casos = 0

    ap, r = run(MESMO, "deblurring")
    assert ap is False and all(x == ["deblurring"] * 3 for x in r)
    print("1 OK  default: idêntico ao upstream (texto TAMBÉM com LoRA = defeito do C6)"); casos += 1

    ap, r = run(None, "deblurring")
    assert ap is True and len(r) == 3
    assert all(x == [None, "deblurring", "deblurring"] for x in r)
    print("2 OK  C6 ativo: texto sem adapter nas TRÊS chamadas"); casos += 1

    reg.clear(); flux.generate(main_adapter="deblurring")
    assert reg[0] == ["deblurring"] * 3
    assert flux.transformer_forward.__name__ == "transformer_forward"
    print("3 OK  original restaurado depois do with"); casos += 1

    ap, r = run(MESMO, None)
    assert r[0] == [None, None, "deblurring"]
    print("4 OK  cond-only: [texto=None, main=None, cond=LoRA] = oficial"); casos += 1

    ap, r = run(None, "deblurring", n_text=2)
    assert r[0] == [None, None, "deblurring", "deblurring"]
    print("5 OK  txt_n sai de text_features, não é assumido como 1"); casos += 1

    try:
        with mod.adapter_de_texto(None):
            assert flux.transformer_forward.__name__ == "wrapper"
            raise ValueError("boom")
    except ValueError:
        pass
    assert flux.transformer_forward.__name__ == "transformer_forward"
    print("6 OK  finally restaura mesmo com exceção"); casos += 1

    print(f"\n{casos}/6 cenários passaram.")


if __name__ == "__main__":
    main()
