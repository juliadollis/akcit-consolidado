"""Configuração de `sys.path` e helpers, compartilhados pelos scripts de `inferencia/`.

Não duplica `scripts/_comum.py`: põe `scripts/` no path e reexporta de lá. Duas
cópias do mesmo helper é a forma clássica de as duas divergirem em silêncio.
"""

from __future__ import annotations

import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent          # retreinar-deblur/inferencia/
RAIZ = AQUI.parent                              # retreinar-deblur/
GENFOCUS_OFICIAL = RAIZ / "third_party" / "Genfocus"
SCRIPTS = RAIZ / "scripts"


def preparar_path() -> None:
    """Põe a raiz do projeto, o clone oficial do Genfocus e `scripts/` no path.

    O `Genfocus/pipeline/flux.py` do repo oficial é importado como
    `Genfocus.pipeline.flux` e NÃO é instalável, então o diretório que o contém
    tem de estar no path.
    """
    # AQUI tem que ficar em PRIMEIRO lugar, sempre. Existem homônimos em
    # `scripts/` (`avaliar_deblur.py`, `infer_deblur.py`) — restos de um move,
    # que sobrevivem no cluster porque o rsync do projeto NUNCA usa --delete.
    # O bug real que isto conserta: como o Python já põe o diretório do script
    # em sys.path[0], o `if s not in sys.path` PULAVA o AQUI, e o `scripts/`
    # inserido depois na posição 0 passava na frente. `import avaliar_deblur`
    # pegava a cópia velha. Falhou em produção assim.
    aqui = str(AQUI)
    if aqui in sys.path:
        sys.path.remove(aqui)
    sys.path.insert(0, aqui)

    for p in (RAIZ, GENFOCUS_OFICIAL):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(1, s)

    # `scripts/` por ÚLTIMO: serve só para o `_comum.py`, e não pode sombrear
    # nada de `inferencia/`.
    s = str(SCRIPTS)
    if s not in sys.path:
        sys.path.append(s)


preparar_path()

# Reexporta os helpers de `scripts/_comum.py` (fonte única).
from _comum import (  # noqa: E402
    alinhar16,
    calculate_shift_local,
    imprimir_tabela,
    seq_len_de,
    token_hf,
)

__all__ = [
    "AQUI", "RAIZ", "GENFOCUS_OFICIAL", "SCRIPTS", "preparar_path",
    "token_hf", "imprimir_tabela", "calculate_shift_local", "alinhar16", "seq_len_de",
]
