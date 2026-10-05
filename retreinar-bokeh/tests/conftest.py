"""Coloca a raiz de `retreinar-deblur/` no sys.path.

Assim os testes rodam com `pytest tests/` a partir de qualquer diretório, sem
depender de PYTHONPATH externo.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for p in (str(RAIZ), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
