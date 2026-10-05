"""Utilidades dos testes.

O ponto central: vários módulos do pacote (`trainer`, `backbone`) importam
torch no topo, e torch não existe no ambiente onde estes testes rodam (login
node, CI, laptop). Em vez de copiar o código sob teste para dentro do teste —
o que faria o teste passar mesmo com o código quebrado —, extraímos o nó exato
do AST do arquivo REAL e o executamos isolado. Assim o teste continua acoplado
ao fonte de verdade.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
PACOTE = RAIZ / "genfocus_train"


def ler_fonte(nome_arquivo: str) -> str:
    return (PACOTE / nome_arquivo).read_text(encoding="utf-8")


def extrair_classe(nome_arquivo: str, nome_classe: str, globais: dict[str, Any]):
    """Compila e executa APENAS uma classe do arquivo, sem importar o módulo.

    Evita o `import torch` do topo de trainer.py/backbone.py.
    """
    arvore = ast.parse(ler_fonte(nome_arquivo))
    for no in arvore.body:
        if isinstance(no, ast.ClassDef) and no.name == nome_classe:
            modulo = ast.Module(body=[no], type_ignores=[])
            exec(compile(modulo, f"<{nome_arquivo}:{nome_classe}>", "exec"), globais)
            return globais[nome_classe]
    raise AssertionError(f"classe {nome_classe} não encontrada em {nome_arquivo}")


def extrair_constante(nome_arquivo: str, nome_const: str):
    """Lê o VALOR literal de uma constante de módulo, sem importar o módulo."""
    arvore = ast.parse(ler_fonte(nome_arquivo))
    for no in arvore.body:
        if isinstance(no, ast.Assign):
            for alvo in no.targets:
                if isinstance(alvo, ast.Name) and alvo.id == nome_const:
                    return ast.literal_eval(no.value)
        elif isinstance(no, ast.AnnAssign):
            if isinstance(no.target, ast.Name) and no.target.id == nome_const:
                return ast.literal_eval(no.value)
    raise AssertionError(f"constante {nome_const} não encontrada em {nome_arquivo}")


class OtimizadorFalso:
    """Só o que o WarmupCosineScheduler toca: `param_groups`."""

    def __init__(self, lr: float) -> None:
        self.param_groups = [{"lr": lr}]

    @property
    def lr(self) -> float:
        return self.param_groups[0]["lr"]
