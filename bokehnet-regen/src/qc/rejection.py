"""Registro de rejeições — o que substitui o fallback.

A regra do projeto é: falta de dado NUNCA vira constante. A amostra é rejeitada e o
motivo é gravado. Isso só funciona se o motivo for agregável, então todo run termina
imprimindo o histograma de motivos.

Sem esse histograma não dá para calibrar limiar nenhum, e é ele que denuncia um
fallback novo: se uma categoria de rejeição some de um dia para o outro, alguém
"consertou" o caminho de falha em vez do dado.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from control.contract import SampleRejected


@dataclass
class RejectionLog:
    """Acumula aceitas e rejeitadas, e escreve um JSONL por amostra.

    O JSONL é append-only e serve de retomada: uma amostra com `status == "ok"` não
    é reprocessada. Rejeitadas SÃO reprocessadas, porque um gate recalibrado pode
    aceitá-las depois — e é por isso que o motivo fica gravado em vez de a linha
    sumir.
    """

    path: Optional[Path] = None
    accepted: int = 0
    reasons: Counter = field(default_factory=Counter)
    _handle: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    # -- escrita ---------------------------------------------------------------

    def accept(self, sample_id: str, payload: Optional[dict] = None) -> None:
        self.accepted += 1
        self._write({"sample_id": sample_id, "status": "ok", **(payload or {})})

    def reject(self, sample_id: str, reason: str, detail: str = "", payload: Optional[dict] = None) -> None:
        self.reasons[reason] += 1
        self._write({
            "sample_id": sample_id, "status": "rejected",
            "reason": reason, "detail": detail, **(payload or {}),
        })

    def reject_from(self, sample_id: str, exc: SampleRejected, payload: Optional[dict] = None) -> None:
        self.reject(sample_id, exc.reason, exc.detail, payload)

    def _write(self, row: dict) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        self._handle.flush()   # crash no meio do run não pode perder o histograma

    # -- leitura ---------------------------------------------------------------

    @property
    def rejected(self) -> int:
        return sum(self.reasons.values())

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    def completed_ids(self) -> set[str]:
        """IDs já aceitos, para retomada. Rejeitadas ficam de fora de propósito."""
        done: set[str] = set()
        if self.path is None or not self.path.exists():
            return done
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") == "ok" and row.get("sample_id"):
                    done.add(row["sample_id"])
        return done

    # -- o histograma ----------------------------------------------------------

    def summary(self) -> str:
        if self.total == 0:
            return "[rejeições] nenhuma amostra processada."
        lines = [
            "",
            "=" * 62,
            f"  aceitas   : {self.accepted:>7}  ({100 * self.accepted / self.total:5.1f}%)",
            f"  rejeitadas: {self.rejected:>7}  ({100 * self.rejected / self.total:5.1f}%)",
            f"  total     : {self.total:>7}",
        ]
        if self.reasons:
            lines.append("-" * 62)
            lines.append("  motivos de rejeição:")
            width = max(len(r) for r in self.reasons)
            for reason, count in self.reasons.most_common():
                share = 100 * count / self.total
                lines.append(f"    {reason:<{width}}  {count:>6}  ({share:5.1f}%)")
        lines.append("=" * 62)
        return "\n".join(lines)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RejectionLog":
        return self

    def __exit__(self, *exc_info) -> None:
        print(self.summary())
        self.close()
