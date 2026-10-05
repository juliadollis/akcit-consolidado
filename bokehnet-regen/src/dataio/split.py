"""Split treino/validação POR CENA, materializado no dataset.

Este módulo existe porque o risco mudou de tamanho. Com **um par por cena**, um split
por imagem era quase inofensivo. Agora a RealBokeh entrega **2 a 21 aberturas da mesma
cena** e o LFDOF entrega N desfocadas por AIF — split por imagem é vazamento garantido,
e infla a métrica de validação sem que nada denuncie.

Duas regras:

1. **A unidade do split é `scene_id`, nunca `sample_id`.**
2. **O split é materializado no dataset**, não deixado para o config do treino. Config
   é editável, some no rsync e diverge entre runs; um arquivo no release não.

Bônus já medido: o `timseizinger/RealBokeh_3MP` **já traz** `train` / `test` /
`validation` separados por cena (220 cenas em test, 220 em validation). Quando a origem
tem split, `split_from_source` respeita o que existe em vez de inventar outro.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


def _scene_bucket(scene_id: str, salt: str, buckets: int = 10_000) -> int:
    """Bucket determinístico e estável — mesma cena, mesmo bucket, sempre.

    `hash()` do Python NÃO serve: é randomizado por processo (PYTHONHASHSEED), então
    o split mudaria a cada execução e a validação de ontem viraria treino hoje.
    """
    digest = hashlib.sha256(f"{salt}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


@dataclass(frozen=True)
class SceneSplit:
    assignment: dict[str, str]          # scene_id -> "train" | "val"
    salt: str
    val_fraction: float

    def of(self, scene_id: str) -> str:
        if scene_id not in self.assignment:
            raise KeyError(
                f"cena {scene_id!r} não está no split. O split é materializado: uma cena "
                "que aparece na geração e não no split é erro de manifesto, não caso a "
                "resolver em runtime."
            )
        return self.assignment[scene_id]

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.assignment.values()))

    def to_json(self) -> str:
        return json.dumps({
            "salt": self.salt,
            "val_fraction": self.val_fraction,
            "counts": self.counts(),
            "assignment": self.assignment,
        }, indent=2, ensure_ascii=False, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SceneSplit":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["assignment"], data["salt"], data["val_fraction"])


def build_scene_split(
    scene_ids: Iterable[str], *, val_fraction: float = 0.05, salt: str = "bokehnet-regen-v1",
) -> SceneSplit:
    """Atribui cada CENA a treino ou validação, de forma determinística."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError(f"val_fraction fora de (0, 0.5): {val_fraction}")
    unique = sorted(set(scene_ids))
    if not unique:
        raise ValueError("nenhuma cena para dividir")
    threshold = int(round(val_fraction * 10_000))
    assignment = {
        scene: ("val" if _scene_bucket(scene, salt) < threshold else "train")
        for scene in unique
    }
    return SceneSplit(assignment, salt, val_fraction)


def split_from_source(
    scene_to_source_split: dict[str, str], *, val_names: tuple[str, ...] = ("validation", "val", "test"),
) -> SceneSplit:
    """Respeita o split que a origem já traz, em vez de inventar outro.

    A RealBokeh_3MP separa `train` / `test` / `validation` por cena. Reusar isso é
    melhor que sortear: preserva a intenção de quem montou o dataset e mantém a
    comparabilidade com quem já publicou número nele.
    """
    train_names = ("train", "training")
    assignment: dict[str, str] = {}
    for scene, source in scene_to_source_split.items():
        # `(source or "")` mandava split ausente para TREINO em silêncio — e treino é
        # justamente o lado que infla a métrica de validação. O gate contra vazamento
        # não pegaria: a cena ESTARIA no split, só que do lado errado.
        if not source:
            raise ValueError(
                f"cena {scene!r} sem `source_split`. Split materializado não inventa "
                "lado: ou a origem diz, ou a cena não entra."
            )
        norm = str(source).lower()
        if norm in val_names:
            assignment[scene] = "val"
        elif norm in train_names:
            assignment[scene] = "train"
        else:
            raise ValueError(
                f"split de origem desconhecido para {scene!r}: {source!r}. "
                f"Esperado um de {sorted(set(val_names) | set(train_names))}."
            )
    if not assignment:
        raise ValueError("nenhuma cena para dividir")
    n_val = sum(1 for v in assignment.values() if v == "val")
    return SceneSplit(assignment, "from_source", n_val / len(assignment))


# --------------------------------------------------------------------------------
# Verificação — o gate que o `data-contract` exige
# --------------------------------------------------------------------------------

@dataclass
class LeakReport:
    scenes_in_both: list[str]
    samples_without_scene: list[str]
    scenes_missing_from_split: list[str]
    samples_per_scene: dict[str, int]
    samples_without_split: list[str] = field(default_factory=list)
    scenes_disagreeing_with_split: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.scenes_in_both or self.samples_without_scene
                    or self.scenes_missing_from_split or self.samples_without_split
                    or self.scenes_disagreeing_with_split)

    def summary(self) -> str:
        scenes = len(self.samples_per_scene)
        samples = sum(self.samples_per_scene.values())
        lines = [
            "",
            "=" * 62,
            f"  cenas   : {scenes}",
            f"  amostras: {samples}   ({samples / max(scenes, 1):.1f} por cena)",
        ]
        if self.samples_per_scene:
            worst = max(self.samples_per_scene.values())
            lines.append(f"  máximo de amostras numa cena: {worst}")
        if self.clean:
            lines.append("  -> SEM VAZAMENTO: nenhuma cena em treino e validação")
        else:
            if self.scenes_in_both:
                lines.append(f"  -> VAZAMENTO: {len(self.scenes_in_both)} cenas nos dois splits")
                lines.append(f"     {self.scenes_in_both[:5]}")
            if self.scenes_missing_from_split:
                lines.append(f"  -> {len(self.scenes_missing_from_split)} cenas fora do split")
            if self.samples_without_scene:
                lines.append(f"  -> {len(self.samples_without_scene)} amostras sem scene_id")
            if self.samples_without_split:
                lines.append(f"  -> {len(self.samples_without_split)} amostras sem `split` gravado")
            if self.scenes_disagreeing_with_split:
                lines.append(f"  -> {len(self.scenes_disagreeing_with_split)} cenas cujo split "
                             "gravado diverge do split materializado")
                lines.append(f"     {self.scenes_disagreeing_with_split[:5]}")
        lines.append("=" * 62)
        return "\n".join(lines)


def check_no_leak(manifest_rows: Iterable[dict], split: SceneSplit) -> LeakReport:
    """Confere vazamento comparando o split GRAVADO EM CADA LINHA contra o split file.

    A versão anterior deste gate era **tautológica** e dava garantia falsa: ela fazia
    `scene_splits[scene].add(split.of(scene))` e depois procurava cenas com mais de um
    split. Como `split.of()` é função pura de `scene_id` sobre um `dict[str, str]`,
    `len(v) > 1` era impossível por construção — o ramo de detecção era código morto,
    com cobertura zero, e o teste que se chamava "detecta vazamento" afirmava
    `clean is True`. Um gate que não pode reprovar é pior que gate nenhum.

    Agora cada linha do manifesto carrega o `split` com que foi **gravada**, e o gate
    compara isso com o split materializado. Assim ele pega o que interessa de verdade:
    o release foi gerado com um split e alguém trocou o arquivo depois; duas execuções
    usaram `salt` diferente; ou uma cena mudou de lado entre lotes.

    Reporta também a razão amostras-por-cena, que é o número que diz se a diversidade
    efetiva é a que se pensa: 20.554 amostras vindas de 3.960 cenas não são 20.554
    unidades de diversidade.
    """
    per_scene: dict[str, int] = defaultdict(int)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    sem_cena: list[str] = []
    sem_split_na_linha: list[str] = []
    fora_do_split: set[str] = set()
    divergentes: set[str] = set()

    for row in manifest_rows:
        scene = row.get("scene_id")
        if not scene:
            sem_cena.append(row.get("sample_id", "?"))
            continue
        per_scene[scene] += 1

        gravado = row.get("split")
        if gravado is None:
            sem_split_na_linha.append(row.get("sample_id", "?"))
        else:
            scene_splits[scene].add(gravado)      # o que a LINHA diz

        try:
            esperado = split.of(scene)
        except KeyError:
            fora_do_split.add(scene)
            continue
        if gravado is not None and gravado != esperado:
            divergentes.add(scene)

    return LeakReport(
        scenes_in_both=sorted(s for s, v in scene_splits.items() if len(v) > 1),
        samples_without_scene=sem_cena,
        samples_without_split=sem_split_na_linha,
        scenes_missing_from_split=sorted(fora_do_split),
        scenes_disagreeing_with_split=sorted(divergentes),
        samples_per_scene=dict(per_scene),
    )
