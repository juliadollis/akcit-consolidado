"""Adaptadores de FONTE — enumeram pares de origem, não geram rótulo nenhum.

Um adaptador aqui responde a uma pergunta só: **quais pares existem, e o que a
origem afirma sobre cada um.** Profundidade, máscara, K e mapa de defocus vivem em
`control/`, `model_runtime/` e `routes/` — nunca aqui.

A fronteira existe porque foi ela que faltou antes: a rota C antiga escolhia a AIF
dentro de `gt/` no meio do laço de geração, e o defeito D6 (uma foto de f/5.6 a f/14
entrando como "all-in-focus") ficou escondido dentro de um `for` de 200 linhas.
Aqui a escolha da AIF é uma linha, auditável, com o sha256 que a prova.
"""
from .realbokeh import (
    ALIGNMENT_ALIGNED, ALIGNMENT_MISALIGNED, ALIGNMENT_SHIFT,
    MIRROR_AIF_COLUMN, MIRROR_BOKEH_COLUMN, MIRROR_DATASET, MIRROR_IMAGE_HW,
    PENDING_REJECTION_REASONS, RAW_DATASET, SOURCE_SPLITS, ParsedName, RealBokehPair,
    enumerate_pairs, enumeration_summary, f_number_for_level, level_count,
    load_scene_metadata, load_split_metadata, pair_from_name, parse_file_name_base,
    parse_full_name, scene_key,
    scene_source_splits,
)

# O LFDOF entra pelos MÓDULOS, não por nomes soltos: `scene_key`,
# `parse_file_name_base`, `enumerate_pairs`, `SOURCE_SPLITS` e outros oito nomes existem
# nas DUAS fontes com semântica diferente (o split do LFDOF são dois, o da RealBokeh
# três; o `sample_id` tem prefixo diferente). Reexportá-los planos aqui deixaria
# `from sources import enumerate_pairs` ambíguo, e a última linha de import ganharia em
# silêncio — que é exatamente o tipo de escolha invisível que este projeto está
# desfazendo. Use `lfdof.enumerate_pairs` / `realbokeh.enumerate_pairs`.
from . import lfdof, lfdof_images

# A rota A entra pelos MÓDULOS, pela mesma razão do LFDOF. `enumerate_candidates` e
# `enumeration_summary` de `genphoto_ebb` têm a mesma cara dos homônimos das outras duas
# fontes e semântica diferente: aqui a unidade é a IMAGEM (uma cena, N variantes), não o
# par. Use `genphoto_ebb.enumerate_candidates`.
#
# `k_distribution` é fonte no mesmo sentido: ela enumera o que existe — a distribuição de
# K medida nas rotas B e C — afirma o que a origem afirma, e não produz rótulo nenhum. O
# sorteio em si é decisão da rota A e vive em `routes/route_a.py`.
from . import genphoto_ebb, k_distribution
from .genphoto_ebb import AifImage, PAPER_SOURCES
from .k_distribution import KDistribution, SampledK
from .lfdof import LFDOF_DATASET, LFDOF_IMAGE_HW, LFDOFPair
from .lfdof_images import LFDOFImageLoader

__all__ = [
    "lfdof", "lfdof_images", "genphoto_ebb", "k_distribution",
    "AifImage", "PAPER_SOURCES", "KDistribution", "SampledK",
    "LFDOF_DATASET", "LFDOF_IMAGE_HW", "LFDOFImageLoader", "LFDOFPair",
    "ALIGNMENT_ALIGNED", "ALIGNMENT_MISALIGNED", "ALIGNMENT_SHIFT",
    "MIRROR_AIF_COLUMN", "MIRROR_BOKEH_COLUMN", "MIRROR_DATASET", "MIRROR_IMAGE_HW",
    "PENDING_REJECTION_REASONS", "RAW_DATASET", "SOURCE_SPLITS", "ParsedName",
    "RealBokehPair", "enumerate_pairs", "enumeration_summary", "f_number_for_level",
    "level_count", "load_scene_metadata", "load_split_metadata", "pair_from_name",
    "parse_file_name_base", "scene_key",
    "parse_full_name", "scene_source_splits",
]
