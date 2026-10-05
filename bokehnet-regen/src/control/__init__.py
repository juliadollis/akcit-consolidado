"""Contrato canônico do sinal de controle. Ver `contract.py` e `reference/CONTRATO.md`."""
from .contract import (
    DEPTH_BACKENDS, GATE_REJECTION_REASONS, reject,
    FOCUS_DEPTH_MAX_M, FOCUS_DEPTH_MIN_M, MIN_DEPTH_RANGE_RATIO,
    CONTROL_VERSION, MAX_COC, MM_PER_M, MetricDepth, REJECTION_REASONS, SampleRejected,
    control_metadata, decode_defocus_uint16, defocus_map, encode_defocus_uint16,
    focus_disparity_from_mask, k_at_resolution, k_eq3_mm, k_for_bokehme, k_from_exif,
    k_official, pixel_ratio, sensor_width_mm, signed_coc_px, validate_metric_depth,
)

__all__ = [
    "DEPTH_BACKENDS", "GATE_REJECTION_REASONS", "reject",
    "FOCUS_DEPTH_MAX_M", "FOCUS_DEPTH_MIN_M", "MIN_DEPTH_RANGE_RATIO",
    "CONTROL_VERSION", "MAX_COC", "MM_PER_M", "MetricDepth", "REJECTION_REASONS",
    "SampleRejected", "control_metadata", "decode_defocus_uint16", "defocus_map",
    "encode_defocus_uint16", "focus_disparity_from_mask", "k_at_resolution", "k_eq3_mm",
    "k_for_bokehme", "k_from_exif", "k_official", "pixel_ratio", "sensor_width_mm",
    "signed_coc_px", "validate_metric_depth",
]
