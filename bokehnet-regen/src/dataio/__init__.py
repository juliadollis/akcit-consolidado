"""Contrato de dados: o que uma amostra é, como é codificada e como é dividida."""
from .encoding import (
    DEPTH_LONG_SIDE, EncodedDepth, decode_depth_m, decode_disparity, encode_depth,
    quantization_coc_error_px, resize_depth_nearest,
)
from .layout import (
    DEFAULT_DEPTH_LAYOUT, DEPTH_DIR, DepthLayout, depth_layout_of_run_config, depth_ref,
    depth_ref_of, require_filename_safe_scene_id, resolve_depth_path,
)
from .sample import (
    FOCUS_SOURCE_TO_MASK_SOURCE, REQUIRED_METADATA_FIELDS, ControlLabel,
    FocusRegionRecord, KSource, MaskSource, Sample, SampleProvenance, SampleRefs,
    metadata_to_json, validate_metadata,
)
from .split import (
    LeakReport, SceneSplit, build_scene_split, check_no_leak, split_from_source,
)
from .writer import (
    FileSampleWriter, estimate_disk_budget, iter_manifest, read_depth_u16,
    read_metadata,
)

__all__ = [
    "DEPTH_LONG_SIDE", "EncodedDepth", "decode_depth_m", "decode_disparity",
    "encode_depth", "quantization_coc_error_px", "resize_depth_nearest",
    "DEFAULT_DEPTH_LAYOUT", "DEPTH_DIR", "DepthLayout", "depth_layout_of_run_config",
    "depth_ref", "depth_ref_of", "require_filename_safe_scene_id", "resolve_depth_path",
    "FOCUS_SOURCE_TO_MASK_SOURCE", "REQUIRED_METADATA_FIELDS", "ControlLabel",
    "FocusRegionRecord", "KSource", "MaskSource", "Sample", "SampleProvenance",
    "SampleRefs", "metadata_to_json", "validate_metadata",
    "LeakReport", "SceneSplit", "build_scene_split", "check_no_leak", "split_from_source",
    "FileSampleWriter", "estimate_disk_budget", "iter_manifest", "read_depth_u16",
    "read_metadata",
]
