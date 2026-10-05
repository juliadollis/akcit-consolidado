"""Renderer e calibração da Eq. 5.

`bokehme` importa torch, então não é reexportado aqui — importe explicitamente
`from renderer.bokehme import BokehMeRenderer` onde houver GPU.
"""
from .calibration import (
    K_ABSOLUTE_MAX_DEFAULT, K_MAX_DEFAULT, K_MIN_DEFAULT, KCalibration, calibrate_k,
)
from .verification import (
    DISC_EDGE_RATIO_MAX, GAUSSIAN_EDGE_RATIO, RadiusCheck, RadiusResponse,
    fit_radius_response,
    check_radius_is_linear_in_k, check_radius_matches_contract, edge_width_ratio,
    measure_blur_radius_px, point_light_scene, radial_profile,
)

__all__ = [
    "K_ABSOLUTE_MAX_DEFAULT", "K_MAX_DEFAULT", "K_MIN_DEFAULT", "KCalibration",
    "calibrate_k", "DISC_EDGE_RATIO_MAX", "GAUSSIAN_EDGE_RATIO", "RadiusCheck",
    "RadiusResponse", "fit_radius_response",
    "check_radius_is_linear_in_k", "check_radius_matches_contract", "edge_width_ratio",
    "measure_blur_radius_px", "point_light_scene", "radial_profile",
]
