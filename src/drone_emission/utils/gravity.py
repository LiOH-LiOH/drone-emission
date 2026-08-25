"""Shared gravity-model transformations used by calibration and OD generation."""

from __future__ import annotations

import numpy as np


def boxcox_transform(values, shape: float):
    """Return the stable Box-Cox transform for strictly positive values."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Box-Cox inputs must be finite and strictly positive")
    shape = float(shape)
    if not np.isfinite(shape):
        raise ValueError("Box-Cox shape must be finite")
    logarithm = np.log(values)
    if abs(shape) <= 1e-10:
        return logarithm
    return np.expm1(shape * logarithm) / shape


def boxcox_gravity_log_weights(
    volumes,
    distances_km,
    alpha: float,
    gamma: float,
    shape: float,
):
    """Return log weights for volume attraction and Box-Cox distance decay."""
    volumes = np.asarray(volumes, dtype=float)
    distances_km = np.asarray(distances_km, dtype=float)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Gravity-model volumes must be finite and strictly positive")
    return (
        float(alpha) * np.log(volumes)
        - float(gamma) * boxcox_transform(distances_km, shape)
    )
