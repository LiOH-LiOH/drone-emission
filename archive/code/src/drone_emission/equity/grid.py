"""
Traffic Zone Grid Management Module

Creates and manages grid-based traffic zones for equity analysis.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class TrafficZoneGrid:
    """Traffic zone grid data structure."""

    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    unit_lat: float
    unit_lng: float
    n_rows: int
    n_cols: int

    def get_zone_center(self, row: int, col: int) -> Tuple[float, float]:
        """Get the center coordinates of a zone."""
        center_lat = self.min_lat + (row + 0.5) * self.unit_lat
        center_lng = self.min_lng + (col + 0.5) * self.unit_lng
        return center_lat, center_lng

    def get_zone_id(self, row: int, col: int) -> int:
        """Get zone ID from row and column indices."""
        return row * self.n_cols + col

    def get_row_col(self, zone_id: int) -> Tuple[int, int]:
        """Get row and column indices from zone ID."""
        row = zone_id // self.n_cols
        col = zone_id % self.n_cols
        return row, col

    def total_zones(self) -> int:
        """Get total number of zones."""
        return self.n_rows * self.n_cols


def create_traffic_zone_grid(
    boundaries_dict: Dict[str, float], unit_lat: float, unit_lng: float
) -> TrafficZoneGrid:
    """
    Create a traffic zone grid from boundaries.

    Args:
        boundaries_dict: Dictionary with min_lat, max_lat, min_lng, max_lng
        unit_lat: Grid cell height in degrees (~0.01 = ~1.1 km)
        unit_lng: Grid cell width in degrees (~0.01 = ~0.9 km)

    Returns:
        TrafficZoneGrid object
    """
    min_lat = boundaries_dict["min_lat"]
    max_lat = boundaries_dict["max_lat"]
    min_lng = boundaries_dict["min_lng"]
    max_lng = boundaries_dict["max_lng"]

    n_rows = int(np.ceil((max_lat - min_lat) / unit_lat))
    n_cols = int(np.ceil((max_lng - min_lng) / unit_lng))

    return TrafficZoneGrid(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lng=min_lng,
        max_lng=max_lng,
        unit_lat=unit_lat,
        unit_lng=unit_lng,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def assign_points_to_zones(
    grid: TrafficZoneGrid, lats: np.ndarray, lngs: np.ndarray
) -> np.ndarray:
    """
    Assign points to traffic zones using vectorized operations.

    Args:
        grid: TrafficZoneGrid object
        lats: Array of latitudes
        lngs: Array of longitudes

    Returns:
        Array of zone IDs for each point (-1 for out-of-bounds points)
    """
    # Calculate row and column indices
    rows = ((lats - grid.min_lat) / grid.unit_lat).astype(int)
    cols = ((lngs - grid.min_lng) / grid.unit_lng).astype(int)

    # Check bounds
    valid_mask = (
        (rows >= 0)
        & (rows < grid.n_rows)
        & (cols >= 0)
        & (cols < grid.n_cols)
    )

    # Calculate zone IDs
    zone_ids = np.where(valid_mask, rows * grid.n_cols + cols, -1)

    return zone_ids
