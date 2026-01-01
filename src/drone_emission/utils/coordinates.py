import numpy as np
from coord_convert.transform import gcj2wgs
from pyproj import Transformer

# Alias directly to avoid redundant function wrappers
gcj02_to_wgs84 = gcj2wgs

# Initialize transformer globally
# EPSG:4326 (WGS84) -> EPSG:3857 (Web Mercator)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
wgs84_to_3857 = _TRANSFORMER.transform


def convert_coords_array(lng_array, lat_array):
    """
    Convert arrays of GCJ02 coordinates to WGS84
    """
    wgs_lngs = []
    wgs_lats = []
    for lng, lat in zip(lng_array, lat_array):
        w_lng, w_lat = gcj2wgs(lng, lat)
        wgs_lngs.append(w_lng)
        wgs_lats.append(w_lat)
    return np.array(wgs_lngs), np.array(wgs_lats)


# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in meters.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (in degrees)
        lat2, lon2: Latitude and longitude of point 2 (in degrees)

    Returns:
        Distance in meters
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c  # in kilometers
