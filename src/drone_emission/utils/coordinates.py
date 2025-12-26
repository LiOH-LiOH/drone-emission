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
