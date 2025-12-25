import numpy as np
from coord_convert.transform import gcj2wgs


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(Mars Coordinates) to WGS84
    :param lng: GCJ02 longitude
    :param lat: GCJ02 latitude
    :return: (wgs84_lng, wgs84_lat)
    """
    return gcj2wgs(lng, lat)


def convert_coords_array(lng_array, lat_array):
    """
    Convert arrays of GCJ02 coordinates to WGS84
    """
    wgs_lngs = []
    wgs_lats = []
    for lng, lat in zip(lng_array, lat_array):
        w_lng, w_lat = gcj02_to_wgs84(lng, lat)
        wgs_lngs.append(w_lng)
        wgs_lats.append(w_lat)
    return np.array(wgs_lngs), np.array(wgs_lats)
