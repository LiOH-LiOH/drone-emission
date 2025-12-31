import time
import math
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_lat_lon(city):
    # Create a session with retry strategy
    session = create_retry_session()

    # Query Nominatim for city location
    nom_url = f"https://nominatim.openstreetmap.org/search?q={quote(city)}&format=json&addressdetails=1&limit=1"
    headers = {"User-Agent": "MatsimEmissionProject/1.0 (contact: info@matsim.org)"}

    print(f"Querying Nominatim for city: {city}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = session.get(nom_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                raise ValueError(f"Failed to query Nominatim: {resp.status_code}")
            break
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise ValueError(
                    f"Failed to connect to Nominatim after {max_retries} attempts: {e}"
                )

    data = resp.json()
    if not data:
        raise ValueError(f"City '{city}' not found in Nominatim.")

    city_data = data[0]
    lat = float(city_data["lat"])
    lon = float(city_data["lon"])
    print(f"City: {city}, Lat: {lat}, Lon: {lon}")
    return lat, lon


def get_epsg_code(city_name: str) -> str:
    """
    Get the EPSG code for the UTM zone of a city.

    Args:
        city_name (str): The name of the city.

    Returns:
        str: The EPSG code (e.g., "EPSG:32650").
    """
    lat, lon = get_lat_lon(city_name)

    # Calculate UTM zone
    zone = math.floor((lon + 180) / 6) + 1

    # Determine EPSG code
    if lat >= 0:
        epsg_code = 32600 + zone
    else:
        epsg_code = 32700 + zone

    return epsg_code
