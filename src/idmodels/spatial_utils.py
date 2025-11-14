"""
Spatial utilities for computing directional wave features.

Provides location coordinates and functions for computing distances,
bearings, and directional neighbors.
"""

import math
import warnings
from importlib import resources
from typing import Dict, List, Tuple

import pandas as pd


# Direction angles (degrees, 0° = North, clockwise)
DIRECTION_ANGLES = {
    'N': 0,
    'NE': 45,
    'E': 90,
    'SE': 135,
    'S': 180,
    'SW': 225,
    'W': 270,
    'NW': 315
}

# Cone width for each direction (degrees)
CONE_WIDTH = 45.0  # ±22.5° around center

# Cache for loaded centroid data
_CENTROID_CACHE = {}


def _load_state_centroids() -> Dict[str, Tuple[float, float]]:
    """
    Load US state centroids from bundled CSV file.

    Returns
    -------
    dict
        Mapping from FIPS code to (latitude, longitude) tuple
    """
    if 'state' in _CENTROID_CACHE:
        return _CENTROID_CACHE['state']

    # Load the CSV file from package data
    try:
        # Python 3.9+
        with resources.files('idmodels.data').joinpath('state_centroids.csv').open('r') as f:
            df = pd.read_csv(f, dtype={'fips': str})
    except AttributeError:
        # Python 3.7-3.8 fallback
        import pkg_resources
        csv_path = pkg_resources.resource_filename('idmodels', 'data/state_centroids.csv')
        df = pd.read_csv(csv_path, dtype={'fips': str})

    # Convert to dictionary
    centroids = {}
    for _, row in df.iterrows():
        centroids[row['fips']] = (row['latitude'], row['longitude'])

    # Cache the result
    _CENTROID_CACHE['state'] = centroids

    return centroids


def get_location_centroids(agg_level: str = 'state') -> Dict[str, Tuple[float, float]]:
    """
    Get location centroids for a given aggregation level.

    Loads location coordinates from bundled CSV files in the package data directory.

    Parameters
    ----------
    agg_level : str
        Aggregation level ('state', 'county', 'hsa', etc.)
        Currently only 'state' is supported.

    Returns
    -------
    dict
        Mapping from location code to (latitude, longitude) tuple

    Raises
    ------
    ValueError
        If agg_level is not supported

    Notes
    -----
    Data is loaded from `idmodels/data/{agg_level}_centroids.csv` and cached
    for subsequent calls. See `idmodels/data/README.md` for data sources.
    """
    if agg_level == 'state':
        return _load_state_centroids().copy()
    else:
        raise ValueError(
            f"Aggregation level '{agg_level}' not supported. "
            f"Currently only 'state' is implemented. "
            f"To add support for other levels, add a {agg_level}_centroids.csv "
            f"file to idmodels/data/ and extend _load_state_centroids()."
        )


def haversine_distance(coord1: Tuple[float, float],
                       coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Parameters
    ----------
    coord1 : tuple
        (latitude, longitude) of first point in degrees
    coord2 : tuple
        (latitude, longitude) of second point in degrees

    Returns
    -------
    float
        Distance in kilometers
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Earth's radius in kilometers
    r = 6371.0

    return c * r


def compute_bearing(coord1: Tuple[float, float],
                    coord2: Tuple[float, float]) -> float:
    """
    Calculate the bearing (direction) from coord1 to coord2.

    Parameters
    ----------
    coord1 : tuple
        (latitude, longitude) of origin point in degrees
    coord2 : tuple
        (latitude, longitude) of destination point in degrees

    Returns
    -------
    float
        Bearing in degrees (0° = North, clockwise, range [0, 360))
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate bearing
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)

    # Normalize to [0, 360)
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


def get_directional_neighbors(
        origin_loc: str,
        origin_coord: Tuple[float, float],
        all_coords: Dict[str, Tuple[float, float]],
        direction: str,
        max_distance_km: float
) -> List[Tuple[str, float]]:
    """
    Find neighbors of origin location within a directional cone.

    Parameters
    ----------
    origin_loc : str
        Location code of origin
    origin_coord : tuple
        (latitude, longitude) of origin location
    all_coords : dict
        Mapping from location codes to (lat, lon) tuples
    direction : str
        Direction name (one of: N, NE, E, SE, S, SW, W, NW)
    max_distance_km : float
        Maximum distance in kilometers to consider as neighbor

    Returns
    -------
    list of tuples
        List of (location_code, distance) for neighbors in the cone,
        sorted by distance (nearest first)

    Raises
    ------
    ValueError
        If direction is not recognized
    """
    if direction not in DIRECTION_ANGLES:
        raise ValueError(
            f"Invalid direction '{direction}'. "
            f"Must be one of: {', '.join(sorted(DIRECTION_ANGLES.keys()))}"
        )

    direction_angle = DIRECTION_ANGLES[direction]
    half_cone = CONE_WIDTH / 2.0

    neighbors = []

    for loc_code, loc_coord in all_coords.items():
        # Skip origin location
        if loc_code == origin_loc:
            continue

        # Calculate distance
        distance = haversine_distance(origin_coord, loc_coord)

        # Skip if too far
        if distance > max_distance_km:
            continue

        # Calculate bearing
        bearing = compute_bearing(origin_coord, loc_coord)

        # Check if bearing is within directional cone
        # Handle wraparound at 0°/360°
        angle_diff = abs((bearing - direction_angle + 180) % 360 - 180)

        if angle_diff <= half_cone:
            neighbors.append((loc_code, distance))

    # Sort by distance (nearest first)
    neighbors.sort(key=lambda x: x[1])

    return neighbors


def validate_wave_directions(wave_directions: List[str]) -> None:
    """
    Validate that all directions are recognized.

    Parameters
    ----------
    wave_directions : list of str
        List of direction names to validate

    Raises
    ------
    ValueError
        If any direction is not recognized

    Warnings
    --------
    UserWarning
        If opposite directions are both included. In datasets with uniform spatial patterns,
        this may lead to correlation between opposite direction features.
    """
    valid_directions = set(DIRECTION_ANGLES.keys())

    # Check for invalid directions
    for direction in wave_directions:
        if direction not in valid_directions:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Must be one of: {', '.join(sorted(valid_directions))}"
            )

    # Check for opposite direction pairs (potential multicollinearity warning)
    opposite_pairs = [('N', 'S'), ('E', 'W'), ('NE', 'SW'), ('NW', 'SE')]
    wave_set = set(wave_directions)

    for dir1, dir2 in opposite_pairs:
        if dir1 in wave_set and dir2 in wave_set:
            warnings.warn(
                f"Both {dir1} and {dir2} directions are included. "
                f"In datasets with uniform spatial patterns, opposite directions may be correlated. "
                f"Consider checking correlation if multicollinearity is a concern "
                f"(note: tree-based models like GBQR are robust to multicollinearity).",
                UserWarning
            )
